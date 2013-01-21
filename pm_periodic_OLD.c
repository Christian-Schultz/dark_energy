#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <mpi.h>
#ifdef DEBUG
#include <assert.h>
#endif

/*! \file pm_periodic.c
 *  \brief routines for periodic PM-force computation
 */

#ifdef PMGRID
#ifdef PERIODIC

#ifdef NOTYPEPREFIX_FFTW
#include        <rfftw_mpi.h>
#else
#ifdef DOUBLEPRECISION_FFTW
#include     <drfftw_mpi.h>	/* double precision FFTW */
#else
#include     <srfftw_mpi.h>
#endif
#endif


#include "allvars.h"
#include "proto.h"

#define  PMGRID2 (2*(PMGRID/2 + 1))  /* Last dimension (nz) of the in-place FFT. Only PMGRID values are ultimately stored, however */

/* Dark energy macros */
#define LOGICAL_INDEX(x)  ((x<0) ? x+PMGRID : (x>=PMGRID) ? x-PMGRID : x ) /* Map index to the range [0,PMGRID[ */
#define INDMAP(i,j,k) ((i)*PMGRID*PMGRID2+(j)*PMGRID2+k) /* Map (i,j,k) in 3 dimensional array (dimx X PMGRID X PMGRID2) to 1 dimensional array */

#ifdef DEBUG
void dbg_print(int, fftw_real *,fftw_real *);
void dbg_print_fftw_format(int, fftw_real *,fftw_real *);
#endif

static rfftwnd_mpi_plan fft_forward_plan, fft_inverse_plan;

static int slab_to_task[PMGRID];
static int *slabs_per_task;
static int *first_slab_of_task;
static int *meshmin_list, *meshmax_list;

static int slabstart_x, nslab_x, slabstart_y, nslab_y, smallest_slab;

static int fftsize, maxfftsize;

static fftw_real *rhogrid, *forcegrid, *workspace;
static fftw_complex *fft_of_rhogrid;
#ifdef DYNAMICAL_DE
static fftw_complex *fft_of_rhogrid_DE;
#endif

static FLOAT to_slab_fac;

#ifdef DYNAMICAL_DE
static int recv_tasks[4]; /* The 4 tasks that have the slabs this process needs (ordered left left, left, right, right right) */
static int send_tasks[5]; /* The (up to 5) processes that needs this process' slabs. Only in the case where some processes, but not all, is it neccessary to communicate with 5 others, otherwise this is normally 4*/
static int slabs_to_send[5]; /* The slabs this process needs to send in the order defined in send_tasks */
static int slabs_to_recv[4]; /* The slabs this process needs to send in the order defined in send_tasks */
static int nslabs_to_send;  /* How many slabs does this process need to send? Normally 4 */

/* TODO: Purge U and rho send arrays */
static fftw_real *rhogrid_tot,*U_recv, *rho_recv, *phi_recv;

/* TODO: Does it make a difference whether ugrid is a FLOAT or a fftw_real? */
static fftw_real *rhogrid_DE, (*ugrid_DE)[3];

void check_omegas(void);

int comm_order(int nslabs){
	if(nslabs>0){
		int index=LOGICAL_INDEX(slabstart_x-1);
		slabs_to_recv[1]=index;
		int my_task_l=slab_to_task[index];

		index=LOGICAL_INDEX(slabstart_x-2);
		slabs_to_recv[0]=index;
		int my_task_ll=slab_to_task[index];

		index=LOGICAL_INDEX(slabstart_x+nslab_x);
		slabs_to_recv[2]=index;
		int my_task_r=slab_to_task[index];

		index= LOGICAL_INDEX(slabstart_x+nslab_x+1);
		slabs_to_recv[3]=index;
		int my_task_rr=slab_to_task[index];

		/* The task(s) that has the four slabs this processor needs */
		recv_tasks[0]=my_task_ll;
		recv_tasks[1]=my_task_l;
		recv_tasks[2]=my_task_r;
		recv_tasks[3]=my_task_rr;

		/* Now find the processors that need this processor's slabs (could be more than 4!) */

		int task, first_slab,last_slab,i;
		int nslabs_to_send=0;

		if(ThisTask==0){
			for( task=1 ; task<NTask ; ++task )
			{
				if(slabs_per_task[task]<4){
					master_fprintf(stderr,"Warning: A bad number of processes has been chosen to run the PM part of the code.\n"
							"Normally you want each process to have at least 4 slabs\n");
					break;
				}
			}
			if(PMGRID<5 ){
				master_fprintf(stderr,"Need more than 4 slabs to run the Dark Energy code. Increase PMGRID\n");
				endrun(1);
			}
		}


		int slabs[4];
		for( task=0 ; task<NTask ; ++task )
		{
			if(slabs_per_task[task]==0)
				continue;
			if(task==ThisTask && NTask!=1)
				continue;
			/* Note: Program will hang if there is only 2 processes in the FFT pool, and one process only has 1 slab. 
			 * This should be impossible however, due to the requirement that a minimum of 4 slabs is needed */

			first_slab=first_slab_of_task[task];
			last_slab=first_slab+slabs_per_task[task]-1;


			slabs[0]=LOGICAL_INDEX(first_slab-2); /* your ll slab */

			slabs[1]=LOGICAL_INDEX(first_slab-1); /* your l slab */

			slabs[2]=LOGICAL_INDEX(last_slab+1);  /* your r slab */

			slabs[3]=LOGICAL_INDEX(last_slab+2);  /* your rr slab */


			for( i=0 ; i<4 ; ++i )
			{
				if(slabs[i]>=slabstart_x && slabs[i]<= (slabstart_x+nslab_x-1) ){
					send_tasks[nslabs_to_send]=task;
					slabs_to_send[nslabs_to_send]=slabs[i];
					++nslabs_to_send;
				}
			}


		}

		return nslabs_to_send;
	}
	else
		return 0;
}

void DE_allocate(int nx){

	if(nx>0){

		const double rhocrit=3*All.Hubble*All.Hubble/(8*M_PI*All.G);
		const double OmegaLambda=All.OmegaLambda/pow(All.Time,3.0*All.DarkEnergyW);
		const FLOAT de_mass=OmegaLambda*rhocrit*All.BoxSize*All.BoxSize*All.BoxSize/(PMGRID*PMGRID*PMGRID);

		rhogrid_DE=malloc(nx*PMGRID*PMGRID*sizeof(fftw_real));
		ugrid_DE=malloc(3*nx*PMGRID*PMGRID*sizeof(fftw_real));

		mpi_printf("Allocated %lu bytes (%lu MB) for DE arrays\n",4*nx*PMGRID*PMGRID*sizeof(fftw_real),4*nx*PMGRID*PMGRID*sizeof(fftw_real)/(1024*1024));

		int i,j;
		for( i=0 ; i<nslab_x*PMGRID*PMGRID ; ++i )
		{
			rhogrid_DE[i]=de_mass;
			for( j=0 ; j<3 ; ++j )
			{
				ugrid_DE[i][j]=0;
			}
		}
	}
	else{
		rhogrid_DE=NULL;
		ugrid_DE=NULL;
	}


}

void PM_cleanup(int nx){ /* Like free_memory(), this is not actually called by the program */
	free(rhogrid_DE);
	free(ugrid_DE);
	free(slabs_per_task);
	free(first_slab_of_task);
	rfftwnd_mpi_destroy_plan(fft_forward_plan);
	rfftwnd_mpi_destroy_plan(fft_inverse_plan);
}

#endif

/*! This routines generates the FFTW-plans to carry out the parallel FFTs
 *  later on. Some auxiliary variables are also initialized.
 */
void pm_init_periodic(void)
{
	int i;
	int slab_to_task_local[PMGRID];

	All.Asmth[0] = ASMTH * All.BoxSize / PMGRID;
	All.Rcut[0] = RCUT * All.Asmth[0];

	/* Set up the FFTW plan files. */

	fft_forward_plan = rfftw3d_mpi_create_plan(MPI_COMM_WORLD, PMGRID, PMGRID, PMGRID,
			FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE | FFTW_IN_PLACE);
	fft_inverse_plan = rfftw3d_mpi_create_plan(MPI_COMM_WORLD, PMGRID, PMGRID, PMGRID,
			FFTW_COMPLEX_TO_REAL, FFTW_ESTIMATE | FFTW_IN_PLACE);

	/* Workspace out the ranges on each processor. */

	rfftwnd_mpi_local_sizes(fft_forward_plan, &nslab_x, &slabstart_x, &nslab_y, &slabstart_y, &fftsize);

	for(i = 0; i < PMGRID; i++)
		slab_to_task_local[i] = 0;

	for(i = 0; i < nslab_x; i++)
		slab_to_task_local[slabstart_x + i] = ThisTask;

	MPI_Allreduce(slab_to_task_local, slab_to_task, PMGRID, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	MPI_Allreduce(&nslab_x, &smallest_slab, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

	slabs_per_task = malloc(NTask * sizeof(int));
	MPI_Allgather(&nslab_x, 1, MPI_INT, slabs_per_task, 1, MPI_INT, MPI_COMM_WORLD);

	if(ThisTask == 0)
	{
		for(i = 0; i < NTask; i++)
			printf("Task=%d  FFT-Slabs=%d\n", i, slabs_per_task[i]);
	}

	first_slab_of_task = malloc(NTask * sizeof(int));
	MPI_Allgather(&slabstart_x, 1, MPI_INT, first_slab_of_task, 1, MPI_INT, MPI_COMM_WORLD);

	meshmin_list = malloc(3 * NTask * sizeof(int));
	meshmax_list = malloc(3 * NTask * sizeof(int));


	to_slab_fac = PMGRID / All.BoxSize;

	MPI_Allreduce(&fftsize, &maxfftsize, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#ifdef DYNAMICAL_DE
	DE_allocate(nslab_x);
	nslabs_to_send=comm_order(nslab_x);
#endif
}


/*! This function allocates the memory neeed to compute the long-range PM
 *  force. Three fields are used, one to hold the density (and its FFT, and
 *  then the real-space potential), one to hold the force field obtained by
 *  finite differencing, and finally a workspace field, which is used both as
 *  workspace for the parallel FFT, and as buffer for the communication
 *  algorithm used in the force computation.
 */
void pm_init_periodic_allocate(int dimprod)
{
	static int first_alloc = 1;
	int dimprodmax;
	double bytes_tot = 0;
	size_t bytes;

	MPI_Allreduce(&dimprod, &dimprodmax, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

	/* allocate the memory to hold the FFT fields */
	if(!(rhogrid = (fftw_real *) malloc(bytes = fftsize * sizeof(fftw_real))))
	{
		printf("failed to allocate memory for `FFT-rhogrid' (%g MB).\n", bytes / (1024.0 * 1024.0));
		endrun(1);
	}
	bytes_tot += bytes;

#ifdef DYNAMICAL_DE
	if(!(rhogrid_tot = (fftw_real *) malloc(bytes = fftsize * sizeof(fftw_real))))
	{
		printf("failed to allocate memory for `FFT-rhogrid_tot' (%g MB).\n", bytes / (1024.0 * 1024.0));
		endrun(1);
	}
	bytes_tot += bytes;

	if(nslab_x>0){
		if(!(U_recv = (fftw_real *) malloc(bytes = 4*3*PMGRID*PMGRID * sizeof(fftw_real))))
		{
			printf("failed to allocate memory for `U_recv' (%g MB).\n", bytes / (1024.0 * 1024.0));
			endrun(1);
		}
		bytes_tot += bytes;

		if(!(rho_recv = (fftw_real *) malloc(bytes = 4*PMGRID*PMGRID * sizeof(fftw_real))))
		{
			printf("failed to allocate memory for `rho_recv' (%g MB).\n", bytes / (1024.0 * 1024.0));
			endrun(1);
		}
		bytes_tot += bytes;

		if(!(phi_recv = (fftw_real *) malloc(bytes = 4*PMGRID*PMGRID2 * sizeof(fftw_real))))
		{
			printf("failed to allocate memory for `phi_recv' (%g MB).\n", bytes / (1024.0 * 1024.0));
			endrun(1);
		}
		bytes_tot += bytes;
	}
	else{
		U_recv=rho_recv=phi_recv=NULL;
	}
#endif

	if(!(forcegrid = (fftw_real *) malloc(bytes = imax(fftsize, dimprodmax) * sizeof(fftw_real))))
	{
		printf("failed to allocate memory for `FFT-forcegrid' (%g MB).\n", bytes / (1024.0 * 1024.0));
		endrun(1);
	}
	bytes_tot += bytes;

	if(!(workspace = (fftw_real *) malloc(bytes = imax(maxfftsize, dimprodmax) * sizeof(fftw_real))))
	{
		printf("failed to allocate memory for `FFT-workspace' (%g MB).\n", bytes / (1024.0 * 1024.0));
		endrun(1);
	}
	bytes_tot += bytes;

	if(first_alloc == 1)
	{
		first_alloc = 0;
		if(ThisTask == 0)
			printf("\nAllocated %g MByte for FFT data.\n\n", bytes_tot / (1024.0 * 1024.0));
	}

	fft_of_rhogrid = (fftw_complex *) & rhogrid[0];
#ifdef DYNAMICAL_DE
	fft_of_rhogrid_DE = (fftw_complex *) & rhogrid_tot[0];
#endif
}



/*! This routine frees the space allocated for the parallel FFT algorithm.
*/
void pm_init_periodic_free(void)
{
	/* allocate the memory to hold the FFT fields */
	free(workspace);
	free(forcegrid);
	free(rhogrid);
#ifdef DYNAMICAL_DE
	free(rhogrid_tot);
	free(U_recv);
	free(rho_recv);
	free(phi_recv);
#endif
}


#ifdef DYNAMICAL_DE
#ifdef DEBUG
void DE_array_debug(void){
	int i,j,k;
	unsigned int num;
	const unsigned int offset=slabstart_x*PMGRID*PMGRID;

	if(nslab_x>0)
		for( i=0 ; i<nslab_x ; ++i )
		{
			for( j=0 ; j<PMGRID ; ++j )
			{
				for( k=0 ; k<PMGRID ; ++k )
				{
					num=i*PMGRID*PMGRID+j*PMGRID+k;
					//mpi_printf("num=%u\n",num);
					rhogrid_DE[num]=(fftw_real) num+offset;
					ugrid_DE[num][0]=(fftw_real) -num-offset;
					ugrid_DE[num][1]=(fftw_real) -num-offset;
					ugrid_DE[num][2]=(fftw_real) -num-offset;
				}
			}
		}


}
#endif
#endif

/*! Calculates the long-range periodic force given the particle positions
 *  using the PM method.  The force is Gaussian filtered with Asmth, given in
 *  mesh-cell units. We carry out a CIC charge assignment, and compute the
 *  potenial by Fourier transform methods. The potential is finite differenced
 *  using a 4-point finite differencing formula, and the forces are
 *  interpolated tri-linearly to the particle positions. The CIC kernel is
 *  deconvolved. Note that the particle distribution is not in the slab
 *  decomposition that is used for the FFT. Instead, overlapping patches
 *  between local domains and FFT slabs are communicated as needed.
 */
void pmforce_periodic(void)
{
	double k2, kx, ky, kz, smth;
	double dx, dy, dz;
	double fx, fy, fz, ff;
	double asmth2, fac, acc_dim;
	int i, j, k, slab, level, sendTask, recvTask;
	int x, y, z, xl, yl, zl, xr, yr, zr, xll, yll, zll, xrr, yrr, zrr, ip, dim;
	int slab_x, slab_y, slab_z;
	int slab_xx, slab_yy, slab_zz;
	int meshmin[3], meshmax[3], sendmin, sendmax, recvmin, recvmax;
	int rep, ncont, cont_sendmin[2], cont_sendmax[2], cont_recvmin[2], cont_recvmax[2];
	int dimx, dimy, dimz, recv_dimx, recv_dimy, recv_dimz;
	MPI_Status status;


	if(ThisTask == 0)
	{
		printf("Starting periodic PM calculation.\n");
		fflush(stdout);
	}

	force_treefree();

	asmth2 = (2 * M_PI) * All.Asmth[0] / All.BoxSize; /* Default value for Asmth is 1.25, can be changed in the Makefile */
	asmth2 *= asmth2;

	fac = All.G / (M_PI * All.BoxSize);	/* to get potential */
	fac *= 1 / (2 * All.BoxSize / PMGRID);	/* for finite differencing */

	/* first, establish the extension of the local patch in the PMGRID  */

	for(j = 0; j < 3; j++)
	{
		meshmin[j] = PMGRID;
		meshmax[j] = 0;
	}

	for(i = 0; i < NumPart; i++)
	{
		for(j = 0; j < 3; j++)
		{
			slab = to_slab_fac * P[i].Pos[j];
			if(slab >= PMGRID)
				slab = PMGRID - 1;

			if(slab < meshmin[j])
				meshmin[j] = slab;

			if(slab > meshmax[j])
				meshmax[j] = slab;
		}
	}

	MPI_Allgather(meshmin, 3, MPI_INT, meshmin_list, 3, MPI_INT, MPI_COMM_WORLD);
	MPI_Allgather(meshmax, 3, MPI_INT, meshmax_list, 3, MPI_INT, MPI_COMM_WORLD);

	dimx = meshmax[0] - meshmin[0] + 2;
	dimy = meshmax[1] - meshmin[1] + 2;
	dimz = meshmax[2] - meshmin[2] + 2;

	pm_init_periodic_allocate((dimx + 4) * (dimy + 4) * (dimz + 4));

#ifdef DYNAMICAL_DE
	/* Insert non-blocking send/receive statements for rhogrid_DE and ugrid_DE */

#ifdef DEBUG
	DE_array_debug();
#endif
	check_omegas();

	MPI_Request *comm_reqs=NULL;
	MPI_Status *status_DE=NULL;
	if(nslab_x>0){
		comm_reqs=malloc(2*(nslabs_to_send+4)*sizeof(MPI_Request));
		status_DE=malloc(2*(nslabs_to_send+4)*sizeof(MPI_Status));

		/* Copy rhogrid_DE to rhogrid_tot. Note rhogrid_DE is an nslab_x*PMGRID*PMGRID array whilce rhogrid_tot is in the fftw format nslab_x*PMGRID*PMGRID2 */
		for( i=0 ; i<nslab_x ; ++i )
			for( j=0 ; j<PMGRID ; j++ )
				memcpy(rhogrid_tot+INDMAP(i,j,0),rhogrid_DE+i*PMGRID*PMGRID+j*PMGRID,PMGRID*sizeof(fftw_real));	

		/* Send slabs */
		unsigned int slab;

		char buf[128];
		char out[512]="Send information\n";
		unsigned short req_count=0;
		for( i=0 ; i<nslabs_to_send ; ++i )
		{
			slab=slabs_to_send[i]-slabstart_x;
			/* Send ugrid slabs. Tagged with the absolute slab index */
			MPI_Isend(ugrid_DE+slab*PMGRID*PMGRID,3*PMGRID*PMGRID*sizeof(fftw_real),MPI_BYTE,send_tasks[i],slabs_to_send[i]       ,MPI_COMM_WORLD,&comm_reqs[req_count++]);
			/* Send rhogrid_DE slabs. Tagged with absolute slab_index + PMGRID so as to not coincide with the ugrid tag */
			MPI_Isend(rhogrid_DE+slab*PMGRID*PMGRID,PMGRID*PMGRID*sizeof(fftw_real),MPI_BYTE,send_tasks[i],slabs_to_send[i]+PMGRID,MPI_COMM_WORLD,&comm_reqs[req_count++]);
			sprintf(buf,"*Sending slab %i to task %i\n",slabs_to_send[i],send_tasks[i]);
			strcat(out,buf);
		}

		/* Receive slabs */
		unsigned int recv_index;
		for( i=0 ; i<4 ; ++i )
		{
			if(slabs_to_recv[i]==LOGICAL_INDEX(slabstart_x-2))
				recv_index=0;
			else if(slabs_to_recv[i]==LOGICAL_INDEX(slabstart_x-1))
				recv_index=1;
			else if(slabs_to_recv[i]==LOGICAL_INDEX(slabstart_x+nslab_x))
				recv_index=2;
			else if(slabs_to_recv[i]==LOGICAL_INDEX(slabstart_x+nslab_x+1))
				recv_index=3;
			else{
				mpi_fprintf(stderr,"Error in determining which slab to receive (slab to receive: %i, nslabs: %i, slabstart: %i)\n",slabs_to_recv[i],nslab_x,slabstart_x);
				endrun(1);
			}

			sprintf(buf,"*Receiving slab %i from task %i\n",slabs_to_recv[i],recv_tasks[i]);
			strcat(out,buf);

			/* Receive ugrid slabs. Tagged with the absolute slab index */
			MPI_Irecv(U_recv+recv_index*3*PMGRID*PMGRID,3*PMGRID*PMGRID*sizeof(fftw_real),MPI_BYTE,recv_tasks[i],slabs_to_recv[i],MPI_COMM_WORLD,&comm_reqs[req_count++]);
			/* Receive rhogrid_DE slabs. Tagged with absolute slab_index + PMGRID so as to not coincide with the ugrid tag */
			MPI_Irecv(rho_recv+recv_index*PMGRID*PMGRID,PMGRID*PMGRID*sizeof(fftw_real),MPI_BYTE,recv_tasks[i],slabs_to_recv[i]+PMGRID,MPI_COMM_WORLD,&comm_reqs[req_count++]);
		}
		mpi_printf("%s\n",out);
	}

#endif

	for(i = 0; i < dimx * dimy * dimz; i++)
		workspace[i] = 0;

	for(i = 0; i < NumPart; i++)
	{
		slab_x = to_slab_fac * P[i].Pos[0];
		if(slab_x >= PMGRID)
			slab_x = PMGRID - 1;
		dx = to_slab_fac * P[i].Pos[0] - slab_x;
		slab_x -= meshmin[0];
		slab_xx = slab_x + 1;

		slab_y = to_slab_fac * P[i].Pos[1];
		if(slab_y >= PMGRID)
			slab_y = PMGRID - 1;
		dy = to_slab_fac * P[i].Pos[1] - slab_y;
		slab_y -= meshmin[1];
		slab_yy = slab_y + 1;

		slab_z = to_slab_fac * P[i].Pos[2];
		if(slab_z >= PMGRID)
			slab_z = PMGRID - 1;
		dz = to_slab_fac * P[i].Pos[2] - slab_z;
		slab_z -= meshmin[2];
		slab_zz = slab_z + 1;

		workspace[(slab_x * dimy + slab_y) * dimz + slab_z] += P[i].Mass * (1.0 - dx) * (1.0 - dy) * (1.0 - dz);
		workspace[(slab_x * dimy + slab_yy) * dimz + slab_z] += P[i].Mass * (1.0 - dx) * dy * (1.0 - dz);
		workspace[(slab_x * dimy + slab_y) * dimz + slab_zz] += P[i].Mass * (1.0 - dx) * (1.0 - dy) * dz;
		workspace[(slab_x * dimy + slab_yy) * dimz + slab_zz] += P[i].Mass * (1.0 - dx) * dy * dz;

		workspace[(slab_xx * dimy + slab_y) * dimz + slab_z] += P[i].Mass * (dx) * (1.0 - dy) * (1.0 - dz);
		workspace[(slab_xx * dimy + slab_yy) * dimz + slab_z] += P[i].Mass * (dx) * dy * (1.0 - dz);
		workspace[(slab_xx * dimy + slab_y) * dimz + slab_zz] += P[i].Mass * (dx) * (1.0 - dy) * dz;
		workspace[(slab_xx * dimy + slab_yy) * dimz + slab_zz] += P[i].Mass * (dx) * dy * dz;
	}


	for(i = 0; i < fftsize; i++)	/* clear local density field */
		rhogrid[i] = 0;

#ifdef DYNAMICAL_DE
	/* Wait point for non-blocking exchange of DE slabs, only continue when all rhogrid_DE and ugrid_DE slabs haven been exchanged.
	 * Still need to calculate and exchange rhogrid_tot for the total potential */

	if(nslab_x>0)
		MPI_Waitall(2*(nslabs_to_send+4),comm_reqs,status_DE);

	free(comm_reqs);
	comm_reqs=NULL;
	free(status_DE);
	status_DE=NULL;
#endif

	for(level = 0; level < (1 << PTask); level++)	/* note: for level=0, target is the same task */
	{
		sendTask = ThisTask;
		recvTask = ThisTask ^ level;
		if(recvTask < NTask)
		{
			/* check how much we have to send */
			sendmin = 2 * PMGRID;
			sendmax = -1;
			for(slab_x = meshmin[0]; slab_x < meshmax[0] + 2; slab_x++)
				if(slab_to_task[slab_x % PMGRID] == recvTask)
				{
					if(slab_x < sendmin)
						sendmin = slab_x;
					if(slab_x > sendmax)
						sendmax = slab_x;
				}
			if(sendmax == -1)
				sendmin = 0;

			/* check how much we have to receive */
			recvmin = 2 * PMGRID;
			recvmax = -1;
			for(slab_x = meshmin_list[3 * recvTask]; slab_x < meshmax_list[3 * recvTask] + 2; slab_x++)
				if(slab_to_task[slab_x % PMGRID] == sendTask)
				{
					if(slab_x < recvmin)
						recvmin = slab_x;
					if(slab_x > recvmax)
						recvmax = slab_x;
				}
			if(recvmax == -1)
				recvmin = 0;


			if((recvmax - recvmin) >= 0 || (sendmax - sendmin) >= 0)	/* ok, we have a contribution to the slab */
			{
				recv_dimx = meshmax_list[3 * recvTask + 0] - meshmin_list[3 * recvTask + 0] + 2;
				recv_dimy = meshmax_list[3 * recvTask + 1] - meshmin_list[3 * recvTask + 1] + 2;
				recv_dimz = meshmax_list[3 * recvTask + 2] - meshmin_list[3 * recvTask + 2] + 2;

				if(level > 0) /* workspace is the sending buffer in both cases */
				{
					MPI_Sendrecv(workspace + (sendmin - meshmin[0]) * dimy * dimz,
							(sendmax - sendmin + 1) * dimy * dimz * sizeof(fftw_real), MPI_BYTE, recvTask,
							TAG_PERIODIC_A, forcegrid,
							(recvmax - recvmin + 1) * recv_dimy * recv_dimz * sizeof(fftw_real), MPI_BYTE,
							recvTask, TAG_PERIODIC_A, MPI_COMM_WORLD, &status);
				}
				else
				{
					memcpy(forcegrid, workspace + (sendmin - meshmin[0]) * dimy * dimz,
							(sendmax - sendmin + 1) * dimy * dimz * sizeof(fftw_real));
				}

				for(slab_x = recvmin; slab_x <= recvmax; slab_x++)
				{
					slab_xx = (slab_x % PMGRID) - first_slab_of_task[ThisTask];

					if(slab_xx >= 0 && slab_xx < slabs_per_task[ThisTask])
					{
						for(slab_y = meshmin_list[3 * recvTask + 1];
								slab_y <= meshmax_list[3 * recvTask + 1] + 1; slab_y++)
						{
							slab_yy = slab_y;
							if(slab_yy >= PMGRID)
								slab_yy -= PMGRID;

							for(slab_z = meshmin_list[3 * recvTask + 2];
									slab_z <= meshmax_list[3 * recvTask + 2] + 1; slab_z++)
							{
								slab_zz = slab_z;
								if(slab_zz >= PMGRID)
									slab_zz -= PMGRID;

								rhogrid[PMGRID * PMGRID2 * slab_xx + PMGRID2 * slab_yy + slab_zz] +=
									forcegrid[((slab_x - recvmin) * recv_dimy +
											(slab_y - meshmin_list[3 * recvTask + 1])) * recv_dimz +
									(slab_z - meshmin_list[3 * recvTask + 2])];
							}
						}
					}
				}
			}
		}
	}

#ifdef DYNAMICAL_DE
#ifdef DEBUG
	for(i = 0; i < nslab_x; i++)	
		for(j = 0; j < PMGRID; j++)
			for(k = 0; k < PMGRID; k++)
				rhogrid[INDMAP(i,j,k)] = (i+slabstart_x)*PMGRID*PMGRID+j*PMGRID+k;
	dbg_print(0,rho_recv,rhogrid);

#endif
#endif

	/* Do the FFT of the density field */

	rfftwnd_mpi(fft_forward_plan, 1, rhogrid, workspace, FFTW_TRANSPOSED_ORDER);
#ifdef DYNAMICAL_DE
	/* Do the FFT of the rhogrid_DE (rhogrid_DE has already been copied to the rhogrid_tot) */
	rfftwnd_mpi(fft_forward_plan, 1, rhogrid_tot, workspace, FFTW_TRANSPOSED_ORDER);
	fftw_complex rho_temp;

#ifdef DEBUG
	master_printf("***Starting FFT***\n");
#endif
#endif
	/* multiply with Green's function for the potential, deconvolve */
	for(y = slabstart_y; y < slabstart_y + nslab_y; y++)
		for(x = 0; x < PMGRID; x++)
			for(z = 0; z < PMGRID / 2 + 1; z++)
			{
				if(x > PMGRID / 2)
					kx = x - PMGRID;
				else
					kx = x;
				if(y > PMGRID / 2)
					ky = y - PMGRID;
				else
					ky = y;
				if(z > PMGRID / 2)
					kz = z - PMGRID;
				else
					kz = z;

				k2 = kx * kx + ky * ky + kz * kz; /* Note: k2 is the integer wave number squared. The physical k is k_phys=2 M_PI/BoxSize k */

#ifndef DEBUG
				if(k2 > 0)
#endif
				{
					ip = PMGRID * (PMGRID / 2 + 1) * (y - slabstart_y) + (PMGRID / 2 + 1) * x + z;
					smth = -exp(-k2 * asmth2) / k2; /* -4 M_PI G/k2 is the Green's function for the Poisson equation */
					/* TODO: Add 3P term from DE Poisson? */
#ifdef DYNAMICAL_DE
#ifdef DEBUG
					smth=1.0;
#endif
					/* Long range smoothing of the dark energy part */
					fft_of_rhogrid_DE[ip].re *= smth;
					fft_of_rhogrid_DE[ip].im *= smth;
#endif	
					/* do deconvolution */
					/* Note: Actual deconvolution is sinc(k_phys BoxSize/(2*PMGRID)), but in the code k = k_phys/(2*M_MPI)*BoxSize  */
					fx = fy = fz = 1;
					if(kx != 0)
					{
						fx = (M_PI * kx) / PMGRID;
						fx = sin(fx) / fx;
					}
					if(ky != 0)
					{
						fy = (M_PI * ky) / PMGRID;
						fy = sin(fy) / fy;
					}
					if(kz != 0)
					{
						fz = (M_PI * kz) / PMGRID;
						fz = sin(fz) / fz;
					}
					ff = 1 / (fx * fy * fz);
					smth *=  ff * ff; 

#ifdef DYNAMICAL_DE
#ifdef DEBUG
					smth=1.0;
					ff=1.0;
#endif
#endif
					fft_of_rhogrid[ip].re *= smth;
					fft_of_rhogrid[ip].im *= smth;
					/* Have now done one deconvolution of the dark matter potential */
#ifdef DYNAMICAL_DE
#ifndef DEBUG
					/* Deconvolve rhogrid_DE once and add it to rhogrid (which has been deconvolved twice). 
					 * Deconvolve rhogrid once and add it to rhogrid_tot, also add rhogrid_DE unconvolved */

					/* TODO: IMPORTANT: Consistent units before addition */
					rho_temp=fft_of_rhogrid_DE[ip];
					fft_of_rhogrid_DE[ip].re += fft_of_rhogrid[ip].re;
					fft_of_rhogrid_DE[ip].im += fft_of_rhogrid[ip].im;

					fft_of_rhogrid[ip].re += ff*ff*rho_temp.re;
					fft_of_rhogrid[ip].im += ff*ff*rho_temp.im;
#endif
#endif
					/* Now do second deconvolution of dark matter potential */
					fft_of_rhogrid[ip].re *= ff * ff;
					fft_of_rhogrid[ip].im *= ff * ff;
					/* end deconvolution */
				}

			}

	/* TODO: REMEMBER to uncomment */
#ifdef NEVER
	if(slabstart_y == 0) /* This sets the mean to zero, meaning that we get the relative density delta_rho (since the k=0 part is the constant contribution) */
		fft_of_rhogrid[0].re = fft_of_rhogrid[0].im = 0.0;

#ifdef DYNAMICAL_DE
	if(slabstart_y == 0) 
		fft_of_rhogrid_DE[0].re = fft_of_rhogrid_DE[0].im = 0.0;
#endif

#endif

	/* Do the FFT to get the potential */

	rfftwnd_mpi(fft_inverse_plan, 1, rhogrid, workspace, FFTW_TRANSPOSED_ORDER);
	/* Note: The inverse FFT scales the data by PMGRID*PMGRID*PMGRID */
#ifdef DYNAMICAL_DE
	/* Do the FFT of rhogrid_tot to get the total potential of the singly deconvolved dm+de */
	rfftwnd_mpi(fft_inverse_plan, 1, rhogrid_tot, workspace, FFTW_TRANSPOSED_ORDER);
#endif

#ifdef DYNAMICAL_DE
#ifdef DEBUG
	for( i=0 ; i<fftsize ; ++i )
	{
		rhogrid[i]/=PMGRID*PMGRID*PMGRID;
		rhogrid_tot[i]/=PMGRID*PMGRID*PMGRID;
	}
#endif
#endif

	/* Now rhogrid holds the potential */
	/* construct the potential for the local patch */

	dimx = meshmax[0] - meshmin[0] + 6;
	dimy = meshmax[1] - meshmin[1] + 6;
	dimz = meshmax[2] - meshmin[2] + 6;

	for(level = 0; level < (1 << PTask); level++)	/* note: for level=0, target is the same task */
	{
		sendTask = ThisTask;
		recvTask = ThisTask ^ level;

		if(recvTask < NTask)
		{

			/* check how much we have to send */
			sendmin = 2 * PMGRID;
			sendmax = -PMGRID;
			for(slab_x = meshmin_list[3 * recvTask] - 2; slab_x < meshmax_list[3 * recvTask] + 4; slab_x++)
				if(slab_to_task[(slab_x + PMGRID) % PMGRID] == sendTask)
				{
					if(slab_x < sendmin)
						sendmin = slab_x;
					if(slab_x > sendmax)
						sendmax = slab_x;
				}
			if(sendmax == -PMGRID)
				sendmin = sendmax + 1;


			/* check how much we have to receive */
			recvmin = 2 * PMGRID;
			recvmax = -PMGRID;
			for(slab_x = meshmin[0] - 2; slab_x < meshmax[0] + 4; slab_x++)
				if(slab_to_task[(slab_x + PMGRID) % PMGRID] == recvTask)
				{
					if(slab_x < recvmin)
						recvmin = slab_x;
					if(slab_x > recvmax)
						recvmax = slab_x;
				}
			if(recvmax == -PMGRID)
				recvmin = recvmax + 1;

			if((recvmax - recvmin) >= 0 || (sendmax - sendmin) >= 0)	/* ok, we have a contribution to the slab */
			{
				recv_dimx = meshmax_list[3 * recvTask + 0] - meshmin_list[3 * recvTask + 0] + 6;
				recv_dimy = meshmax_list[3 * recvTask + 1] - meshmin_list[3 * recvTask + 1] + 6;
				recv_dimz = meshmax_list[3 * recvTask + 2] - meshmin_list[3 * recvTask + 2] + 6;

				ncont = 1;
				cont_sendmin[0] = sendmin;
				cont_sendmax[0] = sendmax;
				cont_sendmin[1] = sendmax + 1;
				cont_sendmax[1] = sendmax;

				cont_recvmin[0] = recvmin;
				cont_recvmax[0] = recvmax;
				cont_recvmin[1] = recvmax + 1;
				cont_recvmax[1] = recvmax;

				for(slab_x = sendmin; slab_x <= sendmax; slab_x++)
				{
					if(slab_to_task[(slab_x + PMGRID) % PMGRID] != ThisTask)
					{
						/* non-contiguous */
						cont_sendmax[0] = slab_x - 1;
						while(slab_to_task[(slab_x + PMGRID) % PMGRID] != ThisTask)
							slab_x++;
						cont_sendmin[1] = slab_x;
						ncont++;
					}
				}

				for(slab_x = recvmin; slab_x <= recvmax; slab_x++)
				{
					if(slab_to_task[(slab_x + PMGRID) % PMGRID] != recvTask)
					{
						/* non-contiguous */
						cont_recvmax[0] = slab_x - 1;
						while(slab_to_task[(slab_x + PMGRID) % PMGRID] != recvTask)
							slab_x++;
						cont_recvmin[1] = slab_x;
						if(ncont == 1)
							ncont++;
					}
				}


				for(rep = 0; rep < ncont; rep++)
				{
					sendmin = cont_sendmin[rep];
					sendmax = cont_sendmax[rep];
					recvmin = cont_recvmin[rep];
					recvmax = cont_recvmax[rep];

					/* prepare what we want to send */
					if(sendmax - sendmin >= 0)
					{
						for(slab_x = sendmin; slab_x <= sendmax; slab_x++)
						{
							slab_xx = ((slab_x + PMGRID) % PMGRID) - first_slab_of_task[ThisTask];

							for(slab_y = meshmin_list[3 * recvTask + 1] - 2;
									slab_y < meshmax_list[3 * recvTask + 1] + 4; slab_y++)
							{
								slab_yy = (slab_y + PMGRID) % PMGRID;

								for(slab_z = meshmin_list[3 * recvTask + 2] - 2;
										slab_z < meshmax_list[3 * recvTask + 2] + 4; slab_z++)
								{
									slab_zz = (slab_z + PMGRID) % PMGRID;

									forcegrid[((slab_x - sendmin) * recv_dimy +
											(slab_y - (meshmin_list[3 * recvTask + 1] - 2))) * recv_dimz +
										slab_z - (meshmin_list[3 * recvTask + 2] - 2)] =
										rhogrid[PMGRID * PMGRID2 * slab_xx + PMGRID2 * slab_yy + slab_zz];
								}
							}
						}
					}

					if(level > 0) /* workspace is the receiving buffer in both cases, any data present gets overwritten */
					{
						MPI_Sendrecv(forcegrid,
								(sendmax - sendmin + 1) * recv_dimy * recv_dimz * sizeof(fftw_real),
								MPI_BYTE, recvTask, TAG_PERIODIC_B,
								workspace + (recvmin - (meshmin[0] - 2)) * dimy * dimz,
								(recvmax - recvmin + 1) * dimy * dimz * sizeof(fftw_real), MPI_BYTE,
								recvTask, TAG_PERIODIC_B, MPI_COMM_WORLD, &status);
					}
					else
					{
						memcpy(workspace + (recvmin - (meshmin[0] - 2)) * dimy * dimz,
								forcegrid, (recvmax - recvmin + 1) * dimy * dimz * sizeof(fftw_real));
					}
				}
			}
		}
	}

	dimx = meshmax[0] - meshmin[0] + 2;
	dimy = meshmax[1] - meshmin[1] + 2;
	dimz = meshmax[2] - meshmin[2] + 2;

	recv_dimx = meshmax[0] - meshmin[0] + 6;
	recv_dimy = meshmax[1] - meshmin[1] + 6;
	recv_dimz = meshmax[2] - meshmin[2] + 6;

#ifdef DYNAMICAL_DE
	/* Send/receive rhogrid_tot slabs (which now are the singly deconvolved potential) 
	 * Note that phi_recv, unlike rho_recv and u_recv, is in the fftw slab format (y,z dimensions PMGRID, PMGRID2 respectively)
	 * that rhogrid_tot is also in (whereas rhogrid_DE and ugrid_DE is in the normal PMGRID, PMGRID format).
	 * This makes the indexing consistent over the different arrays - at the price of communication of some junk-data ( (PMGRID2-PMGRID)*PMGRID*PMGRID*sizeof(fftw_real) bytes)*/
	if(nslab_x>0){
		comm_reqs=malloc((nslabs_to_send+4)*sizeof(MPI_Request));
		status_DE=malloc((nslabs_to_send+4)*sizeof(MPI_Status));

		/* Send slabs */
		unsigned int slab;
		unsigned short req_count=0;
		for( i=0 ; i<nslabs_to_send ; ++i )
		{
			slab=slabs_to_send[i]-slabstart_x;
			/* Send rhogrid_tot slabs. */ 
			MPI_Isend(rhogrid_tot+slab*PMGRID*PMGRID2,PMGRID*PMGRID2*sizeof(fftw_real),MPI_BYTE,send_tasks[i],slabs_to_send[i],MPI_COMM_WORLD,&comm_reqs[req_count++]);
		}

		/* Receive slabs */
		unsigned int recv_index;
		for( i=0 ; i<4 ; ++i )
		{
			if(slabs_to_recv[i]==LOGICAL_INDEX(slabstart_x-2))
				recv_index=0;
			else if(slabs_to_recv[i]==LOGICAL_INDEX(slabstart_x-1))
				recv_index=1;
			else if(slabs_to_recv[i]==LOGICAL_INDEX(slabstart_x+nslab_x))
				recv_index=2;
			else if(slabs_to_recv[i]==LOGICAL_INDEX(slabstart_x+nslab_x+1))
				recv_index=3;
			else{
				mpi_fprintf(stderr,"Error in determining which slab to receive (slab to receive: %i, nslabs: %i, slabstart: %i)\n",slabs_to_recv[i],nslab_x,slabstart_x);
				endrun(1);
			}
			/* Receive rhogrid_tot slabs (called phi for convenience since it is really the potential now) */
			MPI_Irecv(phi_recv+recv_index*PMGRID*PMGRID2,PMGRID*PMGRID2*sizeof(fftw_real),MPI_BYTE,recv_tasks[i],slabs_to_recv[i],MPI_COMM_WORLD,&comm_reqs[req_count++]);
		}
	}
#endif


	/*TODO: DElete entire block */
#ifdef DYNAMICAL_DE
#ifdef DEBUG
	if(nslab_x>0)
		MPI_Waitall(nslabs_to_send+4,comm_reqs,status_DE);
	dbg_print_fftw_format(0,phi_recv,rhogrid_tot);
	dbg_print_fftw_format(slab_to_task[PMGRID-1],phi_recv,rhogrid_tot);
	endrun(1);
#endif
#endif

	for(dim = 0; dim < 3; dim++)	/* Calculate each component of the force. */
	{
		/* get the force component by finite differencing the potential */
		/* note: "workspace" now contains the potential for the local patch, plus a suffiently large buffer region */

		for(x = 0; x < meshmax[0] - meshmin[0] + 2; x++)
			for(y = 0; y < meshmax[1] - meshmin[1] + 2; y++)
				for(z = 0; z < meshmax[2] - meshmin[2] + 2; z++)
				{
					xrr = xll = xr = xl = x;
					yrr = yll = yr = yl = y;
					zrr = zll = zr = zl = z;

					switch (dim)
					{
						case 0:
							xr = x + 1;
							xrr = x + 2;
							xl = x - 1;
							xll = x - 2;
							break;
						case 1:
							yr = y + 1;
							yl = y - 1;
							yrr = y + 2;
							yll = y - 2;
							break;
						case 2:
							zr = z + 1;
							zl = z - 1;
							zrr = z + 2;
							zll = z - 2;
							break;
					}

					forcegrid[(x * dimy + y) * dimz + z]
						=
						fac * ((4.0 / 3) *
								(workspace[((xl + 2) * recv_dimy + (yl + 2)) * recv_dimz + (zl + 2)]
								 - workspace[((xr + 2) * recv_dimy + (yr + 2)) * recv_dimz + (zr + 2)]) -
								(1.0 / 6) *
								(workspace[((xll + 2) * recv_dimy + (yll + 2)) * recv_dimz + (zll + 2)] -
								 workspace[((xrr + 2) * recv_dimy + (yrr + 2)) * recv_dimz + (zrr + 2)]));
				}

		/* read out the forces */

		for(i = 0; i < NumPart; i++)
		{
			slab_x = to_slab_fac * P[i].Pos[0];
			if(slab_x >= PMGRID)
				slab_x = PMGRID - 1;
			dx = to_slab_fac * P[i].Pos[0] - slab_x;
			slab_x -= meshmin[0];
			slab_xx = slab_x + 1;

			slab_y = to_slab_fac * P[i].Pos[1];
			if(slab_y >= PMGRID)
				slab_y = PMGRID - 1;
			dy = to_slab_fac * P[i].Pos[1] - slab_y;
			slab_y -= meshmin[1];
			slab_yy = slab_y + 1;

			slab_z = to_slab_fac * P[i].Pos[2];
			if(slab_z >= PMGRID)
				slab_z = PMGRID - 1;
			dz = to_slab_fac * P[i].Pos[2] - slab_z;
			slab_z -= meshmin[2];
			slab_zz = slab_z + 1;

			acc_dim =
				forcegrid[(slab_x * dimy + slab_y) * dimz + slab_z] * (1.0 - dx) * (1.0 - dy) * (1.0 - dz);
			acc_dim += forcegrid[(slab_x * dimy + slab_yy) * dimz + slab_z] * (1.0 - dx) * dy * (1.0 - dz);
			acc_dim += forcegrid[(slab_x * dimy + slab_y) * dimz + slab_zz] * (1.0 - dx) * (1.0 - dy) * dz;
			acc_dim += forcegrid[(slab_x * dimy + slab_yy) * dimz + slab_zz] * (1.0 - dx) * dy * dz;

			acc_dim += forcegrid[(slab_xx * dimy + slab_y) * dimz + slab_z] * (dx) * (1.0 - dy) * (1.0 - dz);
			acc_dim += forcegrid[(slab_xx * dimy + slab_yy) * dimz + slab_z] * (dx) * dy * (1.0 - dz);
			acc_dim += forcegrid[(slab_xx * dimy + slab_y) * dimz + slab_zz] * (dx) * (1.0 - dy) * dz;
			acc_dim += forcegrid[(slab_xx * dimy + slab_yy) * dimz + slab_zz] * (dx) * dy * dz;

			P[i].GravPM[dim] = acc_dim;
		}
	}
#ifdef DYNAMICAL_DE
	/* Wait point for non-blocking exchange of rhogrid_tot slabs.
	 * Calculate the next PM timestep and update the DE equations - use finite differences */
	if(nslab_x>0)
		MPI_Waitall(nslabs_to_send+4,comm_reqs,status_DE);
	free(comm_reqs);
	comm_reqs=NULL;
	free(status_DE);
	status_DE=NULL;

	advance_DE();
#endif

	pm_init_periodic_free();
	force_treeallocate(All.TreeAllocFactor * All.MaxPart, All.MaxPart);

	All.NumForcesSinceLastDomainDecomp = 1 + All.TotNumPart * All.TreeDomainUpdateFrequency;

	if(ThisTask == 0)
	{
		printf("done PM.\n");
		fflush(stdout);
	}
}


/*! Calculates the long-range potential using the PM method.  The potential is
 *  Gaussian filtered with Asmth, given in mesh-cell units. We carry out a CIC
 *  charge assignment, and compute the potenial by Fourier transform
 *  methods. The CIC kernel is deconvolved.
 */
void pmpotential_periodic(void)
{
	double k2, kx, ky, kz, smth;
	double dx, dy, dz;
	double fx, fy, fz, ff;
	double asmth2, fac;
	int i, j, slab, level, sendTask, recvTask;
	int x, y, z, ip;
	int slab_x, slab_y, slab_z;
	int slab_xx, slab_yy, slab_zz;
	int meshmin[3], meshmax[3], sendmin, sendmax, recvmin, recvmax;
	int rep, ncont, cont_sendmin[2], cont_sendmax[2], cont_recvmin[2], cont_recvmax[2];
	int dimx, dimy, dimz, recv_dimx, recv_dimy, recv_dimz;
	MPI_Status status;

	if(ThisTask == 0)
	{
		printf("Starting periodic PM calculation.\n");
		fflush(stdout);
	}

	asmth2 = (2 * M_PI) * All.Asmth[0] / All.BoxSize;
	asmth2 *= asmth2;

	fac = All.G / (M_PI * All.BoxSize);	/* to get potential */

	force_treefree();

	/* first, establish the extension of the local patch in the PMGRID  */

	for(j = 0; j < 3; j++)
	{
		meshmin[j] = PMGRID;
		meshmax[j] = 0;
	}

	for(i = 0; i < NumPart; i++)
	{
		for(j = 0; j < 3; j++)
		{
			slab = to_slab_fac * P[i].Pos[j];
			if(slab >= PMGRID)
				slab = PMGRID - 1;

			if(slab < meshmin[j])
				meshmin[j] = slab;

			if(slab > meshmax[j])
				meshmax[j] = slab;
		}
	}

	MPI_Allgather(meshmin, 3, MPI_INT, meshmin_list, 3, MPI_INT, MPI_COMM_WORLD);
	MPI_Allgather(meshmax, 3, MPI_INT, meshmax_list, 3, MPI_INT, MPI_COMM_WORLD);

	dimx = meshmax[0] - meshmin[0] + 2;
	dimy = meshmax[1] - meshmin[1] + 2;
	dimz = meshmax[2] - meshmin[2] + 2;

	pm_init_periodic_allocate((dimx + 4) * (dimy + 4) * (dimz + 4));

	for(i = 0; i < dimx * dimy * dimz; i++)
		workspace[i] = 0;

	for(i = 0; i < NumPart; i++)
	{
		slab_x = to_slab_fac * P[i].Pos[0];
		if(slab_x >= PMGRID)
			slab_x = PMGRID - 1;
		dx = to_slab_fac * P[i].Pos[0] - slab_x;
		slab_x -= meshmin[0];
		slab_xx = slab_x + 1;

		slab_y = to_slab_fac * P[i].Pos[1];
		if(slab_y >= PMGRID)
			slab_y = PMGRID - 1;
		dy = to_slab_fac * P[i].Pos[1] - slab_y;
		slab_y -= meshmin[1];
		slab_yy = slab_y + 1;

		slab_z = to_slab_fac * P[i].Pos[2];
		if(slab_z >= PMGRID)
			slab_z = PMGRID - 1;
		dz = to_slab_fac * P[i].Pos[2] - slab_z;
		slab_z -= meshmin[2];
		slab_zz = slab_z + 1;

		workspace[(slab_x * dimy + slab_y) * dimz + slab_z] += P[i].Mass * (1.0 - dx) * (1.0 - dy) * (1.0 - dz);
		workspace[(slab_x * dimy + slab_yy) * dimz + slab_z] += P[i].Mass * (1.0 - dx) * dy * (1.0 - dz);
		workspace[(slab_x * dimy + slab_y) * dimz + slab_zz] += P[i].Mass * (1.0 - dx) * (1.0 - dy) * dz;
		workspace[(slab_x * dimy + slab_yy) * dimz + slab_zz] += P[i].Mass * (1.0 - dx) * dy * dz;

		workspace[(slab_xx * dimy + slab_y) * dimz + slab_z] += P[i].Mass * (dx) * (1.0 - dy) * (1.0 - dz);
		workspace[(slab_xx * dimy + slab_yy) * dimz + slab_z] += P[i].Mass * (dx) * dy * (1.0 - dz);
		workspace[(slab_xx * dimy + slab_y) * dimz + slab_zz] += P[i].Mass * (dx) * (1.0 - dy) * dz;
		workspace[(slab_xx * dimy + slab_yy) * dimz + slab_zz] += P[i].Mass * (dx) * dy * dz;
	}


	for(i = 0; i < fftsize; i++)	/* clear local density field */
		rhogrid[i] = 0;

	for(level = 0; level < (1 << PTask); level++)	/* note: for level=0, target is the same task */
	{
		sendTask = ThisTask;
		recvTask = ThisTask ^ level;
		if(recvTask < NTask)
		{
			/* check how much we have to send */
			sendmin = 2 * PMGRID;
			sendmax = -1;
			for(slab_x = meshmin[0]; slab_x < meshmax[0] + 2; slab_x++)
				if(slab_to_task[slab_x % PMGRID] == recvTask)
				{
					if(slab_x < sendmin)
						sendmin = slab_x;
					if(slab_x > sendmax)
						sendmax = slab_x;
				}
			if(sendmax == -1)
				sendmin = 0;

			/* check how much we have to receive */
			recvmin = 2 * PMGRID;
			recvmax = -1;
			for(slab_x = meshmin_list[3 * recvTask]; slab_x < meshmax_list[3 * recvTask] + 2; slab_x++)
				if(slab_to_task[slab_x % PMGRID] == sendTask)
				{
					if(slab_x < recvmin)
						recvmin = slab_x;
					if(slab_x > recvmax)
						recvmax = slab_x;
				}
			if(recvmax == -1)
				recvmin = 0;


			if((recvmax - recvmin) >= 0 || (sendmax - sendmin) >= 0)	/* ok, we have a contribution to the slab */
			{
				recv_dimx = meshmax_list[3 * recvTask + 0] - meshmin_list[3 * recvTask + 0] + 2;
				recv_dimy = meshmax_list[3 * recvTask + 1] - meshmin_list[3 * recvTask + 1] + 2;
				recv_dimz = meshmax_list[3 * recvTask + 2] - meshmin_list[3 * recvTask + 2] + 2;

				if(level > 0)
				{
					MPI_Sendrecv(workspace + (sendmin - meshmin[0]) * dimy * dimz,
							(sendmax - sendmin + 1) * dimy * dimz * sizeof(fftw_real), MPI_BYTE, recvTask,
							TAG_PERIODIC_C, forcegrid,
							(recvmax - recvmin + 1) * recv_dimy * recv_dimz * sizeof(fftw_real), MPI_BYTE,
							recvTask, TAG_PERIODIC_C, MPI_COMM_WORLD, &status);
				}
				else
				{
					memcpy(forcegrid, workspace + (sendmin - meshmin[0]) * dimy * dimz,
							(sendmax - sendmin + 1) * dimy * dimz * sizeof(fftw_real));
				}

				for(slab_x = recvmin; slab_x <= recvmax; slab_x++)
				{
					slab_xx = (slab_x % PMGRID) - first_slab_of_task[ThisTask];

					if(slab_xx >= 0 && slab_xx < slabs_per_task[ThisTask])
					{
						for(slab_y = meshmin_list[3 * recvTask + 1];
								slab_y <= meshmax_list[3 * recvTask + 1] + 1; slab_y++)
						{
							slab_yy = slab_y;
							if(slab_yy >= PMGRID)
								slab_yy -= PMGRID;

							for(slab_z = meshmin_list[3 * recvTask + 2];
									slab_z <= meshmax_list[3 * recvTask + 2] + 1; slab_z++)
							{
								slab_zz = slab_z;
								if(slab_zz >= PMGRID)
									slab_zz -= PMGRID;

								rhogrid[PMGRID * PMGRID2 * slab_xx + PMGRID2 * slab_yy + slab_zz] +=
									forcegrid[((slab_x - recvmin) * recv_dimy +
											(slab_y - meshmin_list[3 * recvTask + 1])) * recv_dimz +
									(slab_z - meshmin_list[3 * recvTask + 2])];
							}
						}
					}
				}
			}
		}
	}



	/* Do the FFT of the density field */

	rfftwnd_mpi(fft_forward_plan, 1, rhogrid, workspace, FFTW_TRANSPOSED_ORDER);

	/* multiply with Green's function for the potential */

	for(y = slabstart_y; y < slabstart_y + nslab_y; y++)
		for(x = 0; x < PMGRID; x++)
			for(z = 0; z < PMGRID / 2 + 1; z++)
			{
				if(x > PMGRID / 2)
					kx = x - PMGRID;
				else
					kx = x;
				if(y > PMGRID / 2)
					ky = y - PMGRID;
				else
					ky = y;
				if(z > PMGRID / 2)
					kz = z - PMGRID;
				else
					kz = z;

				k2 = kx * kx + ky * ky + kz * kz;

				if(k2 > 0)
				{
					smth = -exp(-k2 * asmth2) / k2 * fac;
					/* do deconvolution */
					fx = fy = fz = 1;
					if(kx != 0)
					{
						fx = (M_PI * kx) / PMGRID;
						fx = sin(fx) / fx;
					}
					if(ky != 0)
					{
						fy = (M_PI * ky) / PMGRID;
						fy = sin(fy) / fy;
					}
					if(kz != 0)
					{
						fz = (M_PI * kz) / PMGRID;
						fz = sin(fz) / fz;
					}
					ff = 1 / (fx * fy * fz);
					smth *= ff * ff * ff * ff;
					/* end deconvolution */

					ip = PMGRID * (PMGRID / 2 + 1) * (y - slabstart_y) + (PMGRID / 2 + 1) * x + z;
					fft_of_rhogrid[ip].re *= smth;
					fft_of_rhogrid[ip].im *= smth;
				}
			}

	if(slabstart_y == 0)
		fft_of_rhogrid[0].re = fft_of_rhogrid[0].im = 0.0;

	/* Do the FFT to get the potential */

	rfftwnd_mpi(fft_inverse_plan, 1, rhogrid, workspace, FFTW_TRANSPOSED_ORDER);

	/* note: "rhogrid" now contains the potential */



	dimx = meshmax[0] - meshmin[0] + 6;
	dimy = meshmax[1] - meshmin[1] + 6;
	dimz = meshmax[2] - meshmin[2] + 6;

	for(level = 0; level < (1 << PTask); level++)	/* note: for level=0, target is the same task */
	{
		sendTask = ThisTask;
		recvTask = ThisTask ^ level;

		if(recvTask < NTask)
		{

			/* check how much we have to send */
			sendmin = 2 * PMGRID;
			sendmax = -PMGRID;
			for(slab_x = meshmin_list[3 * recvTask] - 2; slab_x < meshmax_list[3 * recvTask] + 4; slab_x++)
				if(slab_to_task[(slab_x + PMGRID) % PMGRID] == sendTask)
				{
					if(slab_x < sendmin)
						sendmin = slab_x;
					if(slab_x > sendmax)
						sendmax = slab_x;
				}
			if(sendmax == -PMGRID)
				sendmin = sendmax + 1;


			/* check how much we have to receive */
			recvmin = 2 * PMGRID;
			recvmax = -PMGRID;
			for(slab_x = meshmin[0] - 2; slab_x < meshmax[0] + 4; slab_x++)
				if(slab_to_task[(slab_x + PMGRID) % PMGRID] == recvTask)
				{
					if(slab_x < recvmin)
						recvmin = slab_x;
					if(slab_x > recvmax)
						recvmax = slab_x;
				}
			if(recvmax == -PMGRID)
				recvmin = recvmax + 1;

			if((recvmax - recvmin) >= 0 || (sendmax - sendmin) >= 0)	/* ok, we have a contribution to the slab */
			{
				recv_dimx = meshmax_list[3 * recvTask + 0] - meshmin_list[3 * recvTask + 0] + 6;
				recv_dimy = meshmax_list[3 * recvTask + 1] - meshmin_list[3 * recvTask + 1] + 6;
				recv_dimz = meshmax_list[3 * recvTask + 2] - meshmin_list[3 * recvTask + 2] + 6;

				ncont = 1;
				cont_sendmin[0] = sendmin;
				cont_sendmax[0] = sendmax;
				cont_sendmin[1] = sendmax + 1;
				cont_sendmax[1] = sendmax;

				cont_recvmin[0] = recvmin;
				cont_recvmax[0] = recvmax;
				cont_recvmin[1] = recvmax + 1;
				cont_recvmax[1] = recvmax;

				for(slab_x = sendmin; slab_x <= sendmax; slab_x++)
				{
					if(slab_to_task[(slab_x + PMGRID) % PMGRID] != ThisTask)
					{
						/* non-contiguous */
						cont_sendmax[0] = slab_x - 1;
						while(slab_to_task[(slab_x + PMGRID) % PMGRID] != ThisTask)
							slab_x++;
						cont_sendmin[1] = slab_x;
						ncont++;
					}
				}

				for(slab_x = recvmin; slab_x <= recvmax; slab_x++)
				{
					if(slab_to_task[(slab_x + PMGRID) % PMGRID] != recvTask)
					{
						/* non-contiguous */
						cont_recvmax[0] = slab_x - 1;
						while(slab_to_task[(slab_x + PMGRID) % PMGRID] != recvTask)
							slab_x++;
						cont_recvmin[1] = slab_x;
						if(ncont == 1)
							ncont++;
					}
				}


				for(rep = 0; rep < ncont; rep++)
				{
					sendmin = cont_sendmin[rep];
					sendmax = cont_sendmax[rep];
					recvmin = cont_recvmin[rep];
					recvmax = cont_recvmax[rep];

					/* prepare what we want to send */
					if(sendmax - sendmin >= 0)
					{
						for(slab_x = sendmin; slab_x <= sendmax; slab_x++)
						{
							slab_xx = ((slab_x + PMGRID) % PMGRID) - first_slab_of_task[ThisTask];

							for(slab_y = meshmin_list[3 * recvTask + 1] - 2;
									slab_y < meshmax_list[3 * recvTask + 1] + 4; slab_y++)
							{
								slab_yy = (slab_y + PMGRID) % PMGRID;

								for(slab_z = meshmin_list[3 * recvTask + 2] - 2;
										slab_z < meshmax_list[3 * recvTask + 2] + 4; slab_z++)
								{
									slab_zz = (slab_z + PMGRID) % PMGRID;

									forcegrid[((slab_x - sendmin) * recv_dimy +
											(slab_y - (meshmin_list[3 * recvTask + 1] - 2))) * recv_dimz +
										slab_z - (meshmin_list[3 * recvTask + 2] - 2)] =
										rhogrid[PMGRID * PMGRID2 * slab_xx + PMGRID2 * slab_yy + slab_zz];
								}
							}
						}
					}

					if(level > 0)
					{
						MPI_Sendrecv(forcegrid,
								(sendmax - sendmin + 1) * recv_dimy * recv_dimz * sizeof(fftw_real),
								MPI_BYTE, recvTask, TAG_PERIODIC_D,
								workspace + (recvmin - (meshmin[0] - 2)) * dimy * dimz,
								(recvmax - recvmin + 1) * dimy * dimz * sizeof(fftw_real), MPI_BYTE,
								recvTask, TAG_PERIODIC_D, MPI_COMM_WORLD, &status);
					}
					else
					{
						memcpy(workspace + (recvmin - (meshmin[0] - 2)) * dimy * dimz,
								forcegrid, (recvmax - recvmin + 1) * dimy * dimz * sizeof(fftw_real));
					}
				}
			}
		}
	}


	dimx = meshmax[0] - meshmin[0] + 2;
	dimy = meshmax[1] - meshmin[1] + 2;
	dimz = meshmax[2] - meshmin[2] + 2;

	recv_dimx = meshmax[0] - meshmin[0] + 6;
	recv_dimy = meshmax[1] - meshmin[1] + 6;
	recv_dimz = meshmax[2] - meshmin[2] + 6;



	for(x = 0; x < meshmax[0] - meshmin[0] + 2; x++)
		for(y = 0; y < meshmax[1] - meshmin[1] + 2; y++)
			for(z = 0; z < meshmax[2] - meshmin[2] + 2; z++)
			{
				forcegrid[(x * dimy + y) * dimz + z] =
					workspace[((x + 2) * recv_dimy + (y + 2)) * recv_dimz + (z + 2)];
			}


	/* read out the potential */

	for(i = 0; i < NumPart; i++)
	{
		slab_x = to_slab_fac * P[i].Pos[0];
		if(slab_x >= PMGRID)
			slab_x = PMGRID - 1;
		dx = to_slab_fac * P[i].Pos[0] - slab_x;
		slab_x -= meshmin[0];
		slab_xx = slab_x + 1;

		slab_y = to_slab_fac * P[i].Pos[1];
		if(slab_y >= PMGRID)
			slab_y = PMGRID - 1;
		dy = to_slab_fac * P[i].Pos[1] - slab_y;
		slab_y -= meshmin[1];
		slab_yy = slab_y + 1;

		slab_z = to_slab_fac * P[i].Pos[2];
		if(slab_z >= PMGRID)
			slab_z = PMGRID - 1;
		dz = to_slab_fac * P[i].Pos[2] - slab_z;
		slab_z -= meshmin[2];
		slab_zz = slab_z + 1;

		P[i].Potential +=
			forcegrid[(slab_x * dimy + slab_y) * dimz + slab_z] * (1.0 - dx) * (1.0 - dy) * (1.0 - dz);
		P[i].Potential += forcegrid[(slab_x * dimy + slab_yy) * dimz + slab_z] * (1.0 - dx) * dy * (1.0 - dz);
		P[i].Potential += forcegrid[(slab_x * dimy + slab_y) * dimz + slab_zz] * (1.0 - dx) * (1.0 - dy) * dz;
		P[i].Potential += forcegrid[(slab_x * dimy + slab_yy) * dimz + slab_zz] * (1.0 - dx) * dy * dz;

		P[i].Potential += forcegrid[(slab_xx * dimy + slab_y) * dimz + slab_z] * (dx) * (1.0 - dy) * (1.0 - dz);
		P[i].Potential += forcegrid[(slab_xx * dimy + slab_yy) * dimz + slab_z] * (dx) * dy * (1.0 - dz);
		P[i].Potential += forcegrid[(slab_xx * dimy + slab_y) * dimz + slab_zz] * (dx) * (1.0 - dy) * dz;
		P[i].Potential += forcegrid[(slab_xx * dimy + slab_yy) * dimz + slab_zz] * (dx) * dy * dz;
	}

	pm_init_periodic_free();
	force_treeallocate(All.TreeAllocFactor * All.MaxPart, All.MaxPart);

	All.NumForcesSinceLastDomainDecomp = 1 + All.TotNumPart * All.TreeDomainUpdateFrequency;

	if(ThisTask == 0)
	{
		printf("done PM-Potential.\n");
		fflush(stdout);
	}
}

#ifdef DYNAMICAL_DE
void advance_DE(void ){

	const double fac = 1 / (2 * All.BoxSize / PMGRID);	/* for finite differencing */
	unsigned int i;






	/* TODO: Last thing: Add extra mass to DE since DE mass grows in a comoving cell */

}

void check_omegas(void){
		const double rhocrit=3*All.Hubble*All.Hubble/(8*M_PI*All.G);
		const double OmegaLambda=All.OmegaLambda/pow(All.Time,3.0*(1+All.DarkEnergyW));
		double OmegaActual;
		double M=0;
		int i;
		
		for( i=0 ; i<PMGRID*PMGRID*PMGRID ; ++i )
		{
			M+=rhogrid_DE[i];
		}
		MPI_Allreduce(MPI_IN_PLACE,&M,nslab_x*PMGRID*PMGRID,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
		OmegaActual=1.0/(All.BoxSize*All.BoxSize*All.BoxSize*All.Time*All.Time*All.Time);
		OmegaActual*=M/rhocrit;
		if(fabs(OmegaLambda-OmegaActual)>1.0e-3){
			master_fprintf(stderr,"Error: Mass content of dynamic dark energy is OmegaLambda=%d, expected OmegaLambda=%d\n",OmegaActual,OmegaLambda); 
			endrun(1);
		}
		master_printf("OmegaMass at the current timestep is %d, OmegaLambda is %d\n",All.Omega0/(All.Time*All.Time*All.Time),OmegaLambda);

}
#ifdef DEBUG
void dbg_print(int task, fftw_real *recv_arr,fftw_real *rho_arr){
	if(ThisTask==task){
		fftw_real *ll_arr=recv_arr;
		fftw_real *l_arr=recv_arr+PMGRID*PMGRID;
		fftw_real *r_arr=recv_arr+2*PMGRID*PMGRID;
		fftw_real *rr_arr=recv_arr+3*PMGRID*PMGRID;
		printf("Task %i send/recv info\n",ThisTask);
		int second_slab= (nslab_x>1) ? 1 : 0; 
		int last_slab= (nslab_x>1) ? nslab_x-1 : 0; 
		int second_last_slab= (nslab_x>1) ? nslab_x-2 : 0; 
		int i,j;
		printf("Slab:\t\t%5i   %5i   %5i   %5i     %5i   %5i   %5i   %5i\n",slabs_to_recv[0],slabs_to_recv[1],slabstart_x,slabstart_x+second_slab,slabstart_x+second_last_slab,slabstart_x+last_slab,slabs_to_recv[2],slabs_to_recv[3]);
		for( i=0, j=0 ; i<PMGRID ; ++i )
			for( j=0 ; j<PMGRID ; ++j ){
				printf("y=%.3i,z=%.3i\t%li   %li | %li   %li ... %li   %li | %li   %li\n",i,j
						,(long int) ll_arr[i*PMGRID+j],(long int) l_arr[i*PMGRID+j]
						,(long int) rho_arr[INDMAP(0,i,j)],(long int) rho_arr[INDMAP(second_slab,i,j)]
						,(long int) rho_arr[INDMAP(second_last_slab,i,j)],(long int) rho_arr[INDMAP(last_slab,i,j)]
						,(long int) r_arr[i*PMGRID+j],(long int) rr_arr[i*PMGRID+j]);
			}
	}

	MPI_Barrier(MPI_COMM_WORLD);

}

void dbg_print_fftw_format(int task, fftw_real *recv_arr,fftw_real *rho_arr){
	if(ThisTask==task){
		fftw_real *ll_arr=recv_arr;
		fftw_real *l_arr=recv_arr+PMGRID*PMGRID2;
		fftw_real *r_arr=recv_arr+2*PMGRID*PMGRID2;
		fftw_real *rr_arr=recv_arr+3*PMGRID*PMGRID2;
		printf("Task %i send/recv info\n",ThisTask);
		int second_slab= (nslab_x>1) ? 1 : 0; 
		int last_slab= (nslab_x>1) ? nslab_x-1 : 0; 
		int second_last_slab= (nslab_x>1) ? nslab_x-2 : 0; 
		int i,j;
		printf("Slab:\t\t%5i   %5i   %5i   %5i     %5i   %5i   %5i   %5i\n",slabs_to_recv[0],slabs_to_recv[1],slabstart_x,slabstart_x+second_slab,slabstart_x+second_last_slab,slabstart_x+last_slab,slabs_to_recv[2],slabs_to_recv[3]);
		for( i=0, j=0 ; i<PMGRID ; ++i )
			for( j=0 ; j<PMGRID ; ++j ){
				//		#define _INDMAP(i,j,k) ((i)*PMGRID*PMGRID+(j)*PMGRID+k) /* Map (i,j,k) in 3 dimensional array (dimx X PMGRID X PMGRID2) to 1 dimensional array */
				printf("y=%.3i,z=%.3i\t%li   %li | %li   %li ... %li   %li | %li   %li\n",i,j
						,(long int) ll_arr[i*PMGRID2+j],(long int) l_arr[i*PMGRID2+j]
						,(long int) rho_arr[INDMAP(0,i,j)],(long int) rho_arr[INDMAP(second_slab,i,j)]
						,(long int) rho_arr[INDMAP(second_last_slab,i,j)],(long int) rho_arr[INDMAP(last_slab,i,j)]
						,(long int) r_arr[i*PMGRID2+j],(long int) rr_arr[i*PMGRID2+j]);
			}
	}

	MPI_Barrier(MPI_COMM_WORLD);

}
#endif
#endif

#endif
#endif



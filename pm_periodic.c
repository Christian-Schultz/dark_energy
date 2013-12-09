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

/* Check FFTW datatypes and associate MPI datatypes */
#ifdef FFTW_MPITYPE 
	#error "FFTW_MPITYPE already defined"
#else
	#ifdef DOUBLEPRECISION_FFTW
		#define FFTW_MPITYPE MPI_DOUBLE
	#else
		#define FFTW_MPITYPE MPI_FLOAT
	#endif
#endif

/* Dark energy macros */
#define LOGICAL_INDEX(x)  ((x<0) ? x+PMGRID : (x>=PMGRID) ? x-PMGRID : x ) /* Map index to the range [0,PMGRID[ */
#define INDMAP(i,j,k) ((i)*PMGRID*PMGRID2+(j)*PMGRID2+k) /* Map (i,j,k) in 3 dimensional array (dimx X PMGRID X PMGRID2) to 1 dimensional array */

#ifdef DEBUG
#define SHOUT(x) do{if(x) mpi_printf("SHOUT: !(" #x ")\n");} while(0)
void pm_stats(char *);
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

static FLOAT to_slab_fac;

void CIC(int*,int*);

#ifdef DYNAMICAL_DE
static short int first_DE_run=1;
void DE_IC(void);  /* Dark energy initial conditions */
void advance_DE(fftw_real); /* Function prototype for the routine responsible for advancing the dark energy density and velocity perturbations (rhodot and Udot) */

void write_header(FILE *);
void write_dm_grid(char *);
void write_de_grid(char *);
void write_U_grid(char *);

static int recv_tasks[4]; /* The 4 tasks that have the slabs this task needs (ordered left left, left, right, right right) */
static int send_tasks[6]; /* The (up to 6) tasks that needs this task's slabs. Only in the case where some tasks, but not all, only have 1 slab it is neccessary to communicate with 6 others, otherwise this is normally 4*/
static int slabs_to_send[6]; /* The slabs this task needs to send in the order defined in send_tasks */
static int slabs_to_recv[4]; /* The slabs this task needs to receive in the order defined in recv_tasks */
static int nslabs_to_send;  /* How many slabs does this task need to send? Normally 4, but possibly up to 6 if some tasks only have 1 slab */

/* The dark energy arrays come in 2 forms: An array corresponding to the actual slabs of the task, and an expanded array
 * with space for 4 extra slabs (2 in each end) used for finite differencing */
static fftw_real *rhogrid_tot, *rhogrid_tot_expanded, *dPgrid_fftw,*dPgrid_fftw_expanded; /* fftw-format arrays. Note that the P arrays are P/c^2 */
static fftw_real *rhogrid_DE_expanded, (*ugrid_DE_expanded)[3]; /* Normal format arrays */
static fftw_real *rhogrid_DE, (*ugrid_DE)[3], *dPgrid; /* Normal format arrays */
static fftw_complex *fft_of_rhogrid_tot, *fft_of_dPgrid; /* fft of the total dark energy and dark matter density. Also the fft of the velocity divergence (which will become the pressure)*/

/* Mean densities of dark matter and dark energy */
static fftw_real mean_DM, mean_DE;
/* Establish which tasks have to communicate with each other to populate the expanded FFTW arrays
 * (if the current task, for example, has the slabs from 4 to 8 it needs slabs 2 and 3 from the task(s) to its "left"
 * and slabs 9 and 10 from the task(s) to its "right"). The extra slabs are used in the finite differencing of the different 
 * dark energy quantities */
int comm_order(int nslabs){
	if(nslabs>0){
		int index=LOGICAL_INDEX(slabstart_x-2);
		slabs_to_recv[0]=index;
		int my_task_ll=slab_to_task[index];

		index=LOGICAL_INDEX(slabstart_x-1);
		slabs_to_recv[1]=index;
		int my_task_l=slab_to_task[index];
		
		index=LOGICAL_INDEX(slabstart_x+nslab_x);
		slabs_to_recv[2]=index;
		int my_task_r=slab_to_task[index];

		index=LOGICAL_INDEX(slabstart_x+nslab_x+1);
		slabs_to_recv[3]=index;
		int my_task_rr=slab_to_task[index];

		/* The tasks that has the four slabs this task needs */
		recv_tasks[0]=my_task_ll;
		recv_tasks[1]=my_task_l;
		recv_tasks[2]=my_task_r;
		recv_tasks[3]=my_task_rr;

		/* Now find the tasks that need this task's slabs (could be more than 4!) */

		int task, first_slab,last_slab,i;

		if(ThisTask==0){
			for( task=0 ; task<NTask ; ++task )
			{
				if(slabs_per_task[task]<4){
					master_fprintf(stderr,"Warning: A bad number of tasks has been chosen to run the PM part of the code.\n"
							"Normally you want each task to have at the very least 4 slabs. "
							 );
					break;
				}
			}
			if(PMGRID<5 ){
				master_fprintf(stderr,"Need more than 4 slabs to run the dark energy code. Increase PMGRID\n");
				endrun(1);
			}
		}

		int slabs[4];
		int Nsend=0; /* Number of slabs to send. */
		for( task=0 ; task<NTask ; ++task )
		{
			if(slabs_per_task[task]==0)
				continue;
			if(task==ThisTask && NTask!=1)
				continue;
			/* Note: Program will hang if there is only 2 tasks in the FFT pool, and one task only has 1 slab. 
			 * This should be impossible however, due to the requirement that a minimum of 4 slabs is needed */

			first_slab=first_slab_of_task[task];
			last_slab=first_slab+slabs_per_task[task]-1;

			slabs[0]=LOGICAL_INDEX(first_slab-2); /* your ll slab */
			slabs[1]=LOGICAL_INDEX(first_slab-1); /* your l slab */
			slabs[2]=LOGICAL_INDEX(last_slab+1);  /* your r slab */
			slabs[3]=LOGICAL_INDEX(last_slab+2);  /* your rr slab */

			for( i=0 ; i<4 ; ++i )
			{
				if( (slabs[i]>=slabstart_x) && (slabs[i]<= (slabstart_x+nslab_x-1)) ){ /* Overlap between my slabs and your needed slabs */
					send_tasks[Nsend]=task;
					slabs_to_send[Nsend]=slabs[i];
					++Nsend;
				}
			}

		}
#ifdef DEBUG
		if(Nsend>4)
			mpi_printf("WARNING: Sending more than 4 slabs. Is there a processor with only a single slab?\n");
#endif
		unsigned int slab;
		char buf[128];
		char out[1024];
		sprintf(out,"Communication information (My slabs: %i to %i)\n",slabstart_x,slabstart_x+nslabs-1);
		for( i=0 ; i<Nsend ; ++i )
		{
			sprintf(buf,"*Sending slab %i to task %i\n",slabs_to_send[i],send_tasks[i]);
			strcat(out,buf);
		}

		for( i=0 ; i<4 ; ++i )
		{
			slab=LOGICAL_INDEX(slabs_to_recv[i]-slabstart_x+2);
			sprintf(buf,"*Receiving slab %i from task %i (local slab index: %i)\n",slabs_to_recv[i],recv_tasks[i],slab);
			strcat(out,buf);
		}
		mpi_printf("%s\n",out);

		return Nsend;
	}
	else
		return 0;
}

/* One-time allocation of the dark energy arrays */
void DE_allocate(int nx){

	master_printf("Code compiled with DYNAMICAL_DE, setting up dark energy environment\n");

	if(nx>0){
		/* Expand the arrays with 4 extra slabs to store communication */
		rhogrid_DE_expanded=my_malloc((4+nx)*PMGRID*PMGRID*sizeof(fftw_real));
		ugrid_DE_expanded=my_malloc(3*(4+nx)*PMGRID*PMGRID*sizeof(fftw_real));
		dPgrid=my_malloc(nx*PMGRID*PMGRID*sizeof(fftw_real));

		/* Arrays corresponding to the actual slabs of this task */
		rhogrid_DE=rhogrid_DE_expanded+2*PMGRID*PMGRID;
		ugrid_DE=ugrid_DE_expanded+2*PMGRID*PMGRID;
		const unsigned long int size=(4*(4+nx)+nx)*PMGRID*PMGRID*sizeof(fftw_real);
		mpi_printf("Allocated %lu bytes (%lu MB) for DE arrays\n",size,size/(1024*1024));
	}
	else{
		rhogrid_DE_expanded=NULL;
		rhogrid_DE=NULL;
		ugrid_DE_expanded=NULL;
		ugrid_DE=NULL;
		dPgrid=NULL;
	}
}

void PM_cleanup(int nx){ /* Like free_memory(), this is not actually called by the program */
	free(rhogrid_DE_expanded);
	free(ugrid_DE_expanded);
	free(dPgrid);
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

	/* Workspace out the ranges on each task. */

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
	/* Allocate extra memory for the dark energy part and establish communication order */
	nslabs_to_send=comm_order(nslab_x);
#endif
}

#ifdef DYNAMICAL_DE
void DE_periodic_allocate(void){
	/* Expand with 4 slabs to store communication data */
	if(nslab_x>0){
#ifdef DEBUG
		assert(fftsize==nslab_x*PMGRID2*PMGRID);
#endif
		rhogrid_tot_expanded=my_malloc((fftsize+4*PMGRID2*PMGRID) * sizeof(fftw_real));
		dPgrid_fftw_expanded=my_malloc((fftsize+4*PMGRID2*PMGRID) * sizeof(fftw_real));

		/* rhogrid_tot is only the local array, rhogrid_tot_expanded is the local array and the 2 slabs before and 2 after */
		rhogrid_tot=& rhogrid_tot_expanded[INDMAP(2,0,0)];
		dPgrid_fftw=& dPgrid_fftw_expanded[INDMAP(2,0,0)];

		fft_of_rhogrid_tot = (fftw_complex *) & rhogrid_tot[0];
		fft_of_dPgrid = (fftw_complex *) & dPgrid_fftw[0];
	}
	else
	{
		rhogrid_tot_expanded=rhogrid_tot=NULL;
		dPgrid_fftw_expanded=dPgrid_fftw=NULL;
		fft_of_rhogrid_tot=NULL;
		fft_of_dPgrid=NULL;
	}
	if(first_DE_run){
		const unsigned int size=2*(fftsize+4*PMGRID2*PMGRID)*sizeof(fftw_real);
		master_printf("PM force with dark energy toggled (time=%f). Allocated %u bytes (%u MB) for dark energy temporary storage\n",All.Time,size,size/(1024*1024));
	}
}
#endif

/*! This function allocates the memory neeed to compute the long-range PM
 *  force. Three fields are used, one to hold the density (and its FFT, and
 *  then the real-space potential), one to hold the force field obtained by
 *  finite differencing, and finally a workspace field, which is used both as
 *  workspace for the parallel FFT, and as buffer for the communication
 *  algorithm used in the force computation.
 */
void pm_init_periodic_allocate(int dimprod)
{
	static int first_alloc=1;
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

	if(first_alloc==1)
	{
		first_alloc=0;
		if(ThisTask == 0)
			printf("\nAllocated %g MByte for FFT data.\n\n", bytes_tot / (1024.0 * 1024.0));
	}
	fft_of_rhogrid = (fftw_complex *) & rhogrid[0];
}

/*! This routine frees the space allocated for the parallel FFT algorithm (except dark energy arrays).
*/
void pm_init_periodic_free(void)
{
	/* allocate the memory to hold the FFT fields */
	free(workspace);
	free(forcegrid);
	free(rhogrid);
}

#ifdef DYNAMICAL_DE
/* Free dark energy arrays */
void free_dark_energy(void){
	free(rhogrid_tot_expanded);
	rhogrid_tot_expanded=NULL;
	rhogrid_tot=NULL;

	free(dPgrid_fftw_expanded);
	dPgrid_fftw_expanded=NULL;
	dPgrid_fftw=NULL;
}
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
	int i, j, slab, level, sendTask, recvTask;
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


	asmth2 = (2 * M_PI) * All.Asmth[0] / All.BoxSize;
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
					smth = -exp(-k2 * asmth2) / k2;

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

					if(level > 0)
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

	pm_init_periodic_free();
	force_treeallocate(All.TreeAllocFactor * All.MaxPart, All.MaxPart);

	All.NumForcesSinceLastDomainDecomp = 1 + All.TotNumPart * All.TreeDomainUpdateFrequency;

	if(ThisTask == 0)
	{
		printf("done PM.\n");
		fflush(stdout);
	}
}

/* pmforce in a dynamical dark energy cosmology */
#ifdef DYNAMICAL_DE
/* TODO: Fix code when running on a single process (probably in comm order) */
void pmforce_periodic_DE(void)
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
	char statname[MAXLEN_FILENAME];

	double fac_FD, temp; /* Merge temp with fftw_real divU */

	if(ThisTask == 0)
	{
		printf("Starting periodic PM calculation with dynamical dark energy.\n");
		fflush(stdout);
	}

	mean_DM=All.Omega0*3.0*All.Hubble*All.Hubble/(8.0*M_PI*All.G)/pow(All.Time,3.0); /* Mean dark matter density in the universe */
	mean_DE=All.OmegaLambda*3.0*All.Hubble*All.Hubble/(8.0*M_PI*All.G)/pow(All.Time,3.0*(1+All.DarkEnergyW)); /* Mean dark energy density in the universe */

	force_treefree();

	asmth2 = (2 * M_PI) * All.Asmth[0] / All.BoxSize; /* Default value for Asmth is 1.25, can be changed in the Makefile */
	asmth2 *= asmth2;

	fac = All.G / (M_PI * All.BoxSize);	/* to get potential */
	fac_FD = 1 / (2 * All.BoxSize / PMGRID);	/* for finite differencing */
	fac *=fac_FD;

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
	DE_periodic_allocate();
	if(first_DE_run){
		temp=All.DarkEnergySoundSpeed*All.DarkEnergySoundSpeed;
		temp=sqrt(temp/(3*(1+All.DarkEnergyW)*(temp-All.DarkEnergyW)));
		master_printf("Dark energy sound speed: %f, dark energy equation of state: %f\n"
				"Dark energy gauge transformation important for a< %e (z > %e)\n"				
				,All.DarkEnergySoundSpeed,All.DarkEnergyW,
				temp,1/temp-1
			     );
		DE_allocate(nslab_x);
		DE_IC();
	}
	/* Non-blocking send/receive statements for rhogrid_DE and ugrid_DE */

	MPI_Request *comm_reqs=NULL;
	MPI_Status *status_DE=NULL;
	const double vol_fac=(All.BoxSize/PMGRID)*(All.BoxSize/PMGRID)*(All.BoxSize/PMGRID)*(All.Time*All.Time*All.Time); /* Physical volume factor. Converts from density to mass */		
	if(nslab_x>0){
		comm_reqs=malloc(2*(nslabs_to_send+4)*sizeof(MPI_Request)); /* Received/not received status of non-blocking messages (communication handles) */
		status_DE=malloc(2*(nslabs_to_send+4)*sizeof(MPI_Status)); /* The MPI_Status return values of the communication */

		/* Assign dark energy mass to rhogrid_tot. Will (implicitly) be translated to total density.
		 * Note rhogrid_DE is an nslab_x*PMGRID*PMGRID array while rhogrid_tot is in the fftw format nslab_x*PMGRID*PMGRID2 */
		for( i=0 ; i<nslab_x ; ++i )
			for( j=0 ; j<PMGRID ; j++ )/* Change from delta_rho to delta_mass.*/
				for( k=0 ; k<PMGRID ; ++k )
					rhogrid_tot[INDMAP(i,j,k)]=rhogrid_DE[i*PMGRID*PMGRID+j*PMGRID+k]*vol_fac;



		unsigned short req_count=0;
		/* Send slabs one at a time */
		for( i=0 ; i<nslabs_to_send ; ++i )
		{
			slab=slabs_to_send[i]-slabstart_x;
			/* Send ugrid slabs. Tagged with the absolute slab index */
			MPI_Isend(ugrid_DE+slab*PMGRID*PMGRID  ,3*PMGRID*PMGRID*sizeof(fftw_real),MPI_BYTE,send_tasks[i],slabs_to_send[i]       ,MPI_COMM_WORLD,&comm_reqs[req_count++]);
			/* Send rhogrid_DE slabs. Tagged with absolute slab index + PMGRID so as to not coincide with the ugrid tag */
			MPI_Isend(rhogrid_DE+slab*PMGRID*PMGRID,PMGRID*PMGRID*sizeof(fftw_real)  ,MPI_BYTE,send_tasks[i],slabs_to_send[i]+PMGRID,MPI_COMM_WORLD,&comm_reqs[req_count++]);
		}

		/* Receive slabs. Store in expanded array */
		for( i=0 ; i<4 ; ++i )
		{
			slab=LOGICAL_INDEX(slabs_to_recv[i]-slabstart_x+2); /* Index minus start of local patch + 2 since the received patch starts at relative index -2  */
			/* Receive ugrid slabs. Tagged with the absolute slab index */
			MPI_Irecv(ugrid_DE_expanded+slab*PMGRID*PMGRID,3*PMGRID*PMGRID*sizeof(fftw_real),MPI_BYTE,recv_tasks[i],slabs_to_recv[i]       ,MPI_COMM_WORLD,&comm_reqs[req_count++]);
			/* Receive rhogrid_DE slabs. Tagged with absolute slab index + PMGRID so as to not coincide with the ugrid tag */
			MPI_Irecv(rhogrid_DE_expanded+slab*PMGRID*PMGRID,PMGRID*PMGRID*sizeof(fftw_real),MPI_BYTE,recv_tasks[i],slabs_to_recv[i]+PMGRID,MPI_COMM_WORLD,&comm_reqs[req_count++]);
		}
	}

	/* Cloud-in-Cell interpolation. Moved to its own function for readability.
	 * Assigns mass to rhogrid while the dark energy grids are being exchanged. */
	CIC(meshmin,meshmax);


	/* Wait point for non-blocking exchange of DE slabs, only continue when all rhogrid_DE and ugrid_DE slabs have been exchanged.
	 * Still need to calculate and exchange rhogrid_tot for the total potential and dPgrid for the pressure pertubation */
	if(nslab_x>0){
		if(MPI_SUCCESS!=MPI_Waitall(2*(nslabs_to_send+4),comm_reqs,status_DE)){
			mpi_fprintf(stderr,"Error in MPI_Waitall (%s: %i)\n",__FILE__,__LINE__);
			endrun(1);
		}
	}

	free(comm_reqs);
	comm_reqs=NULL;
	free(status_DE);
	status_DE=NULL;

	/* Prepare for dark energy gauge transformation.
	 * Calculate divergence of U and store it in dPgrid_fftw.*/
	fftw_real divU=0; /* Merge this with temp */
	for( x=2 ; x<nslab_x+2 ; ++x ) /* Loop over slabs in expanded array */
		for( y=0 ; y<PMGRID ; ++y )
			for( z=0 ; z<PMGRID ; ++z ){
				divU=0;
				for( dim=0 ; dim<3 ; ++dim ) /* Loop over x,y,z components of the gradients */
				{
					xrr = xll = xr = xl = x;
					yrr = yll = yr = yl = y;
					zrr = zll = zr = zl = z;

					switch (dim)
					{
						case 0:
							xr  = x + 1;
							xrr = x + 2;
							xl  = x - 1;
							xll = x - 2;
							break;
						case 1:
							yr  = LOGICAL_INDEX(y + 1);
							yl  = LOGICAL_INDEX(y - 1);
							yrr = LOGICAL_INDEX(y + 2);
							yll = LOGICAL_INDEX(y - 2);
							break;
						case 2:
							zr  = LOGICAL_INDEX(z + 1);
							zl  = LOGICAL_INDEX(z - 1);
							zrr = LOGICAL_INDEX(z + 2);
							zll = LOGICAL_INDEX(z - 2);
							break;
					}
					divU+=fac_FD*(
							(4.0/3.0)*
							( - ugrid_DE_expanded[(xl * PMGRID + yl) * PMGRID + zl][dim]
							  + ugrid_DE_expanded[(xr * PMGRID + yr) * PMGRID + zr][dim]
							)
							+
							(1.0 / 6.0) *
							( ugrid_DE_expanded[(xll * PMGRID + yll) * PMGRID + zll][dim] 
							  - ugrid_DE_expanded[(xrr * PMGRID + yrr) * PMGRID + zrr][dim]
							)
						     );
				}
				dPgrid_fftw_expanded[INDMAP(x,y,z)]=divU;
			}

#ifdef DEBUG
	char fname_DE[256];
	char fname_DM[256];
	char fname_U[256];
	static int Nruns=0;
	sprintf(fname_DE,"%sDE_a=%.3f.%.3i",All.OutputDir,All.Time,ThisTask);
	sprintf(fname_DM,"%sDM_a=%.3f.%.3i",All.OutputDir,All.Time,ThisTask);
	sprintf(fname_U,"%sU_a=%.3f.%.3i",All.OutputDir,All.Time,ThisTask);
	if(All.Time>All.DarkEnergyOutputStart && Nruns<All.DarkEnergyNumOutputs){
		master_printf("Writing dm+de grids\n");
		write_dm_grid(fname_DM);
		write_de_grid(fname_DE);
		write_U_grid(fname_U);
		++Nruns;
	}
#endif

	/* Print simulation statistics to file */
	sprintf(statname,"%s%s",All.OutputDir,All.DarkEnergyStatFile);
	pm_stats(statname);

	/* Do the FFT of the dark matter density field */
	rfftwnd_mpi(fft_forward_plan, 1, rhogrid, workspace, FFTW_TRANSPOSED_ORDER);

	/* Do the FFT of the dark energy density field (the dark energy density field has been converted to mass in rhogrid_tot) */
	rfftwnd_mpi(fft_forward_plan, 1, rhogrid_tot, workspace, FFTW_TRANSPOSED_ORDER);

	/* Do the FFT of the divergence of U */
	rfftwnd_mpi(fft_forward_plan, 1, dPgrid_fftw, workspace, FFTW_TRANSPOSED_ORDER);

#ifdef DEBUG
	if(slabstart_y==0){
		double tmp=(double) fft_of_rhogrid[0].re/(PMGRID*PMGRID*PMGRID);
		printf("Dark matter mean: %e (mean rho: %e)\n", tmp, tmp/vol_fac);
		tmp=fft_of_rhogrid_tot[0].re/(PMGRID*PMGRID*PMGRID);
		printf("Dark energy mean: %e (mean rho: %e)\n",tmp, tmp/vol_fac);
		tmp=fft_of_dPgrid[0].re/(PMGRID*PMGRID*PMGRID);
		printf("Dark energy pressure mean: %e\n",tmp);
	}
#endif

	/* Enforce mean of pressure perturbation to vanish in case divergence U doesn't */
	if(slabstart_y == 0)
		fft_of_dPgrid[0].re = fft_of_dPgrid[0].im = 0.0;

	/* Dark energy pressure conversion factors */
	const double lightspeed=C/All.UnitVelocity_in_cm_per_s;
	const double H=All.Hubble*sqrt(All.Omega0 / (All.Time * All.Time * All.Time) + (1 - All.Omega0 - All.OmegaLambda) / (All.Time * All.Time) + All.OmegaLambda/pow(All.Time,3.0*(1+All.DarkEnergyW)));
	const double P_prefactor=3*All.Time*H*(1+All.DarkEnergyW)*(All.DarkEnergySoundSpeed*All.DarkEnergySoundSpeed-All.DarkEnergyW)*mean_DE/(lightspeed*lightspeed);

	/* Dummy variable to store rho */
	fftw_complex rho_temp;
	/* Conversion from integer k to comoving k. Needs additional 1/scalefactor to be physical k */
	const double k2_conv=2*M_PI/(All.BoxSize)*2*M_PI/(All.BoxSize);
	double k2_actual;
#ifdef DEBUG
	fftw_complex P_std, P_gauge_std;
	P_std.re=P_std.im=P_gauge_std.re=P_gauge_std.im=0;
	int bad_points=0;
	short trigger=0;
#endif
	/* multiply with the Green's function for the potential, deconvolve */
	for(y = slabstart_y; y < slabstart_y + nslab_y; y++)
		for(x = 0; x < PMGRID; x++)
			for(z = 0; z < PMGRID / 2 + 1; z++)
			{
				k2_actual=x*x+y*y+z*z;
				k2_actual*=k2_conv;

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

				/* k2=0 iff x=y=z=0 iff k2_actual=0, hence k2=0 iff k2_actual=0 */
				if(k2 > 0)
				{
					/* do smoothing and deconvolution */
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

					ip = PMGRID * (PMGRID / 2 + 1) * (y - slabstart_y) + (PMGRID / 2 + 1) * x + z;


					fft_of_rhogrid[ip].re *= ff*ff;
					fft_of_rhogrid[ip].im *= ff*ff;

					/* Have now done one deconvolution of the dark matter potential (corresponding to the CIC from the particles to grid) */
					fft_of_dPgrid[ip].re=All.DarkEnergySoundSpeed*All.DarkEnergySoundSpeed*fft_of_rhogrid_tot[ip].re/vol_fac
						+P_prefactor*fft_of_dPgrid[ip].re/k2_actual;
					fft_of_dPgrid[ip].im=All.DarkEnergySoundSpeed*All.DarkEnergySoundSpeed*fft_of_rhogrid_tot[ip].im/vol_fac
						+P_prefactor*fft_of_dPgrid[ip].im/k2_actual;
#ifdef DEBUG
					temp=All.DarkEnergySoundSpeed*All.DarkEnergySoundSpeed*fft_of_rhogrid_tot[ip].re/vol_fac;
					divU=P_prefactor*fft_of_dPgrid[ip].re/k2_actual;
					if(fabs(divU)>fabs(temp)){
						++bad_points;
						trigger=1;
					}
					P_std.re      +=pow(temp,2);
					P_gauge_std.re+=pow(divU,2);

					temp=All.DarkEnergySoundSpeed*All.DarkEnergySoundSpeed*fft_of_rhogrid_tot[ip].im/vol_fac;
					divU=P_prefactor*fft_of_dPgrid[ip].im/k2_actual;
					if(trigger==0 && fabs(divU)>fabs(temp))
						++bad_points;
					P_std.im      +=pow(temp,2);
					P_gauge_std.im+=pow(divU,2);

					trigger=0;
#endif


					fft_of_dPgrid[ip].re/=PMGRID*PMGRID*PMGRID;
					fft_of_dPgrid[ip].im/=PMGRID*PMGRID*PMGRID;

					/* Add 3dP term from the Poisson equation with dark energy.
					 * This corresponds to effectively adding an extra factor of 3*cs^2*rho_de 
					 * to the dark energy density where cs is the sound speed
					 * that relates the pressure to its density */	
					fft_of_rhogrid_tot[ip].re +=  fft_of_dPgrid[ip].re;
					fft_of_rhogrid_tot[ip].im +=  fft_of_dPgrid[ip].im;

					/* Store dark energy part of Poisson equation */
					rho_temp.re=fft_of_rhogrid_tot[ip].re;
					rho_temp.im=fft_of_rhogrid_tot[ip].im;

					fft_of_rhogrid_tot[ip].re += fft_of_rhogrid[ip].re;
					fft_of_rhogrid_tot[ip].im += fft_of_rhogrid[ip].im;
					/* fft_of_rhogrid_tot now contains FFT(rhogrid)*DC+FFT(rhogrid_DE) where DC is the deconvolution kernel. No smoothing has been done.
					 * This means that fft_of_rhogrid_tot now contains the full dark matter + dark energy potential */

					/* Multiply with the Green's function (constants will be corrected by potfac in advance_DE. Note the difference between the physical k and the integer k used here as mentioned above) */
					fft_of_rhogrid_tot[ip].re *= -1.0/k2;
					fft_of_rhogrid_tot[ip].im *= -1.0/k2;

					smth = -exp(-k2 * asmth2) / k2; /* -4 M_PI G/k2 is the Green's function for the Poisson equation in physical coordinates */
					/* Now do second deconvolution of dark matter potential, and a single deconvolution of the dark energy potential (corresponding to the CIC from the grid to the particles). 
					 * Multiply with the Green's function and smoothing kernel. 
					 * Only the dark matter needs to be long range smoothed, dark energy is applicable on grid scale*/
					fft_of_rhogrid[ip].re = ff*ff*(smth*fft_of_rhogrid[ip].re-rho_temp.re/k2);
					fft_of_rhogrid[ip].im = ff*ff*(smth*fft_of_rhogrid[ip].im-rho_temp.im/k2);
					/* fft_of_rhogrid now contains FFT(rhogrid)*DC*DC+FFT(rhogrid_DE)*DC where DC is the deconvolution kernel (the Green's function and smoothing kernel have also been applied) */
					/* end deconvolution. Note that the pressure term in the Poisson equation has been added above by modifying the dark energy density */
				}

			}

#ifdef DEBUG
	MPI_Allreduce(MPI_IN_PLACE,&P_std.re,1,FFTW_MPITYPE,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE,&P_std.im,1,FFTW_MPITYPE,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE,&P_gauge_std.re,1,FFTW_MPITYPE,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE,&P_gauge_std.im,1,FFTW_MPITYPE,MPI_SUM,MPI_COMM_WORLD);
	P_std.re=sqrt(P_std.re);
	P_std.im=sqrt(P_std.im);
	P_gauge_std.re=sqrt(P_gauge_std.re);
	P_gauge_std.im=sqrt(P_gauge_std.im);

	master_printf("cs^2: %e, velocity unit prefactor: %e\n"
			"P_std.re: %e, P_std.im: %e\n"
			"P_gauge_std.re: %e, P_gauge_std.im: %e\n",
			pow(All.DarkEnergySoundSpeed,2),
			3*All.Time*(1+All.DarkEnergyW)*(pow(All.DarkEnergySoundSpeed,2)-All.DarkEnergyW),
			P_std.re, P_std.im,
			P_gauge_std.re, P_gauge_std.im
		     );

	fftw_real mean_DE_dbg;
	if(slabstart_y == 0) 
		mean_DE_dbg=fft_of_rhogrid_tot[0].re/vol_fac/(PMGRID*PMGRID*PMGRID);

	MPI_Bcast(&mean_DE_dbg,1,FFTW_MPITYPE,0,MPI_COMM_WORLD);
	master_printf("Mean_DE_dbg/mean_DE: %e\n",mean_DE_dbg/mean_DE);

	MPI_Allreduce(MPI_IN_PLACE,&bad_points,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	master_printf("FFTW: Number of gauge term bad points: %i (%e of total)\n",bad_points,bad_points/(1.0*PMGRID*PMGRID*PMGRID));

#endif
	/* Subtract mean */
	if(slabstart_y == 0) 
		fft_of_rhogrid_tot[0].re = fft_of_rhogrid_tot[0].im = 0.0;

	if(slabstart_y == 0) /* This sets the mean to zero, meaning that we get the relative density delta_rho (since the k=0 part is the constant contribution) */
		fft_of_rhogrid[0].re = fft_of_rhogrid[0].im = 0.0;

	/* Do the FFT to get the potential */
	rfftwnd_mpi(fft_inverse_plan, 1, rhogrid, workspace, FFTW_TRANSPOSED_ORDER);
	/* Note: The inverse FFT scales the data by PMGRID*PMGRID*PMGRID */

	/* Do the FFT of rhogrid_tot to get the total potential of the singly deconvolved dm + unconvolved de */
	rfftwnd_mpi(fft_inverse_plan, 1, rhogrid_tot, workspace, FFTW_TRANSPOSED_ORDER);

	/* Now do the inverse FFT to get dP */
	rfftwnd_mpi(fft_inverse_plan, 1, dPgrid_fftw, workspace, FFTW_TRANSPOSED_ORDER);

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

	/* Send/receive rhogrid_tot slabs needed for the finite difference (which now are the singly deconvolved dm+de potential). 
	 * Note that rhogrid_tot, unlike rhogrid_DE and ugrid_DE, is in the fftw slab format (y,z dimensions PMGRID, PMGRID2 respectively).
	 * The following sends more data than what is actually neccesary, however no moving of data is needed.
	 * Also send the pressure grid containing the pressure perturbation.
	 */
	if(nslab_x>0){
		comm_reqs=malloc(2*(nslabs_to_send+4)*sizeof(MPI_Request)); /* Received/not received status of non-blocking messages (communication handles) */
		status_DE=malloc(2*(nslabs_to_send+4)*sizeof(MPI_Status)); /* The MPI_Status return values of the communication */

		/* Send slabs */
		unsigned short req_count=0;
		for( i=0 ; i<nslabs_to_send ; ++i )
		{
			slab=slabs_to_send[i]-slabstart_x;
			/* Send rhogrid_tot slabs. */ 
			MPI_Isend(rhogrid_tot+slab*PMGRID*PMGRID2,PMGRID*PMGRID2*sizeof(fftw_real),MPI_BYTE,send_tasks[i],slabs_to_send[i],MPI_COMM_WORLD,&comm_reqs[req_count++]);
			/* Send pressure slabs. Tagged with absolute slab index + PMGRID so as to not coincide with the rhogrid tag */
			MPI_Isend(dPgrid_fftw+slab*PMGRID*PMGRID2,PMGRID*PMGRID2*sizeof(fftw_real),MPI_BYTE,send_tasks[i],slabs_to_send[i]+PMGRID,MPI_COMM_WORLD,&comm_reqs[req_count++]);
		}

		/* Receive slabs */
		for( i=0 ; i<4 ; ++i )
		{
			slab=LOGICAL_INDEX(slabs_to_recv[i]-slabstart_x+2); /* Index minus start of local patch + 2 since the received patch starts at relative index -2  */
			/* Receive rhogrid_tot slabs (this is the potential) */
			MPI_Irecv(rhogrid_tot_expanded+slab*PMGRID*PMGRID2,PMGRID*PMGRID2*sizeof(fftw_real),MPI_BYTE,recv_tasks[i],slabs_to_recv[i],MPI_COMM_WORLD,&comm_reqs[req_count++]);
			/* Receive pressure slabs. Tagged with absolute slab index + PMGRID so as to not coincide with the rhogrid tag */
			MPI_Irecv(dPgrid_fftw_expanded+slab*PMGRID*PMGRID2,PMGRID*PMGRID2*sizeof(fftw_real),MPI_BYTE,recv_tasks[i],slabs_to_recv[i]+PMGRID,MPI_COMM_WORLD,&comm_reqs[req_count++]);
		}
	}

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

	/* Free all but the dark energy arrays */
	pm_init_periodic_free();

	/* Wait point for non-blocking exchange of rhogrid_tot and dPgrid slabs. */
	if(nslab_x>0)
		if(MPI_SUCCESS!=MPI_Waitall(2*(nslabs_to_send+4),comm_reqs,status_DE)){
			mpi_fprintf(stderr,"Error in MPI_Waitall (%s: %i)\n",__FILE__,__LINE__);
			endrun(1);
		}
	free(comm_reqs);
	comm_reqs=NULL;
	free(status_DE);
	status_DE=NULL;

	/* Calculate the next PM timestep and update the DE equations - use finite differences */
	double hubble_a = All.Omega0 / (All.Time * All.Time * All.Time) + (1 - All.Omega0 - All.OmegaLambda) / (All.Time * All.Time) +  All.OmegaLambda/pow(All.Time,3.0*(1+All.DarkEnergyW));
	hubble_a = All.Hubble * sqrt(hubble_a);
	find_dt_displacement_constraint(hubble_a*All.Time*All.Time);
#ifdef DEBUG
	static int next_integer_timestep=0;
	static double next_timestep=0;
	if(next_timestep!=0){
		if(next_timestep!=All.Time)
			mpi_fprintf(stderr,"Assertion fail: timestep is %f (integer: %i), expected %f (integer: %i)\n",All.Time,All.Ti_Current,next_timestep,next_integer_timestep);
		assert(next_timestep==All.Time);
		assert(next_integer_timestep==All.Ti_Current);
	}
	next_integer_timestep=All.Ti_Current+calc_PM_step();
	next_timestep=All.TimeBegin*exp(next_integer_timestep*All.Timebase_interval);
#else
	int next_integer_timestep=All.Ti_Current+calc_PM_step();
	double next_timestep=All.TimeBegin*exp(next_integer_timestep*All.Timebase_interval);
#endif
	master_printf("Next PM timestep: %f, da=%e\n",next_timestep,next_timestep-All.Time);

	if(nslab_x>0)
		advance_DE(next_timestep-All.Time);
	MPI_Barrier(MPI_COMM_WORLD);
	master_printf("Done with the dark energy PM contribution\n");

	free_dark_energy();

	force_treeallocate(All.TreeAllocFactor * All.MaxPart, All.MaxPart);

	All.NumForcesSinceLastDomainDecomp = 1 + All.TotNumPart * All.TreeDomainUpdateFrequency;

	if(first_DE_run)
		first_DE_run=0;

	if(ThisTask == 0)
	{
		printf("done PM.\n");
		fflush(stdout);
	}
}

#endif


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

/* Cloud-in-Cell interpolation and communication. Moved to seperate function for convenience */
void CIC(int *meshmin,int *meshmax){
	double dx, dy, dz;
	int slab_x, slab_y, slab_z;
	int slab_xx, slab_yy, slab_zz;
	int i, level, sendTask, recvTask;
	int sendmin, sendmax, recvmin, recvmax;
	int dimx, dimy, dimz, recv_dimx, recv_dimy, recv_dimz;
	MPI_Status status;

	dimx = meshmax[0] - meshmin[0] + 2;
	dimy = meshmax[1] - meshmin[1] + 2;
	dimz = meshmax[2] - meshmin[2] + 2;

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


	for(i = 0; i < fftsize; i++){	/* clear local density field */
		rhogrid[i] = 0;
	}

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
}

#ifdef DYNAMICAL_DE
void DE_IC(void){
	int i,j;

	for( i=0 ; i<nslab_x*PMGRID*PMGRID ; ++i )
	{
		rhogrid_DE[i]=mean_DE;
		dPgrid[i]=0;
		for( j=0 ; j<3 ; ++j )
		{
			ugrid_DE[i][j]=0;
		}
	}

	master_printf("Done with dark energy initial conditions\n");
}

void advance_DE(const fftw_real da){
	const double fac_FD = 1 / (2.0 * All.BoxSize / PMGRID);	/* for finite differencing. Factor 1/2 is part of the FD coefficients, could be moved there as well */
	const double potfac =All.G / (All.Time*M_PI * All.BoxSize);	/* to get potential */
	/* Provide easier notation (some of these will be removed by the compiler optimiser anyway) */
	const double a=All.Time;
	const double lightspeed=C/All.UnitVelocity_in_cm_per_s;
	const double H=All.Hubble*sqrt(All.Omega0 / (a * a * a) + (1 - All.Omega0 - All.OmegaLambda) / (a * a) + All.OmegaLambda/pow(a,3.0*(1+All.DarkEnergyW)));

	/* Indexing variables */
	int x,y,z;
	unsigned int index;
	unsigned int fftw_index;
	int xrr,xr,xl,xll, yrr,yr,yl,yll, zrr,zr,zl,zll;
	unsigned short dim,Udim;
	/* Physical quantities */
	fftw_real gradrho[3]; /* Gradient of rho in a single point */
	fftw_real gradphi[3]; /* Gradient of phi in a single point */
	fftw_real gradP[3]; /* Gradient of P in a single point */
	fftw_real gradU[3][3]; /* Gradient of all 3 components of U in a single point */
	fftw_real U_prev[3]; /* U in previous timestep*/
	fftw_real dUda[3]; /* dU/da */
	fftw_real drhoda_current; /* dRho/da */
	fftw_real Pdot; /* dP/c^2/dtau */
	fftw_real rho_plus_P; /* rho + P/c^2 */
	fftw_real rho_plus_P_reci; /* 1/(rho + P/c^2) */
	/* Temporary storage for the new density and velocity fields */
	fftw_real *new_rhogrid_DE_expanded=my_malloc((4+nslab_x)*PMGRID*PMGRID*sizeof(fftw_real));
	fftw_real (*new_ugrid_DE_expanded)[3]=my_malloc(3*(4+nslab_x)*PMGRID*PMGRID*sizeof(fftw_real));
	fftw_real *new_rhogrid_DE=new_rhogrid_DE_expanded+2*PMGRID*PMGRID;
	fftw_real (*new_ugrid_DE)[3]=new_ugrid_DE_expanded+2*PMGRID*PMGRID;

#ifdef DEBUG
	fftw_real U_sq=0;
#endif
	/* Finite differences
	 * The following loops could probably be optimised by changing the data structure to allow cache optimisations */
	for( x=2 ; x<nslab_x+2 ; ++x ) /* Loop over slabs */
		for( y=0 ; y<PMGRID ; ++y )
			for( z=0 ; z<PMGRID ; ++z ){
				for( dim=0 ; dim<3 ; ++dim ) /* Loop over x,y,z components of the gradients */
				{
					xrr = xll = xr = xl = x;
					yrr = yll = yr = yl = y;
					zrr = zll = zr = zl = z;

					switch (dim)
					{
						case 0:
							xr  = x + 1;
							xrr = x + 2;
							xl  = x - 1;
							xll = x - 2;
							break;
						case 1:
							yr  = LOGICAL_INDEX(y + 1);
							yl  = LOGICAL_INDEX(y - 1);
							yrr = LOGICAL_INDEX(y + 2);
							yll = LOGICAL_INDEX(y - 2);
							break;
						case 2:
							zr  = LOGICAL_INDEX(z + 1);
							zl  = LOGICAL_INDEX(z - 1);
							zrr = LOGICAL_INDEX(z + 2);
							zll = LOGICAL_INDEX(z - 2);
							break;
					}
					/* Note the different signs on the FDs than standard Gadget2. This is because Gadget2 needs the force, which has a minus */
					gradrho[dim]=
						fac_FD*(
								(4.0/3.0)*
								( - rhogrid_DE_expanded[(xl * PMGRID + yl) * PMGRID + zl]
								  + rhogrid_DE_expanded[(xr * PMGRID + yr) * PMGRID + zr]
								)
								+
								(1.0 / 6.0) *
								( rhogrid_DE_expanded[(xll * PMGRID + yll) * PMGRID + zll] 
								  - rhogrid_DE_expanded[(xrr * PMGRID + yrr) * PMGRID + zrr]
								)
						       );
					gradphi[dim]=
						potfac*fac_FD*(
								(4.0/3.0)*
								( - rhogrid_tot_expanded[INDMAP(xl,yl,zl)]
								  + rhogrid_tot_expanded[INDMAP(xr,yr,zr)]
								)
								+
								(1.0 / 6.0) *
								( rhogrid_tot_expanded[INDMAP(xll,yll,zll)]
								  - rhogrid_tot_expanded[INDMAP(xrr,yrr,zrr)]
								)
							      );
					gradP[dim]=
						fac_FD*(
								(4.0/3.0)*
								( - dPgrid_fftw_expanded[INDMAP(xl,yl,zl)]
								  + dPgrid_fftw_expanded[INDMAP(xr,yr,zr)]
								)
								+
								(1.0 / 6.0) *
								( dPgrid_fftw_expanded[INDMAP(xll,yll,zll)]
								  - dPgrid_fftw_expanded[INDMAP(xrr,yrr,zrr)]
								)
						       );
					/* In the following Udim is the U index, dim is the derivative index. 
					 * This means that gradU[0][1] is the y derivative of Ux for example*/
					for( Udim=0; Udim<3 ; ++Udim ) 
					{
						gradU[Udim][dim]= /* All 9 components of the gradient of U. */
							fac_FD*(
									(4.0/3.0)*
									( - ugrid_DE_expanded[(xl * PMGRID + yl) * PMGRID + zl][Udim]
									  + ugrid_DE_expanded[(xr * PMGRID + yr) * PMGRID + zr][Udim]
									)
									+
									(1.0 / 6.0) *
									( ugrid_DE_expanded[(xll * PMGRID + yll) * PMGRID + zll][Udim] 
									  - ugrid_DE_expanded[(xrr * PMGRID + yrr) * PMGRID + zrr][Udim]
									)
							       );
					}
				}

				index=(x-2)*PMGRID*PMGRID+y*PMGRID+z;
				fftw_index=INDMAP(x-2,y,z);

				for( dim=0 ;dim<3  ; ++dim )
					U_prev[dim]=ugrid_DE[index][dim];
				rho_plus_P=rhogrid_DE[index]+All.DarkEnergyW*mean_DE+dPgrid_fftw[fftw_index];
				rho_plus_P_reci=1/rho_plus_P;

				Pdot=(dPgrid_fftw[fftw_index]-dPgrid[index])/da; //delta term
				Pdot+=-3*All.DarkEnergyW*(1+All.DarkEnergyW)*mean_DE/All.Time; //Background term
				Pdot*=a*a*H; //Change to time derivative

				fftw_real divU=(gradU[0][0]+gradU[1][1]+gradU[2][2]);
				if(rho_plus_P<0){
					mpi_fprintf(stderr,"WARNING: rho+P<0 for point (%i, %i, %i).\n"
							"Mean rho: %e, full rho: %e (drho: %e), P: %e (dP: %e), sum: %e\n"
							"DivU: %e\n"
							"Gauge term relative to cs term: %e\n",
							x-2+slabstart_x,y,z,
							mean_DE,
							rhogrid_DE[index], /* rho */
							rhogrid_DE[index]-mean_DE, /* drho */
							All.DarkEnergyW*mean_DE+dPgrid_fftw[fftw_index], /* P */
							dPgrid_fftw[fftw_index], /* dP */
							rho_plus_P, /* rho + P */
							divU, /* sgn(divU) */
							(dPgrid_fftw[fftw_index]-All.DarkEnergySoundSpeed*All.DarkEnergySoundSpeed*(rhogrid_DE[index]-mean_DE))/(All.DarkEnergySoundSpeed*All.DarkEnergySoundSpeed*(rhogrid_DE[index]-mean_DE)) /* gauge term ratio */
						   );
					rho_plus_P_reci=0;
				}

				drhoda_current=
					-3.0*a*H*rho_plus_P
					-(U_prev[0]*gradrho[0]+U_prev[1]*gradrho[1]+U_prev[2]*gradrho[2])
					-(U_prev[0]*gradP[0]+U_prev[1]*gradP[1]+U_prev[2]*gradP[2])
					-(rho_plus_P)*(gradU[0][0]+gradU[1][1]+gradU[2][2]);
				drhoda_current=drhoda_current/(a*a*H);

				new_rhogrid_DE[index]=rhogrid_DE[index]+drhoda_current*da;

#ifdef DEBUG
				if(new_rhogrid_DE[index]<0){
					new_rhogrid_DE[index]=0;
				}
				U_sq=0;
#endif
				for( dim=0 ;dim<3  ; ++dim )
				{
					dUda[dim]=
						-a*H*U_prev[dim]
						-(U_prev[0]*gradU[dim][0]+U_prev[1]*gradU[dim][1]+U_prev[2]*gradU[dim][2]) /* U dot grad U[dim] */
						-lightspeed*lightspeed*gradP[dim]*rho_plus_P_reci
						-U_prev[dim]*Pdot*rho_plus_P_reci
						-gradphi[dim]; 
					dUda[dim]=dUda[dim]/(a*a*H);
					new_ugrid_DE[index][dim]=ugrid_DE[index][dim]+dUda[dim]*da;
#ifdef DEBUG
					U_sq+=new_ugrid_DE[index][dim]*new_ugrid_DE[index][dim];
#endif
				}

#ifdef DEBUG
				if(sqrt(U_sq)>C/All.UnitVelocity_in_cm_per_s){
					mpi_printf("Error: Point (%i, %i, %i) has U>c (U=%e).\n" 
							"rho+P: %e\n"
							"Terminating\n"
							,x-2+slabstart_x,y,z,
							U_sq,
							rho_plus_P
						  );
					endrun(1);
				}				

#endif
			}
	/* Now commit the new density and velocity fields and free the old arrays. For debugging purposes this is confusing, since
	 * the global arrays ugrid_DE_expanded and rhogrid_DE_expanded keeps moving around in memory*/
	free(rhogrid_DE_expanded);
	rhogrid_DE_expanded=new_rhogrid_DE_expanded;
	rhogrid_DE=rhogrid_DE_expanded+2*PMGRID*PMGRID;

	free(ugrid_DE_expanded);
	ugrid_DE_expanded=new_ugrid_DE_expanded;
	ugrid_DE=ugrid_DE_expanded+2*PMGRID*PMGRID;

	for( x=0 ; x<nslab_x ; ++x )
		for( y=0 ; y<PMGRID ; ++y )
			for( z=0 ; z<PMGRID ; ++z )
				dPgrid[x*PMGRID*PMGRID+y*PMGRID+z]=dPgrid_fftw[INDMAP(x,y,z)];	
}

void pm_stats(char* fname){
	FILE *fd;
	int i,j,k;
	char buf[128]="";
	char out[512]="";
	const double a=All.Time;
	const double H=All.Hubble*sqrt(All.Omega0 / (a * a * a) + (1 - All.Omega0 - All.OmegaLambda) / (a * a) + All.OmegaLambda/pow(a,3.0*(1+All.DarkEnergyW)));

	master_printf("Hubble parameter: %.3e\nCurrent physical Omega_matter=%.2f, Omega_lambda=%.2f\nCurrent co-moving Omega_matter=%.2f, Omega_lambda=%.2f\n",
			H,
			All.Omega0/(All.Time*All.Time*All.Time),All.OmegaLambda/pow(All.Time,3.0*(1+All.DarkEnergyW)),
			All.Omega0,All.OmegaLambda/pow(All.Time,3.0*All.DarkEnergyW));


	if(slabstart_y==0){
		if(first_DE_run==1){
			fd=fopen(fname,"w");
			fprintf(fd,"Time        \tmean_dm     \tdelta_dm_av    \tstd_dev_dm  \tmin_dm      \tmax_dm      \tmean_de     \tdelta_de_av    \tstd_dev_de  \tmin_de      \tmax_de      \n");
		}else{
			fd=fopen(fname,"a");
		}
	}
	unsigned int index=0;
	/* Dark matter part */
	fftw_real mean=0;
	fftw_real delta=0;
	fftw_real delta_mean=0;
	fftw_real std_dev=0;
	for( i=0 ; i<nslab_x ; ++i )
		for( j=0 ; j<PMGRID ; ++j )
			for( k=0 ; k<PMGRID ; ++k )
			{
				index=INDMAP(i,j,k);
				mean+= rhogrid[index];
			}

	MPI_Allreduce(MPI_IN_PLACE,&mean,1,FFTW_MPITYPE,MPI_SUM,MPI_COMM_WORLD);
	mean=mean/(PMGRID*PMGRID*PMGRID);

	fftw_real min=0;
	fftw_real max=0;

	for( i=0 ; i<nslab_x ; ++i )
		for( j=0 ; j<PMGRID ; ++j )
			for( k=0 ; k<PMGRID ; ++k )
			{
				index=INDMAP(i,j,k);
				delta=rhogrid[index]-mean;
				delta_mean+=delta;
				std_dev+=delta*delta;
				if(delta<min)
					min=delta;
				else if(delta>max)
					max=delta;
			}

	MPI_Allreduce(MPI_IN_PLACE,&delta_mean,1,FFTW_MPITYPE,MPI_SUM,MPI_COMM_WORLD);
	delta_mean=delta_mean/(PMGRID*PMGRID*PMGRID);
	MPI_Allreduce(MPI_IN_PLACE,&std_dev,1,FFTW_MPITYPE,MPI_SUM,MPI_COMM_WORLD);
	std_dev/=PMGRID*PMGRID*PMGRID;
	std_dev=sqrt(std_dev);
	MPI_Allreduce(MPI_IN_PLACE,&min,1,FFTW_MPITYPE,MPI_MIN,MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE,&max,1,FFTW_MPITYPE,MPI_MAX,MPI_COMM_WORLD);

	double print_dummy;
	if(slabstart_y==0){
		print_dummy=mean_DM*All.BoxSize*All.BoxSize*All.BoxSize/(PMGRID*PMGRID*PMGRID);
		printf("Background mass of dark matter in comoving mesh cell is: %e\n",print_dummy);
		print_dummy*=All.Time*All.Time*All.Time;
		printf("Background mass of dark matter in physical mesh cell: " 
				"%e (cosmo term: %e, delta_mean: %e, std dev: %e, min: %e, max: %e)\n"
				"Ratio of mean to cosmo term: %e\n",
				mean,print_dummy,delta_mean,std_dev,min,max,mean/print_dummy);
		sprintf(buf,"%e\t%e\t%e\t%e\t%e\t%e\t",All.Time,mean,delta_mean,std_dev,min,max);
		strcat(out,buf);
	}

	/* Dark energy part */
	mean=0;
	delta=0;
	delta_mean=0;
	std_dev=0;
	min=0;
	max=0;

	for( i=0 ; i<nslab_x ; ++i )
		for( j=0 ; j<PMGRID ; ++j )
			for( k=0 ; k<PMGRID ; ++k )
			{
				index=i*PMGRID*PMGRID+j*PMGRID+k;
				mean+= rhogrid_DE[index];
			}

	MPI_Allreduce(MPI_IN_PLACE,&mean,1,FFTW_MPITYPE,MPI_SUM,MPI_COMM_WORLD);
	mean=mean/(PMGRID*PMGRID*PMGRID);
	mean*=(All.BoxSize/PMGRID)*(All.BoxSize/PMGRID)*(All.BoxSize/PMGRID)*(All.Time*All.Time*All.Time);		

	for( i=0 ; i<nslab_x ; ++i )
		for( j=0 ; j<PMGRID ; ++j )
			for( k=0 ; k<PMGRID ; ++k )
			{
				index=INDMAP(i,j,k);
				delta= rhogrid_tot[index]-mean;
				delta_mean+=delta;
				std_dev+=delta*delta;
				if(delta<min)
					min=delta;
				else if(delta>max)
					max=delta;
			}

	MPI_Allreduce(MPI_IN_PLACE,&delta_mean,1,FFTW_MPITYPE,MPI_SUM,MPI_COMM_WORLD);
	delta_mean=delta_mean/(PMGRID*PMGRID*PMGRID);
	MPI_Allreduce(MPI_IN_PLACE,&std_dev,1,FFTW_MPITYPE,MPI_SUM,MPI_COMM_WORLD);
	std_dev/=PMGRID*PMGRID*PMGRID;
	std_dev=sqrt(std_dev);
	MPI_Allreduce(MPI_IN_PLACE,&min,1,FFTW_MPITYPE,MPI_MIN,MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE,&max,1,FFTW_MPITYPE,MPI_MAX,MPI_COMM_WORLD);

	if(slabstart_y==0){
		print_dummy=mean_DE*All.BoxSize*All.BoxSize*All.BoxSize/(PMGRID*PMGRID*PMGRID);
		printf("Background mass of dark energy in comoving mesh cell is: %e\n",print_dummy);
		print_dummy*=All.Time*All.Time*All.Time;
		printf("Background mass of dark energy in physical mesh cell: "
				"%e (cosmo term: %e, delta_mean: %e, std dev: %e, min: %e, max: %e)\n"
				"Ratio of mean to cosmo term: %e\n",
				mean,print_dummy,delta_mean,std_dev,min,max,mean/print_dummy);
		sprintf(buf,"%e\t%e\t%e\t%e\t%e\n",mean,delta_mean,std_dev,min,max);
		strcat(out,buf);
		fprintf(fd,"%s",out);
	}
	if(slabstart_y==0)
		fclose(fd);
}


void write_dm_grid(char* fname_DM){
	if(nslab_x>0){
		int i,j,k,npts;
		npts=nslab_x*PMGRID*PMGRID;
		float *slabs=my_malloc(npts*sizeof(float));
		unsigned int index;

		for(i=0; i<nslab_x; ++i)
			for(j=0; j<PMGRID; ++j)
				for(k=0; k<PMGRID; ++k)
				{
					index=INDMAP(i,j,k);
					slabs[i*PMGRID*PMGRID+j*PMGRID+k]=(float) rhogrid[index];
				}

		FILE *fd=fopen(fname_DM,"w");
		write_header(fd);
		fwrite(slabs,npts,sizeof(float),fd);
		fclose(fd);

		free(slabs);
	}
}

void write_de_grid(char* fname_DE){
	if(nslab_x>0){
		int i,npts;
		npts=nslab_x*PMGRID*PMGRID;
		float *slabs=my_malloc(npts*sizeof(float));

		for(i=0; i<nslab_x*PMGRID*PMGRID; ++i){
			slabs[i]=(float) rhogrid_DE[i]*(All.BoxSize/PMGRID)*(All.BoxSize/PMGRID)*(All.BoxSize/PMGRID)*All.Time*All.Time*All.Time;
		}

		FILE *fd=fopen(fname_DE,"w");
		write_header(fd);
		fwrite(slabs,npts,sizeof(float),fd);
		fclose(fd);

		free(slabs);
	}

}

void write_U_grid(char* fname_U){
	if(nslab_x>0){
		int i,j,npts;
		npts=3*nslab_x*PMGRID*PMGRID;
		float (*slabs)[3]=my_malloc(npts*sizeof(float));

		for(i=0; i<nslab_x*PMGRID*PMGRID; ++i){
			for( j=0 ; j<3 ; ++j )
			{
				slabs[i][j]=(float) ugrid_DE[i][j];
			}
		}

		FILE *fd=fopen(fname_U,"w");
		write_header(fd);
		fwrite(slabs,npts,sizeof(float),fd);
		fclose(fd);

		free(slabs);
	}
}

void write_header(FILE* fd){
	const unsigned int gridsize=PMGRID;
	fwrite(&All.Time,1,sizeof(double),fd);
	fwrite(&All.BoxSize,1,sizeof(double),fd);
#ifndef DYNAMICAL_DE
	const double dummy=0;
	fwrite(&dummy,1,sizeof(double),fd);
	fwrite(&dummy,1,sizeof(double),fd);
#else
	fwrite(&All.DarkEnergyW,1,sizeof(double),fd);
	fwrite(&All.DarkEnergySoundSpeed,1,sizeof(double),fd);
#endif
	fwrite(&gridsize,1,sizeof(unsigned int),fd);
}
#endif

#endif
#endif



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

#ifdef DE_BACKGROUND
#ifdef DYNAMICAL_DE
#error "Cannot have both DE_BACKGROUND and DYNAMICAL_DE defined at the same time"
#endif
#endif

#ifndef DYNAMICAL_DE
#ifdef NONLINEAR_DE
#warning "WARNING: NONLINEAR_DE defined, but DYNAMICAL_DE is not. Check the makefile."
#undef NONLINEAR_DE
#endif
#endif

/* Dark energy macros */
#define LOGICAL_INDEX(x)  ((x<0) ? x+PMGRID : (x>=PMGRID) ? x-PMGRID : x ) /* Map (negative) index to the range [0,PMGRID[ */
#define INDMAP(i,j,k) ((i)*PMGRID*PMGRID2+(j)*PMGRID2+k) /* Map (i,j,k) in 3 dimensional array (dimx X PMGRID X PMGRID2) to 1 dimensional array */

#ifdef DEBUG
void pm_stats(char *);
/* Debugging macros. If the while(0) was not included the macros would be dangerous to insert inside blocks of code */
#define SHOUT(x) do{if(x) mpi_printf("SHOUT: !(" #x ")\n");} while(0)
#define FILELINE do{if(x) master_printf("At %s:%i\n",__FILE__,__LINE__);} while(0)
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

void CIC(int*,int*); /* Cloud-in-Cell interpolation moved to seperate function */
void calc_powerspec(char *,fftw_complex *); /* Calculate the power spectrum of an array */
void calc_powerspec_detailed(char *,fftw_complex *); /* Alternative way to calculate the power spectrum. Currently unused */
static short int PMTask=0; /* Does this task contribute to the FFT? */
void write_header(FILE *); /* Header written to grid snapshots */
void write_dm_grid(char *); /* Write the dark matter grid to file */

#ifdef DYNAMICAL_DE
static short int first_DE_run=1; /* Is this the initial run? */
void DE_IC(void);  /* Dark energy initial conditions */
void write_de_grid(char *); /* Write the dark energy grid to file */
void write_U_grid(char *); /* Write the divergence of U to file */

#ifdef NONLINEAR_DE /* Only relevant for the non linear (barotropic) dark energy */
void advance_DE_nonlinear(fftw_real); /* Function prototype for the routine responsible for advancing the nonlinear dark energy density and velocity perturbations*/
static int recv_tasks[4]; /* The 4 tasks that have the slabs this task needs (ordered left left, left, right, right right) */
static int send_tasks[6]; /* The (up to 6) tasks that needs this task's slabs. Only in the case where some tasks, but not all, only have 1 slab it is neccessary to communicate with 6 others, otherwise this is normally 4*/
static int slabs_to_send[6]; /* The slabs this task needs to send in the order defined in send_tasks */
static int slabs_to_recv[4]; /* The slabs this task needs to receive in the order defined in recv_tasks */
static int nslabs_to_send;  /* How many slabs does this task need to send? Normally 4, but possibly up to 6 if some tasks only have 1 slab */
#else
void advance_DE_linear(fftw_real);    /* Function prototype for the routine responsible for advancing the linear dark energy density and velocity perturbations */
#endif

/* The dark energy arrays come in 2 forms: An array corresponding to the actual slabs of the task, and an expanded array
 * with space for 4 extra slabs (2 in each end) used for finite differencing */
#ifdef NONLINEAR_DE
static fftw_real *rhogrid_tot, *rhogrid_tot_expanded, *divU_grid; /* fftw-format arrays. Note that the P arrays are P/c^2 */
static fftw_real *rhogrid_DE_expanded, (*ugrid_DE_expanded)[3]; /* Normal format arrays */
static fftw_real *rhogrid_DE, (*ugrid_DE)[3]; /* Normal format arrays */
static fftw_complex *fft_of_rhogrid_tot, *fft_of_divU; /* fft of the total dark energy and dark matter density. Also the fft of the velocity divergence (which will become the pressure)*/
#else
static fftw_complex *rhogrid_tot, *rhogrid_DE, *ugrid_DE;; /* fftw-complex format arrays. Ugrid is divU*/
#endif

/* Mean densities of dark matter and dark energy */
static fftw_real mean_DM, mean_DE;
/* Establish which tasks have to communicate with each other to populate the expanded FFTW arrays
 * (if the current task, for example, has the slabs from 4 to 8 it needs slabs 2 and 3 from the task(s) to its "left"
 * and slabs 9 and 10 from the task(s) to its "right"). The extra slabs are used in the finite differencing of the different 
 * dark energy quantities */
#ifdef NONLINEAR_DE
int comm_order(int nslabs){
	if(PMTask){
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
		/* Brute force communication algorithm. Not pretty but it works */
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
#endif

/* One-time allocation of the dark energy arrays */
void DE_allocate(int nx){
	if(PMTask){
		/* Expand the arrays with 4 extra slabs to store communication */
#ifdef NONLINEAR_DE
		master_printf("Code compiled with DYNAMICAL_DE, setting up non-linear dark energy environment\n");
		rhogrid_DE_expanded=my_malloc((4+nx)*PMGRID*PMGRID*sizeof(fftw_real));
		ugrid_DE_expanded=my_malloc(3*(4+nx)*PMGRID*PMGRID*sizeof(fftw_real));
		const unsigned long int size=(4*(4+nx)+nx)*PMGRID*PMGRID*sizeof(fftw_real);
		/* Arrays corresponding to the actual slabs of this task */
		rhogrid_DE=rhogrid_DE_expanded+2*PMGRID*PMGRID;
		ugrid_DE=ugrid_DE_expanded+2*PMGRID*PMGRID;
#else
		master_printf("Code compiled with DYNAMICAL_DE, setting up linear dark energy environment\n");
		rhogrid_DE=my_malloc(fftsize*sizeof(fftw_real));
		ugrid_DE=my_malloc(fftsize*sizeof(fftw_real));
		const unsigned long int size=2*fftsize*sizeof(fftw_real);
#endif

		mpi_printf("Allocated %lu bytes (%lu MB) for DE arrays\n",size,size/(1024*1024));
	}
	else{
#ifdef NONLINEAR_DE
		rhogrid_DE_expanded=NULL;
		ugrid_DE_expanded=NULL;
#endif
		rhogrid_DE=NULL;
		ugrid_DE=NULL;
	}
}

void PM_cleanup( ){ /* Like free_memory(), this is not actually called by the program */
#ifdef NONLINEAR_DE
	free(rhogrid_DE_expanded);
	free(ugrid_DE_expanded);
#else
	free(rhogrid_DE);
	free(ugrid_DE);
#endif
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
	if(nslab_x>0) /* Do I contribute to the FFT? */
		PMTask=1;
	/* Allocate extra memory for the dark energy part and establish communication order */
#ifdef NONLINEAR_DE
	nslabs_to_send=comm_order(nslab_x);
#endif
}

#ifdef DYNAMICAL_DE
void DE_periodic_allocate(void){
	/* Expand with 4 slabs to store communication data */
	if(PMTask){

		unsigned int size=0;
#ifdef NONLINEAR_DE
		size=(fftsize*PMGRID2*PMGRID)*sizeof(fftw_real);
		divU_grid=my_malloc(size);
		fft_of_divU = (fftw_complex *) & divU_grid[0];
		size=(fftsize+4*PMGRID2*PMGRID)*sizeof(fftw_real);
		rhogrid_tot_expanded=my_malloc(size);
		/* rhogrid_tot is only the local array, rhogrid_tot_expanded is the local array and the 2 slabs before and 2 after */
		rhogrid_tot=& rhogrid_tot_expanded[INDMAP(2,0,0)];
		fft_of_rhogrid_tot = (fftw_complex *) & rhogrid_tot[0];
		size=size*2;
#else
		size=(fftsize*sizeof(fftw_real));
		rhogrid_tot=my_malloc(size);
#endif
		if(first_DE_run){
			master_printf("PM force with dark energy toggled (time=%f). Allocated %u bytes (%u MB) for dark energy temporary storage\n",All.Time,size,size/(1024*1024));
		}
	}
	else
	{
		rhogrid_tot=NULL;
#ifdef NONLINEAR_DE
		fft_of_rhogrid_tot=NULL;
		rhogrid_tot_expanded=NULL;
		divU_grid=NULL;
		fft_of_divU=NULL;
#endif
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
#ifdef NONLINEAR_DE
	free(rhogrid_tot_expanded);
	rhogrid_tot_expanded=NULL;
	rhogrid_tot=NULL;
	free(divU_grid);
	divU_grid=NULL;
#else
	free(rhogrid_tot);
	rhogrid_tot=NULL;
#endif
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
#ifdef DEBUG
	char fname[256];
	static int Nruns=0;
	if(All.Time>All.DarkEnergyOutputStart && Nruns<All.DarkEnergyNumOutputs){
		sprintf(fname,"%sDM_a=%.3f.%.3i",All.OutputDir,All.Time,ThisTask);
		master_printf("Writing dm+de grids\n");
		write_dm_grid(fname);
	}
#endif
	/* Do the FFT of the density field */

	rfftwnd_mpi(fft_forward_plan, 1, rhogrid, workspace, FFTW_TRANSPOSED_ORDER);

	/* Calculate the power spectrum */
	fftw_complex * workspace_powergrid=(fftw_complex *) & workspace[0];
	workspace_powergrid[0].re=0;
	workspace_powergrid[0].im=0;
	fftw_real mean_DM=All.Omega0*3.0*All.Hubble*All.Hubble/(8.0*M_PI*All.G)/pow(All.Time,3.0); /* Mean dark matter density in the universe */
	fftw_real temp=pow(All.BoxSize,3)*pow(All.Time,3.0)*mean_DM; /* Total mass in simulation */
	/* Translate mass grid to delta (doesn't subtract the mean as this is only relevant for the zero mode).
	 * Deconvolve. */
	for(y = slabstart_y; y < slabstart_y + nslab_y; y++)
		for(x = 0; x < PMGRID; x++)
			for(z = 0; z < PMGRID / 2 + 1; z++)
			{
				ip = PMGRID * (PMGRID / 2 + 1) * (y - slabstart_y) + (PMGRID / 2 + 1) * x + z;
				
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

				if(k2 > 0)
				{
					/* Deconvolution */
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
					workspace_powergrid[ip].re=fft_of_rhogrid[ip].re*ff*ff;
					workspace_powergrid[ip].im=fft_of_rhogrid[ip].im*ff*ff;
					/* Done deconvolving */
					/* Convert dimensionless delta to mass */
					workspace_powergrid[ip].re/=temp;
					workspace_powergrid[ip].im/=temp;
				}
			}

	char fname_power[256];
	sprintf(fname_power,"%s%s_DM_a=%.3f",All.OutputDir,All.PowerFileBase,All.Time);
	if(All.DetailedPowerOn)
		calc_powerspec_detailed(fname_power,workspace_powergrid);
	else
		calc_powerspec(fname_power,workspace_powergrid);

	workspace_powergrid=NULL; /* NOT freed, workspace will be freed later */


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
#ifdef NONLINEAR_DE
/* Main function responsible for coupling dark matter and dark energy through the potential.
 * Will also advance the dark energy grid.
 * Note that in case non linear dark energy is triggered the dark energy density grid
 * rhogrid_DE is the full density (rho) and ugrid_DE is the 3-dimensional velocity (U).
 * TODO: Fix code when running on a single process (probably in comm order). */
void pmforce_periodic_DE_nonlinear(void)
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

	double fac_FD; /* Finite difference factor */
	double temp; /* Merge temp with fftw_real divU */
	const double lightspeed=C/All.UnitVelocity_in_cm_per_s;
	const double H=All.Hubble*sqrt(All.Omega0 / (All.Time * All.Time * All.Time) + (1 - All.Omega0 - All.OmegaLambda) / (All.Time * All.Time) + All.OmegaLambda/pow(All.Time,3.0*(1+All.DarkEnergyW)));

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
				"Sound horizon (units of box size): %e\n"
				"Dark energy gauge transformation important for a< %e (z > %e)\n"				
				,All.DarkEnergySoundSpeed,All.DarkEnergyW,
				All.DarkEnergySoundSpeed*lightspeed/(All.Time*H)/All.BoxSize,
				temp,1/temp-1
			     );
		DE_allocate(nslab_x);
	}
	/* Non-blocking send/receive statements for rhogrid_DE and ugrid_DE */

	MPI_Request *comm_reqs=NULL;
	MPI_Status *status_DE=NULL;
	/* Volume factor of grid cell */
	const double vol_fac=(All.BoxSize/PMGRID)*(All.BoxSize/PMGRID)*(All.BoxSize/PMGRID)*(All.Time*All.Time*All.Time); /* Physical volume factor. Converts from density to mass */		
	if(PMTask){
		comm_reqs=malloc(2*(nslabs_to_send+4)*sizeof(MPI_Request)); /* Received/not received status of non-blocking messages (communication handles) */
		status_DE=malloc(2*(nslabs_to_send+4)*sizeof(MPI_Status)); /* The MPI_Status return values of the communication */

		/* Assign dark energy mass to rhogrid_tot.		
		 * Note rhogrid_DE is an nslab_x*PMGRID*PMGRID array while rhogrid_tot is in the fftw format nslab_x*PMGRID*PMGRID2 */
		for( i=0 ; i<nslab_x ; ++i )
			for( j=0 ; j<PMGRID ; j++ )/* Change from rho to mass.*/
				for( k=0 ; k<PMGRID ; ++k )
					rhogrid_tot[INDMAP(i,j,k)]=rhogrid_DE[i*PMGRID*PMGRID+j*PMGRID+k]*vol_fac;



		unsigned short req_count=0;
		/* Send slabs one at a time
		 * Send using non-blocking functions to allow the exchange to run in the background*/
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
			MPI_Irecv(ugrid_DE_expanded+slab*PMGRID*PMGRID  ,3*PMGRID*PMGRID*sizeof(fftw_real),MPI_BYTE,recv_tasks[i],slabs_to_recv[i]       ,MPI_COMM_WORLD,&comm_reqs[req_count++]);
			/* Receive rhogrid_DE slabs. Tagged with absolute slab index + PMGRID so as to not coincide with the ugrid tag */
			MPI_Irecv(rhogrid_DE_expanded+slab*PMGRID*PMGRID,PMGRID*PMGRID*sizeof(fftw_real),MPI_BYTE,recv_tasks[i],slabs_to_recv[i]+PMGRID,MPI_COMM_WORLD,&comm_reqs[req_count++]);
		}
	}

	/* Cloud-in-Cell interpolation. Moved to its own function for readability.
	 * Assigns mass to rhogrid while the dark energy grids are being exchanged. */
	CIC(meshmin,meshmax);


	/* Wait point for non-blocking exchange of DE slabs, only continue when all rhogrid_DE and ugrid_DE slabs have been exchanged.
	 * Still need to calculate and exchange rhogrid_tot for the total potential */
	if(PMTask){
		if(MPI_SUCCESS!=MPI_Waitall(2*(nslabs_to_send+4),comm_reqs,status_DE)){
			mpi_fprintf(stderr,"Error in MPI_Waitall (%s: %i)\n",__FILE__,__LINE__);
			endrun(1);
		}
	}

	free(comm_reqs);
	comm_reqs=NULL;
	free(status_DE);
	status_DE=NULL;

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

	/* Print simulation statistics to file */
	//sprintf(statname,"%s%s",All.OutputDir,All.DarkEnergyStatFile);
	//pm_stats(statname);


#endif

	/* Do the FFT of the dark matter density field */
	rfftwnd_mpi(fft_forward_plan, 1, rhogrid, workspace, FFTW_TRANSPOSED_ORDER);

	fftw_real divU=0; /* Merge this with temp */
	/* Initial conditions for the dark energy */
	if(first_DE_run){
		DE_IC();
	}
	else{

		/*  Calculate divergence of U and store it in divU_grid. Used in potential later.
		 *  At some point I would like to test how big a factor the divU term in the Poisson equation is.
		 *  At the linear level it is neglible, and I assume it's the same at the non-linear level*/
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
					divU_grid[INDMAP(x-2,y,z)]=divU;
				}



		/* Do the FFT of the dark energy density field (the dark energy density field has been converted to mass in rhogrid_tot) */
		rfftwnd_mpi(fft_forward_plan, 1, rhogrid_tot, workspace, FFTW_TRANSPOSED_ORDER);
		/* Do the FFT of the divergence of U */
		rfftwnd_mpi(fft_forward_plan, 1, divU_grid, workspace, FFTW_TRANSPOSED_ORDER);


	}
	/* Prepare to calculate the dark matter power spectrum */
	fftw_complex * workspace_powergrid=(fftw_complex *) & workspace[0];
	workspace_powergrid[0].re=0;
	workspace_powergrid[0].im=0;
	temp=pow(All.BoxSize,3)*pow(All.Time,3.0)*mean_DM; /* Total mass in simulation */
	/* Translate mass grid to delta (doesn't subtract the mean as this is only relevant for the zero mode).
	 * Deconvolve. */
	for(y = slabstart_y; y < slabstart_y + nslab_y; y++)
		for(x = 0; x < PMGRID; x++)
			for(z = 0; z < PMGRID / 2 + 1; z++)
			{
				ip = PMGRID * (PMGRID / 2 + 1) * (y - slabstart_y) + (PMGRID / 2 + 1) * x + z;

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

				if(k2 > 0)
				{
					/* Deconvolution */
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
					workspace_powergrid[ip].re=fft_of_rhogrid[ip].re*ff*ff;
					workspace_powergrid[ip].im=fft_of_rhogrid[ip].im*ff*ff;
					/* Done deconvolving */
					/* Convert to dimensionless delta */
					workspace_powergrid[ip].re/=temp;
					workspace_powergrid[ip].im/=temp;
				}
			}

	char fname_power[256];
	sprintf(fname_power,"%s%s_DM_a=%.3f",All.OutputDir,All.PowerFileBase,All.Time);
	if(All.DetailedPowerOn)
		calc_powerspec_detailed(fname_power,workspace_powergrid);
	else
		calc_powerspec(fname_power,workspace_powergrid);

	/* Prepare to calculate dark energy power spectrum */
	workspace_powergrid[0].re=0;
	workspace_powergrid[0].im=0;
	temp=pow(All.BoxSize,3)*pow(All.Time,3.0)*mean_DE; /* Total dark energy mass in simulation */
	/* Translate rho to delta */
	for(y = slabstart_y; y < slabstart_y + nslab_y; y++)
		for(x = 0; x < PMGRID; x++)
			for(z = 0; z < PMGRID / 2 + 1; z++)
			{
				ip = PMGRID * (PMGRID / 2 + 1) * (y - slabstart_y) + (PMGRID / 2 + 1) * x + z;

				/* Convert to dimensionless delta */
				workspace_powergrid[ip].re=fft_of_rhogrid_tot[ip].re/temp;
				workspace_powergrid[ip].im=fft_of_rhogrid_tot[ip].im/temp;
			}
	sprintf(fname_power,"%s%s_DE_a=%.3f",All.OutputDir,All.PowerFileBase,All.Time);
	if(All.DetailedPowerOn)
		calc_powerspec_detailed(fname_power,workspace_powergrid);
	else
		calc_powerspec(fname_power,workspace_powergrid);


	workspace_powergrid=NULL; /* NOT freed, workspace will be freed later */



#ifdef DEBUG
	if(slabstart_y==0 && PMTask){
		double tmp=(double) fft_of_rhogrid[0].re/(PMGRID*PMGRID*PMGRID);
		printf("Dark matter mean: %e (mean rho: %e)\n", tmp, tmp/vol_fac);
		tmp=fft_of_rhogrid_tot[0].re/(PMGRID*PMGRID*PMGRID);
		printf("Dark energy mean: %e (mean rho: %e)\n",tmp, tmp/vol_fac);
	}
#endif

	/* Enforce mean of pressure perturbation to vanish in case divergence U doesn't */
	if(slabstart_y == 0 && PMTask)
		fft_of_divU[0].re = fft_of_divU[0].im = 0.0;
	/* Dark energy pressure conversion factors */
	const double pot_prefactor=3*(1+All.DarkEnergyW)*mean_DE*vol_fac*H*All.Time*All.BoxSize*All.BoxSize/(lightspeed*lightspeed*4*M_PI*M_PI);

#ifdef DEBUG
	fftw_complex P_std, P_gauge_std;
	P_std.re=P_std.im=P_gauge_std.re=P_gauge_std.im=0;

	int bad_points=0;
	short trigger=0;
#endif

	/* Dummy variables to store rho */
	fftw_complex rho_temp_DE, rho_temp_DM;
	/* Conversion from integer k to comoving k. Needs additional 1/scalefactor to be physical k */
	/* multiply with the Green's function for the potential, deconvolve.
	*/
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

				smth = exp(-k2 * asmth2);
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
#ifdef DEBUG
					temp=All.DarkEnergySoundSpeed*All.DarkEnergySoundSpeed*fft_of_rhogrid_tot[ip].re;

					divU=pot_prefactor*fft_of_divU[ip].re/k2;
					if(fabs(divU)>fabs(temp)){
						++bad_points;
						trigger=1;
					}
					P_std.re      +=pow(temp,2);
					P_gauge_std.re+=pow(divU,2);

					temp=All.DarkEnergySoundSpeed*All.DarkEnergySoundSpeed*fft_of_rhogrid_tot[ip].im;
					divU=pot_prefactor*fft_of_divU[ip].im/k2;
					if(trigger==0 && fabs(divU)>fabs(temp))
						++bad_points;
					P_std.im      +=pow(temp,2);
					P_gauge_std.im+=pow(divU,2);

					trigger=0;
#endif


					/* Store dark matter part */
					rho_temp_DM=fft_of_rhogrid[ip];
					/* Store dark energy part */
					rho_temp_DE=fft_of_rhogrid_tot[ip];

					/* Add divU term from the Poisson equation with dark energy.*/
					fft_of_rhogrid_tot[ip].re=fft_of_rhogrid_tot[ip].re+pot_prefactor*fft_of_divU[ip].re/k2;
					fft_of_rhogrid_tot[ip].im=fft_of_rhogrid_tot[ip].im+pot_prefactor*fft_of_divU[ip].im/k2;

					/* Now do second deconvolution of dark matter potential, and a single deconvolution of the dark energy potential (corresponding to the CIC from the grid to the particles). 
					 * Multiply with the Green's function and smoothing kernel. 
					 * Only the dark matter needs to be long range smoothed, dark energy is applicable on grid scale*/
					fft_of_rhogrid[ip].re = -ff*ff*(smth*fft_of_rhogrid[ip].re+fft_of_rhogrid_tot[ip].re)/k2;
					fft_of_rhogrid[ip].im = -ff*ff*(smth*fft_of_rhogrid[ip].im+fft_of_rhogrid_tot[ip].im)/k2;
					/* fft_of_rhogrid now contains FFT(rhogrid)*DC*DC+FFT(rhogrid_DE)*DC where DC is the deconvolution kernel (the Green's function and smoothing kernel have also been applied) */
					/* end deconvolution. Note that the pressure term in the Poisson equation has been added above by modifying the dark energy density */

					fft_of_rhogrid_tot[ip].re = -(rho_temp_DM.re+fft_of_rhogrid_tot[ip].re)/k2;
					fft_of_rhogrid_tot[ip].im = -(rho_temp_DM.im+fft_of_rhogrid_tot[ip].im)/k2;
					/* fft_of_rhogrid_tot now contains FFT(rhogrid)*DC+FFT(rhogrid_DE) where DC is the deconvolution kernel. No smoothing has been done.
					 * This means that fft_of_rhogrid_tot now contains the full dark matter + dark energy potential (except for a multiplicative constant that will be fixed in advance_DE_nonlinear) */
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
			"P_std.re/P_gauge_std.re: %e, P_std.im/P_gauge_std.im: %e\n",
			pow(All.DarkEnergySoundSpeed,2),
			3*All.Time*(1+All.DarkEnergyW)*(pow(All.DarkEnergySoundSpeed,2)-All.DarkEnergyW),
			P_std.re/P_gauge_std.re, P_std.im/P_gauge_std.im
		     );

	MPI_Allreduce(MPI_IN_PLACE,&bad_points,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	master_printf("Number of gauge term bad points: %i (%e of total)\n",bad_points,bad_points/(1.0*PMGRID*PMGRID*PMGRID));

#endif
	/* Set mean to zero */
	if(slabstart_y == 0 && PMTask) 
		fft_of_rhogrid_tot[0].re = fft_of_rhogrid_tot[0].im = 0.0;

	if(slabstart_y == 0 && PMTask) /* This sets the mean to zero, meaning that we get the relative density delta_rho (since the k=0 part is the constant contribution) */
		fft_of_rhogrid[0].re = fft_of_rhogrid[0].im = 0.0;

	/* Do the FFT to get the potential */
	rfftwnd_mpi(fft_inverse_plan, 1, rhogrid, workspace, FFTW_TRANSPOSED_ORDER);

	/* Do the FFT of rhogrid_tot to get the total potential of the singly deconvolved dm + unconvolved de */
	rfftwnd_mpi(fft_inverse_plan, 1, rhogrid_tot, workspace, FFTW_TRANSPOSED_ORDER);

	/* Now rhogrid holds the potential.
	 * rhogrid_tot holds the potential to be used for the dark energy,
	 * but lacks a factor of G/(pi*a*boxsize) which is the standard Gagdet potential factor seen elsewhere.*/
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
	 */
	if(PMTask){
		comm_reqs=malloc(2*(nslabs_to_send+4)*sizeof(MPI_Request)); /* Received/not received status of non-blocking messages (communication handles) */
		status_DE=malloc(2*(nslabs_to_send+4)*sizeof(MPI_Status)); /* The MPI_Status return values of the communication */

		/* Send slabs */
		unsigned short req_count=0;
		for( i=0 ; i<nslabs_to_send ; ++i )
		{
			slab=slabs_to_send[i]-slabstart_x;
			/* Send rhogrid_tot slabs. */ 
			MPI_Isend(rhogrid_tot+slab*PMGRID*PMGRID2,PMGRID*PMGRID2*sizeof(fftw_real),MPI_BYTE,send_tasks[i],slabs_to_send[i],MPI_COMM_WORLD,&comm_reqs[req_count++]);
		}

		/* Receive slabs */
		for( i=0 ; i<4 ; ++i )
		{
			slab=LOGICAL_INDEX(slabs_to_recv[i]-slabstart_x+2); /* Index minus start of local patch + 2 since the received patch starts at relative index -2  */
			/* Receive rhogrid_tot slabs (this is the potential) */
			MPI_Irecv(rhogrid_tot_expanded+slab*PMGRID*PMGRID2,PMGRID*PMGRID2*sizeof(fftw_real),MPI_BYTE,recv_tasks[i],slabs_to_recv[i],MPI_COMM_WORLD,&comm_reqs[req_count++]);
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

	/* Wait point for non-blocking exchange of rhogrid_tot*/
	if(PMTask)
		if(MPI_SUCCESS!=MPI_Waitall((nslabs_to_send+4),comm_reqs,status_DE)){
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

	int next_integer_timestep=All.Ti_Current+calc_PM_step();
	double next_timestep=All.TimeBegin*exp(next_integer_timestep*All.Timebase_interval);
	master_printf("Next PM timestep: %f, da=%e\n",next_timestep,next_timestep-All.Time);

	if(PMTask)
		advance_DE_nonlinear(next_timestep-All.Time);
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
#else
/* Main function responsible for coupling dark matter and dark energy through the potential.
 * Will also advance the dark energy grid.
 * Note that in case linear dark energy is triggered the dark energy density grid
 * rhogrid_DE is the dark energy density contrast (delta) and ugrid_DE is the divergence of the velocity (divU).
 * TODO: Fix code when running on a single process (probably in comm order). */
void pmforce_periodic_DE_linear(void)
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
	char statname[MAXLEN_FILENAME];

	double fac_FD; /* Finite difference factor */
	double temp; /* Merge temp with fftw_real divU */
	const double lightspeed=C/All.UnitVelocity_in_cm_per_s;
	const double H=All.Hubble*sqrt(All.Omega0 / (All.Time * All.Time * All.Time) + (1 - All.Omega0 - All.OmegaLambda) / (All.Time * All.Time) + All.OmegaLambda/pow(All.Time,3.0*(1+All.DarkEnergyW)));

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
				"Sound horizon (units of box size): %e\n"
				"Dark energy gauge transformation important for a< %e (z > %e)\n"				
				,All.DarkEnergySoundSpeed,All.DarkEnergyW,
				All.DarkEnergySoundSpeed*lightspeed/(All.Time*H)/All.BoxSize,
				temp,1/temp-1
			     );
		DE_allocate(nslab_x);
	}

	const double vol_fac=(All.BoxSize/PMGRID)*(All.BoxSize/PMGRID)*(All.BoxSize/PMGRID)*(All.Time*All.Time*All.Time); /* Physical volume factor. Converts from density to mass */		

	/* Cloud-in-Cell interpolation. Moved to its own function for readability.
	 * Assigns mass to rhogrid while the dark energy grids are being exchanged. */
	CIC(meshmin,meshmax);

	for(y = slabstart_y; y < slabstart_y + nslab_y; y++)
		for(x = 0; x < PMGRID; x++)
			for(z = 0; z < PMGRID / 2 + 1; z++)
			{
				ip = PMGRID * (PMGRID / 2 + 1) * (y - slabstart_y) + (PMGRID / 2 + 1) * x + z;
				/* Convert dimensionless delta to mass */
				rhogrid_tot[ip].re=rhogrid_DE[ip].re*mean_DE*vol_fac;
				rhogrid_tot[ip].im=rhogrid_DE[ip].im*mean_DE*vol_fac;
			}

#ifdef DEBUG
	char fname[256];
	static int Nruns=0;
	if(All.Time>All.DarkEnergyOutputStart && Nruns<All.DarkEnergyNumOutputs){
		sprintf(fname,"%sDM_a=%.3f.%.3i",All.OutputDir,All.Time,ThisTask);
		master_printf("Writing dm+de grids\n");
		write_dm_grid(fname);
	}
#endif

	/* Do the FFT of the dark matter density field */
	rfftwnd_mpi(fft_forward_plan, 1, rhogrid, workspace, FFTW_TRANSPOSED_ORDER);

	if(first_DE_run)
		DE_IC();

#ifdef DEBUG
	if(All.Time>All.DarkEnergyOutputStart && Nruns<All.DarkEnergyNumOutputs){
		sprintf(fname,"%sDE_a=%.3f.%.3i",All.OutputDir,All.Time,ThisTask);
		write_de_grid(fname);
		sprintf(fname,"%sU_a=%.3f.%.3i",All.OutputDir,All.Time,ThisTask);
		write_U_grid(fname);
		++Nruns;
	}
#endif

	/* Prepare to calculate the dark matter power spectrum */
	fftw_complex * workspace_powergrid=(fftw_complex *) & workspace[0];
	workspace_powergrid[0].re=0;
	workspace_powergrid[0].im=0;
	temp=pow(All.BoxSize,3)*pow(All.Time,3.0)*mean_DM; /* Total mass in simulation */
	/* Translate mass grid to delta (doesn't subtract the mean as this is only relevant for the zero mode).
	 * Deconvolve. */
	for(y = slabstart_y; y < slabstart_y + nslab_y; y++)
		for(x = 0; x < PMGRID; x++)
			for(z = 0; z < PMGRID / 2 + 1; z++)
			{
				ip = PMGRID * (PMGRID / 2 + 1) * (y - slabstart_y) + (PMGRID / 2 + 1) * x + z;

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

				if(k2 > 0)
				{
					/* Deconvolution */
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
					workspace_powergrid[ip].re=fft_of_rhogrid[ip].re*ff*ff;
					workspace_powergrid[ip].im=fft_of_rhogrid[ip].im*ff*ff;
					/* Done deconvolving */
					/* Convert dimensionless delta to mass */
					workspace_powergrid[ip].re/=temp;
					workspace_powergrid[ip].im/=temp;
				}
			}

	char fname_power[256];
	sprintf(fname_power,"%s%s_DM_a=%.3f",All.OutputDir,All.PowerFileBase,All.Time);
	if(All.DetailedPowerOn)
		calc_powerspec_detailed(fname_power,workspace_powergrid);
	else
		calc_powerspec(fname_power,workspace_powergrid);


	sprintf(fname_power,"%s%s_DE_a=%.3f",All.OutputDir,All.PowerFileBase,All.Time);
	if(All.DetailedPowerOn)
		calc_powerspec_detailed(fname_power,workspace_powergrid);
	else
		calc_powerspec(fname_power,workspace_powergrid);

	workspace_powergrid=NULL; /* NOT freed, workspace will be freed later */

	const double pot_prefactor=3*(1+All.DarkEnergyW)*mean_DE*vol_fac*H*All.Time*All.BoxSize*All.BoxSize/(lightspeed*lightspeed*4*M_PI*M_PI);
	const fftw_real dm_fac=1/(All.Time*All.Time*All.Time*All.BoxSize*All.BoxSize*All.BoxSize);
	const fftw_real u_fac=3*(1+All.DarkEnergyW)*All.Time*H*pow(All.BoxSize/(2*M_PI*lightspeed),2);
#ifdef DEBUG
	master_printf("***Horizon suppression: %.4e\n",pot_prefactor);
#endif
	/* Dummy variable to store rho */
	fftw_complex rho_temp_DM;

	/* multiply with the Green's function for the potential, deconvolve */
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
				smth = exp(-k2 * asmth2);

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

					rho_temp_DM=fft_of_rhogrid[ip];

					rhogrid_tot[ip].re=(rhogrid_tot[ip].re+pot_prefactor*ugrid_DE[ip].re/k2)*PMGRID*PMGRID*PMGRID;
					rhogrid_tot[ip].im=(rhogrid_tot[ip].im+pot_prefactor*ugrid_DE[ip].im/k2)*PMGRID*PMGRID*PMGRID;

					/* Now do second deconvolution of dark matter potential, and a single deconvolution of the dark energy potential (corresponding to the CIC from the grid to the particles). 
					 * Multiply with the Green's function and smoothing kernel. 
					 * Only the dark matter needs to be long range smoothed, dark energy is applicable on grid scale*/
					fft_of_rhogrid[ip].re = -ff*ff*(smth*rho_temp_DM.re+rhogrid_tot[ip].re)/k2;
					fft_of_rhogrid[ip].im = -ff*ff*(smth*rho_temp_DM.im+rhogrid_tot[ip].im)/k2;
					/* fft_of_rhogrid now contains FFT(rhogrid)*DC*DC+FFT(rhogrid_DE)*DC where DC is the deconvolution kernel (the Green's function and smoothing kernel have also been applied) */

					//	rhogrid_tot[ip].re = -(rho_temp_DM.re+rhogrid_tot[ip].re)/k2;
					//	rhogrid_tot[ip].im = -(rho_temp_DM.im+rhogrid_tot[ip].im)/k2;
					rhogrid_tot[ip].re = -4*M_PI*All.G*All.Time*All.Time*(dm_fac*rho_temp_DM.re+mean_DE*(rhogrid_DE[ip].re+u_fac*ugrid_DE[ip].re/k2));
					rhogrid_tot[ip].im = -4*M_PI*All.G*All.Time*All.Time*(dm_fac*rho_temp_DM.im+mean_DE*(rhogrid_DE[ip].im+u_fac*ugrid_DE[ip].im/k2));
					/* rhogrid_tot now contains FFT(rhogrid)*DC+FFT(rhogrid_DE) where DC is the deconvolution kernel. No smoothing has been done.
					 * This means that fft_of_rhogrid_tot now contains the full dark matter + dark energy potential */

					/* end deconvolution. Note that the pressure term in the Poisson equation has been added above by modifying the dark energy density */
				}

			}
	/* The rhogrid_tot potential array isn't needed and some memory could be saved by keeping everything in the rhogrid array */

	/* Set mean to zero */
	if(slabstart_y == 0 && PMTask) 
		rhogrid_tot[0].re = rhogrid_tot[0].im = 0.0;

	if(slabstart_y == 0 && PMTask) /* This sets the mean to zero, meaning that we get the relative density delta_rho (since the k=0 part is the constant contribution) */
		fft_of_rhogrid[0].re = fft_of_rhogrid[0].im = 0.0;

	/* Do the FFT to get the potential */
	rfftwnd_mpi(fft_inverse_plan, 1, rhogrid, workspace, FFTW_TRANSPOSED_ORDER);
	/* Note: The inverse FFT scales the data by PMGRID*PMGRID*PMGRID */

	/* Now rhogrid holds the potential, rhogrid_tot holds the k2*phi term */
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

	/* Calculate the next PM timestep and update the DE equations - use finite differences */
	double hubble_a = All.Omega0 / (All.Time * All.Time * All.Time) + (1 - All.Omega0 - All.OmegaLambda) / (All.Time * All.Time) +  All.OmegaLambda/pow(All.Time,3.0*(1+All.DarkEnergyW));
	hubble_a = All.Hubble * sqrt(hubble_a);
	find_dt_displacement_constraint(hubble_a*All.Time*All.Time);

	int next_integer_timestep=All.Ti_Current+calc_PM_step();
	double next_timestep=All.TimeBegin*exp(next_integer_timestep*All.Timebase_interval);

	master_printf("Next PM timestep: %f, da=%e\n",next_timestep,next_timestep-All.Time);

	if(PMTask)
		advance_DE_linear(next_timestep-All.Time);
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
/* Calculate power spectrum. Saves to file fname and takes the fftw array to calculate the power spectrum from. */
void calc_powerspec(char * fname, fftw_complex* fft_arr){
	const fftw_real kNorm=2.0*M_PI/(All.BoxSize);
	const fftw_real tophat_scale=8*CM_PER_MPC/All.UnitLength_in_cm;
	const int Nyquist=PMGRID/2;
	const unsigned int nr_freq=1.4*Nyquist-1; /* Doesn't really make sense to go beyond the Nyquist frequency*/ 

	fftw_real * power_x=my_malloc(nr_freq*sizeof(fftw_real));
	fftw_real * power_y=my_malloc(nr_freq*sizeof(fftw_real));
	fftw_real * power_z=my_malloc(nr_freq*sizeof(fftw_real));
	int * num_x=my_malloc(nr_freq*sizeof(int));
	int * num_y=my_malloc(nr_freq*sizeof(int));
	int * num_z=my_malloc(nr_freq*sizeof(int));
	unsigned int * k=my_malloc(nr_freq*sizeof(fftw_real));
	int x,y,z,i,ip;

	master_printf("Calculating power spectrum and saving to file %s.\n",fname);

	for( i=0 ; i<nr_freq ; ++i )
	{
		power_y[i]=0;
		num_y[i]=0;
	}

	int nx,ny,nz;  /* Wavenumber in integer, to go from nx to kx multiply by kNorm*/
	size_t kk; /* Length of k vector (to nearest integer) */
	size_t zdim=PMGRID/2+1;
	fftw_real  sigma_z=0, sigma_x=0, sigma_y=0, kR=0, sigma=0; /* Auxilliary variables */
	fftw_real k2, re_part, im_part; /* k squared and real and imaginary parts of fourier output */

	for(y = slabstart_y; y < slabstart_y + nslab_y; y++)
	{
		for( i=0 ; i<nr_freq ; ++i )
		{
			power_x[i]=0;
			num_x[i]=0;
		}
		sigma_x=0;
		for(x=0 ; x<PMGRID ; ++x )
		{
			for( i=0 ; i<nr_freq ; ++i )
			{
				power_z[i]=0;
				num_z[i]=0;
			}

			sigma_z=0;
			for( z=0 ; z<zdim ; ++z )
			{
				if( x>Nyquist )
					nx=x-PMGRID;
				else
					nx=x;
				if( y>Nyquist )
					ny=y-PMGRID;
				else
					ny=y;
				if(z>Nyquist )
					nz=z-PMGRID;
				else
					nz=z;


				k2=nx*nx+ny*ny+nz*nz;

				if( k2>0 )
				{
					kk=floor(sqrt(k2)+0.5); //Norm of k squared (0.5 factor rounds it to nearest integer)

#ifdef DEBUG
					assert(kk>=1);
#endif
					if( kk <= nr_freq && kk >= 1 )
					{
						/* Symmetry removing part */

						if( z==0 || z == Nyquist )
						{
							if(ny<= 0 && nx <= 0)
							{
								continue;
							}
							if( nx>=0 && ny<=0 )
							{
								if(abs(ny)>nx)
								{
									continue;
								}
							}
							if( nx<=0 && ny>=0 )
							{
								if( abs(nx)>=ny )
								{
									continue;
								}
							}

						}
						/* End symmetry removing part */

						k[kk-1]=k2; /* This is the wave number. Will normalize later */

						ip = PMGRID * (PMGRID / 2 + 1) * (y - slabstart_y) + (PMGRID / 2 + 1) * x + z;

						re_part=fft_arr[ip].re;
						im_part=fft_arr[ip].im;

						power_z[kk-1]+=re_part*re_part+im_part*im_part; //This is the power. Will normalize later
						num_z[kk-1]+=1;

						/* Sigma8 calculation */
						k2=kNorm*sqrt((double) k2);
						kR=k2*tophat_scale;
						kR=3.0*(sin(kR)-kR*cos(kR))/pow(kR,3);
						sigma_z+=power_z[kk-1]*kR*kR*k2*k2;	
					}//kk test
				} //k2>0
			} //z loop
			for(i=0 ; i<nr_freq ; ++i )
			{
				power_x[i]+=power_z[i];
				num_x[i]+=num_z[i];
			}
			sigma_x+=sigma_z;
		} //x loop

		for(i=0 ; i<nr_freq ; ++i )
		{
			power_y[i]+=power_x[i];
			num_y[i]+=num_x[i];
		}
		sigma_y+=sigma_x;
	}//y loop

	MPI_Allreduce(MPI_IN_PLACE,num_y,nr_freq,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE,power_y,nr_freq,FFTW_MPITYPE,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE,&sigma_y,1,FFTW_MPITYPE,MPI_SUM,MPI_COMM_WORLD);

	/* Normalize wavenumbers and power to correct values */
	for(i=0 ; i<nr_freq ; ++i )
	{
		power_y[i]/=num_y[i];
	}

	sigma=sigma_y;
	sigma*=1./(2*M_PI*M_PI)*4.0/3*M_PI*pow(tophat_scale,3);
	sigma=sqrt(sigma);

	if(ThisTask==0){
		FILE *fid;
		fid=fopen(fname,"w");
		if(!fid){
			fprintf(stderr,"Error opening file: %s: %s.\n",fname,strerror(errno));
		}
		else{
			fprintf(fid,"sigma%.2f=%.5e, PMGRID=%i, Boxsize=%.8e.\n",
					tophat_scale*(All.UnitLength_in_cm/CM_PER_MPC),sigma,PMGRID,All.BoxSize);
			fprintf(fid,"k\tmodes\tpower\n");

			size_t freq;
			for( freq=0 ; freq<nr_freq ; ++freq ){
				fprintf(fid,"%.5e\t%i\t%.5e\n",sqrt((double) k[freq]),num_y[freq],power_y[freq]);
			}
			fclose(fid);
		}
	}

	free(power_x);
	free(power_y);
	free(power_z);
	free(num_x);
	free(num_y);
	free(num_z);
	free(k);
}

/* Calculate the power spectrum with all allowed k2 modes. 
 * More noisy, but makes a customised binning possible.  */
void calc_powerspec_detailed(char * fname, fftw_complex* fft_arr){
	const int Nyquist=PMGRID/2;
	/* Maximum number of different k2 modes in box below Nyquist */
	const int k2_max=3*Nyquist*Nyquist;
	int x,y,z, ip, nx, ny, nz;
	int zdim=PMGRID/2+1;
	unsigned long int k2;

	int * k2_multi=my_malloc(k2_max*sizeof(int)); /* Multiplicity of a given k2 index. */
	fftw_real * power_arr=my_malloc(k2_max*sizeof(fftw_real)); /* Power at a given k */

	master_printf("Calculating power spectrum and saving to file %s.\n",fname);
	
	for( k2=0 ; k2<k2_max ; ++k2 )
	{
		k2_multi[k2]=0;
		power_arr[k2]=0;
	}

	const fftw_real kNorm=2.0*M_PI/(All.BoxSize);
	const fftw_real tophat_scale=8*CM_PER_MPC/All.UnitLength_in_cm;

	fftw_real  kR; /* Auxilliary variable */
	fftw_real re_part, im_part; /* k squared and real and imaginary parts of fourier output */

	for(y = slabstart_y; y < slabstart_y + nslab_y; y++)
		for(x=0 ; x<PMGRID ; ++x )
			for( z=0 ; z<zdim ; ++z ){
				if( x>Nyquist )
					nx=x-PMGRID;
				else
					nx=x;
				if( y>Nyquist )
					ny=y-PMGRID;
				else
					ny=y;
				if(z>Nyquist )
					nz=z-PMGRID;
				else
					nz=z;

				k2=nx*nx+ny*ny+nz*nz;
				if(k2==0)
					continue;

				/* Symmetry removing part */
				if( z==0 || z == Nyquist )
				{
					if(ny<= 0 && nx <= 0)
					{
						continue;
					}
					if( nx>=0 && ny<=0 )
					{
						if(abs(ny)>nx)
						{
							continue;
						}
					}
					if( nx<=0 && ny>=0 )
					{
						if( abs(nx)>=ny )
						{
							continue;
						}
					}

				}

				k2_multi[k2]+=1; /* Increase k2 multiplicity. k2 itself is the index */

				ip = PMGRID * (PMGRID / 2 + 1) * (y - slabstart_y) + (PMGRID / 2 + 1) * x + z;

				re_part=fft_arr[ip].re;
				im_part=fft_arr[ip].im;
				power_arr[k2]+=re_part*re_part+im_part*im_part; //This is the power. Will normalize later
			}
	MPI_Allreduce(MPI_IN_PLACE,k2_multi,k2_max,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE,power_arr,k2_max,FFTW_MPITYPE,MPI_SUM,MPI_COMM_WORLD);

	for( k2=0 ; k2<k2_max ; ++k2 )
	{
		power_arr[k2]=power_arr[k2]/k2_multi[k2];
	}

	fftw_real k;
	fftw_real sigma=0;
	for( k2=0 ;k2<k2_max  ; ++k2 )
	{
		if(k2_multi[k2]==0)
			continue;
		k=kNorm*sqrt((double) k2);
		kR=k*tophat_scale;
		kR=3.0*(sin(kR)-kR*cos(kR))/pow(kR,3);
		sigma+=power_arr[k2]*kR*kR*k*k;	
	}

	MPI_Allreduce(MPI_IN_PLACE,&sigma,1,FFTW_MPITYPE,MPI_SUM,MPI_COMM_WORLD);

	sigma*=1./(2*M_PI*M_PI)*4.0/3*M_PI*pow(tophat_scale,3);	
	sigma=sqrt(sigma);

	fftw_real power;
	if(ThisTask==0){
		FILE *fid;
		fid=fopen(fname,"w");
		if(!fid){
			fprintf(stderr,"Error opening file: %s: %s.\n",fname,strerror(errno));
		}
		else{
			fprintf(fid,"sigma%.2f=%.5e, PMGRID=%i, Boxsize=%.8e.\n",
					tophat_scale*(All.UnitLength_in_cm/CM_PER_MPC),sigma,PMGRID,All.BoxSize);
			fprintf(fid,"k\tmodes\tpower\n");

			for( k2=0 ; k2<k2_max ; ++k2 ){
				if(k2_multi[k2]==0)
					continue;
				power=power_arr[k2];
				fprintf(fid,"%.5e\t%i\t%.5e\n",kNorm*sqrt((double) k2),k2_multi[k2],power);
			}
			fclose(fid);
		}
	}
	free(k2_multi);
	free(power_arr);

}

void write_dm_grid(char* fname_DM){
	if(PMTask){
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

#ifdef DYNAMICAL_DE
void DE_IC(void){
	int x,y,z,ip;
	int kx,ky,kz,k2;
	fftw_real fx,fy,fz, ff;
	const fftw_real cs2=All.DarkEnergySoundSpeed*All.DarkEnergySoundSpeed;
	const fftw_real H=All.Hubble*sqrt(All.Omega0 / (All.Time*All.Time*All.Time) + (1 - All.Omega0 - All.OmegaLambda) / (All.Time*All.Time) + All.OmegaLambda/pow(All.Time,3.0*(1+All.DarkEnergyW)));
	const fftw_real mass_tot=pow(All.BoxSize,3)*pow(All.Time,3.0)*mean_DM; /* Total mass in simulation */
	const fftw_real fac_delta=(1.0+All.DarkEnergyW)*(1.0-2*cs2)/(1.0-3*All.DarkEnergyW+cs2);
	master_printf("Dark energy suppression: %.3e\n",fac_delta);
#ifdef NONLINEAR_DE
	int i,index;
	for(y = slabstart_y; y < slabstart_y + nslab_y; y++)
		for(x = 0; x < PMGRID; x++)
			for(z = 0; z < PMGRID / 2 + 1; z++)
			{
				ip = PMGRID * (PMGRID / 2 + 1) * (y - slabstart_y) + (PMGRID / 2 + 1) * x + z;
				/* Convert dimensionless delta to mass */
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

				if(k2 > 0)
				{
					/* Deconvolution */
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
					fft_of_divU[ip].re=fft_of_rhogrid[ip].re*ff*ff;
					fft_of_divU[ip].im=fft_of_rhogrid[ip].im*ff*ff;
					/* Done deconvolving */
					fft_of_divU[ip].re*=fac_delta/mass_tot;
					fft_of_divU[ip].im*=fac_delta/mass_tot;
				}
				else{
					fft_of_divU[ip].re=0;
					fft_of_divU[ip].im=0;
				}


			}
	/* divU_grid is not the divergence of the velocity, it is merely a workspace
	 * Do inverse FFT to get the values in real space */
	rfftwnd_mpi(fft_inverse_plan, 1, divU_grid, workspace, FFTW_TRANSPOSED_ORDER);
	if(PMTask){
		for( x=0 ; x<nslab_x ; ++x )
			for( y=0 ; y<PMGRID ; ++y )
				for( z=0 ; z<PMGRID ; ++z )
				{
					/* FFT array indexing */
					ip = INDMAP(x,y,z);
					/* Standard indexing */
					index=x*PMGRID*PMGRID+y*PMGRID+z;
					
					/* Relative mass pertubation */
					rhogrid_tot[ip]=divU_grid[ip];
					/* Full rho */
					rhogrid_DE[index]=(divU_grid[ip]+1)*mean_DE;
					
					for( i=0 ; i<3 ; ++i )
					{
						ugrid_DE[index][i]=0;
					}
				}
	}


#else
	const fftw_real fac_U=(-1+6*cs2*(cs2-All.DarkEnergyW)/(1-3*All.DarkEnergyW+cs2))*H*All.Time;
	if(PMTask)
		rhogrid_DE[0].re=rhogrid_DE[0].im=ugrid_DE[0].re=ugrid_DE[0].im=0;

	for(y = slabstart_y; y < slabstart_y + nslab_y; y++)
		for(x = 0; x < PMGRID; x++)
			for(z = 0; z < PMGRID / 2 + 1; z++)
			{

				ip = PMGRID * (PMGRID / 2 + 1) * (y - slabstart_y) + (PMGRID / 2 + 1) * x + z;
				/* Convert dimensionless delta to mass */
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

				if(k2 > 0)
				{
					/* Deconvolution */
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
					rhogrid_DE[ip].re=fft_of_rhogrid[ip].re*ff*ff;
					rhogrid_DE[ip].im=fft_of_rhogrid[ip].im*ff*ff;
					ugrid_DE[ip].re=fft_of_rhogrid[ip].re*ff*ff;          			
					ugrid_DE[ip].im=fft_of_rhogrid[ip].im*ff*ff;			
					/* Done deconvolving */
					rhogrid_DE[ip].re=fac_delta*rhogrid_DE[ip].re/mass_tot;
					rhogrid_DE[ip].im=fac_delta*rhogrid_DE[ip].im/mass_tot;
					ugrid_DE[ip].re=fac_U*ugrid_DE[ip].re/mass_tot;
					ugrid_DE[ip].im=fac_U*ugrid_DE[ip].im/mass_tot;
				}
				else{
					rhogrid_DE[ip].re=0;
					rhogrid_DE[ip].im=0;
					ugrid_DE[ip].re=0;
					ugrid_DE[ip].im=0;
				}
			}
#endif

	master_printf("Done with dark energy initial conditions\n");
}

#ifdef NONLINEAR_DE
/* The equation of state for the dark energy. Be sure to get the units right (in Gadget units with h=1).
 * Customise this function according to your favorite model.
 * This function ought to be an integrated part of the advance_DE_nonlinear function for performance.
 * It is kept here as a seperate function for convenience (with the hope that the compiler will inline it)*/
inline fftw_real equation_of_state_DE(fftw_real rho){
	fftw_real P;
	const fftw_real rho_inf=0.95*All.OmegaLambda*3.0*All.Hubble*All.Hubble/(8.0*M_PI*All.G);
	P=2*rho/(1+exp(-2*(rho-rho_inf)/rho_inf))-2*rho;
	return P;
}

/* Gives the derivative dP/drho (that is: the sound speed) from the dark energy equation of state.
 * As above, this function ought to be integrated into advance_DE_nonlinear.*/
inline fftw_real equation_of_state_derivative_DE(fftw_real rho){
	fftw_real dP_drho;
	const fftw_real rho_inf=0.95*All.OmegaLambda*3.0*All.Hubble*All.Hubble/(8.0*M_PI*All.G);
	fftw_real exponent=exp(-2*(rho-rho_inf)/rho_inf);
	dP_drho=2/(1+exponent)-2+2*rho/rho_inf*exponent*pow((1+exponent),-2);
	return dP_drho;
}

/* Advance the dark energy peturbations through the continuity and Euler equation. 
 * Assumes that the dark energy pertubations are parametrised in the full density and 3-d velocity. 
 * Everything is in real space.*/
void advance_DE_nonlinear(const fftw_real da){
	const double fac_FD = 1 / (2.0 * All.BoxSize / PMGRID);	/* for finite differencing. Factor 1/2 is part of the FD coefficients, could be moved there as well */
	const double potfac =All.G / (All.Time*M_PI * All.BoxSize);	/* to get potential */
	/* Provide easier notation (some of these will be removed by the compiler optimiser anyway) */
	const double a=All.Time;
	const double lightspeed=C/All.UnitVelocity_in_cm_per_s;
	const double H=All.Hubble*sqrt(All.Omega0 / (a * a * a) + (1 - All.Omega0 - All.OmegaLambda) / (a * a) + All.OmegaLambda/pow(a,3.0*(1+All.DarkEnergyW)));

	/* Indexing variables */
	int x,y,z;
	unsigned int index;
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
	fftw_real Pll,Pl,Pr,Prr; /* Easy notation for pressure gradient of pressure */
	fftw_real rho_plus_P; /* rho + P/c^2 */
	fftw_real rho_plus_P_reci; /* 1/(rho + P/c^2) */
	/* Temporary storage for the new density and velocity fields */
	fftw_real *new_rhogrid_DE_expanded=my_malloc((4+nslab_x)*PMGRID*PMGRID*sizeof(fftw_real));
	fftw_real (*new_ugrid_DE_expanded)[3]=my_malloc(3*(4+nslab_x)*PMGRID*PMGRID*sizeof(fftw_real));
	fftw_real *new_rhogrid_DE=new_rhogrid_DE_expanded+2*PMGRID*PMGRID;
	fftw_real (*new_ugrid_DE)[3]=new_ugrid_DE_expanded+2*PMGRID*PMGRID;

#ifdef DEBUG
	/* Sanity check to make sure we stay unrelativistic */
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
					Pll=equation_of_state_DE( rhogrid_DE_expanded[(xll * PMGRID + yll) * PMGRID + zll]);
					Pl =equation_of_state_DE( rhogrid_DE_expanded[(xl * PMGRID + yl) * PMGRID + zl]);
					Pr =equation_of_state_DE( rhogrid_DE_expanded[(xr * PMGRID + yr) * PMGRID + zr]);
					Prr=equation_of_state_DE( rhogrid_DE_expanded[(xrr * PMGRID + yrr) * PMGRID + zrr]);
					gradP[dim]=fac_FD*(
							(4.0/3.0)*
							( -Pl 
							  +Pr 
							)
							+
							(1.0 / 6.0) *
							(  Pll 
							   -Prr 
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
									(   ugrid_DE_expanded[(xll * PMGRID + yll) * PMGRID + zll][Udim] 
									    - ugrid_DE_expanded[(xrr * PMGRID + yrr) * PMGRID + zrr][Udim]
									)
							       );
					}
				}

				index=(x-2)*PMGRID*PMGRID+y*PMGRID+z;

				for( dim=0 ;dim<3  ; ++dim )
					U_prev[dim]=ugrid_DE[index][dim];
				rho_plus_P=rhogrid_DE[index]+equation_of_state_DE(rhogrid_DE[index]);
				rho_plus_P_reci=1/rho_plus_P;

				fftw_real divU=(gradU[0][0]+gradU[1][1]+gradU[2][2]);
				/* Unphysical to have rho+P<0 due to the weak energy condition in GR.
				 * Establish sanity check.*/
				if(rho_plus_P<0){
					mpi_fprintf(stderr,"WARNING: rho+P<0 for point (%i, %i, %i).\n"
							"Mean rho: %e, full rho: %e (drho: %e), rho + P: %e\n"
							"DivU: %e\n",
							x-2+slabstart_x,y,z,
							mean_DE,
							rhogrid_DE[index], /* rho */
							rhogrid_DE[index]-mean_DE, /* drho */
							rho_plus_P, /* rho + P */
							divU /* sgn(divU) */
						   );
					rho_plus_P_reci=0;
				}

				/* Calculate drho/dt (conformal time) */
				drhoda_current=
					-3.0*a*H*rho_plus_P
					-(U_prev[0]*gradrho[0]+U_prev[1]*gradrho[1]+U_prev[2]*gradrho[2])
					-(U_prev[0]*gradP[0]+U_prev[1]*gradP[1]+U_prev[2]*gradP[2])
					-(rho_plus_P)*(gradU[0][0]+gradU[1][1]+gradU[2][2]);

				/* Calculate dP/dt=dP/drho*drho/dt */
				Pdot=equation_of_state_derivative_DE(rhogrid_DE[index])*drhoda_current;

				/* Change to drho/da */
				drhoda_current=drhoda_current/(a*a*H);

				/* Update the density pertubation */
				new_rhogrid_DE[index]=rhogrid_DE[index]+drhoda_current*da;

#ifdef DEBUG
				/* Sanity check */
				if(new_rhogrid_DE[index]<0){
					new_rhogrid_DE[index]=0;
					mpi_fprintf(stderr,"WARNING: Negative dark energy rho\n");
				}

				U_sq=0;
#endif
				/* Update the velocity pertubation */
				for( dim=0 ;dim<3  ; ++dim )
				{
					dUda[dim]=
						-a*H*U_prev[dim]
						-(U_prev[0]*gradU[dim][0]+U_prev[1]*gradU[dim][1]+U_prev[2]*gradU[dim][2]) /* U dot grad U[dim] */
						-lightspeed*lightspeed*gradP[dim]*rho_plus_P_reci
						-U_prev[dim]*Pdot*rho_plus_P_reci
						-gradphi[dim]; 
					/* Change from dot U to dU/da */
					dUda[dim]=dUda[dim]/(a*a*H);
					new_ugrid_DE[index][dim]=ugrid_DE[index][dim]+dUda[dim]*da;

#ifdef DEBUG
					/* Speed sanity check. Needs to be safely below the speed of light. */
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
}
#else
/* Advance the linear pertubations on the grid. Everything is in Fourier space unlike the non-linear case. */
void advance_DE_linear(const fftw_real da){
	/* Provide easier notation (some of these will be removed by the compiler optimiser anyway) */
	const fftw_real a=All.Time;
	const fftw_real lightspeed=C/All.UnitVelocity_in_cm_per_s;
	const fftw_real H=All.Hubble*sqrt(All.Omega0 / (a * a * a) + (1 - All.Omega0 - All.OmegaLambda) / (a * a) + All.OmegaLambda/pow(a,3.0*(1+All.DarkEnergyW)));
	const fftw_real Hubble_len_inv=H/lightspeed;
	const fftw_real cs2=All.DarkEnergySoundSpeed*All.DarkEnergySoundSpeed;
	const fftw_real w=All.DarkEnergyW;
	const fftw_real k2Norm=(2*M_PI/All.BoxSize)*(2*M_PI/All.BoxSize);

	int x,y,z,ip;
	int kx,ky,kz;
	unsigned long k2;
	fftw_real k2_phys;
	fftw_complex theta, phi, delta;
	fftw_complex theta_dot,delta_dot;

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

				k2 = kx*kx + ky*ky + kz*kz ; /* Note: k2 is the integer wave number squared. The physical k is k_phys=2 M_PI/BoxSize k */
				if(k2==0)
					continue;
				k2_phys=k2Norm*k2;
				ip = PMGRID * (PMGRID / 2 + 1) * (y - slabstart_y) + (PMGRID / 2 + 1) * x + z;
				theta=ugrid_DE[ip];
				phi=rhogrid_tot[ip];
				delta=rhogrid_DE[ip];

				delta_dot.re=-(1+w)*theta.re-3*(cs2-w)*a*H*delta.re-9*(1+w)*(cs2-w)*a*a*Hubble_len_inv*Hubble_len_inv/k2_phys*theta.re;
				delta_dot.im=-(1+w)*theta.im-3*(cs2-w)*a*H*delta.im-9*(1+w)*(cs2-w)*a*a*Hubble_len_inv*Hubble_len_inv/k2_phys*theta.im;

				theta_dot.re=-(1-3*cs2)*a*H*theta.re+cs2*lightspeed*lightspeed*k2_phys/(1+w)*delta.re+phi.re;
				theta_dot.im=-(1-3*cs2)*a*H*theta.im+cs2*lightspeed*lightspeed*k2_phys/(1+w)*delta.im+phi.im;

				rhogrid_DE[ip].re+=delta_dot.re/(a*a*H)*da;
				rhogrid_DE[ip].im+=delta_dot.im/(a*a*H)*da;

				ugrid_DE[ip].re+=theta_dot.re/(a*a*H)*da;
				ugrid_DE[ip].im+=theta_dot.im/(a*a*H)*da;
			}

}



#endif

#ifdef DEBUG
#ifdef NONLINEAR_DE
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


	if(ThisTask==0){
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
	if(ThisTask==0){
		print_dummy=mean_DM*All.BoxSize*All.BoxSize*All.BoxSize/(PMGRID*PMGRID*PMGRID);
		printf("Background mass of dark matter in comoving mesh cell is: %e\n",print_dummy);
		print_dummy*=All.Time*All.Time*All.Time;
		printf("Background mass of dark matter in physical mesh cell: " 
				"%e (cosmo term: %e, delta_mean: %e, std dev: %e, min: %e, max: %e)\n"
				"Dark matter ratio of mean to cosmo term: %e\n",
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

	if(ThisTask==0){
		print_dummy=mean_DE*All.BoxSize*All.BoxSize*All.BoxSize/(PMGRID*PMGRID*PMGRID);
		printf("Background mass of dark energy in comoving mesh cell is: %e\n",print_dummy);
		print_dummy*=All.Time*All.Time*All.Time;
		printf("Background mass of dark energy in physical mesh cell: "
				"%e (cosmo term: %e, delta_mean: %e, std dev: %e, min: %e, max: %e)\n"
				"Dark energy ratio of mean to cosmo term: %e\n",
				mean,print_dummy,delta_mean,std_dev,min,max,mean/print_dummy);
		sprintf(buf,"%e\t%e\t%e\t%e\t%e\n",mean,delta_mean,std_dev,min,max);
		strcat(out,buf);
		fprintf(fd,"%s",out);
	}
	if(ThisTask==0)
		fclose(fd);
}
#endif //NONLINEAR_DE
#endif



void write_de_grid(char* fname_DE){
	if(PMTask){
#ifdef NONLINEAR_DE
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
#else
		int i,j,k,index;
		int npts=nslab_x*PMGRID*PMGRID;
		size_t size=fftsize*sizeof(fftw_real);
		fftw_real* fftgrid=my_malloc(size);
		float *slabs=my_malloc(npts*sizeof(float));

		memcpy(fftgrid,rhogrid_tot,size);	

		rfftwnd_mpi(fft_inverse_plan, 1, fftgrid, workspace, FFTW_TRANSPOSED_ORDER);
		for( i=0 ; i<nslab_x ; ++i )
			for( j=0 ; j<PMGRID ; ++j )
				for( k=0 ; k<PMGRID ; ++k )
				{
					index=i*PMGRID*PMGRID+j*PMGRID+k;
					slabs[index]=(float) fftgrid[INDMAP(i,j,k)]/(PMGRID*PMGRID*PMGRID);
				}



		FILE *fd=fopen(fname_DE,"w");
		write_header(fd);
		fwrite(slabs,npts,sizeof(float),fd);
		fclose(fd);

		free(fftgrid);
		free(slabs);
#endif
	}

}

void write_U_grid(char* fname_U){
	if(PMTask){
#ifdef NONLINEAR_DE
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
#else
		int i,j,k,index;
		int npts=nslab_x*PMGRID*PMGRID;
		size_t size=fftsize*sizeof(fftw_real);
		fftw_real* fftgrid=my_malloc(size);
		float *slabs=my_malloc(npts*sizeof(float));

		memcpy(fftgrid,ugrid_DE,size);	

		rfftwnd_mpi(fft_inverse_plan, 1, fftgrid, workspace, FFTW_TRANSPOSED_ORDER);
		for( i=0 ; i<nslab_x ; ++i )
			for( j=0 ; j<PMGRID ; ++j )
				for( k=0 ; k<PMGRID ; ++k )
				{
					index=i*PMGRID*PMGRID+j*PMGRID+k;
					slabs[index]=(float) fftgrid[INDMAP(i,j,k)]/(PMGRID*PMGRID*PMGRID);
				}



		FILE *fd=fopen(fname_U,"w");
		write_header(fd);
		fwrite(slabs,npts,sizeof(float),fd);
		fclose(fd);

		free(fftgrid);
		free(slabs);
#endif
	}
}
#endif //DYNAMICAL_DE


#endif
#endif



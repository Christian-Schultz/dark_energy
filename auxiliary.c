/*
 * =====================================================================================
 *
 *       Filename:  aux.c
 *
 *    Description:  Auxillary functions
 *
 *        Version:  1.0
 *        Created:  12.12.2011 14:24:56
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Christian Schultz (CS), christian@phys.au.dk
 *        Company:  Aarhus University
 *
 * =====================================================================================
 */
#include "allvars.h"
#include <stdarg.h>
#include <string.h>
#include <assert.h>
#include "proto.h"
#include "mpi.h"

#ifdef NOTYPEPREFIX_FFTW
#include        <rfftw_mpi.h>
#else
#ifdef DOUBLEPRECISION_FFTW
#include     <drfftw_mpi.h>	/* double precision FFTW */
#else
#include     <srfftw_mpi.h>
#endif
#endif


void error_check(void){

	if( (All.SofteningGas==0.0 || All.SofteningGasMaxPhys==0.0) && Ntype[0]!=0){
		master_fprintf(stderr,"Error with softening values for particle type 0. Exiting\n");
		endrun(0);
	}

	if( (All.SofteningHalo==0.0 || All.SofteningHaloMaxPhys==0.0) && Ntype[1]!=0 )
	{
		master_fprintf(stderr,"Error with softening values for particle type 1. Exiting\n");
		endrun(0);

	}

	if( (All.SofteningDisk==0.0 || All.SofteningDiskMaxPhys==0.0) && Ntype[2]!=0 )
	{

		master_fprintf(stderr,"Error with softening values for particle type 2. Exiting\n");
		endrun(0);
	}

	if( (All.SofteningBulge==0.0 || All.SofteningBulgeMaxPhys==0.0) && Ntype[3]!=0 )
	{

		master_fprintf(stderr,"Error with softening values for particle type 3. Exiting\n");
		endrun(0);
	}

	if( (All.SofteningStars==0.0 || All.SofteningStarsMaxPhys==0.0) && Ntype[4]!=0 )
	{

		master_fprintf(stderr,"Error with softening values for particle type 4. Exiting\n");
		endrun(0);
	}

	if( (All.SofteningBndry==0.0 || All.SofteningBndryMaxPhys==0.0) && Ntype[5]!=0 )
	{

		master_fprintf(stderr,"Error with softening values for particle type 5. Exiting\n");
		endrun(0);
	}
#ifdef DYNAMICAL_DE
	if(All.ComovingIntegrationOn==0){
		master_fprintf(stderr,"Error: Comoving integration is disabled, but dark energy enabled. This is not currently supported. Terminating\n");
		endrun(0);
	}
#endif


}

int mpi_printf(const char * format, ...){
	int n; 
	va_list ap;
	char *str;
	char tmp[20];
	va_start(ap,format);

	sprintf(tmp,"Task %i: ",ThisTask);
	str=malloc(sizeof(char)*(1+strlen(format)+strlen(tmp)));
	memcpy(str,tmp,strlen(tmp));
	memcpy(str+strlen(tmp),format,strlen(format)+1);

	n=vprintf(str,ap);
	fflush(stdout);
	va_end(ap);

	free(str);
	return n;
}


int mpi_fprintf(FILE * stream, const char * format, ...){
	int n; 
	va_list ap;
	char *str;
	char tmp[20];
	va_start(ap,format);

	sprintf(tmp,"Task %i: ",ThisTask);
	str=malloc(sizeof(char)*(1+strlen(format)+strlen(tmp)));
	memcpy(str,tmp,strlen(tmp));
	memcpy(str+strlen(tmp),format,strlen(format)+1);

	n=vfprintf(stream,str,ap);
	fflush(stream);
	va_end(ap);

	free(str);
	return n;
}


int master_printf( const char * format, ...){
	int n=0;

	if(ThisTask==0){
		va_list ap;
		va_start(ap,format);
		n=vprintf(format,ap);
		fflush(stdout);
		va_end(ap);

	}
	return n;
}

int master_fprintf(FILE *stream, const char * format, ...){
	int n=0;

	if(ThisTask==0){
		va_list ap;
		va_start(ap,format);
		n=vfprintf(stream,format,ap);
		fflush(stdout);
		va_end(ap);

	}
	return n;
}

#ifdef DEBUG
/*__attribute__(( alloc_size(1) )) */  void* my_malloc_debug(size_t size,const char *file,int line) 
{
	void* ret;
	if(!( ret = malloc(size))){
		mpi_fprintf(stderr,"Error allocating memory in %s:%i. Error: %s\n",file,line,strerror(errno));
		endrun(2);
	}
	return ret;
} 
#endif

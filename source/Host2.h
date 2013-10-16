#include <stdio.h>
#include <stdlib.h>
#include "define.h"

#ifndef HOST_CLASS
#define HOST_CLASS

class Host{

public:
	FILE *masterfile;
	char masterfilename[64];
	FILE *pathfile;				//used in multisim mode, contains list of directories
	char pathfilename[64];

	int Nst;				//Number of simulations

	int devCount;
	int runtimeVersion;
	int driverVersion;

	int *NB;				//number of bodies increased to integer block size
	int *icNB;				//initial number of NB
	int *N4;				//number of bodies divided by 4
	int *N2;				//number of bodies divided by 2
	int *Nconst;				//number of massive bodies in test particle run
	int *Nmin;				//minimal number of bodies
	double *rho;				//default density of bodies


	struct Parameter P;			//parameters, for all Simulations the same
	struct GSFiles *GSF;			//Information for the different simulations
	struct GridaeParameter Gridae;
	// These are the parameters for the multi simulation run mode //
	double *n1_h, *n1_d;			//factor for Hill size
	double *n2_h, *n2_d;			//factor for velocity
	int *N_h, *N_d;				//number of bodies
	int *Nsmall_h, *Nsmall_d;		//number of test particles
	double *Msun_h, *Msun_d;		//Mass of the star
	double *dtiMsun_h, *dtiMsun_d;		//dt/Msun

	double dt;
	double dtksq;

	//Total sizes
	int NT;
	int NsmallT;
	int NB2T;
	int Nsmall2T;
	int NEnergyT;

	int *NsmallS_h;
	int *NB2S;
	int *Nsmall2S;
	int *NEnergy;

	int *NBS_h, *NBS_d;
	
	__host__ Host();
	__host__ int NSimulations(int, char*argv[]);
	__host__ int DeviceInfo();
	__host__ void Halloc();
	__host__ int Param(int , char*argv[]);
	__host__ int size();
	__host__ void Calloc();
	__host__ void Info();
	__host__ void Tsizes();
	__host__ int freeHost();
private:
	__host__ int readparam(FILE *, int , int , char*argv[]);
	__host__ int icSize(int);

};
#endif

#include <stdio.h>
#include <stdlib.h>
#include "define.h"

extern FILE *masterfile;
extern char masterfilename[64];
extern FILE *pathfile;				//used in multisim mode, contains list of directories
extern char pathfilename[64];

extern int Nst;					//Number of simulations

extern int devCount;
extern int runtimeVersion;
extern int driverVersion;

extern int *NB;					//number of bodies increased to integer block size
extern int *icNB;				//initial number of NB
extern int *N4;					//number of bodies divided by 4
extern int *N2;					//number of bodies divided by 2
extern int *Nconst;				//number of massive bodies in test particle run
extern int *Nmin;				//minimal number of bodies
extern double *rho;				//default density of bodies


extern struct Parameter P;			//parameters, for all Simulations the same
extern struct GSFiles *GSF;			//Information for the different simulations
extern struct GridaeParameter Gridae;
// These are the parameters for the multi simulation run mode //
extern double *n1_h, *n1_d;			//factor for Hill size
extern double *n2_h, *n2_d;			//factor for velocity
extern int *N_h, *N_d;				//number of bodies
extern int *Nsmall_h, *Nsmall_d;		//number of test particles
extern double *Msun_h, *Msun_d;			//Mass of the star
extern double *dtiMsun_h, *dtiMsun_d;		//dt/Msun

extern double dt;
extern double dtksq;

//Total sizes
extern int NT;
extern int NsmallT;
extern int NB2T;
extern int Nsmall2T;
extern int NEnergyT;

extern int *NsmallS_h;
extern int *NB2S;
extern int *Nsmall2S;
extern int *NEnergy;

extern int *NBS_h, *NBS_d;



extern __host__ void Hostinit();
extern __host__ int NSimulations(int, char*argv[]);
extern __host__ int DeviceInfo();
extern __host__ void Halloc();
extern __host__ int readparam(FILE *, int , int , char*argv[]);
extern __host__ int Param(int , char*argv[]);
extern __host__ int icSize(int);
extern __host__ int size();
extern __host__ void Calloc();
extern __host__ void Info();
extern __host__ void Tsizes();
extern __host__ int freeHost();

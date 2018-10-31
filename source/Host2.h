#include "define.h"

#ifndef HOST_CLASS
#define HOST_CLASS
// ***********************************
// Authors: Simon Grimm, Joachim Stadel
// March 2014
//
// ************************************
class Host{

public:
	int Lock;
	FILE *masterfile;
	char masterfilename[64];
	FILE *pathfile;				//used in multisim mode, contains list of directories
	char pathfilename[64];

	int Nst;				//Number of simulations
	int NconstT;
	int MTFlag;
	int ForceFlag;
	int devCount;
	int runtimeVersion;
	int driverVersion;

	int MultiSim;
	int interrupt;				//signal handling

	int *NB;				//number of bodies increased to integer block size
	int *N4;				//number of bodies divided by 4
	int *N2;				//number of bodies divided by 2
	int *Nmin;				//minimal number of bodies
	double *rho;				//default density of bodies
	long long *delta_h, *delta_d;		//number of timesteps
	int MaxIndex;				//highest index of all bodies and test particles

	struct Parameter P;			//parameters, for all Simulations the same
	struct GSFiles *GSF;			//Information for the different simulations
	struct GridaeParameter Gridae;
	// These are the parameters for the multi simulation run mode //
	double *n1_h, *n1_d;			//factor for Hill size
	double *n2_h, *n2_d;			//factor for velocity
	int *N_h, *N_d;				//number of bodies
	int *Nsmall_h, *Nsmall_d;		//number of test particles
	double4 *Msun_h, *Msun_d;		//Mass of the star, Radius, Love Number, fluid Love Number
	double4 *Spinsun_h, *Spinsun_d;		//Spin of the star x, y, z and time lag
	double *idt_h, *idt_d;			//initial time step 
	double *ict_h, *ict_d;			//initial time
	double *Rcut_h, *Rcut_d;		//inner truncation radius
	double *RcutSun_h, *RcutSun_d;		//outer truncation radius
	double *time_h, *time_d;
	double *dt_h, *dt_d;			//time step in code units
	//Total sizes
	int NT;
	int Nstart;
	int NsmallT;
	long long int NB2T;
	int NBNencT;
	int NEnergyT;

	int *NsmallS_h;
	int *NEnergy;

	int *NBS_h, *NBS_d;			//starting point in memory of individual simulations

	//data for irregular outputs
	int bufferCount;
	int bufferCountIrr;
	int irrTimeStep;
	int irrTimeStepOut;
	int NIrrOutputs;
	double *IrrOutputs;
	
	//Transits Data
	int2 *NtransitsT_h, *NtransitsT_d;			//Total number of computed transits per planet, old number
	int *NtransitsTObs_h, *NtransitsTObs_d;			//Total number of observed transits per planet
	double2 *TransitTimeObs_h, *TransitTimeObs_d;		//contains all observed transit times
	double *TransitTime_h, *TransitTime_d;			//contains all computed transit times
	__host__ int readTransits();

	double *setElementsData_h, *setElementsData_d;
	int *setElementsLine_d;
	double4 *GasData_h, *GasData_d;
	int GasDatanr;
	double2 GasDatatime;
	
	__host__ Host(long long);
	__host__ int NSimulations(int, char*argv[]);
	__host__ int DeviceInfo();
	__host__ void Halloc();
	__host__ int Param(int , char*argv[]);
	__host__ int size();
	__host__ void Calloc();
	__host__ void Info();
	__host__ void Tsizes();
	__host__ int readIrregularOutputs();
	__host__ int readSetElements();
	__host__ int readGasFile();
	__host__ int readGasFile2(double);
	__host__ int freeHost();

	//force
	__host__ void constantCopy3(int *, int, int, int, int);

private:
	__host__ int readparam(FILE *, int , int , char*argv[]);
	__host__ int icSize(int);
	__host__ int icict(int, int);
};
#endif

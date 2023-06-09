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

	char fileFormat[def_Ninformat][5];

	int Nst;				//Number of simulations
	int NconstT;
	int MTFlag;
	int ForceFlag;
	int devCount;
	int runtimeVersion;
	int driverVersion;
	int WarpSize;
	int Nomp;				//Number of openMP threads

	int MultiSim;
	int interrupt;				//signal handling

	int *NB;				//number of bodies increased to integer block size
	int *NBT;				//number of bodies + number of test particles, increased to integer block size
	int NBmax;				//maximum of NB[i]

	int2 *Nmin;				//minimal number of bodies
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
	double2 *Msun_h, *Msun_d;		//Mass of the star, Radius, Love Number, fluid Love Number
	double4 *Spinsun_h, *Spinsun_d;		//Spin of the star x, y, z and Ic
	double3 *Lovesun_h, *Lovesun_d;		//Love number of the star, fluid Love number, time lag
	double2 *J2_h, *J2_d;			//J2 of multipole gravity, mean radius of particle distribution
	
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
	int NBNencT;
	int NEnergyT;

	int *NsmallS_h;
	int *NEnergy;

	int *NBS_h, *NBS_d;			//starting point in memory of individual simulations


	//kernel tuning parameters
	//FG kernel
	int FTX;
	//Rcrit kernel
	int RTX;
	//kick kernel
	int KP;
	int KTX;
	int KTY;
	int UseAcc;
	//kick TP 2
	int KP2;
	int KTX2;
	int KTY2;
	//Force
	int FrTX;
	//BVH
	int UseBVH;
	//Multi simulation mode
	int KTM3;
	int HCTM3;
	int UseM3;

	//data for irregular outputs
	int bufferCount;
	int bufferCountIrr;
	long long int irrTimeStep;
	long long int irrTimeStepOut;
	long long int NIrrOutputs;
	double *IrrOutputs;
	
	//Transits Data
	int doTransits;						//1: do transits in step(), 0: don't do transits in step
	int2 *NtransitsT_h, *NtransitsT_d;			//Total number of computed transits per planet, old number
	int *NtransitsTObs_h, *NtransitsTObs_d;			//Total number of observed transits per planet
	double2 *TransitTimeObs_h, *TransitTimeObs_d;		//contains all observed transit times and uncertainties
	double *TransitTime_h, *TransitTime_d;			//contains all computed transit times
	__host__ int readTransits();
	__host__ int readRV();

	//RV Data
	int2 *NRVT_h, *NRVT_d;					//Total number of computed RV data, old number
	int *NRVTObs_h, *NRVTObs_d;				//Total number of observed RV data
	double2 *RV_h, *RV_d;					//contains all times and computed RV data
	double3 *RVObs_h, *RVObs_d;				//contains all times and observed RV data and uncertainties
	int RVTimeStep;
	double *RVP_d;						//RV probability

	double *setElementsData_h, *setElementsData_d;
	int *setElementsLine_d;
	double4 *GasData_h, *GasData_d;
	int GasDatanr;
	double2 GasDatatime;


#if def_CPU == 1
	int *setElementsLine_h;
#endif	
	__host__ Host(long long);
	__host__ int NSimulations(int, char*argv[]);
	__host__ int DeviceInfo();
	__host__ void Halloc();
	__host__ int assignInformat(char *, int &, int);
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
	__host__ void constantCopy3(int *, int, int, int);

	__host__ int readOutLine(double &, int &, double4 &, double4 &, double4 &, double3 &, double3 &, float4 &, double &, double &, unsigned long long &, double &, double &, FILE *, int);

private:
	__host__ int readparam(FILE *, int , int , char*argv[]);
	__host__ int icSize(int);
	__host__ int icict(int);
};
#endif

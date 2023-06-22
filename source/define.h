#ifndef M_PI
#define _USE_MATH_DEFINES  //for Windows
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define def_Version 3.179

#define def_OldShuffle 0 		//set this to 1 when an old cuda version is used which doesn't have shfl_sync operations


//Default parameter values
// ---------------------------------
//The following block defines default values for the parameters from the param.dat file
#define def_TimeStep 6
#define def_Name "test"
#define def_EnergyOutputInterval 100
#define def_CoordinatesOutputInterval 100
#define def_OutputsPerInterval 1
#define def_Buffer 1
#define def_IntegrationSteps 1000
#define def_CentralMass 1.0
#define def_CentralRadius 0.00465475877
#define def_StarK2 1.0
#define def_StarK2f 1.0
#define def_StarTau 0.0
#define def_StarSpinx 0.0
#define def_StarSpiny 0.0
#define def_StarSpinz 0.0
#define def_StarIc 0.4				//Moment of inertia
#define def_J2 0.0
#define def_J2R 0.0
#define def_SolarConstant 1367.0		//Solar Constant at 1 AU in W /m^2
#define def_n1 3.0
#define def_n2 0.4
#define def_InputFile "inital.dat"
#define def_InputFileFormat "<< t i m r x y z vx vy vz >>"
#define def_OutputFileFormat "<< t i m r x y z vx vy vz Sx Sy Sz amin amax emin emax aec aecT encc test >>"
#define def_OutBinary 0				//0: text files, 1: binary files
#define def_AngleUnits 0			//0: radians, 1:degrees
#define def_rho 2.0
#define def_UseTestParticles 0			//0 or 1
#define def_MinMass 0.0			//Minimal mass for massive particles in Test Particle mode, lighter particles are treated as test particles		
#define def_RestartTimeStep 0
#define def_MinimumNumberOfBodies 0
#define def_MinimumNumberOfTestParticles 0
#define def_RcutSun 0.2
#define def_Rcut 50.0
#define def_NencMax 512
#define def_Nfragments 0		//Additional array size for debris particles
#define def_OderOfIntegrator 2			//2, 4  or 6
#define def_UseaeGrid 0				// 1 or 0
#define def_aeGridamin 0.0f
#define def_aeGridamax 5.0f
#define def_aeGridemin 0.0f
#define def_aeGridemax 1.0f
#define def_aeGridimin 0.0f
#define def_aeGridimax 0.1f
#define def_aeGridNa 10
#define def_aeGridNe 10
#define def_aeGridNi 10
#define def_aeGridStartCount 0
#define def_aeGridName "A"
#define def_Usegas 0			//Gas Grid. See Morishima, Stadel and Moore 2010 for more details
#define def_UsegasEnhance 2		//Gas Grid. See Morishima, Stadel and Moore 2010 for more details
#define def_UsegasPotential 1		//Gas Grid. See Morishima, Stadel and Moore 2010 for more details
#define def_UsegasDrag 2		//Gas Grid. See Morishima, Stadel and Moore 2010 for more details
#define def_UsegasTidalDamping 2	//Gas Grid. See Morishima, Stadel and Moore 2010 for more details
#define def_GasdTau_diss 10000
#define def_GasRg0 0.1			//Gas disk inner edge
#define def_GasRg1 35.0			//Gas disk outer edge
#define def_GasRp1 15.0			//Gas disk grid outer edge
#define def_GasDrg 0.1			//Gas disk grid spacing
#define def_GasAlpha 1
#define def_GasBeta 0.25
#define def_G_Sigma_10 2000		 //surface density at 1AU
#define def_Mgiant  1.0E-4
#define def_UseForce 0			//Use additional forces, which can be specified in the file force.h
#define def_UseGR 0			//Flag for GR, 1:Hamiltonian splitting, 2: implicit midpoint, 3:direct force
#define def_UseTides 0			//Flag for tidal force
#define def_UseRotationalDeformation 0	//Flag for Rotational deformation
#define def_UseYarkovsky 0		//Flag for Yarkovsky effect
#define def_UsePR 0			//Flag for Poynting-Robertson effect
#define def_UseMigrationForce 0		//Flag for artifitial migration force
#define def_Qpr 1.0			//radiation pressure coefficient, 1 pure absortion
#define def_SolarWind 0.0		//ratio of solar wind drag to Poynting-Robertson drag
#define def_Asteroid_eps 0.95		//Emissivity
#define def_Asteroid_rho 3500.0		//density of body in kg/m^3		Hebe 3500,	Veritas 2250
#define def_Asteroid_C 680.0		//Specific Heat Capacity in J/kgK 	Hebe 680,	Veritas 500
#define def_Asteroid_A 0.2		//Bond albedo				Hebe 0.2,	Veritas 0.069
#define def_Asteroid_K 2.65		//Thermal conductivity in W/mK		Hebe 2.65,	Veritas 1.0
#define def_Asteroid_V 5000.0		//Collisional velocity in m/s 
#define def_Asteroid_rmin 0.01		//minimal radius of new generated particles in m 
#define def_Asteroid_rdel 0.01		//remove limit for new generated particles in n
#define def_UseSmallCollisions 0	//fragmentation and rotation reset model
#define def_CreateParticles 0		//flag for particle creation mode
#define def_CreateParticlesN 0		//Maximum number of particles to be created
#define def_FormatS 0			//0: one file per simulation, 1: all simulations in the same file
#define def_FormatT 0			//0: one file per time step, 1: all time steps in the same file
#define def_FormatP 1			//0: one file per particle, 1: all particles in the same file
#define def_FormatO 0			//output numbering, 0: time steps or 1: output steps
#define def_WriteEncounters 0		//Write all close encounters to a file
#define def_WriteEncountersRadius 3.0	//factor in terms fo physical radii
#define def_WriteEncountersCloudSize 1	//Index range of particles belonging to a cloud of particles
#define def_StopAtEncounter 0		//Stop simulations at close encounters
#define def_StopAtEncounterRadius 1.0	//factor in terms of Hill radii
#define def_StopAtCollision 0		//1 Stop Simulation when a Collision occurs, 0 continue simulation with merged bodies (default)
#define def_StopMinMass 0.0		//when def_StopAtCollision = 1, then stop simulations only when both bodies are more massive than def_StopMinMass
#define def_CollisionPrecision 1.0e-4	//Tolerance for collision time precision. In units of physical radius fraction.
#define def_CollTshift 1.0		//Collision output before Collision happens, default is 1.0
#define def_CollisionModel 0		//0 is perfect merger, other models can be added in directAcc.h
#define def_SLevels 1			//Number of recursive symplectic sub step levels
#define def_SLSteps 2			//number of time steps per level
#define def_SERIAL_GROUPING 0
#define def_doTuning 1			//Flag to enable kernel tunings
#define def_doSLTuning -1		//Flag to enable symplectic levels
#define def_KickFloat 0			//1: Do kick operation in large N runs in single precision, 0 use double precision
// End default parameters
// --------------------------------


//The following parameters can not be set in the param.dat file
#define def_NAFvars 1
#define def_NAFn0 10
#define def_NAFnfreqs 1
#define def_NAFformat 1
#define def_NAFinterval 1
#define def_Ninformat 55		//number of entries in informat array
#define def_NColl 25			//number of parameters in Coll and writeEnc arrays
#define def_BufferSize 29		//number of parameters in the Buffer arrays

#define def_pc 3.0			//Factor in Prechecker, Pairs with rij^2 < pc * rcrit^2 are considered as close encounter candidates
#define def_pcf 3.0f			//float version of def_pc
#define def_MaxColl 120			//Maximum number of Collisions per time step, needed for memory allocation
#define def_MaxWriteEnc 128		//Maximum number of Encounter per time step which can be written to file
#define def_cef 1.0 			//Close encounter factor, pairs with rij^2 < f * rcrit^2 are considered as close encounter pairs.
#define def_tol 1.0e-12			//Tolerance in Bulirsh Stoer
#define def_dtmin 1.0e-17		//minimal time step in Bulirsh Stoer 
#define def_NFileNameDigits 12		//number of digits in output filenames
#define def_NSetElementsMax 10000000	//maximum number of lines in the set-elements file


//print Poincare Section of surface, in this mode the code can be very slow
#define def_poincareFlag 0			//1: print, 0: no print

//ignore the lock file and start GENGA anyway
#define def_IgnoreLockFile 1

#define def_SLevelsMax 3

#define def_SLn1 0			//1: apply substeps refinement also to n1 terms

//gas disk constants 
// See Morishima, Stadel and Moore 2010 for more details
#define def_Gasnz_g 50
#define def_Gasnz_p 51
#define def_h_1 0.03358			//scale height at 1AU for c = 1km/s
#define def_M_Enhance 5.98/1.98*1.E-8	// 1% of the Earth's mass
#define def_Mass_pl  0.502E-14		// corresponding to 10^19g
#define def_fMass_min 7.55E-9
#define def_Gas_cd 2			//numerical gas drag coefficient

#define def_MgasSmall 1.0e-14  //minimal mass that is taken for test particles

//Units and constants
#define def_ksq 1.0			//Squared Gaussian gravitational constant in current units
#define def_Kg 2.959122082855911e-4	//Squared Gaussian gravitational constant in AU^3 day^-2 M_Sun^-1 , used for conversion
#define dayUnit 0.01720209895
//#define dayUnit 0.01720412578565452474399499749324604636058
#define def_AU 149597870700.0		//AU in m
#define def_Solarmass 1.98855e30	//Solar Mass in kg
#define def_c 299792458.0		//speed of light in m/s
#define def_cm 10065.3201686		//speed of light in AU / day * 0.0172020989	
#define def_sigma 5.670373e-8		//Stefan Boltzmann constant J m^-2 s^-1 K^-4

//Block Sizes for multi simulation run
#define HCM_Bl 128
#define NmaxM 32			//maximal size of sub simulations
//#define NmaxM 4				//maximal size of sub simulations
#define HCM_Bl2 (HCM_Bl - NmaxM - NmaxM / 2)

#define KM_Bl 128
#define KM_Bl2 (KM_Bl - NmaxM)

#define def_MaxIndex 100			//this is the maximum id for the multi simulation mode


//Maximum close encounter group size  = 2^(def_GMax)
#define def_GMax 20



//Parameters for fastfg
#define def_FGN 127				//Number of elements in table for fastfg
#define PI_N M_PI/def_FGN
#define N_PI def_FGN/M_PI

//Build Data
#ifndef GIT_BRANCH
#define GIT_BRANCH "Undefined"
#endif

#ifndef GIT_COMMIT
#define GIT_COMMIT "Undefined"
#endif

#ifndef BUILD_DATE
#define BUILD_DATE "Undefined"
#endif

#ifndef BUILD_SYSTEM
#define BUILD_SYSTEM "Undefined"
#endif

#ifndef BUILD_PATH
#define BUILD_PATH "Undefined"
#endif

#ifndef BUILD_SM
#define BUILD_SM "Undefined"
#endif

// * Only for testing **
#define def_G3 0				//New integrator scheme
#define def_G3Limit	1.5e-12	//2.0e-12
#define def_G3Limit2 2.0e-16 //2.0e-16
// *********************


//only here for testing
#define USE_NAF 0

//------------------------------
//only for TTV or RV sampling
#define def_TTV 0			//1: to transit detection and MCMC sampling
#define def_NtransitMax 6000		//only used in def_TTV 1
#define def_NtransitTimeMax 6000	//Maximum number of transit times per object
//#define def_NtransitMax 20000
//#define def_NtransitTimeMax 2000	//Maximum number of transit times per object
#define def_TransitTol 1.0e-12

#define def_RV 0			
#define def_NRVMax 6000			//Maximum number of RV data
#define MCMC_BLOCK 4			
					//4 DEMCMC
					//5 RMSPROP
					//6 Nelder Mead
					//7 lbfgs

#define MCMC_Q 0			//1 quadratic estimator
					//2 iterative adjustment of P
#define MCMC_NQ 1

#define MCMC_NE 5 			//2: a M; 3: a M m; 5: a M m e w; 7: a m M e w inc Omega,; 8: + r
#define MCMC_NT 1			//number of temperature levels in parallel tempering
#define MCMC_NCOV 0			//number of parameters per planet in covariance matrix
#define def_NoEncounters 0			
//----------------------------


#define USE_RANDOM 1			//USE_RANDOM 1 is needed for def_TTV and for 'Use Small Collisions' > 0

#if USE_RANDOM == 1
  #include <curand_kernel.h>
#else
  #define curandState int
#endif


#define def_CPU 0
#if def_CPU == 1

 #ifndef CPU_H
 #define CPU_H
/*
	#define cudaError_t int

	int cudaGetLastError(){
		return 0;
	}
	char cudaGetErrorString(int error){
		return '';
	}

	struct cudaDeviceProp{
		char name[256];
		int major;
		int minor;
		int warpSize;

	};
	struct double3{
		double x;
		double y;
		double z;
	};
	struct double4{
		double x;
		double y;
		double z;
		double w;
	};
	struct float4{
		float x;
		float y;
		float z;
		float w;
	};

	void cudaGetDeviceProperties(cudaDeviceProp &devProp, int device){
		sprintf(devProp.name, "%s", "cpu");
		devProp.major = 0;
		devProp.minor = 0;
		warpSize = 1;
	}
*/
/*
	inline double __dmul_rn_cpu(double x, double y){
		return (x * y);
	}
	inline float __fmul_rn_cpu(float x, float y){
		return (x * y);
	}
	inline double __fma_rn_cpu(double x, double y, double z){
		return (x * y + z);
	}
*/
 #endif
#endif


#if def_OldShuffle == 1
 #ifndef OldShuffle_H
 #define OldShuffle_H
//Use this for older CUDA version where shfl_xor is not available in double precision
__device__ inline
double __shfld_xor(double x, int k) {
	int2 a = *reinterpret_cast<int2*>(&x);
	a.x = __shfl_xor(a.x, k);
	a.y = __shfl_xor(a.y, k);
	return *reinterpret_cast<double*>(&a);
}

__device__ inline
double __shfld_up(double x, int k) {
	int2 a = *reinterpret_cast<int2*>(&x);
	a.x = __shfl_up(a.x, k);
	a.y = __shfl_up(a.y, k);
	return *reinterpret_cast<double*>(&a);
}
__device__ inline
double __shfld_down(double x, int k) {
	int2 a = *reinterpret_cast<int2*>(&x);
	a.x = __shfl_down(a.x, k);
	a.y = __shfl_down(a.y, k);
	return *reinterpret_cast<double*>(&a);
}
 #endif
#endif

#ifndef STRUCT_H
#define STRUCT_H

__constant__ double Rcut_c[1];
__constant__ double RcutSun_c[1];
__constant__ int StopAtCollision_c[1];
__constant__ double StopMinMass_c[1];
__constant__ double CollisionPrecision_c[1]; 
__constant__ double CollTshift_c[1]; 
__constant__ int CollisionModel_c[1];
__constant__ int2 CollTshiftpairs_c[1]; 
__constant__ int WriteEncounters_c[1]; 
__constant__ double WriteEncountersRadius_c[1]; 
__constant__ int WriteEncountersCloudSize_c[1]; 
__constant__ int StopAtEncounter_c[1]; 
__constant__ double StopAtEncounterRadius_c[1]; 
__constant__ double Asteroid_eps_c[1];
__constant__ double Asteroid_rho_c[1];
__constant__ double Asteroid_C_c[1];
__constant__ double Asteroid_A_c[1];
__constant__ double Asteroid_K_c[1];
__constant__ double Asteroid_V_c[1];
__constant__ double Asteroid_rmin_c[1];
__constant__ double Asteroid_rdel_c[1];
__constant__ double SolarConstant_c[1];
__constant__ double Qpr_c[1];
__constant__ double SolarWind_c[1];

__constant__ double BSddt_c[8];		//time stepping factors in Bulirsch-Stoer method
__constant__ double BSt0_c[8 * 8];

__constant__ double CreateParticlesParameters_c[12];




struct Parameter{
	int dev[32];			//device number array 
	int ndev;			//number of devices
	int ei;				//Energy output interval
	int ci;				//Coordinate output interval
	int nci;			//Number of outputs per interval
	int UseTestParticles;
	long long tRestart;		//timestep for restart
	long long deltaT;		//Number of time steps to do
	int SIO;
	double G_dTau_diss;		//Dissipation time for Gas Disc
	double G_rg0;			//gas disk inner edge
	double G_rg1;			//gas disk outer edge
	double G_rp1;			//gas disk grid outer edge
	double G_drg;			//gas disk grid spacing
	double G_alpha;			//alpha parameter for Gas Disc
	double G_beta;			//beta parameter for Gas Disc
	double G_Sigma_10;		//Gas Sigma_10
	double G_Mgiant;		//Mass limit for gas effects
	int UseaeGrid;			
	int FormatS;			//Output file structure
	int FormatT;			
	int FormatP;
	int FormatO;			//output numbering, 0: time steps or 1: output steps
	int Buffer;
	int Usegas;
	int UsegasPotential;
	int UsegasEnhance;
	int UsegasDrag;
	int UsegasTidalDamping;
	int UseForce;
	int UseGR;
	int UseTides;
	int UseRotationalDeformation;
	int UseJ2;
	int UseYarkovsky;
	int UseMigrationForce;
	int UseSmallCollisions;		//fragmentation and rotation reset model
	int CreateParticles;
	int CreateParticlesN;
	char CreateParticlesfilename[128];
	int UsePR;			//Poynting Robertson drag
	double Qpr;			//radiation pressure coefficient
	double SolarWind;		//ratio of solar wind drag to Poynting-Robertson drag
	double SolarConstant;
	double Asteroid_eps;
	double Asteroid_rho;
	double Asteroid_C;
	double Asteroid_A;
	double Asteroid_K;
	double Asteroid_V;
	double Asteroid_rmin;
	double Asteroid_rdel;
	int IrregularOutputs;
	char IrregularOutputsfilename[128];
	int UseTransits;
	int UseRV;
	char Transitsfilename[128];
	char RVfilename[128];
	int TransitSteps;
	int PrintTransits;
	int PrintRV;
	int PrintMCMC;
	int mcmcNE;
	int mcmcRestart;
	int setElements;
	int setElementsV;
	char setElementsfilename[128];
	int setElementsN;
	char Gasfilename[128];
	int WriteEncounters;
	double WriteEncountersRadius;
	int WriteEncountersCloudSize;
	int StopAtEncounter;
	double StopAtEncounterRadius;
	int StopAtCollision;
	double StopMinMass;
	double CollisionPrecision;
	double CollTshift;
	int CollisionModel;
	int NAFvars;
	int NAFn0;
	int NAFnfreqs;
	int NAFformat;
	int NAFinterval;
	int NencMax;
	int Nfragments;
	double MinMass;
	int OutBinary;
	int AngleUnits;
	int SLevels;
	int SLSteps;
	int SERIAL_GROUPING;
	int doTuning;
	int doSLTuning;
	int KickFloat;
};

struct elements{
	double P;
	double T;
	double m;
	double e;
	double w;
	double f;
};
struct elements8{
	double P;
	double T;
	double m;
	double e;
	double w;
	double inc;
	double O;
	double r;
};
struct elements10{
	double P;
	double T;
	double m;
	double e;
	double w;
	double inc;
	double O;
	double r;
	double a;
	double M;
};

struct elementsS{
	double pP;	//direction
	double pT;
	double pm;
	double pe;
	double pw;
	double pf;
	double gP;	//old derivatives
	double gT;
	double gm;
	double ge;
	double gw;
	double gf;
	double P0;	//old derivatives
	double T0;
	double m0;
	double e0;
	double w0;
	double f0;
	double alpha;	//step length
	int count;
};

#define MCMC_NH 30
struct elementsH{	//History
	double sP;
	double sT;
	double sm;
	double se;
	double sw;
	double yP;
	double yT;
	double ym;
	double ye;
	double yw;
};


//File names of Simulations
struct GSFiles{
	char X[128];
	char path[128];
	
	//The following file names must have a lenght of size(X) + size(path) + name size
	FILE *outputfile, *logfile;
	char outputfilename[384];
	char inputfilename[384];
	char Originputfilename[384];
	char Energyfilename[384];
	char EnergyIrrfilename[384];
	char logfilename[384];
	char timefilename[384];
	char collisionfilename[384];
	char collisionTshiftfilename[384];
	char ejectfilename[384];
	char encounterfilename[384];
	char fragmentfilename[384];
	char starfilename[384];
	char starIrrfilename[384];
	int informat[def_Ninformat];
	int outformat[def_Ninformat];
};

//Parameters for the ae grid, for all Simulations the same
struct GridaeParameter{
	float amin;
	float amax;
	float emin;
	float emax;
	float imin;
	float imax;
	float deltaa;
	float deltae;
	float deltai;
	int Na;
	int Ne;
	int Ni;
	long long Start;
	char X[64];
	FILE *file;
	char filename[128];
};

struct Node{
        Node *childL;
        Node *childR;
        Node *parent;

        int rangeL;
        int rangeR;
        int counter;

        bool isLeaf;
        unsigned int nodeID;

        //Axis Aligned Bounding Box, AABB
        float xmin;
        float xmax;
        float ymin;
        float ymax;
        float zmin;
        float zmax;
};


#endif

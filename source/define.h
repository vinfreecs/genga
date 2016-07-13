#ifndef M_PI
#define _USE_MATH_DEFINES  //for Windows
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define Version 3.48

//Default parameter values
#define def_TimeStep 6
#define def_Name "test"
#define def_EnergyOutputInterval 100
#define def_CoordinatesOutputInterval 100
#define def_OutputsPerInterval 1
#define def_Buffer 1
#define def_IntegrationSteps 1000
#define def_CentralMass 1.0
#define def_CentralRadius 0.00465475877
#define def_CentralK2 1.0
#define def_CentralK2f 1.0
#define def_n1 3.0
#define def_n2 0.4
#define def_InputFile "inital.dat"
#define def_InputFileFormat "<< t i m r x y z vx vy vz >>"
#define def_rho 2.0
#define def_UseTestParticles 0			//0 or 1
#define def_RestartTimeStep 0
#define def_MinimumNumberOfBodies 0
#define def_Rcut 50.0
#define def_RcutSun 0.2
#define def_OderOfIntegrator 2			//2, 4  or 6
#define def_UseaeGrid 0				// 1 or 0
#define def_aeGridamin 0.0
#define def_aeGridamax 5.0
#define def_aeGridemin 0.0
#define def_aeGridemax 1.0
#define def_aeGridimin 0.0
#define def_aeGridimax 0.1
#define def_aeGridNa 10
#define def_aeGridNe 10
#define def_aeGridNi 10
#define def_aeGridStartCount 0
#define def_aeGridName "A"
#define def_Usegas 0			//Gas Grid. See Morishima, Stadel and Moore 2010 for more details
#define def_UseForce 0			//Use additional forces, which can be specified in the file force.h
#define def_GasdTau_diss 10000
#define def_GasAlpha 1
#define def_FormatS 0			//0: one file per simulation, 1: all simulations in the same file
#define def_FormatT 0			//0: one file per time step, 1: all time steps in the same file
#define def_FormatP 1			//0: one file per particle, 1: all particles in the same file
#define def_WriteEncounters 0		//Write all close encounters to a file
#define def_WriteEncountersRadius 3
#define def_NAFvars 1
#define def_NAFn0 10
#define def_NAFnfreqs 1
#define def_NAFformat 1
#define def_NAFinterval 1
#define def_NencMax 512


#define def_pc 3.0			//Factor in Prechecker, Pairs with rij^2 < pc * rcrit^2 are considered as close encounter candidates
#define MaxColl 120			//Maximum number of Collisions per time step, needed for memory allocation
#define MaxWriteEnc 128			//Maximum number of Encounter per time step which can be written to file
#define def_cef 1.0 			//Close encounter factor, pairs with rij^2 < f * rcrit^2 are considered as close encounter pairs.
#define def_tol 1.0e-12			//Tolerance in Bulirsh Stoer 
#define def_Nfragments 100		//Additional array size for debris particles
#define def_MinMass 0.0			

//The serial grouping mode can be chosen to reproduce simulations exactly, but there is a performance penalty
#define SERIAL_GROUPING 0

//print Poincare Section of surface, in this mode the code can be very slow
#define poincareFlag 0			//1: print, 0: no print

//ignore the lock file and start GENGA anyway
#define IgnoreLockFile 1

//stop Simulation at close encounters
#define StopAtEncounter 0


//gas disk constants 
// See Morishima, Stadel and Moore 2010 for more details
#define Gasnr_g 189
#define Gasnz_g 50
#define Gasnr_p 150
#define Gasnz_p 51
#define h_1 0.03358 //scale height at 1AU for c = 1km/s*
#define Sigma_10 2000*1.49598*1.49598/1.98892*1.0e-7 //surface density at 1AU
#define Mgiant  1.0E-4
#define M_Enhance 5.98/1.98*1.E-8 /* 1% of the Earth's mass */
#define Mass_pl  0.502E-14 /* corresponding to 10^19g */
#define fMass_min 7.55E-9

#define MgasSmall 1.0e-14  //minimal mass that is taken for test particles

//Units
#define ksq 1.0				//Squared Gaussion gravitation constant in current units
#define def_Kg 2.959122082855911e-4	//Squared Gaussion gravitation constant, used for conversion
#define dayUnit 0.01720209895
#define def_AU 149597870700.0		//AU in m
#define def_Solarmass 1.98855e30       //solar mass in Kg
#define def_c 299792458.0  	       //speed of light im m/s

//Block Sizes for multi simulation run
#define HCM_Bl 128
#define NmaxM 16
#define HCM_Bl2 (HCM_Bl - NmaxM - NmaxM / 2)

#define KM_Bl 128
#define KM_Bl2 (KM_Bl - NmaxM)


//Maximum close encounter group size  = 2^(def_GMax)
#define def_GMax 20



//Parameters for fastfg
#define FGN 127				//Number of elements in table for fastfg
#define PI_N M_PI/FGN
#define N_PI FGN/M_PI

//Build Data
#ifndef HG_BRANCH
#define HG_BRANCH "Undefined"
#endif

#ifndef HG_COMMIT
#define HG_COMMIT "Undefined"
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
#define G3 0				//New integrator scheme
#define G3Limit	1.5e-12	//2.0e-12
#define G3Limit2 2.0e-16 //2.0e-16
// *********************

//only here for testing
#define USE_NAF 0


#define USE_RANDOM 1

#if USE_RANDOM
#include <curand_kernel.h>
#endif

//values for Asteroids in force function
//#define Asteroid_rho 3500.0	//density of body in kg/m^3	Hebe
#define Asteroid_rho 2250.0	//density of body in kg/m^3	Veritas

//#define Asteroid_C 680	//Specific Heat Capacity in J/kgK Hebe
#define Asteroid_C 500		//Specific Heat Capacity in J/kgK Veritas

//#define Asteroid_A 0.2	//Bond albedo			Hebe
#define Asteroid_A 0.069	//Bond albedo			Veritas
	

//#define Asteroid_K 2.65       //Thermal conductivity in W/mK	Hebe
//#define Asteroid_K 1.0         //Thermal conductivity in W/mK	Veritas
#define Asteroid_K 0.001         //Thermal conductivity in W/mK	Veritas


#ifndef STRUCT_H
#define STRUCT_H


struct Parameter{
	int dev;                        //Number of device
	int ei;                         //Energy output intervall
	int ci;                         //Coordinate output intervall
	int nci;                        //Number of outputs per intervall
	int UseTestParticles;
	long long tRestart;             //timestep for restart
	long long deltaT;               //Number of time steps to do
	int SIO;
	double G_dTau_diss;		//Dissipation time for Gas Disc
	int G_alpha;			//alpha parameter for Gas Disc
	int UseaeGrid;			
	int FormatS;			//Output file structure
	int FormatT;			
	int FormatP;
	int Buffer;
	int Usegas;
	int UseForce;
	int IrregularOutputs;
	char IrregularOutputsfilename[128];
	int setElements;
	char setElementsfilename[128];
	int WriteEncounters;
	double WriteEncountersRadius;
	int NAFvars;
	int NAFn0;
	int NAFnfreqs;
	int NAFformat;
	int NAFinterval;
	int NencMax;
};

//File names of Simulstions
struct GSFiles{
	FILE *outputfile, *Energyfile, *logfile, *timefile, *collisionfile, *ejectfile, *encounterfile, *fragmentfile;
	char outputfilename[128];
	char inputfilename[128];
	char Originputfilename[128];
	char Energyfilename[128];
	char logfilename[128];
	char timefilename[128];
	char collisionfilename[128];
	char ejectfilename[128];
	char encounterfilename[128];
	char fragmentfilename[128];
	char X[128];
	int informat[25];
	char path[128];
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
	char filename[64];
};

#endif

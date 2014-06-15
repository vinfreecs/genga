#include <stdio.h>
#include <stdlib.h>

#define Version 3.01

//output file structure
#define FormatS 0			//0: one file per simulation, 1: all simulations in the same file
#define FormatT 0			//0: one file per time step, 1: all time steps in the same file
#define FormatP 1			//0: one file per particle, 1: all particles in the same file

#define Rcut 50.0			//bodies with r > Rcut are Ejected
#define RcutSun 0.2			//bodies with r < RcutSun fall into the Sun
#define pc 3.0 				//Factor in Prechecker, Pairs with rij^2 < pc * rcrit^2 are considered as close encounter candidates
#define MaxColl 120			//Maximum number of Collisions per time step, needed for memory allocation
#define cef 1.0 			//Close encounter factor, pairs with rij^2 < f * rcrit^2 are considered as close encounter pairs.

//The serial grouping mode can be chosen to reproduce simulations exactly, but there is a performance penalty
#define SERIAL_GROUPING 0

//use aeGrid
#define useGridae 0			// 1 or 0

//print Poincare Section of surface, in this mode the code can be very slow
#define poincareFlag 0			//1: print, 0: no print

//ignore the lock file and start GENGA anyway
#define IgnoreLockFile 1

//Gas Grid. See Morishima, Stadel and Moore 2010 for more details
#define useGas 0			// 1 or 0
#define Gasnr_g 189
#define Gasnz_g 50
#define Gasnr_p 150
#define Gasnz_p 51
#define h_1 0.03358 //scale height at 1AU for c = 1km/s*
#define Sigma_10 2000*1.49598*1.49598/1.98892*1.0e-7 //surface density at 1AU
#define Mgiant  1.0E-4
#define M_Enhance 5.98/1.98*1.E-8 /* 1% of the Earth's mass */
#define Mass_pl  0.502E-14; /* corresponding to 10^19g */
#define fMass_min 7.55E-9

//Units
#define ksq 1.0				//Squared Gaussion gravitation constant in current units
#define Kg 2.959122082855911e-4         //Squared Gaussion gravitation constant, used for conversion
#define dayUnit 0.01720209895

//Block Sizes for multi simulation run
#define HCM_Bl 128
#define NmaxM 16
#define HCM_Bl2 (HCM_Bl - NmaxM - NmaxM / 2)

#define KM_Bl 128
#define KM_Bl2 (KM_Bl - NmaxM)

//Parameters for fastfg
#define FGN 127				//Number of elements in table for fastfg
#define PI_N M_PI/FGN
#define N_PI FGN/M_PI

#ifndef STRUCT_H
#define STRUCT_H

// * Only for testing **
#define G3 0				//New integrator scheme
#define G3Limit	2.0e-13
#define G3Limit2 2.0e-14 //1.0e-13
// *********************

struct Parameter{
	int dev;                        //Number of device
	int ei;                         //Energy output intervall
	int ci;                         //Coordinate output intervall
	int nci;                        //Number of outputs per intervall
	int UseTestParticles;
	long long tRestart;             //timestep for restart
	long long delta;                //Number of time steps to do
	int SIO;
	double idt;                     //modified time step
	double ict;                     //initial time
	double G_dTau_diss;		//Dissipation time for Gas Disc
	int G_alpha;			//alpha parameter for Gas Disc
};

//File names of Simulstions
struct GSFiles{
	FILE *outputfile, *Energyfile, *logfile, *timefile, *collisionfile, *ejectfile;
	char outputfilename[128];
	char inputfilename[128];
	char Energyfilename[128];
	char logfilename[128];
	char timefilename[128];
	char collisionfilename[128];
	char ejectfilename[128];
	char X[128];
	int informat[22];
	char path[128];
};

//Parameters for the ae grid, for all Simulations the same
struct GridaeParameter{
	float amin;
	float amax;
	float emin;
	float emax;
	float deltaa;
	float deltae;
	int Na;
	int Ne;
	long long Start;
	char X[64];
	FILE *file;
	char filename[64];
};

#endif

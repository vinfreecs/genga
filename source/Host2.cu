#include "Host2.h"
// ******************************
//Costructor for Host class
//Authors: Simon Grimm, Joachim Stadel
//April 2014
// *******************************
__host__ Host::Host(long long Restart){
	
	sprintf(masterfilename, "%s", "master.out");
	
	FILE *lockfile;
	char lockfilename[64];
	sprintf(lockfilename, "%s", "lock.dat");
	
	Lock = 0;
	
	if(Restart == 0LL){
		lockfile = fopen(lockfilename, "r");
		
		if(lockfile == NULL){
			lockfile = fopen(lockfilename, "w");
			fprintf(lockfile, "%d\n", 0);
			fclose(lockfile);
		}
		else{
			Lock = 1;
			fclose(lockfile);
		}
		
	}
	else{
		lockfile = fopen(lockfilename, "r");
		if(lockfile == NULL){
			lockfile = fopen(lockfilename, "w");
			fprintf(lockfile, "%lld\n", Restart);
			fclose(lockfile);
		}
		else{
			long long R;
			fscanf(lockfile, "%lld", &R);
			fclose(lockfile);
			
			if(R == Restart) Lock = 1;
			
			lockfile = fopen(lockfilename, "w");
			fprintf(lockfile, "%lld\n", Restart);
			fclose(lockfile);
			
		}
	}
	
	#if IgnoreLockFile == 1
	Lock = 0;
	#endif
	if(Lock == 0){
		if(Restart == 0LL){
			masterfile = fopen(masterfilename, "w");
		}
		else{
			masterfile = fopen(masterfilename, "a");
		}
	}
	else{
		masterfile = fopen(masterfilename, "a");
	}
	
	
	pathfilename[0] = 0; // = ""
	
	Nst = 1;
	devCount = 0;
	runtimeVersion = 0;
	driverVersion = 0;
	
	NT = 0;
	NsmallT = 0;
	NBNencT = 0;
	NEnergyT = 0;
}


// ************************************************
//This function determines the number of simulations by reading the pathfile specified in the -M console argument
//Authors: Simon Grimm
//January 2017
// ************************************************
__host__ int Host::NSimulations(int argc, char*argv[]){
	
	MTFlag = 0;
	for(int i = 1; i < argc; i += 2){
		if(strcmp(argv[i], "-M") == 0){
			char t[160];
			int er;
			int Np = 0;
			sprintf(pathfilename, "%s", argv[i + 1]);
			pathfile = fopen(pathfilename, "r");
			if(pathfile == NULL){
				printf("Error: pathfile %s doesn't exist!\n", pathfilename);
				fprintf(masterfile, "Error: pathfile %s doesn't exist!\n", pathfilename);
				return 0;
			}
			for(int i = 0; i < 1000000; ++i){
				er = fscanf(pathfile, "%s", t);
				if(er <= 0) break;
				++Np;
			}
			fclose(pathfile);
			if(Np > 1) Nst = Np;
		}
		if(strcmp(argv[i], "-MT") == 0){
			Nst = atoi(argv[i + 1]);
			MTFlag = 1;
			if(Nst <= 2){
				printf("Error, needed at least 3 chains for DEMCMC runs\n");
				return 0;
			}
		}
	}
	if(Nst <= 0){
		printf("Error: No Simulations!\n");
		fprintf(masterfile, "Error: No Simulations!\n");
		return 0;
	}
	
	return Nst;
}

// ************************************************
//Check Device Properties
//Authors: Simon Grimm, Joachim Stadel
//March 2014
// **********************************************
__host__ int Host::DeviceInfo(){
	
	cudaError_t error;
	error = cudaGetDeviceCount(&devCount);
	if(error > 0){
		printf("device error = %d = %s\n",error, cudaGetErrorString(error));
		return 0;
	}
	if(devCount == 0){
		fprintf(masterfile, "Error: No valid cuda device!\n");
		printf("Error: No valid cuda device!\n");
		return 0;
	}
	
	error = cudaGetLastError();
	fprintf(masterfile,"initial error = %d = %s\n",error, cudaGetErrorString(error));
	if(error > 0){
		printf("initial error = %d = %s\n",error, cudaGetErrorString(error));
		return 0;
	}
	
	cudaSetDeviceFlags(cudaDeviceMapHost);
	error = cudaGetLastError();
	fprintf(masterfile,"set Flags error = %d = %s\n",error, cudaGetErrorString(error));
	if(error > 0){
		printf("set Flags error = %d = %s\n",error, cudaGetErrorString(error));	
		return 0;
	}
	
	cudaDeviceProp devProp;
	
	cudaRuntimeGetVersion(&runtimeVersion);
	cudaDriverGetVersion(&driverVersion);
	
	fprintf(masterfile, "There are %d CUDA devices.\n", devCount);
	fprintf(masterfile, "Runtime Version: %d\n", runtimeVersion);
	fprintf(masterfile, "Driver Version: %d\n", driverVersion);
	
	for(int i = 0; i < devCount; ++i){
		cudaGetDeviceProperties(&devProp, i);
		fprintf(masterfile,"Name:%s, Major:%d, Minor:%d, Max threads per Block:%d, Max x dim:%d, #Multiprocessors:%d, Can Map Memory:%d, Clock Rate:%d, Memory Clock Rate:%d, Can Overlap:%d, Concurrent Kernels:%d, regsPerBlock:%d, sharedMemPerBlock:%lu\n",  
			devProp.name, devProp.major, devProp.minor, devProp.maxThreadsPerBlock, devProp.maxThreadsDim[0], devProp.multiProcessorCount, devProp.canMapHostMemory,devProp.clockRate, devProp.memoryClockRate, devProp.deviceOverlap, devProp.concurrentKernels, devProp.regsPerBlock, devProp.sharedMemPerBlock);
		if(!devProp.canMapHostMemory) {
			fprintf(masterfile, "Device %d cannot map host memory!\n", i);
			return 0;
		}
	}
	return 1;
}



__host__ int assignInformat(char *ff, int &format){
	int cartesian = 0;
	int keplerian = 0;

	if(strcmp(ff, "x") == 0){
		format = 1;
		cartesian = 1;
	}
	else if(strcmp(ff, "y") == 0){
		format = 2;
		cartesian = 1;
	}
	else if(strcmp(ff, "z") == 0){
		format = 3;
		cartesian = 1;
	}
	else if(strcmp(ff, "m") == 0){
		format = 4;
	}
	else if(strcmp(ff, "vx") == 0){
		format = 5;
		cartesian = 1;
	}
	else if(strcmp(ff, "vy") == 0){
		format = 6;
		cartesian = 1;
	}
	else if(strcmp(ff, "vz") == 0){
		format = 7;
		cartesian = 1;
	}
	else if(strcmp(ff, "r") == 0){
		format = 8;
	}
	else if(strcmp(ff, "rho") == 0){
		format = 9;
	}
	else if(strcmp(ff, "Sx") == 0){
		format = 10;
	}
	else if(strcmp(ff, "Sy") == 0){
		format = 11;
	}
	else if(strcmp(ff, "Sz") == 0){
		format = 12;
	}
	else if(strcmp(ff, "i") == 0){
		format = 13;
	}
	else if(strcmp(ff, "-") == 0){
		format = 14;
	}
	else if(strcmp(ff, "amin") == 0){
		format = 15;
	}
	else if(strcmp(ff, "amax") == 0){
		format = 16;
	}
	else if(strcmp(ff, "emin") == 0){
		format = 17;
	}
	else if(strcmp(ff, "emax") == 0){
		format = 18;
	}
	else if(strcmp(ff, "t") == 0){
		format = 19;
	}
	else if(strcmp(ff, "k2") == 0){
		format = 20;
	}
	else if(strcmp(ff, "k2f") == 0){
		format = 21;
	}
	else if(strcmp(ff, "tau") == 0){
		format = 22;
	}
	else if(strcmp(ff, "a") == 0){
		format = 23;
		keplerian = 1;
	}
	else if(strcmp(ff, "e") == 0){
		format = 24;
		keplerian = 1;
	}
	else if(strcmp(ff, "inc") == 0){
		format = 25;
		keplerian = 1;
	}
	else if(strcmp(ff, "O") == 0){
		format = 26;
		keplerian = 1;
	}
	else if(strcmp(ff, "w") == 0){
		format = 27;
		keplerian = 1;
	}
	else if(strcmp(ff, "M") == 0){
		format = 28;
		keplerian = 1;
	}
	else if(strcmp(ff, "aL") == 0){		//tunig lengths for mcmc step
		format = 29;
	}
	else if(strcmp(ff, "eL") == 0){
		format = 30;
	}
	else if(strcmp(ff, "incL") == 0){
		format = 31;
	}
	else if(strcmp(ff, "mL") == 0){
		format = 32;
	}
	else if(strcmp(ff, "OL") == 0){
		format = 33;
	}
	else if(strcmp(ff, "wL") == 0){
		format = 34;
	}
	else if(strcmp(ff, "ML") == 0){
		format = 35;
	}
	else if(strcmp(ff, "rL") == 0){
		format = 36;
	}
	else if(strcmp(ff, "saT") == 0){
		format = 37;
	}
	else if(strcmp(ff, "P") == 0){
		format = 38;
	}
	else if(strcmp(ff, "PL") == 0){
		format = 39;
	}
	else if(strcmp(ff, "T") == 0){
		format = 40;
	}
	else if(strcmp(ff, "TL") == 0){
		format = 41;
	}
	else if(strcmp(ff, "Rc") == 0){		//Rcrit
		format = 42;
	}
	else if(strcmp(ff, "gw") == 0){		//gamma w
		format = 43;
	}
	else if(strcmp(ff, ">>") == 0){
		return 2;
	}
	else if(strcmp(ff, "<<") == 0){
	}
	else {
		printf("Error: Input format not valid! Maybe the spaces in << ... >> have been forgotten\n");
		return 1;
	}

	if(cartesian == 1 && keplerian == 1){
		printf("Error: Input file format is not valid! Kartesian and Keplerian coordinates can not be mixed.\n");
		return 1;
		
	}

	return 0;
}

// ************************************************
//This function allocates memory on the Host
//Authors: Simon Grimm
//September 2016
// ************************************************
__host__ void Host::Halloc(){
	NB = (int*)malloc(Nst*sizeof(int));
	NBT = (int*)malloc(Nst*sizeof(int));
	Nmin = (int2*)malloc(Nst*sizeof(int2));				// x: masive particles, y: test particles
	rho = (double*)malloc(Nst*sizeof(double));	
	
	P.dev = 0;
	GSF = (struct GSFiles*)malloc(Nst*sizeof(struct GSFiles));
	
	n1_h = (double*)malloc(Nst*sizeof(double));
	n2_h = (double*)malloc(Nst*sizeof(double));
	N_h = (int*)malloc(Nst*sizeof(int));
	Nsmall_h = (int*)malloc(Nst*sizeof(int));
	Msun_h = (double4*)malloc(Nst*sizeof(double4));
	Spinsun_h = (double4*)malloc(Nst*sizeof(double4));
	idt_h = (double*)malloc(Nst*sizeof(double));
	ict_h = (double*)malloc(Nst*sizeof(double));
	Rcut_h = (double*)malloc(Nst*sizeof(double));
	RcutSun_h = (double*)malloc(Nst*sizeof(double));
	time_h = (double*)malloc(Nst*sizeof(double));
	dt_h = (double*)malloc(Nst*sizeof(double));
	delta_h = (long long*)malloc(Nst*sizeof(long long));
	
	//Initialize parameters with default values
	P.ei = def_EnergyOutputInterval;
	P.ci = def_CoordinatesOutputInterval;
	P.nci = def_OutputsPerInterval;
	P.Buffer = def_Buffer;
	P.deltaT = def_IntegrationSteps;
	P.UseTestParticles = def_UseTestParticles;
	P.MinMass = def_MinMass;
	P.tRestart = def_RestartTimeStep;	
	P.SIO = def_OderOfIntegrator;
	P.NencMax = def_NencMax;
	P.SLevels = def_SLevels;
	P.SLSteps = def_SLSteps;
	P.AngleUnits = 0;		//0: radians, 1:degrees
	P.UseaeGrid = def_UseaeGrid;
	Gridae.amin = def_aeGridamin;
	Gridae.amax = def_aeGridamax;		
	Gridae.emin = def_aeGridemin;
	Gridae.emax = def_aeGridemax;	
	Gridae.imin = def_aeGridimin;
	Gridae.imax = def_aeGridimax;	
	Gridae.Na = def_aeGridNa;
	Gridae.Ne = def_aeGridNe;	
	Gridae.Ni = def_aeGridNi;	
	Gridae.Start = def_aeGridStartCount;
	sprintf(Gridae.X, def_aeGridName);
	P.Usegas = def_Usegas;
	P.UsegasEnhance = def_UsegasEnhance;
	P.UseForce = def_UseForce;
	P.UseYarkovsky = def_UseYarkovsky;
	P.UseSmallCollisions = def_UseSmallCollisions;
	P.UsePR = def_UsePR;
	P.Qpr = def_Qpr;
	P.SolarConstant = def_SolarConstant;
	P.G_dTau_diss = def_GasdTau_diss;
	P.G_alpha = def_GasAlpha;
	P.G_Sigma_10 = def_G_Sigma_10 * 1.49598*1.49598/1.98892*1.0e-7;
	P.FormatS = def_FormatS;
	P.FormatT = def_FormatT;
	P.FormatP = def_FormatP;
	P.FormatO = def_FormatO;
	P.WriteEncounters = def_WriteEncounters;
	P.WriteEncountersRadius = def_WriteEncountersRadius;
	P.StopAtEncounter = def_StopAtEncounter;
	P.StopAtEncounterRadius = def_StopAtEncounterRadius;
	P.StopAtCollision = def_StopAtCollision;
	P.StopMinMass = def_StopMinMass;
	P.CollisionPrecision = def_CollisionPrecision;
	P.CollTshift = def_CollTshift;
	P.NAFvars = def_NAFvars;
	P.NAFn0 = def_NAFn0;
	P.NAFnfreqs = def_NAFnfreqs;
	P.NAFformat = def_NAFformat;
	P.NAFinterval = def_NAFinterval;
	
	P.IrregularOutputs = 0;
	sprintf(P.IrregularOutputsfilename, "%s", "-");
	P.setElements = 0;
	P.setElementsV = 0;
	sprintf(P.setElementsfilename, "%s", "-");
	P.setElementsN = 0;
	P.UseTransits = 0;
	P.UseRV = 0;
	P.TransitSteps = 1;
	sprintf(P.Transitsfilename, "%s", "-");
	sprintf(P.RVfilename, "%s", "-");
	P.PrintTransits = 0;
	P.PrintRV = 0;
	P.PrintMCMC = 0;
	P.mcmcNE = MCMC_NE;
	P.mcmcRestart = 0;
	sprintf(P.Gasfilename, "%s", "-");
	
	char format[def_Ninformat];
	sprintf(format, def_InputFileFormat);
	
	for(int st = 0; st < Nst; ++st){
		for(int i = 0; i < def_Ninformat; ++i){
			GSF[st].informat[i] = 0;
		}

		int pos = 0;		
		for(int f = -1; f < def_Ninformat; ++f){
			char ff[5];
			int n = 0;
			int er = sscanf(format + pos, "%s%n", ff, &n);
			if(er <= 0) break;

			pos += n;

			er = assignInformat(ff, GSF[st].informat[f]);
			if(er == 2) break;

		}	
		n1_h[st] = def_n1;
		n2_h[st] = def_n2;
		N_h[st] = 32;
		Nsmall_h[st] = 0;
		Msun_h[st].x = def_CentralMass;
		Msun_h[st].y = def_CentralRadius;
		Msun_h[st].z = def_CentralK2;
		Msun_h[st].w = def_CentralK2f;
		Spinsun_h[st].x = 0.0;
		Spinsun_h[st].y = 0.0;
		Spinsun_h[st].z = 0.0;
		Spinsun_h[st].w = 0.0;
		idt_h[st] = def_TimeStep;
		ict_h[st] = 0.0;
		Rcut_h[st] = def_Rcut;
		RcutSun_h[st] = def_RcutSun;
		time_h[st] = 0.0;
		dt_h[st] = 0.0;
		delta_h[st] = def_IntegrationSteps;
		
		NB[st] = N_h[st];
		NBT[st] = N_h[st] + Nsmall_h[st];
		Nmin[st].x = def_MinimumNumberOfBodies;
		Nmin[st].y = def_MinimumNumberOfTestParticles;
		rho[st] = def_rho;
		sprintf(GSF[st].X, def_Name);
		sprintf(GSF[st].inputfilename, def_InputFile);
	}
	
	//Read the paths for the individual simulations
	if(Nst > 1){
		if(MTFlag == 0){
			pathfile = fopen(pathfilename, "r");
			for(int st = 0; st < Nst; ++st){
				char t[160];
				fscanf(pathfile, "%s", t);
				sprintf(GSF[st].path, "%s/", t);
			}
			fclose(pathfile);
		}
		else{
			for(int st = 0; st < Nst; ++st){
				GSF[st].path[0] = 0; // = ""
			}
		}
	}
	else GSF[0].path[0] = 0; // = ""
};


// ************************************************
//This function reads the parameters from param.dat and the console input arguments.
//Return 1 by sucess and 0 by an error.
//
//Authors: Simon Grimm
//March 2017
// ***********************************************
__host__ int Host::readparam(FILE *paramfile, int st, int argc, char*argv[]){
	
	char sp[160];
	int er;
	
	for(int j = 0; j < 100; ++j){ //loop around all lines in the param.dat file
		int c;
		for(int i = 0; i < 50; ++i){
			c = fgetc(paramfile);
			if(c == EOF) break;
			sp[i] = char(c);
			if(c == '=' || c == ':'){
				sp[i + 1] = '\0';
				break;
			}
		}
		if(c == EOF) break;
		
		if(strcmp(sp, "Time step in days =") == 0){
			er = fscanf (paramfile, "%lf", &idt_h[st]);
			if(er <= 0){
				printf("Error: time step is not valid!\n");
				return 0;
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Output name =") == 0){
			er = fscanf (paramfile, "%s", GSF[st].X);
			if(er <= 0){
				printf("Error: Output name is not valid!\n");	
				return 0;
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Energy output interval =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.ei);
				if(er <= 0 || P.ei < -1){
					printf("Error: Energy output interval is not valid!\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Coordinates output interval =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.ci);
				
				if(er <= 0 || P.ci < -1){
					printf("Error: Coordinates outut interval is not valid!\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			fgets(sp, 3, paramfile);
		}
		
		else if(strcmp(sp, "Number of outputs per interval =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.nci);
				
				if(er <= 0 || P.nci <= 0 || (P.nci > P.ci && P.ci > 0)){
					printf("Error: Number of outputs per interval is not valid!\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Irregular output calendar =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%s", P.IrregularOutputsfilename);
				
				if(er <= 0){
					printf("Error: Irregular output calendar is not valid!\n");
					return 0;
				}
			}
			else{
				char t;
				er = fscanf (paramfile, "%s", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "TTV file name =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%s", P.Transitsfilename);
				
				if(er <= 0){
					printf("Error: TTV filename is not valid!\n");
					return 0;
				}
			}
			else{
				char t;
				er = fscanf (paramfile, "%s", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "RV file name =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%s", P.RVfilename);
				
				if(er <= 0){
					printf("Error: RV filename is not valid!\n");
					return 0;
				}
			}
			else{
				char t;
				er = fscanf (paramfile, "%s", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "TTV steps =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.TransitSteps);
				
				if(er <= 0){
					printf("Error: TTV Steps is not valid!\n");
					return 0;
				}
			}
			else{
				char t;
				er = fscanf (paramfile, "%s", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Print Transits =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.PrintTransits);
				
				if(er <= 0){
					printf("Error: Print Transits is not valid!\n");
					return 0;
				}
			}
			else{
				char t;
				er = fscanf (paramfile, "%s", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Print RV =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.PrintRV);
				
				if(er <= 0){
					printf("Error: Print RV is not valid!\n");
					return 0;
				}
			}
			else{
				char t;
				er = fscanf (paramfile, "%s", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Print MCMC =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.PrintMCMC);
				
				if(er <= 0){
					printf("Error: Print MCMC is not valid!\n");
					return 0;
				}
			}
			else{
				char t;
				er = fscanf (paramfile, "%s", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "MCMC NE =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.mcmcNE);
				
				if(er <= 0){
					printf("Error: MCMC NE is not valid!\n");
					return 0;
				}
			}
			else{
				char t;
				er = fscanf (paramfile, "%s", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "MCMC Restart =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.mcmcRestart);
				
				if(er <= 0){
					printf("Error: MCMC Restart is not valid!\n");
					return 0;
				}
			}
			else{
				char t;
				er = fscanf (paramfile, "%s", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Integration steps =") == 0){
			er = fscanf (paramfile, "%lld", &delta_h[st]);
			
			if(er <= 0 || delta_h[st] <= 0){
				printf("Error: Inegration steps are not valid!\n");
				return 0;
			}
			fgets(sp, 3, paramfile);
			if(st == 0) P.deltaT = delta_h[st];
			else P.deltaT = max(P.deltaT, delta_h[st]);
		}
		else if(strcmp(sp, "Central Mass =") == 0){
			
			er = fscanf (paramfile, "%lf", &Msun_h[st].x);
			
			if(er <= 0){
				printf("Error: Central mass is not valid!\n");
				return 0;
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Star Radius =") == 0){
			
			er = fscanf (paramfile, "%lf", &Msun_h[st].y);
			
			if(er <= 0){
				printf("Error: Star Raius is not valid!\n");
				return 0;
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Star Love Number =") == 0){
			
			er = fscanf (paramfile, "%lf", &Msun_h[st].z);
			
			if(er <= 0){
				printf("Error: Star Love Number is not valid!\n");
				return 0;
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Star fluid Love Number =") == 0){
			
			er = fscanf (paramfile, "%lf", &Msun_h[st].w);
			
			if(er <= 0){
				printf("Error: Star fluid Love Number is not valid!\n");
				return 0;
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Star spin_x =") == 0){
			
			er = fscanf (paramfile, "%lf", &Spinsun_h[st].x);
			
			if(er <= 0){
				printf("Error: Star spin_x is not valid!\n");
				return 0;
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Star spin_y =") == 0){
			
			er = fscanf (paramfile, "%lf", &Spinsun_h[st].y);
			
			if(er <= 0){
				printf("Error: Star spin_y is not valid!\n");
				return 0;
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Star spin_z =") == 0){
			
			er = fscanf (paramfile, "%lf", &Spinsun_h[st].z);
			
			if(er <= 0){
				printf("Error: Star spin_z is not valid!\n");
				return 0;
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Star tau =") == 0){
			
			er = fscanf (paramfile, "%lf", &Spinsun_h[st].w);
			
			if(er <= 0){
				printf("Error: Star tau is not valid!\n");
				return 0;
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Solar Constant =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%lf", &P.SolarConstant);
				if(er <= 0){
					printf("Error: Solar Constant value is not valid!\n");
					return 0;
				}
			}
			else{
				double t;
				er = fscanf (paramfile, "%lf", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "n1 =") == 0){
			er = fscanf (paramfile, "%lf", &n1_h[st]);
			
			if(er <= 0){
				printf("Error: n1 is not valid!\n");
				return 0;
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "n2 =") == 0){
			
			er = fscanf (paramfile, "%lf", &n2_h[st]);
			if(er <= 0){
				printf("Error: n2 is not valid!\n");
				return 0;
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Input file =") == 0){
			er = fscanf (paramfile, "%s", GSF[st].inputfilename);
			
			if(er <= 0){
				printf("Error: Input file name is not valid!\n");
				return 0;
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Input file Format:") == 0){
			for(int i = 0; i < def_Ninformat; ++i){
				GSF[st].informat[i] = 0;
			}
			//Read input file Format
			int f;
			for(f = -1; f < 50; ++f){
				er = fscanf (paramfile, "%s", sp);
	

				int er2 = assignInformat(sp, GSF[st].informat[f]);
				if(er2 == 2) break;
				if(er2 == 1) return 0;

			}
			if(er <= 0){
				printf("Error: Input file format is not valid!\n");
				return 0;
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Angle units =") == 0){
			if(st == 0){
				char angle[16];
				er = fscanf (paramfile, "%s", angle);
				if(strcmp(angle, "radians") == 0){
					P.AngleUnits = 0;
				}
				else if(strcmp(angle, "degrees") == 0){
					P.AngleUnits = 1;
				}
				else{
					er = -1;
				}

				if(er <= 0){
					printf("Error: Angle units value is not valid!\n");
					return 0;
				}
			}
			else{
				char angle[16];
				er = fscanf (paramfile, "%s", angle);
			}
			fgets(sp, 3, paramfile);
		}
		
		else if(strcmp(sp, "Default rho =") == 0){
			er = fscanf (paramfile, "%lf", &rho[st]);
			if(er <= 0 ){
				printf("Error: Default value for rho is not valid!\n");
				return 0;
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Use Test Particles =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.UseTestParticles);
				if(er <= 0 || P.UseTestParticles < 0 || P.UseTestParticles > 2){
					printf("Error: Test Particle Mode not valid\n");
					return 0;
				}
			}
			else{
				long long t;
				er = fscanf (paramfile, "%lld", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Particle Minimum Mass =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%lf", &P.MinMass);
				if(er <= 0 || P.MinMass < 0){
					printf("Error: Particle Minimum Mass not valid\n");
					return 0;
				}
			}
			else{
				double t;
				er = fscanf (paramfile, "%lf", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Restart timestep =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%lld", &P.tRestart);
				if(er <= 0 || P.tRestart < -1){
					printf("Error: Restart time step not valid\n");
					return 0;
				}
			}
			else{
				long long t;
				er = fscanf (paramfile, "%lld", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Minimum number of bodies =") == 0){
			er = fscanf (paramfile, "%d", &Nmin[st].x);
			if(er <= 0 || Nmin[st].x < 0){
				printf("Error: Minimal number of bodies not valid\n");
				return 0;
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Minimum number of test particles =") == 0){
			er = fscanf (paramfile, "%d", &Nmin[st].y);
			if(er <= 0 || Nmin[st].y < 0){
				printf("Error: Minimal number of test particles not valid\n");
				return 0;
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Inner truncation radius =") == 0){
			er = fscanf (paramfile, "%lf", &RcutSun_h[st]);
			if(er <= 0 || RcutSun_h[st] < 0){
				printf("Error: Inner truncation radius not valid\n");
				return 0;
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Outer truncation radius =") == 0){
			er = fscanf (paramfile, "%lf", &Rcut_h[st]);
			if(er <= 0 || Rcut_h[st] < 0){
				printf("Error: Outer truncation radius not valid\n");
				return 0;
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Order of integrator =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.SIO);
				if(er <= 0 || P.SIO < 2 || P.SIO > 6 || P.SIO % 2 == 1){
					printf("Error: Order of integrator not valid\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Use aeGrid =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.UseaeGrid);
				if(er <= 0){
					printf("Error: Use aeGrid not valid\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "aeGrid amin =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%f", &Gridae.amin);
				if(er <= 0){
					printf("Error: Grid amin not valid\n");
					return 0;
				}
			}
			else{
				float t;
				er = fscanf (paramfile, "%f", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "aeGrid amax =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%f", &Gridae.amax);
				if(er <= 0){
					printf("Error: Grid amax not valid\n");
					return 0;
				}
			}
			else{
				float t;
				er = fscanf (paramfile, "%f", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "aeGrid emin =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%f", &Gridae.emin);
				if(er <= 0){
					printf("Error: Grid emin not valid\n");
					return 0;
				}
			}
			else{
				float t;
				er = fscanf (paramfile, "%f", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "aeGrid emax =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%f", &Gridae.emax);
				if(er <= 0){
					printf("Error: Grid emax not valid\n");
					return 0;
				}
			}
			else{
				float t;
				er = fscanf (paramfile, "%f", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "aeGrid imin =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%f", &Gridae.imin);
				if(er <= 0){
					printf("Error: Grid imin not valid\n");
					return 0;
				}
			}
			else{
				float t;
				er = fscanf (paramfile, "%f", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "aeGrid imax =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%f", &Gridae.imax);
				if(er <= 0){
					printf("Error: Grid imax not valid\n");
					return 0;
				}
			}
			else{
				float t;
				er = fscanf (paramfile, "%f", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "aeGrid Na =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &Gridae.Na);
				if(er <= 0){
					printf("Error: Grid Na not valid\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "aeGrid Ne =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &Gridae.Ne);
				if(er <= 0){
					printf("Error: Grid Ne not valid\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "aeGrid Ni =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &Gridae.Ni);
				if(er <= 0){
					printf("Error: Grid Ni not valid\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "aeGrid Start Count =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%lld", &Gridae.Start);
				if(er <= 0){
					printf("Error: Grid Start not valid\n");
					return 0;
				}
			}
			else{
				long long t;
				er = fscanf (paramfile, "%lld", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "aeGrid name =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%s", Gridae.X);
				
				if(er <= 0){
					printf("Error: Grid name is not valid!\n");
					return 0;
				}	
			}
			else{
				char t[64];
				er = fscanf (paramfile, "%s", t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Use gas disk =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.Usegas);
				if(er <= 0){
					printf("Error: Use gas Disk value is not valid!\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Use gas disk enhancement =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.UsegasEnhance);
				if(er <= 0){
					printf("Error: Use gas disk enhancement value is not valid!\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Gas dTau_diss =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%lf", &P.G_dTau_diss);
				if(er <= 0){
					printf("Error: dTau_diss value is not valid!\n");
					return 0;
				}
			}
			else{
				double t;
				er = fscanf (paramfile, "%lf", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Gas alpha =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.G_alpha);
				if(er <= 0 || !(P.G_alpha == 1 || P.G_alpha == 2)){
					printf("Error: Gas alpha value is not valid!\n");
					return 0;
				}
				
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Gas Sigma_10 =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%lf", &P.G_Sigma_10);
				P.G_Sigma_10 *= 1.49598*1.49598/1.98892*1.0e-7;
				if(er <= 0 || P.G_Sigma_10 < 0){
					printf("Error: Gas Sigma_10 value is not valid!\n");
					return 0;
				}
				
			}
			else{
				double t;
				er = fscanf (paramfile, "%lf", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Use force =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.UseForce);
				if(er <= 0){
					printf("Error: Use force value is not valid!\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Use Yarkovsky =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.UseYarkovsky);
				if(er <= 0){
					printf("Error: Use Yarkovsky value is not valid!\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Use Small Collisions =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.UseSmallCollisions);
				if(er <= 0){
					printf("Error: Use Small Collisions value is not valid!\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Use Poynting-Robertson =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.UsePR);
				if(er <= 0){
					printf("Error: Use Poynting-Robertson value is not valid!\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Radiation Pressure Coefficient Qpr =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%lf", &P.Qpr);
				if(er <= 0){
					printf("Error: Radiation Pressure Coefficient value is not valid!\n");
					return 0;
				}
			}
			else{
				double t;
				er = fscanf (paramfile, "%lf", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "FormatS =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.FormatS);
				if(er <= 0){
					printf("Error: FormatS value is not valid!\n");
					return 0;
				}
				
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "FormatT =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.FormatT);
				if(er <= 0){
					printf("Error: FormatT value is not valid!\n");
					return 0;
				}
				
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "FormatP =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.FormatP);
				if(er <= 0){
					printf("Error: FormatP value is not valid!\n");
					return 0;
				}
				
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "FormatO =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.FormatO);
				if(er <= 0){
					printf("Error: FormatO value is not valid!\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Report Encounters =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.WriteEncounters);
				if(er <= 0){
					printf("Error: Report Encounters value is not valid!\n");
					return 0;
				}
				
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Report Encounters Radius =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%lf", &P.WriteEncountersRadius);
				if(er <= 0){
					printf("Error: Report Encounters Radius value is not valid!\n");
					return 0;
				}
				
			}
			else{
				double t;
				er = fscanf (paramfile, "%lf", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Stop at Encounter =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.StopAtEncounter);
				if(er <= 0){
					printf("Error: Stop at Encounter value is not valid!\n");
					return 0;
				}
				
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Stop at Encounter Radius =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%lf", &P.StopAtEncounterRadius);
				if(er <= 0){
					printf("Error: Stop at Encounter Radius value is not valid!\n");
					return 0;
				}
				
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Stop at Collision =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.StopAtCollision);
				if(er <= 0){
					printf("Error: Stop at Collision value is not valid!\n");
					return 0;
				}
				
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Stop Minimum Mass =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%lf", &P.StopMinMass);
				if(er <= 0){
					printf("Error: Stop Minumun Mass value is not valid!\n");
					return 0;
				}
				
			}
			else{
				double t;
				er = fscanf (paramfile, "%lf", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Collision Precision =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%lf", &P.CollisionPrecision);
				if(er <= 0){
					printf("Error: Collision Precision value is not valid!\n");
					return 0;
				}
				
			}
			else{
				double t;
				er = fscanf (paramfile, "%lf", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Collision Time Shift =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%lf", &P.CollTshift);
				if(er <= 0){
					printf("Error: Collision Time Shift value is not valid!\n");
					return 0;
				}
				
			}
			else{
				double t;
				er = fscanf (paramfile, "%lf", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Coordinate output buffer =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.Buffer);
				if(er <= 0){
					printf("Error: Coordinate output buffer value is not valid!\n");
					return 0;
				}
				if(P.Buffer < 1) P.Buffer = 1;
				
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Set Elements file name =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%s", P.setElementsfilename);
				
				if(er <= 0){
					printf("Error: Set Elements file name = is not valid!\n");
					return 0;
				}
			}
			else{
				char t;
				er = fscanf (paramfile, "%s", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Gas file name =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%s", P.Gasfilename);
				
				if(er <= 0){
					printf("Error: Gas file name = is not valid!\n");
					return 0;
				}
			}
			else{
				char t;
				er = fscanf (paramfile, "%s", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "NAF variables =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.NAFvars);
				
				if(er <= 0){
					printf("Error: NAF variables = is not valid!\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "NAF size =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.NAFn0);
				
				if(er <= 0){
					printf("Error: NAF size = is not valid!\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "NAF nfreqs =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.NAFnfreqs);
				
				if(er <= 0){
					printf("Error: NAF nfreqs = is not valid!\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "NAF format =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.NAFformat);
				
				if(er <= 0){
					printf("Error: NAF format = is not valid!\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "NAF interval =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.NAFinterval);
				
				if(er <= 0 || P.NAFinterval <= 0){
					printf("Error: NAF interval = is not valid!\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Maximum encounter pairs =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.NencMax);
				if(P.NencMax < 512) P.NencMax = 512;	
				if(er <= 0){
					printf("Error: Maximum encounter pairs = is not valid!\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Symplectic recursion levels =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.SLevels);
				if(P.SLevels > def_SLevelsMax){
					printf("Error, Symplectic recursion levels larger than def_SLevelsMax %d %d\n", P.SLevels, def_SLevelsMax);
					P.SLevels = def_SLevelsMax;
				}
				if(er <= 0){
					printf("Error: Symplectic recursion levels = is not valid!\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Symplectic recursion sub steps =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.SLSteps);
				if(er <= 0){
					printf("Error: Symplectic recursion sub steps = is not valid!\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			fgets(sp, 3, paramfile);
		}
		else{
			printf("Undefined line in param.dat file: line %d\n", j);
			return 0;
		}
	}
	
	
	
	if(st == 0){
		
		Gridae.deltaa = (Gridae.amax - Gridae.amin) / ((float)(Gridae.Na));
		Gridae.deltae = (Gridae.emax - Gridae.emin) / ((float)(Gridae.Ne));
		Gridae.deltai = (Gridae.imax - Gridae.imin) / ((float)(Gridae.Ni));
		
	}
	
	//Read console input arguments
	for(int i = 1; i < argc; i += 2){
		
		if(strcmp(argv[i], "-dt") == 0){
			idt_h[st] = atof(argv[i + 1]);
		}
		else if(strcmp(argv[i], "-ei") == 0){
			P.ei = atoi(argv[i + 1]);
		}
		else if(strcmp(argv[i], "-ci") == 0){
			P.ci = atoi(argv[i + 1]);
		}
		else if(strcmp(argv[i], "-I") == 0){
			P.deltaT = atol(argv[i + 1]);
			delta_h[st] = P.deltaT;
		}
		else if(strcmp(argv[i], "-n1") == 0){
			n1_h[st] = atof(argv[i + 1]);
		}
		else if(strcmp(argv[i], "-n2") == 0){ 
			n2_h[st] = atof(argv[i + 1]);
		}
		else if(strcmp(argv[i], "-dev") == 0){
			P.dev = atof(argv[i + 1]);
		}
		else if(strcmp(argv[i], "-in") == 0){
			sprintf(GSF[st].inputfilename, "%s", argv[i + 1]);
		}
		else if(strcmp(argv[i], "-out") == 0){
			sprintf(GSF[st].X, "%s", argv[i + 1]);
		}
		else if(strcmp(argv[i], "-R") == 0){
			P.tRestart = atol(argv[i + 1]);
			if(P.tRestart < -1){
				printf("Error: Restart time step not valid\n");
				return 0;
			}
		}
		else if(strcmp(argv[i], "-TP") == 0){
			P.UseTestParticles = atoi(argv[i + 1]);
		}
		else if(strcmp(argv[i], "-M") == 0){
		}
		else if(strcmp(argv[i], "-Nmin") == 0){
			Nmin[st].x = atoi(argv[i + 1]);
		}
		else if(strcmp(argv[i], "-NminTP") == 0){
			Nmin[st].y = atoi(argv[i + 1]);
		}
		else if(strcmp(argv[i], "-SIO") == 0){
			P.SIO = atoi(argv[i + 1]);
		}
		else if(strcmp(argv[i], "-aeN") == 0){
			sprintf(Gridae.X, "%s", argv[i + 1]);
		}
		else if(strcmp(argv[i], "-t") == 0){
			ict_h[st] = atof(argv[i + 1]);
		}
		else if(strcmp(argv[i], "-MT") == 0){
			Nst = atoi(argv[i + 1]);
		}
		else if(strcmp(argv[i], "-sl") == 0){
			P.SLevels = atoi(argv[i + 1]);
		}
		else if(strcmp(argv[i], "-sls") == 0){
			P.SLSteps = atoi(argv[i + 1]);
		}
		else{
			printf("Error: Console arguments not valid!\n");
			return 0;
		}
	}
	sprintf(GSF[st].Originputfilename, "%s", GSF[st].inputfilename);
	if(strcmp(P.IrregularOutputsfilename, "-") != 0){
		P.IrregularOutputs = 1;
	}
	if(strcmp(P.Transitsfilename, "-") != 0){
		P.UseTransits = 1;
		#if def_TTV == 0
		printf("Error: TTV file not allowed for def_TTV = 0!\n");
		return 0;
		
		#endif
	}
	if(strcmp(P.RVfilename, "-") != 0){
		P.UseRV = 1;
		#if def_RV == 0
		printf("Error: RV file not allowed for def_RV = 0!\n");
		return 0;
		
		#endif
	}
	if(strcmp(P.setElementsfilename, "-") != 0){
		P.setElements = 1;
	}
	if(strcmp(P.Gasfilename, "-") != 0){
		P.Usegas = 2;
	}
	if(P.Usegas == 1 && P.G_dTau_diss <= 0.0){
		printf("Error: dTau_diss value is not valid!\n");
		return 0;
	}
	if(P.UseTestParticles == 0){
		P.MinMass = 0.0;
		Nmin[st].y = 0;
	}
	
	
	if(P.StopAtCollision != 0 && Nst > 1){
		printf("Error: Stop at Collision not available in multi simulation mode!\n");
		return 0;
	}
	
	if(P.CollTshift < 1.0){
		printf("Error: Collision Time Shift not valid! %g\n", P.CollTshift);
		return 0;
	}
	
	if(P.CollisionPrecision <= 0.0){
		printf("Error: Collision Precision not valid! %g\n", P.CollisionPrecision);
		return 0;
	}


	if(P.SLevels > def_SLevelsMax){
		printf("Error, Symplectic recursion levels bigger than def_SLevelsMax %d %d\n", P.SLevels, def_SLevelsMax);
		return 0;
	} 

	if(log10(delta_h[st]) >= def_NFileNameDigits && P.FormatO == 0){
		printf("Error, number of time steps larger than number of digits in the output filenames. Increase def_NFileNameDigits in the define.h file or use P.FormatO = 1.\n");
		return 0;

	}
	if(P.ci > 0 && log10(delta_h[st] / P.ci) >= def_NFileNameDigits && P.FormatO == 1){
		printf("Error, number of time steps larger than number of digits in the output filenames. Increase def_NFileNameDigits in the define.h file.\n");
		return 0;

	}

	ForceFlag = 0;
	if(P.UseForce > 0 || P.Usegas > 0 || P.UseYarkovsky > 0 || P.UsePR > 0){
		ForceFlag = 1;
	}
	
	return 1;
}


// ************************************************
//This function calls the function readparam
//
//Authors: Simon Grimm
//February 2018
// *********************************************3
__host__ int Host::Param(int argc, char*argv[]){
	FILE *paramfile;
	char paramfilename[300];
	// Read parameters from param file //
	for(int st = 0; st < Nst; ++st){
		sprintf(paramfilename, "%s%s", GSF[st].path, "param.dat");
		paramfile = fopen(paramfilename, "r");
		if(paramfile == NULL){
			if(Nst == 1) printf("Error: file param.dat doesn't exist!\n");
			else printf("Error in Simulation %s: file param.dat or path doesn't exist!\n", GSF[st].path);
			fprintf(masterfile, "Error in Simulation %s\n", GSF[st].path);
			return 0;
		}
		int er;
		er = readparam(paramfile, st, argc, argv);
		if(dayUnit == 1){
			Msun_h[st].x *= def_Kg;	//convert to mercury units
		}
		if(er == 0) return 0;
		fclose(paramfile);
		
		if(Nst > 1){
			char tname[300];
			sprintf(tname, "%s%s", GSF[st].path, GSF[st].inputfilename);
			sprintf(GSF[st].inputfilename, "%s", tname);
			P.UseTestParticles = 0;
		}
		dt_h[st] = idt_h[st] * dayUnit;
	}
	if((P.ei > P.ci && P.ci > 0) || (P.ci == -1 && P.ei == 0)){
		P.ei = P.ci;
		printf("**** Energy output interval decreased equal to coordinate output interval ****\n");
		fprintf(masterfile, "**** Energy output interval decreased equal to coordinate output interval ****\n");
	}
	//if Restart == -1, find last printed output
	int RestartBackup = 0;	//Flag, used to find last output in P.FormatO 1 restarts
	if(P.tRestart == -1){
		RestartBackup = 1;
		long long Restart = -1;
		for(int st = 0; st < Nst; ++st){
			FILE *timefile;
			char timefilename[300];
			sprintf(timefilename, "%stime%s.dat", GSF[st].path, GSF[st].X);
			int er = 0;
			timefile = fopen(timefilename, "r");
			
			if(timefile == NULL){
				printf("Warning: file %s not found. Restore last time step not possible -> begin new simulation\n", timefilename);
				fprintf(masterfile, "Warning: file %s not found. Restore last time step not possible -> begin new simulation\n", timefilename);
				Restart = 0;
			}
			else{
				long long ts = 0LL;
				double time = 0.0;
				
				for(int i = 0; i < 1e8; ++i){
					er = fscanf (timefile, "%lld",&ts);
					er = fscanf (timefile, "%lf",&time);
					if(er < 0){
						Restart = ts;
						break;
					}
				}
				fclose(timefile);
				if(Restart < 0){
					printf("Error: restore last time step failed\n");
					fprintf(masterfile, "Error: restore last time step failed\n");
					return 0;
				}
			}
		P.tRestart = max(Restart, P.tRestart);
		}
	}
	if(P.ci != -1 && P.ci != 0){
		 if(P.tRestart % P.ci == 0) RestartBackup = 0;
	}
//printf("restart %lld %d\n", P.tRestart, RestartBackup);
	
	for(int st = 0; st < Nst; ++st){
		//restart -> inputfilename
		if(P.tRestart > 0 && P.FormatP == 1){
			if(Nst == 1 || P.FormatS == 0){
				if(P.FormatT == 0){
					long long scale = 1ll;
					if(P.FormatO == 1){
						scale = (long long)(P.ci);
						if(P.ci == -1) scale = P.tRestart;
					}
					sprintf(GSF[st].inputfilename, "%sOut%s_%.*lld.dat", GSF[st].path, GSF[st].X, def_NFileNameDigits, P.tRestart);
					if(P.FormatO == 1) sprintf(GSF[st].inputfilename, "%sOut%s_%.*lld.dat", GSF[st].path, GSF[st].X, def_NFileNameDigits, P.tRestart / scale);
					if(P.FormatO == 1 && RestartBackup == 1) sprintf(GSF[st].inputfilename, "%sOutbackup%s_%.20lld.dat", GSF[st].path, GSF[st].X, P.tRestart);
				}
				if(P.FormatT == 1) sprintf(GSF[st].inputfilename, "%sOut%s.dat", GSF[st].path, GSF[st].X);
			}
			else{
				if(P.FormatT == 0){
					long long scale = 1ll;
					if(P.FormatO == 1){
						scale = (long long)(P.ci);
						if(P.ci == -1) scale = P.tRestart;
					}
					sprintf(GSF[st].inputfilename, "Out%s_%.*lld.dat", GSF[st].X, def_NFileNameDigits, P.tRestart);
					if(P.FormatO == 1) sprintf(GSF[st].inputfilename, "Out%s_%.*lld.dat", GSF[st].X, def_NFileNameDigits, P.tRestart / scale);
					if(P.FormatO == 1 && RestartBackup == 1) sprintf(GSF[st].inputfilename, "Outbackup%s_%.20lld.dat", GSF[st].X, P.tRestart);
				}
				if(P.FormatT == 1) sprintf(GSF[st].inputfilename, "Out%s.dat", GSF[st].X);
			}
		}
		sprintf(GSF[st].logfilename, "%sinfo%s.dat", GSF[st].path, GSF[st].X);
		sprintf(GSF[st].timefilename, "%stime%s.dat", GSF[st].path, GSF[st].X);
		sprintf(GSF[st].Energyfilename, "%sEnergy%s.dat", GSF[st].path, GSF[st].X);
		sprintf(GSF[st].EnergyIrrfilename, "%sEnergyIrr%s.dat", GSF[st].path, GSF[st].X);
		sprintf(GSF[st].collisionfilename, "%sCollisions%s.dat", GSF[st].path, GSF[st].X);
		sprintf(GSF[st].collisionTshiftfilename, "%sCollisionsTShift%s.dat", GSF[st].path, GSF[st].X);
		sprintf(GSF[st].ejectfilename, "%sEjections%s.dat", GSF[st].path, GSF[st].X);
		sprintf(GSF[st].encounterfilename, "%sEncounters%s.dat", GSF[st].path, GSF[st].X);
		sprintf(GSF[st].fragmentfilename, "%sFragments%s.dat", GSF[st].path, GSF[st].X);
		
		//create files or erase content//
		if(P.tRestart == 0){
			FILE *tfile;
			GSF[st].logfile = fopen(GSF[st].logfilename, "w");
			fclose(GSF[st].logfile);

			tfile = fopen(GSF[st].timefilename, "w");
			fclose(tfile);

			tfile = fopen(GSF[st].Energyfilename, "w");
			fclose(tfile);

			tfile = fopen(GSF[st].collisionfilename, "w");
			fclose(tfile);  

			if(P.CollTshift > 1.0){
				tfile = fopen(GSF[st].collisionTshiftfilename, "w");
				fclose(tfile); 
			}

			tfile = fopen(GSF[st].ejectfilename, "w");
			fclose(tfile);

			if(P.WriteEncounters > 0){
				tfile = fopen(GSF[st].encounterfilename, "w");
				fclose(tfile);  
			}
			if(P.UseSmallCollisions > 0){
				tfile = fopen(GSF[st].fragmentfilename, "w");
				fclose(tfile);  
			}
		}
		
		GSF[st].logfile = fopen(GSF[st].logfilename, "a");
		
		if(P.tRestart > 0) fprintf(GSF[st].logfile, "\n\n\n************** Restart Simulation at time step %lld *******************\n", P.tRestart);
		fclose(GSF[st].logfile);
	}
	return 1;
}


// **************************************
// This function determines the starting time of the simulation using the input file 
// specified in the param file.

//Authors: Simon Grimm, Joachim Stadel
//Mai 2015
// ************************************3
__host__ int Host::icict(int Nformat, int st){
	double time = 0.0;
	int er = 1;
	FILE *OrigInfile;
	char Origfilename[300];
	sprintf(Origfilename, "%s%s", GSF[st].path, GSF[st].Originputfilename);
	OrigInfile = fopen(Origfilename, "r");
	if(OrigInfile == NULL){
		printf("Error in Simulation %s: Input file not found %s\n", GSF[st].path, GSF[st].inputfilename);
		fprintf(masterfile, "Error in Simulation %s: Input file not found %s\n", GSF[st].path, GSF[st].inputfilename);
		return 0;
	}
	for(int f = 0; f < Nformat; ++f){
		if(GSF[0].informat[f] == 19){
			er = fscanf (OrigInfile, "%lf",&time);
			break;
		}
	}
	if(er > 0 && ict_h[st] == 0.0 && P.tRestart > 0) ict_h[st] = time;
	fclose(OrigInfile);
	return 1;
}
// ************************************************
//This function counts the number of bodies in the initial condition file
//It returns the number of bodies
//
//Authors: Simon Grimm, Joachim Stadel
//April 2014
// *********************************************
__host__ int Host::icSize(int st){
	
	//Determinde the number of coordinates in the input file
	int Nformat = 0;
	for(int f = 0; f < def_Ninformat; ++f){
		if(GSF[st].informat[f] > 0) ++Nformat;
	}
	
	//Determine the simulation start time
	double time = 0.0;
	if(ict_h[st] > 0.0) time = ict_h[st];
	
	int er = icict(Nformat, st);
	if(er == 0) return 0;
	
	if(P.tRestart > 0 && P.FormatP == 1) Nformat = 21; //This is the number of rows in the coordinate output files 
	char t[500];
	er = 1;
	int NN = 0;
	int er1 = 1;
	double m;
	int index;
	char Ets[160]; //exact time at restart time step, must be the same format as the coordinate output
	sprintf(Ets, "%.16g", (P.tRestart * idt_h[st] + ict_h[st] * 365.25) / 365.25);
	double Et = atof(Ets);
	FILE *infile;
	infile = fopen(GSF[st].inputfilename, "r");
	if(infile == NULL){
		if(Nst == 1){
			fprintf(masterfile,"Error in Simulation %s: Input file not found %s\n", GSF[st].path, GSF[st].inputfilename);
			printf("Error in Simulation %s: Input file not found %s\n", GSF[st].path, GSF[st].inputfilename);
			return 0;
		}
		else{
			fprintf(masterfile,"Skip Simulation %s: Input file not found %s\n", GSF[st].path, GSF[st].inputfilename);
			printf("Skip Simulation %s: Input file not found %s\n", GSF[st].path, GSF[st].inputfilename);
			N_h[st] = 0;
			Nsmall_h[st] = 0;
			return 1;
		}
	}
	for(int i = 0; i < 1000000000; ++i){
		for(int f = 0; f < Nformat; ++f){
			
			if(P.tRestart == 0 || P.FormatP == 0){
				if(GSF[st].informat[f] == 4) er = fscanf (infile, "%lf",&m);
				else er = fscanf(infile, "%s", t);
			}
			else{
				if(f == 0) er = fscanf (infile, "%lf",&time);
				else if(f == 1)	er = fscanf (infile, "%d",&index);
				else if(f == 2) er = fscanf (infile, "%lf",&m);
				else er = fscanf (infile, "%s",t);
			}
			if(er <= 0){ //error by reading
				er1 = 0;
				break;
			}
			
		}
		if(P.FormatT == 1 && time > Et) break;
		//if reading was succesfull, check if particles belong to the desired time 
		if(er1 == 1){
			if(P.FormatP == 1){ // All particles in one time file
				if(P.FormatS == 0 || P.tRestart == 0 || Nst == 1){
					if(Et == time){
						if(m > P.MinMass) ++NN;
						else ++Nsmall_h[st];
					}
				}
				else if(index / def_MaxIndex == st){
					if(Et == time){
						if(m > P.MinMass) ++NN;
						else ++Nsmall_h[st];
					}
				}
			}
			if(P.FormatP == 0){
				if(P.tRestart == 0){
					if(m > P.MinMass) ++NN;
					else ++Nsmall_h[st];
				}
				else ++NN;
			}
		}
		else break;
	}
	fclose(infile);
	
	if(P.FormatP == 0 && P.tRestart > 0){//Restart FormatP == 0 data
		int NNN = 0;
		int NNNsmall = 0;
		Nformat = 21;
		FILE *OrigInfile;
		char Origfilename[300];
		sprintf(Origfilename, "%s%s", GSF[st].path, GSF[st].Originputfilename);
		OrigInfile = fopen(Origfilename, "r");
		for(int k = 0; k < 1000000000; ++k){
			double skip = 0.0;
			int eri = 1;
			int i = k;
			//if index is not given in the initial conditions file, i = k, otherwise scan for the index
			for(int f = 0; f < 22; ++f){
				if(GSF[st].informat[f] == 13){
					eri = fscanf (OrigInfile, "%d",&i);
				}
				else if(GSF[st].informat[f] > 0){
					eri = fscanf (OrigInfile, "%lf",&skip);
				}
			}
			if(eri < 0) break;
			
			int NMAX = 0;
			er1 = 1;
			char infilename[300];
			sprintf(infilename, "%sOut%s_p%.6d.dat", GSF[st].path, GSF[st].X, i);
			infile = fopen(infilename, "r");
			if(infile == NULL) continue;
			for(int it = 0; it < 1000000000; ++it){
				
				for(int f = 0; f < Nformat; ++f){
					if(f == 0) er = fscanf (infile, "%lf",&time);
					else if(f == 1)	er = fscanf (infile, "%d",&index);
					else if(f == 2) er = fscanf (infile, "%lf",&m);
					else{
						er = fscanf (infile, "%s",t);
					}
					if(er <= 0){ //error by reading
						er1 = 0;
						break;
					}
				}
//if(st < 10 && i == 1) printf("%d %d %d %.20g %.20g | %g %g\n", st, i, it, time, Et, idt_h[st], ict_h[st]);
				//if(time > Et) break;  //uncomment because of resolution increment in restarting
				
				if(er1 == 1){
					if(Nst == 1 || P.FormatS == 0){
						if(Et == time){
							if(m > P.MinMass) ++NNN;
							else ++NNNsmall;
							break;
						}
						if(NNN + NNNsmall == NN){
							NMAX = 1;
							break;
						}
					}
				}
				else{
					break;
				}
			}
			fclose(infile);
			if(NMAX == 1) break;
		}
		fclose(OrigInfile);
		NN = NNN;
		Nsmall_h[st] = NNNsmall;
	}
	
	
	
	if(P.UseTestParticles == 0){
		NN += Nsmall_h[st];
		Nsmall_h[st] = 0;
	}
	NN = min(NN, 262144);
	N_h[st] = NN;

	if(N_h[st] + Nsmall_h[st] >= 1024 * 1024){

		printf("Error More particles than 1024 * 104: scan call not implemented\n");

		return 0;
	}
	
	if(Nst == 0){
		P.NencMax = min(P.NencMax, N_h[0] + Nsmall_h[0]);
	}
	
	if(Nst > 1 && NN > NmaxM){
		fprintf(masterfile,"Error in Simulation %s: More particles than set in NmaxM: %d\n", GSF[st].path, NN);
		printf("Error in Simulation %s: More particles than set in NmaxM: %d\n", GSF[st].path, NN);
		return 0;
	}
	
	return 1;
}

// ************************************************
//This function calls the function icSize and sets the size parameters
//Authors: Simon Grimm
//January 2017
// ***********************************************3
__host__ int Host::size(){
	for(int st = 0; st < Nst; ++st){
		//Determine the size of the simulations
		int er = icSize(st);
		
		if(er == 0) return 0;
		
		NB[st] = 16;
		if( N_h[st] > 16) NB[st] = 32;
		if( N_h[st] > 32) NB[st] = 64;
		if( N_h[st] > 64) NB[st] = 128;
		if( N_h[st] > 128) NB[st] = 256;
		if( N_h[st] > 256) NB[st] = 512;
		if( N_h[st] > 512) NB[st] = 1024;
		if( N_h[st] > 1024) NB[st] = 2048;
		if( N_h[st] > 2048) NB[st] = 4096;
		if( N_h[st] > 4096) NB[st] = 8192;
		if( N_h[st] > 8192) NB[st] = 16384;
		if( N_h[st] > 16384) NB[st] = 32768;
		if( N_h[st] > 32768) NB[st] = 65536;
		if( N_h[st] > 65536) NB[st] = 131072;
		if( N_h[st] > 131072) NB[st] = 262144;
		
		NBT[st] = 16;
		if( (N_h[st] + Nsmall_h[st]) > 16) NBT[st] = 32;
		if( (N_h[st] + Nsmall_h[st]) > 32) NBT[st] = 64;
		if( (N_h[st] + Nsmall_h[st]) > 64) NBT[st] = 128;
		if( (N_h[st] + Nsmall_h[st]) > 128) NBT[st] = 256;
		if( (N_h[st] + Nsmall_h[st]) > 256) NBT[st] = 512;
		if( (N_h[st] + Nsmall_h[st]) > 512) NBT[st] = 1024;
		if( (N_h[st] + Nsmall_h[st]) > 1024) NBT[st] = 2048;
		if( (N_h[st] + Nsmall_h[st]) > 2048) NBT[st] = 4096;
		if( (N_h[st] + Nsmall_h[st]) > 4096) NBT[st] = 8192;
		if( (N_h[st] + Nsmall_h[st]) > 8192) NBT[st] = 16384;
		if( (N_h[st] + Nsmall_h[st]) > 16384) NBT[st] = 32768;
		if( (N_h[st] + Nsmall_h[st]) > 32768) NBT[st] = 65536;
		if( (N_h[st] + Nsmall_h[st]) > 65536) NBT[st] = 131072;
		if( (N_h[st] + Nsmall_h[st]) > 131072) NBT[st] = 262144;

		GSF[st].logfile = fopen(GSF[st].logfilename, "a");
		fclose(GSF[st].logfile);
		if(MTFlag == 1){
			for(int sst = 1; sst < Nst; ++sst){
				N_h[sst] = N_h[0];
				Nsmall_h[sst] = Nsmall_h[0];
				NB[sst] = NB[0];
			}
			break;
		}
	}
	return 1;
}


// ************************************************
//This function allocates memory on the device
//Author: Simon Grimm
//September 2016
// ***********************************************
__host__ void Host::Calloc(){
	cudaMalloc((void **) &n1_d,Nst*sizeof(double));
	cudaMalloc((void **) &n2_d,Nst*sizeof(double));
	cudaMalloc((void **) &N_d,Nst*sizeof(int));
	cudaMalloc((void **) &Nsmall_d,Nst*sizeof(int));
	cudaMalloc((void **) &Msun_d,Nst*sizeof(double4));
	cudaMalloc((void **) &Spinsun_d,Nst*sizeof(double4));
	cudaMalloc((void **) &idt_d,Nst*sizeof(double));
	cudaMalloc((void **) &ict_d,Nst*sizeof(double));
	cudaMalloc((void **) &Rcut_d,Nst*sizeof(double));
	cudaMalloc((void **) &RcutSun_d,Nst*sizeof(double));
	cudaMalloc((void **) &time_d,Nst*sizeof(double));
	cudaMalloc((void **) &dt_d,Nst*sizeof(double));
	cudaMalloc((void **) &delta_d,Nst*sizeof(long long));
	
	cudaMemcpy(n1_d, n1_h, Nst*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(n2_d, n2_h, Nst*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(N_d, N_h, Nst*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Msun_d, Msun_h, Nst*sizeof(double4), cudaMemcpyHostToDevice);
	cudaMemcpy(Spinsun_d, Spinsun_h, Nst*sizeof(double4), cudaMemcpyHostToDevice);
	cudaMemcpy(idt_d, idt_h, Nst*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(ict_d, ict_h, Nst*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Rcut_d, Rcut_h, Nst*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(RcutSun_d, RcutSun_h, Nst*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(time_d, time_h, Nst*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dt_d, dt_h, Nst*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(delta_d, delta_h, Nst*sizeof(long long), cudaMemcpyHostToDevice);
}

//************************************************
//This function prints the parametes on screen and into the infofiles
//Authors: Simon Grimm
//January 2017
//**************************************************
__host__ void Host::Info(){
	FILE *infofile;
	
	for(int st = 0; st < Nst; ++st){
		GSF[st].logfile = fopen(GSF[st].logfilename, "a");
		if(P.dev > devCount){
			P.dev = P.dev % devCount;
			fprintf(GSF[st].logfile,"selected device not allowed; changed to %d", P.dev);
		}
		
		for(int i = 0; i < 2; ++i){
			if(i == 1){
				infofile = stdout;
				if(Nst > 1) break;
			}
			else infofile = GSF[st].logfile;
			fprintf(infofile, "\n ******** Simulation path %s ********\n\n", GSF[st].path);
			fprintf(infofile, "Genga Version: %g\n", def_Version);
			fprintf(infofile, "Mercurial Branch: %s\n", HG_BRANCH);
			fprintf(infofile, "Mercurial Commit: %s\n", HG_COMMIT);
			fprintf(infofile, "Build Date: %s\n", BUILD_DATE);
			fprintf(infofile, "Build Path: %s\n", BUILD_PATH);
			fprintf(infofile, "Build System: %s\n", BUILD_SYSTEM);
			fprintf(infofile, "Build Compute Capability: SM=%s\n", BUILD_SM);
			fprintf(infofile, "Serial Grouping: %d\n", SERIAL_GROUPING);
			fprintf(infofile, "Compute Poincare Section: %d\n", poincareFlag);
			fprintf(infofile, "FormatS: %d\n", P.FormatS);						// use only argument in simulation 0
			fprintf(infofile, "FormatT: %d\n", P.FormatT);						// use only argument in simulation 0
			fprintf(infofile, "FormatP: %d\n", P.FormatP);						// use only argument in simulation 0
			fprintf(infofile, "FormatO: %d\n", P.FormatO);						// use only argument in simulation 0
			fprintf(infofile, "NmaxM: %d\n", NmaxM);
			fprintf(infofile, "Time step in days: %g \n", idt_h[st]);
			fprintf(infofile, "Output name: %s\n", GSF[st].X);
			fprintf(infofile, "Energy output interval: %d\n", P.ei);				// use only argument in simulation 0
			fprintf(infofile, "Coordinates output interval: %d\n", P.ci);				// use only argument in simulation 0
			fprintf(infofile, "Number of outputs per interval: %d\n", P.nci);			// use only argument in simulation 0
			fprintf(infofile, "Coordinate output buffer: %d\n", P.Buffer);				// use only argument in simulation 0
			fprintf(infofile, "Use Irregular outputs: %d\n", P.IrregularOutputs);			// use only argument in simulation 0
			fprintf(infofile, "Irregular output calendar: %s\n", P.IrregularOutputsfilename);	// use only argument in simulation 0
			fprintf(infofile, "Use Transits: %d\n", P.UseTransits);					// use only argument in simulation 0
			fprintf(infofile, "Use RV: %d\n", P.UseRV);						// use only argument in simulation 0
			fprintf(infofile, "TTV file name: %s\n", P.Transitsfilename);				// use only argument in simulation 0
			fprintf(infofile, "RV file name: %s\n", P.RVfilename);					// use only argument in simulation 0
			fprintf(infofile, "Print Transits: %d\n", P.PrintTransits);				// use only argument in simulation 0
			fprintf(infofile, "Print RV: %d\n", P.PrintRV);						// use only argument in simulation 0
			fprintf(infofile, "Print MCMC: %d\n", P.PrintMCMC);					// use only argument in simulation 0
			fprintf(infofile, "MCMC NE: %d\n", P.mcmcNE);						// use only argument in simulation 0
			fprintf(infofile, "MCMC Restart: %d\n", P.mcmcRestart);					// use only argument in simulation 0
			fprintf(infofile, "Integration steps: %lld\n", delta_h[st]);
			fprintf(infofile, "Central Mass: %g\n", Msun_h[st].x);
			fprintf(infofile, "Star Radius: %g\n", Msun_h[st].y);
			fprintf(infofile, "Star Love Number: %g\n", Msun_h[st].z);
			fprintf(infofile, "Star fluid Love Number: %g\n", Msun_h[st].w);
			fprintf(infofile, "Star spin_x: %g\n", Spinsun_h[st].x);
			fprintf(infofile, "Star spin_y: %g\n", Spinsun_h[st].y);
			fprintf(infofile, "Star spin_z: %g\n", Spinsun_h[st].z);
			fprintf(infofile, "Star tau: %g\n", Spinsun_h[st].w);
			fprintf(infofile, "Solar Constant: %g\n", P.SolarConstant);				// use only argument in simulation 0
			fprintf(infofile, "n1: %g\n", n1_h[st]);
			fprintf(infofile, "n2: %g\n", n2_h[st]);
			#if G3 > 0
			fprintf(infofile, "G3Limit: %g\n", G3Limit);
			fprintf(infofile, "G3Limit2: %g\n", G3Limit2);
			#endif
			fprintf(infofile, "Input file: %s\n", GSF[st].Originputfilename);
			fprintf(infofile, "Input file format: ");
			for(int f = 0; f < def_Ninformat; ++f){
				if(GSF[st].informat[f] == 1) fprintf(infofile, "x ");
				else if(GSF[st].informat[f] == 2) fprintf(infofile, "y ");
				else if(GSF[st].informat[f] == 3) fprintf(infofile, "z ");
				else if(GSF[st].informat[f] == 4) fprintf(infofile, "m ");
				else if(GSF[st].informat[f] == 5) fprintf(infofile, "vx ");
				else if(GSF[st].informat[f] == 6) fprintf(infofile, "vy ");
				else if(GSF[st].informat[f] == 7) fprintf(infofile, "vz ");
				else if(GSF[st].informat[f] == 8) fprintf(infofile, "r ");
				else if(GSF[st].informat[f] == 9) fprintf(infofile, "rho ");
				else if(GSF[st].informat[f] == 10) fprintf(infofile, "Sx ");
				else if(GSF[st].informat[f] == 11) fprintf(infofile, "Sy ");
				else if(GSF[st].informat[f] == 12) fprintf(infofile, "Sz ");
				else if(GSF[st].informat[f] == 13) fprintf(infofile, "i ");
				else if(GSF[st].informat[f] == 14) fprintf(infofile, "- ");
				else if(GSF[st].informat[f] == 15) fprintf(infofile, "amin ");
				else if(GSF[st].informat[f] == 16) fprintf(infofile, "amax ");
				else if(GSF[st].informat[f] == 17) fprintf(infofile, "emin ");
				else if(GSF[st].informat[f] == 18) fprintf(infofile, "emax ");
				else if(GSF[st].informat[f] == 19) fprintf(infofile, "t ");
				else if(GSF[st].informat[f] == 20) fprintf(infofile, "k2 ");
				else if(GSF[st].informat[f] == 21) fprintf(infofile, "k2f ");
				else if(GSF[st].informat[f] == 22) fprintf(infofile, "tau ");
				else if(GSF[st].informat[f] == 23) fprintf(infofile, "a ");
				else if(GSF[st].informat[f] == 24) fprintf(infofile, "e ");
				else if(GSF[st].informat[f] == 25) fprintf(infofile, "inc ");
				else if(GSF[st].informat[f] == 26) fprintf(infofile, "O ");
				else if(GSF[st].informat[f] == 27) fprintf(infofile, "w ");
				else if(GSF[st].informat[f] == 28) fprintf(infofile, "M ");
				else if(GSF[st].informat[f] == 29) fprintf(infofile, "aL ");
				else if(GSF[st].informat[f] == 30) fprintf(infofile, "eL ");
				else if(GSF[st].informat[f] == 31) fprintf(infofile, "incL ");
				else if(GSF[st].informat[f] == 32) fprintf(infofile, "mL ");
				else if(GSF[st].informat[f] == 33) fprintf(infofile, "OL ");
				else if(GSF[st].informat[f] == 34) fprintf(infofile, "wL ");
				else if(GSF[st].informat[f] == 35) fprintf(infofile, "ML ");
				else if(GSF[st].informat[f] == 36) fprintf(infofile, "rL ");
				else if(GSF[st].informat[f] == 37) fprintf(infofile, "saT ");
				else if(GSF[st].informat[f] == 38) fprintf(infofile, "P ");
				else if(GSF[st].informat[f] == 39) fprintf(infofile, "PL ");
				else if(GSF[st].informat[f] == 40) fprintf(infofile, "T ");
				else if(GSF[st].informat[f] == 41) fprintf(infofile, "TL ");
				else if(GSF[st].informat[f] == 42) fprintf(infofile, "Rc "); //critical radius
				else if(GSF[st].informat[f] == 43) fprintf(infofile, "gw "); //gamma w in MCMC
				else if(GSF[st].informat[f] == 0) break;
			}
			fprintf(infofile, "\n");
			fprintf(infofile, "Angle units: %d\n", P.AngleUnits);
			fprintf(infofile, "Default rho: %g\n", rho[st]);
			fprintf(infofile, "Device number: %d\n", P.dev);                           // use only argument in simulation 0
			fprintf(infofile, "Inner truncation radius: %g\n", RcutSun_h[st]);
			fprintf(infofile, "Outer truncation radius: %g\n", Rcut_h[st]);
			fprintf(infofile, "MaxColl: %d\n", def_MaxColl);
			fprintf(infofile, "pc: %g\n", def_pc);
			fprintf(infofile, "cef: %g\n", def_cef);
			fprintf(infofile, "Number of bodies: %d\n", N_h[st]);
			fprintf(infofile, "Number of test particles: %d\n", Nsmall_h[st]);
			fprintf(infofile, "Minimal number of bodies: %d\n", Nmin[st].x);
			fprintf(infofile, "Minimal number of test particles: %d\n", Nmin[st].y);
			fprintf(infofile, "Test Particle Mode: %d\n", P.UseTestParticles);              // use only argument in simulation 0
			fprintf(infofile, "Particle Minimum Mass : %g\n", P.MinMass);			// use only argument in simulation 0
			fprintf(infofile, "Symplectic recursion Max levels : %d\n", def_SLevelsMax);	// use only argument in simulation 0
			fprintf(infofile, "Symplectic recursion levels : %d\n", P.SLevels);		// use only argument in simulation 0
			fprintf(infofile, "Symplectic recursion sub steps : %d\n", P.SLSteps);		// use only argument in simulation 0
			fprintf(infofile, "Restart time step: %lld\n", P.tRestart);                     // use only argument in simulation 0
			fprintf(infofile, "Order of Symplectic integrator: %d\n", P.SIO);               // use only argument in simulation 0
			fprintf(infofile, "Maximum encounter pairs: %d\n", P.NencMax); 	                // use only argument in simulation 0
			fprintf(infofile, "Nfragments: %d\n", def_Nfragments);
			fprintf(infofile, "Use aeGrid: %d\n", P.UseaeGrid);                           	// use only argument in simulation 0
			fprintf(infofile, "aeGrid amin: %f\n", Gridae.amin);                            // use only argument in simulation 0
			fprintf(infofile, "aeGrid amax: %f\n", Gridae.amax);                            // use only argument in simulation 0
			fprintf(infofile, "aeGrid emin: %f\n", Gridae.emin);                            // use only argument in simulation 0
			fprintf(infofile, "aeGrid emax: %f\n", Gridae.emax);                            // use only argument in simulation 0
			fprintf(infofile, "aeGrid imin: %f\n", Gridae.imin);                            // use only argument in simulation 0
			fprintf(infofile, "aeGrid imax: %f\n", Gridae.imax);                            // use only argument in simulation 0
			fprintf(infofile, "aeGrid Na: %d\n", Gridae.Na);                                // use only argument in simulation 0
			fprintf(infofile, "aeGrid Ne: %d\n", Gridae.Ne);                                // use only argument in simulation 0
			fprintf(infofile, "aeGrid Ni: %d\n", Gridae.Ne);                                // use only argument in simulation 0
			fprintf(infofile, "aeGrid Count Start: %lld\n", Gridae.Start);                  // use only argument in simulation 0
			fprintf(infofile, "aeGrid name: %s\n", Gridae.X);                               // use only argument in simulation 0
			fprintf(infofile, "Use gas disk: %d\n", P.Usegas);				// use only argument in simulation 0
			fprintf(infofile, "Use gas disk enhancement: %d\n", P.UsegasEnhance);		// use only argument in simulation 0
			fprintf(infofile, "Gas dTau_diss: %g\n", P.G_dTau_diss);                        // use only argument in simulation 0
			fprintf(infofile, "Gas alpha: %d\n", P.G_alpha);                                // use only argument in simulation 0
			fprintf(infofile, "Gas Sigma_10: %g\n", P.G_Sigma_10 / (1.49598*1.49598/1.98892*1.0e-7));// use only argument in simulation 0
			fprintf(infofile, "Use force: %d\n", P.UseForce);				// use only argument in simulation 0
			fprintf(infofile, "Use Yarkovsky: %d\n", P.UseYarkovsky);			// use only argument in simulation 0
			fprintf(infofile, "Use Poynting-Robertson: %d\n", P.UsePR);			// use only argument in simulation 0
			fprintf(infofile, "Radiation Pressure Coefficient Qpr: %g\n", P.Qpr);		// use only argument in simulation 0
			fprintf(infofile, "Use Small Collisions: %d\n", P.UseSmallCollisions);		// use only argument in simulation 0
			fprintf(infofile, "Use Set Elemets function: %d\n", P.setElements);		// use only argument in simulation 0
			fprintf(infofile, "Set Elements file name: %s\n", P.setElementsfilename);	// use only argument in simulation 0
			fprintf(infofile, "Gas file name: %s\n", P.Gasfilename);			// use only argument in simulation 0
			fprintf(infofile, "Report Encounters: %d\n", P.WriteEncounters);		// use only argument in simulation 0
			fprintf(infofile, "Report Encounters Radius: %g\n", P.WriteEncountersRadius);	// use only argument in simulation 0
			fprintf(infofile, "Stop at close Encounters: %d\n", P.StopAtEncounter);
			fprintf(infofile, "Stop at close Encounter Radius: %g\n", P.StopAtEncounterRadius);
			fprintf(infofile, "Stop at Collision: %d\n", P.StopAtCollision);
			fprintf(infofile, "Stop collision minimum mass: %g\n", P.StopMinMass);
			fprintf(infofile, "Collision precision: %g\n", P.CollisionPrecision);
			fprintf(infofile, "Collision Time Shift: %g\n", P.CollTshift);
			fprintf(infofile, "Asteroid density: %g\n", Asteroid_rho);
			fprintf(infofile, "Asteroid specific heat capacity: %g\n", Asteroid_C);
			fprintf(infofile, "Asteroid albedo: %g\n", Asteroid_A);
			fprintf(infofile, "Asteroid thermal conductivity: %g\n", Asteroid_K);
			fprintf(infofile, "Asteroid collisional velocity V: %g\n", Asteroid_V);
			fprintf(infofile, "Runtime Version: %d\n", runtimeVersion);
			fprintf(infofile, "Driver Version: %d\n", driverVersion);
		}
		fclose(GSF[st].logfile);
		if(MTFlag == 1) break;
	}
}


// **************************************
//This function determines the starting points of the individual simulations
//Authors: Simon Grimm, Joachim Stadel
//March 2014
// ******************************************
__host__ void Host::Tsizes(){
	NBS_h = (int*)malloc(Nst*sizeof(int));
	NsmallS_h = (int*)malloc(Nst*sizeof(int));
	NEnergy = (int*)malloc(Nst*sizeof(int));
	
	cudaMalloc((void **) &NBS_d, Nst*sizeof(int));
	
	for(int st = 0; st < Nst; ++st){
		NBS_h[st] = NT;
		NsmallS_h[st] = NsmallT;
		NEnergy[st] = NEnergyT;
		NT += N_h[st];
		NsmallT += Nsmall_h[st];
		NBNencT += NB[st] * NmaxM;
		NEnergyT += max(N_h[st], 8);
	}
	
	NconstT = NT + NsmallT + def_Nfragments;
	if(Nst == 1){
		NBNencT = NconstT * P.NencMax;
		NEnergyT = max(NconstT, 8);
	}
}

// **************************************
// This function reads the irregular output times and stores them in IrrOutputs
// Authors: Simon Grimm
// June 2015
// ******************************************
__host__ int Host::readIrregularOutputs(){
	
	FILE *Irrfile;
	Irrfile = fopen(P.IrregularOutputsfilename, "r");
	if(Irrfile == NULL){
		printf("Error: Irregular output file not found: %s\n", P.IrregularOutputsfilename);		
		fprintf(masterfile, "Error: Irregular output file not found: %s\n", P.IrregularOutputsfilename);		
		return 0;
	}
	
	//determine the lengh of the file
	double t;
	int er;
	int n = 0;
	for(int i = 0; i < 100000000; ++i){
		er = fscanf(Irrfile, "%lf", &t);
		if(er <= 0){
			n = i;
			break;
		}
	}
	fclose(Irrfile);
	Irrfile = fopen(P.IrregularOutputsfilename, "r");
	
	IrrOutputs = (double*)malloc(n * sizeof(double));
	for(int i = 0; i < n; ++i){
		er = fscanf(Irrfile, "%lf", &IrrOutputs[i]);
		IrrOutputs[i] *= 365.25;
		if(er <= 0){
			n = i;
			break;
		}
	}
	NIrrOutputs = n;
	
	return 1;
}

// **************************************
// This function reads the transit times 
// Authors: Simon Grimm
// April 2017
// ******************************************
__host__ int Host::readTransits(){
	
	FILE *Transitfile;
	Transitfile = fopen(P.Transitsfilename, "r");
	if(Transitfile == NULL){
		printf("Error: TTV file not found: %s\n", P.Transitsfilename);		
		fprintf(masterfile, "Error: TTV file not found: %s\n", P.Transitsfilename);		
		return 0;
	}
	//determine the length of the file
	int t;
	double t1, t2;
	int er;
	int n = 0;
	//read header: Epoch and Period
	for(int i = 0; i < 1000000; ++i){
		er = fscanf(Transitfile, "%d", &t);
		er = fscanf(Transitfile, "%lf", &t1);
		er = fscanf(Transitfile, "%lf", &t2);
//printf("file a %d %d %d %g %g\n", i, er, t, t1, t2); 
		if(er <= 0){
			n += i;
			break;
		}
	}
	//read *
	char skip[160];
	er = fscanf(Transitfile, "%s", skip);
	er = fscanf(Transitfile, "%s", skip);
	er = fscanf(Transitfile, "%s", skip);
	//read Transit times
	for(int i = 0; i < 1000000; ++i){
		er = fscanf(Transitfile, "%d", &t);
		er = fscanf(Transitfile, "%lf", &t1);
		er = fscanf(Transitfile, "%lf", &t2);
//printf("file b %d %d %d %g %g\n", i, er, t, t1, t2); 
		if(er <= 0){
			n += i;
			break;
		}
	}
	++n;
	fclose(Transitfile);
	Transitfile = fopen(P.Transitsfilename, "r");
	
	for(int i = 0; i < NconstT; ++i){
		NtransitsTObs_h[i] = 0;
		
	}
	for(int i = 0; i < def_NtransitTimeMax * NconstT; ++i){
		TransitTimeObs_h[i].x = 0.0;
		TransitTimeObs_h[i].y = 1.0;
	}
	//read header: Epoch and Period
	for(int i = 0; i < n; ++i){
		int index;
		double T0, P;
		er = fscanf(Transitfile, "%d", &index);
		er = fscanf(Transitfile, "%lf", &P);
		er = fscanf(Transitfile, "%lf", &T0);
//printf("file c %d %d %d %g %g\n", i, er, index, T0, P); 
		if(er <= 0){
			n += i;
			break;
		}
		TransitTimeObs_h[index * def_NtransitTimeMax + 0].x = T0;
		TransitTimeObs_h[index * def_NtransitTimeMax + 0].y = P;
	}
	//read *
	er = fscanf(Transitfile, "%s", skip);
	er = fscanf(Transitfile, "%s", skip);
	er = fscanf(Transitfile, "%s", skip);

	int index = -1;
	int indexOld = -1;
	double T = -1E10;	
	double TOld = -1E10;
	int Epoch;
	for(int i = 0; i < n; ++i){

		TOld = T;
		indexOld = index;
		double error;
		er = fscanf(Transitfile, "%d", &index);
		er = fscanf(Transitfile, "%lf", &T);
		er = fscanf(Transitfile, "%lf", &error);

		if(index > indexOld){
			Epoch = 0;
		}
		
		if(er <= 0){
			n = i;
			break;
		}
		double T0 = TransitTimeObs_h[index * def_NtransitTimeMax].x;
		double P = TransitTimeObs_h[index * def_NtransitTimeMax].y;

		int dEpoch = 0;
		if(Epoch > 0){
			dEpoch = (T - TOld + 0.5 * P) / P;
			Epoch += dEpoch;
		}
		if(Epoch == 0){
			Epoch = (T - T0 + 0.5 * P) / P;

		}
	
		TransitTimeObs_h[index * def_NtransitTimeMax + Epoch + 1].x = T; //time
		TransitTimeObs_h[index * def_NtransitTimeMax + Epoch + 1].y = error; //error
		NtransitsTObs_h[index] = max(NtransitsTObs_h[index], Epoch);
//printf("A %d %d %g %g %d %d\n", index, i, T, TOld, dEpoch, Epoch); 
//printf("read NTobs %d %d | %d %d %.20g\n", index, NtransitsTObs_h[index], Epoch,  index * def_NtransitTimeMax + Epoch + 1, TransitTimeObs_h[index * def_NtransitTimeMax + Epoch + 1].x);
		
	}
	fclose(Transitfile);
	//copy first sub-simulation to all the others
	for(int st = 1; st < Nst; ++st){
		for(int i = 0; i < N_h[0]; ++i){
			//assume that all sub simulations are of equal size
			NtransitsTObs_h[i + st * N_h[0]] = NtransitsTObs_h[i];
			for(int Epoch = 0; Epoch <= NtransitsTObs_h[i] + 1; ++Epoch){
				TransitTimeObs_h[(i + st * N_h[0]) * def_NtransitTimeMax + Epoch] = TransitTimeObs_h[i * def_NtransitTimeMax + Epoch];
			}
		}
	}
	cudaMemcpy(TransitTimeObs_d, TransitTimeObs_h, def_NtransitTimeMax * NconstT * sizeof(double2), cudaMemcpyHostToDevice);
	cudaMemcpy(NtransitsTObs_d, NtransitsTObs_h, NconstT * sizeof(int), cudaMemcpyHostToDevice);
	
	return 1;
}

// **************************************
// This function reads the RV data 
// Authors: Simon Grimm
// November 2019
// ******************************************
__host__ int Host::readRV(){
	
	FILE *RVfile;
	RVfile = fopen(P.RVfilename, "r");
	if(RVfile == NULL){
		printf("Error: RV file not found: %s\n", P.RVfilename);		
		fprintf(masterfile, "Error: RV file not found: %s\n", P.RVfilename);		
		return 0;
	}
	//determine the lengh of the file
	double time;
	double t1, t2;
	int er;
	int n = 0;

	//read RV data
	for(int i = 0; i < 1000000; ++i){
		er = fscanf(RVfile, "%lf", &time);
		er = fscanf(RVfile, "%lf", &t1);
		er = fscanf(RVfile, "%lf", &t2);
//printf("file b %d %d %d %g %g\n", i, er, t, t1, t2); 
		if(er <= 0){
			n += i;
			break;
		}
	}
	fclose(RVfile);
	RVfile = fopen(P.RVfilename, "r");
	
	for(int i = 0; i < Nst; ++i){
		NRVTObs_h[i] = 0;
		
	}
	for(int i = 0; i < def_NRVMax * Nst; ++i){
		RVObs_h[i].x = 0.0;
		RVObs_h[i].y = 0.0;
		RVObs_h[i].z = 1.0;
	}

	//read RV data
	for(int i = 0; i < n; ++i){
		double T, error;
		er = fscanf(RVfile, "%lf", &time);
		er = fscanf(RVfile, "%lf", &T);
		er = fscanf(RVfile, "%lf", &error);
		
		if(er <= 0){
			n = i;
			break;
		}

		RVObs_h[i].x = time; //RV
		RVObs_h[i].y = T; //RV
		RVObs_h[i].z = error; //error
printf("read RV %d %.20g %g %g\n", i, time, T, error);
		
	}
	NRVTObs_h[0] = n;
	fclose(RVfile);

	//copy first sub-simulation to all the others
	for(int st = 1; st < Nst; ++st){
		//assume that all sub simulations are of equal size
		NRVTObs_h[st] = NRVTObs_h[0];
		for(int i = 0; i < n; ++i){
			RVObs_h[st * def_NRVMax + i] = RVObs_h[i];
		}
	}

	cudaMemcpy(RVObs_d, RVObs_h, def_NRVMax * Nst * sizeof(double3), cudaMemcpyHostToDevice);
	cudaMemcpy(NRVTObs_d, NRVTObs_h, Nst * sizeof(int), cudaMemcpyHostToDevice);
	
	return 1;
}


// **************************************
// This function reads the Set Elements file with the Kepler elements
// Authors: Simon Grimm
// June 2015
// ******************************************
__host__ int Host::readSetElements(){
	
	FILE *Efile;
	Efile = fopen(P.setElementsfilename, "r");
	if(Efile == NULL){
		printf("Error: Set Elements file not found: %s\n", P.setElementsfilename);		
		fprintf(masterfile, "Error: Set Elements file not found: %s\n", P.setElementsfilename);		
		return 0;
	}
	
	int Elements[25];
	for(int i = 0; i < 25; ++i){
		Elements[i] = 0;
	}
	
	//read the number of planets
	P.setElementsN = 1;
	int er = fscanf(Efile, "%d", &P.setElementsN);
	if(er <= 0) return 0;
	
	int nelements = 0;
	char sp[64];
	int useKeplerElements = 0;
	int useXYZ = 0;
	//determine the specified elements
	for(int i = 0; i < 25; ++i){
		//m r a e i W w M are set after the drift
		// x y z vy vy vz before the drift
		fscanf (Efile, "%s", sp);
		
		if(strcmp(sp, "t") == 0){	
			Elements[i] = 1;
			printf("t ");
			++nelements;
		}
		else if(strcmp(sp, "j") == 0){
			//index
			Elements[i] = 2;
			printf("j ");
			++nelements;
		}
		else if(strcmp(sp, "a") == 0){	
			Elements[i] = 3;
			printf("a ");
			++nelements;
			P.setElements = 2;
			useKeplerElements = 1;
		}
		else if(strcmp(sp, "e") == 0){	
			Elements[i] = 4;
			printf("e ");
			++nelements;
			P.setElements = 2;
			useKeplerElements = 1;
		}
		else if(strcmp(sp, "i") == 0){	
			Elements[i] = 5;
			printf("i ");
			++nelements;
			P.setElements = 2;
			useKeplerElements = 1;
		}
		else if(strcmp(sp, "O") == 0){	
			Elements[i] = 6;
			printf("O ");
			++nelements;
			P.setElements = 2;
			useKeplerElements = 1;
		}
		else if(strcmp(sp, "w") == 0){	
			Elements[i] = 7;
			printf("w ");
			++nelements;
			P.setElements = 2;
			useKeplerElements = 1;
		}
		else if(strcmp(sp, "m") == 0){	
			Elements[i] = 8;
			printf("m ");
			++nelements;
			P.setElements = 2;
		}
		else if(strcmp(sp, "r") == 0){	
			Elements[i] = 9;
			printf("r ");
			++nelements;
			P.setElements = 2;
		}
		else if(strcmp(sp, "T") == 0){	
			Elements[i] = 10;
			printf("T ");
			++nelements;
			P.setElements = 2;
			useKeplerElements = 1;
		}
		else if(strcmp(sp, "x") == 0){	
			Elements[i] = 11;
			printf("x ");
			++nelements;
			useXYZ = 1;
		}
		else if(strcmp(sp, "y") == 0){	
			Elements[i] = 12;
			printf("y ");
			++nelements;
			useXYZ = 1;
		}
		else if(strcmp(sp, "z") == 0){	
			Elements[i] = 13;
			printf("z ");
			++nelements;
			useXYZ = 1;
		}
		else if(strcmp(sp, "-") == 0){	
			Elements[i] = 14;
			printf("- ");
			++nelements;
		}
		else if(strcmp(sp, "vx") == 0){		//heliocentric velocities	
			Elements[i] = 15;
			printf("vx ");
			++nelements;
			P.setElementsV = 2;
			useXYZ = 1;
		}
		else if(strcmp(sp, "vy") == 0){	
			Elements[i] = 16;
			printf("vy ");
			++nelements;
			P.setElementsV = 2;
			useXYZ = 1;
		}
		else if(strcmp(sp, "vz") == 0){	
			Elements[i] = 17;
			printf("vz ");
			++nelements;
			P.setElementsV = 2;
			useXYZ = 1;
		}
		else if(strcmp(sp, "vxb") == 0){	//barycentric velocities
			Elements[i] = 18;
			printf("vxb ");
			++nelements;
			P.setElementsV = 3;
			useXYZ = 1;
		}
		else if(strcmp(sp, "vyb") == 0){	
			Elements[i] = 19;
			printf("vyb ");
			++nelements;
			P.setElementsV = 3;
			useXYZ = 1;
		}
		else if(strcmp(sp, "vzb") == 0){	
			Elements[i] = 20;
			printf("vzb ");
			++nelements;
			P.setElementsV = 3;
			useXYZ = 1;
		}
		else{
			printf("\n");
			break;
		}
		
	}
	if(useXYZ == 1 && useKeplerElements == 0 && P.setElements == 2){
		P.setElements = 1;
	} 
	er = 0;
	if(Elements[0] != 1) er = 1;
	if(er == 1){
		printf("Error: time is missing in Set Elements file\n");
		return 0;
	}
	er = 1;
	fclose(Efile);
	//determine the lenght of the file
	Efile = fopen(P.setElementsfilename, "r");
	//skip header
	double t;
	fscanf(Efile, "%lf", &t);
	for(int i = 0; i < nelements; ++i){
		char c[64];
		er = fscanf(Efile, "%s", c);
	}
	int nlines = 0;
	int nlinesToSkip = 0;
	double time = -2.0e10;
	double timeOld = -1.0e10;
	double timeOld2 = time;
	//start time
	double time0 = ict_h[0] + P.tRestart * idt_h[0] / 365.25;
	//end time
	double time1 = ict_h[0] + P.deltaT * idt_h[0] / 365.25;
	for(int j = 0; j < 1000000; ++j){
		for(int i = 0; i < nelements; ++i){
			er = fscanf(Efile, "%lf", &t);
			if(er <= 0) break;
			//find starting time of the simulation
			if(Elements[i] == 1){
				if(j % P.setElementsN == 0){
					timeOld2 = timeOld;
					timeOld = time;
				}
				time = t;
//printf("time %.10g %.10g start time %.10g| end time %.10g | %d %d\n", time, timeOld2, time0, time1, nlinesToSkip, nlines);

				//cubic interpolation
				if(j < P.setElementsN  && time > time0){
					printf("Error, set Elements start time smaller than time in datafile\n");
					return 0;
				}
			}

		}
		if(er <= 0){
			break;
		}
		if((time >= ict_h[0] && timeOld2 <= time1) || nlines < 4){	//need at least 4 lines for cubic interpolation
			++nlines;
		}
		if(time < ict_h[0] && j >= P.setElementsN){
			++nlinesToSkip;
		}
	}
	if(nlines < 4 * P.setElementsN){
		printf("Error, set Elements less than 4 data points, need at least 4\n");
		return 0;
	}
	//cubic interpolation
	if(time < time1){
		printf("Error, set Elements end time larger than time in datafile: %g %g\n", time1, time);
		return 0;
	}

	fclose(Efile);
	printf("%d lines, %d linesToSkip, %d bodies, %d elements\n", nlines, nlinesToSkip, P.setElementsN, nelements);
	
	constantCopy3(Elements, nelements, P.setElementsN, nlines);
	//allocate memory
	setElementsData_h = (double*)malloc(nelements * nlines * sizeof(double));	
	cudaMalloc((void **) &setElementsData_d, nelements * nlines * sizeof(double));

	cudaError_t error = cudaGetLastError();
	if(error != 0){
		printf("read set elements error = %d = %s\n",error, cudaGetErrorString(error));
		fprintf(masterfile, "read set elements error = %d = %s\n",error, cudaGetErrorString(error));
		return 0;
	}


	Efile = fopen(P.setElementsfilename, "r");
	//read file	
	//skip header and linesToSkip
	fscanf(Efile, "%lf", &t);
	for(int j = 0; j < nlinesToSkip + 1; ++j){
		for(int i = 0; i < nelements; ++i){
			char c[64];
			er = fscanf(Efile, "%s", c);
			if(Elements[i] == 1){
//printf("skip time %d %s start time %g\n", j, c, ict_h[0]);
			}

		}
	}

	for(int j = 0; j < nlines; ++j){
		for(int i = 0; i < nelements; ++i){
			er = fscanf(Efile, "%lf", &setElementsData_h[j * nelements + i]);
			if(Elements[i] == 1){
//printf("read time %d %g start time %g\n", j, setElementsData_h[j * nelements + i], ict_h[0]);
			}
		}
	}
	cudaMemcpy(setElementsData_d, setElementsData_h, nelements * nlines * sizeof(double), cudaMemcpyHostToDevice);
	error = cudaGetLastError();
	if(error != 0){
		printf("read set elements error = %d = %s\n",error, cudaGetErrorString(error));
		fprintf(masterfile, "read set elements error = %d = %s\n",error, cudaGetErrorString(error));
		return 0;
	}
	
	cudaMalloc((void **) &setElementsLine_d, sizeof(int));
	cudaMemset(setElementsLine_d, 0, sizeof(int));

	return 1;
}

// **************************************
// This function reads the Gas disk file
// Authors: Simon Grimm
// July 2016
// ******************************************
__host__ int Host::readGasFile(){
	
	FILE *Efile;
	Efile = fopen(P.Gasfilename, "r");
	if(Efile == NULL){
		printf("Error: Gas file not found: %s\n", P.Gasfilename);		
		fprintf(masterfile, "Error: Gas file not found: %s\n", P.Gasfilename);		
		return 0;
	}
	int er = 0;
	//The elements are time, r, Sigma and h
	//read time 0
	double t0, r0, Sigma0, h0;
	double t1, r1, Sigma1, h1;
	er = fscanf(Efile, "%lf", &t0);
	er = fscanf(Efile, "%lf", &r0);
	er = fscanf(Efile, "%lf", &Sigma0);
	er = fscanf(Efile, "%lf", &h0);
	
	//determine the number of values in r
	int nr;
	for(nr = 1; nr < 10000; ++nr){
		er = fscanf(Efile, "%lf", &t1);
		er = fscanf(Efile, "%lf", &r1);
		er = fscanf(Efile, "%lf", &Sigma1);
		er = fscanf(Efile, "%lf", &h1);
		if(t1 > t0) break;
		if(er <= 0){
			printf("Error: Gas file not correct: %s\n", P.Gasfilename);		
			fprintf(masterfile, "Error: Gas file not correct: %s\n", P.Gasfilename);		
			return 0;
		}
	}
	printf("nr %d\n", nr);
	
	fclose(Efile);
	
	//allocate memory
	GasData_h = (double4*)malloc(nr *  sizeof(double4));	//2 time steps and 2 values
	cudaMalloc((void **) &GasData_d, nr * sizeof(double4));
	Efile = fopen(P.Gasfilename, "r");
	
	double skip;
	//read data0
	for(int i = 0; i < nr; ++i){
		er = fscanf(Efile, "%lf", &skip);
		er = fscanf(Efile, "%lf", &skip);
		er = fscanf(Efile, "%lf", &GasData_h[i].x);
		er = fscanf(Efile, "%lf", &GasData_h[i].y);
	}
	//read data1
	for(int i = 0; i < nr; ++i){
		er = fscanf(Efile, "%lf", &skip);
		er = fscanf(Efile, "%lf", &skip);
		er = fscanf(Efile, "%lf", &GasData_h[i].z);
		er = fscanf(Efile, "%lf", &GasData_h[i].w);
	}
	cudaMemcpy(GasData_d, GasData_h, nr * sizeof(double4), cudaMemcpyHostToDevice);
	GasDatanr = nr;
	GasDatatime.x = t0;
	GasDatatime.y = t1;
	fclose(Efile);
	printf("Read Gas file OK\n");
	return 1;
}
// **************************************
// This function reads the next time step of the Gas File
// Authors: Simon Grimm
// July 2016
// ******************************************
__host__ int Host::readGasFile2(double time){
	
	FILE *Efile;
	Efile = fopen(P.Gasfilename, "r");
	int nr = GasDatanr;
	//The elements are time, r, Sigma and h
	//read time 0
	double t, r, t0, t1;
	int er = 0;
	//determine the number of values in r
	for(int j = 0; j < 10000; ++j){
		for(int i = 0; i < nr; ++i){
			er = fscanf(Efile, "%lf", &t);
			er = fscanf(Efile, "%lf", &r);
			if(t < time){
				t0 = t;
				er = fscanf(Efile, "%lf", &GasData_h[i].x);
				er = fscanf(Efile, "%lf", &GasData_h[i].y);
			}
			else{
				t1 = t;
				er = fscanf(Efile, "%lf", &GasData_h[i].z);
				er = fscanf(Efile, "%lf", &GasData_h[i].w);
			}
		}
		if(t > time){
			printf("Gas Data line %d t0 %.20g t1 %.20g \n", j * nr, t0, t1);
			break;
		}
		if(er <= 0){
			printf("Error: Gas file not correct: %s\n", P.Gasfilename);		
			return 0;
		}
	}
	
	fclose(Efile);
	cudaMemcpy(GasData_d, GasData_h, nr * sizeof(double4), cudaMemcpyHostToDevice);
	GasDatatime.x = t0;
	GasDatatime.y = t1;
	return 1;
}


__host__ int Host::freeHost(){
	cudaError_t error;
	free(NB);
	free(Nmin);
	free(rho);
	
	free(n1_h);
	free(n2_h);
	free(N_h);
	free(Nsmall_h);
	free(Msun_h);
	free(Spinsun_h);
	free(idt_h);
	free(ict_h);
	free(Rcut_h);
	free(RcutSun_h);
	free(time_h);
	free(dt_h);
	free(delta_h);
	
	cudaFree(n1_d);
	cudaFree(n2_d);
	cudaFree(N_d);
	cudaFree(Nsmall_d);
	cudaFree(Msun_d);
	cudaFree(Spinsun_d);
	cudaFree(idt_d);
	cudaFree(ict_d);
	cudaFree(Rcut_d);
	cudaFree(RcutSun_d);
	cudaFree(time_d);
	cudaFree(dt_d);
	cudaFree(delta_d);
	
	error = cudaGetLastError();
	if(error != 0){
		printf("Cuda Orbit free error = %d = %s\n",error, cudaGetErrorString(error));
		fprintf(masterfile, "Cuda Orbit free error = %d = %s\n",error, cudaGetErrorString(error));
		return 0;
	}
	return 1;
	
	
}

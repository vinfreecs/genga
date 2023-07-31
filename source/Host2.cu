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
			int er = fscanf(lockfile, "%lld", &R);
			if(er <= 0){
				printf("Error when reading lockfile\n");
			}
			fclose(lockfile);
			
			if(R == Restart) Lock = 1;
			
			lockfile = fopen(lockfilename, "w");
			fprintf(lockfile, "%lld\n", Restart);
			fclose(lockfile);
			
		}
	}
	
	#if def_IgnoreLockFile == 1
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

	Nomp = 1;
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
			for(int j = 0; j < 1000000; ++j){
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

#if def_CPU == 0	
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
		fprintf(masterfile,"Name: %s, Major:%d, Minor:%d, Max threads per Block:%d, Max x dim:%d, #Multiprocessors:%d, Can Map Memory:%d, Clock Rate:%d, Memory Clock Rate:%d, Can Overlap:%d, Concurrent Kernels:%d, regsPerBlock:%d, sharedMemPerBlock:%zu, warp size:%d\n",  
			devProp.name, devProp.major, devProp.minor, devProp.maxThreadsPerBlock, devProp.maxThreadsDim[0], devProp.multiProcessorCount, devProp.canMapHostMemory,devProp.clockRate, devProp.memoryClockRate, devProp.deviceOverlap, devProp.concurrentKernels, devProp.regsPerBlock, devProp.sharedMemPerBlock, devProp.warpSize);
		if(!devProp.canMapHostMemory) {
			fprintf(masterfile, "Device %d cannot map host memory!\n", i);
			return 0;
		}
	}
#else
	devCount = 1;
#endif
	return 1;
}



//E =  1: input
//E = -1: output
__host__ int Host::assignInformat(char *ff, int &format, int E){
	int cartesian = 0;
	int keplerian = 0;
	int check = 0;

	for(int i = 0; i < def_Ninformat; ++i){
		if(strcmp(ff, fileFormat[i]) == 0){
			format = i;
			check = 1;
			if(i == 1 || i == 2 || i == 3 || i == 5 || i == 6 || i == 7){
				cartesian = 1;
			}
			if(i == 23 || i == 24 || i == 25 || i == 26 || i == 27 || i == 28){
				keplerian = 1;
			}
		}
	}
	
	if(check == 0){
		if(strcmp(ff, ">>") == 0){
			return 2;
		}
		else if(strcmp(ff, "<<") == 0){
		}
		else {
			printf("Error: Input or output format not valid! Maybe the spaces in << ... >> have been forgotten\n");
			return 1;
		}

	}
	if(cartesian == 1 && keplerian == 1){
		printf("Error: Input or output file format is not valid! Cartesian and Keplerian coordinates can not be mixed.\n");
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
	
	for(int i = 0; i < 32; ++i){
		P.dev[i] = i;
	}
	P.ndev = 1;
	GSF = (struct GSFiles*)malloc(Nst*sizeof(struct GSFiles));
	
	n1_h = (double*)malloc(Nst*sizeof(double));
	n2_h = (double*)malloc(Nst*sizeof(double));
	N_h = (int*)malloc(Nst*sizeof(int));
	Nsmall_h = (int*)malloc(Nst*sizeof(int));
	Msun_h = (double2*)malloc(Nst*sizeof(double2));
	Spinsun_h = (double4*)malloc(Nst*sizeof(double4));
	Lovesun_h = (double3*)malloc(Nst*sizeof(double3));
	J2_h = (double2*)malloc(Nst*sizeof(double2));
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
	P.Nfragments = def_Nfragments;
	P.SLevels = def_SLevels;
	P.SLSteps = def_SLSteps;
	P.AngleUnits = def_AngleUnits;
	P.OutBinary = def_OutBinary;
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
	P.UsegasPotential = def_UsegasPotential;
	P.UsegasEnhance = def_UsegasEnhance;
	P.UsegasDrag = def_UsegasDrag;
	P.UsegasTidalDamping = def_UsegasTidalDamping;
	P.UseForce = def_UseForce;
	P.UseGR = def_UseGR;
	P.UseTides = def_UseTides;
	P.UseRotationalDeformation = def_UseRotationalDeformation;
	P.UseYarkovsky = def_UseYarkovsky;
	P.UseMigrationForce = def_UseMigrationForce;
	P.UseSmallCollisions = def_UseSmallCollisions;
	P.CreateParticles = def_CreateParticles;
	P.CreateParticlesN = def_CreateParticlesN;
	sprintf(P.CreateParticlesfilename, "%s", "-");
	P.UsePR = def_UsePR;
	P.Qpr = def_Qpr;
	P.SolarWind = def_SolarWind;
	P.SolarConstant = def_SolarConstant;
	P.Asteroid_eps = def_Asteroid_eps;
	P.Asteroid_rho = def_Asteroid_rho;
	P.Asteroid_C = def_Asteroid_C;
	P.Asteroid_A = def_Asteroid_A;
	P.Asteroid_K = def_Asteroid_K;
	P.Asteroid_V = def_Asteroid_V;
	P.Asteroid_rmin = def_Asteroid_rmin;
	P.Asteroid_rdel = def_Asteroid_rdel;
	P.G_dTau_diss = def_GasdTau_diss;
	P.G_rg0 = def_GasRg0;
	P.G_rg1 = def_GasRg1;
	P.G_rp1 = def_GasRp1;
	P.G_drg = def_GasDrg;
	P.G_alpha = def_GasAlpha;
	P.G_beta = def_GasBeta;
	P.G_Sigma_10 = def_G_Sigma_10 * 1.49598*1.49598/1.98892*1.0e-7;
	P.G_Mgiant = def_Mgiant;
	P.FormatS = def_FormatS;
	P.FormatT = def_FormatT;
	P.FormatP = def_FormatP;
	P.FormatO = def_FormatO;
	P.WriteEncounters = def_WriteEncounters;
	P.WriteEncountersRadius = def_WriteEncountersRadius;
	P.WriteEncountersCloudSize = def_WriteEncountersCloudSize;
	P.StopAtEncounter = def_StopAtEncounter;
	P.StopAtEncounterRadius = def_StopAtEncounterRadius;
	P.StopAtCollision = def_StopAtCollision;
	P.StopMinMass = def_StopMinMass;
	P.CollisionPrecision = def_CollisionPrecision;
	P.CollTshift = def_CollTshift;
	P.CollisionModel = def_CollisionModel;
	P.NAFvars = def_NAFvars;
	P.NAFn0 = def_NAFn0;
	P.NAFnfreqs = def_NAFnfreqs;
	P.NAFformat = def_NAFformat;
	P.NAFinterval = def_NAFinterval;
	
	P.UseJ2 = 0.0;
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

	P.SERIAL_GROUPING = def_SERIAL_GROUPING;
	P.doTuning = def_doTuning;
	P.doSLTuning = def_doSLTuning;
	P.KickFloat = def_KickFloat;
	

	for(int i = 0; i < def_Ninformat; ++i){
		sprintf(fileFormat[i], "%s", "_");
	}

	//parameters must be less than 5 characters long
	sprintf(fileFormat[ 1], "%s", "x");
	sprintf(fileFormat[ 2], "%s", "y");
	sprintf(fileFormat[ 3], "%s", "z");
	sprintf(fileFormat[ 4], "%s", "m");
	sprintf(fileFormat[ 5], "%s", "vx");
	sprintf(fileFormat[ 6], "%s", "vy");
	sprintf(fileFormat[ 7], "%s", "vz");
	sprintf(fileFormat[ 8], "%s", "r");
	sprintf(fileFormat[ 9], "%s", "rho");
	sprintf(fileFormat[10], "%s", "Sx");
	sprintf(fileFormat[11], "%s", "Sy");
	sprintf(fileFormat[12], "%s", "Sz");
	sprintf(fileFormat[13], "%s", "i");
	sprintf(fileFormat[14], "%s", "-");
	sprintf(fileFormat[15], "%s", "amin");	//aelimits
	sprintf(fileFormat[16], "%s", "amax");
	sprintf(fileFormat[17], "%s", "emin");
	sprintf(fileFormat[18], "%s", "emax");
	sprintf(fileFormat[19], "%s", "t");
	sprintf(fileFormat[20], "%s", "k2");
	sprintf(fileFormat[21], "%s", "k2f");
	sprintf(fileFormat[22], "%s", "tau");
	sprintf(fileFormat[23], "%s", "a");
	sprintf(fileFormat[24], "%s", "e");
	sprintf(fileFormat[25], "%s", "inc");
	sprintf(fileFormat[26], "%s", "O");
	sprintf(fileFormat[27], "%s", "w");
	sprintf(fileFormat[28], "%s", "M");
	sprintf(fileFormat[29], "%s", "aL");
	sprintf(fileFormat[30], "%s", "eL");
	sprintf(fileFormat[31], "%s", "incL");
	sprintf(fileFormat[32], "%s", "mL");
	sprintf(fileFormat[33], "%s", "OL");
	sprintf(fileFormat[34], "%s", "wL");
	sprintf(fileFormat[35], "%s", "ML");
	sprintf(fileFormat[36], "%s", "rL");
	sprintf(fileFormat[37], "%s", "saT");
	sprintf(fileFormat[38], "%s", "P");
	sprintf(fileFormat[39], "%s", "PL");
	sprintf(fileFormat[40], "%s", "T");
	sprintf(fileFormat[41], "%s", "TL");
	sprintf(fileFormat[42], "%s", "Rc");	//Rcrit
	sprintf(fileFormat[43], "%s", "gw");	//gamma w
	sprintf(fileFormat[44], "%s", "Ic");	//Moment of Inertia
	sprintf(fileFormat[45], "%s", "test");
	sprintf(fileFormat[46], "%s", "encc");	//enccountT
	sprintf(fileFormat[47], "%s", "aec");	//aecount
	sprintf(fileFormat[48], "%s", "aecT");	//aecountT
	sprintf(fileFormat[49], "%s", "mig");	//artificial migration time scale
	sprintf(fileFormat[50], "%s", "mige");	//artificial migration time scale e
	sprintf(fileFormat[51], "%s", "migi");	//artifitial migration time scale i

	char iformat[def_Ninformat * 5];
	sprintf(iformat, def_InputFileFormat);
	char oformat[def_Ninformat * 5];
	sprintf(oformat, def_OutputFileFormat);
	
	for(int st = 0; st < Nst; ++st){
		for(int i = 0; i < def_Ninformat; ++i){
			GSF[st].informat[i] = 0;
		}

		int pos = 0;		
		for(int f = -1; f < def_Ninformat; ++f){
			char ff[5];
			int n = 0;
			int er = sscanf(iformat + pos, "%s%n", ff, &n);
			if(er <= 0) break;

			pos += n;

			er = assignInformat(ff, GSF[st].informat[f], 1);
			if(er == 2) break;

		}
	}
	for(int st = 0; st < Nst; ++st){
		for(int i = 0; i < def_Ninformat; ++i){
			GSF[st].outformat[i] = 0;
		}

		int pos = 0;		
		for(int f = -1; f < def_Ninformat; ++f){
			char ff[5];
			int n = 0;
			int er = sscanf(oformat + pos, "%s%n", ff, &n);
			if(er <= 0) break;

			pos += n;

			er = assignInformat(ff, GSF[st].outformat[f], -1);
			if(er == 2) break;

		}
	}
	for(int st = 0; st < Nst; ++st){
		n1_h[st] = def_n1;
		n2_h[st] = def_n2;
		N_h[st] = 32;
		Nsmall_h[st] = 0;
		Msun_h[st].x = def_CentralMass;
		Msun_h[st].y = def_CentralRadius;
		Spinsun_h[st].x = def_StarSpinx;
		Spinsun_h[st].y = def_StarSpiny;
		Spinsun_h[st].z = def_StarSpinz;
		Spinsun_h[st].w = def_StarIc;
		Lovesun_h[st].x = def_StarK2;
		Lovesun_h[st].y = def_StarK2f;
		Lovesun_h[st].z = def_StarTau;
		J2_h[st].x = def_J2;
		J2_h[st].y = def_J2R;
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
				char t[120];
				int er = fscanf(pathfile, "%s", t);
				if(er <= 0){
					printf("Error when reading pathfile %s\n", pathfilename);
				}
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
	
	for(int j = 0; j < 1000; ++j){ //loop around all lines in the param.dat file
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

		//continue statements are used because there would be too many 'eles if' branches
		if(strcmp(sp, "Time step in days =") == 0){
			er = fscanf (paramfile, "%lf", &idt_h[st]);
			if(er <= 0){
				printf("Error: time step is not valid!\n");
				return 0;
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Output name =") == 0){
			er = fscanf (paramfile, "%s", GSF[st].X);
			if(er <= 0){
				printf("Error: Output name is not valid!\n");	
				return 0;
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
 		}
		if(strcmp(sp, "Energy output interval =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Coordinates output interval =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Number of outputs per interval =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Irregular output calendar =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "TTV file name =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "RV file name =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "TTV steps =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.TransitSteps);
				
				if(er <= 0){
					printf("Error: TTV Steps is not valid!\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Print Transits =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.PrintTransits);
				
				if(er <= 0){
					printf("Error: Print Transits is not valid!\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Print RV =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.PrintRV);
				
				if(er <= 0){
					printf("Error: Print RV is not valid!\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Print MCMC =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.PrintMCMC);
				
				if(er <= 0){
					printf("Error: Print MCMC is not valid!\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "MCMC NE =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.mcmcNE);
				
				if(er <= 0){
					printf("Error: MCMC NE is not valid!\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "MCMC Restart =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.mcmcRestart);
				
				if(er <= 0){
					printf("Error: MCMC Restart is not valid!\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Integration steps =") == 0){
			er = fscanf (paramfile, "%lld", &delta_h[st]);
			
			if(er <= 0 || delta_h[st] <= 0){
				printf("Error: Inegration steps are not valid!\n");
				return 0;
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			{
				if(st == 0){
					P.deltaT = delta_h[st];
				}
				else{
					//avoid max for long long int
					if(delta_h[st] > P.deltaT){
						P.deltaT = delta_h[st];
					}
				}
				continue;
			}
		}
		if(strcmp(sp, "Central Mass =") == 0){
			
			er = fscanf (paramfile, "%lf", &Msun_h[st].x);
			
			if(er <= 0){
				printf("Error: Central mass is not valid!\n");
				return 0;
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Star Radius =") == 0){
			
			er = fscanf (paramfile, "%lf", &Msun_h[st].y);
			
			if(er <= 0){
				printf("Error: Star Raius is not valid!\n");
				return 0;
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Star Love Number =") == 0){
			
			er = fscanf (paramfile, "%lf", &Lovesun_h[st].x);
			
			if(er <= 0){
				printf("Error: Star Love Number is not valid!\n");
				return 0;
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Star fluid Love Number =") == 0){
			
			er = fscanf (paramfile, "%lf", &Lovesun_h[st].y);
			
			if(er <= 0){
				printf("Error: Star fluid Love Number is not valid!\n");
				return 0;
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Star tau =") == 0){
			
			er = fscanf (paramfile, "%lf", &Lovesun_h[st].z);
			
			if(er <= 0){
				printf("Error: Star tau is not valid!\n");
				return 0;
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Star spin_x =") == 0){
			
			er = fscanf (paramfile, "%lf", &Spinsun_h[st].x);
			
			if(er <= 0){
				printf("Error: Star spin_x is not valid!\n");
				return 0;
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Star spin_y =") == 0){
			
			er = fscanf (paramfile, "%lf", &Spinsun_h[st].y);
			
			if(er <= 0){
				printf("Error: Star spin_y is not valid!\n");
				return 0;
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Star spin_z =") == 0){
			
			er = fscanf (paramfile, "%lf", &Spinsun_h[st].z);
			
			if(er <= 0){
				printf("Error: Star spin_z is not valid!\n");
				return 0;
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Star Ic =") == 0){
			
			er = fscanf (paramfile, "%lf", &Spinsun_h[st].w);
			
			if(er <= 0){
				printf("Error: Star Ic is not valid!\n");
				return 0;
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "J2 =") == 0){
			
			er = fscanf (paramfile, "%lf", &J2_h[st].x);
			
			if(er <= 0){
				printf("Error: J2 is not valid!\n");
				return 0;
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "J2 radius =") == 0){
			
			er = fscanf (paramfile, "%lf", &J2_h[st].y);
			
			if(er <= 0){
				printf("Error: J2 radius is not valid!\n");
				return 0;
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Solar Constant =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "n1 =") == 0){
			er = fscanf (paramfile, "%lf", &n1_h[st]);
			
			if(er <= 0){
				printf("Error: n1 is not valid!\n");
				return 0;
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "n2 =") == 0){
			
			er = fscanf (paramfile, "%lf", &n2_h[st]);
			if(er <= 0){
				printf("Error: n2 is not valid!\n");
				return 0;
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Input file =") == 0){
			er = fscanf (paramfile, "%s", GSF[st].inputfilename);
			
			if(er <= 0){
				printf("Error: Input file name is not valid!\n");
				return 0;
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Input file Format:") == 0){
			for(int i = 0; i < def_Ninformat; ++i){
				GSF[st].informat[i] = 0;
			}
			//Read input file Format
			int f;
			for(f = -1; f < def_Ninformat; ++f){
				er = fscanf (paramfile, "%s", sp);
	

				int er2 = assignInformat(sp, GSF[st].informat[f], 1);
				if(er2 == 2) break;
				if(er2 == 1) return 0;

			}
			if(er <= 0){
				printf("Error: Input file format is not valid!\n");
				return 0;
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Output file Format:") == 0){
			for(int i = 0; i < def_Ninformat; ++i){
				GSF[st].outformat[i] = 0;
			}
			//Read output file Format
			int f;
			for(f = -1; f < def_Ninformat; ++f){
				er = fscanf (paramfile, "%s", sp);
	

				int er2 = assignInformat(sp, GSF[st].outformat[f], -1);
				if(er2 == 2) break;
				if(er2 == 1) return 0;

			}
			if(er <= 0){
				printf("Error: Output file format is not valid!\n");
				return 0;
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Angle units =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Use output binary files =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.OutBinary);
				if(er <= 0 || P.OutBinary < 0 || P.OutBinary > 1){
					printf("Error: Use output binary files not valid\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Default rho =") == 0){
			er = fscanf (paramfile, "%lf", &rho[st]);
			if(er <= 0 ){
				printf("Error: Default value for rho is not valid!\n");
				return 0;
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Use Test Particles =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.UseTestParticles);
				if(er <= 0 || P.UseTestParticles < 0 || P.UseTestParticles > 2){
					printf("Error: Test Particle Mode not valid\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Particle Minimum Mass =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Restart timestep =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Minimum number of bodies =") == 0){
			er = fscanf (paramfile, "%d", &Nmin[st].x);
			if(er <= 0 || Nmin[st].x < 0){
				printf("Error: Minimal number of bodies not valid\n");
				return 0;
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Minimum number of test particles =") == 0){
			er = fscanf (paramfile, "%d", &Nmin[st].y);
			if(er <= 0 || Nmin[st].y < 0){
				printf("Error: Minimal number of test particles not valid\n");
				return 0;
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Inner truncation radius =") == 0){
			er = fscanf (paramfile, "%lf", &RcutSun_h[st]);
			if(er <= 0 || RcutSun_h[st] < 0){
				printf("Error: Inner truncation radius not valid\n");
				return 0;
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Outer truncation radius =") == 0){
			er = fscanf (paramfile, "%lf", &Rcut_h[st]);
			if(er <= 0 || Rcut_h[st] < 0){
				printf("Error: Outer truncation radius not valid\n");
				return 0;
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Order of integrator =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Use aeGrid =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "aeGrid amin =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "aeGrid amax =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "aeGrid emin =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "aeGrid emax =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "aeGrid imin =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "aeGrid imax =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "aeGrid Na =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "aeGrid Ne =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "aeGrid Ni =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "aeGrid Start Count =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "aeGrid name =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Use gas disk =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Use gas disk potential =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.UsegasPotential);
				if(er <= 0){
					printf("Error: Use gas disk potential value is not valid!\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Use gas disk enhancement =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Use gas disk drag =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.UsegasDrag);
				if(er <= 0){
					printf("Error: Use gas disk drag value is not valid!\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Use gas disk tidal damping =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.UsegasTidalDamping);
				if(er <= 0){
					printf("Error: Use gas disk tidal damping value is not valid!\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Gas dTau_diss =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Gas disk inner edge =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%lf", &P.G_rg0);
				if(er <= 0){
					printf("Error: Gas disk inner edge value is not valid!\n");
					return 0;
				}
				
			}
			else{
				double t;
				er = fscanf (paramfile, "%lf", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Gas disk outer edge =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%lf", &P.G_rg1);
				if(er <= 0){
					printf("Error: Gas disk outer edge value is not valid!\n");
					return 0;
				}
				
			}
			else{
				double t;
				er = fscanf (paramfile, "%lf", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Gas disk grid outer edge =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%lf", &P.G_rp1);
				if(er <= 0){
					printf("Error: Gas disk grid outer edge value is not valid!\n");
					return 0;
				}
				
			}
			else{
				double t;
				er = fscanf (paramfile, "%lf", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Gas disk grid dr =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%lf", &P.G_drg);
				if(er <= 0){
					printf("Error: Gas disk grid dr value is not valid!\n");
					return 0;
				}
				
			}
			else{
				double t;
				er = fscanf (paramfile, "%lf", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Gas alpha =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%lf", &P.G_alpha);
				if(er <= 0){
					printf("Error: Gas alpha value is not valid!\n");
					return 0;
				}
				
			}
			else{
				double t;
				er = fscanf (paramfile, "%lf", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Gas beta =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%lf", &P.G_beta);
				if(er <= 0 || P.G_beta < 0.0){
					printf("Error: Gas beta value is not valid!\n");
					return 0;
				}
				
			}
			else{
				double t;
				er = fscanf (paramfile, "%lf", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Gas Sigma_10 =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Gas Mgiant =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%lf", &P.G_Mgiant);
				if(er <= 0 || P.G_Mgiant < 0){
					printf("Error: Gas Mgiant value is not valid!\n");
					return 0;
				}
				
			}
			else{
				double t;
				er = fscanf (paramfile, "%lf", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Use force =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Use GR =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.UseGR);
				if(er <= 0){
					printf("Error: Use GR value is not valid!\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Use Tides =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.UseTides);
				if(er <= 0){
					printf("Error: Use Tides value is not valid!\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Use Rotational Deformation =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.UseRotationalDeformation);
				if(er <= 0){
					printf("Error: Use Rotational Deformation value is not valid!\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Use Yarkovsky =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Use Small Collisions =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Create Particles file name =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%s", P.CreateParticlesfilename);
				if(er <= 0){
					printf("Error: Create Particles file name value is not valid!\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Use Poynting-Robertson =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Radiation Pressure Coefficient Qpr =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Solar Wind factor =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%lf", &P.SolarWind);
				if(er <= 0){
					printf("Error: Solar Wind factor value is not valid!\n");
					return 0;
				}
			}
			else{
				double t;
				er = fscanf (paramfile, "%lf", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Use Migration Force =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.UseMigrationForce);
				if(er <= 0){
					printf("Error: Use Migration Force value is not valid!\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Asteroid emissivity eps =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%lf", &P.Asteroid_eps);
				if(er <= 0){
					printf("Error: Asteroid eps value is not valid!\n");
					return 0;
				}
			}
			else{
				double t;
				er = fscanf (paramfile, "%lf", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Asteroid density rho =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%lf", &P.Asteroid_rho);
				if(er <= 0){
					printf("Error: Asteroid rho value is not valid!\n");
					return 0;
				}
			}
			else{
				double t;
				er = fscanf (paramfile, "%lf", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Asteroid specific heat capacity C =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%lf", &P.Asteroid_C);
				if(er <= 0){
					printf("Error: Asteroid C value is not valid!\n");
					return 0;
				}
			}
			else{
				double t;
				er = fscanf (paramfile, "%lf", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Asteroid albedo A =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%lf", &P.Asteroid_A);
				if(er <= 0){
					printf("Error: Asteroid A value is not valid!\n");
					return 0;
				}
			}
			else{
				double t;
				er = fscanf (paramfile, "%lf", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Asteroid thermal conductivity K =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%lf", &P.Asteroid_K);
				if(er <= 0){
					printf("Error: Asteroid K value is not valid!\n");
					return 0;
				}
			}
			else{
				double t;
				er = fscanf (paramfile, "%lf", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Asteroid collisional velocity V =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%lf", &P.Asteroid_V);
				if(er <= 0){
					printf("Error: Asteroid V value is not valid!\n");
					return 0;
				}
			}
			else{
				double t;
				er = fscanf (paramfile, "%lf", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Asteroid minimal fragment radius =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%lf", &P.Asteroid_rmin);
				if(er <= 0){
					printf("Error: Asteroid minimal fragment radius value is not valid!\n");
					return 0;
				}
			}
			else{
				double t;
				er = fscanf (paramfile, "%lf", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Asteroid fragment remove radius =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%lf", &P.Asteroid_rdel);
				if(er <= 0){
					printf("Error: Asteroid fragment remove radius value is not valid!\n");
					return 0;
				}
			}
			else{
				double t;
				er = fscanf (paramfile, "%lf", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "FormatS =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "FormatT =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "FormatP =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "FormatO =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Report Encounters =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Report Encounters Radius =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Report Encounters Cloud Size =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.WriteEncountersCloudSize);
				if(er <= 0){
					printf("Error: Report Encounters Cloud Size value is not valid!\n");
					return 0;
				}
				
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Stop at Encounter =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Stop at Encounter Radius =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%lf", &P.StopAtEncounterRadius);
				if(er <= 0){
					printf("Error: Stop at Encounter Radius value is not valid!\n");
					return 0;
				}
				
			}
			else{
				double t;
				er = fscanf (paramfile, "%lf", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Stop at Collision =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Stop Minimum Mass =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Collision Precision =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Collision Time Shift =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Collision Model =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.CollisionModel);
				if(er <= 0){
					printf("Error: Collison Model value is not valid!\n");
					return 0;
				}
				
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Coordinate output buffer =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Set Elements file name =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Gas file name =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "NAF variables =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "NAF size =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "NAF nfreqs =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "NAF format =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "NAF interval =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Maximum encounter pairs =") == 0){
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
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Nfragments =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.Nfragments);	
				if(er <= 0){
					printf("Error: Nfragments = is not valid!\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Symplectic recursion levels =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.SLevels);
				if(P.SLevels > def_SLevelsMax){
					printf("Error, Symplectic recursion levels larger than def_SLevelsMax %d %d\n", P.SLevels, def_SLevelsMax);
					P.SLevels = def_SLevelsMax;
				}
				if(er <= 0 || P.SLevels == 0){
					printf("Error: Symplectic recursion levels = is not valid!\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Symplectic recursion sub steps =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.SLSteps);
				if(er <= 0 || P.SLSteps <= 0){
					printf("Error: Symplectic recursion sub steps = is not valid!\n");
					return 0;
				}
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Serial Grouping =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.SERIAL_GROUPING);
				if(er <= 0){
					printf("Error: Serial Grouping value is not valid!\n");
					return 0;
				}
				
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Do kernel tuning =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.doTuning);
				if(er <= 0){
					printf("Error: Do kernel tuning value is not valid!\n");
					return 0;
				}
				
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
		if(strcmp(sp, "Do Kick in single precision =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.KickFloat);
				if(er <= 0){
					printf("Error: Use Kick float value is not valid!\n");
					return 0;
				}
				
			}
			else{
				int t;
				er = fscanf (paramfile, "%d", &t);
			}
			if(fgets(sp, 3, paramfile) != nullptr)
			continue;
		}
	
		printf("Undefined line in param.dat file: line %d\n", j);
		return 0;
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
		else if(strcmp(argv[i], "-ndev") == 0){
			P.ndev = atoi(argv[i + 1]);
		}
		else if(strcmp(argv[i], "-dev") == 0){
			for(int j = 0; j < P.ndev; ++j){
				P.dev[j] = atoi(argv[i + 1]);
				++i;
			}
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
		else if(strcmp(argv[i], "-collPrec") == 0){
			P.CollisionPrecision = atof(argv[i + 1]);
		}
		else if(strcmp(argv[i], "-collTshift") == 0){
			P.CollTshift = atof(argv[i + 1]);
		}
		else if(strcmp(argv[i], "-Nomp") == 0){
			Nomp = atoi(argv[i + 1]);
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
	if(strcmp(P.CreateParticlesfilename, "-") != 0){
		P.CreateParticles = 1;
	}
	if(P.Usegas == 1 && P.G_dTau_diss <= 0.0){
		printf("Error: dTau_diss value is not valid!\n");
		return 0;
	}
	if(P.UseTestParticles == 0){
		P.MinMass = 0.0;
		Nmin[st].y = 0;
		if(P.WriteEncounters == 2){
			P.WriteEncounters = 1;
		}
	}
	
	
	if(P.StopAtCollision != 0 && Nst > 1){
		printf("Error: Stop at Collision not available in multi simulation mode!\n");
		return 0;
	}
	
	if(P.CollTshift < 1.0){
		printf("Error: Collision Time Shift not valid! %g\n", P.CollTshift);
		return 0;
	}
	
	if(P.CollisionPrecision == 0.0){
		printf("Error: Collision Precision not valid! %g\n", P.CollisionPrecision);
		return 0;
	}
	if(fabs(P.CollisionPrecision) <= 1.0e-10){
		printf("Error: Collision Precision too small! %g  Limit is at 1.0e-10\n", P.CollisionPrecision);
		return 0;
	}

	if(P.SLevels == -1){
		//use atomatic SLevels tuning
		P.SLevels = def_SLevelsMax;
		P.doSLTuning = 1;
	}

	if(P.SLevels > def_SLevelsMax){
		printf("Error, Symplectic recursion levels bigger than def_SLevelsMax %d %d\n", P.SLevels, def_SLevelsMax);
		return 0;
	}
	if(Nst > 1 && (P.SLevels > 1 || P.SLSteps > 2)){
		printf("Error, Symplectic recursion levels are not supported in the multi simulation mode\n");
		return 0;
	}

	if(P.FormatS == 1 && P.FormatP == 0){
		P.FormatS = 0;
	}

	if(log10(delta_h[st]) >= def_NFileNameDigits && P.FormatO == 0){
		printf("Error, number of time steps larger than number of digits in the output filenames. Increase def_NFileNameDigits in the define.h file or use P.FormatO = 1.\n");
		return 0;

	}
	if(P.ci > 0 && log10(delta_h[st] / P.ci) >= def_NFileNameDigits && P.FormatO == 1){
		printf("Error, number of time steps larger than number of digits in the output filenames. Increase def_NFileNameDigits in the define.h file.\n");
		return 0;

	}

	if(P.G_alpha > 1.0 && P.UsegasPotential > 0){
		printf("Error, Gas alpha value not valid for UsegasPotential. Code needs to be updated\n");
		return 0;
	}

	if(P.UseSmallCollisions > 0 && USE_RANDOM != 1){
		printf("Error, Use Small Collisions must use USE_RANDOM 1\n");
		return 0;
	}
	if(P.CreateParticles > 0 && USE_RANDOM != 1){
		printf("Error, Create Particles mode must use USE_RANDOM 1\n");
		return 0;
	}
	if(def_TTV > 0 && USE_RANDOM != 1){
		printf("Error, TTV must use USE_RANDOM 1\n");
		return 0;
	}

	if(J2_h[0].x != 0.0){
		P.UseJ2 = 1;
	}

	ForceFlag = 0;
	if(P.UseForce > 0 || P.Usegas > 0 || P.UseYarkovsky > 0 || P.UsePR > 0 || P.UseGR > 0 || P.UseTides > 0 || P.UseRotationalDeformation > 0 || P.UseJ2 > 0 || P.UseMigrationForce > 0){
		ForceFlag = 1;
	}
	//set UseGR when old UseForce is used, choose Hamiltonian splitting
	if(P.UseForce & 1){
		P.UseGR = 1;
	}
	if(P.UseForce >> 1 & 1){
		P.UseTides = 1;
	}
	if(P.UseForce >> 2 & 1){
		P.UseRotationalDeformation = 1;
	}

	//Nomp is only used on CPUs
	if(Nomp <= 0) Nomp = 1;

#if def_CPU == 0
	Nomp = 1;
#else
	omp_set_num_threads(Nomp);
	printf("Nomp: %d, omp_get_num: %d\n", Nomp, omp_get_num_threads());

	#pragma omp parallel for
	for(int i = 0; i < Nomp; ++i){
		int k = omp_get_thread_num();
		int cpuid = sched_getcpu();
		printf("used cpus: %d, thread_id: %d, cpu_id: %d\n", i, k, cpuid);
	}
 
#endif

	//check output format
	{
		int check_time = 0;
		int check_m = 0;
		int check_r = 0;
		int check_i = 0;
		int check_xv = 0;
		int check_S = 0;
		
		int check_k2 = 0;
		int check_k2f = 0;
		int check_tau = 0;
		int check_Ic = 0;
		int check_mig = 0;
		int check_mige = 0;
		int check_migi = 0;

		for(int f = 0; f < def_Ninformat; ++f){
			if(GSF[st].outformat[f] == 19)	check_time = 1;
			else if(GSF[st].outformat[f] == 13)	check_i = 1;
			else if(GSF[st].outformat[f] == 4)	check_m = 1;
			else if(GSF[st].outformat[f] == 8)	check_r = 1;
			else if(GSF[st].outformat[f] == 1)	check_xv += 1;
			else if(GSF[st].outformat[f] == 2)	check_xv += 2;
			else if(GSF[st].outformat[f] == 3)	check_xv += 4;
			else if(GSF[st].outformat[f] == 5)	check_xv += 8;
			else if(GSF[st].outformat[f] == 6)	check_xv += 16;
			else if(GSF[st].outformat[f] == 7)	check_xv += 32;
			else if(GSF[st].outformat[f] == 10)	check_S += 1;
			else if(GSF[st].outformat[f] == 11)	check_S += 2;
			else if(GSF[st].outformat[f] == 12)	check_S += 4;
			else if(GSF[st].outformat[f] == 20)	check_k2 = 1;
			else if(GSF[st].outformat[f] == 21)	check_k2f = 1;
			else if(GSF[st].outformat[f] == 22)	check_tau = 1;
			else if(GSF[st].outformat[f] == 44)	check_Ic = 1;
			else if(GSF[st].outformat[f] == 49)	check_mig = 1;
			else if(GSF[st].outformat[f] == 50)	check_mige = 1;
			else if(GSF[st].outformat[f] == 51)	check_migi = 1;

			else if(GSF[st].outformat[f] == 15) ;	//amin	
			else if(GSF[st].outformat[f] == 16) ;	//amax
			else if(GSF[st].outformat[f] == 17) ;	//emin	
			else if(GSF[st].outformat[f] == 18) ;	//emax	
			else if(GSF[st].outformat[f] == 47) ;	//aecount
			else if(GSF[st].outformat[f] == 48) ;	//aecountT
			else if(GSF[st].outformat[f] == 46) ;	//enccount
			else if(GSF[st].outformat[f] == 42) ;	//Rc
			else if(GSF[st].outformat[f] == 44) ;	//Ic
			else if(GSF[st].outformat[f] == 45) ;	//test
			else if(GSF[st].outformat[f] > 0){
				printf("Error, Output file format not valid: %s\n", fileFormat[GSF[st].outformat[f]]);
				return 0;
			}
		}

		//check if parameters are given in the initial conditions
		//otherwise these are constant for all particles and does not need to be stored
		int check_k2_in = 0;
		int check_k2f_in = 0;
		int check_tau_in = 0;
		int check_Ic_in = 0;
		int check_mig_in = 0;
		int check_mige_in = 0;
		int check_migi_in = 0;
		for(int f = 0; f < def_Ninformat; ++f){
			if(GSF[st].informat[f] == 20)	check_k2_in = 1;
			if(GSF[st].informat[f] == 21)	check_k2f_in = 1;
			if(GSF[st].informat[f] == 22)	check_tau_in = 1;
			if(GSF[st].informat[f] == 44)	check_Ic_in = 1;
			if(GSF[st].informat[f] == 49)	check_mig_in = 1;
			if(GSF[st].informat[f] == 50)	check_mige_in = 1;
			if(GSF[st].informat[f] == 51)	check_migi_in = 1;
		}

		if(check_time == 0){
			printf("Error, Output file format not valid, 't' is not included, needed for restart\n");
			return 0;
		}
		if(check_m == 0){
			printf("Error, Output file format not valid, 'm' is not included, needed for restart\n");
			return 0;
		}
		if(check_r == 0){
			printf("Error, Output file format not valid, 'r' is not included, needed for restart\n");
			return 0;
		}
		if(check_i == 0){
			printf("Error, Output file format not valid, 'i' is not included, needed for restart\n");
			return 0;
		}
		if(check_xv != 63){
			printf("Error, Output file format is not complete. Must include x, y, z, vx, vy, vz, needed for restart\n");
			return 0;
		}
		if(check_S != 7){
			printf("Error, Output file format is not complete. Must include Sx, Sy, Sz, needed for restart\n");
			return 0;
		}
		if(check_k2 == 0 && check_k2_in == 1){
			printf("Error, Output file format not valid, 'k2' is not included, needed for restart\n");
			return 0;
		}
		if(check_tau == 0 && check_tau_in == 1){
			printf("Error, Output file format not valid, 'tau' is not included, needed for restart\n");
			return 0;
		}
		if(check_k2f == 0 && check_k2f_in == 1){
			printf("Error, Output file format not valid, 'k2f' is not included, needed for restart\n");
			return 0;
		}
		if(check_Ic == 0 && check_Ic_in == 1){
			printf("Error, Output file format not valid, 'Ic' is not included, needed for restart\n");
			return 0;
		}
		if(check_mig == 0 && check_mig_in == 1){
			printf("Error, Output file format not valid, 'mig' is not included, needed for restart\n");
			return 0;
		}
		if(check_mige == 0 && check_mige_in == 1){
			printf("Error, Output file format not valid, 'mige' is not included, needed for restart\n");
			return 0;
		}
		if(check_migi == 0 && check_migi_in == 1){
			printf("Error, Output file format not valid, 'migi' is not included, needed for restart\n");
			return 0;
		}

	}

#if def_CPU == 0
	//check peer to peer access for multi GPU runs:
	cudaSetDevice(P.dev[0]);
	for(int i = 1; i < P.ndev; ++i){
		for(int j = 0; j < P.ndev; ++j){
			if(i != j){
				int check = 0;
				cudaDeviceCanAccessPeer(&check, P.dev[i], P.dev[j]);	//check if device i can access device j
				fprintf(masterfile, "device %d can acess device %d: %d\n", P.dev[i], P.dev[j], check);
				printf("device %d can acess device %d: %d\n", P.dev[i], P.dev[j], check);
				if(check == 0){
					fprintf(masterfile, "error: device %d can not acess device %d: %d\n", P.dev[i], P.dev[j], check);
					printf("error: device %d can not acess device %d: %d\n", P.dev[i], P.dev[j], check);
					return 0;
				}
			}
		}
	}
	for(int i = 1; i < P.ndev; ++i){
		cudaSetDevice(P.dev[i]);
		for(int j = 0; j < P.ndev; ++j){
			if(i != j){
				cudaDeviceEnablePeerAccess(P.dev[j], 0);
			}
		}
	}
#endif
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
			char tname[512];
			sprintf(tname, "%s%s", GSF[st].path, GSF[st].inputfilename);
			sprintf(GSF[st].inputfilename, "%.383s", tname);
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
			//avoid max for long long int
			if(Restart > P.tRestart){
				P.tRestart = Restart;
			}
		}
	}
	if(P.ci != -1 && P.ci != 0){
		 if(P.tRestart % P.ci == 0) RestartBackup = 0;
	}
//printf("restart %lld %d\n", P.tRestart, RestartBackup);

	char dat_bin[16];
	if(P.OutBinary == 0){
		sprintf(dat_bin, "%s", "dat");
	}
	else{
		sprintf(dat_bin, "%s", "bin");
	}

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
					sprintf(GSF[st].inputfilename, "%sOut%s_%.*lld.%s", GSF[st].path, GSF[st].X, def_NFileNameDigits, P.tRestart, dat_bin);
					if(P.FormatO == 1) sprintf(GSF[st].inputfilename, "%sOut%s_%.*lld.%s", GSF[st].path, GSF[st].X, def_NFileNameDigits, P.tRestart / scale, dat_bin);
					if(P.FormatO == 1 && RestartBackup == 1) sprintf(GSF[st].inputfilename, "%sOutbackup%s_%.20lld.%s", GSF[st].path, GSF[st].X, P.tRestart, dat_bin);
				}
				if(P.FormatT == 1) sprintf(GSF[st].inputfilename, "%sOut%s.%s", GSF[st].path, GSF[st].X, dat_bin);
			}
			else{
				if(P.FormatT == 0){
					long long scale = 1ll;
					if(P.FormatO == 1){
						scale = (long long)(P.ci);
						if(P.ci == -1) scale = P.tRestart;
					}
					sprintf(GSF[st].inputfilename, "Out%s_%.*lld.%s", GSF[st].X, def_NFileNameDigits, P.tRestart, dat_bin);
					if(P.FormatO == 1) sprintf(GSF[st].inputfilename, "Out%s_%.*lld.%s", GSF[st].X, def_NFileNameDigits, P.tRestart / scale, dat_bin);
					if(P.FormatO == 1 && RestartBackup == 1) sprintf(GSF[st].inputfilename, "Outbackup%s_%.20lld.%s", GSF[st].X, P.tRestart, dat_bin);
				}
				if(P.FormatT == 1) sprintf(GSF[st].inputfilename, "Out%s.%s", GSF[st].X, dat_bin);
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
		sprintf(GSF[st].starfilename, "%sStar%s.dat", GSF[st].path, GSF[st].X);
		sprintf(GSF[st].starIrrfilename, "%sStarIrr%s.dat", GSF[st].path, GSF[st].X);
		
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
			if(P.UseSmallCollisions > 0 || P.CreateParticles > 0){
				tfile = fopen(GSF[st].fragmentfilename, "w");
				fclose(tfile);  
			}

			if(P.UseTides > 0 || P.UseRotationalDeformation > 0){
				tfile = fopen(GSF[st].starfilename, "w");
				fclose(tfile);
			}
		}
		
		GSF[st].logfile = fopen(GSF[st].logfilename, "a");
		
		if(P.tRestart > 0) fprintf(GSF[st].logfile, "\n\n\n************** Restart Simulation at time step %lld *******************\n", P.tRestart);
		fclose(GSF[st].logfile);
	}
	return 1;
}

// ************************************** //
//This function reads 1 line of the output file, used for restarting
//Authors: Simon Grimm
//July 2022
// *****************************************
__host__ int Host::readOutLine(double &time, int &index, double4 &x, double4 &v, double4 &spin, double3 &love, double3 &migration, float4 &aelimits, double &skip, double &aecount, unsigned long long &enccountT, double &rcrit, double &test, FILE *infile, int st){

	int er = 0;
	for(int f = 0; f < def_Ninformat; ++f){
		int ff = GSF[st].outformat[f];
		if(P.OutBinary == 0){
			if(ff == 19){
				er = fscanf (infile, "%lf",&time);
				//printf("time %g\n", time);
			}
			else if(ff == 13){
				er = fscanf (infile, "%d",&index);
				//printf("index %d\n", index);
			}
			else if(ff == 4)	er = fscanf (infile, "%lf",&x.w);
			else if(ff == 8)	er = fscanf (infile, "%lf",&v.w);
			else if(ff == 1)	er = fscanf (infile, "%lf",&x.x);
			else if(ff == 2)	er = fscanf (infile, "%lf",&x.y);
			else if(ff == 3)	er = fscanf (infile, "%lf",&x.z);
			else if(ff == 5)	er = fscanf (infile, "%lf",&v.x);
			else if(ff == 6)	er = fscanf (infile, "%lf",&v.y);
			else if(ff == 7)	er = fscanf (infile, "%lf",&v.z);
			else if(ff == 10)	er = fscanf (infile, "%lf",&spin.x);
			else if(ff == 11)	er = fscanf (infile, "%lf",&spin.y);
			else if(ff == 12)	er = fscanf (infile, "%lf",&spin.z);
			else if(ff == 15)	er = fscanf (infile, "%f",&aelimits.x);
			else if(ff == 16)	er = fscanf (infile, "%f",&aelimits.y);
			else if(ff == 17)	er = fscanf (infile, "%f",&aelimits.z);
			else if(ff == 18)	er = fscanf (infile, "%f",&aelimits.w);
			else if(ff == 20)	er = fscanf (infile, "%lf",&love.x);
			else if(ff == 21)	er = fscanf (infile, "%lf",&love.y);
			else if(ff == 22)	er = fscanf (infile, "%lf",&love.z);
			else if(ff == 47)	er = fscanf (infile, "%lf",&skip);
			else if(ff == 48)	er = fscanf (infile, "%lf",&aecount);
			else if(ff == 46)	er = fscanf (infile, "%llu",&enccountT);
			else if(ff == 42)	er = fscanf (infile, "%lf",&rcrit);
			else if(ff == 44)	er = fscanf (infile, "%lf",&spin.w);
			else if(ff == 45)	er = fscanf (infile, "%lf",&test);
			else if(ff == 49)	er = fscanf (infile, "%lf",&migration.x);
			else if(ff == 50)	er = fscanf (infile, "%lf",&migration.y);
			else if(ff == 51)	er = fscanf (infile, "%lf",&migration.z);
		}
		else{
			if(f == 19)		er = fread(&time, sizeof(double), 1, infile);
			else if(ff == 13)	er = fread(&index, sizeof(int), 1, infile);
			else if(ff == 4)	er = fread(&x.w, sizeof(double), 1, infile);
			else if(ff == 8)	er = fread(&v.w, sizeof(double), 1, infile);
			else if(ff == 1)	er = fread(&x.x, sizeof(double), 1, infile);
			else if(ff == 2)	er = fread(&x.y, sizeof(double), 1, infile);
			else if(ff == 3)	er = fread(&x.z, sizeof(double), 1, infile);
			else if(ff == 5)	er = fread(&v.x, sizeof(double), 1, infile);
			else if(ff == 6)	er = fread(&v.y, sizeof(double), 1, infile);
			else if(ff == 7)	er = fread(&v.z, sizeof(double), 1, infile);
			else if(ff == 10)	er = fread(&spin.x, sizeof(double), 1, infile);
			else if(ff == 11)	er = fread(&spin.y, sizeof(double), 1, infile);
			else if(ff == 12)	er = fread(&spin.z, sizeof(double), 1, infile);
			else if(ff == 15)	er = fread(&aelimits.x, sizeof(float), 1, infile);
			else if(ff == 16)	er = fread(&aelimits.y, sizeof(float), 1, infile);
			else if(ff == 17)	er = fread(&aelimits.z, sizeof(float), 1, infile);
			else if(ff == 18)	er = fread(&aelimits.w, sizeof(float), 1, infile);
			else if(ff == 20)	er = fread(&love.x, sizeof(double), 1, infile);
			else if(ff == 21)	er = fread(&love.y, sizeof(double), 1, infile);
			else if(ff == 22)	er = fread(&love.z, sizeof(double), 1, infile);
			else if(ff == 47)	er = fread(&skip, sizeof(float), 1, infile);
			else if(ff == 48)	er = fread(&aecount, sizeof(float), 1, infile);
			else if(ff == 46)	er = fread(&enccountT, sizeof(unsigned long long), 1, infile);
			else if(ff == 42) 	er = fread(&rcrit, sizeof(double), 1, infile);
			else if(ff == 44)	er = fread(&spin.w, sizeof(double), 1, infile);
			else if(ff == 45) 	er = fread(&test, sizeof(double), 1, infile);
			else if(ff == 49)	er = fread(&migration.x, sizeof(double), 1, infile);
			else if(ff == 50)	er = fread(&migration.y, sizeof(double), 1, infile);
			else if(ff == 51)	er = fread(&migration.z, sizeof(double), 1, infile);
		}
	}
	return er;
}

// **************************************
// This function determines the starting time of the simulation using the input file 
// specified in the param file.

//Authors: Simon Grimm
//July 2022
// ************************************
__host__ int Host::icict(int st){
	double time = 0.0;
	int er = 1;
	FILE *OrigInfile;
	char Origfilename[512];
	sprintf(Origfilename, "%s%s", GSF[st].path, GSF[st].Originputfilename);
	OrigInfile = fopen(Origfilename, "r");
	char t[500];
	if(OrigInfile == NULL){
		printf("Error in Simulation %s: Input file not found %s\n", GSF[st].path, GSF[st].inputfilename);
		fprintf(masterfile, "Error in Simulation %s: Input file not found %s\n", GSF[st].path, GSF[st].inputfilename);
		return 0;
	}
	for(int f = 0; f < def_Ninformat; ++f){
		if(GSF[0].informat[f] == 19){
			er = fscanf (OrigInfile, "%lf",&time);
			break;
		}
		else if(GSF[0].informat[f] > 0){
			er = fscanf(OrigInfile, "%s", t);
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
//Authors: Simon Grimm
//July 2022
// *********************************************
__host__ int Host::icSize(int st){

//printf("Determine the size of the file %s\n", GSF[st].inputfilename);
	
	//Determine the simulation start time
	double time, test, rcrit;
	int index;
	double4 x, v;
	double4 spin;
	double3 love;
	double3 migration;
	float4 aelimits;
	double aecountf, aecountTf;
	unsigned long long enccountT;

	time = 0.0;
	if(ict_h[st] > 0.0) time = ict_h[st];
	
	int er = icict(st);
	if(er == 0) return 0;
	
	char t[500];
	er = 1;
	int NN = 0;
	int er1 = 1;
	double Et;
	if(P.OutBinary == 0){
		char Ets[160]; //exact time at restart time step, must be the same format as the coordinate output
		sprintf(Ets, "%.16g", (P.tRestart * idt_h[st] + ict_h[st] * 365.25) / 365.25);
		Et = atof(Ets);
	}
	else{
		Et = (P.tRestart * idt_h[st] + ict_h[st] * 365.25) / 365.25;
	}

	FILE *infile;
	if(P.OutBinary > 0 && P.tRestart > 0){
		infile = fopen(GSF[st].inputfilename, "rb");
	}
	else{
		infile = fopen(GSF[st].inputfilename, "r");
	}
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

			
		if(P.tRestart == 0 || P.FormatP == 0){
			for(int f = 0; f < def_Ninformat; ++f){
				if(GSF[st].informat[f] == 4){
					er = fscanf (infile, "%lf",&x.w);
				}
				else if(GSF[0].informat[f] > 0){
					er = fscanf(infile, "%s", t);
				}
			}
		}
		else{

			er = readOutLine(time, index, x, v, spin, love, migration, aelimits, aecountf, aecountTf, enccountT, rcrit, test, infile, st);
		}
		if(er <= 0){ //error by reading
			er1 = 0;
			break;
		}

		if(P.FormatT == 1 && ((time > Et && idt_h[st] > 0.0) || (time < Et && idt_h[st] < 0.0))) break;
		//if reading was succesfull, check if particles belong to the desired time 
		if(er1 == 1){
			if(P.FormatP == 1){ // All particles in one time file
				if(P.FormatS == 0 || P.tRestart == 0 || Nst == 1){
					if(Et == time){
						if(x.w > P.MinMass) ++NN;
						else ++Nsmall_h[st];
					}
				}
				else if(index / def_MaxIndex == st){
					if(Et == time){
						if(x.w > P.MinMass) ++NN;
						else ++Nsmall_h[st];
					}
				}
			}
			if(P.FormatP == 0){
				if(P.tRestart == 0){
					if(x.w > P.MinMass) ++NN;
					else ++Nsmall_h[st];
				}
				else ++NN;
			}
		}
		else break;
	}
	fclose(infile);
//	printf("icSize A: st: %d, time: %.20g, N: %d, Nsmall: %d %s\n", st, time, NN, Nsmall_h[st], GSF[st].inputfilename);
	
	if(P.FormatP == 0 && P.tRestart > 0){//Restart FormatP == 0 data
		int NNN = 0;
		int NNNsmall = 0;
		FILE *OrigInfile;
		char Origfilename[512];
		sprintf(Origfilename, "%s%s", GSF[st].path, GSF[st].Originputfilename);
		OrigInfile = fopen(Origfilename, "r");

		FILE *fragmentsfile;
		if(P.UseSmallCollisions > 0 || P.CreateParticles > 0){
			fragmentsfile = fopen(GSF[st].fragmentfilename, "r");
		}
		else{
			fragmentsfile = NULL;
		}


		for(int k = 0; k < 1000000000; ++k){
			double skip = 0.0;
			int eri = 1;
			int i = k;
			//if index is not given in the initial conditions file, i = k, otherwise scan for the index
			for(int f = 0; f < def_Ninformat; ++f){
				if(GSF[st].informat[f] == 13){
					eri = fscanf (OrigInfile, "%d",&i);
				}
				else if(GSF[st].informat[f] > 0){
					eri = fscanf (OrigInfile, "%lf",&skip);
				}
			}
			if(eri < 0){
				if(P.UseSmallCollisions > 0 || P.CreateParticles > 0){
//printf("Search for fragments %s\n", GSF[st].fragmentfilename);
					double ttime, mm;
					double skip;
					eri = fscanf(fragmentsfile, "%lf", &ttime);
					eri = fscanf(fragmentsfile, "%d", &i);
					eri = fscanf(fragmentsfile, "%lf", &mm);

					for(int jj = 0; jj < 11; ++jj){
						eri = fscanf(fragmentsfile, "%lf", &skip);
					}
					if(eri <= 0){
						break;
					}
					if(ttime > Et) continue;
					//NN contains large and small particles
					++NN;

				}
				else{
					break;
				}
			}
			
			int NMAX = 0;
			er1 = 1;
			char infilename[384];
			if(P.OutBinary == 0){
				sprintf(infilename, "%sOut%s_p%.6d.dat", GSF[st].path, GSF[st].X, i);
				infile = fopen(infilename, "r");
			}
			else{
				sprintf(infilename, "%sOut%s_p%.6d.bin", GSF[st].path, GSF[st].X, i);
				infile = fopen(infilename, "rb");
			}
//printf("Read file %d %s %d %d %d\n", k, infilename, NNN, NNNsmall, NN);
			if(infile == NULL) continue;
			for(int it = 0; it < 1000000000; ++it){
				er = readOutLine(time, index, x, v, spin, love, migration, aelimits, aecountf, aecountTf, enccountT, rcrit, test, infile, st);
//printf("%d %d %d %.20g %.20g | %g %g\n", er, i, it, time, Et, idt_h[st], ict_h[st]);

				if(er <= 0){ //error in reading
					er1 = 0;
					break;
				}
//if(st < 10 && i == 1) printf("%d %d %d %.20g %.20g | %g %g\n", st, i, it, time, Et, idt_h[st], ict_h[st]);
				//if(time > Et) break;  //uncomment because of resolution increment in restarting
				
				if(er1 == 1){
					if(Nst == 1 || P.FormatS == 0){
						if(Et == time){
							if(x.w > P.MinMass) ++NNN;
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
		if(P.UseSmallCollisions > 0 || P.CreateParticles > 0){
			fclose(fragmentsfile);
		}
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
	
	if(Nst > 1 && NN > 128){
		fprintf(masterfile,"Error in Simulation %s: More particles than 128: %d\n", GSF[st].path, NN);
		printf("Error in Simulation %s: More particles than 128: %d\n", GSF[st].path, NN);
		return 0;
	}
	if(N_h[st] + Nsmall_h[st] == 0){
		fprintf(masterfile,"Error in Simulation %s: No particles found\n", GSF[st].path);
		printf("Error in Simulation %s: No particles found\n", GSF[st].path);
		return 0;
	}

//	printf("icSize B: st: %d, N: %d, Nsmall: %d\n", st, N_h[st], Nsmall_h[st]);	
	return 1;
}

// ************************************************
//This function calls the function icSize and sets the size parameters
//Authors: Simon Grimm
//January 2017
// ***********************************************3
__host__ int Host::size(){
	NBmax = 0;
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

		//avoid max for long long int
		if(NB[st] > NBmax){
			NBmax = NB[st];
		}
//printf("NBmax %d\n", NBmax);
	
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
	cudaMalloc((void **) &Msun_d,Nst*sizeof(double2));
	cudaMalloc((void **) &Spinsun_d,Nst*sizeof(double4));
	cudaMalloc((void **) &Lovesun_d,Nst*sizeof(double3));
	cudaMalloc((void **) &J2_d,Nst*sizeof(double2));
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
	cudaMemcpy(Msun_d, Msun_h, Nst*sizeof(double2), cudaMemcpyHostToDevice);
	cudaMemcpy(Spinsun_d, Spinsun_h, Nst*sizeof(double4), cudaMemcpyHostToDevice);
	cudaMemcpy(Lovesun_d, Lovesun_h, Nst*sizeof(double3), cudaMemcpyHostToDevice);
	cudaMemcpy(J2_d, J2_h, Nst*sizeof(double2), cudaMemcpyHostToDevice);
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
#if def_CPU == 0
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, P.dev[0]);
#endif	
	for(int st = 0; st < Nst; ++st){
		GSF[st].logfile = fopen(GSF[st].logfilename, "a");
		if(P.ndev > devCount){
			P.ndev = devCount;
			fprintf(GSF[st].logfile,"selected amount of devices not allowed; changed to %d", P.ndev);
		}
		for(int j = 0; j < P.ndev; ++j){
			if(P.dev[j] > devCount){
				int dev0 = P.dev[j];
				P.dev[j] = P.dev[j] % devCount;
				fprintf(GSF[st].logfile,"selected device not allowed; changed %d to %d", dev0, P.dev[j]);
			}
		}
		
		for(int i = 0; i < 2; ++i){
			if(i == 1){
				infofile = stdout;
				if(Nst > 1) break;
			}
			else infofile = GSF[st].logfile;
			fprintf(infofile, "\n ******** Simulation path %s ********\n\n", GSF[st].path);
			fprintf(infofile, "Genga Version: %g\n", def_Version);
#if def_CPU == 1
			fprintf(infofile, "Nomp = %d cpu threads\n", Nomp);
#endif
			fprintf(infofile, "Mercurial Branch: %s\n", GIT_BRANCH);
			fprintf(infofile, "Mercurial Commit: %s\n", GIT_COMMIT);
			fprintf(infofile, "Build Date: %s\n", BUILD_DATE);
			fprintf(infofile, "Build Path: %s\n", BUILD_PATH);
			fprintf(infofile, "Build System: %s\n", BUILD_SYSTEM);
			fprintf(infofile, "Build Compute Capability: SM=%s\n", BUILD_SM);
#if def_CPU == 0
			fprintf(infofile, "Device Name: %s, Major:%d, Minor:%d\n", devProp.name, devProp.major, devProp.minor);
#endif
			fprintf(infofile, "Number of devices: %d\n", P.ndev);					// use only argument in simulation 0
			for(int j = 0; j < P.ndev; ++j){
				fprintf(infofile, "Device number %d: %d\n", j, P.dev[j]);			// use only argument in simulation 0
			}
			fprintf(infofile, "Do Kick in single precision: %d\n", P.KickFloat);			// use only argument in simulation 0
			fprintf(infofile, "Serial Grouping: %d\n", P.SERIAL_GROUPING);				// use only argument in simulation 0
			fprintf(infofile, "Do kernel tuning: %d\n", P.doTuning);				// use only argument in simulation 0
			fprintf(infofile, "Do symplectic level tuning: %d\n", P.doSLTuning);			// use only argument in simulation 0
			fprintf(infofile, "Compute Poincare Section: %d\n", def_poincareFlag);
			fprintf(infofile, "FormatS: %d\n", P.FormatS);						// use only argument in simulation 0
			fprintf(infofile, "FormatT: %d\n", P.FormatT);						// use only argument in simulation 0
			fprintf(infofile, "FormatP: %d\n", P.FormatP);						// use only argument in simulation 0
			fprintf(infofile, "FormatO: %d\n", P.FormatO);						// use only argument in simulation 0
			fprintf(infofile, "NmaxM: %d\n", NmaxM);
			fprintf(infofile, "Time step in days: %g \n", idt_h[st]);
			fprintf(infofile, "Starting time: %g \n", ict_h[st]);
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
			fprintf(infofile, "Star Love Number: %g\n", Lovesun_h[st].x);
			fprintf(infofile, "Star fluid Love Number: %g\n", Lovesun_h[st].y);
			fprintf(infofile, "Star tau: %g\n", Lovesun_h[st].z);
			fprintf(infofile, "Star spin_x: %g\n", Spinsun_h[st].x);
			fprintf(infofile, "Star spin_y: %g\n", Spinsun_h[st].y);
			fprintf(infofile, "Star spin_z: %g\n", Spinsun_h[st].z);
			fprintf(infofile, "Star Ic: %g\n", Spinsun_h[st].w);
			fprintf(infofile, "J2: %g\n", J2_h[st].x);
			fprintf(infofile, "J2 radius: %g\n", J2_h[st].y);
			fprintf(infofile, "Solar Constant: %g\n", P.SolarConstant);				// use only argument in simulation 0
			fprintf(infofile, "n1: %g\n", n1_h[st]);
			fprintf(infofile, "n2: %g\n", n2_h[st]);
			#if def_G3 > 0
			fprintf(infofile, "G3Limit: %g\n", def_G3Limit);
			fprintf(infofile, "G3Limit2: %g\n", def_G3Limit2);
			#endif
			fprintf(infofile, "Input file: %s\n", GSF[st].Originputfilename);
			fprintf(infofile, "Input file format: ");
			for(int f = 0; f < def_Ninformat; ++f){
				int ff = GSF[st].informat[f];
				if(ff > 0){
					fprintf(infofile, "%s ", fileFormat[ff]);
				}
			}
			fprintf(infofile, "\n");
			fprintf(infofile, "Output file format: ");
			for(int f = 0; f < def_Ninformat; ++f){
				int ff = GSF[st].outformat[f];
				if(ff > 0){
					fprintf(infofile, "%s ", fileFormat[ff]);
				}
			}
			fprintf(infofile, "\n");
			fprintf(infofile, "Use output binary files: %d\n", P.OutBinary);		// use only argument in simulation 0
			fprintf(infofile, "Angle units: %d\n", P.AngleUnits);
			fprintf(infofile, "Default rho: %g\n", rho[st]);
			fprintf(infofile, "Inner truncation radius: %g\n", RcutSun_h[st]);
			fprintf(infofile, "Outer truncation radius: %g\n", Rcut_h[st]);
			fprintf(infofile, "MaxColl: %d\n", def_MaxColl);
			fprintf(infofile, "pc: %g\n", def_pc);
			fprintf(infofile, "cef: %g\n", def_cef);
			fprintf(infofile, "Number of bodies: %d\n", N_h[st]);
			fprintf(infofile, "Number of test particles: %d\n", Nsmall_h[st]);
			fprintf(infofile, "Minimal number of bodies: %d\n", Nmin[st].x);
			fprintf(infofile, "Minimal number of test particles: %d\n", Nmin[st].y);
			fprintf(infofile, "Test Particle Mode: %d\n", P.UseTestParticles);		// use only argument in simulation 0
			fprintf(infofile, "Particle Minimum Mass : %g\n", P.MinMass);			// use only argument in simulation 0
			fprintf(infofile, "Symplectic recursion Max levels : %d\n", def_SLevelsMax);	// use only argument in simulation 0
			fprintf(infofile, "Symplectic recursion levels : %d\n", P.SLevels);		// use only argument in simulation 0
			fprintf(infofile, "Symplectic recursion sub steps : %d\n", P.SLSteps);		// use only argument in simulation 0
			fprintf(infofile, "Restart time step: %lld\n", P.tRestart);			// use only argument in simulation 0
			fprintf(infofile, "Order of Symplectic integrator: %d\n", P.SIO);		// use only argument in simulation 0
			fprintf(infofile, "Maximum encounter pairs: %d\n", P.NencMax);			// use only argument in simulation 0
			fprintf(infofile, "Nfragments: %d\n", P.Nfragments);				// use only argument in simulation 0
			fprintf(infofile, "Use aeGrid: %d\n", P.UseaeGrid);				// use only argument in simulation 0
			fprintf(infofile, "aeGrid amin: %f\n", Gridae.amin);				// use only argument in simulation 0
			fprintf(infofile, "aeGrid amax: %f\n", Gridae.amax);				// use only argument in simulation 0
			fprintf(infofile, "aeGrid emin: %f\n", Gridae.emin);				// use only argument in simulation 0
			fprintf(infofile, "aeGrid emax: %f\n", Gridae.emax);				// use only argument in simulation 0
			fprintf(infofile, "aeGrid imin: %f\n", Gridae.imin);				// use only argument in simulation 0
			fprintf(infofile, "aeGrid imax: %f\n", Gridae.imax);				// use only argument in simulation 0
			fprintf(infofile, "aeGrid Na: %d\n", Gridae.Na);				// use only argument in simulation 0
			fprintf(infofile, "aeGrid Ne: %d\n", Gridae.Ne);				// use only argument in simulation 0
			fprintf(infofile, "aeGrid Ni: %d\n", Gridae.Ne);				// use only argument in simulation 0
			fprintf(infofile, "aeGrid Count Start: %lld\n", Gridae.Start);			// use only argument in simulation 0
			fprintf(infofile, "aeGrid name: %s\n", Gridae.X);				// use only argument in simulation 0
			fprintf(infofile, "Use gas disk: %d\n", P.Usegas);				// use only argument in simulation 0
			fprintf(infofile, "Use gas disk enhancement: %d\n", P.UsegasEnhance);		// use only argument in simulation 0
			fprintf(infofile, "Use gas disk potential: %d\n", P.UsegasPotential);		// use only argument in simulation 0
			fprintf(infofile, "Use gas disk drag: %d\n", P.UsegasDrag);			// use only argument in simulation 0
			fprintf(infofile, "Use gas disk tidal damping: %d\n", P.UsegasTidalDamping);	// use only argument in simulation 0
			fprintf(infofile, "Gas dTau_diss: %g\n", P.G_dTau_diss);			// use only argument in simulation 0
			fprintf(infofile, "Gas disk inner edge: %g\n", P.G_rg0);			// use only argument in simulation 0
			fprintf(infofile, "Gas disk outer edge: %g\n", P.G_rg1);			// use only argument in simulation 0
			fprintf(infofile, "Gas disk grid outer edge: %g\n", P.G_rp1);			// use only argument in simulation 0
			fprintf(infofile, "Gas disk grid dr: %g\n", P.G_drg);				// use only argument in simulation 0
			fprintf(infofile, "Gas alpha: %g\n", P.G_alpha);				// use only argument in simulation 0
			fprintf(infofile, "Gas beta: %g\n", P.G_beta);					// use only argument in simulation 0
			fprintf(infofile, "Gas Sigma_10: %g\n", P.G_Sigma_10 / (1.49598*1.49598/1.98892*1.0e-7));// use only argument in simulation 0
			fprintf(infofile, "Gas Mgiant: %g\n", P.G_Mgiant);				// use only argument in simulation 0
			fprintf(infofile, "Use force (Old): %d\n", P.UseForce);				// use only argument in simulation 0
			fprintf(infofile, "Use GR: %d\n", P.UseGR);					// use only argument in simulation 0
			fprintf(infofile, "Use Tides: %d\n", P.UseTides);				// use only argument in simulation 0
			fprintf(infofile, "Use Rotational Deformation: %d\n", P.UseRotationalDeformation);// use only argument in simulation 0
			fprintf(infofile, "Use Yarkovsky: %d\n", P.UseYarkovsky);			// use only argument in simulation 0
			fprintf(infofile, "Use Poynting-Robertson: %d\n", P.UsePR);			// use only argument in simulation 0
			fprintf(infofile, "Radiation Pressure Coefficient Qpr: %g\n", P.Qpr);		// use only argument in simulation 0
			fprintf(infofile, "Use Migration Force: %d\n", P.UseMigrationForce);		// use only argument in simulation 0
			fprintf(infofile, "Solar Wind factor: %g\n", P.SolarWind);			// use only argument in simulation 0
			fprintf(infofile, "Use Small Collisions: %d\n", P.UseSmallCollisions);		// use only argument in simulation 0
			fprintf(infofile, "Create Particles file name: %s\n", P.CreateParticlesfilename);// use only argument in simulation 0
			fprintf(infofile, "Use Set Elemets function: %d\n", P.setElements);		// use only argument in simulation 0
			fprintf(infofile, "Set Elements file name: %s\n", P.setElementsfilename);	// use only argument in simulation 0
			fprintf(infofile, "Gas file name: %s\n", P.Gasfilename);			// use only argument in simulation 0
			fprintf(infofile, "Report Encounters: %d\n", P.WriteEncounters);		// use only argument in simulation 0
			fprintf(infofile, "Report Encounters Radius: %g\n", P.WriteEncountersRadius);	// use only argument in simulation 0
			fprintf(infofile, "Report Encounters Cloud Size: %d\n", P.WriteEncountersCloudSize);	// use only argument in simulation 0
			fprintf(infofile, "Stop at close Encounters: %d\n", P.StopAtEncounter);
			fprintf(infofile, "Stop at close Encounter Radius: %g\n", P.StopAtEncounterRadius);
			fprintf(infofile, "Stop at Collision: %d\n", P.StopAtCollision);
			fprintf(infofile, "Stop collision minimum mass: %g\n", P.StopMinMass);
			fprintf(infofile, "Collision precision: %g\n", P.CollisionPrecision);
			fprintf(infofile, "Collision Time Shift: %g\n", P.CollTshift);
			fprintf(infofile, "Collision Model: %d\n", P.CollisionModel);
			fprintf(infofile, "Asteroid emissivity: %g\n", P.Asteroid_eps);
			fprintf(infofile, "Asteroid density: %g\n", P.Asteroid_rho);
			fprintf(infofile, "Asteroid specific heat capacity: %g\n", P.Asteroid_C);
			fprintf(infofile, "Asteroid albedo: %g\n", P.Asteroid_A);
			fprintf(infofile, "Asteroid thermal conductivity: %g\n", P.Asteroid_K);
			fprintf(infofile, "Asteroid collisional velocity V: %g\n", P.Asteroid_V);
			fprintf(infofile, "Asteroid minimal fragment radius: %g\n", P.Asteroid_rmin);
			fprintf(infofile, "Asteroid fragment remove radius: %g\n", P.Asteroid_rdel);
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
		NBNencT += NB[st] * P.NencMax;
		NEnergyT += 8;
	}
	
	NconstT = NT + NsmallT + P.Nfragments;
	if(Nst == 1){
		NBNencT = NconstT * P.NencMax;
		NEnergyT = 8;
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

	// read now file 
	Transitfile = fopen(P.Transitsfilename, "r");
	
	for(int i = 0; i < N_h[0]; ++i){
		NtransitsTObs_h[i] = 0;
		
	}
	for(int i = 0; i < def_NtransitTimeMax * N_h[0]; ++i){
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
	int Epoch = 0;
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

		//avoid max
		if(Epoch > NtransitsTObs_h[index]){
			NtransitsTObs_h[index] = Epoch;
		}		
//printf("A %d %d %g %g %d %d\n", index, i, T, TOld, dEpoch, Epoch); 
//printf("read NTobs %d %d | %d %d %.20g\n", index, NtransitsTObs_h[index], Epoch,  index * def_NtransitTimeMax + Epoch + 1, TransitTimeObs_h[index * def_NtransitTimeMax + Epoch + 1].x);
		
	}
	fclose(Transitfile);

	cudaMemcpy(TransitTimeObs_d, TransitTimeObs_h, def_NtransitTimeMax * N_h[0] * sizeof(double2), cudaMemcpyHostToDevice);
	cudaMemcpy(NtransitsTObs_d, NtransitsTObs_h, N_h[0] * sizeof(int), cudaMemcpyHostToDevice);

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
		er = fscanf (Efile, "%s", sp);
		
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
		else if(strcmp(sp, "M") == 0){	
			Elements[i] = 21;
			printf("M ");
			++nelements;
			P.setElements = 2;
			useKeplerElements = 1;
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
	er = fscanf(Efile, "%lf", &t);
	for(int i = 0; i < nelements; ++i){
		char c[64];
		er = fscanf(Efile, "%s", c);
	}
	int nlines = 0;
	int nlinesToSkip = 0;
	double time = -1.0e10;
	double timeOld = -2.0e10;
	double timeOld2 = time;
	//start time
	double time0 = ict_h[0] + P.tRestart * idt_h[0] / 365.25;
	//end time
	double time1 = ict_h[0] + P.deltaT * idt_h[0] / 365.25;
	for(int j = 0; j < def_NSetElementsMax; ++j){
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
//printf("time %.10g %.10g start time %.10g| end time %.10g | %d %d %d\n", time, timeOld2, time0, time1, nlinesToSkip, nlines, P.setElementsN);

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
		if((time >= ict_h[0] && timeOld2 <= time1) || nlines < 4 * P.setElementsN){	//need at least 4 lines for cubic interpolation
			++nlines;
		}
		if(time < ict_h[0] && j >= P.setElementsN){
			++nlinesToSkip;
		}
		if(j == def_NSetElementsMax - 1){
				printf("Error, set Elements file is too long: %d, Change limit in def_NSetElementsMax\n", def_NSetElementsMax);
			return 0;
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
	er = fscanf(Efile, "%lf", &t);
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
#if def_CPU == 1
	setElementsLine_h = (int*)malloc(sizeof(int));
	memset(setElementsLine_h, 0, sizeof(int));
#endif

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
	double t, r;
	double t0 = 0.0;
	double t1 = 0.0;
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
	free(NBT);
	free(Nmin);
	free(rho);
	free(GSF);
	if(P.IrregularOutputs == 1){
		free(IrrOutputs);
	}
	free(NEnergy);
	
	free(n1_h);
	free(n2_h);
	free(N_h);
	free(Nsmall_h);
	free(Msun_h);
	free(Spinsun_h);
	free(Lovesun_h);
	free(J2_h);
	free(idt_h);
	free(ict_h);
	free(Rcut_h);
	free(RcutSun_h);
	free(time_h);
	free(dt_h);
	free(delta_h);
	free(NBS_h);
	free(NsmallS_h);
	if(P.setElements > 0){
		free(setElementsData_h);
#if def_CPU == 1
		free(setElementsLine_h);
#endif
	}
	if(P.Usegas == 2){
		free(GasData_h);
	}

	
	cudaFree(n1_d);
	cudaFree(n2_d);
	cudaFree(N_d);
	cudaFree(Nsmall_d);
	cudaFree(Msun_d);
	cudaFree(Spinsun_d);
	cudaFree(Lovesun_d);
	cudaFree(J2_d);
	cudaFree(idt_d);
	cudaFree(ict_d);
	cudaFree(Rcut_d);
	cudaFree(RcutSun_d);
	cudaFree(time_d);
	cudaFree(dt_d);
	cudaFree(delta_d);
	cudaFree(NBS_d);
	if(P.setElements > 0){
		cudaFree(setElementsData_d);
		cudaFree(setElementsLine_d);
	}
	if(P.Usegas == 2){
		cudaFree(GasData_d);
	}
	
	error = cudaGetLastError();
	if(error != 0){
		printf("Cuda Orbit free error = %d = %s\n",error, cudaGetErrorString(error));
		fprintf(masterfile, "Cuda Orbit free error = %d = %s\n",error, cudaGetErrorString(error));
		return 0;
	}
	return 1;
	
	
}

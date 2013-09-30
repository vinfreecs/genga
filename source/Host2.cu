#include "Host2.h"

FILE *masterfile;
char masterfilename[64];
FILE *pathfile;
char pathfilename[64];

int Nst;

int devCount;
int runtimeVersion;
int driverVersion;

int *NB;
int *icNB;
int *N4;
int *N2;
int *Nconst;
int *Nmin;
double *rho;


struct Parameter P;
struct GSFiles *GSF;
struct GridaeParameter Gridae;
// These are the parameters for the multi simulation run mode //
double *n1_h, *n1_d;
double *n2_h, *n2_d;
int *N_h, *N_d;
int *Nsmall_h, *Nsmall_d;
double *Msun_h, *Msun_d;
double *dtiMsun_h, *dtiMsun_d;

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


__host__ void Hostinit(){
	sprintf(masterfilename, "%s", "master.out");
	masterfile = fopen(masterfilename, "w");
	sprintf(pathfilename, "");

	Nst = 1;
	devCount = 0;
	runtimeVersion = 0;
	driverVersion = 0;

	NT = 0;
	NsmallT = 0;
	NB2T = 0;
	Nsmall2T = 0;
	NEnergyT = 0;
}


// ************************************************

//This function determines the number of simulations by reading the pathfile specified in the -M console argument
__host__ int NSimulations(int argc, char*argv[]){

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
	}
	if(Nst == 0){
		printf("Error: No Simulations!\n", pathfilename);
		fprintf(masterfile, "Error: No Simulations!\n", pathfilename);
		return 0;
	}

	return Nst;
}


// ************************************************

//Device Properties
__host__ int DeviceInfo(){

	cudaError_t error;
	cudaGetDeviceCount(&devCount);
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
		fprintf(masterfile,"Name:%s, Major:%d, Minor:%d, Max threads per Block:%d, Max x dim:%d, #Multiprocessors:%d, Can Map Memory:%d, Clock Rate:%d, Memory Clock Rate:%d, Can Overlap:%d, Concurrent Kernels %d\n",  devProp.name, devProp.major, devProp.minor, devProp.maxThreadsPerBlock, devProp.maxThreadsDim[0], devProp.multiProcessorCount, devProp.canMapHostMemory, devProp.clockRate, devProp.memoryClockRate, devProp.deviceOverlap, devProp.concurrentKernels);
		if(!devProp.canMapHostMemory) {
			fprintf(masterfile, "Device %d cannot map host memory!\n", i);
			return 0;
		}
	}
	return 1;
}


// ************************************************

//This function allocates memory on the Host
__host__ void Halloc(){
	NB = (int*)malloc(Nst*sizeof(int));
	icNB = (int*)malloc(Nst*sizeof(int));
	N4 = (int*)malloc(Nst*sizeof(int));
	N2 = (int*)malloc(Nst*sizeof(int));
	Nconst = (int*)malloc(Nst*sizeof(int));
	Nmin = (int*)malloc(Nst*sizeof(int));
	rho = (double*)malloc(Nst*sizeof(double));
	
	P.dev = 0;
	P.ict = 0.0;
	GSF = (struct GSFiles*)malloc(Nst*sizeof(struct GSFiles));
	
	n1_h = (double*)malloc(Nst*sizeof(double));
	n2_h = (double*)malloc(Nst*sizeof(double));
	N_h = (int*)malloc(Nst*sizeof(int));
	Nsmall_h = (int*)malloc(Nst*sizeof(int));
	Msun_h = (double*)malloc(Nst*sizeof(double));
	dtiMsun_h = (double*)malloc(Nst*sizeof(double));
	
	for(int st = 0; st < Nst; ++st){
		Nsmall_h[st] = 0;
		N_h[st] = 32;
		NB[st] = N_h[st];
		N4[st] = N_h[st]/4;
		N2[st] = N_h[st]/2;
		Msun_h[st] = 1.0;
		n1_h[st] = 3.0;
		n2_h[st] = 0.4;
		rho[st] = 0.0;
		for(int i = 0; i < 22; ++i){
			GSF[st].informat[i] = 0;
		}
		Nmin[st] = 1;
	}
	
	//Read the paths for the individual simulations
	if(Nst > 1){
		pathfile = fopen(pathfilename, "r");
		for(int st = 0; st < Nst; ++st){
			char t[160];
			fscanf(pathfile, "%s", t);
			sprintf(GSF[st].path, "%s/", t);
		}
		fclose(pathfile);
	}
	else sprintf(GSF[0].path, "");

};


// ************************************************

//This function reads the parameters from param.dat and the console input arguments.
//Return 1 by sucess and 0 by an error.
__host__ int readparam(FILE *paramfile, int st, int argc, char*argv[]){

	char sp[160];
	int er;

	//Read time step
	er = fscanf (paramfile, "%s%s%s%s%2c",sp, sp, sp, sp, sp);
	er = fscanf (paramfile, "%lf",&P.idt);

	if(er <= 0){
		printf("Error: time step is not valid!\n");
		return 0;
	}

	//Read output name
	er = fscanf (paramfile, "%s%s%2c",sp, sp, sp);
	er = fscanf (paramfile, "%s", &GSF[st].X);

	if(er <= 0){
		printf("Error: Output name is not valid!\n");
		return 0;
	}

	//Read Energy output intervall
	er = fscanf (paramfile, "%s%s%s%2c",sp, sp, sp, sp);
	er = fscanf (paramfile, "%d", &P.ei);

	if(er <= 0 || P.ei <= 0){
		printf("Error: Energy outut intervall is not valid!\n");
		return 0;
	}

	//Read Coordinate  output intervall
	er = fscanf (paramfile, "%s%s%s%2c",sp, sp, sp, sp);
	er = fscanf (paramfile, "%d", &P.ci);

	if(er <= 0 || P.ci <= 0){
		printf("Error: Coordinates outut intervall is not valid!\n");
		return 0;
	}

	//Read Number of outputs per intervall
	er = fscanf (paramfile, "%s%s%s%s%s%2c",sp, sp, sp, sp, sp, sp);
	er = fscanf (paramfile, "%d", &P.nci);

	if(er <= 0 || P.nci <= 0 || P.nci > P.ci){
		printf("Error: Number of outputs per intervall is not valid!\n");
		return 0;
	}

	//Read integration steps
	er = fscanf (paramfile, "%s%s%2c",sp, sp, sp);
	er = fscanf (paramfile, "%lld", &P.delta);

	if(er <= 0 || P.delta <= 0){
		printf("Error: Inegration steps are not valid!\n");
		return 0;
	}

	//Read Solar Mass
	er = fscanf (paramfile, "%s%s%2c",sp, sp, sp);
	er = fscanf (paramfile, "%lf", &Msun_h[st]);

	if(er <= 0){
		printf("Error: Solar mass is not valid!\n");
		return 0;
	}

	//Read n1
	er = fscanf (paramfile, "%s%2c",sp, sp);
	er = fscanf (paramfile, "%lf", &n1_h[st]);

	if(er <= 0){
		printf("Error: n1 is not valid!\n");
		return 0;
	}

	//Read n2
	er = fscanf (paramfile, "%s%2c",sp, sp);
	er = fscanf (paramfile, "%lf", &n2_h[st]);

	if(er <= 0){
		printf("Error: n2 is not valid!\n");
		return 0;
	}

	//Read input file name
	er = fscanf (paramfile, "%s%s%3c",sp, sp, sp);
	er = fscanf (paramfile, "%s", &GSF[st].inputfilename);

	if(er <= 0){
		printf("Error: Input file name is not valid!\n");
		return 0;
	}
	//Read input file Format
	{int f;
	er = fscanf (paramfile, "%s%s%s%3c",sp, sp, sp, sp);
	for(f = 0; f < 22; ++f){
		er = fscanf (paramfile, "%s", sp);
		if(strcmp(sp, "x") == 0){
			GSF[st].informat[f] = 1;
		}
		else if(strcmp(sp, "y") == 0){
			GSF[st].informat[f] = 2;
		}
		else if(strcmp(sp, "z") == 0){
			GSF[st].informat[f] = 3;
		}
		else if(strcmp(sp, "m") == 0){
			GSF[st].informat[f] = 4;
		}
		else if(strcmp(sp, "vx") == 0){
			GSF[st].informat[f] = 5;
		}
		else if(strcmp(sp, "vy") == 0){
			GSF[st].informat[f] = 6;
		}
		else if(strcmp(sp, "vz") == 0){
			GSF[st].informat[f] = 7;
		}
		else if(strcmp(sp, "r") == 0){
			GSF[st].informat[f] = 8;
		}
		else if(strcmp(sp, "rho") == 0){
			GSF[st].informat[f] = 9;
		}
		else if(strcmp(sp, "Sx") == 0){
			GSF[st].informat[f] = 10;
		}
		else if(strcmp(sp, "Sy") == 0){
			GSF[st].informat[f] = 11;
		}
		else if(strcmp(sp, "Sz") == 0){
			GSF[st].informat[f] = 12;
		}
		else if(strcmp(sp, "i") == 0){
			GSF[st].informat[f] = 13;
		}
		else if(strcmp(sp, "-") == 0){
			GSF[st].informat[f] = 14;
		}
		else if(strcmp(sp, "amin") == 0){
			GSF[st].informat[f] = 15;
		}
		else if(strcmp(sp, "amax") == 0){
			GSF[st].informat[f] = 16;
		}
		else if(strcmp(sp, "emin") == 0){
			GSF[st].informat[f] = 17;
		}
		else if(strcmp(sp, "emax") == 0){
			GSF[st].informat[f] = 18;
		}
		else if(strcmp(sp, "t") == 0){
			GSF[st].informat[f] = 19;
		}
		else if(strcmp(sp, ">>") == 0){
			break;
		}
		else {
			printf("Error: Input format not valid! Maybe the spaces in << ... >> have been forgotten\n");
			return 0;
		}
	}
	if(er <= 0){
		printf("Error: Input file format is not valid!\n");
		return 0;
	}
	if(f >=22){
		printf("Error: Input file format is not valid, '>>' not found! Maybe the spaces in << ... >> have been forgotten\n");
		return 0;

	}
	}
	//Read Default rho
	er = fscanf (paramfile, "%s%s%3c",sp, sp, sp);
	er = fscanf (paramfile, "%lf", &rho[st]);
	if(er <= 0 ){
		printf("Error: Default value for rho is not valid!\n");
		return 0;
	}

	//Read Test Particle mode
	er = fscanf (paramfile, "%s%s%s%3c",sp, sp, sp, sp);
	er = fscanf (paramfile, "%d", &P.UseTestParticles);
	if(er <= 0 || P.UseTestParticles < 0 || P.UseTestParticles > 1){
		printf("Error: Test Particle Mode not valid\n");
		return 0;
	}

	//Read Restart time step
	er = fscanf (paramfile, "%s%s%3c",sp, sp, sp);
	er = fscanf (paramfile, "%lld", &P.tRestart);
	if(er <= 0 || P.tRestart < 0){
		printf("Error: Restart time step not valid\n");
		return 0;
	}

	//Read Minimal number of bodies
	er = fscanf (paramfile, "%s%s%s%s%3c",sp, sp, sp, sp, sp);
	er = fscanf (paramfile, "%d", &Nmin[st]);
	if(er <= 0 || Nmin < 0){
		printf("Error: Minimal number of bodies not valid\n");
		return 0;
	}

	if(st == 0){
		//Read Order of integrator
		er = fscanf (paramfile, "%s%s%s%3c",sp, sp, sp, sp);
		er = fscanf (paramfile, "%d", &P.SIO);
		if(er <= 0 || P.SIO < 2 || P.SIO > 6 || P.SIO % 2 == 1){
			printf("Error: Order of integrator not valid\n");
			return 0;
		}


		//Read Grid amin
		er = fscanf (paramfile, "%s%s%3c",sp, sp, sp);
		er = fscanf (paramfile, "%f", &Gridae.amin);
		if(er <= 0){
			printf("Error: Grid amin not valid\n");
			return 0;
		}

		//Read Grid amax
		er = fscanf (paramfile, "%s%s%3c",sp, sp, sp);
		er = fscanf (paramfile, "%f", &Gridae.amax);
		if(er <= 0){
			printf("Error: Grid amax not valid\n");
			return 0;
		}

		//Read Grid emin
		er = fscanf (paramfile, "%s%s%3c",sp, sp, sp);
		er = fscanf (paramfile, "%f", &Gridae.emin);
		if(er <= 0){
			printf("Error: Grid emin not valid\n");
			return 0;
		}

		//Read Grid emax
		er = fscanf (paramfile, "%s%s%3c",sp, sp, sp);
		er = fscanf (paramfile, "%f", &Gridae.emax);
		if(er <= 0){
			printf("Error: Grid emax not valid\n");
			return 0;
		}

		//Read Grid Na
		er = fscanf (paramfile, "%s%s%3c",sp, sp, sp);
		er = fscanf (paramfile, "%d", &Gridae.Na);
		if(er <= 0){
			printf("Error: Grid Na not valid\n");
			return 0;
		}

		//Read Grid Ne
		er = fscanf (paramfile, "%s%s%3c",sp, sp, sp);
		er = fscanf (paramfile, "%d", &Gridae.Ne);
		if(er <= 0){
			printf("Error: Grid Ne not valid\n");
			return 0;
		}

		//Read Grid Start
		er = fscanf (paramfile, "%s%s%s%3c",sp, sp, sp, sp);
		er = fscanf (paramfile, "%d", &Gridae.Start);
		if(er <= 0){
			printf("Error: Grid Start not valid\n");
			return 0;
		}

		//Read Grid name
		er = fscanf (paramfile, "%s%s%2c",sp, sp, sp);
		er = fscanf (paramfile, "%s", &Gridae.X);

		if(er <= 0){
			printf("Error: Grid name is not valid!\n");
			return 0;
		}
		if(er <= 0){
			printf("Error: Grid name is not valid!\n");
			return 0;
		}
		Gridae.deltaa = (Gridae.amax - Gridae.amin) / ((float)(Gridae.Na));
		Gridae.deltae = (Gridae.emax - Gridae.emin) / ((float)(Gridae.Ne));

		//Read Gas dTau_diss
		er = fscanf (paramfile, "%s%s%2c",sp, sp, sp);
		er = fscanf (paramfile, "%lf", &P.G_dTau_diss);
		if(er <= 0 || P.G_dTau_diss <= 0.0){
			printf("Error: dTau_diss value is not valid!\n");
			return 0;
		}
		//Read Gas alpha
		er = fscanf (paramfile, "%s%s%2c",sp, sp, sp);
		er = fscanf (paramfile, "%d", &P.G_alpha);
		if(er <= 0 || !(P.G_alpha == 1 || P.G_alpha == 2)){
			printf("Error: Gas alpha value is not valid!\n");
			return 0;
		}

	}

	//Read console input arguments
	for(int i = 1; i < argc; i += 2){

		if(strcmp(argv[i], "-dt") == 0){
			P.idt = atof(argv[i + 1]);
		}
		else if(strcmp(argv[i], "-ei") == 0){
			P.ei = atoi(argv[i + 1]);
		}
		else if(strcmp(argv[i], "-ci") == 0){
			P.ci = atoi(argv[i + 1]);
		}
		else if(strcmp(argv[i], "-I") == 0){
			P.delta = atol(argv[i + 1]);
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
		}
		else if(strcmp(argv[i], "-TP") == 0){
			P.UseTestParticles = atoi(argv[i + 1]);
		}
		else if(strcmp(argv[i], "-M") == 0){
		}
		else if(strcmp(argv[i], "-Nmin") == 0){
			Nmin[st] = atoi(argv[i + 1]);
		}
		else if(strcmp(argv[i], "-SIO") == 0){
			P.SIO = atoi(argv[i + 1]);
		}
		else if(strcmp(argv[i], "-aeN") == 0){
			sprintf(Gridae.X, "%s", argv[i + 1]);
		}
		else if(strcmp(argv[i], "-t") == 0){
			P.ict = atof(argv[i + 1]);
		}
		else{
			printf("Error: Console arguments not valid!\n");
			return 0;
		}
	}
	return 1;
}


// ************************************************

// ************************************************
//This function calls the function readparam
__host__ int Param(int argc, char*argv[]){
	FILE *paramfile;
	char paramfilename[160];

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
		if(er == 0) return 0;

		fclose(paramfile);

		if(Nst > 1){
			char tname[160];
			sprintf(tname, "%s%s", GSF[st].path, GSF[st].inputfilename);
			sprintf(GSF[st].inputfilename, "%s", tname);
			P.UseTestParticles = 0;
		}
	}
	if(P.ei > P.ci){
		P.ei = P.ci;
		printf("**** Energy output interval decreased equal to coordinate output interval ****\n");
		fprintf(masterfile, "**** Energy output interval decreased equal to coordinate output interval ****\n");
	}
	dt = P.idt * dayUnit;
	dtksq = dt * ksq;
	for(int st = 0; st < Nst; ++st){
		dtiMsun_h[st] = dt / Msun_h[st];
		//restart -> inputfilename
		if(P.tRestart > 0 && FormatP == 1){
			if(Nst == 1 || FormatS == 0){
				if(FormatT == 0) sprintf(GSF[st].inputfilename, "%sOut%s_%.12lld.dat", GSF[st].path, GSF[st].X, P.tRestart);
				if(FormatT == 1) sprintf(GSF[st].inputfilename, "%sOut%s.dat", GSF[st].path, GSF[st].X);
			}
			else{
				if(FormatT == 0) sprintf(GSF[st].inputfilename, "Out%s_%.12lld.dat", GSF[st].X, P.tRestart);
				if(FormatT == 1) sprintf(GSF[st].inputfilename, "Out%s.dat", GSF[st].X);
			}
		}
		sprintf(GSF[st].logfilename, "%sinfo%s.dat", GSF[st].path, GSF[st].X);
		sprintf(GSF[st].timefilename, "%stime%s.dat", GSF[st].path, GSF[st].X);
		sprintf(GSF[st].Energyfilename, "%sEnergy%s.dat", GSF[st].path, GSF[st].X);
		sprintf(GSF[st].collisionfilename, "%sCollisions%s.dat", GSF[st].path, GSF[st].X);
		sprintf(GSF[st].ejectfilename, "%sEjections%s.dat", GSF[st].path, GSF[st].X);
		
		//create files or erase content//
		if(P.tRestart == 0){
			GSF[st].logfile = fopen(GSF[st].logfilename, "w");
			fclose(GSF[st].logfile);
			GSF[st].timefile = fopen(GSF[st].timefilename, "w");
			fclose(GSF[st].timefile);
			GSF[st].Energyfile = fopen(GSF[st].Energyfilename, "w");
			fclose(GSF[st].Energyfile);
			GSF[st].collisionfile = fopen(GSF[st].collisionfilename, "w");
			fclose(GSF[st].collisionfile);  
			GSF[st].ejectfile = fopen(GSF[st].ejectfilename, "w");
			fclose(GSF[st].ejectfile);
		}

		GSF[st].logfile = fopen(GSF[st].logfilename, "a");

		if(P.tRestart > 0) fprintf(GSF[st].logfile, "\n\n\n************** Restart Simulation at time step%lld *******************\n", P.tRestart);
		if(P.UseTestParticles == 1 && N_h[st] > 32){
			printf("Error: Number of bodies in Test particle mode too big: %d\n", N_h[st]);
			fprintf(GSF[st].logfile, "Error: Number of bodies in Test particle mode too big: %d\n", N_h[st]);
			fprintf(masterfile, "Error: Number of bodies in Test particle mode too big: %d\n", N_h[st]);
			return 0;
		}
		fclose(GSF[st].logfile);
	}
	return 1;
}

// ************************************************

//This function counts the number of bodies in the initial condition file
//It returns the number of bodies
__host__ int icSize(int st){

	//Determinde the number of coordinates in the input file
	int Nformat = 0;
	for(int f = 0; f < 22; ++f){
		if(GSF[st].informat[f] > 0) ++Nformat;
	}
	if(P.tRestart > 0 && FormatP == 1) Nformat = 21; //This is the number of rows in the coordinate output files 

	char t[500];
	double time = 0.0;
	int er = 1;
	int NN = 0;
	int er1 = 1;
	double m;
	int index;
	char Ets[160]; //exact time at restart time step, must be the same format as the coordinate output
	sprintf(Ets, "%.16g", (P.tRestart * P.idt) / 365.25);
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
			fprintf(masterfile,"Skip Simulation %s: Input file not found\n", GSF[st].path);
			printf("Skip Simulation %s: Input file not found\n", GSF[st].path);
			N_h[st] = 0;
			Nsmall_h[st] = 0;
		}
	}

	for(int i = 0; i < 1000000000; ++i){
		for(int f = 0; f < Nformat; ++f){

			if(P.tRestart == 0 || FormatP == 0){
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
		if(FormatT == 1 && time > Et) break;

		if(er1 == 1){
			if(FormatP == 1){
				if(FormatS == 0 || P.tRestart == 0 || Nst == 1){
					if(Et == time){
						if(m > 0.0) ++NN;
						else ++Nsmall_h[st];
					}
				}
				else if(index / 100 == st){
					if(Et == time){
						if(m > 0.0) ++NN;
						else ++Nsmall_h[st];
					}
				}
			}
			if(FormatP == 0){
				if(P.tRestart == 0){
					if(m > 0.0) ++NN;
					else ++Nsmall_h[st];
				}
				else ++NN;
			}
		}
		else break;
	}
	fclose(infile);

	if(FormatP == 0 && P.tRestart > 0){//Restart FormatP == 0 data
		int NNN = 0;
		int NNNsmall = 0;
		Nformat = 21;
		for(int i = 0; i < 1000000; ++i){
			int NMAX = 0;
			er1 = 1;
			char infilename[160];
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
				if(time > Et) break;
			
				if(er1 == 1){
					if(Nst == 1 || FormatS == 0){
						if(Et == time){
							if(m > 0.0) ++NNN;
							else ++NNNsmall;
						}
						if(NNN + NNNsmall == NN){
							NMAX = 1;
							break;
						}
					}
				}
				else{
					--NN;
					break;
				}
			}
			fclose(infile);
			if(NMAX == 1) break;
		}
		NN = NNN;
		Nsmall_h[st] = NNNsmall;
	}

	

	if(P.UseTestParticles == 0){
		NN += Nsmall_h[st];
		Nsmall_h[st] = 0;
	}
	NN = min(NN, 2048);
	N_h[st] = NN;
	return 1;
}

// ************************************************

//This function calls the function icSize and sets the size parameters
__host__ int size(){
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

		N4[st] = N_h[st];
		if(N4[st] %4 == 3) N4[st] +=1;
		if(N4[st] %4 == 2) N4[st] +=2;
		if(N4[st] %4 == 1) N4[st] +=3;
		N4[st] /= 4;

		N2[st] = N_h[st];
		if(N2[st] % 2 == 1) N2[st] +=1;
		N2[st] /= 2;
		
		icNB[st] = NB[st];
		Nconst[st] = N_h[st];
	}
	return 1;
}

// ************************************************

//This function allocates memory on the device
__host__ void Calloc(){
	cudaMalloc((void **) &n1_d,Nst*sizeof(double));
	cudaMalloc((void **) &n2_d,Nst*sizeof(double));
	cudaMalloc((void **) &N_d,Nst*sizeof(int));
	cudaMalloc((void **) &Nsmall_d,Nst*sizeof(int));
	cudaMalloc((void **) &Msun_d,Nst*sizeof(double));
	cudaMalloc((void **) &dtiMsun_d,Nst*sizeof(double));

	cudaMemcpy(n1_d, n1_h, Nst*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(n2_d, n2_h, Nst*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(N_d, N_h, Nst*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Msun_d, Msun_h, Nst*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dtiMsun_d, dtiMsun_h, Nst*sizeof(double), cudaMemcpyHostToDevice);
}

// ************************************************

//This function prints the parametes on screen and into the infofiles
__host__ void Info(){
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
			fprintf(infofile, "Genga Version: %g\n", Version);
			fprintf(infofile, "FormatS: %d\n", FormatS);
			fprintf(infofile, "FormatT: %d\n", FormatT);
			fprintf(infofile, "FormatP: %d\n", FormatP);
			fprintf(infofile, "dt: %g \n", P.idt);                                          // use only argument in simulation 0
			fprintf(infofile, "Output name: %s\n", GSF[st].X);
			fprintf(infofile, "Energy output intervall: %d\n", P.ei);                       // use only argument in simulation 0
			fprintf(infofile, "Coordinates output intervall: %d\n", P.ci);                  // use only argument in simulation 0
			fprintf(infofile, "Number of outputs per intervall: %d\n", P.nci);              // use only argument in simulation 0
			fprintf(infofile, "Integration steps: %lld\n", P.delta);                        // use only argument in simulation 0
			fprintf(infofile, "Central Mass: %g\n", Msun_h[st]);
			fprintf(infofile, "n1: %g\n", n1_h[st]);
			fprintf(infofile, "n2: %g\n", n2_h[st]);
			fprintf(infofile, "Input file: %s\n", GSF[st].inputfilename);
			fprintf(infofile, "Using device number %d\n", P.dev);                           // use only argument in simulation 0
			fprintf(infofile, "Rcut: %g\n", (double)(Rcut));
			fprintf(infofile, "RcutSun: %g\n", (double)(RcutSun));
			fprintf(infofile, "MaxColl: %d\n", MaxColl);
			fprintf(infofile, "pc: %g\n", pc);
			fprintf(infofile, "cef: %g\n", cef);
			fprintf(infofile, "Number of bodies: %d\n", N_h[st]);
			fprintf(infofile, "Number of test particles: %d\n", Nsmall_h[st]);
			fprintf(infofile, "Minimal number of bodies: %d\n", Nmin[st]);
			fprintf(infofile, "Test Particle Mode: %d\n", P.UseTestParticles);              // use only argument in simulation 0
			fprintf(infofile, "Restart time step: %lld\n", P.tRestart);                     // use only argument in simulation 0
			fprintf(infofile, "Order of Symplectic integrator: %d\n", P.SIO);               // use only argument in simulation 0
			fprintf(infofile, "aeGrid amin: %f\n", Gridae.amin);                            // use only argument in simulation 0
			fprintf(infofile, "aeGrid amax: %f\n", Gridae.amax);                            // use only argument in simulation 0
			fprintf(infofile, "aeGrid emin: %f\n", Gridae.emin);                            // use only argument in simulation 0
			fprintf(infofile, "aeGrid emax: %f\n", Gridae.emax);                            // use only argument in simulation 0
			fprintf(infofile, "aeGrid Na: %d\n", Gridae.Na);                                // use only argument in simulation 0
			fprintf(infofile, "aeGrid Ne: %d\n", Gridae.Ne);                                // use only argument in simulation 0
			fprintf(infofile, "aeGrid Count Start: %lld\n", Gridae.Start);                  // use only argument in simulation 0
			fprintf(infofile, "aeGrid name: %s\n", Gridae.X);                               // use only argument in simulation 0
#if useGas > 0
			fprintf(infofile, "Gas dTau_diss: %g\n", P.G_dTau_diss);                        // use only argument in simulation 0
			fprintf(infofile, "Gas alpha: %d\n", P.G_alpha);                                // use only argument in simulation 0
#endif
			fprintf(infofile, "Runtime Version: %d\n", runtimeVersion);
			fprintf(infofile, "Driver Version: %d\n", driverVersion);
		}
		fclose(GSF[st].logfile);
	}
}

// **************************************

//This function determines the start points of the individual simulations
__host__ void Tsizes(){
	NBS_h = (int*)malloc(Nst*sizeof(int));
	NsmallS_h = (int*)malloc(Nst*sizeof(int));
	NB2S = (int*)malloc(Nst*sizeof(int));
	Nsmall2S = (int*)malloc(Nst*sizeof(int));
	NEnergy = (int*)malloc(Nst*sizeof(int));

	for(int st = 0; st < Nst; ++st){
		NBS_h[st] = NT;
		NsmallS_h[st] = NsmallT;
		NB2S[st] = NB2T;
		Nsmall2S[st] = Nsmall2T;
		NEnergy[st] = NEnergyT;
		NT += N_h[st];
		NsmallT += Nsmall_h[st];
		NB2T += NB[st] * 16;
		Nsmall2T += Nsmall_h[st] * 16;
		NEnergyT += max(NB[st], 8);

	}

	if(Nst == 1){
		NT = NB[0];
		NB2T = NB[0] * NB[0];
		Nsmall2T = Nsmall_h[0] * 32;
	}
}

__host__ int freeHost(){
	
	cudaError_t error;
	
	free(NB);
	free(icNB);
	free(N4);
	free(N2);
	free(Nconst);
	free(Nmin);
	free(rho);
	
	free(n1_h);
	free(n2_h);
	free(N_h);
	free(Nsmall_h);
	free(Msun_h);
	free(dtiMsun_h);
	
	cudaFree(n1_d);
	cudaFree(n2_d);
	cudaFree(N_d);
	cudaFree(Nsmall_d);
	cudaFree(Msun_d);
	cudaFree(dtiMsun_d);
	
	error = cudaGetLastError();
	if(error != 0){
		printf("Cuda Orbit free error = %d = %s\n",error, cudaGetErrorString(error));
		fprintf(masterfile, "Cuda Orbit free error = %d = %s\n",error, cudaGetErrorString(error));
		return 0;
	}
	return 1;

	
}



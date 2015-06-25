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
//Authors: Simon Grimm, Joachim Stadel
//March 2014
// ************************************************
__host__ int Host::NSimulations(int argc, char*argv[]){

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
//Authors: Simon Grimm, Joachim Stadel
//Mai 2015
// ************************************************
__host__ void Host::Halloc(){
	NB = (int*)malloc(Nst*sizeof(int));
	icNB = (int*)malloc(Nst*sizeof(int));
	N4 = (int*)malloc(Nst*sizeof(int));
	N2 = (int*)malloc(Nst*sizeof(int));
	Nconst = (int*)malloc(Nst*sizeof(int));
	Nmin = (int*)malloc(Nst*sizeof(int));
	rho = (double*)malloc(Nst*sizeof(double));	
	delta = (long long*)malloc(Nst*sizeof(long long));

	P.dev = 0;
	GSF = (struct GSFiles*)malloc(Nst*sizeof(struct GSFiles));
	
	n1_h = (double*)malloc(Nst*sizeof(double));
	n2_h = (double*)malloc(Nst*sizeof(double));
	N_h = (int*)malloc(Nst*sizeof(int));
	Nsmall_h = (int*)malloc(Nst*sizeof(int));
	Msun_h = (double*)malloc(Nst*sizeof(double));
	idt_h = (double*)malloc(Nst*sizeof(double));
	ict_h = (double*)malloc(Nst*sizeof(double));
	dtiMsun_h = (double*)malloc(Nst*sizeof(double));
	Rcut_h = (double*)malloc(Nst*sizeof(double));
	RcutSun_h = (double*)malloc(Nst*sizeof(double));
	time_h = (double*)malloc(Nst*sizeof(double));
	dt_h = (double*)malloc(Nst*sizeof(double));
	dtksq_h = (double*)malloc(Nst*sizeof(double));

	//Initialize parameters with default values
	P.ei = def_EnergyOutputInterval;
	P.ci = def_CoordinatesOutputInterval;
	P.nci = def_OutputsPerInterval;
	P.Buffer = def_Buffer;
	P.deltaT = def_IntegrationSteps;
	P.UseTestParticles = def_UseTestParticles;
	P.tRestart = def_RestartTimeStep;	
	P.SIO = def_OderOfIntegrator;
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
	P.UseForce = def_UseForce;
	P.G_dTau_diss = def_GasdTau_diss;
	P.G_alpha = def_GasAlpha;
	P.FormatS = def_FormatS;
	P.FormatT = def_FormatT;
	P.FormatP = def_FormatP;
	P.IrregularOutputs = 0;
	sprintf(P.IrregularOutputsfilename, "%s", "-");
	sprintf(P.setElementsfilename, "%s", "-");
	P.setElements = 0;

	char format[50];
	sprintf(format, def_InputFileFormat);

	char ff[5 * 30];
	int er = sscanf(format, "%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s",
	 &ff[0 * 5], &ff[1 * 5], &ff[2 * 5], &ff[3 * 5],&ff[4 * 5], &ff[5 * 5], &ff[6 * 5], &ff[7 * 5], &ff[8 * 5],
	 &ff[9 * 5], &ff[10 * 5], &ff[11 * 5], &ff[12 * 5], &ff[13 * 5], &ff[14 * 5], &ff[15 * 5], &ff[16 * 5], 
	 &ff[17 * 5], &ff[18 * 5], &ff[19 * 5], &ff[20 * 5], &ff[21 * 5], &ff[22 * 5], &ff[23 * 5]);

	for(int st = 0; st < Nst; ++st){
		for(int i = 0; i < 22; ++i){
			GSF[st].informat[i] = 0;
		}

		for(int f = -1; f < er; ++f){

			if(strcmp(ff + f * 5, "x") == 0){
				GSF[st].informat[f] = 1;
			}
			else if(strcmp(ff + f * 5, "y") == 0){
				GSF[st].informat[f] = 2;
			}
			else if(strcmp(ff + f * 5, "z") == 0){
				GSF[st].informat[f] = 3;
			}
			else if(strcmp(ff + f * 5, "m") == 0){
				GSF[st].informat[f] = 4;
			}
			else if(strcmp(ff + f * 5, "vx") == 0){
				GSF[st].informat[f] = 5;
			}
			else if(strcmp(ff + f * 5, "vy") == 0){
				GSF[st].informat[f] = 6;
			}
			else if(strcmp(ff + f * 5, "vz") == 0){
				GSF[st].informat[f] = 7;
			}
			else if(strcmp(ff + f * 5, "r") == 0){
				GSF[st].informat[f] = 8;
			}
			else if(strcmp(ff + f * 5, "rho") == 0){
				GSF[st].informat[f] = 9;
			}
			else if(strcmp(ff + f * 5, "Sx") == 0){
				GSF[st].informat[f] = 10;
			}
			else if(strcmp(ff + f * 5, "Sy") == 0){
				GSF[st].informat[f] = 11;
			}
			else if(strcmp(ff + f * 5, "Sz") == 0){
				GSF[st].informat[f] = 12;
			}
			else if(strcmp(ff + f * 5, "i") == 0){
				GSF[0].informat[f] = 13;
			}
			else if(strcmp(ff + f * 5, "-") == 0){
				GSF[st].informat[f] = 14;
			}
			else if(strcmp(ff + f * 5, "amin") == 0){
				GSF[st].informat[f] = 15;
			}
			else if(strcmp(ff + f * 5, "amax") == 0){
				GSF[st].informat[f] = 16;
			}
			else if(strcmp(ff + f * 5, "emin") == 0){
				GSF[st].informat[f] = 17;
			}
			else if(strcmp(ff + f * 5, "emax") == 0){
				GSF[st].informat[f] = 18;
			}
			else if(strcmp(ff + f * 5, "t") == 0){
				GSF[st].informat[f] = 19;
			}
		}	
		n1_h[st] = def_n1;
		n2_h[st] = def_n2;
		N_h[st] = 32;
		Nsmall_h[st] = 0;
		Msun_h[st] = def_CentralMass;
		idt_h[st] = def_TimeStep;
		ict_h[st] = 0.0;
		dtiMsun_h[st] = 0.0;		
		Rcut_h[st] = def_Rcut;
		RcutSun_h[st] = def_RcutSun;
		time_h[st] = 0.0;
		dt_h[st] = 0.0;
		dtksq_h[st] = 0.0;

		NB[st] = N_h[st];
		icNB[st] = N_h[st];
		N4[st] = N_h[st]/4;
		N2[st] = N_h[st]/2;
		Nconst[st] = N_h[st] + 1;
		Nmin[st] = def_MinimumNumberOfBodies;
		rho[st] = def_rho;
		delta[st] = def_IntegrationSteps;
		sprintf(GSF[st].X, def_Name);
		sprintf(GSF[st].inputfilename, def_InputFile);

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
//
//Authors: Simon Grimm, Joachim Stadel
//Mai 2015
// ***********************************************
__host__ int Host::readparam(FILE *paramfile, int st, int argc, char*argv[]){

	char sp[160];
	int er;

	for(int j = 0; j < 40; ++j){
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
			er = fscanf (paramfile, "%s", &GSF[st].X);
			if(er <= 0){
				printf("Error: Output name is not valid!\n");	
				return 0;
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Energy output interval =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%d", &P.ei);
				if(er <= 0 || P.ei <= 0){
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
	
				if(er <= 0 || P.ci <= 0){
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
	
				if(er <= 0 || P.nci <= 0 || P.nci > P.ci){
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
				er = fscanf (paramfile, "%s", &P.IrregularOutputsfilename);
	
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
		else if(strcmp(sp, "Integration steps =") == 0){
			er = fscanf (paramfile, "%lld", &delta[st]);
	
			if(er <= 0 || delta[st] <= 0){
				printf("Error: Inegration steps are not valid!\n");
				return 0;
			}
			fgets(sp, 3, paramfile);
			if(st == 0) P.deltaT = delta[st];
			else P.deltaT = max(P.deltaT, delta[st]);
		}
		else if(strcmp(sp, "Central Mass =") == 0){

			er = fscanf (paramfile, "%lf", &Msun_h[st]);

			if(er <= 0){
				printf("Error: Central mass is not valid!\n");
				return 0;
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
			er = fscanf (paramfile, "%s", &GSF[st].inputfilename);

			if(er <= 0){
				printf("Error: Input file name is not valid!\n");
				return 0;
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Input file Format:") == 0){
			for(int i = 0; i < 22; ++i){
				GSF[st].informat[i] = 0;
			}
			//Read input file Format
			int f;
			for(f = -1; f < 22; ++f){
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
				else if(strcmp(sp, "<<") == 0){
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
				if(er <= 0 || P.UseTestParticles < 0 || P.UseTestParticles > 1){
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
		else if(strcmp(sp, "Restart timestep =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%lld", &P.tRestart);
				if(er <= 0 || P.tRestart < 0){
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
			er = fscanf (paramfile, "%d", &Nmin[st]);
			if(er <= 0 || Nmin < 0){
				printf("Error: Minimal number of bodies not valid\n");
				return 0;
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Inner truncation radius =") == 0){
			er = fscanf (paramfile, "%lf", &RcutSun_h[st]);
			if(er <= 0 || Nmin < 0){
				printf("Error: Inner truncation radius not valid\n");
				return 0;
			}
			fgets(sp, 3, paramfile);
		}
		else if(strcmp(sp, "Outer truncation radius =") == 0){
			er = fscanf (paramfile, "%lf", &Rcut_h[st]);
			if(er <= 0 || Nmin < 0){
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
				double t;
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
				double t;
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
				double t;
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
				double t;
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
				double t;
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
				double t;
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
				er = fscanf (paramfile, "%s", &Gridae.X);

				if(er <= 0){
					printf("Error: Grid name is not valid!\n");
					return 0;
				}	
			}
			else{
				char t[64];
				er = fscanf (paramfile, "%s", &t);
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
		else if(strcmp(sp, "Gas dTau_diss =") == 0){
			if(st == 0){
				er = fscanf (paramfile, "%lf", &P.G_dTau_diss);
				if(er <= 0 || P.G_dTau_diss <= 0.0){
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
				er = fscanf (paramfile, "%s", &P.setElementsfilename);
	
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
		else{
			printf("Unefined line in param.dat file: line %d\n", j);
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
			delta[st] = P.deltaT;
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
			ict_h[st] = atof(argv[i + 1]);
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
	if(strcmp(P.setElementsfilename, "-") != 0){
		P.setElements = 1;
	}

	return 1;
}


// ************************************************
//This function calls the function readparam
//
//Authors: Simon Grimm, Joachim Stadel
//March 2014
// *********************************************3
__host__ int Host::Param(int argc, char*argv[]){
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
		dt_h[st] = idt_h[st] * dayUnit;
		dtksq_h[st] = dt_h[st] * ksq;
	}
	if(P.ei > P.ci){
		P.ei = P.ci;
		printf("**** Energy output interval decreased equal to coordinate output interval ****\n");
		fprintf(masterfile, "**** Energy output interval decreased equal to coordinate output interval ****\n");
	}

	for(int st = 0; st < Nst; ++st){
		dtiMsun_h[st] = dt_h[st] / Msun_h[st];
		//restart -> inputfilename
		if(P.tRestart > 0 && P.FormatP == 1){
			if(Nst == 1 || P.FormatS == 0){
				if(P.FormatT == 0) sprintf(GSF[st].inputfilename, "%sOut%s_%.12lld.dat", GSF[st].path, GSF[st].X, P.tRestart);
				if(P.FormatT == 1) sprintf(GSF[st].inputfilename, "%sOut%s.dat", GSF[st].path, GSF[st].X);
			}
			else{
				if(P.FormatT == 0) sprintf(GSF[st].inputfilename, "Out%s_%.12lld.dat", GSF[st].X, P.tRestart);
				if(P.FormatT == 1) sprintf(GSF[st].inputfilename, "Out%s.dat", GSF[st].X);
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
	char Origfilename[160];
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
	for(int f = 0; f < 22; ++f){
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
	sprintf(Ets, "%.16g", (P.tRestart * idt_h[st]) / 365.25 + ict_h[st]);
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
			if(P.FormatP == 0){
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

	if(P.FormatP == 0 && P.tRestart > 0){//Restart FormatP == 0 data
		int NNN = 0;
		int NNNsmall = 0;
		Nformat = 21;
		FILE *OrigInfile;
		char Origfilename[160];
		sprintf(Origfilename, "%s%s", GSF[st].path, GSF[st].Originputfilename);
		OrigInfile = fopen(Origfilename, "r");
		for(int k = 0; k < 1000000000; ++k){
			int i;
			double skip = 0.0;
			int eri = 1;
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
					if(Nst == 1 || P.FormatS == 0){
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
	NN = min(NN, 8192);
	N_h[st] = NN;
	return 1;
}

// ************************************************
//This function calls the function icSize and sets the size parameters
//Authors: Simon Grimm, Joachim Stadel
//March 2014
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


		N4[st] = N_h[st];
		if(N4[st] %4 == 3) N4[st] +=1;
		if(N4[st] %4 == 2) N4[st] +=2;
		if(N4[st] %4 == 1) N4[st] +=3;
		N4[st] /= 4;

		N2[st] = N_h[st];
		if(N2[st] % 2 == 1) N2[st] +=1;
		N2[st] /= 2;
		
		icNB[st] = NB[st];
		Nconst[st] = N_h[st] + 1;

		GSF[st].logfile = fopen(GSF[st].logfilename, "a");

		if(P.UseTestParticles == 1 && N_h[st] > min(NmaxTestParticles, 8192)){
			printf("Error: Number of massive bodies in Test particle mode is too big: %d, Maximum number is %d\n", N_h[st], min(NmaxTestParticles, 8192));
			fprintf(GSF[st].logfile, "Error: Number of massive bodies in Test particle mode is too big: %d, Maximum number is %d\n", N_h[st], min(NmaxTestParticles, 8192));
			fprintf(masterfile, "Error: Number of masive bodies in Test particle mode is too big: %d, Maximum number is %d\n", N_h[st], min(NmaxTestParticles, 8192));
			return 0;
		}
		fclose(GSF[st].logfile);
	}
	return 1;
}


// ************************************************
//This function allocates memory on the device
//Authors: Simon Grimm, Joachim Stadel
//March 2014
// ***********************************************
__host__ void Host::Calloc(){
	cudaMalloc((void **) &n1_d,Nst*sizeof(double));
	cudaMalloc((void **) &n2_d,Nst*sizeof(double));
	cudaMalloc((void **) &N_d,Nst*sizeof(int));
	cudaMalloc((void **) &Nsmall_d,Nst*sizeof(int));
	cudaMalloc((void **) &Msun_d,Nst*sizeof(double));
	cudaMalloc((void **) &idt_d,Nst*sizeof(double));
	cudaMalloc((void **) &ict_d,Nst*sizeof(double));
	cudaMalloc((void **) &dtiMsun_d,Nst*sizeof(double));
	cudaMalloc((void **) &Rcut_d,Nst*sizeof(double));
	cudaMalloc((void **) &RcutSun_d,Nst*sizeof(double));
	cudaMalloc((void **) &time_d,Nst*sizeof(double));
	cudaMalloc((void **) &dt_d,Nst*sizeof(double));
	cudaMalloc((void **) &dtksq_d,Nst*sizeof(double));

	cudaMemcpy(n1_d, n1_h, Nst*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(n2_d, n2_h, Nst*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(N_d, N_h, Nst*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Msun_d, Msun_h, Nst*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(idt_d, idt_h, Nst*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(ict_d, ict_h, Nst*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dtiMsun_d, dtiMsun_h, Nst*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Rcut_d, Rcut_h, Nst*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(RcutSun_d, RcutSun_h, Nst*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(time_d, time_h, Nst*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dt_d, dt_h, Nst*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dtksq_d, dtksq_h, Nst*sizeof(double), cudaMemcpyHostToDevice);
}

// ************************************************
//This function prints the parametes on screen and into the infofiles
//Authors: Simon Grimm, Joachim Stadel
//March 2014
//  **************************************************
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
			fprintf(infofile, "Genga Version: %g\n", Version);
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
			fprintf(infofile, "NmaxTestParticles: %d\n", NmaxTestParticles);
			fprintf(infofile, "Time step in days: %g \n", idt_h[st]);
			fprintf(infofile, "Output name: %s\n", GSF[st].X);
			fprintf(infofile, "Energy output interval: %d\n", P.ei);				// use only argument in simulation 0
			fprintf(infofile, "Coordinates output interval: %d\n", P.ci);				// use only argument in simulation 0
			fprintf(infofile, "Number of outputs per interval: %d\n", P.nci);			// use only argument in simulation 0
			fprintf(infofile, "Coordinate output buffer: %d\n", P.Buffer);				// use only argument in simulation 0
			fprintf(infofile, "Use Irregular outputs: %d\n", P.IrregularOutputs);			// use only argument in simulation 0
			fprintf(infofile, "Irregular output calendar: %s\n", P.IrregularOutputsfilename);	// use only argument in simulation 0
			fprintf(infofile, "Integration steps: %lld\n", delta[st]);
			fprintf(infofile, "Central Mass: %g\n", Msun_h[st]);
			fprintf(infofile, "n1: %g\n", n1_h[st]);
			fprintf(infofile, "n2: %g\n", n2_h[st]);
#if G3 == 1
			fprintf(infofile, "G3Limit: %g\n", G3Limit);
			fprintf(infofile, "G3Limit2: %g\n", G3Limit2);
#endif
			fprintf(infofile, "Input file: %s\n", GSF[st].Originputfilename);
			fprintf(infofile, "Input file format: ");
			for(int f = 0; f < 22; ++f){
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
				else if(GSF[st].informat[f] == 0) break;
			}
			fprintf(infofile, "\n");
			fprintf(infofile, "Default rho: %g\n", rho[st]);
			fprintf(infofile, "Device number: %d\n", P.dev);                           // use only argument in simulation 0
			fprintf(infofile, "Inner truncation radius: %g\n", RcutSun_h[st]);
			fprintf(infofile, "Outer truncation radius: %g\n", Rcut_h[st]);
			fprintf(infofile, "MaxColl: %d\n", MaxColl);
			fprintf(infofile, "pc: %g\n", pc);
			fprintf(infofile, "cef: %g\n", cef);
			fprintf(infofile, "Number of bodies: %d\n", N_h[st]);
			fprintf(infofile, "Number of test particles: %d\n", Nsmall_h[st]);
			fprintf(infofile, "Minimal number of bodies: %d\n", Nmin[st]);
			fprintf(infofile, "Test Particle Mode: %d\n", P.UseTestParticles);              // use only argument in simulation 0
			fprintf(infofile, "Restart time step: %lld\n", P.tRestart);                     // use only argument in simulation 0
			fprintf(infofile, "Order of Symplectic integrator: %d\n", P.SIO);               // use only argument in simulation 0
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
			fprintf(infofile, "Gas dTau_diss: %g\n", P.G_dTau_diss);                        // use only argument in simulation 0
			fprintf(infofile, "Gas alpha: %d\n", P.G_alpha);                                // use only argument in simulation 0
			fprintf(infofile, "Use force: %d\n", P.UseForce);				// use only argument in simulation 0
			fprintf(infofile, "Use Set Elemets function: %d\n", P.setElements);		// use only argument in simulation 0
			fprintf(infofile, "Set Elements file name: %s\n", P.setElementsfilename);	// use only argument in simulation 0
			fprintf(infofile, "Runtime Version: %d\n", runtimeVersion);
			fprintf(infofile, "Driver Version: %d\n", driverVersion);
		}
		fclose(GSF[st].logfile);
	}
}


// **************************************
//This function determines the start points of the individual simulations
//Authors: Simon Grimm, Joachim Stadel
//March 2014
// ******************************************
__host__ void Host::Tsizes(){
	NBS_h = (int*)malloc(Nst*sizeof(int));
	NsmallS_h = (int*)malloc(Nst*sizeof(int));
	NB2S = (int*)malloc(Nst*sizeof(int));
	Nsmall2S = (int*)malloc(Nst*sizeof(int));
	NEnergy = (int*)malloc(Nst*sizeof(int));

	cudaMalloc((void **) &NBS_d, Nst*sizeof(int));

	for(int st = 0; st < Nst; ++st){
		NBS_h[st] = NT;
		NsmallS_h[st] = NsmallT;
		NB2S[st] = NB2T;
		Nsmall2S[st] = Nsmall2T;
		NEnergy[st] = NEnergyT;
		NT += N_h[st];
		NsmallT += Nsmall_h[st];
		NB2T += NB[st] * NmaxTestParticles;
		Nsmall2T += Nsmall_h[st] * NmaxTestParticles;
		NEnergyT += max(NB[st], 8);

	}

	if(Nst == 1){
		NT = NB[0];
		NB2T = NB[0] * NB[0];
		Nsmall2T = Nsmall_h[0] * 2 * NmaxTestParticles;
	}
	NconstT = NT + NsmallT;
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
	for(int i = 0; i < 1000000; ++i){
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

	int Elements[12];
	for(int i = 0; i < 12; ++i){
		Elements[i] = 0;
	}


	//read the number of planets
	int nbodies = 1;
	int er = fscanf(Efile, "%d", &nbodies);
	if(er <= 0) return 0;

printf("%d\n", nbodies);
	int nelements = 0;
	char sp[16];
	fgets(sp, 2, Efile);
	//determine the specified elements
	for(int i = 0; i < 12; ++i){
		int c = fgetc(Efile);

		if(c == 't'){
			Elements[i] = 1;
			printf("t ");
			++nelements;
		}
		else if(c == 'j'){
			Elements[i] = 2;
			printf("j ");
			++nelements;
		}
		else if(c == 'a'){
			Elements[i] = 3;
			printf("a ");
			++nelements;
		}
		else if(c == 'e'){
			Elements[i] = 4;
			printf("e ");
			++nelements;
		}
		else if(c == 'i'){
			Elements[i] = 5;
			printf("i ");
			++nelements;
		}
		else if(c == 'N'){
			Elements[i] = 6;
			printf("N ");
			++nelements;
		}
		else if(c == 'w'){
			Elements[i] = 7;
			printf("w ");
			++nelements;
		}
		else if(c == 'm'){
			Elements[i] = 8;
			printf("m ");
			++nelements;
		}
		else if(c == 'r'){
			Elements[i] = 9;
			printf("r ");
			++nelements;
		}
		else if(c == 'T'){
			Elements[i] = 10;
			printf("T ");
			++nelements;
		}
		else{
			printf("\n");
			break;
		}

		fgets(sp, 2, Efile);
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
	fscanf(Efile, "%g", &t);
	for(int i = 0; i < nelements; ++i){
		char c[16];
		er = fscanf(Efile, "%s", &c);
	}
	int nlines;
	int ncolumns = (nelements - 1) * nbodies + 1;
	for(int j = 0; j < 1000000; ++j){
		for(int i = 0; i < ncolumns; ++i){
			er = fscanf(Efile, "%lf", &t);
			if(er <= 0) break;
		}
		if(er <= 0){
			nlines = j;
			break;
		}
	}
	fclose(Efile);
printf("%d lines, %d bodies %d elements %d columns\n", nlines, nbodies, nelements, ncolumns);

	constantCopy3(Elements, nelements, nbodies, nlines, ncolumns);
	//allocate memory
	setElementsData_h = (double*)malloc(ncolumns * nlines * sizeof(double));	
	cudaMalloc((void **) &setElementsData_d, ncolumns * nlines * sizeof(double));
	Efile = fopen(P.setElementsfilename, "r");

	//skip header
	fscanf(Efile, "%g", &t);
	for(int i = 0; i < nelements; ++i){
		char c[16];
		er = fscanf(Efile, "%s", &c);
	}
	for(int j = 0; j < nlines; ++j){
		for(int i = 0; i < ncolumns; ++i){
			er = fscanf(Efile, "%lf", &setElementsData_h[j * ncolumns + i]);
		}
	}
	cudaMemcpy(setElementsData_d, setElementsData_h, ncolumns * nlines * sizeof(double), cudaMemcpyHostToDevice);

	cudaMalloc((void **) &setElementsLine_d, sizeof(int));
	cudaMemset(setElementsLine_d, 0, sizeof(int));
	return 1;
}


__host__ int Host::freeHost(){
	cudaError_t error;
	free(NB);
	free(icNB);
	free(N4);
	free(N2);
	free(Nconst);
	free(Nmin);
	free(rho);
	free(delta);
	
	free(n1_h);
	free(n2_h);
	free(N_h);
	free(Nsmall_h);
	free(Msun_h);
	free(idt_h);
	free(ict_h);
	free(dtiMsun_h);
	free(Rcut_h);
	free(RcutSun_h);
	free(time_h);
	free(dt_h);
	free(dtksq_h);
	
	cudaFree(n1_d);
	cudaFree(n2_d);
	cudaFree(N_d);
	cudaFree(Nsmall_d);
	cudaFree(Msun_d);
	cudaFree(idt_d);
	cudaFree(ict_d);
	cudaFree(dtiMsun_d);
	cudaFree(Rcut_d);
	cudaFree(RcutSun_d);
	cudaFree(time_d);
	cudaFree(dt_d);
	cudaFree(dtksq_d);
	
	error = cudaGetLastError();
	if(error != 0){
		printf("Cuda Orbit free error = %d = %s\n",error, cudaGetErrorString(error));
		fprintf(masterfile, "Cuda Orbit free error = %d = %s\n",error, cudaGetErrorString(error));
		return 0;
	}
	return 1;

	
}

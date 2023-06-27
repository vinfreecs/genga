#include "Orbit2.h"
#include "Rcrit.h"
#include "Kick3.h"
#include "HC.h"
#include "FG2.h"
#include "Encounter3.h"
#include "BSB.h"
#include "BSBM.h"
#include "BSBM3.h"
#include "ComEnergy.h"
#include "convert.h"
#include "force.h"
#include "forceYarkovskyOld.h"
#include "Kick4.h"
#include "BSA.h"
#include "BSAM3.h"
#if def_TTV > 0
  #include "BSTTV.h"
#endif
#if def_TTV == 2
  #include "TTVAll.h"
#endif
#if def_RV == 1
  #include "BSRV.h"
#endif
#include "Scan.h"
#include "createparticles.h"
#include "bvh.h"
#if def_G3 > 0
	#include "BSBG3.h"
#endif


int SIn;		//Number of direction steps
int SIM;		//half of steps
double *Ct;		//time factor for HC Kick steps
double *FGt;		//time factor for Drift steps
double *Kt;		//time factor for Kick steps

int EjectionFlag2 = 0;
int StopAtEncounterFlag2 = 0;


#if def_CPU == 1
double Rcut_c[1];
double RcutSun_c[1];
int StopAtCollision_c[1];
double StopMinMass_c[1];
double CollisionPrecision_c[1]; 
double CollTshift_c[1]; 
int CollisionModel_c[1];
int2 CollTshiftpairs_c[1]; 
int WriteEncounters_c[1]; 
double WriteEncountersRadius_c[1]; 
int WriteEncountersCloudSize_c[1]; 
int StopAtEncounter_c[1]; 
double StopAtEncounterRadius_c[1]; 
double Asteroid_eps_c[1];
double Asteroid_rho_c[1];
double Asteroid_C_c[1];
double Asteroid_A_c[1];
double Asteroid_K_c[1];
double Asteroid_V_c[1];
double Asteroid_rmin_c[1];
double Asteroid_rdel_c[1];
double SolarConstant_c[1];
double Qpr_c[1];
double SolarWind_c[1];

double BSddt_c[8];		//time stepping factors in Bulirsch-Stoer method
double BSt0_c[8 * 8];

#endif


__host__ void Data::constantCopyDirectAcc(){
#if def_CPU == 0
	cudaMemcpyToSymbol(Rcut_c, &Rcut_h[0], sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(RcutSun_c, &RcutSun_h[0], sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(StopAtCollision_c, &P.StopAtCollision, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(StopMinMass_c, &P.StopMinMass, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(CollisionPrecision_c, &P.CollisionPrecision, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(CollTshift_c, &P.CollTshift, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(CollisionModel_c, &P.CollisionModel, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(WriteEncounters_c, &P.WriteEncounters, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(WriteEncountersRadius_c, &P.WriteEncountersRadius, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(WriteEncountersCloudSize_c, &P.WriteEncountersCloudSize, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(StopAtEncounter_c, &P.StopAtEncounter, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(StopAtEncounterRadius_c, &P.StopAtEncounterRadius, sizeof(double), 0, cudaMemcpyHostToDevice);

	int2 ij;
	ij.x = -1;
	ij.y = -1;
	cudaMemcpyToSymbol(CollTshiftpairs_c, &ij, sizeof(int2), 0, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(SolarConstant_c, &P.SolarConstant, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Qpr_c, &P.Qpr, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(SolarWind_c, &P.SolarWind, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Asteroid_eps_c, &P.Asteroid_eps, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Asteroid_rho_c, &P.Asteroid_rho, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Asteroid_C_c, &P.Asteroid_C, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Asteroid_A_c, &P.Asteroid_A, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Asteroid_K_c, &P.Asteroid_K, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Asteroid_V_c, &P.Asteroid_V, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Asteroid_rmin_c, &P.Asteroid_rmin, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Asteroid_rdel_c, &P.Asteroid_rdel, sizeof(double), 0, cudaMemcpyHostToDevice);
#else
	*Rcut_c = *Rcut_h;
	*RcutSun_c = *RcutSun_h;
	*StopAtCollision_c = P.StopAtCollision;
	*StopMinMass_c = P.StopMinMass;
	*CollisionPrecision_c = P.CollisionPrecision;
	*CollTshift_c = P.CollTshift;
	*CollisionModel_c = P.CollisionModel;
	*WriteEncounters_c = P.WriteEncounters;
	*WriteEncountersRadius_c = P.WriteEncountersRadius;
	*WriteEncountersCloudSize_c = P.WriteEncountersCloudSize;
	*StopAtEncounter_c = P.StopAtEncounter;
	*StopAtEncounterRadius_c = P.StopAtEncounterRadius;

	CollTshiftpairs_c[0].x = -1;
	CollTshiftpairs_c[0].y = -1;

	*SolarConstant_c = P.SolarConstant;
	*Qpr_c = P.Qpr;
	*SolarWind_c = P.SolarWind;
	*Asteroid_eps_c = P.Asteroid_eps;
	*Asteroid_rho_c = P.Asteroid_rho;
	*Asteroid_C_c = P.Asteroid_C;
	*Asteroid_A_c = P.Asteroid_A;
	*Asteroid_K_c = P.Asteroid_K;
	*Asteroid_V_c = P.Asteroid_V;
	*Asteroid_rmin_c = P.Asteroid_rmin;
	*Asteroid_rdel_c = P.Asteroid_rdel;
#endif
}


__host__ void Data::constantCopyBS(){

	double ddt[8];
	double t0[8 * 8];

	for(int n = 1; n <= 8; ++n){
		ddt[n-1] = 0.25 / (n*n);
	}

	for(int n = 1; n <= 8; ++n){
		for(int j = n-1; j >=1; --j){
			t0[(n-1) * 8 + (j -1)] = 1.0 / (ddt[j-1] - ddt[n-1]);
		}
	}
#if def_CPU == 0
	 cudaMemcpyToSymbol(BSddt_c, ddt, 8 * sizeof(double), 0, cudaMemcpyHostToDevice);
	 cudaMemcpyToSymbol(BSt0_c, t0, 8 * 8 * sizeof(double), 0, cudaMemcpyHostToDevice);
#else
	 memcpy(BSddt_c, ddt, 8 * sizeof(double));
	 memcpy(BSt0_c, t0, 8 * 8 * sizeof(double));
#endif
}


__global__ void initialb_kernel(int2 *Encpairs_d, int2 *Encpairs2_d, const int NBNencT){
	
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(id < NBNencT){
		Encpairs_d[id].x = -1;
		Encpairs_d[id].y = -1;
		
		Encpairs2_d[id].x = -1;
		Encpairs2_d[id].y = -1;
	}
}

/*
 __global__ void test_kernel(double4 *x4_d, double3 *a_d, int *index_d, const int N){
 
	int id = blockIdx.x * blockDim.x + threadIdx.x;
 
	if(id < N){
		 if(fabs(a_d[id].x) > 10) printf("test %d %.g\n", id, a_d[id].x);
 	}
 }
 */


__global__ void save_kernel(double4 *x4_d, double4 *v4_d, double4 *x4bb_d, double4 *v4bb_d, double4 *spin_d, double4 *spinbb_d, double *rcrit_d, double *rcritv_d, double *rcritbb_d, double *rcritvbb_d, int *index_d, int *indexbb_d, const int N, const int NconstT, const int SLevels, const int f){

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < N){
		if(f == 1){
			x4bb_d[id] = x4_d[id];
			v4bb_d[id] = v4_d[id];
			spinbb_d[id] = spin_d[id];
			indexbb_d[id] = index_d[id];
			for(int l = 0; l < SLevels; ++l){
				rcritbb_d[id + l * NconstT] = rcrit_d[id + l * NconstT];
				rcritvbb_d[id + l * NconstT] = rcritv_d[id + l * NconstT];
			}
		}
		if(f == -1){
			x4_d[id] = x4bb_d[id];
			v4_d[id] = v4bb_d[id];
			spin_d[id] = spinbb_d[id];
			index_d[id] = indexbb_d[id];
			for(int l = 0; l < SLevels; ++l){
				rcrit_d[id + l * NconstT] = rcritbb_d[id + l * NconstT];
				rcritv_d[id + l * NconstT] = rcritvbb_d[id + l * NconstT];
			}
		}
	}
}


__host__ int Data::beforeTimeStepLoop1(){

	int er;
#if def_CPU == 0
	cudaEventCreate(&KickEvent);	
	cudaStreamCreateWithFlags(&copyStream, cudaStreamNonBlocking);
	for(int st = 0; st < 12; ++st){
		cudaStreamCreate(&BSStream[st]);
	}
	for(int st = 0; st < 16; ++st){
		cudaStreamCreate(&hstream[st]);
	}
#endif


	//Allocate orbit data on Host and Device
	er = AllocateOrbit();
	if(er == 0) return 0;

	//allocate mapped memory//
	er = CMallocateOrbit();
	if(er == 0) return 0;

	//copy constant memory
	constantCopyDirectAcc();
	constantCopyBS();

	//Allocate aeGride
	constantCopy2();
	if(P.UseaeGrid == 1){
		er = GridaeAlloc();
		if(er == 0) return 0;
	}
	if(P.Usegas == 1){
		GasAlloc();
	}

	//Table for fastfg//
	er = FGAlloc();
	if(er == 0) return 0;

	//initialize memory//
	er = init();
	printf("\nInitialize Memory\n");

#if def_TTV > 0
  #if MCMC_NCOV > 0
	er = readMCMC_COV();
	if(er == 0) return 0;
  #endif
#endif

	cudaDeviceSynchronize();
	//read initial conditions//
	printf("\nRead Initial Conditions\n");
	er = ic();
	if(er == 0) return 0;
	printf("Initial Conditions OK\n");

#if USE_NAF == 1
	er = naf.alloc1(NT, N_h[0], Nsmall_h[0], Nst, P.tRestart, idt_h, ict_h, P.NAFn0, P.NAFnfreqs);
	if(er == 0) return 0;

	er = naf.alloc2(NT, N_h[0], Nsmall_h[0], Nst, GSF, P.NAFformat, P.tRestart, index_h);
	if(er == 0) return 0;
#endif


#if def_CPU == 0
	//Check warp size
	{
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, P.dev[0]);
		WarpSize = devProp.warpSize;
	}
#endif

	//remove ghost particles and reorder arrays//
	int NminFlag = remove();
	//remove stopped simulations//
	if(NminFlag > 0){
		stopSimulations();
		NminFlag = 0;
		if(Nst == 0)  return 0;
	}

	cudaDeviceSynchronize();
	error = cudaGetLastError();
	if(error != 0){
		fprintf(masterfile, "Start1 error = %d = %s\n",error, cudaGetErrorString(error));
		printf("Start1 error = %d = %s\n",error, cudaGetErrorString(error));
		return 0;
	}
	printf("Compute initial Energy\n");

	er = firstEnergy();
	if(er == 0) return 0;

	cudaDeviceSynchronize();


	printf("Write initial Energy\n");

	//write first output
	er = firstoutput(0);
	if(er == 0) return 0;

	if(P.IrregularOutputs == 1){
		er = firstoutput(1);
	}
	if(er == 0) return 0;
	printf("Energy OK\n");

	//read aeGrid at restart time step 
	if(P.UseaeGrid == 1){
		readGridae();
	}

	//Set Gas Disc and Gas Table
	if(P.Usegas == 1){
		printf("Set Gas Table\n");
		er = setGasDisk();
		if(er == 0) return 0;
		printf("Gas Table OK\n");
	}

	// Set Order and Coefficients of the symplectic integrator //
	SymplecticP(0);

	cudaDeviceSynchronize();

#if def_CPU == 0
	if(Nst == 1){

		//set default kernel parameters
		FTX = 128;
		RTX = 128;
		FrTX = 128;
		KP = 1;
		KTX = 1;
		KTY = 256;
		KP2 = 1;
		KTX2 = 1;
		KTY2 = 256;
		UseAcc = 1;
		UseBVH = 1;

		FILE *tuneFile;

		if(P.doTuning == 0){

			//check if tuneParameters file exists.
			tuneFile = fopen("tuningParameters.dat", "r");
			if(tuneFile == NULL){
				printf("tuningParameters.dat file not available, use default settings\n");
				GSF[0].logfile = fopen(GSF[0].logfilename, "a");
				fprintf(GSF[0].logfile, "tuningParameters.dat file not available, use default settings\n");
				fclose(GSF[0].logfile);
			}
			else{
				printf("Read tuningParameters.dat file\n");
			
				char sp[16];	
				int er = 0;

				for(int i = 0; i < 20; ++i){
					er = fscanf(tuneFile, "%s", sp);
					if(er <= 0) break;

					if(strcmp(sp, "FTX") == 0){
						fscanf(tuneFile, "%d", &FTX);
					}
					if(strcmp(sp, "RTX") == 0){
						fscanf(tuneFile, "%d", &RTX);
					}
					if(strcmp(sp, "KP") == 0){
						fscanf(tuneFile, "%d", &KP);
					}
					if(strcmp(sp, "KTX") == 0){
						fscanf(tuneFile, "%d", &KTX);
					}
					if(strcmp(sp, "KTY") == 0){
						fscanf(tuneFile, "%d", &KTY);
					}
					if(strcmp(sp, "KP2") == 0){
						fscanf(tuneFile, "%d", &KP2);
					}
					if(strcmp(sp, "KTX2") == 0){
						fscanf(tuneFile, "%d", &KTX2);
					}
					if(strcmp(sp, "KTY2") == 0){
						fscanf(tuneFile, "%d", &KTY2);
					}
					if(strcmp(sp, "FrTX") == 0){
						fscanf(tuneFile, "%d", &FrTX);
					}
					if(strcmp(sp, "UseAcc") == 0){
						fscanf(tuneFile, "%d", &UseAcc);
					}
					if(strcmp(sp, "UseBVH") == 0){
						fscanf(tuneFile, "%d", &UseBVH);
					}
				}

				fclose(tuneFile);

				printf("FTX %d\n", FTX);
				printf("RTX %d\n", RTX);
				printf("KP %d\n", KP);
				printf("KTX %d\n", KTX);
				printf("KTY %d\n", KTY);
				printf("KP2 %d\n", KP2);
				printf("KTX2 %d\n", KTX2);
				printf("KTY2 %d\n", KTY2);
				printf("FrTX %d\n", FrTX);
				printf("UseAcc %d\n", UseAcc);
				printf("UseBVH %d\n", UseBVH);

				GSF[0].logfile = fopen(GSF[0].logfilename, "a");
				fprintf(GSF[0].logfile, "Read tuningParameters.dat file\n");
				fprintf(GSF[0].logfile, "FTX %d\n", FTX);
				fprintf(GSF[0].logfile, "RTX %d\n", RTX);
				fprintf(GSF[0].logfile, "KP %d\n", KP);
				fprintf(GSF[0].logfile, "KTX %d\n", KTX);
				fprintf(GSF[0].logfile, "KTY %d\n", KTY);
				fprintf(GSF[0].logfile, "KP2 %d\n", KP2);
				fprintf(GSF[0].logfile, "KTX2 %d\n", KTX2);
				fprintf(GSF[0].logfile, "KTY2 %d\n", KTY2);
				fprintf(GSF[0].logfile, "FrTX %d\n", FrTX);
				fprintf(GSF[0].logfile, "UseAcc %d\n", UseAcc);
				fprintf(GSF[0].logfile, "UseBVH %d\n", UseBVH);
				fclose(GSF[0].logfile);
		
			}
		}
		else{
			//Tune kernel parameters
			er = tuneFG(FTX);
			if(er == 0) return 0;
			er = tuneRcrit(RTX);
			if(er == 0) return 0;
			if(P.UseTestParticles == 0){
				er = tuneKick(0, KP, KTX, KTY);
				if(er == 0) return 0;
			}
			if(P.UseTestParticles == 1){
				er = tuneKick(1, KP, KTX, KTY);
				if(er == 0) return 0;
			}
			if(P.UseTestParticles == 2){
				er = tuneKick(1, KP, KTX, KTY);
				if(er == 0) return 0;
				er = tuneKick(2, KP2, KTX2, KTY2);
				if(er == 0) return 0;
			}
			if(ForceFlag > 0){
				er = tuneForce(FrTX);
				if(er == 0) return 0;
			}
			if(P.WriteEncounters == 2){
				er = tuneBVH(UseBVH);
				if(er == 0) return 0;
			}


			tuneFile = fopen("tuningParameters.dat", "w");
			fprintf(tuneFile, "FTX %d\n", FTX);
			fprintf(tuneFile, "RTX %d\n", RTX);
			fprintf(tuneFile, "KP %d\n", KP);
			fprintf(tuneFile, "KTX %d\n", KTX);
			fprintf(tuneFile, "KTY %d\n", KTY);
			fprintf(tuneFile, "KP2 %d\n", KP2);
			fprintf(tuneFile, "KTX2 %d\n", KTX2);
			fprintf(tuneFile, "KTY2 %d\n", KTY2);
			fprintf(tuneFile, "FrTX %d\n", FrTX);
			fprintf(tuneFile, "UseAcc %d\n", UseAcc);
			fprintf(tuneFile, "UseBVH %d\n", UseBVH);
			fclose(tuneFile);
		}
		if(P.doSLTuning == 1){
			er = tuneBS();
			if(er == 0) return 0;
		}
	}
	else{
		//Nst > 1
		//set default kernel parameters
		KTM3 = 32;
		HCTM3 = 32;
		UseM3 = 0;

		FILE *tuneFile;

		if(P.doTuning == 0){

			//check if tuneParameters file exists.
			tuneFile = fopen("tuningParameters.dat", "r");
			if(tuneFile == NULL){
				printf("tuningParameters.dat file not available, use default settings\n");
				GSF[0].logfile = fopen(GSF[0].logfilename, "a");
				fprintf(GSF[0].logfile, "tuningParameters.dat file not available, use default settings\n");
				fclose(GSF[0].logfile);
			}
			else{
				printf("Read tuningParameters.dat file\n");
			
				char sp[16];	
				int er = 0;

				for(int i = 0; i < 20; ++i){
					er = fscanf(tuneFile, "%s", sp);
					if(er <= 0) break;
					if(strcmp(sp, "KTM3") == 0){
						fscanf(tuneFile, "%d", &KTM3);
					}
					if(strcmp(sp, "HCTM3") == 0){
						fscanf(tuneFile, "%d", &HCTM3);
					}
					if(strcmp(sp, "UseM3") == 0){
						fscanf(tuneFile, "%d", &UseM3);
					}
				}
				if(NBmax > NmaxM){
					UseM3 = 1;
				}

				fclose(tuneFile);
				printf("KTM3 %d\n", KTM3);
				printf("HCTM3 %d\n", HCTM3);
				printf("UseM3 %d\n", UseM3);

				GSF[0].logfile = fopen(GSF[0].logfilename, "a");
				fprintf(GSF[0].logfile, "Read tuningParameters.dat file\n");
				fprintf(GSF[0].logfile, "KTM3 %d\n", KTM3);
				fprintf(GSF[0].logfile, "HCTM3 %d\n", HCTM3);
				fprintf(GSF[0].logfile, "UseM3 %d\n", UseM3);
				fclose(GSF[0].logfile);
			}
		}
		else{
			//Tune kernel parameters
			er = tuneKickM3(KTM3);
			if(er == 0) return 0;

			er = tuneHCM3(HCTM3);
			if(er == 0) return 0;

			if(NBmax > NmaxM){
				UseM3 = 1;
			}

			tuneFile = fopen("tuningParameters.dat", "w");
			fprintf(tuneFile, "KTM3 %d\n", KTM3);
			fprintf(tuneFile, "HCTM3 %d\n", HCTM3);
			fprintf(tuneFile, "UseM3 %d\n", UseM3);
			fclose(tuneFile);
		}
	}
#else
	if(P.doSLTuning == 1){
		er = tuneBS();
		if(er == 0) return 0;
	}
	UseBVH = 2;
#endif 

	if(Nst == 1) printf("Start integration with %d simulation\n", Nst);
	else printf("Start integration with %d simulations\n", Nst);
	error = cudaGetLastError();
	if(error != 0){
		fprintf(masterfile, "Start2 error = %d = %s\n",error, cudaGetErrorString(error));
		printf("Start2 error = %d = %s\n",error, cudaGetErrorString(error));
		return 0;
	}



	fflush(masterfile);
#if USE_NAF == 1
	//compute the x and y arrays for the naf algorithm
	int NAFstep = 0;
	naf.getnafvarsCall(x4_d, v4_d, index_d, NBS_d, vcom_d, test_d, P.NAFvars, naf.x_d, naf.y_d, Msun_d, Msun_h[0].x, NT, Nst, naf.n, NAFstep, NB[0], N_h[0], Nsmall_h[0], P.UseTestParticles);
	++NAFstep;
#endif

	return 1;

}
// *****************************************************
// This function calls all necessary steps before the time step loop

// Authors: Simon Grimm
// February 2019
// ****************************************************
__host__ int Data::beforeTimeStepLoop(int ittv){

	int er;


	if(P.setElements > 0){
		er = readSetElements();
		if(er == 0){
			return 0;
		}
	}
	if(P.CreateParticles > 0){
		er = createReadFile2();
		if(er == 0){
			return 0;
		}
	}

	firstStep(0);

	cudaDeviceSynchronize();
	error = cudaGetLastError();
	if(error != 0){
		fprintf(masterfile, "first kick error = %d = %s\n",error, cudaGetErrorString(error));
		printf("first kick error = %d = %s\n", error, cudaGetErrorString(error));
		return 0;
	}
	else{
		if(ittv == 0) printf("first kick OK\n");
	}

	if(EncFlag_m[0] > 0){
		printf("Error: more encounters than allowed: %d %d. Increase 'Maximum encounter pairs'.\n", EncFlag_m[0], P.NencMax);
		fprintf(masterfile, "Error: more encounters than allowed: %d %d. Increase 'Maximum encounter pairs'.\n", EncFlag_m[0], P.NencMax);
		return 0;
	}


	//Print first informations about close encounter pairs
	if(ittv == 0) firstInfo();
	setStartTime();

#if def_poincareFlag == 1
	sprintf(poincarefilename, "%sPoincare%s_%.*d.dat", GSF[0].path, GSF[0].X, def_NFileNameDigits, 0);
	poincarefile = fopen(poincarefilename, "w");
#endif

	irrTimeStep = 0ll;
	irrTimeStepOut = 0ll;
	if(P.IrregularOutputs == 1){
		er = readIrregularOutputs();
		if(er == 0){
			return 0;
		}
		//skip Irregular output times which are before the simulation starts
		double starttime = (P.tRestart) * idt_h[0] + ict_h[0] * 365.25;
		for(long long int i = 0ll; i < NIrrOutputs; ++i){
			if(IrrOutputs[i] >= starttime){
				break;
			}
			++irrTimeStep;
			++irrTimeStepOut;
		}
	}
	if(P.UseTransits == 1 && ittv == 0){
		er = readTransits();
		if(er == 0){
			return 0;
		}
	}
	if(P.UseRV == 1 && ittv == 0){
		er = readRV();
		if(er == 0){
			return 0;
		}
	}


	if(P.Usegas == 2){
		er = readGasFile();
		er = readGasFile2(time_h[0] / 365.25);
		if(er == 0){
			return 0;
		}
	}

	bufferCount = 0;
	bufferCountIrr = 0;
	MultiSim = 0;
	if(Nst > 1) MultiSim = 1;
	interrupt = 0;

	return 1;

}


// *****************************************************
// This function calls all necessary sub steps for computing 
// one time step.
// Authors: Simon Grimm
// March 2017
// ****************************************************
__host__ int Data::timeStepLoop(int interrupted, int ittv){
	time_h[0] = timeStep * idt_h[0] + ict_h[0] * 365.25;
	
	int er;	
	er = step(0);
	if(er == 0){
		return 0;
	}
	if(doTransits == 0 && timeStep == P.tRestart + 1){
		if(P.ei == 0 || (P.ei != 0 && timeStep % P.ei != 0)){
			firstInfoB();
		}
	}
	
	if(interrupted == 1){
		printf("GENGA is interrupted by SIGINT signal at time step %lld\n", timeStep);
		fprintf(masterfile, "GENGA is interrupted by SIGINT signal at time step %lld\n", timeStep);
		interrupt = 1;
	}

	if(ErrorFlag_m[0] > 0){
		printf("Error detected, GENGA stopped\n");
		fprintf(masterfile, "Error detected, GENGA stopped\n");
		return 0;
	}

	//cudaEventSynchronize(KickEvent);
	//do not synchronize here, to save time. but that means that the error message could be delayed
	//Check for too many encounters
	if(EncFlag_m[0] > 0){
		printf("Error: more encounters than allowed: %d %d. Increase 'Maximum encounter pairs'.\n", EncFlag_m[0], P.NencMax);
		fprintf(masterfile, "Error: more encounters than allowed: %d %d. Increase 'Maximum encounter pairs'.\n", EncFlag_m[0], P.NencMax);
		return 0;
	}
	
	//Check for too big groups//
	if(Nst == 1){
		er = MaxGroups();
		if(er == 0) return 0;
	}

	if(interrupt == 1){
		if(Nst == 1){
			RemoveCall();
		}
	}

	//Print Energy and log information//
	int CallEnergy = 0;
	if(interrupt == 1) CallEnergy = 1;
	if(P.ei != 0 && timeStep == P.deltaT) CallEnergy = 1;
	if(P.ci != 0 && timeStep == P.deltaT) CallEnergy = 1;
	if(P.ci > 0 && timeStep % P.ci == 0) CallEnergy = 1;

	if((P.ei > 0 && timeStep % P.ei == 0) || CallEnergy == 1){
		if(bufferCount + 1 >= P.Buffer || CallEnergy == 1){
			er = EnergyOutput(0);
			if(er == 0) return 0;
		}
	}
	
	if(P.UseaeGrid == 1){ 
		if(timeStep % 10000 == 0){
			er = copyGridae();
			if(er == 0){
				return 0;
			}
		}
	}
	//update Gas Disk
	if(P.Usegas == 2 && time_h[0] / 365.25 > GasDatatime.y){
		er = readGasFile2(time_h[0] / 365.25);
		if(er == 0){
			return 0;
		}
	}
	
//test_kernel <<< 1, 16 >>> (x4_d, v4_d, index_d);
	
	//Print Output//
	if((P.ci > 0 && ((timeStep - 1) % P.ci >= P.ci - P.nci)) || interrupt == 1 || (P.ci != 0 && timeStep == P.deltaT)){
		if(P.Buffer == 1){
			CoordinateOutput(0);
		}
		else if(bufferCount + 1 >= P.Buffer || interrupt == 1){
			//write out buffer
			timestepBuffer[bufferCount] = timeStep;
			for(int st = 0; st < Nst; ++st){
				NBuffer[Nst * (bufferCount) + st].x = N_h[st];
				NBuffer[Nst * (bufferCount) + st].y = Nsmall_h[st];
			}
			CoordinateToBuffer(bufferCount, 0, 0.0);
			++bufferCount;	
			CoordinateOutputBuffer(0);
		}
		else{
			//store in buffer
			timestepBuffer[bufferCount] = timeStep;
			for(int st = 0; st < Nst; ++st){
				NBuffer[Nst * (bufferCount) + st].x = N_h[st];
				NBuffer[Nst * (bufferCount) + st].y = Nsmall_h[st];
			}
			CoordinateToBuffer(bufferCount, 0, 0.0);
			++bufferCount;	
		}
		if(P.UseaeGrid == 1){
			GridaeOutput();
		}
		
#if def_poincareFlag == 1
		if((timeStep - 1) % P.ci == P.ci - P.nci){
			fclose(poincarefile);
			sprintf(poincarefilename, "%sPoincare%s_%.*lld.dat", GSF[0].path, GSF[0].X, def_NFileNameDigits, timeStep);
			//Erase old Poincare files
			poincarefile = fopen(poincarefilename, "w");
		}
#endif
	}
	
	//print irregular outputs
	if(interrupt == 1 && P.Buffer > 1){
		//write out buffer
		CoordinateOutputBuffer(1);
	}
	if(P.IrregularOutputs == 1 && irrTimeStep < NIrrOutputs && time_h[0] >= IrrOutputs[irrTimeStep]){
		
		int ni = 1; //multiple outputs per time step
		for(int i = 0; i < ni; ++i){
			double dTau = -(time_h[0] - IrrOutputs[irrTimeStep]) / idt_h[0];
			IrregularStep(dTau);
			for(int st = 0; st < Nst; ++st){
				time_h[st] += dTau * idt_h[st];
			}
			if(Nst > 1){
				cudaMemcpy(time_d, time_h, Nst * sizeof(double), cudaMemcpyHostToDevice);
			}
			
			step(0);
			
			if(P.Buffer == 1){
				CoordinateOutput(1);
				int er = EnergyOutput(1);
				if(er == 0) return 0;
			}
			else if(bufferCountIrr + 1 >= P.Buffer){
				//write out buffer
				timestepBufferIrr[bufferCountIrr] = timeStep;
				for(int st = 0; st < Nst; ++st){
					NBufferIrr[Nst * (bufferCountIrr) + st].x = N_h[st];
					NBufferIrr[Nst * (bufferCountIrr) + st].y = Nsmall_h[st];
				}
				CoordinateToBuffer(bufferCountIrr, 1, dTau);
				++bufferCountIrr;
				CoordinateOutputBuffer(1);
				bufferCountIrr = 0;
				irrTimeStepOut += P.Buffer;
			}
			else{
				//store in buffer
				timestepBufferIrr[bufferCountIrr] = timeStep;
				for(int st = 0; st < Nst; ++st){
					NBufferIrr[Nst * (bufferCountIrr) + st].x = N_h[st];
					NBufferIrr[Nst * (bufferCountIrr) + st].y = Nsmall_h[st];
				}
				CoordinateToBuffer(bufferCountIrr, 1, dTau);
				++bufferCountIrr;
			}
			
			IrregularStep(-dTau);
			for(int st = 0; st < Nst; ++st){
				time_h[st] -= dTau * idt_h[st];
			}
			if(Nst > 1){
				cudaMemcpy(time_d, time_h, Nst * sizeof(double), cudaMemcpyHostToDevice);
			}
			
			step(0);
			SymplecticP(1);
			
			++irrTimeStep;
			
			dTau = -(time_h[0] - IrrOutputs[irrTimeStep]) / idt_h[0];
			if(dTau <= 0) ++ni;
			
			if(ni + irrTimeStep - 1 > NIrrOutputs) break;
		}
	}
	
#if USE_NAF == 1
	//compute the x and y arrays for the naf algorithm
	if(timeStep % P.NAFinterval == 0){
		naf.getnafvarsCall(x4_d, v4_d, index_d, NBS_d, vcom_d, test_d, P.NAFvars, naf.x_d, naf.y_d, Msun_d, Msun_h[0].x, NT, Nst, naf.n, NAFstep, NB[0], N_h[0], Nsmall_h[0], P.UseTestParticles);
		++NAFstep;
		if(NAFstep % P.NAFn0 == 0){
			er = naf.nafCall(NT, N_h, N_d, Nsmall_h, Nsmall_d, Nst, GSF, time_h, time_d, idt_h, P.NAFformat, P.NAFinterval, index_h, index_d, NBS_h);
			if(er == 0) return 0;
			NAFstep = 0;
		}
	}
#endif
	// print time information //
	// this should be the last thing to print, because it is used to restart at the last possible timestep
	if((P.ci > 0 && timeStep % P.ci == 0) || interrupt == 1){
		if(bufferCount >= P.Buffer || P.Buffer == 1 || interrupt == 1){
			er = printTime(0);
			if(er == 0) return 0;
			fflush(masterfile);
			bufferCount = 0;
		}
	}
	if(interrupt == 1){
		printf("GENGA is terminated by SIGINT signal at time step %lld\n", timeStep);
		fprintf(masterfile, "GENGA is terminated by SIGINT signal at time step %lld\n", timeStep);
		cudaDeviceSynchronize();
		return 0;
	}
	
	error = cudaGetLastError();
	if(error != 0){
		printf("Step error = %d = %s at time step: %lld\n",error, cudaGetErrorString(error), timeStep);
		fprintf(masterfile, "Step error = %d = %s at time step: %lld\n",error, cudaGetErrorString(error), timeStep);
		CoordinateOutput(4);
		return 0;
	}
	
	
	return 1;
	
}
// *****************************************************
// This function calls all necessary steps after the main loop
//
// Authors: Simon Grimm
// FEbruary 2019
// ****************************************************
__host__ int Data::Remaining(){

	int er;
#if def_CPU == 0
	cudaEventDestroy(KickEvent);	
	cudaStreamDestroy(copyStream);	
	for(int st = 0; st < 12; ++st){
		cudaStreamDestroy(BSStream[st]);
	}
	for(int st = 0; st < 16; ++st){
		cudaStreamDestroy(hstream[st]);
	}
#endif

	error = cudaGetLastError();
	if(error != 0){
		printf("Stream error = %d = %s %lld\n",error, cudaGetErrorString(error), timeStep);
		return 0;
	}
	

	//write out the remaining buffer
	if(P.IrregularOutputs == 1){
		if(bufferCountIrr > 1){
			CoordinateOutputBuffer(1);
		}
	}
	if(bufferCount > 0){
		CoordinateOutputBuffer(0);
	}

#if def_poincareFlag == 1
	fclose(poincarefile);
#endif


	//print last informations
	printLastTime(0);
	LastInfo();

	//free all the memory on the Host and on the Device
	er = freeOrbit();
	if(er == 0) return 0;

	if(P.UseaeGrid == 1){
		free(Gridaecount_h);
		cudaFree(Gridaecount_d);
	}

	if(P.Usegas == 1){
		er = freeGas();
		if(er == 0) return 0;
	}


#if USE_NAF == 1
		er = naf.naffree();
		if(er == 0) return 0;
#endif


	er = freeHost();
	if(er == 0) return 0;

	printf("GENGA terminated successfully\n");
	fprintf(masterfile, "GENGA terminated successfully\n");

	return 1;

}


// *****************************************************
// This function set the time factors fot the symplectic integrator for a given order
// The first time it must be called with E = 0, afterwards with E = 1
// Authors: Simon Grimm
// June 2015
// ****************************************************
__host__ void Data::SymplecticP(int E){
	SIn = 1;
	SIM = 1;
	double SIw[4]; //for maximal SI6
	
	//second order
	if(P.SIO == 2){
		//SI2
		SIn = 1;
		SIM = (SIn + 1) / 2;
		
		SIw[0] = 1.0;
	}
	//4th order
	//From Yoshida
	if(P.SIO == 4){
		//SI4
		SIn = 3;
		SIM = (SIn + 1) / 2;
		
		double two3r = cbrt(2.0);
		SIw[0] = - two3r / (2.0 - two3r);
		SIw[1] = 1.0 / (2.0 - two3r);
		
	}
	//6th order
	if(P.SIO == 6){
		//SI6
		SIn = 7;
		SIM = (SIn + 1) / 2;
		
		//Solution A from Yoshida
		SIw[1] = -0.117767998417887e1;
		SIw[2] = 0.235573213359357e0;
		SIw[3] = 0.784513610477560e0;
		SIw[0] = 1.0 - 2.0 * (SIw[1] + SIw[2] + SIw[3]);
	}
	if(E == 0){
		Ct = (double*)malloc(SIn*sizeof(double));
		FGt = (double*)malloc(SIn*sizeof(double));
		Kt = (double*)malloc(SIn*sizeof(double));
	}
	
	for(int sim = 0; sim < SIM; ++sim){
		FGt[sim] = SIw[SIM - sim - 1];
	}
	for(int sim = SIM; sim < SIn; ++sim){
		FGt[sim] = SIw[sim - SIM + 1];
	}
	
	for(int si = 0; si < SIn; ++si){
		Ct[si] = 0.5 * FGt[si];
	}
	for(int si = 0; si < SIn - 1; ++si){
		Kt[si] = 0.5 * (FGt[si] + FGt[si + 1]);
	}
	Kt[SIn - 1] = 0.5 * FGt[SIn - 1];

//	for(int si = 0; si <= SIn - 1; ++si){
//		printf("%d %g\n", si, Kt[si]);
//	}
}

// *****************************************************
// This function sets the time factors for an irregular output step
// dTau is the modified time step
// Authors: Simon Grimm
// June 2015
// ****************************************************
__host__ void Data::IrregularStep(double dTau){
	SIn = 1;
	SIM = 1;
	
	FGt[0] = dTau;
	Ct[0] = dTau * 0.5;
	Kt[0] = dTau * 0.5;
	
}






__host__ int Data::tuneFG(int &TX){

	int ttx[4] = {32, 64, 128, 256};

	TX = 0;

	int NN = N_h[0] + Nsmall_h[0];

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float times;
	float timesMin = 1000000.0f;

	GSF[0].logfile = fopen(GSF[0].logfilename, "a");
	printf("Starting FG kernel parameters tuning\n");
	fprintf(GSF[0].logfile, "Starting FG kernel parameters tuning\n");
	for(int i = 0; i < 4; ++i){
		int tx = ttx[i];
		cudaEventRecord(start, 0);
		//revert fg operation
		//launch with si = -1
		fg_kernel <<< (NN + tx - 1) / tx, tx >>> (x4_d, v4_d, xold_d, vold_d, index_d, dt_h[0] * FGt[0], Msun_h[0].x, NN, aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, -1, P.UseGR);
		
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&times, start, stop); //time in microseconds
		error = cudaGetLastError();
		if(error == 7){
			//skip choices with too many resources requested
			continue;

		}
		if(error != 0 && error != 7){
			fprintf(masterfile, "FG Tune error = %d = %s\n",error, cudaGetErrorString(error));
			printf("FG Tune error = %d = %s\n",error, cudaGetErrorString(error));
			return 0;
		}
		printf("tx:%d    \ttime: %.15f s\n", tx, times * 0.001);	//time in seconds
		fprintf(GSF[0].logfile,"\ttx:%d    \ttime: %.15f s\n", tx, times * 0.001);	//time in seconds
		if(times < timesMin){
			TX = tx;
		}
		timesMin = fmin(times, timesMin);
	}
	if(TX == 0){
		printf("FG kernel tunig failed\n");
		fprintf(masterfile, "FG kernel tunig failed\n");
		return 0;
	}
	printf("Best parameters: tx:%d\t time: %.15f s\n", TX, timesMin * 0.001);	//time in seconds
	fprintf(GSF[0].logfile, "Best parameters: tx:%d\t time: %.15f s\n", TX, timesMin * 0.001);	//time in seconds
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	fclose(GSF[0].logfile);
	return 1;
}


__host__ int Data::tuneRcrit(int &TX){

	int ttx[4] = {32, 64, 128, 256};

	TX = 0;

	int NN = N_h[0] + Nsmall_h[0];

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float times;
	float timesMin = 1000000.0f;

	GSF[0].logfile = fopen(GSF[0].logfilename, "a");
	printf("Starting Rcrit kernel parameters tuning\n");
	fprintf(GSF[0].logfile, "Starting Rcrit kernel parameters tuning\n");

	for(int i = 0; i < 4; ++i){
		int tx = ttx[i];
		cudaEventRecord(start, 0);	
		Rcrit_kernel <<< (NN + tx - 1) / tx, tx >>> (x4_d, v4_d, x4b_d, v4b_d, spin_d, spinb_d, 1.0 / (3.0 * Msun_h[0].x), rcrit_d, rcritb_d, rcritv_d, rcritvb_d, index_d, indexb_d, dt_h[0], n1_h[0], n2_h[0], time_d, time_h[0], EjectionFlag_d, NN, NconstT, P.SLevels, 0);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&times, start, stop); //time in microseconds
		error = cudaGetLastError();
		if(error != 0){
			fprintf(masterfile, "Rcrit Tune error = %d = %s\n",error, cudaGetErrorString(error));
			printf("Rcrit Tune error = %d = %s\n",error, cudaGetErrorString(error));
			return 0;
		}
		printf("tx:%d    \ttime: %.15f s\n", tx, times * 0.001);	//time in seconds
		fprintf(GSF[0].logfile,"\ttx:%d    \ttime: %.15f s\n", tx, times * 0.001);	//time in seconds
		if(times < timesMin){
			TX = tx;
		}
		timesMin = fmin(times, timesMin);
	}
	if(TX == 0){
		printf("Rcrit kernel tunig failed\n");
		fprintf(masterfile, "Rcrit kernel tunig failed\n");
		return 0;
	}
	printf("Best parameters: tx:%d\t time: %.15f s\n", TX, timesMin * 0.001);	//time in seconds
	fprintf(GSF[0].logfile, "Best parameters: tx:%d\t time: %.15f s\n", TX, timesMin * 0.001);	//time in seconds
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	fclose(GSF[0].logfile);
	return 1;
}

__host__ int Data::tuneForce(int &TX){

	int ttx[5] = {32, 64, 128, 256, 512};

	TX = 0;

	int NN = N_h[0] + Nsmall_h[0];

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float times;
	float timesMin = 1000000.0f;

	GSF[0].logfile = fopen(GSF[0].logfilename, "a");
	printf("Starting Force kernel parameters tuning\n");
	fprintf(GSF[0].logfile, "Starting Force kernel parameters tuning\n");

	for(int i = 0; i < 5; ++i){
		int tx = ttx[i];
		cudaEventRecord(start, 0);
		int nn = (NN + tx - 1) / tx;
		force_kernel <<< nn, tx, WarpSize * sizeof(double3) >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, Lovesun_d, J2_d, vold_d, dt_d, Kt[SIn - 1], time_d, NN, Nst, P.UseGR, P.UseTides, P.UseRotationalDeformation, 0, 0);

		if(P.UseTides > 0 || P.UseRotationalDeformation > 0){
			int ncb = min(nn, 1024);
			if(N_h[0] + Nsmall_h[0] > tx){
				forced2_kernel <<< 1, ((ncb + WarpSize - 1) / WarpSize) * WarpSize, WarpSize * sizeof(double3) >>> (vold_d, Spinsun_d, nn, 0);
			}
		}

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&times, start, stop); //time in microseconds
		error = cudaGetLastError();
		if(error == 701){
			//skip choices with too many resources requested
			continue;

		}
		if(error != 0 && error != 701){
			fprintf(masterfile, "Force Tune error = %d = %s\n",error, cudaGetErrorString(error));
			printf("Force Tune error = %d = %s\n",error, cudaGetErrorString(error));
			return 0;
		}
		printf("tx:%d    \ttime: %.15f s\n", tx, times * 0.001);	//time in seconds
		fprintf(GSF[0].logfile,"\ttx:%d    \ttime: %.15f s\n", tx, times * 0.001);	//time in seconds
		if(times < timesMin){
			TX = tx;
		}
		timesMin = fmin(times, timesMin);
	}
	if(TX == 0){
		printf("Force kernel tunig failed\n");
		fprintf(masterfile, "Force kernel tunig failed\n");
		return 0;
	}
	printf("Best parameters: tx:%d\t time: %.15f s\n", TX, timesMin * 0.001);	//time in seconds
	fprintf(GSF[0].logfile, "Best parameters: tx:%d\t time: %.15f s\n", TX, timesMin * 0.001);	//time in seconds
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	fclose(GSF[0].logfile);
	return 1;
}


// *******************************************************************
// This function tests different kernel parameters for the Kick kernel, 
// and times them. It selects the fastest configuration and sets the
// values in KPP, KTX and KTY.
//
// Date: April 2019
// Author: Simon Grimm
__host__ int Data::tuneKick(int EE, int &PP, int &TX, int &TY){

	int pp[3] =  {1,2,4};
	int ttx[8] = {1, 32, 128, 256};
	int tty[9] = {256, 128, 64, 32, 16, 8, 4, 2, 1};


	PP = 0;
	TX = 0;
	TY = 0;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float times;
	float timesMin = 1000000.0f;

	//limit the tuning test to 8192. More particles behave most likely the same
	//This is to save some time at the beginning
	int NN, N0, N1;
	if(EE == 0){
		NN = min(N_h[0], 8192);
		N1 = NN;
		N0 = 0;
	}
	if(EE == 1){
		NN = N_h[0] + Nsmall_h[0];
		N1 = N_h[0];
		N0 = 0;
	}
	if(EE == 2){
		NN = N_h[0];
		N1 = min(N_h[0] + Nsmall_h[0], 8192);	//TP 1 has to be run before
		N0 = N_h[0];
	}
	int f = 0; //flag for comparison

	int T = 1;	//number of kick calls
	if(NN <= 256)  T = 100;
	if(NN <= 1024)  T = 10;


	//test if kick kernel or acc kernel is faster
	float kickTime = -1.0f;
	float accTime = -1.0f;

	GSF[0].logfile = fopen(GSF[0].logfilename, "a");
	printf("\nStarting Kick kernel parameters tuning\n");
	fprintf(GSF[0].logfile, "\nStarting Kick kernel parameters tuning\n");
	
	Rcrit_kernel <<< (NN + 255) / 256, 256 >>> (x4_d, v4_d, x4b_d, v4b_d, spin_d, spinb_d, 1.0 / (3.0 * Msun_h[0].x), rcrit_d, rcritb_d, rcritv_d, rcritvb_d, index_d, indexb_d, dt_h[0], n1_h[0], n2_h[0], time_d, time_h[0], EjectionFlag_d, NN, NconstT, P.SLevels, 0);


	if(EE == 0){
		if(NB[0] <= WarpSize){
			cudaEventRecord(start, 0);	
			for(int t = 0; t < T; ++t){
				EncpairsZeroC_kernel <<< (NN + 255) / 256, 256 >>> (Encpairs2_d, a_d, Nencpairs_d, Nencpairs2_d, P.NencMax, NN);
				if(P.KickFloat == 0){
					kick16c_kernel <<< NN, NB[0] >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, EncFlag_d, P.NencMax, NN, 0);
				}
				else{
					kick16cf_kernel <<< NN, NB[0] >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, EncFlag_d, P.NencMax, NN, 0);
				}
			}
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&times, start, stop); //time in microseconds
			printf("kick16c time: %.15f s\n", times * 0.001);	//time in seconds
			fprintf(GSF[0].logfile, "kick16c time: %.15f s\n", times * 0.001);	//time in seconds
			kickTime = times;
		}		
		else if(NB[0] <= 1024){
			cudaEventRecord(start, 0);	
			for(int t = 0; t < T; ++t){
				EncpairsZeroC_kernel <<< (NN + 255) / 256, 256 >>> (Encpairs2_d, a_d, Nencpairs_d, Nencpairs2_d, P.NencMax, NN);
				if(P.KickFloat == 0){
					kick32c_kernel <<< NN, NB[0], 2 * WarpSize * sizeof(double3) >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, EncFlag_d, P.NencMax, NN, 0);
				}
				else{
					kick32cf_kernel <<< NN, NB[0], 2 * WarpSize * sizeof(float3) >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, EncFlag_d, P.NencMax, NN, 0);
				}
			}
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&times, start, stop); //time in microseconds
			printf("kick32c time: %.15f s\n", times * 0.001);	//time in seconds
			fprintf(GSF[0].logfile, "kick32c time: %.15f s\n", times * 0.001);	//time in seconds
			kickTime = times;
		}		
	}

	for(int i = 0; i < 3; ++i){
		for(int j = 0; j < 4; ++j){
			for(int k = 0; k < 9; ++k){
	//for(int i = 0; i < 1; ++i){
	//	for(int j = 0; j < 1; ++j){
	//		for(int k = 0; k < 1; ++k){
				int p = pp[i];
				int tx = ttx[j];
				int ty = tty[k];

				if(tx * ty > 512) continue;
				if(tx * ty < min(32, NN)) continue;

				//set Encpairs to zero
				cudaEventRecord(start, 0);	
				for(int t = 0; t < T; ++t){
					EncpairsZeroC_kernel <<< (NN + 255) / 256, 256 >>> (Encpairs2_d, a_d, Nencpairs_d, Nencpairs2_d, P.NencMax, NN);
					if(P.KickFloat == 0){
						acc4C_kernel <<< dim3( (((NN + p - 1)/ p) + tx - 1) / tx, 1, 1), dim3(tx,ty,1), tx * ty * p * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, NN, N0, N1, P.NencMax, p, EE);
					}
					else{
						acc4Cf_kernel <<< dim3( (((NN + p - 1)/ p) + tx - 1) / tx, 1, 1), dim3(tx,ty,1), tx * ty * p * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, NN, N0, N1, P.NencMax, p, EE);
					}
				}

				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&times, start, stop); //time in microseconds
				error = cudaGetLastError();
				if(error == 7 || error == 701){
					//skip choices with too many resources requested
					continue;

				}
				if(error != 0 && error != 7){
					fprintf(masterfile, "Kick Tune error = %d = %s\n",error, cudaGetErrorString(error));
					printf("Kick Tune error = %d = %s\n",error, cudaGetErrorString(error));
					return 0;
				}
				if(EE < 2){
					printf("p:%d    \ttx:%d    \tty:%d    \ttime: %.15f s\n", p, tx, ty, times * 0.001);	//time in seconds
					fprintf(GSF[0].logfile,"p:%d    \ttx:%d    \tty:%d    \ttime: %.15f s\n", p, tx, ty, times * 0.001);	//time in seconds
				}
				else{
					printf("p2:%d   \ttx2:%d   \tty2:%d   \ttime: %.15f s\n", p, tx, ty, times * 0.001);	//time in seconds
					fprintf(GSF[0].logfile,"p2:%d   \ttx2:%d   \tty2:%d   \ttime: %.15f s\n", p, tx, ty, times * 0.001);	//time in seconds
				}
				if(times < timesMin){
					PP = p;
					TX = tx;
					TY = ty;
				}
				timesMin = fmin(times, timesMin);
				//check if different tunig parameters give the same result
				compare_a_kernel <<< (NN + 255) / 256, 256 >>> (a_d, ab_d, P.KickFloat, NN, f);
				if(f == 0) f = 1;
			}
		}

	}
	if(PP == 0){
		printf("Kick kernel tuning failed\n");
		fprintf(masterfile, "Kick kernel tuning failed\n");
		return 0;
	}
	if(EE < 2){
		printf("Best parameters: p:%d\t tx:%d\t ty:%d\t time: %.15f s\n", PP, TX, TY, timesMin * 0.001);	//time in seconds
		fprintf(GSF[0].logfile, "Best parameters: p:%d\t tx:%d\t ty:%d\t time: %.15f s\n", PP, TX, TY, timesMin * 0.001);	//time in seconds
	}
	else{
		printf("Best parameters: p2:%d\t tx2:%d\t ty2:%d\t time: %.15f s\n", PP, TX, TY, timesMin * 0.001);	//time in seconds
		fprintf(GSF[0].logfile, "Best parameters: p2:%d\t tx2:%d\t ty2:%d\t time: %.15f s\n", PP, TX, TY, timesMin * 0.001);	//time in seconds
	}

	//test now the total time for the kick operation	
	cudaEventRecord(start, 0);	
	for(int t = 0; t < T; ++t){
		EncpairsZeroC_kernel <<< (NN + 255) / 256, 256 >>> (Encpairs2_d, a_d, Nencpairs_d, Nencpairs2_d, P.NencMax, NN);
		if(P.KickFloat == 0){
			acc4C_kernel <<< dim3( (((NN + PP - 1)/ PP) + TX - 1) / TX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, NN, N0, N1, P.NencMax, KP, EE);
		}
		else{
			acc4Cf_kernel <<< dim3( (((NN + PP - 1)/ PP) + TX - 1) / TX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, NN, N0, N1, P.NencMax, KP, EE);
		}
		kick32Ab_kernel <<< (NN + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs2_d, 0, NN, P.NencMax, 0);
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&times, start, stop); //time in microseconds
	printf("Total acc4C + kick32Ab time: %.15f s\n", times * 0.001);	//time in seconds
	fprintf(GSF[0].logfile, "Total acc4C + kick32Ab time: %.15f s\n", times * 0.001);	//time in seconds
	accTime = times;

	if(EncFlag_m[0] > 0){
		printf("Error: more encounters than allowed: %d %d. Increase 'Maximum encounter pairs'.\n", EncFlag_m[0], P.NencMax);
		fprintf(masterfile, "Error: more encounters than allowed: %d %d. Increase 'Maximum encounter pairs'.\n", EncFlag_m[0], P.NencMax);
		return 0;
	}

	if(kickTime > 0.0f && accTime > 0.0f){
		if(kickTime < accTime){
			UseAcc = 0;
		}
	}
	if(kickTime <= 0.0f && accTime <= 0.0f){
		printf("Error in Kick tuning, %g %g\n", kickTime, accTime);
		return 0;
	}

	if(UseAcc == 1){
		printf("Use acc kernel\n");
		fprintf(GSF[0].logfile, "Use acc kernel\n");
	}
	else{
		printf("Use kick kernel\n");
		fprintf(GSF[0].logfile, "Use kick kernel\n");
	}

	if(P.SERIAL_GROUPING == 1){
		printf("Using Serial Grouping, this disables the tuning parameters\n");
		fprintf(GSF[0].logfile, "Using Serial Grouping, this disables the tuning parameters\n");
		if(EE == 0 || EE == 2){
			PP = 4;
			TX = 1;
			TY = 128;
		}
		else{
			PP = 1;
			TX = 64;
			TY = 1;
		}
	}

	//Set again the Encpairs arrays to zero
	EncpairsZeroC_kernel <<< (N_h[0] + Nsmall_h[0] + 255) / 256, 256 >>> (Encpairs2_d, a_d, Nencpairs_d, Nencpairs2_d, P.NencMax, N_h[0] + Nsmall_h[0]);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	fclose(GSF[0].logfile);
	return 1;
}

__host__ int Data::tuneKickM3(int &KTM3){

	int ttM3[3] =  {32, 64, 128};

	KTM3 = 32;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float times;
	float timesMin = 1000000.0f;

	int T = 10;	//number of kick calls
	int f = 0; //flag for comparison

	GSF[0].logfile = fopen(GSF[0].logfilename, "a");
	printf("\nStarting KickM3 kernel parameters tuning\n");
	fprintf(GSF[0].logfile, "\nStarting KickM3 kernel parameters tuning\n");
	
	RcritM_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, x4b_d, v4b_d, spin_d, spinb_d, Msun_d, rcrit_d, rcritb_d, rcritv_d, rcritvb_d, dt_d, n1_d, n2_d, Rcut_d, RcutSun_d, EjectionFlag_d, index_d, indexb_d, Nst, NT, time_d, idt_d, ict_d, delta_d, timeStep, StopFlag_d, NconstT, P.SLevels, 0, Nstart);

	for(int i = 0; i < 3; ++i){
		int tM3 = ttM3[i];
		cudaEventRecord(start, 0);	
		for(int t = 0; t < T; ++t){

			KickM3_kernel <<< NT, dim3(tM3, 1, 1), WarpSize * sizeof(double3) >>> (x4_d, v4_d, a_d, rcritv_d, Nencpairs_d, Encpairs_d, dt_d, Kt[SIn - 1], index_d, NT, N_d, NBS_d, 0, 0);
		}

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&times, start, stop); //time in microseconds

		printf("KTM3:%d   \ttime: %.15f s\n", tM3, times * 0.001);	//time in seconds
		fprintf(GSF[0].logfile,"KTM3:%d   \ttime: %.15f s\n", tM3, times * 0.001);	//time in seconds
		if(times < timesMin){
			KTM3 = tM3;
		}
		timesMin = fmin(times, timesMin);
		//check if different tunig parameters give the same result
		compare_a_kernel <<< (NT + 255) / 256, 256 >>> (a_d, ab_d, P.KickFloat, NT, f);
		if(f == 0) f = 1;
	}

	if(EncFlag_m[0] > 0){
		printf("Error: more encounters than allowed: %d %d. Increase 'Maximum encounter pairs'.\n", EncFlag_m[0], P.NencMax);
		fprintf(masterfile, "Error: more encounters than allowed: %d %d. Increase 'Maximum encounter pairs'.\n", EncFlag_m[0], P.NencMax);
		return 0;
	}

	if(timesMin <= 0.0f){
		printf("Error in KickM3 tuning, %g\n", timesMin);
		return 0;
	}

	if(P.SERIAL_GROUPING == 1){
		printf("Using Serial Grouping, this disables the tuning parameters\n");
		fprintf(GSF[0].logfile, "Using Serial Grouping, this disables the tuning parameters\n");
		KTM3 = 32;
	}

	printf("Best parameters: KTM3:%d\t time: %.15f s\n", KTM3, timesMin * 0.001);	//time in seconds
	fprintf(GSF[0].logfile, "Best parameters: KTM3:%d\t time: %.15f s\n", KTM3, timesMin * 0.001);	//time in seconds

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	fclose(GSF[0].logfile);
	return 1;
}
__host__ int Data::tuneHCM3(int &HCTM3){

	int hhcM3[3] =  {32, 64, 128};

	HCTM3 = 32;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float times;
	float timesMin = 1000000.0f;

	int T = 10;	//number of kick calls

	GSF[0].logfile = fopen(GSF[0].logfilename, "a");
	printf("\nStarting HCM3 kernel parameters tuning\n");
	fprintf(GSF[0].logfile, "\nStarting HCM3 kernel parameters tuning\n");
	
	for(int i = 0; i < 3; ++i){
		int hcM3 = hhcM3[i];
		cudaEventRecord(start, 0);	
		for(int t = 0; t < T; ++t){
			HC32aM_kernel <<< Nst, hcM3, WarpSize * sizeof(double3) >>> (x4_d, v4_d, dt_d, Msun_d, N_d, NBS_d, Nst, Ct[0], P.UseGR, 0);
		}

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&times, start, stop); //time in microseconds

		printf("HCTM3:%d   \ttime: %.15f s\n", hcM3, times * 0.001);	//time in seconds
		fprintf(GSF[0].logfile,"HCTM3:%d   \ttime: %.15f s\n", hcM3, times * 0.001);	//time in seconds
		if(times < timesMin){
			HCTM3 = hcM3;
		}
		timesMin = fmin(times, timesMin);
	}

	if(timesMin <= 0.0f){
		printf("Error in HCM3 tuning, %g\n", timesMin);
		return 0;
	}

	if(P.SERIAL_GROUPING == 1){
		printf("Using Serial Grouping, this disables the tuning parameters\n");
		fprintf(GSF[0].logfile, "Using Serial Grouping, this disables the tuning parameters\n");
		HCTM3 = 32;
	}

	printf("Best parameters: HCTM3:%d\t time: %.15f s\n", HCTM3, timesMin * 0.001);	//time in seconds
	fprintf(GSF[0].logfile, "Best parameters: HCTM3:%d\t time: %.15f s\n", HCTM3, timesMin * 0.001);	//time in seconds

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	fclose(GSF[0].logfile);
	return 1;
}

__host__ int Data::tuneBVH(int &useBVH){

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float times;
	float timesMin = 1000000.0f;

	GSF[0].logfile = fopen(GSF[0].logfilename, "a");
	printf("\nStarting BVH parameters tuning\n");
	fprintf(GSF[0].logfile, "\nStarting BVH parameters tuning\n");

	cudaEventRecord(start, 0);

	BVHCall1();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&times, start, stop); //time in microseconds

	printf("\tBVH1:\t time: %.15f s\n", times * 0.001); //time in seconds
	fprintf(GSF[0].logfile,"\tBVH1:\t time: %.15f s\n", times * 0.001);	//time in seconds
	if(times < timesMin){
		UseBVH = 1;
	}
	timesMin = fmin(times, timesMin);

	cudaEventRecord(start, 0);

	BVHCall2();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&times, start, stop); //time in microseconds

	printf("\tBVH2:\t time: %.15f s\n", times * 0.001); //time in seconds
	fprintf(GSF[0].logfile,"\tBVH2:\t time: %.15f s\n", times * 0.001);	//time in seconds
	if(times < timesMin){
		UseBVH = 2;
	}
	timesMin = fmin(times, timesMin);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("Use BVH method %d\n", UseBVH);
	fprintf(GSF[0].logfile, "Use BVH method %d\n", UseBVH);


	fclose(GSF[0].logfile);

	return 1;
}

__host__ int Data::tuneBS(){

	cudaMemset(BSstop_d, 0, sizeof(int));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float times;
	float timesMin = 1000000.0f;

	int L[9] =  {1, 2, 2, 2, 2,  3, 3, 3, 3};
	int LS[9] = {2, 2, 4, 8, 10, 2, 4, 8, 10};

	int LMin = 1;
	int LSMin = 2;

	GSF[0].logfile = fopen(GSF[0].logfilename, "a");
	printf("\nStarting close-encounter parameters tuning\n");
	fprintf(GSF[0].logfile, "\nStarting close-encounter parameters tuning\n");


#if def_CPU == 0
	int2 ij;
	ij.x = -1;
	ij.y = -1;
	cudaMemcpyToSymbol(CollTshiftpairs_c, &ij, sizeof(int2), 0, cudaMemcpyHostToDevice);
#else
	CollTshiftpairs_c[0].x = -1;
	CollTshiftpairs_c[0].y = -1;
#endif

	int NN = N_h[0] + Nsmall_h[0];
	//save backup values
	save_kernel <<< (NN + 127) / 128, 128 >>> (x4_d, v4_d, x4bb_d, v4bb_d, spin_d, spinbb_d, rcrit_d, rcritv_d, rcritbb_d, rcritvbb_d, index_d, indexbb_d, NN, NconstT, P.SLevels, 1);

	int noColl = 3;

	for(int i = 0; i < 9; ++i){

		P.SLevels = L[i];
		P.SLSteps = LS[i];

		EncpairsZeroC_kernel <<< (NN + 255) / 256, 256 >>> (Encpairs2_d, a_d, Nencpairs_d, Nencpairs2_d, P.NencMax, NN);
		save_kernel <<< (NN + 127) / 128, 128 >>> (x4_d, v4_d, x4bb_d, v4bb_d, spin_d, spinbb_d, rcrit_d, rcritv_d, rcritbb_d, rcritvbb_d, index_d, indexbb_d, NN, NconstT, P.SLevels, -1);


		Rcrit_kernel <<< (NN + 255) / 256, 256 >>> (x4_d, v4_d, x4b_d, v4b_d, spin_d, spinb_d, 1.0 / (3.0 * Msun_h[0].x), rcrit_d, rcritb_d, rcritv_d, rcritvb_d, index_d, indexb_d, dt_h[0], n1_h[0], n2_h[0], time_d, time_h[0], EjectionFlag_d, NN, NconstT, P.SLevels, 0);

		if(P.UseTestParticles == 0){
#if def_CPU == 0
			acc4C_kernel <<< dim3( (((N_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> (x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0], 0, N_h[0], P.NencMax, KP, 0);
#else
			if(Nomp == 1){
				acc4E_cpu();
			}
			else{
				acc4D_cpu();
			}
#endif
		}
		if(P.UseTestParticles == 1){
#if def_CPU == 0
			acc4C_kernel <<< dim3( (((N_h[0] + Nsmall_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> (x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0] + Nsmall_h[0], 0, N_h[0], P.NencMax, KP, 1);
#else
			if(Nomp == 1){
				acc4E_cpu();
				acc4Esmall_cpu();
			}
			else{
				acc4D_cpu();
				acc4Dsmall_cpu();
			}
#endif
		}
		if(P.UseTestParticles == 2){
#if def_CPU == 0
			acc4C_kernel <<< dim3( (((N_h[0] + KP2 - 1)/ KP2) + KTX2 - 1) / KTX2, 1, 1), dim3(KTX2,KTY2,1), KTX2 * KTY2 * KP2 * sizeof(double3) >>> (x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0], N_h[0], N_h[0] + Nsmall_h[0], P.NencMax, KP2, 2);
#else
			if(Nomp == 1){
				acc4E_cpu();
				acc4Esmall_cpu();
			}
			else{
				acc4D_cpu();
				acc4Dsmall_cpu();
			}
#endif
		}

		cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);

		kick32Ab_kernel <<< (NN + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs2_d, 0, NN, P.NencMax, 1);

		HCCall(Ct[0], 1);
		fg_kernel <<< (NN + FTX - 1) / FTX, FTX >>> (x4_d, v4_d, xold_d, vold_d, index_d, dt_h[0] * FGt[0], Msun_h[0].x, NN, aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, 1, P.UseGR);
		cudaDeviceSynchronize();

		printf("    Precheck-pairs:    %d\n", Nencpairs_h[0]);
		fprintf(GSF[0].logfile,"    Precheck-pairs:    %d\n", Nencpairs_h[0]);

		if(Nenc_m[0] > 0){
			for(int i = 0; i < def_GMax; ++i){
				Nenc_m[i] = 0;
			}
			Nencpairs2_h[0] = 0;		
			setNencpairs_kernel <<< 1, 1 >>> (Nencpairs2_d, 1);
		}

		if(EncFlag_m[0] > 0){
			printf("Error: more encounters than allowed: %d %d. Increase 'Maximum encounter pairs'.\n", EncFlag_m[0], P.NencMax);
			fprintf(masterfile, "Error: more encounters than allowed: %d %d. Increase 'Maximum encounter pairs'.\n", EncFlag_m[0], P.NencMax);
			return 0;
		}


		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 63)/ 64, 64 >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, dt_h[0] * FGt[0], Nencpairs_h[0], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, enccount_d, 1, NN, time_h[0], P.StopAtEncounter, Ncoll_d, P.MinMass);

			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);

			cudaEventRecord(start, 0);

			if(P.SLevels > 1){
				if(Nencpairs2_h[0] > 0){
					double time = time_h[0];
					SEnc(time, 0, 1.0, 0, noColl);
				}
			}
			else{
				if(Nencpairs2_h[0] > 0){
#if def_CPU == 0
					groupCall();
#else
					group_cpu (Nenc_m, Nencpairs2_h, Encpairs2_h, Encpairs_h, P.NencMax, N_h[0] + Nsmall_h[0], N_h[0], P.SERIAL_GROUPING);
#endif
				}
				cudaDeviceSynchronize();

				BSCall(0, time_h[0], noColl, 1.0);
			}

			cudaEventRecord(stop, 0);
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);

			printf("    CE:    %d; ", Nencpairs2_h[0]);
			printf("groups: %d; ", Nenc_m[0]);
			fprintf(GSF[0].logfile, "    CE:    %d; ", Nencpairs2_h[0]);
			fprintf(GSF[0].logfile, "groups: %d; ", Nenc_m[0]);
			int nn = 2;
			for(int st = 1; st < def_GMax; ++st){
				if(Nenc_m[st] > 0){
					printf("%d: %d; ", nn, Nenc_m[st]);
					fprintf(GSF[0].logfile, "%d: %d; ", nn, Nenc_m[st]);
				}
				nn *= 2;
			}
			printf("\n");
			fprintf(GSF[0].logfile, "\n");


			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&times, start, stop); //time in microseconds
			printf("Close-encounter time: Symplectic levels: %d, Symplectic sub steps: %d,  %.15f s\n", P.SLevels, P.SLSteps, times * 0.001);	//time in seconds
			fprintf(GSF[0].logfile, "Close-encounter time: Symplectic levels: %d, Symplectic sub steps: %d,  %.15f s\n", P.SLevels, P.SLSteps, times * 0.001);	//time in seconds
			if(times < timesMin){
				LMin = L[i];
				LSMin = LS[i];
			}
			timesMin = fmin(times, timesMin);

		}
	}
	P.SLevels = LMin;
	P.SLSteps = LSMin;

	printf("\nBest parameters: Symplectic levels: %d, Symplectic sub steps: %d,  %.15f s\n\n", P.SLevels, P.SLSteps, timesMin * 0.001);
	fprintf(GSF[0].logfile, "\nBest parameters: Symplectic levels: %d, Symplectic sub steps: %d,  %.15f s\n\n", P.SLevels, P.SLSteps, timesMin * 0.001);

	//restore old coordinate values
	EncpairsZeroC_kernel <<< (NN + 255) / 256, 256 >>> (Encpairs2_d, a_d, Nencpairs_d, Nencpairs2_d, P.NencMax, NN);
	save_kernel <<< (NN + 127) / 128, 128 >>> (x4_d, v4_d, x4bb_d, v4bb_d, spin_d, spinbb_d, rcrit_d, rcritv_d, rcritbb_d, rcritvbb_d, index_d, indexbb_d, NN, NconstT, P.SLevels, -1);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	fclose(GSF[0].logfile);
	return 1;

}

//called during the integration, before BSCall
__host__ int Data::tuneBS2(){

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float times;
	float timesMin = 1000000.0f;

	//int L[9] =  {1, 2, 2, 2, 2,  3, 3, 3, 3};
	//int LS[9] = {2, 2, 4, 8, 10, 2, 4, 8, 10};
	int L[9] =  {2, 1, 2, 2, 2,  3, 3, 3, 3};
	int LS[9] = {2, 2, 4, 8, 10, 2, 4, 8, 10};

	int LMin = 1;
	int LSMin = 2;

	GSF[0].logfile = fopen(GSF[0].logfilename, "a");
	printf("\nStarting close-encounter parameters tuning\n");
	fprintf(GSF[0].logfile, "\nStarting close-encounter parameters tuning\n");


#if def_CPU == 0
	int2 ij;
	ij.x = -1;
	ij.y = -1;
	cudaMemcpyToSymbol(CollTshiftpairs_c, &ij, sizeof(int2), 0, cudaMemcpyHostToDevice);
#else
	CollTshiftpairs_c[0].x = -1;
	CollTshiftpairs_c[0].y = -1;
#endif

	int NN = N_h[0] + Nsmall_h[0];
	//save backup values
	save_kernel <<< (NN + 127) / 128, 128 >>> (x4_d, v4_d, x4bb_d, v4bb_d, spin_d, spinbb_d, rcrit_d, rcritv_d, rcritbb_d, rcritvbb_d, index_d, indexbb_d, NN, NconstT, P.SLevels, 1);
	save_kernel <<< (NN + 127) / 128, 128 >>> (xold_d, vold_d, x4b_d, v4b_d, spin_d, spinb_d, rcrit_d, rcritv_d, rcritb_d, rcritvb_d, index_d, indexb_d, NN, NconstT, P.SLevels, 1);

	int noColl = 3;

	int Nencpairs = Nencpairs_h[0];

	//for(int i = 0; i < 9; ++i){
	for(int i = 0; i < 2; ++i){


		//if(i > 0) noColl = 2;
		Nencpairs_h[0] = Nencpairs;
		P.SLevels = L[i];
		P.SLSteps = LS[i];

		save_kernel <<< (NN + 127) / 128, 128 >>> (x4_d, v4_d, x4bb_d, v4bb_d, spin_d, spinbb_d, rcrit_d, rcritv_d, rcritbb_d, rcritvbb_d, index_d, indexbb_d, NN, NconstT, P.SLevels, -1);
		save_kernel <<< (NN + 127) / 128, 128 >>> (xold_d, vold_d, x4b_d, v4b_d, spin_d, spinb_d, rcrit_d, rcritv_d, rcritb_d, rcritvb_d, index_d, indexb_d, NN, NconstT, P.SLevels, -1);


		printf("    Precheck-pairs:    %d\n", Nencpairs_h[0]);
		fprintf(GSF[0].logfile,"    Precheck-pairs:    %d\n", Nencpairs_h[0]);


		cudaEventRecord(start, 0);

		if(P.SLevels > 1){
			if(Nencpairs2_h[0] > 0){
				double time = time_h[0];
				SEnc(time, 0, 1.0, 0, noColl);
			}
		}
		else{
			if(Nencpairs2_h[0] > 0){
				groupCall();
			}
			cudaDeviceSynchronize();

			BSCall(0, time_h[0], noColl, 1.0);
		}

		cudaEventRecord(stop, 0);
		cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);

		printf("    CE:    %d; ", Nencpairs2_h[0]);
		printf("groups: %d; ", Nenc_m[0]);
		fprintf(GSF[0].logfile, "    CE:    %d; ", Nencpairs2_h[0]);
		fprintf(GSF[0].logfile, "groups: %d; ", Nenc_m[0]);
		int nn = 2;
		for(int st = 1; st < def_GMax; ++st){
			if(Nenc_m[st] > 0){
				printf("%d: %d; ", nn, Nenc_m[st]);
				fprintf(GSF[0].logfile, "%d: %d; ", nn, Nenc_m[st]);
			}
			nn *= 2;
		}
		printf("\n");
		fprintf(GSF[0].logfile, "\n");


		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&times, start, stop); //time in microseconds
		printf("Close-encounter time: Symplectic levels: %d, Symplectic sub steps: %d,  %.15f s\n", P.SLevels, P.SLSteps, times * 0.001);	//time in seconds
		fprintf(GSF[0].logfile, "Close-encounter time: Symplectic levels: %d, Symplectic sub steps: %d,  %.15f s\n", P.SLevels, P.SLSteps, times * 0.001);	//time in seconds
		if(times < timesMin){
			LMin = L[i];
			LSMin = LS[i];
		}
		timesMin = fmin(times, timesMin);

	}
	P.SLevels = LMin;
	P.SLSteps = LSMin;

	printf("\nBest parameters: Symplectic levels: %d, Symplectic sub steps: %d,  %.15f s\n\n", P.SLevels, P.SLSteps, timesMin * 0.001);
	fprintf(GSF[0].logfile, "\nBest parameters: Symplectic levels: %d, Symplectic sub steps: %d,  %.15f s\n\n", P.SLevels, P.SLSteps, timesMin * 0.001);

	//restore old coordinate values
	Nencpairs_h[0] = Nencpairs;
	save_kernel <<< (NN + 127) / 128, 128 >>> (x4_d, v4_d, x4bb_d, v4bb_d, spin_d, spinbb_d, rcrit_d, rcritv_d, rcritbb_d, rcritvbb_d, index_d, indexbb_d, NN, NconstT, P.SLevels, -1);
	save_kernel <<< (NN + 127) / 128, 128 >>> (xold_d, vold_d, x4b_d, v4b_d, spin_d, spinb_d, rcrit_d, rcritv_d, rcritb_d, rcritvb_d, index_d, indexb_d, NN, NconstT, P.SLevels, -1);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	fclose(GSF[0].logfile);
	return 1;

}


#if def_CPU == 1
void Data::firstKick_cpu(int noColl){
	//use last time information, the beginning of the time step
	double time = (P.tRestart + 1) * idt_h[0] + ict_h[0] * 365.25; //in the set Elements kernel, timestep wil be decreased by 1 again
	for(int i = 0; i < NconstT; ++i){
		a_h[i] = {0.0, 0.0, 0.0};
		ab_h[i] = {0.0, 0.0, 0.0};
	}
	BSstop_h[0] = 0;
	initialb_cpu (Encpairs_h, Encpairs2_h, NBNencT);

	Rcrit_cpu (x4_h, v4_h, x4b_h, v4b_h, spin_h, spinb_h, 1.0 / (3.0 * Msun_h[0].x), rcrit_h, rcritb_h, rcritv_h, rcritvb_h, index_h, indexb_h, dt_h[0], n1_h[0], n2_h[0], time_h, time, EjectionFlag_m, N_h[0], NconstT, P.SLevels, noColl);
	if(P.setElementsV == 2){ // convert barycentric velocities to heliocentric
		comCall(1);
	}
	if(P.setElementsV > 0){
		setElements_cpu (x4_h, v4_h, index_h, setElementsData_h, setElementsLine_h, Msun_h, dt_h, time_h, N_h[0], Nst, 0);
	}
	if(P.setElementsV == 2){
		comCall(-1);
	}
	if(P.KickFloat == 0){
		if(Nomp == 1){
			acc4E_cpu();
		}
		else{
			acc4D_cpu();
		}
	}
	else{
		if(Nomp == 1){
			acc4Ef_cpu();
		}
		else{
			acc4Df_cpu();
		}
	}
}
void Data::firstKick_small_cpu(int noColl){
	//use last time information, the beginning of the time step
	double time = (P.tRestart + 1) * idt_h[0] + ict_h[0] * 365.25; //in the set Elements kernel, timestep wil be decreased by 1 again
	for(int i = 0; i < NconstT; ++i){
		a_h[i] = {0.0, 0.0, 0.0};
		ab_h[i] = {0.0, 0.0, 0.0};
	}
	BSstop_h[0] = 0;
	initialb_cpu (Encpairs_h, Encpairs2_h, NBNencT);

	Rcrit_cpu (x4_h, v4_h, x4b_h, v4b_h, spin_h, spinb_h, 1.0 / (3.0 * Msun_h[0].x), rcrit_h, rcritb_h, rcritv_h, rcritvb_h, index_h, indexb_h, dt_h[0], n1_h[0], n2_h[0], time_h, time, EjectionFlag_m, N_h[0] + Nsmall_h[0], NconstT, P.SLevels, noColl);
	if(P.setElementsV == 2){ // convert barycentric velocities to heliocentric
		comCall(1);
	}
	if(P.setElementsV > 0){
		setElements_cpu (x4_h, v4_h, index_h, setElementsData_h, setElementsLine_h, Msun_h, dt_h, time_h, N_h[0], Nst, 0);
	}
	if(P.setElementsV == 2){
		comCall(-1);
	}

	if(P.KickFloat == 0){
		if(Nomp == 1){
			acc4E_cpu();
			acc4Esmall_cpu();
		}
		else{
			acc4D_cpu();
			acc4Dsmall_cpu();
		}
	}
	else{
		if(Nomp == 1){
			acc4Ef_cpu();
			acc4Efsmall_cpu();
		}
		else{
			acc4Df_cpu();
			acc4Dfsmall_cpu();
		}
	}

	if(P.SERIAL_GROUPING == 1){
		Sortb_cpu(Encpairs2_h, 0, N_h[0] + Nsmall_h[0], P.NencMax);
	}
}
#endif
__host__ void Data::firstKick_16(int noColl){
	//use last time information, the beginning of the time step
	double time = (P.tRestart + 1) * idt_h[0] + ict_h[0] * 365.25; //in the set Elements kernel, timestep wil be decreased by 1 again
	cudaMemset(a_d, 0, NconstT * sizeof(double3));
	cudaMemset(ab_d, 0, NconstT * sizeof(double3));
	cudaMemset(BSstop_d, 0, sizeof(int));
	initialb_kernel <<< (NBNencT + 255) / 256, 256 >>> (Encpairs_d, Encpairs2_d, NBNencT);
	Rcrit_kernel <<< (N_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, x4b_d, v4b_d, spin_d, spinb_d, 1.0 / (3.0 * Msun_h[0].x), rcrit_d, rcritb_d, rcritv_d, rcritvb_d, index_d, indexb_d, dt_h[0], n1_h[0], n2_h[0], time_d, time, EjectionFlag_d, N_h[0], NconstT, P.SLevels, noColl);
	if(P.setElementsV == 2){ // convert barycentric velocities to heliocentric
		comCall(1);
	}
	if(P.setElementsV > 0){
		setElements_kernel <<< (P.setElementsN + 63) / 64, 64 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst, 0);
	}
	if(P.setElementsV == 2){
		comCall(-1);
	}
	if(P.KickFloat == 0){
		kick16c_kernel <<< N_h[0], WarpSize >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, EncFlag_d, P.NencMax, N_h[0], 0);
	}
	else{
		kick16cf_kernel <<< N_h[0], WarpSize >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, EncFlag_d, P.NencMax, N_h[0], 0);
	}
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
}

__host__ void Data::firstKick_largeN(int noColl){
	//use last time information, the beginning of the time step
	double time = (P.tRestart + 1) * idt_h[0] + ict_h[0] * 365.25; //in the set Elements kernel, timestep wil be decreased by 1 again
	cudaMemset(a_d, 0, NconstT * sizeof(double3));
	cudaMemset(ab_d, 0, NconstT * sizeof(double3));
	cudaMemset(BSstop_d, 0, sizeof(int));
	initialb_kernel <<< (NBNencT + 255) / 256, 256 >>> (Encpairs_d, Encpairs2_d, NBNencT);
	Rcrit_kernel <<< (N_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, x4b_d, v4b_d, spin_d, spinb_d, 1.0 / (3.0 * Msun_h[0].x), rcrit_d, rcritb_d, rcritv_d, rcritvb_d, index_d, indexb_d, dt_h[0], n1_h[0], n2_h[0], time_d, time, EjectionFlag_d, N_h[0], NconstT, P.SLevels, noColl);
	if(P.setElementsV == 2){ // convert barycentric velocities to heliocentric
		comCall(1);
	}
	if(P.setElementsV > 0){
		setElements_kernel <<< (P.setElementsN + 63) / 64, 64 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst, 0);
	}
	if(P.setElementsV == 2){
		comCall(-1);
	}
	if(UseAcc == 0){
		if(P.KickFloat == 0){
			kick32c_kernel <<< N_h[0] , min(NB[0], 1024), 2 * WarpSize * sizeof(double3) >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, EncFlag_d, P.NencMax, N_h[0], 0);
		}
		else{
			kick32cf_kernel <<< N_h[0] , min(NB[0], 1024), 2 * WarpSize * sizeof(float3) >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, EncFlag_d, P.NencMax, N_h[0], 0);
		}
	}
	else{
		if(P.ndev == 1){

			if(P.KickFloat == 0){
				acc4C_kernel <<< dim3( (((N_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0], 0, N_h[0], P.NencMax, KP, 0);
			}
			else{
				acc4Cf_kernel <<< dim3( (((N_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0], 0, N_h[0], P.NencMax, KP, 0);
			}
		}
		else{
			KickfirstndevCall(0);
		}
	}
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
}

__host__ void Data::firstKick_small(int noColl){
	//use last time information, the beginning of the time step
	double time = (P.tRestart + 1) * idt_h[0] + ict_h[0] * 365.25; //in the set Elements kernel, timestep wil be decreased by 1 again
	cudaMemset(a_d, 0, NconstT*sizeof(double3));
	cudaMemset(ab_d, 0, NconstT * sizeof(double3));
	cudaMemset(BSstop_d, 0, sizeof(int));
	initialb_kernel <<< (NBNencT + 255) / 256, 256 >>> (Encpairs_d, Encpairs2_d, NBNencT);
	Rcrit_kernel <<< (N_h[0] + Nsmall_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, x4b_d, v4b_d, spin_d, spinb_d, 1.0 / (3.0 * Msun_h[0].x), rcrit_d, rcritb_d, rcritv_d, rcritvb_d, index_d, indexb_d, dt_h[0], n1_h[0], n2_h[0], time_d, time, EjectionFlag_d, N_h[0] + Nsmall_h[0], NconstT, P.SLevels, noColl);
	if(P.setElementsV == 2){ // convert barycentric velocities to heliocentric
		comCall(1);
	}	
	if(P.setElementsV > 0){
		setElements_kernel <<< (P.setElementsN + 63) / 64, 64 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst, 0);
	}
	if(P.setElementsV == 2){
		comCall(-1);
	}

	if(P.KickFloat == 0){
		acc4C_kernel <<< dim3( (((N_h[0] + Nsmall_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0] + Nsmall_h[0], 0, N_h[0], P.NencMax, KP, 1);
	}
	else{
		acc4Cf_kernel <<< dim3( (((N_h[0] + Nsmall_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0] + Nsmall_h[0], 0, N_h[0], P.NencMax, KP, 1);
	}
	if(P.UseTestParticles == 2){
		if(P.KickFloat == 0){
			acc4C_kernel <<< dim3( (((N_h[0] + KP2 - 1)/ KP2) + KTX2 - 1) / KTX2, 1, 1), dim3(KTX2,KTY2,1), KTX2 * KTY2 * KP2 * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0], N_h[0], N_h[0] + Nsmall_h[0], P.NencMax, KP2, 2);
		}
		else{
			acc4Cf_kernel <<< dim3( (((N_h[0] + KP2 - 1)/ KP2) + KTX2 - 1) / KTX2, 1, 1), dim3(KTX2,KTY2,1), KTX2 * KTY2 * KP2 * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0], N_h[0], N_h[0] + Nsmall_h[0], P.NencMax, KP2, 2);
		}
	}
		
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);

	if(P.SERIAL_GROUPING == 1){
		Sortb_kernel<<< (N_h[0] + Nsmall_h[0] + 127) / 128, 128 >>>(Encpairs2_d, 0, N_h[0] + Nsmall_h[0], P.NencMax);
	}
}

__host__ void Data::firstKick_M(long long ts, int noColl){
	cudaMemset(a_d, 0, NconstT*sizeof(double3));
	cudaMemset(ab_d, 0, NconstT * sizeof(double3));
	cudaMemset(BSstop_d, 0, sizeof(int));
	initialb_kernel <<< (NBNencT + 255) / 256, 256 >>> (Encpairs_d, Encpairs2_d, NBNencT);
	RcritM_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, x4b_d, v4b_d, spin_d, spinb_d, Msun_d, rcrit_d, rcritb_d, rcritv_d, rcritvb_d, dt_d, n1_d, n2_d, Rcut_d, RcutSun_d, EjectionFlag_d, index_d, indexb_d, Nst, NT, time_d, idt_d, ict_d, delta_d, ts, StopFlag_d, NconstT, P.SLevels, noColl, Nstart);
	KickM2_kernel < KM_Bl, KM_Bl2, NmaxM, 0 > <<< (NT + KM_Bl2 - 1) / KM_Bl2, KM_Bl >>> (x4_d, v4_d, a_d, rcritv_d, Nencpairs_d, Encpairs_d, dt_d, Kt[SIn - 1] * def_ksq, index_d, NT, Nstart);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
}

__host__ void Data::firstKick_M3(long long ts, int noColl){
	cudaMemset(a_d, 0, NconstT*sizeof(double3));
	cudaMemset(ab_d, 0, NconstT * sizeof(double3));
	cudaMemset(BSstop_d, 0, sizeof(int));
	initialb_kernel <<< (NBNencT + 255) / 256, 256 >>> (Encpairs_d, Encpairs2_d, NBNencT);
	RcritM_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, x4b_d, v4b_d, spin_d, spinb_d, Msun_d, rcrit_d, rcritb_d, rcritv_d, rcritvb_d, dt_d, n1_d, n2_d, Rcut_d, RcutSun_d, EjectionFlag_d, index_d, indexb_d, Nst, NT, time_d, idt_d, ict_d, delta_d, ts, StopFlag_d, NconstT, P.SLevels, noColl, Nstart);
	KickM3_kernel <<< NT, dim3(KTM3, 1, 1), WarpSize * sizeof(double3) >>> (x4_d, v4_d, a_d, rcritv_d, Nencpairs_d, Encpairs_d, dt_d, Kt[SIn - 1], index_d, NT, N_d, NBS_d, 0, 1);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
}

__host__ void Data::BSCall(int si, double time, int noColl, double ll){
	
	time -= dt_h[0] / dayUnit;
	int N = N_h[0] + Nsmall_h[0];
	double dt = dt_h[0] / ll * FGt[si];
	
//printf(" %d | %d %d %d | %d %d %d | %d %d %d | %d %d %d\n", Nenc_m[0], Nenc_m[1], Nenc_m[2], Nenc_m[3], Nenc_m[4], Nenc_m[5], Nenc_m[6], Nenc_m[7], Nenc_m[8], Nenc_m[9],  Nenc_m[10],  Nenc_m[11],  Nenc_m[12]);
#if def_CPU == 0	
	//if(Nenc_m[1] > 0) BSB2Step_kernel <2 * 16, 2> <<< (Nenc_m[1] + 15)/ 16, 2 * 16, 0, BSStream[0] >>> (random_d, x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 0, index_d, Nenc_m[1], BSstop_d, Ncoll_d, Coll_d, time, spin_d, love_d, createFlag_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, N, NconstT, NWriteEnc_d, writeEnc_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	//if(Nenc_m[1] > 0) BSB2Step_kernel <2, 2> <<< Nenc_m[1], 2, 0, BSStream[0] >>> (random_d, x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 0, index_d, Nenc_m[1], BSstop_d, Ncoll_d, Coll_d, time, spin_d, love_d, createFlag_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, N, NconstT, NWriteEnc_d, writeEnc_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);

	if(Nenc_m[1] > 0) BSBStep_kernel <2, 2> <<< Nenc_m[1], 4, 0, BSStream[0] >>> (random_d, x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 0, index_d, BSstop_d, Ncoll_d, Coll_d, time, spin_d, love_d, createFlag_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, N, NconstT, NWriteEnc_d, writeEnc_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	

	if(Nenc_m[2] > 0) BSBStep_kernel <4, 4> <<< Nenc_m[2], 16, 0, BSStream[1] >>> (random_d, x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 1, index_d, BSstop_d, Ncoll_d, Coll_d, time, spin_d, love_d, createFlag_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, N, NconstT, NWriteEnc_d, writeEnc_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[3] > 0) BSBStep_kernel <8, 8> <<< Nenc_m[3], 64, 0, BSStream[2] >>> (random_d, x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 2, index_d, BSstop_d, Ncoll_d, Coll_d, time, spin_d, love_d, createFlag_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, N, NconstT, NWriteEnc_d, writeEnc_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[4] > 0) BSBStep_kernel <16, 16> <<< Nenc_m[4], 256, 0, BSStream[3] >>> (random_d, x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 3, index_d, BSstop_d, Ncoll_d, Coll_d, time, spin_d, love_d, createFlag_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, N, NconstT, NWriteEnc_d, writeEnc_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[5] > 0) BSBStep_kernel <32, 8> <<< Nenc_m[5], 256, 0, BSStream[4] >>> (random_d, x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 4, index_d, BSstop_d, Ncoll_d, Coll_d, time, spin_d, love_d, createFlag_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, N, NconstT, NWriteEnc_d, writeEnc_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);

	/*
	if(Nenc_m[1] > 0) BSA_kernel < 2 > <<< Nenc_m[1], 2 , 0, BSStream[0] >>> (random_d, x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, love_d, createFlag_d, Encpairs_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 0, N, NconstT, P.NencMax, BSstop_d, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[2] > 0) BSA_kernel < 4 > <<< Nenc_m[2], 4 , 0, BSStream[1] >>> (random_d, x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, love_d, createFlag_d, Encpairs_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 1, N, NconstT, P.NencMax, BSstop_d, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[3] > 0) BSA_kernel < 8 > <<< Nenc_m[3], 8 , 0, BSStream[2] >>> (random_d, x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, love_d, createFlag_d, Encpairs_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 2, N, NconstT, P.NencMax, BSstop_d, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[4] > 0) BSA_kernel < 16 > <<< Nenc_m[4], 16 , 0, BSStream[3] >>> (random_d, x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, love_d, createFlag_d, Encpairs_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 3, N, NconstT, P.NencMax, BSstop_d, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[5] > 0) BSA_kernel < 32 > <<< Nenc_m[5], 32 , 0, BSStream[4] >>> (random_d, x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, love_d, createFlag_d, Encpairs_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 4, N, NconstT, P.NencMax, BSstop_d, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	*/	

	if(Nenc_m[6] > 0) BSA_kernel < 64 > <<< Nenc_m[6], 64 , 0, BSStream[5] >>> (random_d, x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, love_d, createFlag_d, Encpairs_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 5, N, NconstT, P.NencMax, BSstop_d, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[7] > 0) BSA_kernel < 128 > <<< Nenc_m[7], 128 , 0, BSStream[6] >>> (random_d, x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, love_d, createFlag_d, Encpairs_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 6, N, NconstT, P.NencMax, BSstop_d, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[8] > 0) BSA_kernel < 256 > <<< Nenc_m[8], 256 , 0, BSStream[7] >>> (random_d, x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, love_d, createFlag_d, Encpairs_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 7, N, NconstT, P.NencMax, BSstop_d, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
 

/*		
	if(Nenc_m[1] > 0) BSA512_kernel < 2, 2 > <<< Nenc_m[1], 2 , 0, BSStream[0] >>> (random_d, x4_d, v4_d, xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, rcrit_d, rcritv_d, index_d, spin_d, love_d, createFlag_d, Encpairs_d, Encpairs2_d, dx_d, dv_d, dt, Msun_h[0].x, U_d, 0, N, NconstT, P.NencMax, BSstop_d, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, dtgr_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[2] > 0) BSA512_kernel < 4, 4 > <<< Nenc_m[2], 4 , 0, BSStream[0] >>> (random_d, x4_d, v4_d, xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, rcrit_d, rcritv_d, index_d, spin_d, love_d, createFlag_d, Encpairs_d, Encpairs2_d, dx_d, dv_d, dt, Msun_h[0].x, U_d, 1, N, NconstT, P.NencMax, BSstop_d, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, dtgr_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[3] > 0) BSA512_kernel < 8, 8 > <<< Nenc_m[3], 8 , 0, BSStream[0] >>> (random_d, x4_d, v4_d, xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, rcrit_d, rcritv_d, index_d, spin_d, love_d, createFlag_d, Encpairs_d, Encpairs2_d, dx_d, dv_d, dt, Msun_h[0].x, U_d, 2, N, NconstT, P.NencMax, BSstop_d, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, dtgr_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[4] > 0) BSA512_kernel < 16, 16 > <<< Nenc_m[4], 16 , 0, BSStream[0] >>> (random_d, x4_d, v4_d, xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, rcrit_d, rcritv_d, index_d, spin_d, love_d, createFlag_d, Encpairs_d, Encpairs2_d, dx_d, dv_d, dt, Msun_h[0].x, U_d, 3, N, NconstT, P.NencMax, BSstop_d, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, dtgr_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[5] > 0) BSA512_kernel < 32, 32 > <<< Nenc_m[5], 32 , 0, BSStream[0] >>> (random_d, x4_d, v4_d, xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, rcrit_d, rcritv_d, index_d, spin_d, love_d, createFlag_d, Encpairs_d, Encpairs2_d, dx_d, dv_d, dt, Msun_h[0].x, U_d, 4, N, NconstT, P.NencMax, BSstop_d, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, dtgr_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[6] > 0) BSA512_kernel < 64, 64 > <<< Nenc_m[6], 64 , 0, BSStream[0] >>> (random_d, x4_d, v4_d, xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, rcrit_d, rcritv_d, index_d, spin_d, love_d, createFlag_d, Encpairs_d, Encpairs2_d, dx_d, dv_d, dt, Msun_h[0].x, U_d, 5, N, NconstT, P.NencMax, BSstop_d, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, dtgr_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[7] > 0) BSA512_kernel < 128, 128 > <<< Nenc_m[7], 128 , 0, BSStream[0] >>> (random_d, x4_d, v4_d, xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, rcrit_d, rcritv_d, index_d, spin_d, love_d, createFlag_d, Encpairs_d, Encpairs2_d, dx_d, dv_d, dt, Msun_h[0].x, U_d, 6, N, NconstT, P.NencMax, BSstop_d, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, dtgr_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[8] > 0) BSA512_kernel < 256, 256 > <<< Nenc_m[8], 256 , 0, BSStream[0] >>> (random_d, x4_d, v4_d, xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, rcrit_d, rcritv_d, index_d, spin_d, love_d, createFlag_d, Encpairs_d, Encpairs2_d, dx_d, dv_d, dt, Msun_h[0].x, U_d, 7, N, NconstT, P.NencMax, BSstop_d, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, dtgr_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
*/	
	
	//if(Nenc_m[9] > 0) BSA512_kernel < 512, 512 > <<< Nenc_m[9], 512 , 0, BSStream[0] >>> (random_d, x4_d, v4_d, xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, rcrit_d, rcritv_d, index_d, spin_d, love_d, createFlag_d, Encpairs_d, Encpairs2_d, dx_d, dv_d, dt, Msun_h[0].x, U_d, 8, N, NconstT, P.NencMax, BSstop_d, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, dtgr_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	

	/*
	if(Nenc_m[1] > 0) BSACall(0, 2, Nenc_m[1], si, time, FGt[si] / ll, noColl);
	if(Nenc_m[2] > 0) BSACall(1, 4, Nenc_m[2], si, time, FGt[si] / ll, noColl);
	if(Nenc_m[3] > 0) BSACall(2, 8, Nenc_m[3], si, time, FGt[si] / ll, noColl);
	if(Nenc_m[4] > 0) BSACall(3, 16, Nenc_m[4], si, time, FGt[si] / ll, noColl);
	if(Nenc_m[5] > 0) BSACall(4, 32, Nenc_m[5], si, time, FGt[si] / ll, noColl);
	if(Nenc_m[6] > 0) BSACall(5, 64, Nenc_m[6], si, time, FGt[si] / ll, noColl);
	if(Nenc_m[7] > 0) BSACall(6, 128, Nenc_m[7], si, time, FGt[si] / ll, noColl);
	if(Nenc_m[8] > 0) BSACall(7, 256, Nenc_m[8], si, time, FGt[si] / ll, noColl);
	*/	
	if(Nenc_m[9] > 0) BSACall(8, 512, Nenc_m[9], si, time, FGt[si] / ll, noColl);

	if(Nenc_m[10] > 0) BSACall(9, 1024, Nenc_m[10], si, time, FGt[si] / ll, noColl);
	if(Nenc_m[11] > 0) BSACall(10, 2048, Nenc_m[11], si, time, FGt[si] / ll, noColl);
	
	int nn = 4096;
	for(int st = 11; st < def_GMax - 1; ++st){
		if(Nenc_m[st + 1] > 0) BSACall(st, nn, Nenc_m[st + 1], si, time, FGt[si] / ll, noColl);
		nn *= 2;
	}

	//make sure that Nenc_m is updated and available on host before continue
	cudaDeviceSynchronize();
	//for(int i = 0; i < def_GMax; ++i){
	//	Nenc_m[i] = 0;
	//}
#else

//omp task
//omp taskloop
	#pragma omp parallel
	{
	#pragma omp for nowait
	for(int idx = 0; idx < Nenc_m[1]; ++idx){ 
		BSBStep_cpu <2> (random_h, x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, Encpairs2_h, dt, Msun_h[0].x, U_h, 0, index_h, BSstop_h, Ncoll_m, Coll_h, time, spin_h, love_h, createFlag_h, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, N, NconstT, NWriteEnc_m, writeEnc_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl, idx);
	}
	#pragma omp for nowait
	for(int idx = 0; idx < Nenc_m[2]; ++idx){ 
		BSBStep_cpu <4> (random_h, x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, Encpairs2_h, dt, Msun_h[0].x, U_h, 1, index_h, BSstop_h, Ncoll_m, Coll_h, time, spin_h, love_h, createFlag_h, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, N, NconstT, NWriteEnc_m, writeEnc_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl, idx);
	}
	#pragma omp for nowait
	for(int idx = 0; idx < Nenc_m[3]; ++idx){ 
		BSBStep_cpu <8> (random_h, x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, Encpairs2_h, dt, Msun_h[0].x, U_h, 2, index_h, BSstop_h, Ncoll_m, Coll_h, time, spin_h, love_h, createFlag_h, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, N, NconstT, NWriteEnc_m, writeEnc_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl, idx);
	}
	#pragma omp for nowait
	for(int idx = 0; idx < Nenc_m[4]; ++idx){ 
		BSBStep_cpu <16> (random_h, x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, Encpairs2_h, dt, Msun_h[0].x, U_h, 3, index_h, BSstop_h, Ncoll_m, Coll_h, time, spin_h, love_h, createFlag_h, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, N, NconstT, NWriteEnc_m, writeEnc_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl, idx);
	}
	#pragma omp for nowait
	for(int idx = 0; idx < Nenc_m[5]; ++idx){ 
		BSBStep_cpu <32> (random_h, x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, Encpairs2_h, dt, Msun_h[0].x, U_h, 4, index_h, BSstop_h, Ncoll_m, Coll_h, time, spin_h, love_h, createFlag_h, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, N, NconstT, NWriteEnc_m, writeEnc_h, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl, idx);
	}

	/*
	#pragma omp for nowait
	for(int idx = 0; idx < Nenc_m[1]; ++idx){ 
		BSA_cpu < 2 > (random_d, x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, love_d, createFlag_d, Encpairs_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 0, N, NconstT, P.NencMax, BSstop_d, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl, idx);
	}
	#pragma omp for nowait
	for(int idx = 0; idx < Nenc_m[2]; ++idx){ 
		BSA_cpu < 4 > (random_d, x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, love_d, createFlag_d, Encpairs_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 1, N, NconstT, P.NencMax, BSstop_d, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl, idx);
	}
	#pragma omp for nowait
	for(int idx = 0; idx < Nenc_m[3]; ++idx){ 
		BSA_cpu < 8 > (random_d, x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, love_d, createFlag_d, Encpairs_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 2, N, NconstT, P.NencMax, BSstop_d, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl, idx);
	}
	#pragma omp for nowait
	for(int idx = 0; idx < Nenc_m[4]; ++idx){ 
		BSA_cpu < 16 > (random_d, x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, love_d, createFlag_d, Encpairs_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 3, N, NconstT, P.NencMax, BSstop_d, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl, idx);
	}
	#pragma omp for nowait
	for(int idx = 0; idx < Nenc_m[5]; ++idx){ 
		BSA_cpu < 32 > (random_d, x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, love_d, createFlag_d, Encpairs_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 4, N, NconstT, P.NencMax, BSstop_d, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl, idx);
	}
	*/


	#pragma omp for nowait
	for(int idx = 0; idx < Nenc_m[6]; ++idx){ 
		BSA_cpu < 64 > (random_d, x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, love_d, createFlag_d, Encpairs_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 5, N, NconstT, P.NencMax, BSstop_d, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl, idx);
	}
	#pragma omp for nowait
	for(int idx = 0; idx < Nenc_m[7]; ++idx){ 
		BSA_cpu < 128 > (random_d, x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, love_d, createFlag_d, Encpairs_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 6, N, NconstT, P.NencMax, BSstop_d, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl, idx);
	}
	#pragma omp for nowait
	for(int idx = 0; idx < Nenc_m[8]; ++idx){ 
		BSA_cpu < 256 > (random_d, x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, love_d, createFlag_d, Encpairs_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 7, N, NconstT, P.NencMax, BSstop_d, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl, idx);
	}

	}

	/*
	if(Nenc_m[1] > 0) BSACall(0, 2, Nenc_m[1], si, time, FGt[si] / ll, noColl);
	if(Nenc_m[2] > 0) BSACall(1, 4, Nenc_m[2], si, time, FGt[si] / ll, noColl);
	if(Nenc_m[3] > 0) BSACall(2, 8, Nenc_m[3], si, time, FGt[si] / ll, noColl);
	if(Nenc_m[4] > 0) BSACall(3, 16, Nenc_m[4], si, time, FGt[si] / ll, noColl);
	if(Nenc_m[5] > 0) BSACall(4, 32, Nenc_m[5], si, time, FGt[si] / ll, noColl);
	if(Nenc_m[6] > 0) BSACall(5, 64, Nenc_m[6], si, time, FGt[si] / ll, noColl);
	if(Nenc_m[7] > 0) BSACall(6, 128, Nenc_m[7], si, time, FGt[si] / ll, noColl);
	if(Nenc_m[8] > 0) BSACall(7, 256, Nenc_m[8], si, time, FGt[si] / ll, noColl);
	*/
	if(Nenc_m[9] > 0) BSACall(8, 512, Nenc_m[9], si, time, FGt[si] / ll, noColl);

	if(Nenc_m[10] > 0) BSACall(9, 1024, Nenc_m[10], si, time, FGt[si] / ll, noColl);
	if(Nenc_m[11] > 0) BSACall(10, 2048, Nenc_m[11], si, time, FGt[si] / ll, noColl);
	
	int nn = 4096;
	for(int st = 11; st < def_GMax - 1; ++st){
		if(Nenc_m[st + 1] > 0) BSACall(st, nn, Nenc_m[st + 1], si, time, FGt[si] / ll, noColl);
		nn *= 2;
	}

#endif
}

__host__ void Data::BSBMCall(int si, int noColl, double ll){

//printf(" %d | %d %d %d | %d %d %d | %d %d %d | %d %d %d\n", Nenc_m[0], Nenc_m[1], Nenc_m[2], Nenc_m[3], Nenc_m[4], Nenc_m[5], Nenc_m[6], Nenc_m[7], Nenc_m[8], Nenc_m[9],  Nenc_m[10],  Nenc_m[11],  Nenc_m[12]);

	if(Nenc_m[1] > 0) BSBMStep_kernel <2, 2> <<< Nenc_m[1], 4, 0, BSStream[0] >>> (random_d, x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs_d, Encpairs2_d, dt_d, FGt[si] / ll, Msun_d, U_d, 0, index_d, BSstop_d, Ncoll_d, Coll_d, time_d, spin_d, love_d, createFlag_d, Nst, NconstT, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl, P.NencMax);
	if(Nenc_m[2] > 0) BSBMStep_kernel <4, 4> <<< Nenc_m[2], 16, 0, BSStream[1] >>> (random_d, x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs_d, Encpairs2_d, dt_d, FGt[si] / ll, Msun_d, U_d, 1, index_d, BSstop_d, Ncoll_d, Coll_d, time_d, spin_d, love_d, createFlag_d, Nst, NconstT, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl, P.NencMax);
	if(Nenc_m[3] > 0) BSBMStep_kernel <8, 8> <<< Nenc_m[3], 64, 0, BSStream[2] >>> (random_d, x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs_d, Encpairs2_d, dt_d, FGt[si] / ll, Msun_d, U_d, 2, index_d, BSstop_d, Ncoll_d, Coll_d, time_d, spin_d, love_d, createFlag_d, Nst, NconstT, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl, P.NencMax);
	if(Nenc_m[4] > 0) BSBMStep_kernel <16, 16> <<< Nenc_m[4], 256, 0, BSStream[3] >>> (random_d, x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs_d, Encpairs2_d, dt_d, FGt[si] / ll, Msun_d, U_d, 3, index_d, BSstop_d, Ncoll_d, Coll_d, time_d, spin_d, love_d, createFlag_d, Nst, NconstT, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl, P.NencMax);
	if(Nenc_m[5] > 0) BSBMStep_kernel <32, 8> <<< Nenc_m[5], 256, 0, BSStream[4] >>> (random_d, x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs_d, Encpairs2_d, dt_d, FGt[si] / ll, Msun_d, U_d, 4, index_d, BSstop_d, Ncoll_d, Coll_d, time_d, spin_d, love_d, createFlag_d, Nst, NconstT, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl, P.NencMax);

	cudaDeviceSynchronize();
}

__host__ void Data::BSBM3Call(int si, int noColl, double ll){

//printf(" %d | %d %d %d | %d %d %d | %d %d %d | %d %d %d\n", Nenc_m[0], Nenc_m[1], Nenc_m[2], Nenc_m[3], Nenc_m[4], Nenc_m[5], Nenc_m[6], Nenc_m[7], Nenc_m[8], Nenc_m[9],  Nenc_m[10],  Nenc_m[11],  Nenc_m[12]);

	if(Nenc_m[1] > 0) BSBM3Step_kernel <2, 2> <<< Nenc_m[1], 4, 0, BSStream[0] >>> (random_d, x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs2_d, groupIndex_d, dt_d, FGt[si] / ll, Msun_d, U_d, 0, index_d, BSstop_d, Ncoll_d, Coll_d, time_d, spin_d, love_d, createFlag_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NT, NconstT, NWriteEnc_d, writeEnc_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[2] > 0) BSBM3Step_kernel <4, 4> <<< Nenc_m[2], 16, 0, BSStream[1] >>> (random_d, x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs2_d, groupIndex_d, dt_d, FGt[si] / ll, Msun_d, U_d, 1, index_d, BSstop_d, Ncoll_d, Coll_d, time_d, spin_d, love_d, createFlag_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NT, NconstT, NWriteEnc_d, writeEnc_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[3] > 0) BSBM3Step_kernel <8, 8> <<< Nenc_m[3], 64, 0, BSStream[2] >>> (random_d, x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs2_d, groupIndex_d, dt_d, FGt[si] / ll, Msun_d, U_d, 2, index_d, BSstop_d, Ncoll_d, Coll_d, time_d, spin_d, love_d, createFlag_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NT, NconstT, NWriteEnc_d, writeEnc_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[4] > 0) BSBM3Step_kernel <16, 16> <<< Nenc_m[4], 256, 0, BSStream[3] >>> (random_d, x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs2_d, groupIndex_d, dt_d, FGt[si] / ll, Msun_d, U_d, 3, index_d, BSstop_d, Ncoll_d, Coll_d, time_d, spin_d, love_d, createFlag_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NT, NconstT, NWriteEnc_d, writeEnc_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[5] > 0) BSBM3Step_kernel <32, 8> <<< Nenc_m[5], 256, 0, BSStream[4] >>> (random_d, x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs2_d, groupIndex_d, dt_d, FGt[si] / ll, Msun_d, U_d, 4, index_d, BSstop_d, Ncoll_d, Coll_d, time_d, spin_d, love_d, createFlag_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NT, NconstT, NWriteEnc_d, writeEnc_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);


	/*
	if(Nenc_m[1] > 0) BSAM3_kernel < 64 > <<< Nenc_m[1], 64 , 0, BSStream[0] >>> (random_d, x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, love_d, createFlag_d, Encpairs_d, Encpairs2_d, groupIndex_d, dt_d, FGt[si] / ll, Msun_d, U_d, 0, NT, NconstT, P.NencMax, BSstop_d, Ncoll_d, Coll_d, time_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[2] > 0) BSAM3_kernel < 64 > <<< Nenc_m[2], 64 , 0, BSStream[1] >>> (random_d, x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, love_d, createFlag_d, Encpairs_d, Encpairs2_d, groupIndex_d, dt_d, FGt[si] / ll, Msun_d, U_d, 1, NT, NconstT, P.NencMax, BSstop_d, Ncoll_d, Coll_d, time_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[3] > 0) BSAM3_kernel < 64 > <<< Nenc_m[3], 64 , 0, BSStream[2] >>> (random_d, x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, love_d, createFlag_d, Encpairs_d, Encpairs2_d, groupIndex_d, dt_d, FGt[si] / ll, Msun_d, U_d, 2, NT, NconstT, P.NencMax, BSstop_d, Ncoll_d, Coll_d, time_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	*/

	if(Nenc_m[6] > 0) BSAM3_kernel < 64 > <<< Nenc_m[6], 64 , 0, BSStream[5] >>> (random_d, x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, love_d, createFlag_d, Encpairs_d, Encpairs2_d, groupIndex_d, dt_d, FGt[si] / ll, Msun_d, U_d, 5, NT, NconstT, P.NencMax, BSstop_d, Ncoll_d, Coll_d, time_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseGR, P.MinMass, P.UseTestParticles, P.SLevels, noColl);


	cudaDeviceSynchronize();
}

__host__ int Data::RemoveCall(){
#if def_TTV == 0
	int NminFlag = remove();
	if(NminFlag == 1){
		fprintf(masterfile, "Number of bodies smaller than Nmin, simulation stopped\n");
		printf("Number of bodies smaller than Nmin, simulation stopped\n");
		return 0;
	}
	if(NminFlag == 2){
		fprintf(masterfile, "Number of test particles smaller than NminTP, simulation stopped\n");
		printf("Number of test particles smaller than NminTP, simulation stopped\n");
		return 0;
	}
	CollisionFlag = 0;
#endif
	return 1;
}

__host__ int Data::CollisionCall(int noColl){
#if def_TTV == 0
	if(Ncoll_m[0] > def_MaxColl){
		fprintf(masterfile, "Error: More Collisions than def_MaxColl, simulation stopped\n");
		printf("Error: More Collisions than def_MaxColl, simulation stopped\n");
		return 0;
	}
	int stopAtCollision = 0;
	if(noColl == 0){
		stopAtCollision = printCollisions();
	}
	CollisionFlag = 1;

	if(noColl == 0 && stopAtCollision == 1 && (P.StopAtCollision == 1 || P.CollTshift > 1.0)){
		
printf("save  1\n");
		save_kernel <<< (N_h[0] + Nsmall_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, x4bb_d, v4bb_d, spin_d, spinbb_d, rcrit_d, rcritv_d, rcritbb_d, rcritvbb_d, index_d, indexbb_d, N_h[0] + Nsmall_h[0], NconstT, P.SLevels, 1);
		
		if(P.CollTshift > 1.0){
			//restore old step and increase radius 
			int NOld = Ncoll_m[0];
			for(int i = 0; i < NOld; ++i){

printf("Backup step 1 %d %.20g %.20g %.20g\n", i, Coll_h[i * def_NColl] * 365.25, time_h[0] - idt_h[0], (Coll_h[i * def_NColl] * 365.25 - (time_h[0] - idt_h[0])) / idt_h[0]);
				int2 ij;
				ij.x = int(Coll_h[i * def_NColl + 1]);
				ij.y = int(Coll_h[i * def_NColl + 13]);
printf("Tshiftpairs %d %d\n", ij.x, ij.y);
#if def_CPU == 0
				cudaMemcpyToSymbol(CollTshiftpairs_c, &ij, sizeof(int2), 0, cudaMemcpyHostToDevice);
#else
				CollTshiftpairs_c[0].x = ij.x;
				CollTshiftpairs_c[0].y = ij.y;
#endif

				IrregularStep(1.0);

				int N0 = Ncoll_m[0];
				BSAstop_h[0] = 0;
				cudaMemset(BSstop_d, 0, sizeof(int));
				bStep(1);
				cudaDeviceSynchronize();
				error = cudaGetLastError();
				if(error != 0){
					printf("Backup step 1  error = %d = %s\n",error, cudaGetErrorString(error));
					return 0;
				}
				//If all collisions are found, the number of collisions must be doubled now

printf("N0 %d %d %d\n", NOld, N0, Ncoll_m[0]);
				if(N0 == Ncoll_m[0]){
printf("Revert time step\n");
					IrregularStep(-1.0);
printf("Backup step -1 %.20g %.20g %.20g\n", (time_h[0] - idt_h[0]) - idt_h[0], time_h[0] - idt_h[0], -1.0);
					N0 = Ncoll_m[0];
					BSAstop_h[0] = 0;
					cudaMemset(BSstop_d, 0, sizeof(int));
					bStep(-1);
					cudaDeviceSynchronize();
					error = cudaGetLastError();
					if(error != 0){
						printf("Backup step -1  error = %d = %s\n",error, cudaGetErrorString(error));
						return 0;
					}
					if(N0 == Ncoll_m[0]){
						printf("Error: Collision time could not be reconstructed. Maybe CollTshift is too large.\n");
						return 0;
					}

				}
				

			}
			cudaDeviceSynchronize();
			cudaMemcpy(Coll_h, Coll_d, sizeof(double) * def_NColl * Ncoll_m[0], cudaMemcpyDeviceToHost);	
			printCollisionsTshift();
printf("print Collision Tshift\n");
		}


		if(P.StopAtCollision == 1){
			double Coltime = 1.0e100;
			for(int i = 0; i < Ncoll_m[0]; ++i){
				if(Coll_h[i * def_NColl + 2] >= P.StopMinMass && Coll_h[i * def_NColl + 14] >= P.StopMinMass){
					Coltime = fmin(Coltime, Coll_h[i * def_NColl]);
				}
 
			}
printf("Backup step 2 %.20g %.20g %.20g\n", Coltime * 365.25, time_h[0] - idt_h[0], (Coltime * 365.25 - (time_h[0] - idt_h[0])) / idt_h[0]);

			IrregularStep(1.0 * ((Coltime * 365.25 - time_h[0] + idt_h[0]) / idt_h[0]));
			BSAstop_h[0] = 0;
			cudaMemset(BSstop_d, 0, sizeof(int));
			bStep(2);
			cudaDeviceSynchronize();
			error = cudaGetLastError();
			if(error != 0){
				printf("Backup step 2  error = %d = %s\n",error, cudaGetErrorString(error));
				return 0;
			}

			time_h[0] = Coltime * 365.25;
			CoordinateOutput(2);
			P.ci = -1;
			return 0;
		}

		Ncoll_m[0] = 0;
		BSAstop_h[0] = 0;
		cudaMemset(BSstop_d, 0, sizeof(int));
	
		// P.StopAtCollision = 0
printf("save -1\n");
		save_kernel <<< (N_h[0] + Nsmall_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, x4bb_d, v4bb_d, spin_d, spinbb_d, rcrit_d, rcritv_d, rcritbb_d, rcritvbb_d, index_d, indexbb_d, N_h[0] + Nsmall_h[0], NconstT, P.SLevels, -1);
		IrregularStep(1.0);
		return 1;
	}
	else{
		Ncoll_m[0] = 0;
		return 1;
	}
#else

	return 1;
#endif
}

__host__ int Data::CollisionMCall(int noColl){
#if def_TTV == 0
	
	
	if(Ncoll_m[0] > def_MaxColl){
		fprintf(masterfile, "Error: More Collisions than def_MaxColl, simulation stopped\n");
		printf("Error: More Collisions than def_MaxColl, simulation stopped\n");
		return 0;
	}
	int stopAtCollision = 0;
	if(noColl == 0){
		stopAtCollision = printCollisions();
	}
	CollisionFlag = 1;

	if(noColl == 0 && stopAtCollision == 1 && (P.StopAtCollision == 1 || P.CollTshift > 1.0)){

printf("save  1\n");
		save_kernel <<< (NT + NsmallT + 127) / 128, 128 >>> (x4_d, v4_d, x4bb_d, v4bb_d, spin_d, spinbb_d, rcrit_d, rcritv_d, rcritbb_d, rcritvbb_d, index_d, indexbb_d, NT + NsmallT, NconstT, P.SLevels, 1);

		if(P.CollTshift > 1.0){
			//restore old step and increase radius 
			int NOld = Ncoll_m[0];
			for(int i = 0; i < NOld; ++i){

printf("Backup step 1 %d %.20g\n", i, Coll_h[i * def_NColl] * 365.25);
				int2 ij;
				ij.x = int(Coll_h[i * def_NColl + 1]);
				ij.y = int(Coll_h[i * def_NColl + 13]);
printf("Tshiftpairs %d %d\n", ij.x, ij.y);
#if def_CPU == 0
				cudaMemcpyToSymbol(CollTshiftpairs_c, &ij, sizeof(int2), 0, cudaMemcpyHostToDevice);
#else
				CollTshiftpairs_c[0].x = ij.x;
				CollTshiftpairs_c[0].y = ij.y;
#endif

				IrregularStep(1.0);

				int N0 = Ncoll_m[0];
				cudaMemset(BSstop_d, 0, sizeof(int));
				if(UseM3 == 0){
					bStepM(1);
				}
				else{
					bStepM3(1);
				}
				cudaDeviceSynchronize();
				error = cudaGetLastError();
				if(error != 0){
					printf("Backup step 1  error = %d = %s\n",error, cudaGetErrorString(error));
					return 0;
				}
				//If all collisions are found, the number of collisions must be doubled now
printf("N0 %d %d %d\n", NOld, N0, Ncoll_m[0]);

				if(N0 == Ncoll_m[0]){
printf("Revert time step\n");
					IrregularStep(-1.0);
printf("Backup step -1 %.20g\n", -1.0);
					N0 = Ncoll_m[0];
					cudaMemset(BSstop_d, 0, sizeof(int));
					if(UseM3 == 0){
						bStepM(-1);
					}
					else{
						bStepM3(-1);
					}
					cudaDeviceSynchronize();
					error = cudaGetLastError();
					if(error != 0){
						printf("Backup step -1  error = %d = %s\n",error, cudaGetErrorString(error));
						return 0;
					}
					if(N0 == Ncoll_m[0]){
						printf("Error: Collision time could not be reconstructed. Maybe CollTshift is too large.\n");
						return 0;
					}

				}

			}
			cudaDeviceSynchronize();
			cudaMemcpy(Coll_h, Coll_d, sizeof(double) * def_NColl * Ncoll_m[0], cudaMemcpyDeviceToHost);	
			printCollisionsTshift();
printf("print Collision Tshift\n");
		}

		if(P.StopAtCollision == 1){
			double Coltime = 1.0e100;
			for(int i = 0; i < Ncoll_m[0]; ++i){
				if(Coll_h[i * def_NColl + 2] >= P.StopMinMass && Coll_h[i * def_NColl + 14] >= P.StopMinMass){
					Coltime = fmin(Coltime, Coll_h[i * def_NColl]);
				}
			}
printf("Backup step 2 %.20g\n", Coltime * 365.25);
	//st
			IrregularStep(1.0 * ((Coltime * 365.25 - time_h[0] + idt_h[0]) / idt_h[0]));
			cudaMemset(BSstop_d, 0, sizeof(int));
			if(UseM3 == 0){
				bStepM(2);
			}
			else{
				bStepM3(2);
			}
			cudaDeviceSynchronize();
			error = cudaGetLastError();
			if(error != 0){
				printf("Backup step 2  error = %d = %s\n",error, cudaGetErrorString(error));
				return 0;
			}

			time_h[0] = Coltime * 365.25;
			CoordinateOutput(2);
			P.ci = -1;
			return 0;
		}


		// P.StopAtCollision = 0
printf("save -1\n");
		save_kernel <<< (NT + NsmallT + 127) / 128, 128 >>> (x4_d, v4_d, x4bb_d, v4bb_d, spin_d, spinbb_d, rcrit_d, rcritv_d, rcritbb_d, rcritvbb_d, index_d, indexbb_d, NT + NsmallT, NconstT, P.SLevels, -1);
		IrregularStep(1.0);
		cudaMemset(BSstop_d, 0, sizeof(int));

		int NminFlag = remove();
		if(NminFlag > 0){
			stopSimulations();
		}
		Ncoll_m[0] = 0;

	}
	else{

		int NminFlag = remove();
		if(NminFlag > 0){
			stopSimulations();
		}
		Ncoll_m[0] = 0;
	}
#endif
	return 1;
}

__host__ int Data::writeEncCall(){
	int er = printEncounters();
	if(er == 0){
		return 0;
	}
	NWriteEnc_m[0] = 0;
	return 1;
}

__host__ int Data::EjectionCall(){
#if def_TTV == 0
	Ejection();
	EjectionFlag_m[0] = 0;
	EjectionFlag2 = 1;
	int NminFlag = remove();
	if(NminFlag > 0){
		//at least one simulation has less bodies than Nmin and must be stopped
		
		if(P.ci != 0){
			CoordinateOutput(3);
		}
		
		stopSimulations();
		if(Nst == 0){
			return 0;
		}
	}
#endif
	return 1;
}

__host__ int Data::StopAtEncounterCall(){
#if def_TTV == 0
	
	if(Nst == 1){
		n1_h[0] = -1;
		
	}
	else{
		cudaMemcpy(n1_h, n1_d, Nst * sizeof(double), cudaMemcpyDeviceToHost);
	}
	if(P.ci != 0){
		CoordinateOutput(3);
	}
	stopSimulations();
	if(Nst == 0){
		return 0;
	}
#endif
	return 1;
}



// ******************************************
// This fucntions calls the PoincareSection kernel
// It prints the section of surface: time, particle ID, x, v, to the file Poincare_X.dat
//Authors: Simon Grimm, Joachim Stadel
//March 2014
// *******************************************
#if def_poincareFlag == 1
__host__ int Data::PoincareSectionCall(double t){
	if(SIn > 1){
		printf("Compute Poincare Sections only with the second Order integrator!\n");
		fprintf(masterfile, "Compute Poincare Sections only with the second Order integrator!\n");
		return 0;
	}
	PoincareSection_kernel <<< (N_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, xold_d, vold_d, index_d, Msun_h[0].x, N_h[0], 0, PFlag_d);
	
	cudaMemcpy(PFlag_h, PFlag_d, sizeof(int), cudaMemcpyDeviceToHost);
	if(PFlag_h[0] == 1){
		cudaMemcpy(x4_h, xold_d, N_h[0] * sizeof(double4), cudaMemcpyDeviceToHost);
		cudaMemcpy(v4_h, vold_d, N_h[0] * sizeof(double4), cudaMemcpyDeviceToHost);
		cudaMemcpy(index_h, index_d, N_h[0] * sizeof(int), cudaMemcpyDeviceToHost);
		for(int i = 0; i < N_h[0]; ++i){
			if(v4_h[i].w < 0.0 && x4_h[i].w >= 0.0){
				fprintf(poincarefile, "%.16g %d %g %g\n", t/365.25, index_h[i], x4_h[i].x, v4_h[i].x);
				
			}
		}
		PFlag_h[0] = 0; 
		cudaMemcpy(PFlag_d, PFlag_h, sizeof(int), cudaMemcpyHostToDevice);
	}
	return 1;
}
#endif

//Recursive symplectic close encounter Step.
//At the last level BS is called
//noColl == 3 is used in tuning step
//noColl == 1 or -1 is used in collision precision backtracing
__host__ void Data::SEnc(double &time, int SLevel, double ll, int si, int noColl){

#if def_CPU == 0
	int nt = min(N_h[0] + Nsmall_h[0], 512);
	int nb = (N_h[0] + Nsmall_h[0] + nt - 1) / nt;
#endif
	if((noColl == 1 || noColl == -1) && BSAstop_h[0] == 3){
		printf("stop SEnc call b\n");
		return; 
	}

//printf("SEnc %d %d %d\n", SLevel, Nencpairs_h[0], Nencpairs2_h[0]);	
	if(noColl == 0 || SLevel > 0 || noColl == 3){	
		int NN = N_h[0] + Nsmall_h[0];
		setEnc3_kernel <<< nb, nt >>> (NN, Nencpairs3_d + SLevel, Encpairs3_d + SLevel * NBNencT, scan_d, P.NencMax);
		groupS2_kernel <<< (Nencpairs2_h[0] + 511) / 512, 512 >>> (Nencpairs2_d, Encpairs2_d, Nencpairs3_d + SLevel, Encpairs3_d + SLevel * NBNencT, scan_d, P.NencMax, P.UseTestParticles, N_h[0], SLevel);	

#if def_CPU == 0
		if(NN <= WarpSize){
			Scan32c_kernel <<< 1, WarpSize >>> (scan_d, Encpairs3_d + SLevel * NBNencT, Nencpairs3_d + SLevel, NN, P.NencMax);

		}
		else if(NN <= 1024){
			int nn = (NN + WarpSize - 1) / WarpSize;
			Scan32a_kernel <<< 1, nn * WarpSize, WarpSize * sizeof(int) >>> (scan_d, Encpairs3_d + SLevel * NBNencT, Nencpairs3_d + SLevel, NN, P.NencMax);
		}
		else{
			int nct = 1024;
			int ncb = min((NN + nct - 1) / nct, 1024);

			Scan32d1_kernel <<< ncb, nct, WarpSize * sizeof(int) >>> (scan_d, NN);
			Scan32d2_kernel <<< 1, ((ncb + WarpSize - 1) / WarpSize) * WarpSize, WarpSize * sizeof(int)  >>> (scan_d, NN);
			Scan32d3_kernel  <<< ncb, nct >>>  (Encpairs3_d + SLevel * NBNencT, scan_d, Nencpairs3_d + SLevel, NN, P.NencMax);
		}
#else
		Scan_cpu(scan_d, Encpairs3_d + SLevel * NBNencT, Nencpairs3_d + SLevel, NN, P.NencMax);
#endif

		cudaMemcpy(Nencpairs3_h + SLevel, Nencpairs3_d + SLevel, sizeof(int), cudaMemcpyDeviceToHost);
	}
	cudaDeviceSynchronize();
// /*if(timeStep % 1000 == 0) */printf("Base0 %d %d %d\n", Nencpairs_h[0], Nencpairs2_h[0], Nencpairs3_h[SLevel]);

#if def_CPU == 0
	int nt3 = min(Nencpairs3_h[SLevel], 512);
	int nb3 = (Nencpairs3_h[SLevel] + nt3 - 1) / nt3;
	int ntf3 = min(Nencpairs3_h[SLevel], 128);
	int nbf3 = (Nencpairs3_h[SLevel] + ntf3 - 1) / ntf3;
#endif
	if(P.SERIAL_GROUPING == 1){
		if(noColl == 0 || SLevel > 0 || noColl == 3){	
			SortSb_kernel <<< nb3, nt3 >>> (Encpairs3_d + SLevel * NBNencT, Nencpairs3_d + SLevel, N_h[0] + Nsmall_h[0], P.NencMax);
		}
	}
	
	SLevel += 1;
	double l = P.SLSteps;	//number of sub steps
	ll *= l;
	//loop over sub time steps
	for(int s = 0; s < l; ++s){
// /*if(timeStep % 1000 == 0)*/ printf("Level %d %d %g %g\n", SLevel, s, ll, time);
		
		if(s == 0){
#if def_SLn1 == 0
			RcritS_kernel <<< nb3, nt3 >>>  (xold_d, vold_d, 1.0 / (3.0 * Msun_h[0].x), rcrit_d, rcritv_d, dt_h[0] / ll, n1_h[0], n2_h[0], N_h[0] + Nsmall_h[0], Nencpairs_d, Nencpairs2_d, Nencpairs3_d + SLevel - 1, Encpairs3_d + (SLevel - 1) * NBNencT, P.NencMax, NconstT, SLevel);
#else
			RcritS_kernel <<< nb3, nt3 >>>  (xold_d, vold_d, 1.0 / (3.0 * Msun_h[0].x), rcrit_d, rcritv_d, dt_h[0] / ll, n1_h[0] / ll, n2_h[0], N_h[0] + Nsmall_h[0], Nencpairs_d, Nencpairs2_d, Nencpairs3_d + SLevel - 1, Encpairs3_d + (SLevel - 1) * NBNencT, P.NencMax, NconstT, SLevel);
#endif
			kickS_kernel <<< nb3, nt3 >>> (x4_d, v4_d, xold_d, vold_d, rcritv_d, dt_h[0] / ll * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, Nencpairs3_d + SLevel - 1, Encpairs3_d + (SLevel - 1) * NBNencT, N_h[0] + Nsmall_h[0], NconstT, P.NencMax, SLevel, P.SLevels, 0);
		}
		else{
#if def_SLn1 == 0
			RcritS_kernel <<< nb3, nt3 >>>  (x4_d, v4_d, 1.0 / (3.0 * Msun_h[0].x), rcrit_d, rcritv_d, dt_h[0] / ll, n1_h[0], n2_h[0], N_h[0] + Nsmall_h[0], Nencpairs_d, Nencpairs2_d, Nencpairs3_d + SLevel - 1, Encpairs3_d + (SLevel - 1) * NBNencT, P.NencMax, NconstT, SLevel);
#else
			RcritS_kernel <<< nb3, nt3 >>>  (x4_d, v4_d, 1.0 / (3.0 * Msun_h[0].x), rcrit_d, rcritv_d, dt_h[0] / ll, n1_h[0] / ll, n2_h[0], N_h[0] + Nsmall_h[0], Nencpairs_d, Nencpairs2_d, Nencpairs3_d + SLevel - 1, Encpairs3_d + (SLevel - 1) * NBNencT, P.NencMax, NconstT, SLevel);
#endif
			kickS_kernel <<< nb3, nt3 >>> (x4_d, v4_d, x4_d, v4_d, rcritv_d, dt_h[0] / ll * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, Nencpairs3_d + SLevel - 1, Encpairs3_d + (SLevel - 1) * NBNencT, N_h[0] + Nsmall_h[0], NconstT, P.NencMax, SLevel, P.SLevels, 2);
		}
		cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
// /*if(timeStep % 1000 == 0) */printf("Nencpairs %d\n", Nencpairs_h[0]);
		fgS_kernel <<< nbf3, ntf3 >>> (x4_d, v4_d, xold_d, vold_d, index_d, dt_h[0] / ll * FGt[si], Msun_h[0].x, N_h[0] + Nsmall_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, 1, P.UseGR, Nencpairs3_d + SLevel - 1, Encpairs3_d + (SLevel - 1) * NBNencT, P.NencMax);
		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 63)/ 64, 64 >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d + SLevel * NconstT, rcritv_d + SLevel * NconstT, dt_h[0] / ll * FGt[si], Nencpairs_h[0], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, enccount_d, 1, N_h[0] + Nsmall_h[0], time, P.StopAtEncounter, Ncoll_d, P.MinMass);
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
// /*if(timeStep % 1000 == 0) */printf("Nencpairs2: %d %d n1: %g\n", Nencpairs_h[0], Nencpairs2_h[0], n1_h[0] / ll);
			if(Nencpairs2_h[0] > 0){
				
				if(SLevel < P.SLevels - 1){
					SEnc(time, SLevel, ll, si, noColl);
				}
				else{
#if def_CPU == 0
					groupCall();
#else
					group_cpu (Nenc_m, Nencpairs2_h, Encpairs2_h, Encpairs_h, P.NencMax, N_h[0] + Nsmall_h[0], N_h[0], P.SERIAL_GROUPING);
#endif


					cudaDeviceSynchronize();
					BSCall(si, time, noColl, ll);
					time += dt_h[0] / ll / dayUnit;

					if(noColl == 1 || noColl == -1){
						if(BSAstop_h[0] == 3){
							printf("stop SEnc call\n");
							return; 
						}
					}
				}
			}
		}
		kickS_kernel <<< nb3, nt3 >>> (x4_d, v4_d, x4_d, v4_d, rcritv_d, dt_h[0] / ll * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, Nencpairs3_d + SLevel - 1, Encpairs3_d + (SLevel - 1) * NBNencT, N_h[0] + Nsmall_h[0], NconstT, P.NencMax, SLevel, P.SLevels, 1);
	}
} 

#if def_CPU == 1
int Data::bStep(int noColl){
	Rcrit_cpu (x4_h, v4_h, x4b_h, v4b_h, spin_h, spinb_h, 1.0 / (3.0 * Msun_h[0].x), rcrit_h, rcritb_h, rcritv_h, rcritvb_h, index_h, indexb_h, dt_h[0], n1_h[0], n2_h[0], time_h, time_h[0], EjectionFlag_m, N_h[0] + Nsmall_h[0], NconstT, P.SLevels, noColl);
	kick32C_cpu (x4_h, v4_h, ab_h, N_h[0] + Nsmall_h[0], dt_h[0] * Kt[0]);
	HCCall(Ct[0], 1);
	fg_cpu (x4_h, v4_h, xold_h, vold_h, index_h, dt_h[0] * FGt[0], Msun_h[0].x, N_h[0] + Nsmall_h[0], aelimits_h, aecount_h, Gridaecount_h, Gridaicount_h, 0, P.UseGR);

	if(P.SLevels > 1){
		if(Nencpairs2_h[0] > 0){
			double time = time_h[0];
			SEnc(time, 0, 1.0, 0, noColl);
		}
	}
	else{
		BSCall(0, time_h[0], noColl, 1.0);
	}


	HCCall(Ct[0], -1);
	
	kick32C_cpu (x4_h, v4_h, ab_h, N_h[0] + Nsmall_h[0], dt_h[0] * Kt[0]);

	return 0;
}
#else
__host__ int Data::bStep(int noColl){
	Rcrit_kernel <<< (N_h[0] + Nsmall_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, x4b_d, v4b_d, spin_d, spinb_d, 1.0 / (3.0 * Msun_h[0].x), rcrit_d, rcritb_d, rcritv_d, rcritvb_d, index_d, indexb_d, dt_h[0], n1_h[0], n2_h[0], time_d, time_h[0], EjectionFlag_d, N_h[0] + Nsmall_h[0], NconstT, P.SLevels, noColl);
	kick32C_kernel <<< (N_h[0] + Nsmall_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, ab_d, N_h[0] + Nsmall_h[0], dt_h[0] * Kt[0]);
	HCCall(Ct[0], 1);
	fg_kernel <<<(N_h[0] + Nsmall_h[0] + FTX - 1)/FTX, FTX >>> (x4_d, v4_d, xold_d, vold_d, index_d, dt_h[0] * FGt[0], Msun_h[0].x, N_h[0] + Nsmall_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, 0, P.UseGR);
	cudaDeviceSynchronize();

	if(P.SLevels > 1){
		if(Nencpairs2_h[0] > 0){
			double time = time_h[0];
			SEnc(time, 0, 1.0, 0, noColl);
		}
	}
	else{
		BSCall(0, time_h[0], noColl, 1.0);
	}


	HCCall(Ct[0], -1);
	
	kick32C_kernel <<< (N_h[0] + Nsmall_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, ab_d, N_h[0] + Nsmall_h[0], dt_h[0] * Kt[0]);

	return 0;
}
#endif
__host__ int Data::bStepM(int noColl){
	RcritM_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, x4b_d, v4b_d, spin_d, spinb_d, Msun_d, rcrit_d, rcritb_d, rcritv_d, rcritvb_d, dt_d, n1_d, n2_d, Rcut_d, RcutSun_d, EjectionFlag_d, index_d, indexb_d, Nst, NT, time_d, idt_d, ict_d, delta_d, timeStep, StopFlag_d, NconstT, P.SLevels, noColl, Nstart);

	KickM2Simple_kernel < KM_Bl, KM_Bl2, NmaxM, 1 > <<< (NT + KM_Bl2 - 1) / KM_Bl2, KM_Bl>>> (x4_d, v4_d, v4b_d, a_d, dt_d, Kt[0], index_d, NT, Nst, Nstart);

	if(P.UseGR == 1){
		convertVToPseidovM <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, ErrorFlag_m, Msun_d, NT);
	}
	HCM2_kernel < HCM_Bl, HCM_Bl2, NmaxM, 0 > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, dt_d, Msun_d, index_d, NT, Ct[0], Nencpairs_d, Nencpairs2_d, Nenc_d, Nst, P.UseGR, Nstart);
	fgMSimple_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, xold_d, vold_d, dt_d, Msun_d, index_d, NT, FGt[0], 0, P.UseGR, Nstart);
	cudaDeviceSynchronize();

	BSBMCall(0, noColl, 1.0);

	HCM2_kernel < HCM_Bl, HCM_Bl2, NmaxM, 0 > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, dt_d, Msun_d, index_d, NT, Ct[0], Nencpairs_d, Nencpairs2_d, Nenc_d, Nst, P.UseGR, Nstart);
	if(P.UseGR == 1){
		convertPseudovToVM <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, Msun_d, NT);
	}
	KickM2Simple_kernel < KM_Bl, KM_Bl2, NmaxM, 2 > <<< (NT + KM_Bl2 - 1) / KM_Bl2, KM_Bl>>> (x4_d, v4_d, v4b_d, a_d, dt_d, Kt[0], index_d, NT, Nst, Nstart);

	return 0;
}
__host__ int Data::bStepM3(int noColl){
	RcritM_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, x4b_d, v4b_d, spin_d, spinb_d, Msun_d, rcrit_d, rcritb_d, rcritv_d, rcritvb_d, dt_d, n1_d, n2_d, Rcut_d, RcutSun_d, EjectionFlag_d, index_d, indexb_d, Nst, NT, time_d, idt_d, ict_d, delta_d, timeStep, StopFlag_d, NconstT, P.SLevels, noColl, Nstart);

	KickM3_kernel <<< NT, dim3(KTM3, 1, 1), WarpSize * sizeof(double3) >>> (x4_d, v4_d, a_d, rcritv_d, Nencpairs_d, Encpairs_d, dt_d, Kt[0], index_d, NT, N_d, NBS_d, 1, 0);

	if(P.UseGR == 1){
		convertVToPseidovM <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, ErrorFlag_m, Msun_d, NT);
	}
	HC32aM_kernel <<< Nst, HCTM3, WarpSize * sizeof(double3) >>> (x4_d, v4_d, dt_d, Msun_d, N_d, NBS_d, Nst, Ct[0], P.UseGR, 1);

	fgMSimple_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, xold_d, vold_d, dt_d, Msun_d, index_d, NT, FGt[0], 0, P.UseGR, Nstart);
	cudaDeviceSynchronize();

	BSBM3Call(0, noColl, 1.0);

	HC32aM_kernel <<< Nst, HCTM3, WarpSize * sizeof(double3) >>> (x4_d, v4_d, dt_d, Msun_d, N_d, NBS_d, Nst, Ct[0], P.UseGR, 1);
	if(P.UseGR == 1){
		convertPseudovToVM <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, Msun_d, NT);
	}
	KickM3_kernel <<< NT, dim3(KTM3, 1, 1), WarpSize * sizeof(double3) >>> (x4_d, v4_d, a_d, rcritv_d, Nencpairs_d, Encpairs_d, dt_d, Kt[0], index_d, NT, N_d, NBS_d, 2, 0);

	return 0;
}

__host__ int Data::KickndevCall(int si, int EE){

	int NN = (N_h[0] + P.ndev - 1) / P.ndev;
	int Nx0 = 0;
	int Nx1 = NN;
	int nb = (((NN + KP - 1)/ KP) + KTX - 1) / KTX;
	cudaDeviceSynchronize();

	if(P.ndev > 1){
		cudaMemcpy(rcritv_d1, rcritv_d, NconstT * P.SLevels * sizeof(double), cudaMemcpyDefault);
		cudaMemcpy(x4_d1, x4_d, NconstT * sizeof(double4), cudaMemcpyDefault);
	}
	if(P.ndev > 2){
		cudaMemcpy(rcritv_d2, rcritv_d, NconstT * P.SLevels * sizeof(double), cudaMemcpyDefault);
		cudaMemcpy(x4_d2, x4_d, NconstT * sizeof(double4), cudaMemcpyDefault);
	}
	if(P.ndev > 3){
		cudaMemcpy(rcritv_d3, rcritv_d, NconstT * P.SLevels * sizeof(double), cudaMemcpyDefault);
		cudaMemcpy(x4_d3, x4_d, NconstT * sizeof(double4), cudaMemcpyDefault);
	}
	cudaDeviceSynchronize();
	if(P.ndev > 0){
		Nx1 = min(Nx1, N_h[0]);
//printf("****** %d %d %d | %d %d %d %d\n", 0, N_h[0], (((N_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, nb, Nx0, Nx1, Nx1 - Nx0);
		if(P.KickFloat == 0){
			acc4C_kernel <<< dim3(nb, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> (x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, Nx0, Nx1, 0, N_h[0], P.NencMax, KP, EE);
		}
		else{
			acc4Cf_kernel <<< dim3(nb, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> (x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, Nx0, Nx1, 0, N_h[0], P.NencMax, KP, EE);
		}

		Nx0 += NN;
		Nx1 += NN;
	}
	if(P.ndev > 1){
		cudaSetDevice(P.dev[1]);
		Nx1 = min(Nx1, N_h[0]);
//printf("****** %d %d %d | %d %d %d %d\n", 1, N_h[0], (((N_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, nb, Nx0, Nx1, Nx1 - Nx0);
		if(P.KickFloat == 0){
			acc4C_kernel <<< dim3(nb, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> (x4_d1, a_d, rcritv_d1, Encpairs_d1, Encpairs2_d1, Nencpairs_d1, EncFlag_d, Nx0, Nx1, 0, N_h[0], P.NencMax, KP, EE);
		}
		else{
			acc4Cf_kernel <<< dim3(nb, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> (x4_d1, a_d, rcritv_d1, Encpairs_d1, Encpairs2_d1, Nencpairs_d1, EncFlag_d, Nx0, Nx1, 0, N_h[0], P.NencMax, KP, EE);
		}

		Nx0 += NN;
		Nx1 += NN;
	}
	if(P.ndev > 2){
		cudaSetDevice(P.dev[2]);
		Nx1 = min(Nx1, N_h[0]);
//printf("****** %d %d %d | %d %d %d %d\n", 2, N_h[0], (((N_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, nb, Nx0, Nx1, Nx1 - Nx0);
		if(P.KickFloat == 0){
			acc4C_kernel <<< dim3(nb, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> (x4_d2, a_d, rcritv_d2, Encpairs_d2, Encpairs2_d2, Nencpairs_d2, EncFlag_d, Nx0, Nx1, 0, N_h[0], P.NencMax, KP, EE);
		}
		else{
			acc4Cf_kernel <<< dim3(nb, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> (x4_d2, a_d, rcritv_d2, Encpairs_d2, Encpairs2_d2, Nencpairs_d2, EncFlag_d, Nx0, Nx1, 0, N_h[0], P.NencMax, KP, EE);
		}

		Nx0 += NN;
		Nx1 += NN;
	}
	if(P.ndev > 3){
		cudaSetDevice(P.dev[3]);
		Nx1 = min(Nx1, N_h[0]);
//printf("****** %d %d %d | %d %d %d %d\n", 3, N_h[0], (((N_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, nb, Nx0, Nx1, Nx1 - Nx0);
		if(P.KickFloat == 0){
			acc4C_kernel <<< dim3(nb, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> (x4_d3, a_d, rcritv_d3, Encpairs_d3, Encpairs2_d3, Nencpairs_d3, EncFlag_d, Nx0, Nx1, 0, N_h[0], P.NencMax, KP, EE);
		}
		else{
			acc4Cf_kernel <<< dim3(nb, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> (x4_d3, a_d, rcritv_d3, Encpairs_d3, Encpairs2_d3, Nencpairs_d3, EncFlag_d, Nx0, Nx1, 0, N_h[0], P.NencMax, KP, EE);
		}

		Nx0 += NN;
		Nx1 += NN;
	}

//printf("%d %d %d %d\n", Nencpairs_h[0], Nencpairs_h[1], Nencpairs_h[2], Nencpairs_h[3]);

	//Synchronize all devices
	for(int i = 0; i < P.ndev; ++i){
		cudaSetDevice(P.dev[i]);
		if(i == 0){
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
		}
		if(i == 1){
			cudaMemcpy(Nencpairs_h + 1, Nencpairs_d1, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemset(Nencpairs_d1, 0, sizeof(int));
		}
		if(i == 2){
			cudaMemcpy(Nencpairs_h + 2, Nencpairs_d2, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemset(Nencpairs_d2, 0, sizeof(int));
		}
		if(i == 3){
			cudaMemcpy(Nencpairs_h + 3, Nencpairs_d3, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemset(Nencpairs_d3, 0, sizeof(int));
		}
		cudaDeviceSynchronize();
	}

	Nx0 = 0;
	Nx1 = NN;

	for(int i = 0; i < P.ndev; ++i){
		Nx1 = min(Nx1, N_h[0]);
		cudaSetDevice(P.dev[i]);
		if(i == 0){
			cudaMemcpy(Encpairs_d + Nencpairs_h[0], Encpairs_d1, Nencpairs_h[1] * sizeof(int2), cudaMemcpyDefault);
			Nencpairs_h[0] += Nencpairs_h[1];
			Nencpairs_h[1] = 0;
			cudaMemcpy(Encpairs_d + Nencpairs_h[0], Encpairs_d2, Nencpairs_h[2] * sizeof(int2), cudaMemcpyDefault);
			Nencpairs_h[0] += Nencpairs_h[2];
			Nencpairs_h[2] = 0;
			cudaMemcpy(Encpairs_d + Nencpairs_h[0], Encpairs_d3, Nencpairs_h[3] * sizeof(int2), cudaMemcpyDefault);
			Nencpairs_h[0] += Nencpairs_h[3];
			Nencpairs_h[3] = 0;
			cudaMemcpy(Nencpairs_d, Nencpairs_h, sizeof(int), cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();


			if(P.SERIAL_GROUPING == 1){
				Sortb_kernel <<< (NN + RTX - 1) / RTX, RTX >>> (Encpairs2_d, Nx0, Nx1, P.NencMax);
			}
			kick32Ab_kernel <<< (NN + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[si] * def_ksq, Nencpairs_d, Encpairs2_d, Nx0, Nx1, P.NencMax, 1);
		}
		if(i == 1){
			CollectGPUsAb_kernel <<< (NN + RTX - 1) / RTX, RTX >>> (x4_d1, Encpairs2_d1, Encpairs2_d, Nx0, Nx1, P.NencMax);
	
			if(P.SERIAL_GROUPING == 1){
				Sortb_kernel <<< (NN + RTX - 1) / RTX, RTX >>> (Encpairs2_d1, Nx0, Nx1, P.NencMax);
			}
			kick32Ab_kernel <<< (NN + RTX - 1) / RTX, RTX >>> (x4_d1, v4_d, a_d, ab_d, rcritv_d1, dt_h[0] * Kt[si] * def_ksq, Nencpairs_d, Encpairs2_d1, Nx0, Nx1, P.NencMax, 1);
		}
		if(i == 2){
			CollectGPUsAb_kernel <<< (NN + RTX - 1) / RTX, RTX >>> (x4_d2, Encpairs2_d2, Encpairs2_d, Nx0, Nx1, P.NencMax);

			if(P.SERIAL_GROUPING == 1){
				Sortb_kernel <<< (NN + RTX - 1) / RTX, RTX >>> (Encpairs2_d2, Nx0, Nx1, P.NencMax);
			}
			kick32Ab_kernel <<< (NN + RTX - 1) / RTX, RTX >>> (x4_d2, v4_d, a_d, ab_d, rcritv_d2, dt_h[0] * Kt[si] * def_ksq, Nencpairs_d, Encpairs2_d2, Nx0, Nx1, P.NencMax, 1);
		}
		if(i == 3){
			CollectGPUsAb_kernel <<< (NN + RTX - 1) / RTX, RTX >>> (x4_d3, Encpairs2_d3, Encpairs2_d, Nx0, Nx1, P.NencMax);

			if(P.SERIAL_GROUPING == 1){
				Sortb_kernel <<< (NN + RTX - 1) / RTX, RTX >>> (Encpairs2_d3, Nx0, Nx1, P.NencMax);
			}
			kick32Ab_kernel <<< (NN + RTX - 1) / RTX, RTX >>> (x4_d3, v4_d, a_d, ab_d, rcritv_d3, dt_h[0] * Kt[si] * def_ksq, Nencpairs_d, Encpairs2_d3, Nx0, Nx1, P.NencMax, 1);
		}
		Nx0 += NN;
		Nx1 += NN;
	}
	//Synchronize all devices
	for(int i = 0; i < P.ndev; ++i){
		cudaSetDevice(P.dev[i]);
		cudaDeviceSynchronize();
	}
	cudaSetDevice(P.dev[0]);

	return 1;
}

__host__ int Data::KickfirstndevCall(int EE){

	int NN = (N_h[0] + P.ndev - 1) / P.ndev;
	int Nx0 = 0;
	int Nx1 = NN;
	int nb = (((NN + KP - 1)/ KP) + KTX - 1) / KTX;
	cudaDeviceSynchronize();

	if(P.ndev > 1){
		cudaMemcpy(rcritv_d1, rcritv_d, NconstT * P.SLevels * sizeof(double), cudaMemcpyDefault);
		cudaMemcpy(x4_d1, x4_d, NconstT * sizeof(double4), cudaMemcpyDefault);
	}
	if(P.ndev > 2){
		cudaMemcpy(rcritv_d2, rcritv_d, NconstT * P.SLevels * sizeof(double), cudaMemcpyDefault);
		cudaMemcpy(x4_d2, x4_d, NconstT * sizeof(double4), cudaMemcpyDefault);
	}
	if(P.ndev > 3){
		cudaMemcpy(rcritv_d3, rcritv_d, NconstT * P.SLevels * sizeof(double), cudaMemcpyDefault);
		cudaMemcpy(x4_d3, x4_d, NconstT * sizeof(double4), cudaMemcpyDefault);
	}
	cudaDeviceSynchronize();
	if(P.ndev > 0){
		Nx1 = min(Nx1, N_h[0]);
//printf("****** %d %d %d | %d %d %d %d\n", 0, N_h[0], (((N_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, nb, Nx0, Nx1, Nx1 - Nx0);
		if(P.KickFloat == 0){
			acc4C_kernel <<< dim3(nb, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> (x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, Nx0, Nx1, 0, N_h[0], P.NencMax, KP, EE);
		}
		else{
			acc4Cf_kernel <<< dim3(nb, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> (x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, Nx0, Nx1, 0, N_h[0], P.NencMax, KP, EE);
		}

		Nx0 += NN;
		Nx1 += NN;
	}
	if(P.ndev > 1){
		cudaSetDevice(P.dev[1]);
		Nx1 = min(Nx1, N_h[0]);
//printf("****** %d %d %d | %d %d %d %d\n", 1, N_h[0], (((N_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, nb, Nx0, Nx1, Nx1 - Nx0);
		if(P.KickFloat == 0){
			acc4C_kernel <<< dim3(nb, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> (x4_d1, a_d, rcritv_d1, Encpairs_d1, Encpairs2_d1, Nencpairs_d1, EncFlag_d, Nx0, Nx1, 0, N_h[0], P.NencMax, KP, EE);
		}
		else{
			acc4Cf_kernel <<< dim3(nb, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> (x4_d1, a_d, rcritv_d1, Encpairs_d1, Encpairs2_d1, Nencpairs_d1, EncFlag_d, Nx0, Nx1, 0, N_h[0], P.NencMax, KP, EE);
		}

		Nx0 += NN;
		Nx1 += NN;
	}
	if(P.ndev > 2){
		cudaSetDevice(P.dev[2]);
		Nx1 = min(Nx1, N_h[0]);
//printf("****** %d %d %d | %d %d %d %d\n", 2, N_h[0], (((N_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, nb, Nx0, Nx1, Nx1 - Nx0);
		if(P.KickFloat == 0){
			acc4C_kernel <<< dim3(nb, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> (x4_d2, a_d, rcritv_d2, Encpairs_d2, Encpairs2_d2, Nencpairs_d2, EncFlag_d, Nx0, Nx1, 0, N_h[0], P.NencMax, KP, EE);
		}
		else{
			acc4Cf_kernel <<< dim3(nb, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> (x4_d2, a_d, rcritv_d2, Encpairs_d2, Encpairs2_d2, Nencpairs_d2, EncFlag_d, Nx0, Nx1, 0, N_h[0], P.NencMax, KP, EE);
		}

		Nx0 += NN;
		Nx1 += NN;
	}
	if(P.ndev > 3){
		cudaSetDevice(P.dev[3]);
		Nx1 = min(Nx1, N_h[0]);
//printf("****** %d %d %d | %d %d %d %d\n", 3, N_h[0], (((N_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, nb, Nx0, Nx1, Nx1 - Nx0);
		if(P.KickFloat == 0){
			acc4C_kernel <<< dim3(nb, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> (x4_d3, a_d, rcritv_d3, Encpairs_d3, Encpairs2_d3, Nencpairs_d3, EncFlag_d, Nx0, Nx1, 0, N_h[0], P.NencMax, KP, EE);
		}
		else{
			acc4Cf_kernel <<< dim3(nb, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> (x4_d3, a_d, rcritv_d3, Encpairs_d3, Encpairs2_d3, Nencpairs_d3, EncFlag_d, Nx0, Nx1, 0, N_h[0], P.NencMax, KP, EE);
		}

		Nx0 += NN;
		Nx1 += NN;
	}

//printf("%d %d %d %d\n", Nencpairs_h[0], Nencpairs_h[1], Nencpairs_h[2], Nencpairs_h[3]);

	//Synchronize all devices
	for(int i = 0; i < P.ndev; ++i){
		cudaSetDevice(P.dev[i]);
		if(i == 0){
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
		}
		if(i == 1){
			cudaMemcpy(Nencpairs_h + 1, Nencpairs_d1, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemset(Nencpairs_d1, 0, sizeof(int));
		}
		if(i == 2){
			cudaMemcpy(Nencpairs_h + 2, Nencpairs_d2, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemset(Nencpairs_d2, 0, sizeof(int));
		}
		if(i == 3){
			cudaMemcpy(Nencpairs_h + 3, Nencpairs_d3, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemset(Nencpairs_d3, 0, sizeof(int));
		}
		cudaDeviceSynchronize();
	}

	Nx0 = 0;
	Nx1 = NN;

	for(int i = 0; i < P.ndev; ++i){
		Nx1 = min(Nx1, N_h[0]);

		cudaSetDevice(P.dev[i]);
		if(i == 0){
			cudaMemcpy(Encpairs_d + Nencpairs_h[0], Encpairs_d1, Nencpairs_h[1] * sizeof(int2), cudaMemcpyDefault);
			Nencpairs_h[0] += Nencpairs_h[1];
			Nencpairs_h[1] = 0;
			cudaMemcpy(Encpairs_d + Nencpairs_h[0], Encpairs_d2, Nencpairs_h[2] * sizeof(int2), cudaMemcpyDefault);
			Nencpairs_h[0] += Nencpairs_h[2];
			Nencpairs_h[2] = 0;
			cudaMemcpy(Encpairs_d + Nencpairs_h[0], Encpairs_d3, Nencpairs_h[3] * sizeof(int2), cudaMemcpyDefault);
			Nencpairs_h[0] += Nencpairs_h[3];
			Nencpairs_h[3] = 0;
			cudaMemcpy(Nencpairs_d, Nencpairs_h, sizeof(int), cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();
		}
		if(i == 1){
			CollectGPUsAb_kernel <<< (NN + RTX - 1) / RTX, RTX >>> (x4_d1, Encpairs2_d1, Encpairs2_d, Nx0, Nx1, P.NencMax);
		}
		if(i == 2){
			CollectGPUsAb_kernel <<< (NN + RTX - 1) / RTX, RTX >>> (x4_d2, Encpairs2_d2, Encpairs2_d, Nx0, Nx1, P.NencMax);
		}
		if(i == 3){
			CollectGPUsAb_kernel <<< (NN + RTX - 1) / RTX, RTX >>> (x4_d3, Encpairs2_d3, Encpairs2_d, Nx0, Nx1, P.NencMax);
		}
		Nx0 += NN;
		Nx1 += NN;
	}
	//Synchronize all devices
	for(int i = 0; i < P.ndev; ++i){
		cudaSetDevice(P.dev[i]);
		cudaDeviceSynchronize();
	}
	cudaSetDevice(P.dev[0]);

	return 1;
}

#if def_CPU == 1
// *************************************************
// Step CPU
// *************************************************
void Data::step1_cpu(){

	Rcrit_cpu (x4_h, v4_h, x4b_h, v4b_h, spin_h, spinb_h, 1.0 / (3.0 * Msun_h[0].x), rcrit_h, rcritb_h, rcritv_h, rcritvb_h, index_h, indexb_h, dt_h[0], n1_h[0], n2_h[0], time_h, time_h[0], EjectionFlag_m, N_h[0], NconstT, P.SLevels, 0);

	double Msun = Msun_h[0].x;
	double dt05 = dt_h[0] * 0.5;
	double dt05Msun = dt05 / Msun;

	//Kick  
	kick32Ab_cpu (x4_h, v4_h, a_h, ab_h, rcritv_h, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_h, Encpairs2_h, 0, N_h[0], P.NencMax, 1);
	

	for(int si = 0; si < SIn; ++si){
		
		//HC
/*		double3 a = {0.0, 0.0, 0.0};

		//#pragma omp parallel for
		for(int i = 0; i < N_h[0]; ++i){
			a.x += x4_h[i].w * v4_h[i].x;
			a.y += x4_h[i].w * v4_h[i].y;
			a.z += x4_h[i].w * v4_h[i].z;
		//printf("HCA %d %d %g %g %g %g\n", id, j, v4.x, v4j.x, mj, a.x);
		}

		a.x *= dt05Msun;
		a.y *= dt05Msun;
		a.z *= dt05Msun;


		for(int i = 0; i < N_h[0]; ++i){
			x4_h[i].x += a.x;
			x4_h[i].y += a.y;
			x4_h[i].z += a.z;
		}
*/
		HCCall(Ct[si], 1);

		//FG

		#pragma omp parallel for 
		for(int i = 0; i < N_h[0]; ++i){
			unsigned int aecount = 0u;
			xold_h[i] = x4_h[i];
			vold_h[i] = v4_h[i];
			int index = index_h[i];
			float4 aelimits = aelimits_h[i];
			//fgcfull(x4_h[i], v4_h[i], dt, mu, P.UseGR);
			fgfull(x4_h[i], v4_h[i], dt_h[0], def_ksq * Msun, Msun, aelimits, aecount, Gridaecount_h, Gridaicount_h, 0, i, index, P.UseGR);
			aecount_h[i] += aecount;
		}

//                fg_cpu (x4_h, v4_h, xold_h, vold_h, index_h, dt_h[0] * FGt[si], Msun_h[0].x, N_h[0], aelimits_h, aecount_h, Gridaecount_h, Gridaicount_h, si, P.UseGR);

		//HC
	/*	a = {0.0, 0.0, 0.0};
		for(int j = 0; j < N_h[0]; ++j){
			a.x += x4_h[j].w * v4_h[j].x;
			a.y += x4_h[j].w * v4_h[j].y;
			a.z += x4_h[j].w * v4_h[j].z;
		//printf("HCA %d %d %g %g %g %g\n", id, j, v4.x, v4j.x, mj, a.x);
		}

		a.x *= dt05Msun;
		a.y *= dt05Msun;
		a.z *= dt05Msun;

		for(int i = 0; i < N_h[0]; ++i){
			x4_h[i].x += a.x;
			x4_h[i].y += a.y;
			x4_h[i].z += a.z;
		}
	*/
		HCCall(Ct[si], -1);
	}

	if(Nomp == 1){
		acc4E_cpu();
	}
	else{
		acc4D_cpu();
	}
	kick32Ab_cpu (x4_h, v4_h, a_h, ab_h, rcritv_h, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_h, Encpairs2_h, 0, N_h[0], P.NencMax, 1);

}


int Data::step_cpu(int noColl){
	Rcrit_cpu (x4_h, v4_h, x4b_h, v4b_h, spin_h, spinb_h, 1.0 / (3.0 * Msun_h[0].x), rcrit_h, rcritb_h, rcritv_h, rcritvb_h, index_h, indexb_h, dt_h[0], n1_h[0], n2_h[0], time_h, time_h[0], EjectionFlag_m, N_h[0], NconstT, P.SLevels, noColl);

	//use last time step information for setElements function, the beginning of the time step
	if(P.setElementsV == 2){ // convert barycentric velocities to heliocentric
		comCall(1);
	}
	if(P.setElementsV > 0){
		setElements_cpu(x4_h, v4_h, index_h, setElementsData_h, setElementsLine_h, Msun_h, dt_h, time_h, N_h[0], Nst, 0);
	}
	if(P.setElementsV == 2){
		comCall(-1);
	}

	if(P.SERIAL_GROUPING == 1){
		Sortb_cpu(Encpairs2_h, 0, N_h[0], P.NencMax);
	}
	if(doTransits == 0){
		if(EjectionFlag2 == 0){
			kick32Ab_cpu(x4_h, v4_h, a_h, ab_h, rcritv_h, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_h, Encpairs2_h, 0, N_h[0], P.NencMax, 1);
		}
		else{
			if(P.KickFloat == 0){
				if(Nomp == 1){
					acc4E_cpu();
				}
				else{
					acc4D_cpu();
				}
			}
			else{
				if(Nomp == 1){
					acc4Ef_cpu();
				}
				else{
					acc4Df_cpu();
				}
			}
			if(P.SERIAL_GROUPING == 1){
				Sortb_cpu (Encpairs2_h, 0, N_h[0], P.NencMax);
			}
			kick32Ab_cpu (x4_h, v4_h, a_h, ab_h, rcritv_h, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_h, Encpairs2_h, 0, N_h[0], P.NencMax, 1);

		}
	}
	if(ForceFlag > 0 || P.setElements > 1){
		comCall(1);
		if(P.setElements > 1){
			setElements_cpu(x4_h, v4_h, index_h, setElementsData_h, setElementsLine_h, Msun_h, dt_h, time_h, N_h[0], Nst, 1);
		}
		if(P.Usegas == 1) GasAccCall(time_h, dt_h, Kt[SIn - 1]);
		if(P.UseGR > 0 || P.UseTides > 0 || P.UseRotationalDeformation > 0 || P.UseJ2 > 0){
			force_cpu (x4_h, v4_h, index_h, spin_h, love_h, Msun_h, Spinsun_h, Lovesun_h, J2_h, vold_h, dt_h, Kt[SIn - 1], time_h, N_h[0], Nst, P.UseGR, P.UseTides, P.UseRotationalDeformation, 0, 1);
		}
		if(P.UseMigrationForce > 0){
			artificialMigration_cpu (x4_h, v4_h, index_h, migration_h, Msun_h, dt_h, Kt[SIn - 1], N_h[0], Nst, 0, 1);
			//artificialMigration2_cpu (x4_h, v4_h, index_h, dt_h, Kt[SIn - 1], N_h[0], Nst, 0, 1);
		}
		if(P.UseYarkovsky == 1) CallYarkovsky2_cpu (x4_h, v4_h, spin_h, index_h, Msun_h, dt_h, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky_averaged_cpu (x4_h, v4_h, spin_h, index_h, Msun_h, dt_h, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonEffect2_cpu (x4_h, v4_h, index_h, dt_h, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonEffect_averaged_cpu (x4_h, v4_h, index_h, Msun_h, dt_h, Kt[SIn - 1], N_h[0], Nst, 0);
		comCall(-1);
	}
	EjectionFlag2 = 0;

	for(int si = 0; si < SIn; ++si){
		HCCall(Ct[si], 1);
		fg_cpu (x4_h, v4_h, xold_h, vold_h, index_h, dt_h[0] * FGt[si], Msun_h[0].x, N_h[0], aelimits_h, aecount_h, Gridaecount_h, Gridaicount_h, si, P.UseGR);

		if(Nenc_m[0] > 0){
			for(int i = 0; i < def_GMax; ++i){
				Nenc_m[i] = 0;
			}
			Nencpairs2_h[0] = 0;		
		}
//printf("Nencpairs %d\n", Nencpairs_h[0]);
		if(Nencpairs_h[0] > 0){
			encounter_cpu (x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, dt_h[0] * FGt[si], Nencpairs_h[0], Nencpairs_h, Encpairs_h, Nencpairs2_h, Encpairs2_h, enccount_h, si, N_h[0], time_h[0], P.StopAtEncounter, Ncoll_m, P.MinMass);

			if(P.StopAtEncounter > 0 && Ncoll_m[0] > 0){
				Ncoll_m[0] = 0;
				StopAtEncounterFlag2 = 1;
			}
			
			if(P.SLevels > 1){
				if(Nencpairs2_h[0] > 0){
					double time = time_h[0];
					SEnc(time, 0, 1.0, si, noColl);
				}
			}
			else{
//printf("Nencpairs2 %d\n", Nencpairs2_h[0]);
				if(Nencpairs2_h[0] > 0){
					group_cpu (Nenc_m, Nencpairs2_h, Encpairs2_h, Encpairs_h, P.NencMax, N_h[0], N_h[0], P.SERIAL_GROUPING);
				}
				BSCall(si, time_h[0], noColl, 1.0);
			}
		}

		if(StopAtEncounterFlag2 == 1){
			StopAtEncounterFlag2 = 0;
			int enc = StopAtEncounterCall();
			if(enc == 0) return 0;
		}
		if(Ncoll_m[0] > 0){
			int col = CollisionCall(noColl);
			if(col == 0) return 0;
		}
		if(CollisionFlag == 1 && P.ei > 0 && timeStep % P.ei == 0){
			int rem = RemoveCall();
			if( rem == 0) return 0;
		}
		if(NWriteEnc_m[0] > 0){
			int enc = writeEncCall();
			if(enc == 0) return 0;
		}
		HCCall(Ct[si], -1);
		if(si < SIn - 1){
			if(P.KickFloat == 0){
				if(Nomp == 1){
					acc4E_cpu();
				}
				else{
					acc4D_cpu();
				}
			}
			else{
				if(Nomp == 1){
					acc4Ef_cpu();
				}
				else{
					acc4Df_cpu();
				}
			}
			if(P.SERIAL_GROUPING == 1){
				Sortb_cpu (Encpairs2_h, 0, N_h[0], P.NencMax);
			}
			kick32Ab_cpu (x4_h, v4_h, a_h, ab_h, rcritv_h, dt_h[0] * Kt[si] * def_ksq, Nencpairs_h, Encpairs2_h, 0, N_h[0], P.NencMax, 1);

			if(ForceFlag > 0){
				comCall(1);
				if(P.Usegas == 1) GasAccCall(time_h, dt_h, Kt[si]);
				if(P.UseGR > 0 || P.UseTides > 0 || P.UseRotationalDeformation > 0 || P.UseJ2 > 0){
					force_cpu (x4_h, v4_h, index_h, spin_h, love_h, Msun_h, Spinsun_h, Lovesun_h, J2_h, vold_h, dt_h, Kt[si], time_h, N_h[0], Nst, P.UseGR, P.UseTides, P.UseRotationalDeformation, 0, 1);
				}
				if(P.UseMigrationForce > 0){
					artificialMigration_cpu (x4_h, v4_h, index_h, migration_h, Msun_h, dt_h, Kt[si], N_h[0], Nst, 0, 1);
					//artificialMigration2_cpu (x4_h, v4_h, index_h, dt_h, Kt[si], N_h[0], Nst, 0, 1);
				}

				if(P.UseYarkovsky == 1) CallYarkovsky2_cpu (x4_h, v4_h, spin_h, index_h, Msun_h, dt_h, Kt[si], N_h[0], Nst, 0);
				if(P.UseYarkovsky == 2) CallYarkovsky_averaged_cpu (x4_h, v4_h, spin_h, index_h, Msun_h, dt_h, Kt[si], N_h[0], Nst, 0);
				if(P.UsePR == 1) PoyntingRobertsonEffect2_cpu (x4_h, v4_h, index_h, dt_h, Kt[si], N_h[0], Nst, 0);
				if(P.UsePR == 2) PoyntingRobertsonEffect_averaged_cpu (x4_h, v4_h, index_h, Msun_h, dt_h, Kt[si], N_h[0], Nst, 0);
				comCall(-1);
			}
		}
	}
	if(P.KickFloat == 0){
		if(Nomp == 1){
			acc4E_cpu();
		}
		else{
			acc4D_cpu();
		}
	}
	else{
		if(Nomp == 1){
			acc4Ef_cpu();
		}
		else{
			acc4Df_cpu();
		}
	}
	if(P.SERIAL_GROUPING == 1){
		Sortb_cpu (Encpairs2_h, 0, N_h[0], P.NencMax);
	}
	kick32Ab_cpu (x4_h, v4_h, a_h, ab_h, rcritv_h, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_h, Encpairs2_h, 0, N_h[0], P.NencMax, 1);

	if(ForceFlag > 0){
		comCall(1);
		if(P.Usegas == 1) GasAccCall(time_h, dt_h, Kt[SIn - 1]);
		if(P.UseGR > 0 || P.UseTides > 0 || P.UseRotationalDeformation > 0 || P.UseJ2 > 0){
			force_cpu (x4_h, v4_h, index_h, spin_h, love_h, Msun_h, Spinsun_h, Lovesun_h, J2_h, vold_h, dt_h, Kt[SIn - 1], time_h, N_h[0], Nst, P.UseGR, P.UseTides, P.UseRotationalDeformation, 0, 1);
		}
		if(P.UseMigrationForce > 0){
			artificialMigration_cpu (x4_h, v4_h, index_h, migration_h, Msun_h, dt_h, Kt[SIn - 1], N_h[0], Nst, 0, 1);
			//artificialMigration2_cpu (x4_h, v4_h, index_h, dt_h, Kt[SIn - 1], N_h[0], Nst, 0, 1);
		}
		if(P.UseYarkovsky == 1) CallYarkovsky2_cpu (x4_h, v4_h, spin_h, index_h, Msun_h, dt_h, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky_averaged_cpu (x4_h, v4_h, spin_h, index_h, Msun_h, dt_h, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonEffect2_cpu (x4_h, v4_h, index_h, dt_h, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonEffect_averaged_cpu (x4_h, v4_h, index_h, Msun_h, dt_h, Kt[SIn - 1], N_h[0], Nst, 0);
		comCall(-1);
	}
	if(EjectionFlag_m[0] > 0){
		int Ej = EjectionCall();
		if(Ej == 0) return 0;
	}

 #if def_poincareFlag == 1
	int per = PoincareSectionCall(time_h[0]);
	if(per == 0) return 0;
 #endif
	return 1;
}

// *************************************************
// Step small CPU
// *************************************************
int Data::step_small_cpu(int noColl){
	Rcrit_cpu (x4_h, v4_h, x4b_h, v4b_h, spin_h, spinb_h, 1.0 / (3.0 * Msun_h[0].x), rcrit_h, rcritb_h, rcritv_h, rcritvb_h, index_h, indexb_h, dt_h[0], n1_h[0], n2_h[0], time_h, time_h[0], EjectionFlag_m, N_h[0] + Nsmall_h[0], NconstT, P.SLevels, noColl);

	//use last time step information for setElements function, the beginning of the time step
	if(P.setElementsV == 2){ // convert barycentric velocities to heliocentric
		comCall(1);
	}
	if(P.setElementsV > 0){
		setElements_cpu(x4_h, v4_h, index_h, setElementsData_h, setElementsLine_h, Msun_h, dt_h, time_h, N_h[0], Nst, 0);
	}
	if(P.setElementsV == 2){
		comCall(-1);
	}

	if(P.SERIAL_GROUPING == 1){
		Sortb_cpu(Encpairs2_h, 0, N_h[0] + Nsmall_h[0], P.NencMax);
	}
	if(EjectionFlag2 == 0){
		kick32Ab_cpu(x4_h, v4_h, a_h, ab_h, rcritv_h, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_h, Encpairs2_h, 0, N_h[0] + Nsmall_h[0], P.NencMax, 1);
	}
	else{
//
		if(P.KickFloat == 0){
			if(Nomp == 1){
				acc4E_cpu();
				acc4Esmall_cpu();
			}
			else{
				acc4D_cpu();
				acc4Dsmall_cpu();
			}
		}
		else{
			if(Nomp == 1){
				acc4Ef_cpu();
				acc4Efsmall_cpu();
			}
			else{
				acc4Df_cpu();
				acc4Dfsmall_cpu();
			}
		}
		if(P.SERIAL_GROUPING == 1){
			Sortb_cpu (Encpairs2_h, 0, N_h[0] + Nsmall_h[0], P.NencMax);
		}
		kick32Ab_cpu (x4_h, v4_h, a_h, ab_h, rcritv_h, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_h, Encpairs2_h, 0, N_h[0] + Nsmall_h[0], P.NencMax, 1);

	}
//
	if(ForceFlag > 0 || P.setElements > 1){
		comCall(1);
		if(P.setElements > 1){
			setElements_cpu(x4_h, v4_h, index_h, setElementsData_h, setElementsLine_h, Msun_h, dt_h, time_h, N_h[0], Nst, 1);
		}
		if(P.Usegas == 1){
			GasAccCall(time_h, dt_h, Kt[SIn - 1]);
			GasAccCall_small(time_d, dt_d, Kt[SIn - 1]);
		}
		if(P.Usegas == 2){
			//GasAccCall(time_d, dt_d, Kt[SIn - 1]);
			GasAccCall2_small(time_d, dt_d, Kt[SIn - 1]);
		}
		if(P.UseGR > 0 || P.UseTides > 0 || P.UseRotationalDeformation > 0 || P.UseJ2 > 0){
			force_cpu (x4_h, v4_h, index_h, spin_h, love_h, Msun_h, Spinsun_h, Lovesun_h, J2_h, vold_h, dt_h, Kt[SIn - 1], time_h, N_h[0] + Nsmall_h[0], Nst, P.UseGR, P.UseTides, P.UseRotationalDeformation, 0, 1);
		}
		if(P.UseMigrationForce > 0){
			artificialMigration_cpu (x4_h, v4_h, index_h, migration_h, Msun_h, dt_h, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0, 1);
			//artificialMigration2_cpu (x4_h, v4_h, index_h, dt_h, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0, 1);
		}
		if(P.UseYarkovsky == 1) CallYarkovsky2_cpu (x4_h, v4_h, spin_h, index_h, Msun_h, dt_h, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky_averaged_cpu (x4_h, v4_h, spin_h, index_h, Msun_h, dt_h, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonEffect2_cpu (x4_h, v4_h, index_h, dt_h, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonEffect_averaged_cpu (x4_h, v4_h, index_h, Msun_h, dt_h, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0);
		comCall(-1);
	}
	EjectionFlag2 = 0;

	for(int si = 0; si < SIn; ++si){

		HCCall(Ct[si], 1);

		fg_cpu (x4_h, v4_h, xold_h, vold_h, index_h, dt_h[0] * FGt[si], Msun_h[0].x, N_h[0] + Nsmall_h[0], aelimits_h, aecount_h, Gridaecount_h, Gridaicount_h, si, P.UseGR);

		if(P.WriteEncounters == 2 && si == 0){
			Nencpairs2_h[0] = 0;		
			if(UseBVH == 1){
				BVHCall1();
			}
			if(UseBVH == 2){
				BVHCall2();
			}

			if(Nencpairs2_h[0] > 0){
				encounter_small_cpu (x4_h, v4_h, xold_h, vold_h, index_h, spin_h, dt_h[0] * FGt[si], Nencpairs2_h[0], Encpairs2_h, NWriteEnc_m, writeEnc_h, time_h[0]);
			}

			Nencpairs2_h[0] = 0;		
		}

		if(Nenc_m[0] > 0){
			for(int i = 0; i < def_GMax; ++i){
				Nenc_m[i] = 0;
			}
			Nencpairs2_h[0] = 0;		
		}
//printf("Nencpairs %d\n", Nencpairs_h[0]);
		if(Nencpairs_h[0] > 0){
			encounter_cpu (x4_h, v4_h, xold_h, vold_h, rcrit_h, rcritv_h, dt_h[0] * FGt[si], Nencpairs_h[0], Nencpairs_h, Encpairs_h, Nencpairs2_h, Encpairs2_h, enccount_h, si, N_h[0] + Nsmall_h[0], time_h[0], P.StopAtEncounter, Ncoll_m, P.MinMass);

			if(P.StopAtEncounter > 0 && Ncoll_m[0] > 0){
				Ncoll_m[0] = 0;
				StopAtEncounterFlag2 = 1;
			}
			
			if(P.SLevels > 1){
				if(Nencpairs2_h[0] > 0){
					double time = time_h[0];
					SEnc(time, 0, 1.0, si, noColl);
				}
			}
			else{
//printf("Nencpairs2 %d\n", Nencpairs2_h[0]);
				if(Nencpairs2_h[0] > 0){
					group_cpu (Nenc_m, Nencpairs2_h, Encpairs2_h, Encpairs_h, P.NencMax, N_h[0] + Nsmall_h[0], N_h[0], P.SERIAL_GROUPING);
				}
				BSCall(si, time_h[0], noColl, 1.0);
			}
		}

		if(StopAtEncounterFlag2 == 1){
			StopAtEncounterFlag2 = 0;
			int enc = StopAtEncounterCall();
			if(enc == 0) return 0;
		}
		if(Ncoll_m[0] > 0){
			int col = CollisionCall(noColl);
			if(col == 0) return 0;
		}
		if(P.UseSmallCollisions == 1 || P.UseSmallCollisions == 3){
			fragmentCall();
			if(nFragments_m[0] > 0){
				int er = printFragments(nFragments_m[0]);
				if(er == 0) return 0;
				er = RemoveCall();
				if(er == 0) return 0;
			}
		}
		if(P.UseSmallCollisions == 1 || P.UseSmallCollisions == 2){
			rotationCall();
			if(nFragments_m[0] > 0){
				int er = printRotation();
				if(er == 0) return 0;
			}
		}
		if(CollisionFlag == 1 && P.ei > 0 && timeStep % P.ei == 0){
			int rem = RemoveCall();
			if( rem == 0) return 0;
		}
		if(NWriteEnc_m[0] > 0){
			int enc = writeEncCall();
			if(enc == 0) return 0;
		}

		if(P.CreateParticles > 0){
			int er = createCall();
			if(er == 0) return 0;
			if(nFragments_m[0] > 0){
				printCreateparticle(nFragments_m[0]);
			}
		}

		HCCall(Ct[si], -1);
		if(si < SIn - 1){
			if(P.KickFloat == 0){
				if(Nomp == 1){
					acc4E_cpu();
					acc4Esmall_cpu();
				}
				else{
					acc4D_cpu();
					acc4Dsmall_cpu();
				}
			}
			else{
				if(Nomp == 1){
					acc4Ef_cpu();
					acc4Efsmall_cpu();
				}
				else{
					acc4Df_cpu();
					acc4Dfsmall_cpu();
				}
			}
			if(P.SERIAL_GROUPING == 1){
				Sortb_cpu(Encpairs2_h, 0, N_h[0] + Nsmall_h[0], P.NencMax);
			}
			kick32Ab_cpu(x4_h, v4_h, a_h, ab_h, rcritv_h, dt_h[0] * Kt[si] * def_ksq, Nencpairs_h, Encpairs2_h, 0, N_h[0] + Nsmall_h[0], P.NencMax, 1);

			if(ForceFlag > 0){
				comCall(1);
				if(P.Usegas == 1){
					GasAccCall(time_h, dt_h, Kt[si]);
					GasAccCall_small(time_h, dt_h, Kt[si]);
				}
				if(P.Usegas == 2){
					//GasAccCall(time_d, dt_d, Kt[SIn - 1]);
					GasAccCall2_small(time_d, dt_d, Kt[si]);
				}
				if(P.UseGR > 0 || P.UseTides > 0 || P.UseRotationalDeformation > 0 || P.UseJ2 > 0){
					force_cpu (x4_h, v4_h, index_h, spin_h, love_h, Msun_h, Spinsun_h, Lovesun_h, J2_h, vold_h, dt_h, Kt[si], time_h, N_h[0] + Nsmall_h[0], Nst, P.UseGR, P.UseTides, P.UseRotationalDeformation, 0, 1);
				}
				if(P.UseMigrationForce > 0){
					artificialMigration_cpu (x4_h, v4_h, index_h, migration_h, Msun_h, dt_h, Kt[si], N_h[0] + Nsmall_h[0], Nst, 0, 1);
					//artificialMigration2_cpu (x4_h, v4_h, index_h, dt_h, Kt[si], N_h[0] + Nsmall_h[0], Nst, 0, 1);
				}

				if(P.UseYarkovsky == 1) CallYarkovsky2_cpu (x4_h, v4_h, spin_h, index_h, Msun_h, dt_h, Kt[si], N_h[0] + Nsmall_h[0], Nst, 0);
				if(P.UseYarkovsky == 2) CallYarkovsky_averaged_cpu (x4_h, v4_h, spin_h, index_h, Msun_h, dt_h, Kt[si], N_h[0] + Nsmall_h[0], Nst, 0);
				if(P.UsePR == 1) PoyntingRobertsonEffect2_cpu (x4_h, v4_h, index_h, dt_h, Kt[si], N_h[0] + Nsmall_h[0], Nst, 0);
				if(P.UsePR == 2) PoyntingRobertsonEffect_averaged_cpu (x4_h, v4_h, index_h, Msun_h, dt_h, Kt[si], N_h[0] + Nsmall_h[0], Nst, 0);
				comCall(-1);
			}
		}
	}
	if(P.KickFloat == 0){
		if(Nomp == 1){
			acc4E_cpu();
			acc4Esmall_cpu();
		}
		else{
			acc4D_cpu();
			acc4Dsmall_cpu();
		}
	}
	else{
		if(Nomp == 1){
			acc4Ef_cpu();
			acc4Efsmall_cpu();
		}
		else{
			acc4Df_cpu();
			acc4Dfsmall_cpu();
		}
	}

	if(P.SERIAL_GROUPING == 1){
		Sortb_cpu (Encpairs2_h, 0, N_h[0] + Nsmall_h[0], P.NencMax);
	}
	kick32Ab_cpu (x4_h, v4_h, a_h, ab_h, rcritv_h, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_h, Encpairs2_h, 0, N_h[0] + Nsmall_h[0], P.NencMax, 1);

	
	if(ForceFlag > 0){
		comCall(1);
		if(P.Usegas == 1){
			GasAccCall(time_h, dt_h, Kt[SIn - 1]);
			GasAccCall_small(time_h, dt_h, Kt[SIn - 1]);
		}
		if(P.Usegas == 2){
			//GasAccCall(time_d, dt_d, Kt[SIn - 1]);
			GasAccCall2_small(time_d, dt_d, Kt[SIn - 1]);
		}
		if(P.UseGR > 0 || P.UseTides > 0 || P.UseRotationalDeformation > 0 || P.UseJ2 > 0){
			force_cpu (x4_h, v4_h, index_h, spin_h, love_h, Msun_h, Spinsun_h, Lovesun_h, J2_h, vold_h, dt_h, Kt[SIn - 1], time_h, N_h[0] + Nsmall_h[0], Nst, P.UseGR, P.UseTides, P.UseRotationalDeformation, 0, 1);
		}
		if(P.UseMigrationForce > 0){
			artificialMigration_cpu (x4_h, v4_h, index_h, migration_h, Msun_h, dt_h, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0, 1);
			//artificialMigration2_cpu (x4_h, v4_h, index_h, dt_h, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0, 1);
		}
		if(P.UseYarkovsky == 1) CallYarkovsky2_cpu (x4_h, v4_h, spin_h, index_h, Msun_h, dt_h, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky_averaged_cpu (x4_h, v4_h, spin_h, index_h, Msun_h, dt_h, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonEffect2_cpu (x4_h, v4_h, index_h, dt_h, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonEffect_averaged_cpu (x4_h, v4_h, index_h, Msun_h, dt_h, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0);
		comCall(-1);
	}
	if(EjectionFlag_m[0] > 0){
		int Ej = EjectionCall();
		if(Ej == 0) return 0;
	}

 #if def_poincareFlag == 1
	int per = PoincareSectionCall(time_h[0]);
	if(per == 0) return 0;
 #endif
	return 1;
}
#endif

__host__ int Data::step_1kernel(int noColl){
	//no Set Elements, no sort, noPoincare, no floatkick, no force
	// moStopatEncounter, no writeEnc
	Rcrit_kernel <<< (N_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, x4b_d, v4b_d, spin_d, spinb_d, 1.0 / (3.0 * Msun_h[0].x), rcrit_d, rcritb_d, rcritv_d, rcritvb_d, index_d, indexb_d, dt_h[0], n1_h[0], n2_h[0], time_d, time_h[0], EjectionFlag_d, N_h[0], NconstT, P.SLevels, noColl);
	//Rcritb_kernel <<< (N_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, x4b_d, v4b_d, spin_d, spinb_d, 1.0 / (3.0 *Msun_h[0].x), rcrit_d, rcritb_d, rcritv_d, rcritvb_d, index_d, indexb_d, dt_h[0], n1_h[0], n2_h[0], time_d, time_h[0], EjectionFlag_d, N_h[0], noColl);

	if(EjectionFlag2 == 0){
		kick32Ab_kernel <<< (N_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs2_d, 0, N_h[0], P.NencMax, 1);
	}
	else{
		kick16c_kernel <<< N_h[0], WarpSize >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, EncFlag_d, P.NencMax, N_h[0], 1);
		cudaEventRecord(KickEvent, 0);
		cudaStreamWaitEvent(copyStream, KickEvent, 0);
		cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);
	}
	EjectionFlag2 = 0;

	for(int si = 0; si < SIn; ++si){
		HCCall(Ct[si], 1);
		//HCfg_kernel <<< (N_h[0] + FTX - 1) / FTX, FTX >>> (x4_d, v4_d, xold_d, vold_d, index_d, dt_h[0] * FGt[si], dt_h[0] * Ct[si], dt_h[0] / Msun_h[0].x * Ct[si], Msun_h[0].x, N_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, si, P.UseGR);
		fg_kernel <<< (N_h[0] + FTX - 1) / FTX, FTX >>> (x4_d, v4_d, xold_d, vold_d, index_d, dt_h[0] * FGt[si], Msun_h[0].x, N_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, si, P.UseGR);
		cudaStreamSynchronize(copyStream);

		if(Nenc_m[0] > 0){
			for(int i = 0; i < def_GMax; ++i){
				Nenc_m[i] = 0;
			}			
			setNencpairs_kernel <<< 1, 1 >>> (Nencpairs2_d, 1);
		}
//printf("Nencpairs %d\n", Nencpairs_h[0]);
		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 63)/ 64, 64 >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, dt_h[0] * FGt[si], Nencpairs_h[0], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, enccount_d, si, NB[0], time_h[0], P.StopAtEncounter, Ncoll_d, P.MinMass);
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);

			if(P.StopAtEncounter > 0 && Ncoll_m[0] > 0){
				Ncoll_m[0] = 0;
				StopAtEncounterFlag2 = 1;
			}
			
			if(P.SLevels > 1){
				if(Nencpairs2_h[0] > 0){
					double time = time_h[0];
					SEnc(time, 0, 1.0, si, noColl);
				}
			}
			else{
//printf("Nencpairs2 %d\n", Nencpairs2_h[0]);
				if(Nencpairs2_h[0] > 0){
					if(NB[0] < 32){
						group_kernel < 16, 512 > <<< 1, 512 >>> (Nenc_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0], N_h[0], P.SERIAL_GROUPING);
					}
					else{
						group_kernel < 32, 512 > <<< 1, 512 >>> (Nenc_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0], N_h[0], P.SERIAL_GROUPING);
					}
				}
				cudaDeviceSynchronize();
				BSCall(si, time_h[0], noColl, 1.0);

			}
		}
		if(Ncoll_m[0] > 0){
			int col = CollisionCall(noColl);
			if(col == 0) return 0;
		}
		if(CollisionFlag == 1 && P.ei > 0 && timeStep % P.ei == 0){
			int rem = RemoveCall();
			if( rem == 0) return 0;
		}
		HCCall(Ct[si], -1);
		if(si < SIn - 1){
			kick16c_kernel <<< N_h[0], WarpSize >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[si] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, EncFlag_d, P.NencMax, N_h[0], 2);
			cudaEventRecord(KickEvent, 0);
			cudaStreamWaitEvent(copyStream, KickEvent, 0);
			cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);
	
		}
	}
	kick16c_kernel <<< N_h[0], WarpSize >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, EncFlag_d, P.NencMax, N_h[0], 1);
	cudaEventRecord(KickEvent, 0);
	cudaStreamWaitEvent(copyStream, KickEvent, 0);
	cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);
	
	if(EjectionFlag_m[0] > 0){
		int Ej = EjectionCall();
		if(Ej == 0) return 0;
	}

	return 1;
}

__host__ int Data::step_16(int noColl){
	Rcrit_kernel <<< (N_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, x4b_d, v4b_d, spin_d, spinb_d, 1.0 / (3.0 * Msun_h[0].x), rcrit_d, rcritb_d, rcritv_d, rcritvb_d, index_d, indexb_d, dt_h[0], n1_h[0], n2_h[0], time_d, time_h[0], EjectionFlag_d, N_h[0], NconstT, P.SLevels, noColl);
	//Rcritb_kernel <<< (N_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, x4b_d, v4b_d, spin_d, spinb_d, 1.0 / (3.0 * Msun_h[0].x), rcrit_d, rcritb_d, rcritv_d, rcritvb_d, index_d, indexb_d, dt_h[0], n1_h[0], n2_h[0], time_d, time_h[0], EjectionFlag_d, N_h[0], noColl);

	//use last time step information for setElements function, the beginning of the time step
	if(P.setElementsV == 2){ // convert barycentric velocities to heliocentric
		comCall(1);
	}
	if(P.setElementsV > 0){
		setElements_kernel <<< (P.setElementsN + 63) / 64, 64 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst, 0);
	}
	if(P.setElementsV == 2){
		comCall(-1);
	}

	if(P.SERIAL_GROUPING == 1){
		Sortb_kernel <<< (N_h[0] + RTX - 1) / RTX, RTX >>> (Encpairs2_d, 0, N_h[0], P.NencMax);
	}
	if(doTransits == 0){
		if(EjectionFlag2 == 0){
			kick32Ab_kernel <<< (N_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs2_d, 0, N_h[0], P.NencMax, 1);
		}
		else{
			if(P.KickFloat == 0){
				kick16c_kernel <<< N_h[0], WarpSize >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, EncFlag_d, P.NencMax, N_h[0], 1);
			}
			else{
				kick16cf_kernel <<< N_h[0], WarpSize >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, EncFlag_d, P.NencMax, N_h[0], 1);
			}
			cudaEventRecord(KickEvent, 0);
			cudaStreamWaitEvent(copyStream, KickEvent, 0);
			cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);
		}
	}
#if def_TTV == 1
	if(doTransits == 1){
		if(EjectionFlag2 == 0){
			kick32ATTV_kernel <<<1, WarpSize >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs2_d, N_h[0], P.NencMax, time_h[0], dt_h[0], Msun_h[0].x, Msun_h[0].y, Ntransit_d, Transit_d);
		}
		else{
			if(P.KickFloat == 0){
				kick16c_kernel <<< N_h[0], WarpSize >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, EncFlag_d, P.NencMax, N_h[0], 1);
			}
			else{
				kick16cf_kernel <<< N_h[0], WarpSize >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, EncFlag_d, P.NencMax, N_h[0], 1);
			}
			cudaEventRecord(KickEvent, 0);
			cudaStreamWaitEvent(copyStream, KickEvent, 0);
			cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);
			kick32ATTV_kernel <<<1, WarpSize >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs2_d, N_h[0], P.NencMax, time_h[0], dt_h[0], Msun_h[0].x, Msun_h[0].y, Ntransit_d, Transit_d);
		}
		cudaDeviceSynchronize();
		if(Ntransit_m[0] > 0){
			if(Ntransit_m[0] >= def_NtransitMax - 1){
				printf("more Transits than allowed in def_NtransitMax: %d\n", def_NtransitMax);
				return 0;
			}
			BSTTVStep_kernel < 8, 8 > <<< Ntransit_m[0], 64 >>> (x4_d, v4b_d, Transit_d, N_d, dt_d, Msun_d, index_d, time_d, NBS_d, P.UseGR, P.MinMass, P.UseTestParticles, Nst, TransitTime_d, NtransitsT_d);
			Ntransit_m[0] = 0;
		}
	}
#endif
	if(ForceFlag > 0 || P.setElements > 1){
		comCall(1);
		if(P.setElements > 1){
			setElements_kernel <<< (P.setElementsN + 63) / 64, 64 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst, 1);
		}
		if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[SIn - 1]);
		if(P.UseGR > 0 || P.UseTides > 0 || P.UseRotationalDeformation > 0 || P.UseJ2 > 0){
			int nn = (N_h[0] + Nsmall_h[0] + FrTX - 1) / FrTX;
			int ncb = min(nn, 1024);
			force_kernel <<< nn, FrTX, WarpSize * sizeof(double3) >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, Lovesun_d, J2_d, vold_d, dt_d, Kt[SIn - 1], time_d, N_h[0], Nst, P.UseGR, P.UseTides, P.UseRotationalDeformation, 0, 1);
			if(P.UseTides > 0 || P.UseRotationalDeformation > 0){
				if(N_h[0] + Nsmall_h[0] > FrTX){
					forced2_kernel <<< 1, ((ncb + WarpSize - 1) / WarpSize) * WarpSize, WarpSize * sizeof(double3) >>> (vold_d, Spinsun_d, nn, 1);
				}
			}
		}
		if(P.UseMigrationForce > 0){
			artificialMigration_kernel <<< (N_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, index_d, migration_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0, 1);
			//artificialMigration2_kernel <<< (N_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0, 1);
		}
		if(P.UseYarkovsky == 1) CallYarkovsky2_kernel <<< (N_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky_averaged_kernel <<< (N_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonEffect2_kernel <<< (N_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonEffect_averaged_kernel <<< (N_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		comCall(-1);
	}
	EjectionFlag2 = 0;

	for(int si = 0; si < SIn; ++si){
		HCCall(Ct[si], 1);
		//HCfg_kernel <<< (N_h[0] + FTX - 1) / FTX, FTX >>> (x4_d, v4_d, xold_d, vold_d, index_d, dt_h[0] * FGt[si], dt_h[0] * Ct[si], dt_h[0] / Msun_h[0].x * Ct[si], Msun_h[0].x, N_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, si, P.UseGR);
		fg_kernel <<< (N_h[0] + FTX - 1) / FTX, FTX >>> (x4_d, v4_d, xold_d, vold_d, index_d, dt_h[0] * FGt[si], Msun_h[0].x, N_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, si, P.UseGR);
		cudaStreamSynchronize(copyStream);

		if(Nenc_m[0] > 0){
			for(int i = 0; i < def_GMax; ++i){
				Nenc_m[i] = 0;
			}			
			setNencpairs_kernel <<< 1, 1 >>> (Nencpairs2_d, 1);
		}
//printf("Nencpairs %d\n", Nencpairs_h[0]);
		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 63)/ 64, 64 >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, dt_h[0] * FGt[si], Nencpairs_h[0], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, enccount_d, si, NB[0], time_h[0], P.StopAtEncounter, Ncoll_d, P.MinMass);
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);

			if(P.StopAtEncounter > 0 && Ncoll_m[0] > 0){
				Ncoll_m[0] = 0;
				StopAtEncounterFlag2 = 1;
			}
			
			if(P.SLevels > 1){
				if(Nencpairs2_h[0] > 0){
					double time = time_h[0];
					SEnc(time, 0, 1.0, si, noColl);
				}
			}
			else{
//printf("Nencpairs2 %d\n", Nencpairs2_h[0]);
				if(Nencpairs2_h[0] > 0){
					if(NB[0] < 32){
						group_kernel < 16, 512 > <<< 1, 512 >>> (Nenc_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0], N_h[0], P.SERIAL_GROUPING);
					}
					else if(NB[0] < 64){
						group_kernel < 32, 512 > <<< 1, 512 >>> (Nenc_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0], N_h[0], P.SERIAL_GROUPING);
					}
					else{
						group_kernel < 64, 512 > <<< 1, 512 >>> (Nenc_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0], N_h[0], P.SERIAL_GROUPING);
					}

				}
				cudaDeviceSynchronize();
				BSCall(si, time_h[0], noColl, 1.0);

			}
		}
		if(StopAtEncounterFlag2 == 1){
			StopAtEncounterFlag2 = 0;
			int enc = StopAtEncounterCall();
			if(enc == 0) return 0;
		}
		if(Ncoll_m[0] > 0){
			int col = CollisionCall(noColl);
			if(col == 0) return 0;
		}
		if(CollisionFlag == 1 && P.ei > 0 && timeStep % P.ei == 0){
			int rem = RemoveCall();
			if( rem == 0) return 0;
		}
		if(NWriteEnc_m[0] > 0){
			int enc = writeEncCall();
			if(enc == 0) return 0;
		}
		HCCall(Ct[si], -1);
		if(si < SIn - 1){
			if(P.KickFloat == 0){
				kick16c_kernel <<< N_h[0], WarpSize >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[si] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, EncFlag_d, P.NencMax, N_h[0], 2);
			}
			else{
				kick16cf_kernel <<< N_h[0], WarpSize >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[si] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, EncFlag_d, P.NencMax, N_h[0], 2);
			}
			cudaEventRecord(KickEvent, 0);
			cudaStreamWaitEvent(copyStream, KickEvent, 0);
			cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);
	
			if(ForceFlag > 0){
				comCall(1);
				if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[si]);
				if(P.UseGR > 0 || P.UseTides > 0 || P.UseRotationalDeformation > 0 || P.UseJ2 > 0){
					int nn = (N_h[0] + Nsmall_h[0] + FrTX - 1) / FrTX;
					int ncb = min(nn, 1024);
					force_kernel <<< nn, FrTX, WarpSize * sizeof(double3) >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, Lovesun_d, J2_d, vold_d, dt_d, Kt[si], time_d, N_h[0], Nst, P.UseGR, P.UseTides, P.UseRotationalDeformation, 0, 1);
					if(P.UseTides > 0 || P.UseRotationalDeformation > 0){
						if(N_h[0] + Nsmall_h[0] > FrTX){
							forced2_kernel <<< 1, ((ncb + WarpSize - 1) / WarpSize) * WarpSize, WarpSize * sizeof(double3) >>> (vold_d, Spinsun_d, nn, 1);
						}
					}
				}
				if(P.UseMigrationForce > 0){
					artificialMigration_kernel <<< (N_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, index_d, migration_d, Msun_d, dt_d, Kt[si], N_h[0], Nst, 0, 1);
					//artificialMigration2_kernel <<< (N_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, index_d, dt_d, Kt[si], N_h[0], Nst, 0, 1);
				}

				if(P.UseYarkovsky == 1) CallYarkovsky2_kernel <<< (N_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], N_h[0], Nst, 0);
				if(P.UseYarkovsky == 2) CallYarkovsky_averaged_kernel <<< (N_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], N_h[0], Nst, 0);
				if(P.UsePR == 1) PoyntingRobertsonEffect2_kernel <<< (N_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, index_d, dt_d, Kt[si], N_h[0], Nst, 0);
				if(P.UsePR == 2) PoyntingRobertsonEffect_averaged_kernel <<< (N_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[si], N_h[0], Nst, 0);
				comCall(-1);
			}
		}
	}
	if(P.KickFloat == 0){
		kick16c_kernel <<< N_h[0], WarpSize >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, EncFlag_d, P.NencMax, N_h[0], 1);
	}
	else{
		kick16cf_kernel <<< N_h[0], WarpSize >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, EncFlag_d, P.NencMax, N_h[0], 1);
	}
	cudaEventRecord(KickEvent, 0);
	cudaStreamWaitEvent(copyStream, KickEvent, 0);
	cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);
	
	if(ForceFlag > 0){
		comCall(1);
		if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[SIn - 1]);
		if(P.UseGR > 0 || P.UseTides > 0 || P.UseRotationalDeformation > 0 || P.UseJ2 > 0){
			int nn = (N_h[0] + Nsmall_h[0] + FrTX - 1) / FrTX;
			int ncb = min(nn, 1024);
			force_kernel <<< nn, FrTX, WarpSize * sizeof(double3) >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, Lovesun_d, J2_d, vold_d, dt_d, Kt[SIn - 1], time_d, N_h[0], Nst, P.UseGR, P.UseTides, P.UseRotationalDeformation, 0, 1);
			if(P.UseTides > 0 || P.UseRotationalDeformation > 0){
				if(N_h[0] + Nsmall_h[0] > FrTX){
					forced2_kernel <<< 1, ((ncb + WarpSize - 1) / WarpSize) * WarpSize, WarpSize * sizeof(double3) >>> (vold_d, Spinsun_d, nn, 1);
				}
			}
		}
		if(P.UseMigrationForce > 0){
			artificialMigration_kernel <<< (N_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, index_d, migration_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0, 1);
			//artificialMigration2_kernel <<< (N_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0, 1);
		}
		if(P.UseYarkovsky == 1) CallYarkovsky2_kernel <<< (N_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky_averaged_kernel <<< (N_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonEffect2_kernel <<< (N_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonEffect_averaged_kernel <<< (N_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		comCall(-1);
	}
	if(EjectionFlag_m[0] > 0){
		int Ej = EjectionCall();
		if(Ej == 0) return 0;
	}

#if def_poincareFlag == 1
	int per = PoincareSectionCall(time_h[0]);
	if(per == 0) return 0;
#endif
	return 1;
}

// *************************************************
// Step large N
// *************************************************
__host__ int Data::step_largeN(int noColl){

	Rcrit_kernel <<< (N_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, x4b_d, v4b_d, spin_d, spinb_d, 1.0 / (3.0 * Msun_h[0].x), rcrit_d, rcritb_d, rcritv_d, rcritvb_d, index_d, indexb_d, dt_h[0], n1_h[0], n2_h[0], time_d, time_h[0], EjectionFlag_d, N_h[0], NconstT, P.SLevels, noColl);
	//use last time step information for setElements function, the beginning of the time step
	if(P.setElementsV == 2){ // convert barycentric velocities to heliocentric
		comCall(1);
	}
	if(P.setElementsV > 0){
		setElements_kernel <<< (P.setElementsN + 63) / 64, 64 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst, 0);
	}
	if(P.setElementsV == 2){
		comCall(-1);
	}
//This is not needed, remove after checking
//	if(P.SERIAL_GROUPING == 1){
//		Sortb_kernel <<< (N_h[0] + RTX - 1) / RTX, RTX >>> (Encpairs2_d, 0, N_h[0], P.NencMax);
//	}
	if(EjectionFlag2 == 0){
		kick32Ab_kernel <<< (N_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs2_d, 0, N_h[0], P.NencMax, 1);
	}
	else{
		if(UseAcc == 0){
			if(P.KickFloat == 0){
				kick32c_kernel <<< N_h[0] , min(NB[0], 1024), 2 * WarpSize * sizeof(double3) >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, EncFlag_d, P.NencMax, N_h[0], 1);
			}
			else{
				kick32cf_kernel <<< N_h[0] , min(NB[0], 1024), 2 * WarpSize * sizeof(float3) >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, EncFlag_d, P.NencMax, N_h[0], 1);
			}
			cudaEventRecord(KickEvent, 0);
			cudaStreamWaitEvent(copyStream, KickEvent, 0);
			cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);
		}
		else{
			if(P.ndev == 1){
				if(P.KickFloat == 0){
					acc4C_kernel <<< dim3( (((N_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0], 0, N_h[0], P.NencMax, KP, 0);
				}
				else{
					acc4Cf_kernel <<< dim3( (((N_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0], 0, N_h[0], P.NencMax, KP, 0);
				}
				cudaEventRecord(KickEvent, 0);
				cudaStreamWaitEvent(copyStream, KickEvent, 0);
				cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);

				if(P.SERIAL_GROUPING == 1){
					Sortb_kernel <<< (N_h[0] + RTX - 1) / RTX, RTX >>> (Encpairs2_d, 0, N_h[0], P.NencMax);
				}
				kick32Ab_kernel <<< (N_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs2_d, 0, N_h[0], P.NencMax, 1);
			}
			else{
				KickndevCall(SIn - 1, 0);
			}
		}
	}
	if(ForceFlag > 0 || P.setElements > 1){
		comCall(1);
		if(P.setElements > 1){
			setElements_kernel <<< (P.setElementsN + 63) / 64, 64 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst, 1);
		}
		if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[SIn - 1]);
		if(P.UseGR > 0 || P.UseTides > 0 || P.UseRotationalDeformation > 0 || P.UseJ2 > 0){
			int nn = (N_h[0] + Nsmall_h[0] + FrTX - 1) / FrTX;
			int ncb = min(nn, 1024);
			force_kernel <<< nn, FrTX, WarpSize * sizeof(double3) >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, Lovesun_d, J2_d, vold_d, dt_d, Kt[SIn - 1], time_d, N_h[0], Nst, P.UseGR, P.UseTides, P.UseRotationalDeformation, 0, 1);
			if(P.UseTides > 0 || P.UseRotationalDeformation > 0){
				if(N_h[0] + Nsmall_h[0] > FrTX){
					forced2_kernel <<< 1, ((ncb + WarpSize - 1) / WarpSize) * WarpSize, WarpSize * sizeof(double3) >>> (vold_d, Spinsun_d, nn, 1);
				}
			}
		}
		if(P.UseMigrationForce > 0){
			artificialMigration_kernel <<< (N_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, index_d, migration_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0, 1);
		}
		if(P.UseYarkovsky == 1) CallYarkovsky2_kernel <<< (N_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky_averaged_kernel <<< (N_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonEffect2_kernel <<< (N_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonEffect_averaged_kernel <<< (N_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		comCall(-1);
	}
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		HCCall(Ct[si], 1);
		fg_kernel <<< (N_h[0] + FTX - 1) / FTX, FTX >>> (x4_d, v4_d, xold_d, vold_d, index_d, dt_h[0] * FGt[si], Msun_h[0].x, N_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, si, P.UseGR);
		cudaStreamSynchronize(copyStream);

		if(Nenc_m[0] > 0){
			for(int i = 0; i < def_GMax; ++i){
				Nenc_m[i] = 0;
			}			
			setNencpairs_kernel <<< 1, 1 >>> (Nencpairs2_d, 1);
		}
		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 63)/ 64, 64 >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, dt_h[0] * FGt[si], Nencpairs_h[0], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, enccount_d, si, NB[0], time_h[0], P.StopAtEncounter, Ncoll_d, P.MinMass);
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			if(P.StopAtEncounter > 0 && Ncoll_m[0] > 0){
				Ncoll_m[0] = 0;
				StopAtEncounterFlag2 = 1;
			}
//tuneBS2();

			if(P.SLevels > 1){
				if(Nencpairs2_h[0] > 0){
					double time = time_h[0];
					SEnc(time, 0, 1.0, si, noColl);
				}
			}
			else{
				if(Nencpairs2_h[0] > 0){
					if(NB[0] < 128){
						group_kernel < 64, 512 > <<< 1, 512 >>> (Nenc_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0], N_h[0], P.SERIAL_GROUPING);
					}
					else if(NB[0] < 256){
						group_kernel < 128, 512 > <<< 1, 512 >>> (Nenc_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0], N_h[0], P.SERIAL_GROUPING);
					}
					else if(NB[0] < 512){
						group_kernel < 256, 512 > <<< 1, 512 >>> (Nenc_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0], N_h[0], P.SERIAL_GROUPING);
					}
					else if(NB[0] < 1024){
						group_kernel < 512, 512 > <<< 1, 512 >>> (Nenc_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0], N_h[0], P.SERIAL_GROUPING);
					}
					else{
						group_kernel < 1, 512 > <<< 1, 512 >>> (Nenc_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0], N_h[0], P.SERIAL_GROUPING);
					}
				}
				cudaDeviceSynchronize();
				BSCall(si, time_h[0], noColl, 1.0);
			}
		}
		if(StopAtEncounterFlag2 == 1){
			StopAtEncounterFlag2 = 0;
			int enc = StopAtEncounterCall();
			if(enc == 0) return 0;
		}
		if(Ncoll_m[0] > 0){
			int col = CollisionCall(noColl);
			if(col == 0) return 0;
		}
		if(CollisionFlag == 1 && P.ei > 0 && timeStep % P.ei == 0){
			int rem = RemoveCall();
			if( rem == 0) return 0;
		}
		if(NWriteEnc_m[0] > 0){
			int enc = writeEncCall();
			if(enc == 0) return 0;
		}

		HCCall(Ct[si], -1);
		
		if(si < SIn - 1){
			//kick
			if(UseAcc == 0){
				if(P.KickFloat == 0){
					kick32c_kernel <<< N_h[0] , min(NB[0], 1024), 2 * WarpSize * sizeof(double3) >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[si] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, EncFlag_d, P.NencMax, N_h[0], 2);
				}
				else{
					kick32cf_kernel <<< N_h[0] , min(NB[0], 1024), 2 * WarpSize * sizeof(float3) >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[si] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, EncFlag_d, P.NencMax, N_h[0], 2);
				}
				cudaEventRecord(KickEvent, 0);
				cudaStreamWaitEvent(copyStream, KickEvent, 0);
				cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);
			}
			else{
				if(P.ndev == 1){
					if(P.KickFloat == 0){
						acc4C_kernel <<< dim3( (((N_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0], 0, N_h[0], P.NencMax, KP, 0);
					}
					else{
						acc4Cf_kernel <<< dim3( (((N_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0], 0, N_h[0], P.NencMax, KP, 0);
					}
					cudaEventRecord(KickEvent, 0);
					cudaStreamWaitEvent(copyStream, KickEvent, 0);
					cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);

					if(P.SERIAL_GROUPING == 1){
						Sortb_kernel <<< (N_h[0] + RTX - 1) / RTX, RTX >>> (Encpairs2_d, 0, N_h[0], P.NencMax);
					}
					kick32Ab_kernel <<< (N_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[si] * def_ksq, Nencpairs_d, Encpairs2_d, 0, N_h[0], P.NencMax, 1);
				}
				else{
					KickndevCall(si, 0);
				}
			}
			if(ForceFlag > 0){
				comCall(1);
				if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[si]);
				if(P.UseGR > 0 || P.UseTides > 0 || P.UseRotationalDeformation > 0 || P.UseJ2 > 0){
					int nn = (N_h[0] + Nsmall_h[0] + FrTX - 1) / FrTX;
					int ncb = min(nn, 1024);
					force_kernel <<< nn, FrTX, WarpSize * sizeof(double3) >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, Lovesun_d, J2_d, vold_d, dt_d, Kt[si], time_d, N_h[0], Nst, P.UseGR, P.UseTides, P.UseRotationalDeformation, 0, 1);
					if(P.UseTides > 0 || P.UseRotationalDeformation > 0){
						if(N_h[0] + Nsmall_h[0] > FrTX){
							forced2_kernel <<< 1, ((ncb + WarpSize - 1) / WarpSize) * WarpSize, WarpSize * sizeof(double3) >>> (vold_d, Spinsun_d, nn, 1);
						}
					}
				}
				if(P.UseMigrationForce > 0){
					artificialMigration_kernel <<< (N_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, index_d, migration_d, Msun_d, dt_d, Kt[si], N_h[0], Nst, 0, 1);
				}
				if(P.UseYarkovsky == 1) CallYarkovsky2_kernel <<< (N_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], N_h[0], Nst, 0);
				if(P.UseYarkovsky == 2) CallYarkovsky_averaged_kernel <<< (N_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], N_h[0], Nst, 0);
				if(P.UsePR == 1) PoyntingRobertsonEffect2_kernel <<< (N_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, index_d, dt_d, Kt[si], N_h[0], Nst, 0);
				if(P.UsePR == 2) PoyntingRobertsonEffect_averaged_kernel <<< (N_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[si], N_h[0], Nst, 0);
				comCall(-1);
			}
		}
	}
	//kick
	if(UseAcc == 0){
		if(P.KickFloat == 0){
			kick32c_kernel <<< N_h[0] , min(NB[0], 1024), 2 * WarpSize * sizeof(double3) >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, EncFlag_d, P.NencMax, N_h[0], 1);
		}
		else{
			kick32cf_kernel <<< N_h[0] , min(NB[0], 1024), 2 * WarpSize * sizeof(float3) >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, EncFlag_d, P.NencMax, N_h[0], 1);
		}
		cudaEventRecord(KickEvent, 0);
		cudaStreamWaitEvent(copyStream, KickEvent, 0);
		cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);
	}
	else{
		if(P.ndev == 1){
			if(P.KickFloat == 0){
				acc4C_kernel <<< dim3( (((N_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> (x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0], 0, N_h[0], P.NencMax, KP, 0);
			}
			else{
				acc4Cf_kernel <<< dim3( (((N_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> (x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0], 0, N_h[0], P.NencMax, KP, 0);
			}

			cudaEventRecord(KickEvent, 0);
			cudaStreamWaitEvent(copyStream, KickEvent, 0);
			cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);

			if(P.SERIAL_GROUPING == 1){
				Sortb_kernel <<< (N_h[0] + RTX - 1) / RTX, RTX >>> (Encpairs2_d, 0, N_h[0], P.NencMax);
			}
			kick32Ab_kernel <<< (N_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs2_d, 0, N_h[0], P.NencMax, 1);
		}
		else{		
			KickndevCall(SIn - 1, 0);
		}

	}
	if(ForceFlag > 0){
		comCall(1);
		if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[SIn - 1]);
		if(P.UseGR > 0 || P.UseTides > 0 || P.UseRotationalDeformation > 0 || P.UseJ2 > 0){
			int nn = (N_h[0] + Nsmall_h[0] + FrTX - 1) / FrTX;
			int ncb = min(nn, 1024);
			force_kernel <<< nn, FrTX, WarpSize * sizeof(double3) >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, Lovesun_d, J2_d, vold_d, dt_d, Kt[SIn - 1], time_d, N_h[0], Nst, P.UseGR, P.UseTides, P.UseRotationalDeformation, 0, 1);
			if(P.UseTides > 0 || P.UseRotationalDeformation > 0){
				if(N_h[0] + Nsmall_h[0] > FrTX){
					forced2_kernel <<< 1, ((ncb + WarpSize - 1) / WarpSize) * WarpSize, WarpSize * sizeof(double3) >>> (vold_d, Spinsun_d, nn, 1);
				}
			}
		}
		if(P.UseMigrationForce > 0){
			artificialMigration_kernel <<< (N_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, index_d, migration_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0, 1);
		}
		if(P.UseYarkovsky == 1) CallYarkovsky2_kernel <<< (N_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky_averaged_kernel <<< (N_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonEffect2_kernel <<< (N_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonEffect_averaged_kernel <<< (N_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		comCall(-1);
	}
	if(EjectionFlag_m[0] > 0){
		int Ej = EjectionCall();
		if(Ej == 0) return 0;
	}
#if def_poincareFlag == 1
	int per = PoincareSectionCall(time_h[0]);
	if(per == 0) return 0;
#endif
	return 1;
	
}
// *************************************************
// Step small
// *************************************************
__host__ int Data::step_small(int noColl){
	Rcrit_kernel <<< (N_h[0] + Nsmall_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, x4b_d, v4b_d, spin_d, spinb_d, 1.0 / (3.0 * Msun_h[0].x), rcrit_d, rcritb_d, rcritv_d, rcritvb_d, index_d, indexb_d, dt_h[0], n1_h[0], n2_h[0], time_d, time_h[0], EjectionFlag_d, N_h[0] + Nsmall_h[0], NconstT, P.SLevels, noColl);
	//use last time step information for setElements function, the beginning of the time step
	if(P.setElementsV == 2){ // convert barycentric velocities to heliocentric
		comCall(1);
	}	
	if(P.setElementsV > 0){
		setElements_kernel <<< (P.setElementsN + 63) / 64, 64 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst, 0);
	}
	if(P.setElementsV == 2){
		comCall(-1);
	}

	if(P.SERIAL_GROUPING == 1){
		Sortb_kernel<<< (N_h[0] + Nsmall_h[0] + RTX - 1) / RTX, RTX >>>(Encpairs2_d, 0, N_h[0] + Nsmall_h[0], P.NencMax);
	}
	if(EjectionFlag2 == 0){
		kick32Ab_kernel <<< (N_h[0] + Nsmall_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs2_d, 0, N_h[0] + Nsmall_h[0], P.NencMax, 1);
	}
	else{
		if(P.KickFloat == 0){
			acc4C_kernel <<< dim3( (((N_h[0] + Nsmall_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0] + Nsmall_h[0], 0, N_h[0], P.NencMax, KP, 1);
		}
		else{
			acc4Cf_kernel <<< dim3( (((N_h[0] + Nsmall_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0] + Nsmall_h[0], 0, N_h[0], P.NencMax, KP, 1);
		}
		if(P.UseTestParticles == 2){
			if(P.KickFloat == 0){
				acc4C_kernel <<< dim3( (((N_h[0] + KP2 - 1)/ KP2) + KTX2 - 1) / KTX2, 1, 1), dim3(KTX2,KTY2,1), KTX2 * KTY2 * KP2 * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0], N_h[0], N_h[0] + Nsmall_h[0], P.NencMax, KP2, 2);
			}
			else{
				acc4Cf_kernel <<< dim3( (((N_h[0] + KP2 - 1)/ KP2) + KTX2 - 1) / KTX2, 1, 1), dim3(KTX2,KTY2,1), KTX2 * KTY2 * KP2 * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0], N_h[0], N_h[0] + Nsmall_h[0], P.NencMax, KP2, 2);
			}
		}
		cudaEventRecord(KickEvent, 0);
		cudaStreamWaitEvent(copyStream, KickEvent, 0);
		cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);

		if(P.SERIAL_GROUPING == 1){
			Sortb_kernel<<< (N_h[0] + Nsmall_h[0] + RTX - 1) / RTX, RTX >>>(Encpairs2_d, 0, N_h[0] + Nsmall_h[0], P.NencMax);
		}
		kick32Ab_kernel <<< (N_h[0] + Nsmall_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs2_d, 0, N_h[0] + Nsmall_h[0], P.NencMax, 1);

	}
	if(ForceFlag > 0 || P.setElements > 1){
		comCall(1);
		if(P.setElements > 1){
			setElements_kernel <<< (P.setElementsN + 63) / 64, 64 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst, 1);
		}
		if(P.Usegas == 1){
			GasAccCall(time_d, dt_d, Kt[SIn - 1]);
			GasAccCall_small(time_d, dt_d, Kt[SIn - 1]);
		}
		if(P.Usegas == 2){
			//GasAccCall(time_d, dt_d, Kt[SIn - 1]);
			GasAccCall2_small(time_d, dt_d, Kt[SIn - 1]);
		}
		if(P.UseGR > 0 || P.UseTides > 0 || P.UseRotationalDeformation > 0 || P.UseJ2 > 0){
			int nn = (N_h[0] + Nsmall_h[0] + FrTX - 1) / FrTX;
			int ncb = min(nn, 1024);
			force_kernel <<< nn, FrTX, WarpSize * sizeof(double3) >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, Lovesun_d, J2_d, vold_d, dt_d, Kt[SIn - 1], time_d, N_h[0] + Nsmall_h[0], Nst, P.UseGR, P.UseTides, P.UseRotationalDeformation, 0, 1);
			if(P.UseTides > 0 || P.UseRotationalDeformation > 0){
				if(N_h[0] + Nsmall_h[0] > FrTX){
					forced2_kernel <<< 1, ((ncb + WarpSize - 1) / WarpSize) * WarpSize, WarpSize * sizeof(double3) >>> (vold_d, Spinsun_d, nn, 1);
				}
			}
		}
		if(P.UseYarkovsky == 1) CallYarkovsky2_kernel <<< (N_h[0] + Nsmall_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky_averaged_kernel <<< (N_h[0] + Nsmall_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonEffect2_kernel <<< (N_h[0] + Nsmall_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonEffect_averaged_kernel <<< (N_h[0] + Nsmall_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0);
		comCall(-1);
	}
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){

		HCCall(Ct[si], 1);
		
		fg_kernel <<<(N_h[0] + Nsmall_h[0] + FTX - 1)/FTX, FTX >>> (x4_d, v4_d, xold_d, vold_d, index_d, dt_h[0] * FGt[si], Msun_h[0].x, N_h[0] + Nsmall_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, si, P.UseGR);
		cudaStreamSynchronize(copyStream);

		if(P.WriteEncounters == 2 && si == 0){
			setNencpairs_kernel <<< 1, 1 >>> (Nencpairs2_d, 1);
			if(UseBVH == 1){
				BVHCall1();
			}
			if(UseBVH == 2){
				BVHCall2();
			}
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			if(Nencpairs2_h[0] > 0){
				encounter_small_kernel <<< (Nencpairs2_h[0] + 63)/ 64, 64 >>> (x4_d, v4_d, xold_d, vold_d, index_d, spin_d, dt_h[0] * FGt[si], Nencpairs2_h[0], Encpairs2_d, NWriteEnc_d, writeEnc_d, time_h[0]);
			}

			setNencpairs_kernel <<< 1, 1 >>> (Nencpairs2_d, 1);
		}

		if(Nenc_m[0] > 0){
			for(int i = 0; i < def_GMax; ++i){
				Nenc_m[i] = 0;
			}			
			setNencpairs_kernel <<< 1, 1 >>> (Nencpairs2_d, 1);
		}

		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 63)/ 64, 64 >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, dt_h[0] * FGt[si], Nencpairs_h[0], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, enccount_d, si, N_h[0] + Nsmall_h[0], time_h[0], P.StopAtEncounter, Ncoll_d, P.MinMass);
			cudaDeviceSynchronize();
			
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			if(P.StopAtEncounter > 0 && Ncoll_m[0] > 0){
				Ncoll_m[0] = 0;
				StopAtEncounterFlag2 = 1;
			}
//printf("Nencpairs2 A %d %d\n", Nencpairs_h[0], Nencpairs2_h[0]);
			if(P.SLevels > 1){
				if(Nencpairs2_h[0] > 0){
					double time = time_h[0];
					SEnc(time, 0, 1.0, si, noColl);
				}
			}
			else{
				if(Nencpairs2_h[0] > 0){
					if(P.UseTestParticles < 2){
//assume here E = 3 or E = 4
						group_kernel < 512, 512 > <<< 1, 512 >>> (Nenc_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0] + Nsmall_h[0], N_h[0], P.SERIAL_GROUPING);
					}
					else{
						group_kernel < 512, 512 > <<< 1, 512 >>> (Nenc_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0] + Nsmall_h[0], N_h[0] + Nsmall_h[0], P.SERIAL_GROUPING);
					}
				}	
				cudaDeviceSynchronize();
				BSCall(si, time_h[0], noColl, 1.0);
			}
		}
		if(StopAtEncounterFlag2 == 1){
			StopAtEncounterFlag2 = 0;
			int enc = StopAtEncounterCall();
			if(enc == 0) return 0;
		}
		if(Ncoll_m[0] > 0){
			int col = CollisionCall(noColl);
			if(col == 0) return 0;
		}
		if(P.UseSmallCollisions == 1 || P.UseSmallCollisions == 3){
			fragmentCall();
			if(nFragments_m[0] > 0){
				int er = printFragments(nFragments_m[0]);
				if(er == 0) return 0;
				er = RemoveCall();
				if(er == 0) return 0;
			}
		}
		if(P.UseSmallCollisions == 1 || P.UseSmallCollisions == 2){
			rotationCall();
			if(nFragments_m[0] > 0){
				int er = printRotation();
				if(er == 0) return 0;
			}
		}
		if(CollisionFlag == 1 && P.ei > 0 && timeStep % P.ei == 0){
			int rem = RemoveCall();
			if( rem == 0) return 0;
		}
		if(NWriteEnc_m[0] > 0){
			int enc = writeEncCall();
			if(enc == 0) return 0;
		}

		if(P.CreateParticles > 0){
			int er = createCall();
			if(er == 0) return 0;
			if(nFragments_m[0] > 0){
				printCreateparticle(nFragments_m[0]);
			}
		}

		HCCall(Ct[si], -1);
		if(si < SIn - 1){
			if(P.KickFloat == 0){
				acc4C_kernel <<< dim3( (((N_h[0] + Nsmall_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0] + Nsmall_h[0], 0, N_h[0], P.NencMax, KP, 1);
			}
			else{
				acc4Cf_kernel <<< dim3( (((N_h[0] + Nsmall_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0] + Nsmall_h[0], 0, N_h[0], P.NencMax, KP, 1);
			}
			if(P.UseTestParticles == 2){
				if(P.KickFloat == 0){
					acc4C_kernel <<< dim3( (((N_h[0] + KP2 - 1)/ KP2) + KTX2 - 1) / KTX2, 1, 1), dim3(KTX2,KTY2,1), KTX2 * KTY2 * KP2 * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0], N_h[0], N_h[0] + Nsmall_h[0], P.NencMax, KP2, 2);
				}
				else{
					acc4Cf_kernel <<< dim3( (((N_h[0] + KP2 - 1)/ KP2) + KTX2 - 1) / KTX2, 1, 1), dim3(KTX2,KTY2,1), KTX2 * KTY2 * KP2 * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0], N_h[0], N_h[0] + Nsmall_h[0], P.NencMax, KP2, 2);
				}
			}
				
			cudaEventRecord(KickEvent, 0);
			cudaStreamWaitEvent(copyStream, KickEvent, 0);
			cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);

			if(P.SERIAL_GROUPING == 1){
				Sortb_kernel<<< (N_h[0] + Nsmall_h[0] + RTX - 1) / RTX, RTX >>>(Encpairs2_d, 0, N_h[0] + Nsmall_h[0], P.NencMax);
			}
			kick32Ab_kernel <<< (N_h[0] + Nsmall_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[si] * def_ksq, Nencpairs_d, Encpairs2_d, 0, N_h[0] + Nsmall_h[0], P.NencMax, 1);
			if(ForceFlag > 0){
				comCall(1);
				if(P.Usegas == 1){
					GasAccCall(time_d, dt_d, Kt[si]);
					GasAccCall_small(time_d, dt_d, Kt[si]);
				}
				if(P.Usegas == 2){
					//GasAccCall(time_d, dt_d, Kt[si]);
					GasAccCall2_small(time_d, dt_d, Kt[si]);
				}
				if(P.UseGR > 0 || P.UseTides > 0 || P.UseRotationalDeformation > 0 || P.UseJ2 > 0){
					int nn = (N_h[0] + Nsmall_h[0] + FrTX - 1) / FrTX;
					int ncb = min(nn, 1024);
					force_kernel <<< nn, FrTX, WarpSize * sizeof(double3) >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, Lovesun_d, J2_d, vold_d, dt_d, Kt[si], time_d, N_h[0] + Nsmall_h[0], Nst, P.UseGR, P.UseTides, P.UseRotationalDeformation, 0, 1);
					if(P.UseTides > 0 || P.UseRotationalDeformation > 0){
						if(N_h[0] + Nsmall_h[0] > FrTX){
							forced2_kernel <<< 1, ((ncb + WarpSize - 1) / WarpSize) * WarpSize, WarpSize * sizeof(double3) >>> (vold_d, Spinsun_d, nn, 1);
						}
					}
				}
				if(P.UseYarkovsky == 1) CallYarkovsky2_kernel <<< (N_h[0] + Nsmall_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], N_h[0] + Nsmall_h[0], Nst, 0);
				if(P.UseYarkovsky == 2) CallYarkovsky_averaged_kernel <<< (N_h[0] + Nsmall_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], N_h[0] + Nsmall_h[0], Nst, 0);
				if(P.UsePR == 1) PoyntingRobertsonEffect2_kernel <<< (N_h[0] + Nsmall_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, index_d, dt_d, Kt[si], N_h[0] + Nsmall_h[0], Nst, 0);
				if(P.UsePR == 2) PoyntingRobertsonEffect_averaged_kernel <<< (N_h[0] + Nsmall_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[si], N_h[0] + Nsmall_h[0], Nst, 0);
				comCall(-1);
			}
		}
	}
	if(P.KickFloat == 0){
		acc4C_kernel <<< dim3( (((N_h[0] + Nsmall_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0] + Nsmall_h[0], 0, N_h[0], P.NencMax, KP, 1);
	}
	else{
		acc4Cf_kernel <<< dim3( (((N_h[0] + Nsmall_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0] + Nsmall_h[0], 0, N_h[0], P.NencMax, KP, 1);
	}
	if(P.UseTestParticles == 2){
		if(P.KickFloat == 0){
			acc4C_kernel <<< dim3( (((N_h[0] + KP2 - 1)/ KP2) + KTX2 - 1) / KTX2, 1, 1), dim3(KTX2,KTY2,1), KTX2 * KTY2 * KP2 * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0], N_h[0], N_h[0] + Nsmall_h[0], P.NencMax, KP2, 2);
		}
		else{
			acc4Cf_kernel <<< dim3( (((N_h[0] + KP2 - 1)/ KP2) + KTX2 - 1) / KTX2, 1, 1), dim3(KTX2,KTY2,1), KTX2 * KTY2 * KP2 * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0], N_h[0], N_h[0] + Nsmall_h[0], P.NencMax, KP2, 2);
		}
	}
	cudaEventRecord(KickEvent, 0);
	cudaStreamWaitEvent(copyStream, KickEvent, 0);
	cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);

	if(P.SERIAL_GROUPING == 1){
		Sortb_kernel<<< (N_h[0] + Nsmall_h[0] + RTX - 1) / RTX, RTX >>>(Encpairs2_d, 0, N_h[0] + Nsmall_h[0], P.NencMax);
	}
	kick32Ab_kernel <<< (N_h[0] + Nsmall_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs2_d, 0, N_h[0] + Nsmall_h[0], P.NencMax, 1);
	
	if(ForceFlag > 0){
		comCall(1);
		if(P.Usegas == 1){
			GasAccCall(time_d, dt_d, Kt[SIn - 1]);
			GasAccCall_small(time_d, dt_d, Kt[SIn - 1]);
		}
		if(P.Usegas == 2){
			//GasAccCall(time_d, dt_d, Kt[SIn - 1]);
			GasAccCall2_small(time_d, dt_d, Kt[SIn - 1]);
		}
		if(P.UseGR > 0 || P.UseTides > 0 || P.UseRotationalDeformation > 0 || P.UseJ2 > 0){
			int nn = (N_h[0] + Nsmall_h[0] + FrTX - 1) / FrTX;
			int ncb = min(nn, 1024);
			force_kernel <<< nn, FrTX, WarpSize * sizeof(double3) >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, Lovesun_d, J2_d, vold_d, dt_d, Kt[SIn - 1], time_d, N_h[0] + Nsmall_h[0], Nst, P.UseGR, P.UseTides, P.UseRotationalDeformation, 0, 1);
			if(P.UseTides > 0 || P.UseRotationalDeformation > 0){
				if(N_h[0] + Nsmall_h[0] > FrTX){
					forced2_kernel <<< 1, ((ncb + WarpSize - 1) / WarpSize) * WarpSize, WarpSize * sizeof(double3) >>> (vold_d, Spinsun_d, nn, 1);
				}
			}
		}
		if(P.UseYarkovsky == 1) CallYarkovsky2_kernel <<< (N_h[0] + Nsmall_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky_averaged_kernel <<< (N_h[0] + Nsmall_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonEffect2_kernel <<< (N_h[0] + Nsmall_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonEffect_averaged_kernel <<< (N_h[0] + Nsmall_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0);
		comCall(-1);
	}
	if(EjectionFlag_m[0] > 0){
		int Ej = EjectionCall();
		if(Ej == 0) return 0;
	}
	return 1;
}

// *************************************************
// Step M
// *************************************************
__host__ int Data::step_M(int noColl){
	RcritM_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, x4b_d, v4b_d, spin_d, spinb_d, Msun_d, rcrit_d, rcritb_d, rcritv_d, rcritvb_d, dt_d, n1_d, n2_d, Rcut_d, RcutSun_d, EjectionFlag_d, index_d, indexb_d, Nst, NT, time_d, idt_d, ict_d, delta_d, timeStep, StopFlag_d, NconstT, P.SLevels, noColl, Nstart);
	if(doTransits == 0){
		if(EjectionFlag2 == 0){
			if(Nencpairs_h[0] == 0){
				kick32BM_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, a_d, ab_d, index_d, NT, dt_d, Kt[SIn - 1], Nstart);
			}
			else{
				KickM2_kernel < KM_Bl, KM_Bl2, NmaxM, 3 > <<< (NT + KM_Bl2 - 1) / KM_Bl2, KM_Bl>>> (x4_d, v4_d, a_d, rcritv_d, Nencpairs_d, Encpairs_d, dt_d, Kt[SIn - 1], index_d, NT, Nstart);
			}
		}
		else{
			KickM2_kernel < KM_Bl, KM_Bl2, NmaxM, 1 > <<< (NT + KM_Bl2 - 1) / KM_Bl2, KM_Bl>>> (x4_d, v4_d, a_d, rcritv_d, Nencpairs_d, Encpairs_d, dt_d, Kt[SIn - 1], index_d, NT, Nstart);
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
			//cudaEventRecord(KickEvent, 0);
			//cudaStreamWaitEvent(copyStream, KickEvent, 0);
			//cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);
		}
	}
#if def_TTV == 1
//printf("%lld %.20g %d %d\n", timeStep, time_h[0], Nencpairs_h[0], EjectionFlag2);
	if(doTransits == 1){
		if(EjectionFlag2 == 0){
			if(Nencpairs_h[0] == 0){
				kick32BMTTV_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, a_d, ab_d, index_d, NT, dt_d, Kt[SIn - 1], Msun_d, Ntransit_d, Transit_d, Nstart);
			}
			else{
				KickM2TTV_kernel < KM_Bl, KM_Bl2, NmaxM, 3 > <<< (NT + KM_Bl2 - 1) / KM_Bl2, KM_Bl>>> (x4_d, v4_d, a_d, rcritv_d, Nencpairs_d, Encpairs_d, dt_d, Kt[SIn - 1], index_d, NT, Msun_d, Ntransit_d, Transit_d, Nstart);

			}
		}
		else{
			KickM2TTV_kernel < KM_Bl, KM_Bl2, NmaxM, 1 > <<< (NT + KM_Bl2 - 1) / KM_Bl2, KM_Bl>>> (x4_d, v4_d, a_d, rcritv_d, Nencpairs_d, Encpairs_d, dt_d, Kt[SIn - 1], index_d, NT, Msun_d, Ntransit_d, Transit_d, Nstart);
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
			//cudaEventRecord(KickEvent, 0);
			//cudaStreamWaitEvent(copyStream, KickEvent, 0);
			//cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);
		}
		cudaDeviceSynchronize();
		if(Ntransit_m[0] > 0){
			if(Ntransit_m[0] >= def_NtransitMax - 1){
				printf("more Transits than allowed in def_NtransitMax: %d\n", def_NtransitMax);
				return 0;
			}
			BSTTVStep_kernel < 8, 8 > <<< Ntransit_m[0], 64 >>> (x4_d, v4b_d, Transit_d, N_d, dt_d, Msun_d, index_d, time_d, NBS_d, P.UseGR, P.MinMass, P.UseTestParticles, Nst, TransitTime_d, NtransitsT_d);
			Ntransit_m[0] = 0;
		}
	}
#endif

#if def_RV == 1
	while(time_h[0] >= RVObs_h[RVTimeStep].x && RVTimeStep < NRVTObs_h[0]){
		//repeat for multiple intertime steps
		double Tau = RVObs_h[RVTimeStep].x - (time_h[0] - dt_h[0] / dayUnit);
		//printf("timeRV %d %d %.20g %.20g %.20g %.20g\n", RVTimeStep, NRVTObs_h[0], time_h[0], time_h[0] - dt_h[0] / dayUnit, RVObs_h[RVTimeStep].x, Tau);
		BSRVStep_kernel < 8, 8 > <<< Nst, 64 >>> (x4_d, v4b_d, N_d, Tau * dayUnit, Msun_d, index_d, time_h[0] - dt_h[0] / dayUnit, NBS_d, P.UseGR, P.MinMass, P.UseTestParticles, Nst, RV_d, NRVT_d);
		++RVTimeStep;
	}
	//printf("time   %d %.20g %.20g %.20g\n", RVTimeStep, time_h[0], time_h[0] - dt_h[0] / dayUnit, RVObs_h[RVTimeStep].x);
	if(P.PrintRV == 2){
		BSRVStep_kernel < 8, 8 > <<< Nst, 64 >>> (x4_d, v4b_d, N_d, dt_h[0], Msun_d, index_d, time_h[0] - dt_h[0] / dayUnit, NBS_d, P.UseGR, P.MinMass, P.UseTestParticles, Nst, RV_d, NRVT_d);
	}


#endif
	if(ForceFlag > 0){
		comM_kernel < HCM_Bl, HCM_Bl2, NmaxM > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, vcom_d, Msun_d, index_d, NBS_d, NT, 1, Nstart);
		if(P.Usegas == 1) GasAccCall_M(time_d, dt_d, Kt[SIn - 1]);
		if(P.UseGR > 0 || P.UseTides > 0 || P.UseRotationalDeformation > 0 || P.UseJ2 > 0){
			force_kernel <<< (NT + 127) / 128, 128, WarpSize * sizeof(double3) >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, Lovesun_d, J2_d, vold_d, dt_d, Kt[SIn - 1], time_d, NT, Nst, P.UseGR, P.UseTides, P.UseRotationalDeformation, Nstart, 1);
			if(P.UseTides > 0 || P.UseRotationalDeformation > 0){
				forceM_kernel < HCM_Bl, HCM_Bl2, NmaxM > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (vold_d, index_d, Spinsun_d, NBS_d, NT, Nstart);
			}
		}
		if(P.UseYarkovsky == 1) CallYarkovsky2_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], NT, Nst, Nstart);
		if(P.UseYarkovsky == 2) CallYarkovsky_averaged_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], NT, Nst, Nstart);
		if(P.UsePR == 1) PoyntingRobertsonEffect2_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], NT, Nst, Nstart);
		if(P.UsePR == 2) PoyntingRobertsonEffect_averaged_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], NT, Nst, Nstart);
		comM_kernel < HCM_Bl, HCM_Bl2, NmaxM > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, vcom_d, Msun_d, index_d, NBS_d, NT, -1, Nstart);
	}
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		if(P.UseGR == 1){
			convertVToPseidovM <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, ErrorFlag_m, Msun_d, NT);
		}
		HCM2_kernel < HCM_Bl, HCM_Bl2, NmaxM, 1 > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, dt_d, Msun_d, index_d, NT, Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, Nst, P.UseGR, Nstart);
		fgM_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, xold_d, vold_d, dt_d, Msun_d, index_d, NT, aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, FGt[si], si, P.UseGR, Nstart);
		cudaStreamSynchronize(copyStream);
		if(Nencpairs_h[0] > 0){
			encounterM_kernel <<< (Nencpairs_h[0] + 127) / 128 , 128 >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, dt_d, Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, index_d, NBS_d, enccount_d, si, FGt[si], Nst, time_d, P.StopAtEncounter, Ncoll_d, n1_d, P.MinMass, P.NencMax);
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			
			if(P.StopAtEncounter > 0 && Ncoll_m[0] > 0){
				Ncoll_m[0] = 0;
				StopAtEncounterFlag2 = 1;
			}
			
			if(Nencpairs2_h[0] > 0){
				if(NBmax < 64){
					groupM1_kernel < 32, 256 > <<< Nencpairs2_h[0], 256 >>> (Nencpairs2_d, Encpairs_d, Encpairs2_d, NBS_d, N_d, Nst, P.NencMax);
				}
				else if(NBmax < 128){
					groupM1_kernel < 64, 256 > <<< Nencpairs2_h[0], 256 >>> (Nencpairs2_d, Encpairs_d, Encpairs2_d, NBS_d, N_d, Nst, P.NencMax);
				}
				else if(NBmax < 256){
					groupM1_kernel < 128, 256 > <<< Nencpairs2_h[0], 256 >>> (Nencpairs2_d, Encpairs_d, Encpairs2_d, NBS_d, N_d, Nst, P.NencMax);
				}
				groupM2_kernel <<< Nencpairs2_h[0], NBmax >>> (Encpairs_d, Encpairs2_d, Nenc_d, NBS_d, N_d, Nst, P.NencMax);
				cudaDeviceSynchronize();
				BSBMCall(si, noColl, 1.0);
			}
		}
		if(StopAtEncounterFlag2 == 1){
			StopAtEncounterFlag2 = 0;
			int enc = StopAtEncounterCall();
			if(enc == 0) return 0;
		}
		if(Ncoll_m[0] > 0){
			int col = CollisionMCall(noColl);
			if(col == 0) return 0;
		}
		if(CollisionFlag == 1 && P.ei > 0 && timeStep % P.ei == 0){
			int rem = RemoveCall();
			if( rem == 0) return 0;
		}
		if(NWriteEnc_m[0] > 0){
			int enc = writeEncCall();
			if(enc == 0) return 0;
		}
		
		HCM2_kernel < HCM_Bl, HCM_Bl2, NmaxM, 2 > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, dt_d, Msun_d, index_d, NT, Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, Nst, P.UseGR, Nstart);
		if(P.UseGR == 1){
			convertPseudovToVM <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, Msun_d, NT);
		}
		if(si < SIn - 1){
			KickM2_kernel < KM_Bl, KM_Bl2, NmaxM, 2 > <<< (NT + KM_Bl2 - 1) / KM_Bl2, KM_Bl>>> (x4_d, v4_d, a_d, rcritv_d, Nencpairs_d, Encpairs_d, dt_d, Kt[si], index_d, NT, Nstart);
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
		//	cudaEventRecord(KickEvent, 0);
		//	cudaStreamWaitEvent(copyStream, KickEvent, 0);
		//	cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);
			if(ForceFlag > 0){
				comM_kernel < HCM_Bl, HCM_Bl2, NmaxM > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, vcom_d, Msun_d, index_d, NBS_d, NT, 1, Nstart);
				if(P.Usegas == 1) GasAccCall_M(time_d, dt_d, Kt[si]);
				if(P.UseGR > 0 || P.UseTides > 0 || P.UseRotationalDeformation > 0 || P.UseJ2 > 0){
					force_kernel <<< (NT + 127) / 128, 128, WarpSize * sizeof(double3) >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, Lovesun_d, J2_d, vold_d, dt_d, Kt[si], time_d, NT, Nst, P.UseGR, P.UseTides, P.UseRotationalDeformation, Nstart, 1);
					if(P.UseTides > 0 || P.UseRotationalDeformation > 0){
						forceM_kernel < HCM_Bl, HCM_Bl2, NmaxM > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (vold_d, index_d, Spinsun_d, NBS_d, NT, Nstart);
					}

				}
				if(P.UseYarkovsky == 1) CallYarkovsky2_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], NT, Nst, Nstart);
				if(P.UseYarkovsky == 2) CallYarkovsky_averaged_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], NT, Nst, Nstart);
				if(P.UsePR == 1) PoyntingRobertsonEffect2_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, dt_d, Kt[si], NT, Nst, Nstart);
				if(P.UsePR == 2) PoyntingRobertsonEffect_averaged_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[si], NT, Nst, Nstart);
				comM_kernel < HCM_Bl, HCM_Bl2, NmaxM > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, vcom_d, Msun_d, index_d, NBS_d, NT, -1, Nstart);
			}
		}
	}
	KickM2_kernel < KM_Bl, KM_Bl2, NmaxM, 1 > <<< (NT + KM_Bl2 - 1) / KM_Bl2, KM_Bl>>> (x4_d, v4_d, a_d, rcritv_d, Nencpairs_d, Encpairs_d, dt_d, Kt[SIn - 1], index_d, NT, Nstart);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
	//cudaEventRecord(KickEvent, 0);
	//cudaStreamWaitEvent(copyStream, KickEvent, 0);
	//cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);
	if(ForceFlag > 0){
		comM_kernel < HCM_Bl, HCM_Bl2, NmaxM > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, vcom_d, Msun_d, index_d, NBS_d, NT, 1, Nstart);
		if(P.Usegas == 1) GasAccCall_M(time_d, dt_d, Kt[SIn - 1]);
		if(P.UseGR > 0 || P.UseTides > 0 || P.UseRotationalDeformation > 0 || P.UseJ2 > 0){
			force_kernel <<< (NT + 127) / 128, 128, WarpSize * sizeof(double3) >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, Lovesun_d, J2_d, vold_d, dt_d, Kt[SIn - 1], time_d, NT, Nst, P.UseGR, P.UseTides, P.UseRotationalDeformation, Nstart, 1);
			if(P.UseTides > 0 || P.UseRotationalDeformation > 0){
				forceM_kernel < HCM_Bl, HCM_Bl2, NmaxM > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (vold_d, index_d, Spinsun_d, NBS_d, NT, Nstart);
			}
		}

		if(P.UseYarkovsky == 1) CallYarkovsky2_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], NT, Nst, Nstart);
		if(P.UseYarkovsky == 2) CallYarkovsky_averaged_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], NT, Nst, Nstart);
		if(P.UsePR == 1) PoyntingRobertsonEffect2_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], NT, Nst, Nstart);
		if(P.UsePR == 2) PoyntingRobertsonEffect_averaged_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], NT, Nst, Nstart);
		comM_kernel < HCM_Bl, HCM_Bl2, NmaxM > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, vcom_d, Msun_d, index_d, NBS_d, NT, -1, Nstart);
	}
	
	if(EjectionFlag_m[0] > 0){
		int Ej = EjectionCall();
		if(Ej == 0) return 0;
	}
	
	if(StopFlag_m[0] == 1){
		if(P.ci != 0){
			CoordinateOutput(3);
			EnergyOutput(3);
			printTime(3);
		}
		printLastTime(3);
		
		stopSimulations();
		StopFlag_m[0] = 0;
	}
	return 1;
}
// *************************************************
// Step M3
// *************************************************
__host__ int Data::step_M3(int noColl){

	RcritM_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, x4b_d, v4b_d, spin_d, spinb_d, Msun_d, rcrit_d, rcritb_d, rcritv_d, rcritvb_d, dt_d, n1_d, n2_d, Rcut_d, RcutSun_d, EjectionFlag_d, index_d, indexb_d, Nst, NT, time_d, idt_d, ict_d, delta_d, timeStep, StopFlag_d, NconstT, P.SLevels, noColl, Nstart);
	if(EjectionFlag2 == 0){
		if(Nencpairs_h[0] == 0){
			kick32BM_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, a_d, ab_d, index_d, NT, dt_d, Kt[SIn - 1], Nstart);
		}
		else{
			KickM3_kernel <<< NT, dim3(KTM3, 1, 1), WarpSize * sizeof(double3) >>> (x4_d, v4_d, a_d, rcritv_d, Nencpairs_d, Encpairs_d, dt_d, Kt[SIn - 1], index_d, NT, N_d, NBS_d, 2, 0);
		}
	}
	else{
		KickM3_kernel <<< NT, dim3(KTM3, 1, 1), WarpSize * sizeof(double3) >>> (x4_d, v4_d, a_d, rcritv_d, Nencpairs_d, Encpairs_d, dt_d, Kt[SIn - 1], index_d, NT, N_d, NBS_d, 1, 1);
		cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
		//cudaEventRecord(KickEvent, 0);
		//cudaStreamWaitEvent(copyStream, KickEvent, 0);
		//cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);
	}
	if(ForceFlag > 0){
		comBM_kernel <<< Nst, 32, WarpSize * sizeof(double3) >>> (x4_d, v4_d, vcom_d, Msun_d, N_d, NBS_d, Nst, 1);
		if(P.Usegas == 1) GasAccCall_M(time_d, dt_d, Kt[SIn - 1]);
		if(P.UseGR > 0 || P.UseTides > 0 || P.UseRotationalDeformation > 0 || P.UseJ2 > 0){
			force_kernel <<< (NT + 127) / 128, 128, WarpSize * sizeof(double3) >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, Lovesun_d, J2_d, vold_d, dt_d, Kt[SIn - 1], time_d, NT, Nst, P.UseGR, P.UseTides, P.UseRotationalDeformation, Nstart, 1);
			if(P.UseTides > 0 || P.UseRotationalDeformation > 0){
				forceBM_kernel <<< Nst, 32, WarpSize * sizeof(double3) >>> (vold_d, Spinsun_d, N_d, NBS_d, Nst);
			}
		}
		if(P.UseMigrationForce > 0){
			artificialMigration_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, migration_d, Msun_d, dt_d, Kt[SIn - 1], NT, Nst, 0, 1);
		}
		if(P.UseYarkovsky == 1) CallYarkovsky2_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], NT, Nst, Nstart);
		if(P.UseYarkovsky == 2) CallYarkovsky_averaged_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], NT, Nst, Nstart);
		if(P.UsePR == 1) PoyntingRobertsonEffect2_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], NT, Nst, Nstart);
		if(P.UsePR == 2) PoyntingRobertsonEffect_averaged_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], NT, Nst, Nstart);
		comBM_kernel <<< Nst, 32, WarpSize * sizeof(double3) >>> (x4_d, v4_d, vcom_d, Msun_d, N_d, NBS_d, Nst, -1);
	}
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		if(P.UseGR == 1){
			convertVToPseidovM <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, ErrorFlag_m, Msun_d, NT);
		}
		HC32aM_kernel <<< Nst, HCTM3, WarpSize * sizeof(double3) >>> (x4_d, v4_d, dt_d, Msun_d, N_d, NBS_d, Nst, Ct[si], P.UseGR, 1);
		fgM_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, xold_d, vold_d, dt_d, Msun_d, index_d, NT, aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, FGt[si], si, P.UseGR, Nstart);
		cudaStreamSynchronize(copyStream);

		for(int i = 0; i < def_GMax; ++i){
			Nenc_m[i] = 0;
		}
		setNencpairs_kernel <<< (Nst + 1 + 127) / 128, 128 >>> (Nencpairs2_d, Nst + 1);
		setNencpairs2_kernel <<< (NT + 127) / 128, 128 >>> (groupIndex_d, NT);

		if(Nencpairs_h[0] > 0){
			encounterM3_kernel <<< (Nencpairs_h[0] + 127) / 128 , 128 >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, dt_d, Nencpairs_h[0], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, index_d, NBS_d, enccount_d, si, FGt[si], Nst, time_d, P.StopAtEncounter, Ncoll_d, n1_d, P.MinMass, P.NencMax);
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			
			if(P.StopAtEncounter > 0 && Ncoll_m[0] > 0){
				Ncoll_m[0] = 0;
				StopAtEncounterFlag2 = 1;
			}
			if(Nencpairs2_h[0] > 0){
				if(NBmax < 64){
					groupM3_kernel < 32, 256 > <<< Nst, 256 >>> (Nenc_d, Nencpairs2_d, Encpairs_d, Encpairs2_d, groupIndex_d, NBS_d, N_d, NT, P.NencMax, P.SERIAL_GROUPING);
				}
				else if(NBmax < 128){
					groupM3_kernel < 64, 256 > <<< Nst, 256 >>> (Nenc_d, Nencpairs2_d, Encpairs_d, Encpairs2_d, groupIndex_d, NBS_d, N_d, NT, P.NencMax, P.SERIAL_GROUPING);
				}
				else if(NBmax < 256){
					groupM3_kernel < 128, 256 > <<< Nst, 256 >>> (Nenc_d, Nencpairs2_d, Encpairs_d, Encpairs2_d, groupIndex_d, NBS_d, N_d, NT, P.NencMax, P.SERIAL_GROUPING);
				}

				groupM3_2_kernel <<< (NT + 127) / 128, 128 >>> (Nenc_d, Nencpairs2_d, Encpairs_d, Encpairs2_d, groupIndex_d, NBS_d, NT, NBmax);
				cudaDeviceSynchronize();
				BSBM3Call(si, noColl, 1.0);
			}
		}
		if(StopAtEncounterFlag2 == 1){
			StopAtEncounterFlag2 = 0;
			int enc = StopAtEncounterCall();
			if(enc == 0) return 0;
		}
		if(Ncoll_m[0] > 0){
			int col = CollisionMCall(noColl);
			if(col == 0) return 0;
		}
		if(CollisionFlag == 1 && P.ei > 0 && timeStep % P.ei == 0){
			int rem = RemoveCall();
			if( rem == 0) return 0;
		}
		if(NWriteEnc_m[0] > 0){
			int enc = writeEncCall();
			if(enc == 0) return 0;
		}
		
		HC32aM_kernel <<< Nst, HCTM3, WarpSize * sizeof(double3) >>>(x4_d, v4_d, dt_d, Msun_d, N_d, NBS_d, Nst, Ct[si], P.UseGR, 1);
		setNencpairs_kernel <<< (Nst + 1 + 127) / 128, 128 >>> (Nencpairs_d, Nst + 1);
		if(P.UseGR == 1){
			convertPseudovToVM <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, Msun_d, NT);
		}
		if(si < SIn - 1){
			KickM3_kernel <<< NT, dim3(KTM3, 1, 1), WarpSize * sizeof(double3) >>> (x4_d, v4_d, a_d, rcritv_d, Nencpairs_d, Encpairs_d, dt_d, Kt[si], index_d, NT, N_d, NBS_d, 2, 1);
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
		//	cudaEventRecord(KickEvent, 0);
		//	cudaStreamWaitEvent(copyStream, KickEvent, 0);
		//	cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);
			if(ForceFlag > 0){
				comBM_kernel <<< Nst, 32, WarpSize * sizeof(double3) >>> (x4_d, v4_d, vcom_d, Msun_d, N_d, NBS_d, Nst, 1);
				if(P.Usegas == 1) GasAccCall_M(time_d, dt_d, Kt[si]);
				if(P.UseGR > 0 || P.UseTides > 0 || P.UseRotationalDeformation > 0 || P.UseJ2 > 0){
					force_kernel <<< (NT + 127) / 128, 128, WarpSize * sizeof(double3) >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, Lovesun_d, J2_d, vold_d, dt_d, Kt[si], time_d, NT, Nst, P.UseGR, P.UseTides, P.UseRotationalDeformation, Nstart, 1);
					if(P.UseTides > 0 || P.UseRotationalDeformation > 0){
						forceBM_kernel <<< Nst, 32, WarpSize * sizeof(double3) >>> (vold_d, Spinsun_d, N_d, NBS_d, Nst);
					}

				}
				if(P.UseMigrationForce > 0){
					artificialMigration_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, migration_d, Msun_d, dt_d, Kt[si], NT, Nst, 0, 1);
				}
				if(P.UseYarkovsky == 1) CallYarkovsky2_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], NT, Nst, Nstart);
				if(P.UseYarkovsky == 2) CallYarkovsky_averaged_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], NT, Nst, Nstart);
				if(P.UsePR == 1) PoyntingRobertsonEffect2_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, dt_d, Kt[si], NT, Nst, Nstart);
				if(P.UsePR == 2) PoyntingRobertsonEffect_averaged_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[si], NT, Nst, Nstart);
				comBM_kernel <<< Nst, 32, WarpSize * sizeof(double3) >>> (x4_d, v4_d, vcom_d, Msun_d, N_d, NBS_d, Nst, -1);
			}
		}
	}
	KickM3_kernel <<< NT, dim3(KTM3, 1, 1), WarpSize * sizeof(double3) >>> (x4_d, v4_d, a_d, rcritv_d, Nencpairs_d, Encpairs_d, dt_d, Kt[SIn - 1], index_d, NT, N_d, NBS_d, 1, 1);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
	//cudaEventRecord(KickEvent, 0);
	//cudaStreamWaitEvent(copyStream, KickEvent, 0);
	//cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);
	if(ForceFlag > 0){
		comBM_kernel <<< Nst, 32, WarpSize * sizeof(double3) >>> (x4_d, v4_d, vcom_d, Msun_d, N_d, NBS_d, Nst, 1);
		if(P.Usegas == 1) GasAccCall_M(time_d, dt_d, Kt[SIn - 1]);
		if(P.UseGR > 0 || P.UseTides > 0 || P.UseRotationalDeformation > 0 || P.UseJ2 > 0){
			force_kernel <<< (NT + 127) / 128, 128, WarpSize * sizeof(double3) >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, Lovesun_d, J2_d, vold_d, dt_d, Kt[SIn - 1], time_d, NT, Nst, P.UseGR, P.UseTides, P.UseRotationalDeformation, Nstart, 1);
			if(P.UseTides > 0 || P.UseRotationalDeformation > 0){
				forceBM_kernel <<< Nst, 32, WarpSize * sizeof(double3) >>> (vold_d, Spinsun_d, N_d, NBS_d, Nst);
			}
		}
		if(P.UseMigrationForce > 0){
			artificialMigration_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, migration_d, Msun_d, dt_d, Kt[SIn - 1], NT, Nst, 0, 1);
		}
		if(P.UseYarkovsky == 1) CallYarkovsky2_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], NT, Nst, Nstart);
		if(P.UseYarkovsky == 2) CallYarkovsky_averaged_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], NT, Nst, Nstart);
		if(P.UsePR == 1) PoyntingRobertsonEffect2_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], NT, Nst, Nstart);
		if(P.UsePR == 2) PoyntingRobertsonEffect_averaged_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], NT, Nst, Nstart);
		comBM_kernel <<< Nst, 32, WarpSize * sizeof(double3) >>> (x4_d, v4_d, vcom_d, Msun_d, N_d, NBS_d, Nst, -1);
	}
	
	if(EjectionFlag_m[0] > 0){
		int Ej = EjectionCall();
		if(Ej == 0) return 0;
	}
	
	if(StopFlag_m[0] == 1){
		if(P.ci != 0){
			CoordinateOutput(3);
			EnergyOutput(3);
			printTime(3);
		}
		printLastTime(3);
		
		stopSimulations();
		StopFlag_m[0] = 0;
	}
	return 1;
}
// *************************************************
// Step M simple
// *************************************************
__host__ int Data::step_MSimple(){
	if(doTransits == 0){
		kick32BMSimple_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, a_d, ab_d, index_d, NT, dt_d, Kt[SIn - 1], time_d, idt_d, ict_d, timeStep, Nst, Nstart);
	}
#if def_TTV == 1
	if(doTransits == 1){
		kick32BMTTVSimple_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, a_d, ab_d, index_d, NT, dt_d, Kt[SIn - 1], Msun_d, Ntransit_d, Transit_d, time_d, idt_d, ict_d, timeStep, Nst, Nstart);
		cudaDeviceSynchronize();
		if(Ntransit_m[0] > 0){
			if(Ntransit_m[0] >= def_NtransitMax - 1){
				printf("more Transits than allowed in def_NtransitMax: %d\n", def_NtransitMax);
				return 0;
			}
			BSTTVStep_kernel < 8, 8 > <<< Ntransit_m[0], 64 >>> (x4_d, v4b_d, Transit_d, N_d, dt_d, Msun_d, index_d, time_d, NBS_d, P.UseGR, P.MinMass, P.UseTestParticles, Nst, TransitTime_d, NtransitsT_d);
			Ntransit_m[0] = 0;
		}
	}
#endif
	for(int si = 0; si < SIn; ++si){
		HCM2_kernel < HCM_Bl, HCM_Bl2, NmaxM, 1 > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, dt_d, Msun_d, index_d, NT, Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, Nst, P.UseGR, Nstart);
		fgMSimple_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, xold_d, vold_d, dt_d, Msun_d, index_d, NT, FGt[si], si, P.UseGR, Nstart);
		HCM2_kernel < HCM_Bl, HCM_Bl2, NmaxM, 2 > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, dt_d, Msun_d, index_d, NT, Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, Nst, P.UseGR, Nstart);
		
		if(si < SIn - 1){
			KickM2Simple_kernel < KM_Bl, KM_Bl2, NmaxM, 2 > <<< (NT + KM_Bl2 - 1) / KM_Bl2, KM_Bl>>> (x4_d, v4_d, v4b_d, a_d, dt_d, Kt[si], index_d, NT, Nst, Nstart);
		}
	}
	KickM2Simple_kernel < KM_Bl, KM_Bl2, NmaxM, 1 > <<< (NT + KM_Bl2 - 1) / KM_Bl2, KM_Bl>>> (x4_d, v4_d, v4b_d, a_d, dt_d, Kt[SIn - 1], index_d, NT, Nst, Nstart);
	
	return 1;
}

#if def_TTV == 2
__host__ int Data::ttv_step(){

	int nsteps = 1;

	ttv_step_kernel < 4 > <<< (Nst + 3) / 4, dim3(7, 4, 1) >>> (x4_d, v4_d, xold_d, vold_d, dt_d, idt_h[0] * dayUnit, Msun_d, N_h[0], Nst, nsteps, time_d, timeold_d, lastTransitTime_d, transitIndex_d, NtransitsT_d, TransitTime_d, TransitTimeObs_d, EpochCount_d, TTV_d, P.PrintTransits);
	
	return 1;

}
#endif

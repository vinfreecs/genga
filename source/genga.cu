/**************************************
*
* Authors: Simon Grimm, Joachmin Stadel
* July 2013
*
****************************************/

#include "define.h"

#include "Host2.h"
#include "Orbit2.h"
#include "signal.h"

#if def_TTV > 0
	#include "TTVStep2.h"
#endif


volatile sig_atomic_t interrupted = 0;
volatile sig_atomic_t terminated = 0;

void catch_signal(int sig){
	interrupted = 1;
	printf("Signal %d received\n", sig);
}
void catch_signal2(int sig){
	terminated = 1;
	printf("Signal %d received\n", sig);
	exit(sig);
}


int main(int argc, char*argv[]){


	//Register signal handler
	signal(SIGINT, catch_signal);	//Ctrl C
	signal(SIGTERM, catch_signal2);	//terminate signal


	long long Restart = 0LL;
	int RRestart = 0;
	//Check if simulation is restarted
	for(int i = 1; i < argc; i += 2){
		if(strcmp(argv[i], "-R") == 0){
			Restart = atol(argv[i + 1]);
			RRestart = 1;
		}
	}

	Data H(Restart);

	if(H.Lock == 1){
		printf("lock.dat file already exists for the current start time. Delete or modify the file to continue\n");
		fprintf(H.masterfile, "lock.dat file already exists for the current start time. Delete or modify the file to continue\n");
		return 0;
	}


	if(RRestart == 0){
		printf("Start GENGA\n");
		fprintf(H.masterfile,"Start GENGA\n");
	}
	if(RRestart == 1){
		printf("Restart GENGA\n");
		fprintf(H.masterfile,"\n \n **************************************** \n \n");
		fprintf(H.masterfile,"Restart GENGA\n");
	}

#if SERIAL_GROUPING > 0
	printf("Using serial grouping!\n");
	fprintf(H.masterfile, "Using serial grouping!\n");
#endif
	//determine the number of simulations
	int Nst = H.NSimulations(argc, argv);
	if(Nst == 0) return 0;

	//Check Device Informations
	int DevError = H.DeviceInfo();
	if(DevError == 0) return 0;

	//Allocate memory for parameters on the host:
	H.Halloc();

	// Read parameters from param file //
	printf("Read parameters\n");
	int er = H.Param(argc, argv);
	if(er == 0) return 0;
	printf("Parameters OK\n");

	

	// Determine the size of the simulations
	printf("Read Size\n");
	er = H.size();
	if(er == 0){
		return 0;
	}
	printf("Size OK\n");

	cudaSetDevice(H.P.dev);
	cudaDeviceSynchronize();

	//Allocate memory for parameters on the device:
	H.Calloc();
	H.Info();

	//Determine the start points of the individual simulations
	H.Tsizes();

	Data D = H;

	er = D.beforeTimeStepLoop1();
	if(er == 0) return 0;

	int ittv = 0;
	D.Nstart = 0;

#if def_RV > 0
	cudaMemset(D.NRVT_d, 0, D.Nst * sizeof(int2));
#endif
#if def_TTV > 0
	cudaMemset(D.NtransitsT_d, 0, D.NconstT * sizeof(int2));
 #if MCMC_BLOCK == 3		
	if(D.Nst > 1) D.NT /= 2;
 #endif
	SetTTVP <<< (Nst + 255) / 256, 256 >>> (D.elementsP_d, D.elementsSA_d, D.Nst);

//start MCMC step loop here
for(ittv = 0; ittv < D.P.TransitSteps; ++ittv){
	D.time_h[0] = (D.P.tRestart + 1) * D.idt_h[0] + D.ict_h[0] * 365.25;
	cudaMemset(D.Nencpairs_d, 0, (D.Nst + 1) * sizeof(int));
	cudaMemset(D.TransitTime_d, 0, def_NtransitTimeMax * D.NconstT * sizeof(double));
	cudaDeviceSynchronize();
 #if def_RV > 0
	D.RVTimeStep = 0; 
 #endif
 #if MCMC_BLOCK == 3		
	if (D.Nst > 1) D.Nstart = (ittv % 2) * D.NT;
if(ittv % 2 == 1) printf("*********** TTV Step %d *********** %d\n", ittv, D.Nstart);
 #else
printf("*********** TTV Step %d *********** %d\n", ittv, D.Nstart);
 #endif

 #if MCMC_BLOCK == 3
//	if(ittv > 1 && ittv % 4 == 0) Mix_kernel <<< (D.NT + 255) / 256, 256 >>> (D.elementsA_d, D.elementsB_d, D.elementsAOld_d, D.elementsBOld_d, D.elementsC_d, D.elementsP_d, D.Nst, D.NT, D.N_h[0]);
	setJ_kernel <<< (Nst + 127) / 128, 128 >>>(D.random_d, D.elementsP_d, D.elementsI_d, D.elementsC_d, D.Nst, D.N_h[0], D.Msun_d, D.elementsM_d, ittv, D.Nstart, D.P.mcmcNE, 3);
		
	if(ittv <= 1) D.modifyElementsCall(ittv, 0); //initialize ensemble walkers
	else D.modifyElementsCall(ittv, 3);
  #if def_TTV == 1 
	HelioToDemo_kernel <<< (D.Nst + 127) / 128, 128 >>> (D.x4_d, D.v4_d, D.NBS_d, D.Msun_h[0].x, D.Nst, D.N_h[0]);
  #endif
  #if def_TTV == 2  
	HelioToBary_kernel <<< (D.Nst + 127) / 128, 128 >>> (D.x4_d, D.v4_d, D.NBS_d, D.Msun_h[0].x, D.Nst, D.N_h[0]);
  #endif
 #endif
 #if MCMC_BLOCK == 4
//sigma_kernel <<< 1, D.N_h[0] >>> (D.elementsAOld_d, D.elementsBOld_d, D.elementsLA_d, D.elementsLB_d, D.time_h[0] - D.dt_h[0] / dayUnit, D.Msun_h[0].x, D.N_h[0], D.Nst);
  #if MCMC_Q == 1
if(ittv % 16 == 0){
  #elif MCMC_Q == 2
if(ittv % MCMC_NQ == 0){
  #else
{
  #endif
	setJ_kernel <<< (Nst + 127) / 128, 128 >>>(D.random_d, D.elementsP_d, D.elementsI_d, D.elementsC_d, D.Nst, D.N_h[0], D.Msun_d, D.elementsM_d, ittv, D.Nstart, D.P.mcmcNE, 4);
}
	cudaMemcpy(D.Msun_h, D.Msun_d, sizeof(double4) * D.Nst, cudaMemcpyDeviceToHost);
		
	if(ittv <= 0) D.modifyElementsCall(ittv, 0); //initialize ensemble walkers
	else D.modifyElementsCall(ittv, 4);
  #if def_TTV == 1 
	HelioToDemo_kernel <<< (D.Nst + 127) / 128, 128 >>> (D.x4_d, D.v4_d, D.NBS_d, D.Msun_h[0].x, D.Nst, D.N_h[0]);
  #endif
  #if def_TTV == 2  
	HelioToBary_kernel <<< (D.Nst + 127) / 128, 128 >>> (D.x4_d, D.v4_d, D.NBS_d, D.Msun_h[0].x, D.Nst, D.N_h[0]);
  #endif
//use the following output for longterm stability runs (from MCMC)
//printf("----- %d %d %d\n", D.NconstT, D.Nst, D.N_h[0]);
//cudaMemcpy(D.x4_h, D.x4_d, sizeof(double4) * D.NconstT, cudaMemcpyDeviceToHost);
//cudaMemcpy(D.v4_h, D.v4_d, sizeof(double4) * D.NconstT, cudaMemcpyDeviceToHost);
//for(int i = 0; i < D.Nst * D.N_h[0]; ++i){
//printf("%d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g\n", i % D.N_h[0], D.x4_h[i].w, D.v4_h[i].w, D.x4_h[i].x, D.x4_h[i].y, D.x4_h[i].z, D.v4_h[i].x, D.v4_h[i].y, D.v4_h[i].z);
//}
//printf("-----\n");
 #endif
	cudaMemcpy(D.elementsA_h, D.elementsA_d, sizeof(double4) * D.NconstT, cudaMemcpyDeviceToHost);
#endif

	er = D.beforeTimeStepLoop(ittv);
	if(er == 0) return 0;

	// ************************************************************************
	// ************************************************************************
	// start time step loop here
	for(D.timeStep = D.P.tRestart + 1; D.timeStep <= D.P.deltaT; ++D.timeStep){
		er = D.timeStepLoop(interrupted);
		if(er == 0){
#if def_TTV > 0
			D.printMCMC(1);
#endif
			return 0;
		}
	} // end of time step loop
	// ***********************************************************************
	// ***********************************************************************



#if def_RV > 0
	cudaMemcpy(D.NRVT_h, D.NRVT_d, D.Nst * sizeof(int2), cudaMemcpyDeviceToHost);
	for(int i = 0; i < D.Nst; ++i){
		if(D.NRVT_h[i].x >= def_NRVMax){
			printf("Error: more RV data than def_NRVMax: %d %d %d\n", i, D.NRVT_h[i].x, def_NRVMax);
			return 0;
		} 
	}
#endif
#if def_TTV > 0
	cudaMemcpy(D.NtransitsT_h, D.NtransitsT_d, D.NconstT * sizeof(int2), cudaMemcpyDeviceToHost);
	for(int i = 0; i < D.NT; ++i){
		if(D.NtransitsT_h[i].x > def_NtransitTimeMax){
			printf("Error: more transits than def_NtransitTimeMax for object %d: %d %d\n", i, D.NtransitsT_h[i].x, def_NtransitTimeMax);
			return 0;
		} 
	}

	TTVstep <<< (D.NT + 255) / 256, 256 >>> (D.TransitTime_d, D.TransitTimeObs_d, D.NtransitsT_d, D.NtransitsTObs_d, D.N_d, D.elementsT_d, D.NT, ittv, D.Nstart);
 #if MCMC_Q == 2
	if(ittv % MCMC_NQ == 6){
	//	TTVstepRefine <<< (D.NT + 255) / 256, 256 >>> (D.TransitTime_d, D.TransitTimeObs_d, D.NtransitsT_d, D.NtransitsTObs_d, D.N_d, D.elementsT_d, D.NT, ittv, D.Nstart);
	}
	if(ittv % MCMC_NQ == MCMC_NQ - 1){
 #else
	{
 #endif

 #if def_RV == 1
	RVstep <<< (D.Nst + 127) / 128, 128 >>> (D.RV_d, D.RVObs_d, D.NRVT_d, D.RVP_d, D.Nst);
 #endif
		TTVstep1 < HCM_Bl, HCM_Bl2, NmaxM > <<< (D.NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (D.index_d, D.TransitTime_d, D.RVP_d, D.elementsA_d, D.elementsB_d, D.elementsT_d, D.elementsAOld_d, D.elementsBOld_d, D.elementsTOld_d, D.elementsP_d, D.elementsSA_d, D.elementsC_d, D.NtransitsT_d, D.Msun_d, D.elementsM_d, D.NT, D.N_h[0], D.Nst, ittv, D.Nstart, D.P.mcmcNE);
	}
 #if MCMC_Q == 1
	if(ittv % 16 == 15){
 #elif MCMC_Q == 2
	if(ittv % MCMC_NQ == MCMC_NQ - 1){
 #else 
	{
 #endif
		if(D.P.PrintMCMC > 0){
			D.printMCMC(0);
		}
	}
	if(D.P.PrintTransits == 1){
		er = D.printTransits();
		if(er <= 0) return 0;
	}
	if(D.P.PrintRV == 1){
		er = D.printRV();
		if(er <= 0) return 0;
	}
	if(D.P.PrintRV == 2){
		er = D.printRV2();
		if(er <= 0) return 0;
	}

	}//end of TTV loop
 #if MCMC_Q == 1
	if(ittv % 16 == 15){
 #elif MCMC_Q == 2
	if(ittv % MCMC_NQ == MCMC_NQ - 1){
 #else 
	{
 #endif
		D.printMCMC(1);
printf("print MCMC");
	}
#endif

	er = D.Remaining();
	if(er == 0) return 0;

	return 0; 
}

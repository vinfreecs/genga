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
	#include "TTVStep.h"
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


	cudaError_t error;

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

	//Allocate orbit data on Host and Device
	D.AllocateOrbitt();

	 //allocate mapped memory//
	er = D.CMallocateOrbit();
	if(er == 0) return 0;

	//Allocate aeGride
	D.constantCopy2();
	if(D.P.UseaeGrid == 1){
		er = D.GridaeAlloc();
		if(er == 0) return 0;
	}
	if(D.P.Usegas == 1){
		D.GasAlloc();
	}

	//Table for fastfg//
	er = D.FGAlloc();
	if(er == 0) return 0;

	//initialize memory//
	er = D.init();
	printf("\nInitialize Memory\n");	

	cudaDeviceSynchronize();
	//read initial conditions//
	printf("\nRead Initial Conditions\n");
	er = D.ic();
	if(er == 0) return 0;
	printf("Initial Conditions OK\n");

#if USE_NAF == 1
	er = D.naf.alloc1(D.NT, D.N_h[0], D.Nsmall_h[0], D.Nst, D.P.tRestart, D.idt_h, D.ict_h, D.P.NAFn0, D.P.NAFnfreqs);
	if(er == 0) return 0;

	er = D.naf.alloc2(D.NT, D.N_h[0], D.Nsmall_h[0], D.Nst, D.GSF, D.P.NAFformat, D.P.tRestart, D.index_h);
	if(er == 0) return 0;
#endif

	//remove ghost particles and reorder arrays//
	int NminFlag = D.remove();

	//remove stopped simulations//
	if(NminFlag == 1){
		D.stopSimulations();
		NminFlag = 0;
	}

	if(D.P.UseTestParticles == 2) D.P.MinMass = 0.0;
	cudaDeviceSynchronize();
	error = cudaGetLastError();
	if(error != 0){
		fprintf(D.masterfile, "Start1 error = %d = %s\n",error, cudaGetErrorString(error));
		printf("Start1 error = %d = %s\n",error, cudaGetErrorString(error));
		return 0;
	}
	printf("Compute initial Energy\n");

	er = D.firstEnergy();
	if(er == 0) return 0;

	cudaDeviceSynchronize();

	printf("Write initial Energy\n");

	//write first output
	er = D.firstoutput();
	if(er == 0) return 0;
	printf("Energy OK\n");
	
	//read aeGrid at restart time step 
	if(D.P.UseaeGrid == 1){
		D.readGridae();	
	}

	//Set Gas Disc and Gas Table
	if(D.P.Usegas == 1){
		printf("Set Gas Table\n");
		er = D.setGasDisk();
		if(er == 0) return 0;
		printf("Gas Table OK\n");
	}

	// Set Order and Coefficients of the symplectic integrator //
	D.SymplecticP(0);

	cudaDeviceSynchronize();
	cudaMemset(D.Energy_d, 0, D.NEnergyT*sizeof(double));
	if(D.Nst == 1) printf("Start integration with %d simulation\n", D.Nst);
	else printf("Start integration with %d simulations\n", D.Nst);
	error = cudaGetLastError();
	if(error != 0){
		fprintf(D.masterfile, "Start2 error = %d = %s\n",error, cudaGetErrorString(error));
		printf("Start2 error = %d = %s\n",error, cudaGetErrorString(error));
		return 0;
	}

	fflush(D.masterfile);
#if USE_NAF == 1
	//compute the x and y arrays for the naf algorithm
	int NAFstep = 0;
	D.naf.getnafvarsCall(D.x4_d, D.v4_d, D.index_d, D.NBS_d, D.vcom_d, D.test_d, D.P.NAFvars, D.naf.x_d, D.naf.y_d, D.Msun_d, D.Msun_h[0].x, D.NT, D.Nst, D.naf.n, NAFstep, D.NB[0], D.N_h[0], D.Nsmall_h[0], D.P.UseTestParticles);
	++NAFstep;
#endif
	int ittv = 0;
#if def_TTV == 1
	cudaMemset(D.NtransitsT_d, 0, D.NconstT * sizeof(int2));
#endif
	D.Nstart = 0;
#if MCMC_BLOCK == 3		
	if(D.Nst > 1) D.NT /= 2;
#endif
#if def_TTV > 0
	SetTTVP <<< (Nst + 255) / 256, 256 >>> (D.elementsP_d, D.elementsSA_d, D.Nst);
	cudaMemset(D.NtransitsT_d, 0, D.NconstT * sizeof(int2));

for(ittv = 0; ittv < D.P.TransitSteps; ++ittv){
	D.time_h[0] = (D.P.tRestart + 1) * D.idt_h[0] + D.ict_h[0] * 365.25;
	cudaMemset(D.Nencpairs_d, 0, (D.Nst + 1) * sizeof(int));
	cudaDeviceSynchronize();
#if MCMC_BLOCK == 3		
	if (D.Nst > 1) D.Nstart = (ittv % 2) * D.NT;
if(ittv % 2 == 1) printf("*********** TTV Step %d *********** %d\n", ittv, D.Nstart);
#else
printf("*********** TTV Step %d *********** %d\n", ittv, D.Nstart);
#endif
	if(ittv > 0){

#if MCMC_BLOCK == 0
		int L = 10;
		if(Nst / MCMC_NT > 1) L = 1;
		if(ittv % L == 0){
			SetTTVPRate <<< (MCMC_NT + 255) / 256, 256 >>> (D.elementsP_d, D.elementsC_d, D.elementsSA_d, D.Nst, L);
		}
#endif
#if MCMC_BLOCK == 1
		int L =D.P.mcmcNE * 10;
		if(Nst / MCMC_NT > 1) L = D.P.mcmcNE;
		if(ittv % L == 0){
			SetTTVPRate2 <<< (D.NT + 255) / 256, 256 >>> (D.elementsSA_d, D.elementsLA_d, D.elementsLB_d, D.elementsCA_d, D.elementsCB_d, D.NT, D.N_h[0], Nst, L, ittv, D.P.mcmcNE);
			cudaMemset(D.elementsCA_d, 0, D.NconstT * sizeof(int4));
			cudaMemset(D.elementsCB_d, 0, D.NconstT * sizeof(int4));
		}
#endif
#if MCMC_BLOCK == 2
		int L = D.N_h[0] * D.P.mcmcNE * 10;
		if(Nst / MCMC_NT > 1) L = D.N_h[0] * D.P.mcmcNE;
		if(ittv % L == 0){
			SetTTVPRate2 <<< (D.NT + 255) / 256, 256 >>> (D.elementsSA_d, D.elementsLA_d, D.elementsLB_d, D.elementsCA_d, D.elementsCB_d, D.NT, D.N_h[0], Nst, L, ittv, D.P.mcmcNE);
			cudaMemset(D.elementsCA_d, 0, D.NconstT * sizeof(int4));
			cudaMemset(D.elementsCB_d, 0, D.NconstT * sizeof(int4));
		}
#endif
#if MCMC_BLOCK < 3
		setJ_kenrnel <<< (Nst + 127) / 128, 128 >>>(D.random_d, D.elementsP_d, D.elementsC_d, D.Nst, D.N_h[0], D.Nstart, MCMC_BLOCK);
if(ittv > 30) TSwap_kernel <<<1, 1 >>> (D.random_d, D.elementsP_d, D.elementsAOld_d, D.elementsBOld_d, D.elementsSA_d, D.N_h[0], D.Nst);
		D.modifyElementsCall(ittv, MCMC_BLOCK);
		HelioToDemo_kernel <<< (D.Nst + 127) / 128, 128 >>> (D.x4_d, D.v4_d, D.NBS_d, D.Msun_h[0].x, D.Nst, D.N_h[0]);
#endif
	}

#if MCMC_BLOCK == 3
	setJ_kenrnel <<< (Nst + 127) / 128, 128 >>>(D.random_d, D.elementsP_d, D.elementsC_d, D.Nst, D.N_h[0], D.Nstart, 3);
		
	if(ittv <= 1) D.modifyElementsCall(ittv, 0); //initialize ensemble walkers
	else D.modifyElementsCall(ittv, 3);
	HelioToDemo_kernel <<< (D.Nst + 127) / 128, 128 >>> (D.x4_d, D.v4_d, D.NBS_d, D.Msun_h[0].x, D.Nst, D.N_h[0]);
#else
	cudaMemcpy(D.elementsA_h, D.elementsA_d, sizeof(double4) * D.NconstT, cudaMemcpyDeviceToHost);
#endif
#endif
	if(D.Nst > 1){
		D.firstKick_M(0);
	}
	else{
		if(D.P.UseTestParticles > 0) D.firstKick_small();
		else{
			switch( D.NB[0] ) {
				case 16: D.firstKick_16();
				break;
				case 32: D.firstKick_32();
				break;
				case 64: D.firstKick_64();
				break;
				case 128: D.firstKick_128();
				break;
				case 256: D.firstKick_256();
				break;
				case 512: D.firstKick_512();
				break;
				case 1024: D.firstKick_1024();
				break;
				case 2048: D.firstKick_2048();
				break;
			}
			if(D.NB[0] > 2048) D.firstKick_largeN();
		}
	}
	cudaDeviceSynchronize();
	error = cudaGetLastError();
	if(error != 0){
		fprintf(D.masterfile, "first kick error = %d = %s\n",error, cudaGetErrorString(error));
		printf("first kick error = %d = %s\n", error, cudaGetErrorString(error));
		return 0;
	}
	else{
		if(ittv == 0) printf("first kick OK\n");
	}

	//Print first informations about close encounter pairs
	if(ittv == 0) D.firstInfo();
	D.setStartTime();

#if poincareFlag == 1
	sprintf(D.poincarefilename, "%sPoincare%s_%.12ld.dat", D.GSF[0].path, D.GSF[0].X, 0);
	D.poincarefile = fopen(D.poincarefilename, "w");
#endif

	D.irrTimeStep = 0;
	D.irrTimeStepOut = 0;
	if(D.P.IrregularOutputs == 1){
		er = D.readIrregularOutputs();
		if(er == 0){
			return 0;
		}
		//skip Irregular output times which are before the simulation starts
		double starttime = (D.P.tRestart) * D.idt_h[0] + D.ict_h[0] * 365.25;
		for(int i = 0; i < D.NIrrOutputs; ++i){
			if(D.IrrOutputs[i] >= starttime){
				break;
			}
			++D.irrTimeStep;
			++D.irrTimeStepOut;
		}
	}
	D.TransitDataStep = 0;
	if(D.P.UseTransits == 1){
		er = D.readTransits(D.elementsA_h);
		if(er == 0){
			return 0;
		}
		//skip transit times which are before the simulation starts
		double starttime = (D.P.tRestart + 1) * D.idt_h[0] + D.ict_h[0] * 365.25;
		for(int i = 0; i < D.NTransitData; ++i){
			if(D.TransitData[i].y + D.TransitMaxError >= starttime){
				break;
			}
			++D.TransitDataStep;
		}

	}
	if(D.P.setElements == 1){
		er = D.readSetElements();
		if(er == 0){
			return 0;
		}
	}
	if(D.P.Usegas == 2){
		er = D.readGasFile();
		er = D.readGasFile2(D.time_h[0] / 365.25);
		if(er == 0){
			return 0;
		}
	}


	D.bufferCount = 0;
	D.bufferCountIrr = 0;
	D.MultiSim = 0;
	if(D.Nst > 1) D.MultiSim = 1;
	D.interrupt = 0;
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
	//write out the remaining buffer
	if(D.P.IrregularOutputs == 1){
		if(D.bufferCountIrr > 1){
			D.P.Buffer = D.bufferCountIrr - 1;
			D.CoordinateOutputBuffer(1);
		}
	}
	if(D.bufferCount > 1){
		D.P.Buffer = D.bufferCount - 1;
		D.CoordinateOutputBuffer(0);
	}

#if poincareFlag == 1
	fclose(D.poincarefile);
#endif

#if def_TTV > 0
	cudaMemcpy(D.NtransitsT_h, D.NtransitsT_d, D.NconstT * sizeof(int2), cudaMemcpyDeviceToHost);
	for(int i = 0; i < D.NT; ++i){
		if(D.NtransitsT_h[i].x > def_NtransitTimeMax){
			printf("Error: more transits than def_NtransitTimeMax for object %d: %d %d\n", i, D.NtransitsT_h[i].x, def_NtransitTimeMax);
			return 0;
		} 
	}


	TTVstep <<< (D.NT + 255) / 256, 256 >>> (D.TransitTime_d, D.TransitTimeObs_d, D.NtransitsT_d, D.NtransitsTObs_d, D.NT, ittv, D.Nstart);
	TTVstep1 < HCM_Bl, HCM_Bl2, NmaxM > <<< (D.NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (D.index_d, D.TransitTime_d, D.elementsA_d, D.elementsB_d, D.elementsAOld_d, D.elementsBOld_d, D.elementsCA_d, D.elementsCB_d, D.elementsP_d, D.elementsSA_d, D.elementsC_d, D.NtransitsT_d, D.NT, D.N_h[0], D.Nst, ittv, D.Nstart, D.P.mcmcNE);


	D.printMCMC(0);
	if(D.P.PrintTransits == 1){
		er = D.printTransits();
		if(er <= 0) return 0;
	}

	}//end of TTV loop
	D.printMCMC(1);
#endif

	//print last informations
	D.printLastTime();
	D.LastInfo();

	//free all the memory on the Host and on the Device
	er = D.freeOrbit();
	if(er == 0) return 0;

	if(D.P.UseaeGrid == 1){
		free(D.Gridaecount_h);
		cudaFree(D.Gridaecount_d);
	}

	if(D.P.Usegas == 1){
		er = D.freeGas();
		if(er == 0) return 0;
	}

#if USE_NAF == 1
	er = D.naf.naffree();
	if(er == 0) return 0;
#endif


	er = D.freeHost();
	if(er == 0) return 0;

	printf("GENGA terminated successfully\n");
	fprintf(H.masterfile, "GENGA terminated successfully\n");

	return 0; 
}

#include "Orbit2.h"
#include "Rcrit.h"
#include "Kick3.h"
#include "HC.h"
#include "FG2.h"
#include "Encounter3.h"
#include "BSB.h"
//#include "BSB2.h"
#include "BSBM.h"
#include "BSB64M.h"
#include "ComEnergy.h"
#include "force.h"
#include "forceYarkovskyOld.h"
#include "Kick4.h"
#include "BSA.h"
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

#if G3 > 0
	#include "BSBG3.h"
#endif

int SIn;		//Number of direction steps
int SIM;		//half of steps
double *Ct;		//time factor for HC Kick steps
double *FGt;		//time factor for Drift steps
double *Kt;		//time factor for Kick steps

int EjectionFlag2 = 0;
int StopAtEncounterFlag2 = 0;

__host__ void Data::constantCopyDirectAcc(){
	cudaMemcpyToSymbol(Rcut_c, &Rcut_h[0], sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(RcutSun_c, &RcutSun_h[0], sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(StopAtCollision_c, &P.StopAtCollision, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(StopMinMass_c, &P.StopMinMass, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(CollisionPrecision_c, &P.CollisionPrecision, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(CollTshift_c, &P.CollTshift, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(WriteEncounters_c, &P.WriteEncounters, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(WriteEncountersRadius_c, &P.WriteEncountersRadius, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(StopAtEncounter_c, &P.StopAtEncounter, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(StopAtEncounterRadius_c, &P.StopAtEncounterRadius, sizeof(double), 0, cudaMemcpyHostToDevice);

	int2 ij;
	ij.x = -1;
	ij.y = -1;
	cudaMemcpyToSymbol(CollTshiftpairs_c, &ij, sizeof(int2), 0, cudaMemcpyHostToDevice);
}



// *****************************************************
// This function calls all necessary steps before the mcmc step loop

// Authors: Simon Grimm
// February 2019
// ****************************************************
__host__ int Data::beforeTimeStepLoop1(){

	int er;
	cudaEventCreate(&KickEvent);	
	cudaStreamCreateWithFlags(&copyStream, cudaStreamNonBlocking);

	//Allocate orbit data on Host and Device
	AllocateOrbit();

	//allocate mapped memory//
	er = CMallocateOrbit();
	if(er == 0) return 0;

	//copy constant memory
	constantCopyDirectAcc();

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
	cudaMemset(Energy_d, 0, NEnergyT*sizeof(double));

	//Tune kernel parameters
	if(Nst == 1){
		er = tuneFG(FTX);
		if(er == 0) return 0;
		er = tuneRcrit(RTX);
		if(er == 0) return 0;
		UseAcc = 1;
		
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
	}

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


	//Print first informations about close encounter pairs
	if(ittv == 0) firstInfo();
	setStartTime();

#if poincareFlag == 1
	sprintf(poincarefilename, "%sPoincare%s_%.*lld.dat", GSF[0].path, GSF[0].X, def_NFileNameDigits, 0);
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

	//cudaEventSynchronize(KickEvent);
	//do not synchronize here, to save time. but that means that the error message could be delayed
	//Check for too many encounters
	if(EncFlag_m[0] > 0){
		printf("Error: more encounters than allowed. %d %d\n", EncFlag_m[0], P.NencMax);
		fprintf(masterfile, "Error: more encounters than allowed. %d %d\n", EncFlag_m[0], P.NencMax);
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
	if((P.ei > 0 && timeStep % P.ei == 0) || interrupt == 1 || (P.ei < 0 && timeStep == P.deltaT)){
		if(bufferCount + 1 >= P.Buffer || interrupt == 1 || (P.ei < 0 && timeStep == P.deltaT)){
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
	if((P.ci > 0 && ((timeStep - 1) % P.ci >= P.ci - P.nci)) || interrupt == 1 || (P.ci < 0 && timeStep == P.deltaT)){
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
		
#if poincareFlag == 1
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
	cudaEventDestroy(KickEvent);	
	cudaStreamDestroy(copyStream);	
	

	//write out the remaining buffer
	if(P.IrregularOutputs == 1){
		if(bufferCountIrr > 1){
			CoordinateOutputBuffer(1);
		}
	}
	if(bufferCount > 0){
		CoordinateOutputBuffer(0);
	}

#if poincareFlag == 1
	fclose(D.poincarefile);
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
__host__ void  Data::SymplecticP(int E){
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
// This function set the time factors for an irregular output step
// dTau is the modified time step
// Authors: Simon Grimm
// June 2015
// ****************************************************
__host__ void  Data::IrregularStep(double dTau){
	SIn = 1;
	SIM = 1;
	
	FGt[0] = dTau;
	Ct[0] = dTau * 0.5;
	Kt[0] = dTau * 0.5;
	
}

// **************************************
// This kernel sets initial values for the Encouter pair arrays
//Authors: Simon Grimm, Joachim Stadel
//March 2016
// **************************************3
template <int Bl>
__global__ void initial_kernel(double *K_d, double *Kold_d, double4 *StopTime_d, int *groupIndex_d, int NB){
	int idy = threadIdx.x;
	int idx = blockIdx.x;
	
	for(int i = 0; i < NB; i += Bl){
		K_d[(idy +i)* NB + idx] = 1.0;
		Kold_d[(idy +i)* NB + idx] = 1.0;
		StopTime_d[(idy +i)* NB + idx].x = -1.0;
		StopTime_d[(idy +i)* NB + idx].y = -1.0;
		StopTime_d[(idy +i)* NB + idx].z = -1.0;
		StopTime_d[(idy +i)* NB + idx].w = -1.0;
		if(idx == 0){
			groupIndex_d[idy + i] = -1;
		}
	}
}

// **************************************
__global__ void initialb_kernel(int2 *Encpairs_d, int2 *Encpairs2_d, int NBNencT){
	
	int idx = blockIdx.x;
	int id = blockIdx.x * blockDim.x + idx;
	
	if(id < NBNencT){
		Encpairs_d[id].x = -1;
		Encpairs_d[id].y = -1;
		
		Encpairs2_d[id].x = -1;
		Encpairs2_d[id].y = -1;
	}
}

/*
 __global__ void test_kernel(double4 *x4_d, double3 *a_d, int *index_d, int N){
 
	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;
 
	if(id < N && fabs(a_d[id].x) > 10) printf("test %d %.g\n", id, a_d[id].x);
 
 }
 */




__global__ void safe_kernel(double4 *x4_d, double4 *v4_d, double4 *x4bb_d, double4 *v4bb_d, double3 *spin_d, double3 *spinbb_d, double *rcrit_d, double *rcritv_d, double *rcritbb_d, double *rcritvbb_d, int *index_d, int *indexbb_d, int N, int NconstT, int SLevels, int f){
	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;


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
		fg_kernel <<< (NN + tx - 1) / tx, tx >>>(x4_d, v4_d, xold_d, vold_d, index_d, dt_h[0] * FGt[0], Msun_h[0].x, test_d, NN, aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, -1, P.UseForce);
		
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
		printf("tx:%d    \ttime:%.15f s\n", tx, times * 0.001);	//time in seconds
		fprintf(GSF[0].logfile,"\ttx:%d    \ttime:%.15f s\n", tx, times * 0.001);	//time in seconds
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
	printf("Best parameters: tx:%d\t time:%.15f s\n", TX, timesMin * 0.001);	//time in seconds
	fprintf(GSF[0].logfile, "Best parameters: tx:%d\t time:%.15f s\n", TX, timesMin * 0.001);	//time in seconds
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
		Rcrit_kernel <<< (NN + tx - 1) / tx, tx >>> (x4_d, v4_d, x4b_d, v4b_d, spin_d, spinb_d, Msun_h[0].x, rcrit_d, rcritb_d, rcritv_d, rcritvb_d, index_d, indexb_d, dt_h[0], test_d, n1_h[0], n2_h[0], time_d, time_h[0], EjectionFlag_d, NN, NconstT, P.SLevels, 0);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&times, start, stop); //time in microseconds
		error = cudaGetLastError();
		if(error != 0){
			fprintf(masterfile, "Rcrit Tune error = %d = %s\n",error, cudaGetErrorString(error));
			printf("Rcrit Tune error = %d = %s\n",error, cudaGetErrorString(error));
			return 0;
		}
		printf("tx:%d    \ttime:%.15f s\n", tx, times * 0.001);	//time in seconds
		fprintf(GSF[0].logfile,"\ttx:%d    \ttime:%.15f s\n", tx, times * 0.001);	//time in seconds
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
	printf("Best parameters: tx:%d\t time:%.15f s\n", TX, timesMin * 0.001);	//time in seconds
	fprintf(GSF[0].logfile, "Best parameters: tx:%d\t time:%.15f s\n", TX, timesMin * 0.001);	//time in seconds
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	fclose(GSF[0].logfile);
	return 1;
}

__host__ int Data::tuneForce(int &TX){

	int ttx[4] = {32, 64, 128, 256};

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

	for(int i = 0; i < 4; ++i){
		int tx = ttx[i];
		cudaEventRecord(start, 0);
		//revert fg operation
		//launch with si = -1
		force <<< (NN + tx - 1) / tx, tx >>>  (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, N_h[0], Nst, P.UseForce, 0, 0);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&times, start, stop); //time in microseconds
		error = cudaGetLastError();
		if(error == 7){
			//skip choices with too many resources requested
			continue;

		}
		if(error != 0 && error != 7){
			fprintf(masterfile, "Force Tune error = %d = %s\n",error, cudaGetErrorString(error));
			printf("Force Tune error = %d = %s\n",error, cudaGetErrorString(error));
			return 0;
		}
		printf("tx:%d    \ttime:%.15f s\n", tx, times * 0.001);	//time in seconds
		fprintf(GSF[0].logfile,"\ttx:%d    \ttime:%.15f s\n", tx, times * 0.001);	//time in seconds
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
	printf("Best parameters: tx:%d\t time:%.15f s\n", TX, timesMin * 0.001);	//time in seconds
	fprintf(GSF[0].logfile, "Best parameters: tx:%d\t time:%.15f s\n", TX, timesMin * 0.001);	//time in seconds
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
		N1 = N_h[0] + Nsmall_h[0];	//TP 1 has to be run before
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
	
	Rcrit_kernel <<< (NN + 255) / 256, 256 >>> (x4_d, v4_d, x4b_d, v4b_d, spin_d, spinb_d, Msun_h[0].x, rcrit_d, rcritb_d, rcritv_d, rcritvb_d, index_d, indexb_d, dt_h[0], test_d, n1_h[0], n2_h[0], time_d, time_h[0], EjectionFlag_d, NN, NconstT, P.SLevels, 0);


	if(EE == 0){
		cudaEventRecord(start, 0);	
		int N4 = (NN + 3) / 4;
		for(int t = 0; t < T; ++t){
			acc4b_kernel < 256 > <<< N4 , 256 >>> (x4_d, v4_d, a_d, rcritv_d, groupIndex_d, N4, NULL /*Encpairsb_d*/, Encpairs2_d, test_d, NN, NconstT, P.NencMax, time_h[0]);
		}
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&times, start, stop); //time in microseconds
		printf("acc4b time:%.15f s\n", times * 0.001);	//time in seconds
		fprintf(GSF[0].logfile, "acc4b time:%.15f s\n", times * 0.001);	//time in seconds


		if(NB[0] <= 32){
			cudaEventRecord(start, 0);	
			for(int t = 0; t < T; ++t){
#if def_KickFloat == 0
				kick16c_kernel < 0 > <<< NN, NB[0] >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, P.NencMax, time_h[0], NN);
#else
				kick16cf_kernel < 0 > <<< NN, NB[0] >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, P.NencMax, time_h[0], NN);
#endif
			}
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&times, start, stop); //time in microseconds
			printf("kick32c time:%.15f s\n", times * 0.001);	//time in seconds
			fprintf(GSF[0].logfile, "kick32c time:%.15f s\n", times * 0.001);	//time in seconds
			kickTime = times;
		}		
		else if(NB[0] <= 1024){
			cudaEventRecord(start, 0);	
			for(int t = 0; t < T; ++t){
#if def_KickFloat == 0
				kick32c_kernel < 0 > <<< NN, NB[0] >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, P.NencMax, time_h[0], NN);
#else
				kick32cf_kernel < 0 > <<< NN, NB[0] >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, P.NencMax, time_h[0], NN);
#endif
			}
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&times, start, stop); //time in microseconds
			printf("kick32c time:%.15f s\n", times * 0.001);	//time in seconds
			fprintf(GSF[0].logfile, "kick32c time:%.15f s\n", times * 0.001);	//time in seconds
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
				EncpairsZeroC <<< (NN + 255) / 256, 256 >>> (Encpairs2_d, a_d, Nencpairs_d, P.NencMax, NN);

				cudaEventRecord(start, 0);	
				for(int t = 0; t < T; ++t){
#if def_KickFloat == 0
					acc4C_kernel <<< dim3( (((NN + p - 1)/ p) + tx - 1) / tx, 1, 1), dim3(tx,ty,1), tx * ty * p * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, NN, N0, N1, P.NencMax, p, EE);
#else
					acc4Cf_kernel <<< dim3( (((NN + p - 1)/ p) + tx - 1) / tx, 1, 1), dim3(tx,ty,1), tx * ty * p * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, NN, N0, N1, P.NencMax, p, EE);
#endif
				}

				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&times, start, stop); //time in microseconds
				error = cudaGetLastError();
				if(error == 7){
					//skip choices with too many resources requested
					continue;

				}
				if(error != 0 && error != 7){
					fprintf(masterfile, "Kick Tune error = %d = %s\n",error, cudaGetErrorString(error));
					printf("Kick Tune error = %d = %s\n",error, cudaGetErrorString(error));
					return 0;
				}
				if(EE < 2){
					printf("p:%d    \ttx:%d    \tty:%d    \ttime:%.15f s\n", p, tx, ty, times * 0.001);	//time in seconds
					fprintf(GSF[0].logfile,"p:%d    \ttx:%d    \tty:%d    \ttime:%.15f s\n", p, tx, ty, times * 0.001);	//time in seconds
				}
				else{
					printf("p2:%d   \ttx2:%d   \tty2:%d   \ttime:%.15f s\n", p, tx, ty, times * 0.001);	//time in seconds
					fprintf(GSF[0].logfile,"p2:%d   \ttx2:%d   \tty2:%d   \ttime:%.15f s\n", p, tx, ty, times * 0.001);	//time in seconds
				}
				if(times < timesMin){
					PP = p;
					TX = tx;
					TY = ty;
				}
				timesMin = fmin(times, timesMin);
				//check if different tunig parameters give the same result
				compare_a_kernel <<< (NN + 255) / 256, 256 >>> (a_d, ab_d, NN, f);
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
		printf("Best parameters: p:%d\t tx:%d\t ty:%d\t time:%.15f s\n", PP, TX, TY, timesMin * 0.001);	//time in seconds
		fprintf(GSF[0].logfile, "Best parameters: p:%d\t tx:%d\t ty:%d\t time:%.15f s\n", PP, TX, TY, timesMin * 0.001);	//time in seconds
	}
	else{
		printf("Best parameters: p2:%d\t tx2:%d\t ty2:%d\t time:%.15f s\n", PP, TX, TY, timesMin * 0.001);	//time in seconds
		fprintf(GSF[0].logfile, "Best parameters: p2:%d\t tx2:%d\t ty2:%d\t time:%.15f s\n", PP, TX, TY, timesMin * 0.001);	//time in seconds
	}

	//test now the total time for the kick operation	
	cudaEventRecord(start, 0);	
	for(int t = 0; t < T; ++t){
	
#if def_KickFloat == 0
		acc4C_kernel <<< dim3( (((NN + PP - 1)/ PP) + TX - 1) / TX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, NN, 0, NN, P.NencMax, KP, 0);
#else
		acc4Cf_kernel <<< dim3( (((NN + PP - 1)/ PP) + TX - 1) / TX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, NN, 0, NN, P.NencMax, KP, 0);
#endif
		kick32Ab_kernel <<< (NN + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs2_d, NN, P.NencMax, 0);
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&times, start, stop); //time in microseconds
	printf("Total acc4C + kick32Ab time:%.15f s\n", times * 0.001);	//time in seconds
	fprintf(GSF[0].logfile, "Total acc4C + kick32Ab time:%.15f s\n", times * 0.001);	//time in seconds
	accTime = times;

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
	

#if SERIAL_GROUPING == 1
	printf("Using Serial Grouping, this disabels the tuning parameters\n");
	fprintf(GSF[0].logfile, "Using Serial Grouping, this disabels the tuning parameters\n");
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

#endif
  	//Set again the Encpairs arrays to zero
	EncpairsZeroC <<< (N_h[0] + Nsmall_h[0] + 255) / 256, 256 >>> (Encpairs2_d, a_d, Nencpairs_d, P.NencMax, N_h[0] + Nsmall_h[0]);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	fclose(GSF[0].logfile);
	return 1;
}


__host__ void Data::firstKick_16(int noColl){
	//use last time information, the beginning of the time step
	double time = (P.tRestart + 1) * idt_h[0] + ict_h[0] * 365.25; //in the set Elements kernel, timestep wil be decreased by 1 again
	cudaMemset(a_d, 0, NconstT * sizeof(double3));
	cudaMemset(ab_d, 0, NconstT * sizeof(double3));
	initialb_kernel <<< (NBNencT + 255) / 256, 256 >>> (Encpairs_d, Encpairs2_d, NBNencT);
	Rcrit_kernel <<< (N_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, x4b_d, v4b_d, spin_d, spinb_d, Msun_h[0].x, rcrit_d, rcritb_d, rcritv_d, rcritvb_d, index_d, indexb_d, dt_h[0], test_d, n1_h[0], n2_h[0], time_d, time, EjectionFlag_d, N_h[0], NconstT, P.SLevels, noColl);
	if(P.setElementsV == 2){ // convert barycentric velocities to heliocentric
		comCall(1);
	}
	if(P.setElementsV > 0) setElements <<< (P.setElementsN + 63) / 64, 64 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst, 0);
	if(P.setElementsV == 2){
		comCall(-1);
	}
#if def_KickFloat == 0
	kick16c_kernel < 0 > <<< N_h[0], 32 >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, P.NencMax, time_h[0], N_h[0]);
#else
	kick16cf_kernel < 0 > <<< N_h[0], 32 >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, P.NencMax, time_h[0], N_h[0]);
#endif
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
}
__host__ void Data::firstKick_largeN(int noColl){
	//use last time information, the beginning of the time step
	double time = (P.tRestart + 1) * idt_h[0] + ict_h[0] * 365.25; //in the set Elements kernel, timestep wil be decreased by 1 again
	cudaMemset(a_d, 0, NconstT * sizeof(double3));
	cudaMemset(ab_d, 0, NconstT * sizeof(double3));
	initialb_kernel <<< (NBNencT + 255) / 256, 256 >>> (Encpairs_d, Encpairs2_d, NBNencT);
	Rcrit_kernel <<< (N_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, x4b_d, v4b_d, spin_d, spinb_d, Msun_h[0].x, rcrit_d, rcritb_d, rcritv_d, rcritvb_d, index_d, indexb_d, dt_h[0], test_d, n1_h[0], n2_h[0], time_d, time, EjectionFlag_d, N_h[0], NconstT, P.SLevels, noColl);
	if(P.setElementsV == 2){ // convert barycentric velocities to heliocentric
		comCall(1);
	}
	if(P.setElementsV > 0) setElements <<< (P.setElementsN + 63) / 64, 64 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst, 0);
	if(P.setElementsV == 2){
		comCall(-1);
	}
	if(UseAcc == 0){
#if def_KickFloat == 0
		kick32c_kernel < 0 > <<< N_h[0] , NB[0] >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, P.NencMax, time_h[0], N_h[0]);
#else
		kick32cf_kernel < 0 > <<< N_h[0] , NB[0] >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, P.NencMax, time_h[0], N_h[0]);
#endif
	}
	else{
#if def_KickFloat == 0
		acc4C_kernel <<< dim3( (((N_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0], 0, N_h[0], P.NencMax, KP, 0);
#else
		acc4Cf_kernel <<< dim3( (((N_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0], 0, N_h[0], P.NencMax, KP, 0);
#endif
	}
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
}
__host__ void Data::firstKick_small(int noColl){
	//use last time information, the beginning of the time step
	double time = (P.tRestart + 1) * idt_h[0] + ict_h[0] * 365.25; //in the set Elements kernel, timestep wil be decreased by 1 again
	cudaMemset(a_d, 0, NconstT*sizeof(double3));
	cudaMemset(ab_d, 0, NconstT * sizeof(double3));
	initialb_kernel <<< (NBNencT + 255) / 256, 256 >>> (Encpairs_d, Encpairs2_d, NBNencT);
	Rcrit_kernel <<< (N_h[0] + Nsmall_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, x4b_d, v4b_d, spin_d, spinb_d, Msun_h[0].x, rcrit_d, rcritb_d, rcritv_d, rcritvb_d, index_d, indexb_d, dt_h[0], test_d, n1_h[0], n2_h[0], time_d, time, EjectionFlag_d, N_h[0] + Nsmall_h[0], NconstT, P.SLevels, noColl);
	if(P.setElementsV == 2){ // convert barycentric velocities to heliocentric
		comCall(1);
	}	
	if(P.setElementsV > 0) setElements <<< (P.setElementsN + 63) / 64, 64 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst, 0);
	if(P.setElementsV == 2){
		comCall(-1);
	}

#if def_KickFloat == 0
	acc4C_kernel <<< dim3( (((N_h[0] + Nsmall_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0] + Nsmall_h[0], 0, N_h[0], P.NencMax, KP, 1);
#else
	acc4Cf_kernel <<< dim3( (((N_h[0] + Nsmall_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0] + Nsmall_h[0], 0, N_h[0], P.NencMax, KP, 1);
#endif
	if(P.UseTestParticles == 2){
#if def_KickFloat == 0
		acc4C_kernel <<< dim3( (((N_h[0] + KP2 - 1)/ KP2) + KTX2 - 1) / KTX2, 1, 1), dim3(KTX2,KTY2,1), KTX2 * KTY2 * KP2 * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0], N_h[0], N_h[0] + Nsmall_h[0], P.NencMax, KP2, 2);
#else
		acc4Cf_kernel <<< dim3( (((N_h[0] + KP2 - 1)/ KP2) + KTX2 - 1) / KTX2, 1, 1), dim3(KTX2,KTY2,1), KTX2 * KTY2 * KP2 * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0], N_h[0], N_h[0] + Nsmall_h[0], P.NencMax, KP2, 2);
#endif
	}
		
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<< (N_h[0] + Nsmall_h[0] + 127) / 128, 128 >>>(Encpairs2_d, N_h[0] + Nsmall_h[0], P.NencMax);
#endif
}

__host__ void Data::firstKick_M(long long ts, int noColl){
	cudaMemset(a_d, 0, NconstT*sizeof(double3));
	cudaMemset(ab_d, 0, NconstT * sizeof(double3));
	initialb_kernel <<< (NBNencT + 255) / 256, 256 >>> (Encpairs_d, Encpairs2_d, NBNencT);
	RcritM_kernel <<< (NT + 31) / 32, 32>>> (x4_d, v4_d, x4b_d, v4b_d, spin_d, spinb_d, Msun_d, rcrit_d, rcritb_d, rcritv_d, rcritvb_d, dt_d, test_d, n1_d, n2_d, Rcut_d, RcutSun_d, EjectionFlag_d, index_d, indexb_d, Nst, NT, time_d, idt_d, ict_d, delta_d, ts, StopFlag_d, NconstT, P.SLevels, noColl, Nstart);
	KickM2_kernel < KM_Bl, KM_Bl2, NmaxM, 0 > <<< (NT + KM_Bl2 - 1) / KM_Bl2, KM_Bl >>> (x4_d, v4_d, a_d, rcritv_d, Nencpairs_d, Encpairs_d, dt_d, Kt[SIn - 1] * def_ksq, index_d, NT, Nstart);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
}

cudaStream_t stream[12];
__host__ void Data::BSCall(int si, double time, int noColl, double ll){
	
	time -= dt_h[0] / dayUnit;
	int N = N_h[0] + Nsmall_h[0];
	double dt = dt_h[0] / ll * FGt[si];
	
//printf(" %d | %d %d %d | %d %d %d | %d %d %d | %d %d %d\n", Nenc_m[0], Nenc_m[1], Nenc_m[2], Nenc_m[3], Nenc_m[4], Nenc_m[5], Nenc_m[6], Nenc_m[7], Nenc_m[8], Nenc_m[9],  Nenc_m[10],  Nenc_m[11],  Nenc_m[12]);
	for(int st = 0; st < 9; ++st)   cudaStreamCreate(&stream[st]);
	
	//if(Nenc_m[1] > 0) BSB2Step_kernel <2 * 16, 2> <<< (Nenc_m[1] + 15)/ 16, 2 * 16, 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 0, index_d, Nenc_m[1], Ncoll_d, Coll_d, time, spin_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, N, NconstT, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	//if(Nenc_m[1] > 0) BSB2Step_kernel <2, 2> <<< Nenc_m[1], 2, 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 0, index_d, Nenc_m[1], Ncoll_d, Coll_d, time, spin_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, N, NconstT, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	
	if(Nenc_m[1] > 0) BSBStep_kernel <2, 2> <<< Nenc_m[1], 4, 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 0, index_d, Ncoll_d, Coll_d, time, spin_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, N, NconstT, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	
	
	if(Nenc_m[2] > 0) BSBStep_kernel <4, 4> <<< Nenc_m[2], 16, 0, stream[1] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 1, index_d, Ncoll_d, Coll_d, time, spin_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, N, NconstT, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[3] > 0) BSBStep_kernel <8, 8> <<< Nenc_m[3], 64, 0, stream[2] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 2, index_d, Ncoll_d, Coll_d, time, spin_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, N, NconstT, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[4] > 0) BSBStep_kernel <16, 16> <<< Nenc_m[4], 256, 0, stream[3] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 3, index_d, Ncoll_d, Coll_d, time, spin_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, N, NconstT, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[5] > 0) BSBStep_kernel <32, 8> <<< Nenc_m[5], 256, 0, stream[4] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 4, index_d, Ncoll_d, Coll_d, time, spin_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, N, NconstT, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	
/*
	if(Nenc_m[1] > 0) BSA_kernel < 2 > <<< Nenc_m[1], 2 , 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 0, N, NconstT, P.NencMax, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[2] > 0) BSA_kernel < 4 > <<< Nenc_m[2], 4 , 0, stream[1] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 1, N, NconstT, P.NencMax, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[3] > 0) BSA_kernel < 8 > <<< Nenc_m[3], 8 , 0, stream[2] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 2, N, NconstT, P.NencMax, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[4] > 0) BSA_kernel < 16 > <<< Nenc_m[4], 16 , 0, stream[3] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 3, N, NconstT, P.NencMax, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[5] > 0) BSA_kernel < 32 > <<< Nenc_m[5], 32 , 0, stream[4] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 4, N, NconstT, P.NencMax, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
*/
	
	if(Nenc_m[6] > 0) BSA_kernel < 64 > <<< Nenc_m[6], 64 , 0, stream[5] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 5, N, NconstT, P.NencMax, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[7] > 0) BSA_kernel < 128 > <<< Nenc_m[7], 128 , 0, stream[6] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 6, N, NconstT, P.NencMax, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[8] > 0) BSA_kernel < 256 > <<< Nenc_m[8], 256 , 0, stream[7] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dt, Msun_h[0].x, U_d, 7, N, NconstT, P.NencMax, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, P.SLevels, noColl);


	/*	
	if(Nenc_m[1] > 0) BSA512_kernel < 2, 2 > <<< Nenc_m[1], 2 , 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dx_d, dv_d, dt, Msun_h[0].x, U_d, 0, N, NconstT, P.NencMax, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, dtgr_d, P.UseForce, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[2] > 0) BSA512_kernel < 4, 4 > <<< Nenc_m[2], 4 , 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dx_d, dv_d, dt, Msun_h[0].x, U_d, 1, N, NconstT,P.NencMax, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, dtgr_d, P.UseForce, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[3] > 0) BSA512_kernel < 8, 8 > <<< Nenc_m[3], 8 , 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dx_d, dv_d, dt, Msun_h[0].x, U_d, 2, N, NconstT. P.NencMax, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, dtgr_d, P.UseForce, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[4] > 0) BSA512_kernel < 16, 16 > <<< Nenc_m[4], 16 , 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dx_d, dv_d, dt, Msun_h[0].x, U_d, 3, N, NconstT, P.NencMax, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, dtgr_d, P.UseForce, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[5] > 0) BSA512_kernel < 32, 32 > <<< Nenc_m[5], 32 , 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dx_d, dv_d, dt, Msun_h[0].x, U_d, 4, N, NconstT, P.NencMax, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, dtgr_d, P.UseForce, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[6] > 0) BSA512_kernel < 64, 64 > <<< Nenc_m[6], 64 , 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dx_d, dv_d, dt, Msun_h[0].x, U_d, 5, N, NconstT, P.NencMax, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, dtgr_d, P.UseForce, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[7] > 0) BSA512_kernel < 128, 128 > <<< Nenc_m[7], 128 , 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dx_d, dv_d, dt, Msun_h[0].x, U_d, 6, N, NconstT, P.NencMax, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, dtgr_d, P.UseForce, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[8] > 0) BSA512_kernel < 256, 256 > <<< Nenc_m[8], 256 , 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dx_d, dv_d, dt, Msun_h[0].x, U_d, 7, N, NconstT, P.NencMax, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, dtgr_d, P.UseForce, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	*/
	
	//if(Nenc_m[9] > 0) BSA512_kernel < 512, 512 > <<< Nenc_m[9], 512 , 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dx_d, dv_d, dt, Msun_h[0].x, U_d, 8, N, NconstT, P.NencMax, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, dtgr_d, P.UseForce, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	

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
	
	for(int st = 0; st < 9; ++st) cudaStreamDestroy(stream[st]);

	//make sure that Nenc_m is updated and available on host before continue
	cudaDeviceSynchronize();
	//for(int i = 0; i < def_GMax; ++i){
	//	Nenc_m[i] = 0;
	//}
}

__host__ void Data::BSBMCall(int si, int noColl, double ll){
	for(int st = 0; st < 6; ++st)  cudaStreamCreate(&stream[st]);
	if(Nenc_m[1] > 0) BSBMStep_kernel <2, 2> <<< Nenc_m[1], 4, 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs_d, Encpairs2_d, dt_d, FGt[si] / ll, Msun_d, U_d, 0, index_d, Ncoll_d, Coll_d, time_d, spin_d, Nst, NconstT, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[2] > 0) BSBMStep_kernel <4, 4> <<< Nenc_m[2], 16, 0, stream[1] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs_d, Encpairs2_d, dt_d, FGt[si] / ll, Msun_d, U_d, 1, index_d, Ncoll_d, Coll_d, time_d, spin_d, Nst, NconstT, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[3] > 0) BSBMStep_kernel <8, 8> <<< Nenc_m[3], 64, 0, stream[2] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs_d, Encpairs2_d, dt_d, FGt[si] / ll, Msun_d, U_d, 2, index_d, Ncoll_d, Coll_d, time_d, spin_d, Nst, NconstT, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[4] > 0) BSBMStep_kernel <16, 16> <<< Nenc_m[4], 256, 0, stream[3] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs_d, Encpairs2_d, dt_d, FGt[si] / ll, Msun_d, U_d, 3, index_d, Ncoll_d, Coll_d, time_d, spin_d, Nst, NconstT, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	if(Nenc_m[5] > 0) BSBMStep_kernel <32, 8> <<< Nenc_m[5], 256, 0, stream[4] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs_d, Encpairs2_d, dt_d, FGt[si] / ll, Msun_d, U_d, 4, index_d, Ncoll_d, Coll_d, time_d, spin_d, Nst, NconstT, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	//if(Nenc_m[6] > 0) BSBMStep64_kernel <64, 4> <<< Nenc_m[6], 256, 0, stream[5] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs_d, Encpairs2_d, dt_d, FGt[si] / ll, Msun_d, U_d, 5, index_d, Ncoll_d, Coll_d, time_d, spin_d, Nst, NconstT, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d. P.UseForce, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
	for(int st = 0; st < 6; ++st)  cudaStreamDestroy(stream[st]);
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
		
printf("safe  1\n");
		safe_kernel <<< (N_h[0] + Nsmall_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, x4bb_d, v4bb_d, spin_d, spinbb_d, rcrit_d, rcritv_d, rcritbb_d, rcritvbb_d, index_d, indexbb_d, N_h[0] + Nsmall_h[0], NconstT, P.SLevels, 1);
		
		if(P.CollTshift > 1.0){
			//restore old step and increase radius 
			int NOld = Ncoll_m[0];
			for(int i = 0; i < NOld; ++i){

printf("Backup step 1 %d %.20g %.20g %.20g\n", i, Coll_h[i * def_NColl] * 365.25, time_h[0] - idt_h[0], (Coll_h[i * def_NColl] * 365.25 - (time_h[0] - idt_h[0])) / idt_h[0]);
				int2 ij;
				ij.x = int(Coll_h[i * def_NColl + 1]);
				ij.y = int(Coll_h[i * def_NColl + 13]);
				cudaMemcpyToSymbol(CollTshiftpairs_c, &ij, sizeof(int2), 0, cudaMemcpyHostToDevice);
				//IrregularStep(1.0 * ((Coll_h[i * def_NColl] * 365.25 - time_h[0] + idt_h[0]) / idt_h[0]));
			IrregularStep(1.0);
				int N0 = Ncoll_m[0];
				bStep(1);
				cudaDeviceSynchronize();
				if(N0 == Ncoll_m[0]){
printf("Revert time step\n");
					IrregularStep(-1.0);
printf("Backup step -1 %.20g %.20g %.20g\n", (time_h[0] - idt_h[0]) - idt_h[0], time_h[0] - idt_h[0], -1.0);
					N0 = Ncoll_m[0];
					bStep(1);
					cudaDeviceSynchronize();
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
				Coltime = fmin(Coltime, Coll_h[i * def_NColl]);
			}
printf("Backup step 2 %.20g %.20g %.20g\n", Coltime * 365.25, time_h[0] - idt_h[0], (Coltime * 365.25 - (time_h[0] - idt_h[0])) / idt_h[0]);

			IrregularStep(1.0 * ((Coltime * 365.25 - time_h[0] + idt_h[0]) / idt_h[0]));
			bStep(2);

			time_h[0] = Coltime * 365.25;
			CoordinateOutput(2);
			P.ci = -1;
			return 0;
		}

		Ncoll_m[0] = 0;
	
		// P.StopAtCollision = 0
printf("safe -1\n");
		safe_kernel <<< (N_h[0] + Nsmall_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, x4bb_d, v4bb_d, spin_d, spinbb_d, rcrit_d, rcritv_d, rcritbb_d, rcritvbb_d, index_d, indexbb_d, N_h[0] + Nsmall_h[0], NconstT, P.SLevels, -1);
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

__host__ int Data::CollisionMCall(){
#if def_TTV == 0
	
	
	if(Ncoll_m[0] > def_MaxColl){
		fprintf(masterfile, "Error: More Collisions than def_MaxColl, simulation stopped\n");
		printf("Error: More Collisions than def_MaxColl, simulation stopped\n");
		return 0;
	}
	printCollisions();
	CollisionFlag = 1;
	int NminFlag = remove();
	if(NminFlag > 0){
		stopSimulations();
	}
	Ncoll_m[0] = 0;
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
#if poincareFlag == 1
__host__ int Data::PoincareSectionCall(double t){
	if(SIn > 1){
		printf("Compute Poincare Sections only with the second Order integrator!\n");
		fprintf(masterfile, "Compute Poincare Sections only with the second Order integrator!\n");
		return 0;
	}
	PoincareSection <<< (N_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, xold_d, vold_d, Msun_h[0].x, N_h[0], 0, PFlag_d);
	
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
__global__ void testA_kernel(double4 *x4_d, double4 *v4_d, int A){
	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;
	
if(id == 203) printf("%d %.20g %.20g %d\n", id, x4_d[id].z, v4_d[id].z, A);
	
}

//Recursive symplectic close encounter Step.
//At the last level BS is called
__host__ void Data::SEnc(double &time,  int SLevel, double ll, int si, int noColl){
	
	int nt = min(N_h[0] + Nsmall_h[0], 512);
	int nb = (N_h[0] + Nsmall_h[0] + nt - 1) / nt;
	
	if(noColl == 0 || SLevel > 0){	
		setEnc3_kernel <<< nb, nt >>> (N_h[0] + Nsmall_h[0], Nencpairs3_d + SLevel, Encpairs3_d + SLevel * NBNencT, EncpairsScan_d, P.NencMax);
		//groupS_kernel <<< (Nencpairs2_h[0] + 511) / 512, 512 >>> (Nencpairs2_d, Encpairs2_d, Nencpairs3_d + SLevel, Encpairs3_d + SLevel * NBNencT, P.NencMax, P.UseTestParticles, N_h[0], SLevel);	
		groupS2_kernel <<< (Nencpairs2_h[0] + 511) / 512, 512 >>> (Nencpairs2_d, Encpairs2_d, Nencpairs3_d + SLevel, Encpairs3_d + SLevel * NBNencT, P.NencMax, P.UseTestParticles, N_h[0], SLevel);	

		if(N_h[0] + Nsmall_h[0] <= 32){
			Scan32c_kernel <<< 1, 32 >>> (Encpairs3_d + SLevel * NBNencT, Nencpairs3_d + SLevel, N_h[0] + Nsmall_h[0], P.NencMax);

		}
		else if(N_h[0] + Nsmall_h[0] <= 1024){
			int nn = (N_h[0] + Nsmall_h[0] + 31) / 32;
			Scan32a_kernel <<< 1, nn * 32 >>> (Encpairs3_d + SLevel * NBNencT, Nencpairs3_d + SLevel, N_h[0] + Nsmall_h[0], P.NencMax);
		}
		else{
			int nct = 1024;
			int ncb = min((N_h[0] + Nsmall_h[0] + nct - 1) / nct, 1024);

			Scan32d1_kernel <<< ncb, nct >>> (Encpairs3_d + SLevel * NBNencT, Nencpairs3_d + SLevel, N_h[0] + Nsmall_h[0], P.NencMax);
			Scan32d2_kernel <<< 1, ((ncb + 31) / 32) * 32  >>> (Encpairs3_d + SLevel * NBNencT, EncpairsScan_d, Nencpairs3_d + SLevel, N_h[0] + Nsmall_h[0], P.NencMax);
			Scan32d3_kernel  <<< ncb, nct >>>  (Encpairs3_d + SLevel * NBNencT, EncpairsScan_d, Nencpairs3_d + SLevel, N_h[0] + Nsmall_h[0], P.NencMax);
		}

		cudaMemcpy(Nencpairs3_h + SLevel, Nencpairs3_d + SLevel, sizeof(int), cudaMemcpyDeviceToHost);
	}
	cudaDeviceSynchronize();
// /*if(timeStep % 1000 == 0) */printf("Base0 %d %d %d\n", Nencpairs_h[0], Nencpairs2_h[0], Nencpairs3_h[SLevel]);

	int nt3 = min(Nencpairs3_h[SLevel], 512);
	int nb3 = (Nencpairs3_h[SLevel] + nt3 - 1) / nt3;
	int ntf3 = min(Nencpairs3_h[SLevel], 128);
	int nbf3 = (Nencpairs3_h[SLevel] + ntf3 - 1) / ntf3;

#if SERIAL_GROUPING == 1
	if(noColl == 0 || SLevel > 0){	
		SortSb_kernel <<< nb3, nt3 >>> (Encpairs3_d + SLevel * NBNencT, Nencpairs3_d + SLevel, N_h[0] + Nsmall_h[0], P.NencMax);
	}
#endif
	
	SLevel += 1;
	double l = P.SLSteps;	//number of sub steps
	ll *= l;
	//loop over sub time steps
	for(int s = 0; s < l; ++s){
// /*if(timeStep % 1000 == 0)*/ printf("Level %d %d %g %g\n", SLevel, s, ll, time);
		
		if(s == 0){
			RcritS_kernel <<< nb3, nt3 >>>  (xold_d, vold_d, Msun_h[0].x, rcrit_d, rcritv_d, dt_h[0] / ll, n1_h[0], n2_h[0], N_h[0] + Nsmall_h[0], Nencpairs_d, Nencpairs2_d, Nencpairs3_d + SLevel - 1, Encpairs3_d + (SLevel - 1) * NBNencT, P.NencMax, NconstT, SLevel);
			kickS_kernel <<< nb3, nt3 >>> (x4_d, v4_d, xold_d, vold_d, rcritv_d, dt_h[0] / ll * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, Nencpairs3_d + SLevel - 1, Encpairs3_d + (SLevel - 1) * NBNencT, N_h[0] + Nsmall_h[0], NconstT, P.NencMax, SLevel, P.SLevels, 0);
		}
		else{
			RcritS_kernel <<< nb3, nt3 >>>  (x4_d, v4_d, Msun_h[0].x, rcrit_d, rcritv_d, dt_h[0] / ll, n1_h[0], n2_h[0], N_h[0] + Nsmall_h[0], Nencpairs_d, Nencpairs2_d, Nencpairs3_d + SLevel - 1, Encpairs3_d + (SLevel - 1) * NBNencT, P.NencMax, NconstT, SLevel);
			kickS_kernel <<< nb3, nt3 >>> (x4_d, v4_d, x4_d, v4_d, rcritv_d, dt_h[0] / ll * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, Nencpairs3_d + SLevel - 1, Encpairs3_d + (SLevel - 1) * NBNencT, N_h[0] + Nsmall_h[0], NconstT, P.NencMax, SLevel, P.SLevels, 2);
		}
		cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
		fgS_kernel <<< nbf3, ntf3 >>> (x4_d, v4_d, xold_d, vold_d, index_d, dt_h[0] / ll * FGt[si], Msun_h[0].x, test_d, N_h[0] + Nsmall_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, si, P.UseForce, Nencpairs3_d + SLevel - 1, Encpairs3_d + (SLevel - 1) * NBNencT, P.NencMax);
		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 63)/ 64, 64 >>> (x4_d, v4_d, xold_d, vold_d, x4G3_d, v4G3_d, rcrit_d + SLevel * NconstT, rcritv_d + SLevel * NconstT, dt_h[0] / ll * FGt[si], Nencpairs_h[0], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, enccount_d, si, K_d, Kold_d, StopTime_d, N_h[0] + Nsmall_h[0], time, P.StopAtEncounter, Ncoll_d, P.MinMass);
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
// /*if(timeStep % 1000 == 0) */printf("Nencpairs2 %d %d\n", Nencpairs_h[0], Nencpairs2_h[0]);
			if(Nencpairs2_h[0] > 0){
				
				if(SLevel < P.SLevels - 1){
					SEnc (time, SLevel, ll, si, noColl);
				}
				else{
					if(Nsmall_h[0] == 0){
						if(NB[0] <= 512){
							switch(NB[0]){
								case 16:{
									group_kernel < 16, 512 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0], N_h[0]);
								}
								break;
								case 32:{
									group_kernel < 32, 512 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0], N_h[0]);
								}
								break;
								case 64:{
									group_kernel < 64, 512 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0], N_h[0]);
								}
								break;
								case 128:{
									group_kernel < 128, 512> <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0], N_h[0]);
								}
								break;
								case 256:{
									group_kernel < 256, 512 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0], N_h[0]);
								}
								break;
								case 512:{
									group_kernel < 512, 512 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0], N_h[0]);
								}
								break;
							}
						}
						else{
							group_kernel < 1, 512 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0], N_h[0]);
						}
					}
					else{
						if(P.UseTestParticles < 2){
							group_kernel < 512, 512 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0] + Nsmall_h[0], N_h[0]);
						}
						else{
							group_kernel < 512, 512 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0] + Nsmall_h[0], N_h[0] + Nsmall_h[0]);
						}
					}
					cudaDeviceSynchronize();
					BSCall(si, time, noColl, ll);
					time += dt_h[0] / ll / dayUnit;
				}
			}
		}
		kickS_kernel <<< nb3, nt3 >>> (x4_d, v4_d, x4_d, v4_d, rcritv_d, dt_h[0] / ll * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, Nencpairs3_d + SLevel - 1, Encpairs3_d + (SLevel - 1) * NBNencT, N_h[0] + Nsmall_h[0], NconstT, P.NencMax, SLevel, P.SLevels, 1);
	}
} 


__host__ int Data::bStep(int noColl){
	Rcrit_kernel <<< (N_h[0] + Nsmall_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, x4b_d, v4b_d, spin_d, spinb_d, Msun_h[0].x, rcrit_d, rcritb_d, rcritv_d, rcritvb_d, index_d, indexb_d, dt_h[0], test_d, n1_h[0], n2_h[0], time_d, time_h[0], EjectionFlag_d, N_h[0] + Nsmall_h[0], NconstT, P.SLevels, noColl);
	kick32C_kernel <<< (N_h[0] + Nsmall_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, ab_d, N_h[0] + Nsmall_h[0], dt_h[0] * Kt[0]);
	HCCall(Ct[0]);
	fg_kernel <<<(N_h[0] + Nsmall_h[0] + FTX - 1)/FTX, FTX >>> (x4_d, v4_d, xold_d, vold_d, index_d, dt_h[0] * FGt[0], Msun_h[0].x, test_d, N_h[0] + Nsmall_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, 0, P.UseForce);
	cudaDeviceSynchronize();

	//BSCall(0, time_h[0], noColl, 1.0);
	if(P.SLevels > 1){
		if(Nencpairs2_h[0] > 0){
			double time = time_h[0];
			SEnc (time, 0, 1.0, 0, noColl);
		}
	}
	else{
		BSCall(0, time_h[0], noColl, 1.0);
	}


	HCCall(Ct[0]);
	
	kick32C_kernel <<< (N_h[0] + Nsmall_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, ab_d, N_h[0] + Nsmall_h[0], dt_h[0] * Kt[0]);

	return 0;
}


__host__ int Data::step_1kernel(int noColl){
	//no Set Elements, no sort, noPoincare, no floatkick, no force
	// moStopatEncounter, no writeEnc
	Rcrit_kernel <<< (N_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, x4b_d, v4b_d, spin_d, spinb_d, Msun_h[0].x, rcrit_d, rcritb_d, rcritv_d, rcritvb_d, index_d, indexb_d, dt_h[0], test_d, n1_h[0], n2_h[0], time_d, time_h[0], EjectionFlag_d, N_h[0], NconstT, P.SLevels, noColl);
	//Rcritb_kernel <<< (N_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, x4b_d, v4b_d, spin_d, spinb_d, Msun_h[0].x, rcrit_d, rcritb_d, rcritv_d, rcritvb_d, index_d, indexb_d, dt_h[0], test_d, n1_h[0], n2_h[0], time_d, time_h[0], EjectionFlag_d, N_h[0], noColl);

	if(EjectionFlag2 == 0){
		kick32Ab_kernel <<< (N_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs2_d, N_h[0], P.NencMax, 1);
	}
	else{
		kick16c_kernel < 1 > <<< N_h[0], 32 >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, P.NencMax, time_h[0], N_h[0]);
		cudaEventRecord(KickEvent, 0);
		cudaStreamWaitEvent(copyStream, KickEvent, 0);
		cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);
	}
	EjectionFlag2 = 0;

	for(int si = 0; si < SIn; ++si){
		HCCall(Ct[si]);
		//HCfg_kernel <<< (N_h[0] + FTX - 1) / FTX, FTX >>> (x4_d, v4_d, xold_d, vold_d, index_d, dt_h[0] * FGt[si], dt_h[0] * Ct[si], dt_h[0] / Msun_h[0].x * Ct[si], Msun_h[0].x, test_d, N_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, si, P.UseForce);
		fg_kernel <<< (N_h[0] + FTX - 1) / FTX, FTX >>> (x4_d, v4_d, xold_d, vold_d, index_d, dt_h[0] * FGt[si], Msun_h[0].x, test_d, N_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, si, P.UseForce);
		cudaStreamSynchronize(copyStream);

		if(Nenc_m[0] > 0){
			for(int i = 0; i < def_GMax; ++i){
				Nenc_m[i] = 0;
			}			
			setNencpairs <<< 1, 1 >>> (Nencpairs2_d);
		}
//printf("Nencpairs %d\n", Nencpairs_h[0]);
		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 63)/ 64, 64 >>> (x4_d, v4_d, xold_d, vold_d, x4G3_d, v4G3_d, rcrit_d, rcritv_d, dt_h[0] * FGt[si], Nencpairs_h[0], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, enccount_d, si, K_d, Kold_d, StopTime_d, NB[0], time_h[0], P.StopAtEncounter, Ncoll_d, P.MinMass);
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);

			if(P.StopAtEncounter > 0 && Ncoll_m[0] > 0){
				Ncoll_m[0] = 0;
				StopAtEncounterFlag2 = 1;
			}
			
			if(P.SLevels > 1){
				if(Nencpairs2_h[0] > 0){
					double time = time_h[0];
					SEnc (time, 0, 1.0, si, noColl);
				}
			}
			else{
//printf("Nencpairs2 %d\n", Nencpairs2_h[0]);
				if(Nencpairs2_h[0] > 0){
					if(NB[0] < 32){
						group_kernel < 16, 512 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0], N_h[0]);
					}
					else{
						group_kernel < 32, 512 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0], N_h[0]);
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
		HCCall(Ct[si]);
		if(si < SIn - 1){
			kick16c_kernel < 2 > <<< N_h[0], 32 >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[si] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, P.NencMax, time_h[0], N_h[0]);
			cudaEventRecord(KickEvent, 0);
			cudaStreamWaitEvent(copyStream, KickEvent, 0);
			cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);
	
		}
	}
	kick16c_kernel < 1 > <<< N_h[0], 32 >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, P.NencMax, time_h[0], N_h[0]);
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
	Rcrit_kernel <<< (N_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, x4b_d, v4b_d, spin_d, spinb_d, Msun_h[0].x, rcrit_d, rcritb_d, rcritv_d, rcritvb_d, index_d, indexb_d, dt_h[0], test_d, n1_h[0], n2_h[0], time_d, time_h[0], EjectionFlag_d, N_h[0], NconstT, P.SLevels, noColl);
	//Rcritb_kernel <<< (N_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, x4b_d, v4b_d, spin_d, spinb_d, Msun_h[0].x, rcrit_d, rcritb_d, rcritv_d, rcritvb_d, index_d, indexb_d, dt_h[0], test_d, n1_h[0], n2_h[0], time_d, time_h[0], EjectionFlag_d, N_h[0], noColl);

	//use last time step information for setElements function, the beginning of the time step
	if(P.setElementsV == 2){ // convert barycentric velocities to heliocentric
		comCall(1);
	}
	if(P.setElementsV > 0) setElements <<< (P.setElementsN + 63) / 64, 64 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst, 0);
	if(P.setElementsV == 2){
		comCall(-1);
	}

#if SERIAL_GROUPING == 1
	Sortb_kernel <<< (N_h[0] + RTX - 1) / RTX, RTX >>> (Encpairs2_d, N_h[0], P.NencMax);
#endif
	if(doTransits == 0){
		if(EjectionFlag2 == 0){
			kick32Ab_kernel <<< (N_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs2_d, N_h[0], P.NencMax, 1);
		}
		else{
#if def_KickFloat == 0
			kick16c_kernel < 1 > <<< N_h[0], 32 >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, P.NencMax, time_h[0], N_h[0]);
#else
			kick16cf_kernel < 1 > <<< N_h[0], 32 >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, P.NencMax, time_h[0], N_h[0]);
#endif
			cudaEventRecord(KickEvent, 0);
			cudaStreamWaitEvent(copyStream, KickEvent, 0);
			cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);
		}
	}
#if def_TTV == 1
	if(doTransits == 1){
		if(EjectionFlag2 == 0){
			kick32ATTV_kernel <<<1, 32 >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs2_d, N_h[0], P.NencMax, time_h[0], dt_h[0], Msun_h[0].x, Msun_h[0].y, Ntransit_d, Transit_d);
		}
		else{
#if def_KickFloat == 0
			kick16c_kernel < 1 > <<< N_h[0], 32 >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, P.NencMax, time_h[0], N_h[0]);
#else
			kick16cf_kernel < 1 > <<< N_h[0], 32 >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, P.NencMax, time_h[0], N_h[0]);
#endif
			cudaEventRecord(KickEvent, 0);
			cudaStreamWaitEvent(copyStream, KickEvent, 0);
			cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);
			kick32ATTV_kernel <<<1, 32 >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs2_d, N_h[0], P.NencMax, time_h[0], dt_h[0], Msun_h[0].x, Msun_h[0].y, Ntransit_d, Transit_d);
		}
		cudaDeviceSynchronize();
		if(Ntransit_m[0] > 0){
			if(Ntransit_m[0] >= def_NtransitMax - 1){
				printf("more Transits than allowed in def_NtransitMax: %d\n", def_NtransitMax);
				return 0;
			}
			BSTTVStep_kernel < 8, 8 > <<< Ntransit_m[0], 64 >>> (x4_d, v4b_d, Transit_d, N_d, dt_d, Msun_d, index_d, time_d, NBS_d, P.UseForce, P.MinMass, P.UseTestParticles, Nst, TransitTime_d, NtransitsT_d);
			Ntransit_m[0] = 0;
		}
	}
#endif
	if(ForceFlag > 0  || P.setElements > 1){
		comCall(1);
		if(P.setElements > 1) setElements <<< (P.setElementsN + 63) / 64, 64 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst, 1);
		if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[SIn - 1]);
		if(P.UseForce > 0) force <<< (N_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, N_h[0], Nst, P.UseForce, 0, 1);
		if(P.UseYarkovsky == 1) CallYarkovsky2 <<< (N_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], P.SolarConstant, N_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky <<< (N_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], P.SolarConstant, N_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< (N_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], P.SolarConstant, P.Qpr, N_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonDrag <<< (N_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], P.SolarConstant, P.Qpr, N_h[0], Nst, 0);
		comCall(-1);
	}
	EjectionFlag2 = 0;

	for(int si = 0; si < SIn; ++si){
		HCCall(Ct[si]);
		//HCfg_kernel <<< (N_h[0] + FTX - 1) / FTX, FTX >>> (x4_d, v4_d, xold_d, vold_d, index_d, dt_h[0] * FGt[si], dt_h[0] * Ct[si], dt_h[0] / Msun_h[0].x * Ct[si], Msun_h[0].x, test_d, N_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, si, P.UseForce);
		fg_kernel <<< (N_h[0] + FTX - 1) / FTX, FTX >>> (x4_d, v4_d, xold_d, vold_d, index_d, dt_h[0] * FGt[si], Msun_h[0].x, test_d, N_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, si, P.UseForce);
		cudaStreamSynchronize(copyStream);

		if(Nenc_m[0] > 0){
			for(int i = 0; i < def_GMax; ++i){
				Nenc_m[i] = 0;
			}			
			setNencpairs <<< 1, 1 >>> (Nencpairs2_d);
		}
//printf("Nencpairs %d\n", Nencpairs_h[0]);
		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 63)/ 64, 64 >>> (x4_d, v4_d, xold_d, vold_d, x4G3_d, v4G3_d, rcrit_d, rcritv_d, dt_h[0] * FGt[si], Nencpairs_h[0], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, enccount_d, si, K_d, Kold_d, StopTime_d, NB[0], time_h[0], P.StopAtEncounter, Ncoll_d, P.MinMass);
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);

			if(P.StopAtEncounter > 0 && Ncoll_m[0] > 0){
				Ncoll_m[0] = 0;
				StopAtEncounterFlag2 = 1;
			}
			
			if(P.SLevels > 1){
				if(Nencpairs2_h[0] > 0){
					double time = time_h[0];
					SEnc (time, 0, 1.0, si, noColl);
				}
			}
			else{
//printf("Nencpairs2 %d\n", Nencpairs2_h[0]);
				if(Nencpairs2_h[0] > 0){
					if(NB[0] < 32){
						group_kernel < 16, 512 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0], N_h[0]);
					}
					else{
						group_kernel < 32, 512 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0], N_h[0]);
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
		HCCall(Ct[si]);
		if(si < SIn - 1){
#if def_KickFloat == 0
			kick16c_kernel < 2 > <<< N_h[0], 32 >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[si] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, P.NencMax, time_h[0], N_h[0]);
#else
			kick16cf_kernel < 2 > <<< N_h[0], 32 >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[si] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, P.NencMax, time_h[0], N_h[0]);
#endif
			cudaEventRecord(KickEvent, 0);
			cudaStreamWaitEvent(copyStream, KickEvent, 0);
			cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);
	
			if(ForceFlag > 0){
				comCall(1);
				if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[si]);
				if(P.UseForce > 0) force <<< (N_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[si], time_d, N_h[0], Nst, P.UseForce, 0, 1);
				if(P.UseYarkovsky == 1) CallYarkovsky2 <<< (N_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], P.SolarConstant, N_h[0], Nst, 0);
				if(P.UseYarkovsky == 2) CallYarkovsky <<< (N_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], P.SolarConstant, N_h[0], Nst, 0);
				if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< (N_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, index_d, dt_d, Kt[si], P.SolarConstant, P.Qpr, N_h[0], Nst, 0);
				if(P.UsePR == 2) PoyntingRobertsonDrag <<< (N_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[si], P.SolarConstant, P.Qpr, N_h[0], Nst, 0);
				comCall(-1);
			}
		}
	}
#if def_KickFloat == 0
	kick16c_kernel < 1 > <<< N_h[0], 32 >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, P.NencMax, time_h[0], N_h[0]);
#else
	kick16cf_kernel < 1 > <<< N_h[0], 32 >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, P.NencMax, time_h[0], N_h[0]);
#endif
	cudaEventRecord(KickEvent, 0);
	cudaStreamWaitEvent(copyStream, KickEvent, 0);
	cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);
	
	if(ForceFlag > 0){
		comCall(1);
		if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[SIn - 1]);
		if(P.UseForce > 0) force <<< (N_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, N_h[0], Nst, P.UseForce, 0, 1);
		if(P.UseYarkovsky == 1) CallYarkovsky2 <<< (N_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], P.SolarConstant, N_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky <<< (N_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], P.SolarConstant, N_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< (N_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], P.SolarConstant, P.Qpr, N_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonDrag <<< (N_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], P.SolarConstant, P.Qpr, N_h[0], Nst, 0);
		comCall(-1);
	}
	if(EjectionFlag_m[0] > 0){
		int Ej = EjectionCall();
		if(Ej == 0) return 0;
	}

#if poincareFlag == 1
	int per = PoincareSectionCall(time_h[0]);
	if(per == 0) return 0;
#endif
	return 1;
}

__host__ int Data::step_largeN(int noColl){
	Rcrit_kernel <<< (N_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, x4b_d, v4b_d, spin_d, spinb_d, Msun_h[0].x, rcrit_d, rcritb_d, rcritv_d, rcritvb_d, index_d, indexb_d, dt_h[0], test_d, n1_h[0], n2_h[0], time_d, time_h[0], EjectionFlag_d, N_h[0], NconstT, P.SLevels, noColl);
	//use last time step information for setElements function, the beginning of the time step
	if(P.setElementsV == 2){ // convert barycentric velocities to heliocentric
		comCall(1);
	}
	if(P.setElementsV > 0) setElements <<< (P.setElementsN + 63) / 64, 64 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst, 0);
	if(P.setElementsV == 2){
		comCall(-1);
	}
#if SERIAL_GROUPING == 1
	Sortb_kernel <<< (N_h[0] + RTX - 1) / RTX, RTX >>> (Encpairs2_d, N_h[0], P.NencMax);
#endif
	if(EjectionFlag2 == 0){
		kick32Ab_kernel <<< (N_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs2_d, N_h[0], P.NencMax, 1);
	}
	else{
		if(UseAcc == 0){
#if def_KickFloat == 0
			kick32c_kernel < 1 > <<< N_h[0] , NB[0] >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, P.NencMax, time_h[0], N_h[0]);
#else
			kick32cf_kernel < 1 > <<< N_h[0] , NB[0] >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, P.NencMax, time_h[0], N_h[0]);
#endif
			cudaEventRecord(KickEvent, 0);
			cudaStreamWaitEvent(copyStream, KickEvent, 0);
			cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);
		}
		else{
#if def_KickFloat == 0
			acc4C_kernel <<< dim3( (((N_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0], 0, N_h[0], P.NencMax, KP, 0);
#else
			acc4Cf_kernel <<< dim3( (((N_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0], 0, N_h[0], P.NencMax, KP, 0);
#endif
			cudaEventRecord(KickEvent, 0);
			cudaStreamWaitEvent(copyStream, KickEvent, 0);
			cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);
#if SERIAL_GROUPING == 1
			Sortb_kernel <<< (N_h[0] + RTX - 1) / RTX, RTX >>> (Encpairs2_d, N_h[0], P.NencMax);
#endif
			kick32Ab_kernel <<< (N_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs2_d, N_h[0], P.NencMax, 1);
		}
	}
	if(ForceFlag > 0 || P.setElements > 1){
		comCall(1);
		if(P.setElements > 1) setElements <<< (P.setElementsN + 63) / 64, 64 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst, 1);
		if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[SIn - 1]);
		if(P.UseForce > 0) force <<< (N_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, N_h[0], Nst, P.UseForce, 0, 1);
		if(P.UseYarkovsky == 1) CallYarkovsky2 <<< (N_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], P.SolarConstant, N_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky <<< (N_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], P.SolarConstant, N_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< (N_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], P.SolarConstant, P.Qpr, N_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonDrag <<< (N_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], P.SolarConstant, P.Qpr, N_h[0], Nst, 0);
		comCall(-1);
	}
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		HCCall(Ct[si]);
		fg_kernel <<< (N_h[0] + FTX - 1) / FTX, FTX >>> (x4_d, v4_d, xold_d, vold_d, index_d, dt_h[0] * FGt[si], Msun_h[0].x, test_d, N_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, si, P.UseForce);
		cudaStreamSynchronize(copyStream);

		if(Nenc_m[0] > 0){
			for(int i = 0; i < def_GMax; ++i){
				Nenc_m[i] = 0;
			}			
			setNencpairs <<< 1, 1 >>> (Nencpairs2_d);
		}

		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 63)/ 64, 64 >>> (x4_d, v4_d, xold_d, vold_d, x4G3_d, v4G3_d, rcrit_d, rcritv_d, dt_h[0] * FGt[si], Nencpairs_h[0], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, enccount_d, si, K_d, Kold_d, StopTime_d, NB[0], time_h[0], P.StopAtEncounter, Ncoll_d, P.MinMass);
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			if(P.StopAtEncounter > 0 && Ncoll_m[0] > 0){
				Ncoll_m[0] = 0;
				StopAtEncounterFlag2 = 1;
			}
			if(P.SLevels > 1){
				if(Nencpairs2_h[0] > 0){
					double time = time_h[0];
					SEnc (time, 0, 1.0, si, noColl);
				}
			}
			else{
				if(Nencpairs2_h[0] > 0){
					if(NB[0] < 128){
						group_kernel < 64, 512 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0], N_h[0]);
					}
					else if(NB[0] < 256){
						group_kernel < 128, 512 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0], N_h[0]);
					}
					else if(NB[0] < 512){
						group_kernel < 256, 512 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0], N_h[0]);
					}
					else if(NB[0] < 1024){
						group_kernel < 512, 512 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0], N_h[0]);
					}
					else{
						group_kernel < 1, 512 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0], N_h[0]);
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

		HCCall(Ct[si]);
		
		if(si < SIn - 1){
			//kick
			if(UseAcc == 0){
#if def_KickFloat == 0
				kick32c_kernel < 2 > <<< N_h[0] , NB[0] >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[si] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, P.NencMax, time_h[0], N_h[0]);
#else
				kick32cf_kernel < 2 > <<< N_h[0] , NB[0] >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[si] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, P.NencMax, time_h[0], N_h[0]);
#endif
				cudaEventRecord(KickEvent, 0);
				cudaStreamWaitEvent(copyStream, KickEvent, 0);
				cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);
			}
			else{
#if def_KickFloat == 0
				acc4C_kernel <<< dim3( (((N_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0], 0, N_h[0], P.NencMax, KP, 0);
#else
				acc4Cf_kernel <<< dim3( (((N_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0], 0, N_h[0], P.NencMax, KP, 0);
#endif
				cudaEventRecord(KickEvent, 0);
				cudaStreamWaitEvent(copyStream, KickEvent, 0);
				cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);
#if SERIAL_GROUPING == 1
				Sortb_kernel <<< (N_h[0] + RTX - 1) / RTX, RTX >>> (Encpairs2_d, N_h[0], P.NencMax);
#endif
				kick32Ab_kernel <<< (N_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[si] * def_ksq, Nencpairs_d, Encpairs2_d, N_h[0], P.NencMax, 1);
			}
			if(ForceFlag > 0){
				comCall(1);
				if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[si]);
				if(P.UseForce > 0) force <<< (N_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[si], time_d, N_h[0], Nst, P.UseForce, 0, 1);
				if(P.UseYarkovsky == 1) CallYarkovsky2 <<< (N_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], P.SolarConstant, N_h[0], Nst, 0);
				if(P.UseYarkovsky == 2) CallYarkovsky <<< (N_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], P.SolarConstant, N_h[0], Nst, 0);
				if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< (N_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, index_d, dt_d, Kt[si], P.SolarConstant, P.Qpr, N_h[0], Nst, 0);
				if(P.UsePR == 2) PoyntingRobertsonDrag <<< (N_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[si], P.SolarConstant, P.Qpr, N_h[0], Nst, 0);
				comCall(-1);
			}
		}
	}
	//kick
	if(UseAcc == 0){
#if def_KickFloat == 0
		kick32c_kernel < 1 > <<< N_h[0] , NB[0] >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, P.NencMax, time_h[0], N_h[0]);
#else
		kick32cf_kernel < 1 > <<< N_h[0] , NB[0] >>> (x4_d, v4_d, a_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, P.NencMax, time_h[0], N_h[0]);
#endif
		cudaEventRecord(KickEvent, 0);
		cudaStreamWaitEvent(copyStream, KickEvent, 0);
		cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);
	}
	else{
#if def_KickFloat == 0
		/*
		{
		//KTX = 4;
		int NN = (N_h[0] + P.ndev - 1) / P.ndev;
		int Nx0 = 0;
		int Nx1 = NN;
		for(int i = 0; i < P.ndev; ++i){
		//for(int i = 0; i < 1; ++i){
			cudaSetDevice(P.dev[i]);
			int nb = (((NN + KP - 1)/ KP) + KTX - 1) / KTX;
			Nx1 = min(Nx1, N_h[0]);
printf("****** %d %d %d | %d %d %d %d\n", i, N_h[0], (((N_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, nb, Nx0, Nx1, Nx1 - Nx0);
			acc4C_kernel <<< dim3(nb, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, Nx0, Nx1, 0, N_h[0], P.NencMax, KP, 0);
			Nx0 += NN;
			Nx1 += NN;
		}
		cudaSetDevice(P.dev[0]);
		}
		*/
		acc4C_kernel <<< dim3( (((N_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0], 0, N_h[0], P.NencMax, KP, 0);
#else
		acc4Cf_kernel <<< dim3( (((N_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0], 0, N_h[0], P.NencMax, KP, 0);
#endif
		cudaEventRecord(KickEvent, 0);
		cudaStreamWaitEvent(copyStream, KickEvent, 0);
		cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);
#if SERIAL_GROUPING == 1
		Sortb_kernel <<< (N_h[0] + RTX - 1) / RTX, RTX >>> (Encpairs2_d, N_h[0], P.NencMax);
#endif
		kick32Ab_kernel <<< (N_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs2_d, N_h[0], P.NencMax, 1);
	}
	if(ForceFlag > 0){
		comCall(1);
		if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[SIn - 1]);
		if(P.UseForce > 0) force <<< (N_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, N_h[0], Nst, P.UseForce, 0, 1);
		if(P.UseYarkovsky == 1) CallYarkovsky2 <<< (N_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], P.SolarConstant, N_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky <<< (N_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], P.SolarConstant, N_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< (N_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], P.SolarConstant, P.Qpr, N_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonDrag <<< (N_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], P.SolarConstant, P.Qpr, N_h[0], Nst, 0);
		comCall(-1);
	}
	if(EjectionFlag_m[0] > 0){
		int Ej = EjectionCall();
		if(Ej == 0) return 0;
	}
#if poincareFlag == 1
	int per = PoincareSectionCall(time_h[0]);
	if(per == 0) return 0;
#endif
	return 1;
	
}

__host__ int Data::step_small(int noColl){
	Rcrit_kernel <<< (N_h[0] + Nsmall_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, x4b_d, v4b_d, spin_d, spinb_d, Msun_h[0].x, rcrit_d, rcritb_d, rcritv_d, rcritvb_d, index_d, indexb_d, dt_h[0], test_d, n1_h[0], n2_h[0], time_d, time_h[0], EjectionFlag_d, N_h[0] + Nsmall_h[0], NconstT, P.SLevels, noColl);
	//use last time step information for setElements function, the beginning of the time step
	if(P.setElementsV == 2){ // convert barycentric velocities to heliocentric
		comCall(1);
	}	
	if(P.setElementsV > 0) setElements <<< (P.setElementsN + 63) / 64, 64 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst, 0);
	if(P.setElementsV == 2){
		comCall(-1);
	}
#if SERIAL_GROUPING == 1
	Sortb_kernel<<< (N_h[0] + Nsmall_h[0] + RTX - 1) / RTX, RTX >>>(Encpairs2_d, N_h[0] + Nsmall_h[0], P.NencMax);
#endif
	if(EjectionFlag2 == 0){
		kick32Ab_kernel <<< (N_h[0] + Nsmall_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs2_d, N_h[0] + Nsmall_h[0], P.NencMax, 1);
	}
	else{
#if def_KickFloat == 0
		acc4C_kernel <<< dim3( (((N_h[0] + Nsmall_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0] + Nsmall_h[0], 0, N_h[0], P.NencMax, KP, 1);
#else
		acc4Cf_kernel <<< dim3( (((N_h[0] + Nsmall_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0] + Nsmall_h[0], 0, N_h[0], P.NencMax, KP, 1);
#endif
		if(P.UseTestParticles == 2){
#if def_KickFloat == 0
			acc4C_kernel <<< dim3( (((N_h[0] + KP2 - 1)/ KP2) + KTX2 - 1) / KTX2, 1, 1), dim3(KTX2,KTY2,1), KTX2 * KTY2 * KP2 * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0], N_h[0], N_h[0] + Nsmall_h[0], P.NencMax, KP2, 2);
#else
			acc4Cf_kernel <<< dim3( (((N_h[0] + KP2 - 1)/ KP2) + KTX2 - 1) / KTX2, 1, 1), dim3(KTX2,KTY2,1), KTX2 * KTY2 * KP2 * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0], N_h[0], N_h[0] + Nsmall_h[0], P.NencMax, KP2, 2);
#endif
		}
		cudaEventRecord(KickEvent, 0);
		cudaStreamWaitEvent(copyStream, KickEvent, 0);
		cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);
#if SERIAL_GROUPING == 1
		Sortb_kernel<<< (N_h[0] + Nsmall_h[0] + RTX - 1) / RTX, RTX >>>(Encpairs2_d, N_h[0] + Nsmall_h[0], P.NencMax);
#endif
		kick32Ab_kernel <<< (N_h[0] + Nsmall_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs2_d, N_h[0] + Nsmall_h[0], P.NencMax, 1);

	}
	if(ForceFlag > 0 || P.setElements > 1){
		comCall(1);
		if(P.setElements > 1) setElements <<< (P.setElementsN + 63) / 64, 64 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst, 1);
		if(P.Usegas == 1){
			GasAccCall(time_d, dt_d, Kt[SIn - 1]);
			GasAccCall_small(time_d, dt_d, Kt[SIn - 1]);
		}
		if(P.Usegas == 2){
			//GasAccCall(time_d, dt_d, Kt[SIn - 1]);
			GasAccCall2_small(time_d, dt_d, Kt[SIn - 1]);
		}
		if(P.UseForce > 0) force <<< (N_h[0] + Nsmall_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, N_h[0] + Nsmall_h[0], Nst, P.UseForce, 0, 1);
		if(P.UseYarkovsky == 1) CallYarkovsky2 <<< (N_h[0] + Nsmall_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], P.SolarConstant, N_h[0] + Nsmall_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky <<< (N_h[0] + Nsmall_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], P.SolarConstant, N_h[0] + Nsmall_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< (N_h[0] + Nsmall_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], P.SolarConstant, P.Qpr, N_h[0] + Nsmall_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonDrag <<< (N_h[0] + Nsmall_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], P.SolarConstant, P.Qpr, N_h[0] + Nsmall_h[0], Nst, 0);
		comCall(-1);
	}
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){

		HCCall(Ct[si]);
		
		fg_kernel <<<(N_h[0] + Nsmall_h[0] + FTX - 1)/FTX, FTX >>> (x4_d, v4_d, xold_d, vold_d, index_d, dt_h[0] * FGt[si], Msun_h[0].x, test_d, N_h[0] + Nsmall_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, si, P.UseForce);
		cudaStreamSynchronize(copyStream);

		if(Nenc_m[0] > 0){
			for(int i = 0; i < def_GMax; ++i){
				Nenc_m[i] = 0;
			}			
			setNencpairs <<< 1, 1 >>> (Nencpairs2_d);
		}

		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 63)/ 64, 64 >>> (x4_d, v4_d, xold_d, vold_d, x4G3_d, v4G3_d, rcrit_d, rcritv_d, dt_h[0] * FGt[si], Nencpairs_h[0], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, enccount_d, si, K_d, Kold_d, StopTime_d, N_h[0] + Nsmall_h[0], time_h[0], P.StopAtEncounter, Ncoll_d, P.MinMass);
			cudaDeviceSynchronize();
			
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			if(P.StopAtEncounter > 0 && Ncoll_m[0] > 0){
				Ncoll_m[0] = 0;
				StopAtEncounterFlag2 = 1;
			}
// printf("Nencpairs2 A %d %d\n", Nencpairs_h[0], Nencpairs2_h[0]);
			if(P.SLevels > 1){
				if(Nencpairs2_h[0] > 0){
					double time = time_h[0];
					SEnc (time, 0, 1.0, si, noColl);
				}
			}
			else{
				if(Nencpairs2_h[0] > 0){
					if(P.UseTestParticles < 2){
//assume here  E = 3 or E = 4
						group_kernel < 512, 512 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0] + Nsmall_h[0], N_h[0]);
					}
					else{
						group_kernel < 512, 512 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0] + Nsmall_h[0], N_h[0] + Nsmall_h[0]);
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
		if(P.UseSmallCollisions == 1){
			fragmentCall(random_d, x4_d, v4_d, spin_d, index_d, N_h, N_d, Nsmall_h, Nsmall_d, dt_d, Nst, NconstT, Fragments_d, time_h[0], nFragments_m, nFragments_d, MaxIndex, x4_h, v4_h, spin_h, index_h);
			if(nFragments_m[0] > 0){
				int er = printFragments(nFragments_m[0]);
				if(er == 0) return 0;
				er = RemoveCall();
				if(er == 0) return 0;
			}
			
			rotationCall(random_d, x4_d, v4_d, spin_d, index_d, N_h, N_d, Nsmall_h, Nsmall_d, dt_d, Nst, Fragments_d, time_h[0], nFragments_m, nFragments_d);
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

		HCCall(Ct[si]);

		if(si < SIn - 1){
#if def_KickFloat == 0
			acc4C_kernel <<< dim3( (((N_h[0] + Nsmall_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0] + Nsmall_h[0], 0, N_h[0], P.NencMax, KP, 1);
#else
			acc4Cf_kernel <<< dim3( (((N_h[0] + Nsmall_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0] + Nsmall_h[0], 0, N_h[0], P.NencMax, KP, 1);
#endif
			if(P.UseTestParticles == 2){
#if def_KickFloat == 0
				acc4C_kernel <<< dim3( (((N_h[0] + KP2 - 1)/ KP2) + KTX2 - 1) / KTX2, 1, 1), dim3(KTX2,KTY2,1), KTX2 * KTY2 * KP2 * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0], N_h[0], N_h[0] + Nsmall_h[0], P.NencMax, KP2, 2);
#else
				acc4Cf_kernel <<< dim3( (((N_h[0] + KP2 - 1)/ KP2) + KTX2 - 1) / KTX2, 1, 1), dim3(KTX2,KTY2,1), KTX2 * KTY2 * KP2 * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0], N_h[0], N_h[0] + Nsmall_h[0], P.NencMax, KP2, 2);
#endif
			}
				
			cudaEventRecord(KickEvent, 0);
			cudaStreamWaitEvent(copyStream, KickEvent, 0);
			cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);
#if SERIAL_GROUPING == 1
			Sortb_kernel<<< (N_h[0] + Nsmall_h[0] + RTX - 1) / RTX, RTX >>>(Encpairs2_d, N_h[0] + Nsmall_h[0], P.NencMax);
#endif
			kick32Ab_kernel <<< (N_h[0] + Nsmall_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[si] * def_ksq, Nencpairs_d, Encpairs2_d, N_h[0] + Nsmall_h[0], P.NencMax, 1);
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
				if(P.UseForce > 0) force <<< (N_h[0] + Nsmall_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[si], time_d, N_h[0] + Nsmall_h[0], Nst, P.UseForce, 0, 1);
				if(P.UseYarkovsky == 1) CallYarkovsky2 <<< (N_h[0] + Nsmall_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], P.SolarConstant, N_h[0] + Nsmall_h[0], Nst, 0);
				if(P.UseYarkovsky == 2) CallYarkovsky <<< (N_h[0] + Nsmall_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], P.SolarConstant, N_h[0] + Nsmall_h[0], Nst, 0);
				if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< (N_h[0] + Nsmall_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, index_d, dt_d, Kt[si], P.SolarConstant, P.Qpr, N_h[0] + Nsmall_h[0], Nst, 0);
				if(P.UsePR == 2) PoyntingRobertsonDrag <<< (N_h[0] + Nsmall_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[si], P.SolarConstant, P.Qpr, N_h[0] + Nsmall_h[0], Nst, 0);
				comCall(-1);
			}
		}
	}
#if def_KickFloat == 0
	acc4C_kernel <<< dim3( (((N_h[0] + Nsmall_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0] + Nsmall_h[0], 0, N_h[0], P.NencMax, KP, 1);
#else
	acc4Cf_kernel <<< dim3( (((N_h[0] + Nsmall_h[0] + KP - 1)/ KP) + KTX - 1) / KTX, 1, 1), dim3(KTX,KTY,1), KTX * KTY * KP * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0] + Nsmall_h[0], 0, N_h[0], P.NencMax, KP, 1);
#endif
	if(P.UseTestParticles == 2){
#if def_KickFloat == 0
		acc4C_kernel <<< dim3( (((N_h[0] + KP2 - 1)/ KP2) + KTX2 - 1) / KTX2, 1, 1), dim3(KTX2,KTY2,1), KTX2 * KTY2 * KP2 * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0], N_h[0], N_h[0] + Nsmall_h[0], P.NencMax, KP2, 2);
#else
		acc4Cf_kernel <<< dim3( (((N_h[0] + KP2 - 1)/ KP2) + KTX2 - 1) / KTX2, 1, 1), dim3(KTX2,KTY2,1), KTX2 * KTY2 * KP2 * sizeof(double3) >>> ( x4_d, a_d, rcritv_d, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, 0, N_h[0], N_h[0], N_h[0] + Nsmall_h[0], P.NencMax, KP2, 2);
#endif
	}
	cudaEventRecord(KickEvent, 0);
	cudaStreamWaitEvent(copyStream, KickEvent, 0);
	cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<< (N_h[0] + Nsmall_h[0] + RTX - 1) / RTX, RTX >>>(Encpairs2_d, N_h[0] + Nsmall_h[0], P.NencMax);
#endif
	kick32Ab_kernel <<< (N_h[0] + Nsmall_h[0] + RTX - 1) / RTX, RTX >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs2_d, N_h[0] + Nsmall_h[0], P.NencMax, 1);
	
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
		if(P.UseForce > 0) force <<< (N_h[0] + Nsmall_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, N_h[0] + Nsmall_h[0], Nst, P.UseForce, 0, 1);
		if(P.UseYarkovsky == 1) CallYarkovsky2 <<< (N_h[0] + Nsmall_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], P.SolarConstant, N_h[0] + Nsmall_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky <<< (N_h[0] + Nsmall_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], P.SolarConstant, N_h[0] + Nsmall_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< (N_h[0] + Nsmall_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], P.SolarConstant, P.Qpr, N_h[0] + Nsmall_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonDrag <<< (N_h[0] + Nsmall_h[0] + FrTX - 1) / FrTX, FrTX >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], P.SolarConstant, P.Qpr, N_h[0] + Nsmall_h[0], Nst, 0);
		comCall(-1);
	}
	if(EjectionFlag_m[0] > 0){
		int Ej = EjectionCall();
		if(Ej == 0) return 0;
	}
	return 1;
}
__host__ int Data::step_M(int noColl){
	RcritM_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, x4b_d, v4b_d, spin_d, spinb_d, Msun_d, rcrit_d, rcritb_d, rcritv_d, rcritvb_d, dt_d, test_d, n1_d, n2_d, Rcut_d, RcutSun_d, EjectionFlag_d, index_d, indexb_d, Nst, NT, time_d, idt_d, ict_d, delta_d, timeStep, StopFlag_d, NconstT, P.SLevels, noColl, Nstart);
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
			BSTTVStep_kernel < 8, 8 > <<< Ntransit_m[0], 64 >>> (x4_d, v4b_d, Transit_d, N_d, dt_d, Msun_d, index_d, time_d, NBS_d, P.UseForce, P.MinMass, P.UseTestParticles, Nst, TransitTime_d, NtransitsT_d);
			Ntransit_m[0] = 0;
		}
	}
#endif

#if def_RV == 1
	while(time_h[0] >= RVObs_h[RVTimeStep].x && RVTimeStep < NRVTObs_h[0]){
		//repeat for multiple intertime steps
		double Tau = RVObs_h[RVTimeStep].x - (time_h[0] - dt_h[0] / dayUnit);
		//printf("timeRV %d %d %.20g %.20g %.20g %.20g\n", RVTimeStep, NRVTObs_h[0], time_h[0], time_h[0] - dt_h[0] / dayUnit, RVObs_h[RVTimeStep].x, Tau);
		BSRVStep_kernel < 8, 8 > <<< Nst, 64 >>> (x4_d, v4b_d, N_d, Tau * dayUnit, Msun_d, index_d, time_h[0] - dt_h[0] / dayUnit, NBS_d, P.UseForce, P.MinMass, P.UseTestParticles, Nst, RV_d, NRVT_d);
		++RVTimeStep;
	}
	//printf("time   %d %.20g %.20g %.20g\n", RVTimeStep, time_h[0], time_h[0] - dt_h[0] / dayUnit, RVObs_h[RVTimeStep].x);
	if(P.PrintRV == 2){
		BSRVStep_kernel < 8, 8 > <<< Nst, 64 >>> (x4_d, v4b_d, N_d, dt_h[0], Msun_d, index_d, time_h[0] - dt_h[0] / dayUnit, NBS_d, P.UseForce, P.MinMass, P.UseTestParticles, Nst, RV_d, NRVT_d);
	}


#endif
	if(ForceFlag > 0){
		comM_kernel < HCM_Bl, HCM_Bl2, NmaxM > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, vcom_d, Msun_d, index_d, NBS_d, NT, test_d, 1, Nstart);
		if(P.Usegas == 1) GasAccCall_M(time_d, dt_d, Kt[SIn - 1]);
		if(P.UseForce > 0) force <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, NT, Nst, P.UseForce, Nstart, 1);
		if(P.UseYarkovsky == 1) CallYarkovsky2 <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], P.SolarConstant, NT, Nst, Nstart);
		if(P.UseYarkovsky == 2) CallYarkovsky <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], P.SolarConstant, NT, Nst, Nstart);
		if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], P.SolarConstant, P.Qpr, NT, Nst, Nstart);
		if(P.UsePR == 2) PoyntingRobertsonDrag <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], P.SolarConstant, P.Qpr, NT, Nst, Nstart);
		comM_kernel < HCM_Bl, HCM_Bl2, NmaxM > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, vcom_d, Msun_d, index_d, NBS_d, NT, test_d, -1, Nstart);
	}
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		HCM2_kernel < HCM_Bl, HCM_Bl2, NmaxM, 1 > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, dt_d, Msun_d, index_d, NT, Ct[si], test_d, Nencpairs_d, Nencpairs2_d, Nenc_d, Nst, P.UseForce, Nstart);
		fgM_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, xold_d, vold_d, dt_d, Msun_d, test_d, index_d, NT, aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, FGt[si], si, P.UseForce, Nstart);
		cudaStreamSynchronize(copyStream);
		if(Nencpairs_h[0] > 0){
			encounterM_kernel < NmaxM > <<< (Nencpairs_h[0] + 31) / 32 , 32 >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, dt_d, Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, index_d, NBS_d, enccount_d, si, FGt[si], Nst, time_d, P.StopAtEncounter, Ncoll_d, n1_d, P.MinMass);
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			
			if(P.StopAtEncounter > 0 && Ncoll_m[0] > 0){
				Ncoll_m[0] = 0;
				StopAtEncounterFlag2 = 1;
			}
			
			if(Nencpairs2_h[0] > 0){
				groupM1_kernel < 256> <<< Nencpairs2_h[0], 256 >>> (Nencpairs2_d, Encpairs_d, Encpairs2_d, NBS_d, N_d, Nst);
				groupM2_kernel <<< Nencpairs2_h[0], 16 >>> (Encpairs_d, Encpairs2_d, Nenc_d, NBS_d, N_d, Nst);
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
			int col = CollisionMCall();
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
		
		HCM2_kernel < HCM_Bl, HCM_Bl2, NmaxM, 2 > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, dt_d, Msun_d, index_d, NT, Ct[si], test_d, Nencpairs_d, Nencpairs2_d, Nenc_d, Nst, P.UseForce, Nstart);
		if(si < SIn - 1){
			KickM2_kernel < KM_Bl, KM_Bl2, NmaxM, 2 > <<< (NT + KM_Bl2 - 1) / KM_Bl2, KM_Bl>>> (x4_d, v4_d, a_d, rcritv_d, Nencpairs_d, Encpairs_d, dt_d, Kt[si], index_d, NT, Nstart);
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
		//	cudaEventRecord(KickEvent, 0);
		//	cudaStreamWaitEvent(copyStream, KickEvent, 0);
		//	cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);
			if(ForceFlag > 0){
				comM_kernel < HCM_Bl, HCM_Bl2, NmaxM > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, vcom_d, Msun_d, index_d, NBS_d, NT, test_d, 1, Nstart);
				if(P.Usegas == 1) GasAccCall_M(time_d, dt_d, Kt[si]);
				if(P.UseForce > 0) force <<< (NT + 127) / 128, 128  >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[si], time_d, NT, Nst, P.UseForce, Nstart, 1);
				if(P.UseYarkovsky == 1) CallYarkovsky2 <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], P.SolarConstant, NT, Nst, Nstart);
				if(P.UseYarkovsky == 2) CallYarkovsky <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], P.SolarConstant, NT, Nst, Nstart);
				if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, dt_d, Kt[si], P.SolarConstant, P.Qpr, NT, Nst, Nstart);
				if(P.UsePR == 2) PoyntingRobertsonDrag <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[si], P.SolarConstant, P.Qpr, NT, Nst, Nstart);
				comM_kernel < HCM_Bl, HCM_Bl2, NmaxM > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, vcom_d, Msun_d, index_d, NBS_d, NT, test_d, -1, Nstart);
			}
		}
	}
	KickM2_kernel < KM_Bl, KM_Bl2, NmaxM, 1 > <<< (NT + KM_Bl2 - 1) / KM_Bl2, KM_Bl>>> (x4_d, v4_d, a_d, rcritv_d, Nencpairs_d, Encpairs_d, dt_d, Kt[SIn - 1], index_d, NT, Nstart);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
	//cudaEventRecord(KickEvent, 0);
	//cudaStreamWaitEvent(copyStream, KickEvent, 0);
	//cudaMemcpyAsync(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost, copyStream);
	if(ForceFlag > 0){
		comM_kernel < HCM_Bl, HCM_Bl2, NmaxM > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, vcom_d, Msun_d, index_d, NBS_d, NT, test_d, 1, Nstart);
		if(P.Usegas == 1) GasAccCall_M(time_d, dt_d, Kt[SIn - 1]);
		if(P.UseForce > 0) force <<< (NT + 127) / 128, 128  >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, NT, Nst, P.UseForce, Nstart, 1);
		if(P.UseYarkovsky == 1) CallYarkovsky2 <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], P.SolarConstant, NT, Nst, Nstart);
		if(P.UseYarkovsky == 2) CallYarkovsky <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], P.SolarConstant, NT, Nst, Nstart);
		if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], P.SolarConstant, P.Qpr, NT, Nst, Nstart);
		if(P.UsePR == 2) PoyntingRobertsonDrag <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], P.SolarConstant, P.Qpr, NT, Nst, Nstart);
		comM_kernel < HCM_Bl, HCM_Bl2, NmaxM > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, vcom_d, Msun_d, index_d, NBS_d, NT, test_d, -1, Nstart);
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
			BSTTVStep_kernel < 8, 8 > <<< Ntransit_m[0], 64 >>> (x4_d, v4b_d, Transit_d, N_d, dt_d, Msun_d, index_d, time_d, NBS_d, P.UseForce, P.MinMass, P.UseTestParticles, Nst, TransitTime_d, NtransitsT_d);
			Ntransit_m[0] = 0;
		}
	}
#endif
	for(int si = 0; si < SIn; ++si){
		HCM2_kernel < HCM_Bl, HCM_Bl2, NmaxM, 1 > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, dt_d, Msun_d, index_d, NT, Ct[si], test_d, Nencpairs_d, Nencpairs2_d, Nenc_d, Nst, P.UseForce, Nstart);
		fgMSimple_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, xold_d, vold_d, dt_d, Msun_d, test_d, index_d, NT, FGt[si], si, P.UseForce, Nstart);
		HCM2_kernel < HCM_Bl, HCM_Bl2, NmaxM, 2 > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, dt_d, Msun_d, index_d, NT, Ct[si], test_d, Nencpairs_d, Nencpairs2_d, Nenc_d, Nst, P.UseForce, Nstart);
		
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

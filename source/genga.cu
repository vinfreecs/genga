/**************************************
*
* Authors: Simon Grimm, Joachmin Stadel
* July 2013
*
****************************************/

#include "define.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <math.h>
#include <cuda.h>

#include "Host2.h"
#include "Orbit2.h"
#include "Energy.h"
#include "output.h"
#include "gas.h"
#include "integrator.h"
/*
#include "FGfull.h"
#include "BS.h"
//#include "BSA.h"
*/


int main(int argc, char*argv[]){

#if useGas > 0
	printf("Start GENGA with Gas Disc\n");
#endif
#if useGridae
	printf("Use ae Grid\n");
#else
	printf("No ae Grid\n");
#endif
	cudaError_t error;

	Hostinit();
	//determine the number of simulations
	int Nst = NSimulations(argc, argv);
	if(Nst == 0) return 0;

	//Check Device Informations
	int DevError = DeviceInfo();
	if(DevError == 0) return 0;

	//Allocate memory for parameters on the host:
	Halloc();
	cudaDeviceSynchronize();

	// Read parameters from param file //
	printf("Read parameters\n");
	int er = Param(argc, argv);
	if(er == 0) return 0;
	printf("Parameters OK\n");

	// Determine the size of the simulations
	printf("Read Size\n");
	er = size();
	if(er == 0) return 0;
	
	printf("Size OK\n");
	cudaDeviceSynchronize();

	cudaSetDevice(P.dev);

	//Allocate memory for parameters on the device:
        Calloc();
	Info();

#if useGas > 0
	GasAlloc();
#endif

	//Table for fastfg//
	er = FGAlloc();
	if(er == 0) return 0;

	//Allocate Grideae 
#if useGridae
	er = GridaeAlloc();
	if(er == 0) return 0;
#endif
	//Determine the start points of the individual simulations
	Tsizes();
	//Allocate orbit data on Host and Device
	AllocateOrbitt();

	 //allocate mapped memory//
	er = CMallocateOrbit();
	if(er == 0) return 0;

	//initialize memory//
	er = init();
	printf("\nInitialize Memory\n");	

	cudaDeviceSynchronize();
	//read initial conditions//
	printf("\nRead Initial Conditions\n");
	er = ic();
	if(er == 0) return 0;
	printf("Initial Conditions OK\n");

	//remove ghost particles and reorder arrays//
	int NminFlag = remove();

	//remove stopped simulations//
	if(NminFlag == 1){
		stopSimulations();
		NminFlag = 0;
	}

	cudaDeviceSynchronize();
	printf("Compute initial Energy\n");

	cudaStream_t hstream[16];
	for(int hst = 0; hst < 16; ++hst) cudaStreamCreate(&hstream[hst]);
	for(int st = 0; st < Nst; ++st){
		int NBS = NBS_h[st];
		Energy (NB[st], x4_d + NBS, v4_d + NBS, Msun_h[st], Energy_d + NEnergy[st], test_d + NBS, U_d, Energy0_d, hstream[st%16], st, N_h[st], 0);
	}
	for(int hst = 0; hst < 16; ++hst) cudaStreamDestroy(hstream[hst]);
	error = cudaGetLastError();
	fprintf(masterfile,"Energy error = %d = %s\n",error, cudaGetErrorString(error));
	if(error != 0){
		printf("Energy error = %d = %s\n",error, cudaGetErrorString(error));
		return 0;
	}
	cudaDeviceSynchronize();

	printf("Write initial Energy\n");

	//write first output
	er = firstoutput();
	if(er == 0) return 0;
	printf("Energy OK\n");

	//read aeGrid at restart time step 
#if useGridae
	readGridae();	
#endif

	//Set Gas Disc and Gas Table
#if useGas > 0
	printf("Set Gas Table\n");
	er = setGasDisk();
	if(er == 0) return 0;
	printf("Gas Table OK\n");
#endif

	// Set Order and Coefficients ot the symplectic integrator //
	SymplecticP();

	cudaDeviceSynchronize();
	cudaMemset(Energy_d, 0, NEnergyT*sizeof(double));
	if(Nst == 1) printf("Start integration with %d simulation\n", Nst);
	else printf("Start integration with %d simulations\n", Nst);
        error = cudaGetLastError();
	if(error != 0){
		fprintf(masterfile, "Cuda error = %d = %s\n",error, cudaGetErrorString(error));
        	printf("Cuda error = %d = %s\n",error, cudaGetErrorString(error));
		return 0;
	}

	if(Nst > 1){
		firstKick_M();
	}
	else{
		if(P.UseTestParticles == 1) firstKick_small();
		else switch( NB[0] ) {
			case 16: firstKick_16();
			break;
			case 32: firstKick_32();
			break;
			case 64: firstKick_64();
			break;
			case 128: firstKick_128();
			break;
			case 256: firstKick_256();
			break;
			case 512: firstKick_512();
			break;
			case 1024: firstKick_1024();
			break;
			case 2048: firstKick_2048();
			break;
		}
	}
	cudaDeviceSynchronize();
	//Print first informations about close encounter pairs
	firstInfo();

	setStartTime();


for(long long ts = P.tRestart + 1; ts <= P.delta; ++ts){
	double t = ts * P.idt + P.ict * 365.25;
	if(Nst > 1){
		er = step_M(t);
		if(er == 0) return 0;
	}
	else{
		if(P.UseTestParticles == 1){
			er = step_small(t);
			if(er == 0) return 0;
		}
		else switch(NB[0]){
			case 16: step_16(t);
			break;
			case 32: step_32(t);
			break;
			case 64: step_64(t);
			break;
			case 128: step_128(t);
			break;
			case 256: step_256(t);
			break;
			case 512: step_512(t);
			break;
			case 1024: step_1024(t);
			break;
			case 2048: step_2048(t);
			break;
		}
	}
		cudaDeviceSynchronize();
		error = cudaGetLastError();
		if(error != 0){
			printf("Cuda error = %d = %s\n",error, cudaGetErrorString(error));
			fprintf(masterfile, "Cuda error = %d = %s\n",error, cudaGetErrorString(error));
			return 0;
		}

		//Check for too big groups//
		if(Nst == 1){
			er = MaxGroups(ts, t);
			if(er == 0) return 0;
		}

		//Check for too many Collisions//
		if(Ncoll_m[0] >= MaxColl-1){
			printfMaxColl(ts);
			return 0;
		}
		//Print Energy and log information//
		if(ts % P.ei == 0){
			EnergyOutput(ts, t);
		}
#if useGridae 
		if(ts % 10000 == 0){
			copyGridae(ts);
		}
#endif
//test_kernel <<< 1, 16 >>> (x4_d, v4_d, index_d);
		//Print Output//
		if((ts - 1) % P.ci >= P.ci - P.nci){
			CoordinateOutput(ts, t);
#if useGridae
			GridaeOutput(ts);
#endif
		}

		// print time information //
		if(ts % P.ci == 0){
			printTime(ts);
		}

} // end of time step loop

	//print last informations
	printLastTime();
	LastInfo();

	//free all the memory on the Host and on the Device
	er = freeOrbit();
	if(er == 0) return 0;

#if useGridae
	free(Gridaecount_h);
	cudaFree(Gridaecount_d);
#endif

#if useGas > 0
	er = freeGas();
	if(er == 0) return 0;
#endif
	er = freeHost();
	if(er == 0) return 0;

        printf("GENGA terminated successfully\n");
	fprintf(masterfile, "GENGA terminated successfully\n");

	return 0; 
}

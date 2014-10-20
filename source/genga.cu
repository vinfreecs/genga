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

#include "Host2.h"
#include "Orbit2.h"
/*
#include "BS.h"
//#include "BSA.h"
*/

int main(int argc, char*argv[]){

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

#if useGas > 0
	printf("Use Gas Disc\n");
	fprintf(H.masterfile,"Use Gas Disc\n");
#else
	printf("No Gas Disc\n");
	fprintf(H.masterfile,"No Gas Disc\n");
#endif
#if useGridae
	printf("Use ae Grid\n");
	fprintf(H.masterfile, "Use ae Grid\n");
#else
	printf("No ae Grid\n");
	fprintf(H.masterfile, "No ae Grid\n");
#endif

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
	cudaDeviceSynchronize();

	// Read parameters from param file //
	printf("Read parameters\n");
	int er = H.Param(argc, argv);
	if(er == 0) return 0;
	printf("Parameters OK\n");

	// Determine the size of the simulations
	printf("Read Size\n");
	er = H.size();
	if(er == 0) return 0;
	
	printf("Size OK\n");
	cudaDeviceSynchronize();

	cudaSetDevice(H.P.dev);

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

        //Allocate Grideae 
#if useGridae
        er = D.GridaeAlloc();
        if(er == 0) return 0;
#endif
#if useGas > 0
	D.GasAlloc();
#endif

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

	//remove ghost particles and reorder arrays//
	int NminFlag = D.remove();

	//remove stopped simulations//
	if(NminFlag == 1){
		D.stopSimulations();
		NminFlag = 0;
	}

	cudaDeviceSynchronize();
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
#if useGridae
	D.readGridae();	
#endif

	//Set Gas Disc and Gas Table
#if useGas > 0
	printf("Set Gas Table\n");
	er = D.setGasDisk();
	if(er == 0) return 0;
	printf("Gas Table OK\n");
#endif

	// Set Order and Coefficients of the symplectic integrator //
	D.SymplecticP();

	cudaDeviceSynchronize();
	cudaMemset(D.Energy_d, 0, D.NEnergyT*sizeof(double));
	if(Nst == 1) printf("Start integration with %d simulation\n", Nst);
	else printf("Start integration with %d simulations\n", Nst);
        error = cudaGetLastError();
	if(error != 0){
		fprintf(D.masterfile, "Start error = %d = %s\n",error, cudaGetErrorString(error));
        	printf("Start error = %d = %s\n",error, cudaGetErrorString(error));
		return 0;
	}

	fflush(D.masterfile);

	if(D.Nst > 1){
		D.firstKick_M();
	}
	else{
		if(D.P.UseTestParticles == 1) D.firstKick_small();
		else switch( D.NB[0] ) {
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
	}
	cudaDeviceSynchronize();
	//Print first informations about close encounter pairs
	D.firstInfo();
	D.setStartTime();
#if poincareFlag == 1
	sprintf(D.poincarefilename, "%sPoincare%s_%.12ld.dat", D.GSF[0].path, D.GSF[0].X, 0);
	D.poincarefile = fopen(D.poincarefilename, "w");
#endif
	for(long long ts = D.P.tRestart + 1; ts <= D.P.delta; ++ts){
		double t = ts * D.P.idt + D.P.ict * 365.25;
		//Multi simulation mode
		if(D.Nst > 1){
			er = D.step_M(t);
			if(er == 0) return 0;
		}
		else{
			//Test particles
			if(D.P.UseTestParticles == 1){
				er = D.step_small(t);
				if(er == 0) return 0;
			}
			//check the number of massive particles
			else switch(D.NB[0]){
				case 16: D.step_16(t);
				break;
				case 32: D.step_32(t);
				break;
				case 64: D.step_64(t);
				break;
				case 128: D.step_128(t);
				break;
				case 256: D.step_256(t);
				break;
				case 512: D.step_512(t);
				break;
				case 1024: D.step_1024(t);
				break;
				case 2048: D.step_2048(t);
				break;
			}
		}

			cudaDeviceSynchronize();
			error = cudaGetLastError();
			if(error != 0){
				printf("Step error = %d = %s\n",error, cudaGetErrorString(error));
				fprintf(D.masterfile, "Step error = %d = %s\n",error, cudaGetErrorString(error));
				return 0;
			}

			//Check for too big groups//
			if(D.Nst == 1){
				er = D.MaxGroups(ts, t);
				if(er == 0) return 0;
			}

			//Check for too many Collisions//
			if(D.Ncoll_m[0] >= MaxColl-1){
				D.printMaxColl(ts);
				return 0;
			}
			//Print Energy and log information//
			if(ts % D.P.ei == 0){
				D.EnergyOutput(ts, t);
			}

#if useGridae 
			if(ts % 10000 == 0){
				D.copyGridae(ts);
			}
#endif
//test_kernel <<< 1, 16 >>> (x4_d, v4_d, index_d);
			//Print Output//
			if((ts - 1) % D.P.ci >= D.P.ci - D.P.nci){
				D.CoordinateOutput(ts, t);
#if useGridae
				D.GridaeOutput(ts);
#endif
#if poincareFlag == 1
				if((ts - 1) % D.P.ci == D.P.ci - D.P.nci){
					fclose(D.poincarefile);
					sprintf(D.poincarefilename, "%sPoincare%s_%.12ld.dat", D.GSF[0].path, D.GSF[0].X, ts);
					//Erase old Poincare files
					D.poincarefile = fopen(D.poincarefilename, "w");
				}
#endif
			}
			// print time information //
			if(ts % D.P.ci == 0){
				D.printTime(ts);
				fflush(D.masterfile);
			}

	} // end of time step loop
#if poincareFlag == 1
	fclose(D.poincarefile);
#endif

	//print last informations
	D.printLastTime();
	D.LastInfo();

	//free all the memory on the Host and on the Device
	er = D.freeOrbit();
	if(er == 0) return 0;

#if useGridae
	free(D.Gridaecount_h);
	cudaFree(D.Gridaecount_d);
#endif

#if useGas > 0
	er = D.freeGas();
	if(er == 0) return 0;
#endif
	er = H.freeHost();
	if(er == 0) return 0;

        printf("GENGA terminated successfully\n");
	fprintf(H.masterfile, "GENGA terminated successfully\n");

	return 0; 
}

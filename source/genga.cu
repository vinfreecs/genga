/**************************************
*
* Authors: Simon Grimm, Joachmin Stadel
* July 2013
*
****************************************/

#include "define.h"

#include "Host2.h"
#include "Orbit2.h"

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
	if(er == 0){
		cudaDeviceSynchronize();
		return 0;
	}
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

	er = D.naf.alloc2(D.NT, D.N_h[0], D.Nsmall_h[0], D.Nst, D.GSF, D.P.NAFformat, D.P.tRestart, D.index_h, D.indexsmall_h);
	if(er == 0) return 0;
#endif

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
		fprintf(D.masterfile, "Start error = %d = %s\n",error, cudaGetErrorString(error));
        	printf("Start error = %d = %s\n",error, cudaGetErrorString(error));
		return 0;
	}

	fflush(D.masterfile);
#if USE_NAF == 1
			//compute the x and y arrays for the naf algorithm
			int NAFstep = 0;
			D.naf.getnafvarsCall(D.x4_d, D.v4_d, D.x4small_d, D.v4small_d, D.index_d, D.indexsmall_d, D.NBS_d, D.vcom_d, D.U_d, D.test_d, D.P.NAFvars, D.naf.x_d, D.naf.y_d, D.Msun_d, D.Msun_h[0], D.NT, D.Nst, D.naf.n, NAFstep, D.NB[0], D.N_h[0], D.Nsmall_h[0], D.P.UseTestParticles);
			++NAFstep;
#endif

	if(D.Nst > 1){
		D.firstKick_M(0);
	}
	else{
		if(D.P.UseTestParticles == 1) D.firstKick_small();
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
		printf("first kick OK\n");
	}
	//Print first informations about close encounter pairs
	D.firstInfo();
	D.setStartTime();
#if poincareFlag == 1
	sprintf(D.poincarefilename, "%sPoincare%s_%.12ld.dat", D.GSF[0].path, D.GSF[0].X, 0);
	D.poincarefile = fopen(D.poincarefilename, "w");
#endif

	D.irrTimeStep = 0;
	if(D.P.IrregularOutputs == 1){
		er = D.readIrregularOutputs();
		if(er == 0){
			return 0;
		}
	}
	if(D.P.setElements == 1){
		er = D.readSetElements();
		if(er == 0){
			return 0;
		}
	}



	int bufferCount = 1;
	int bufferCountIrr = 1;
	D.MultiSim = 0;
	if(D.Nst > 1) D.MultiSim = 1;
	for(D.timeStep = D.P.tRestart + 1; D.timeStep <= D.P.deltaT; ++D.timeStep){
		D.time_h[0] = D.timeStep * D.idt_h[0] + D.ict_h[0] * 365.25;
		cudaMemcpy(D.time_d, D.time_h, sizeof(double), cudaMemcpyHostToDevice);
		
		int er = D.step();
		if(er == 0){
			 return 0;
		}

			cudaDeviceSynchronize();
			//Check for too many encounters
			if(D.EncFlag_m[0] > 0){
				printf("Error: more encounters than allowed. %d %d\n", D.EncFlag_m[0], D.P.NencMax);
				fprintf(D.masterfile, "Error: more encounters than allowed. %d %d\n", D.EncFlag_m[0], D.P.NencMax);
				return 0;
			}

			//Check for too big groups//
			if(D.Nst == 1){
				er = D.MaxGroups();
				if(er == 0) return 0;
			}

			//Check for too many Collisions//
			if(D.Ncoll_m[0] >= MaxColl-1){
				D.printMaxColl();
				return 0;
			}
	
			error = cudaGetLastError();
			if(error != 0){
				printf("Step error = %d = %s\n",error, cudaGetErrorString(error));
				fprintf(D.masterfile, "Step error = %d = %s\n",error, cudaGetErrorString(error));
				return 0;
			}
			//Print Energy and log information//
			if(D.P.ei > 0 && D.timeStep % D.P.ei == 0){
				if(bufferCount >= D.P.Buffer){
					D.EnergyOutput();
				}
			}

			if(H.P.UseaeGrid == 1){ 
				if(D.timeStep % 10000 == 0){
					D.copyGridae();
				}
			}



//test_kernel <<< 1, 16 >>> (x4_d, v4_d, index_d);
			//Print Output//
			if(D.P.ci > 0 && ((D.timeStep - 1) % D.P.ci >= D.P.ci - D.P.nci)){
				if(D.P.Buffer == 1){
					D.CoordinateOutput(0);
				}
				else if(bufferCount >= D.P.Buffer){
					//write out buffer
					D.timestepBuffer[bufferCount - 1] = D.timeStep;
					for(int st = 0; st < D.Nst; ++st){
						D.NBuffer[D.Nst * (bufferCount - 1) + st].x = D.N_h[st];
						D.NBuffer[D.Nst * (bufferCount - 1) + st].y = D.Nsmall_h[st];
					}
					D.CoordinateToBuffer(bufferCount - 1, 0);
					D.CoordinateOutputBuffer(0);
				}
				else{
					//store in buffer
					D.timestepBuffer[bufferCount - 1] = D.timeStep;
					for(int st = 0; st < D.Nst; ++st){
						D.NBuffer[D.Nst * (bufferCount - 1) + st].x = D.N_h[st];
						D.NBuffer[D.Nst * (bufferCount - 1) + st].y = D.Nsmall_h[st];
					}
					D.CoordinateToBuffer(bufferCount - 1, 0);
				}
				if(D.P.UseaeGrid == 1){
					D.GridaeOutput();
				}
#if poincareFlag == 1
				if((D.timeStep - 1) % D.P.ci == D.P.ci - D.P.nci){
					fclose(D.poincarefile);
					sprintf(D.poincarefilename, "%sPoincare%s_%.12ld.dat", D.GSF[0].path, D.GSF[0].X, D.timeStep);
					//Erase old Poincare files
					D.poincarefile = fopen(D.poincarefilename, "w");
				}
#endif
			}
			// print time information //
			if(D.P.ci > 0 && D.timeStep % D.P.ci == 0){
				if(bufferCount >= D.P.Buffer){
					D.printTime();
					fflush(D.masterfile);
				}
			}

			// print irregular outputs
			if(D.P.IrregularOutputs == 1 && D.irrTimeStep < D.NIrrOutputs && D.time_h[0] >= D.IrrOutputs[D.irrTimeStep]){

				int ni = 1;
				for(int i = 0; i < ni; ++i){
					double dTau = -(D.time_h[0] - D.IrrOutputs[D.irrTimeStep]) / D.idt_h[0];

					D.IrregularStep(dTau);
					for(int st = 0; st < D.Nst; ++st){
						D.time_h[st] += dTau * D.idt_h[st];
					}
					if(D.Nst > 1){
						cudaMemcpy(D.time_d, D.time_h, D.Nst * sizeof(double), cudaMemcpyHostToDevice);
					}

					D.step();

					if(D.P.Buffer == 1){
						D.CoordinateOutput(1);
					}
					else if(bufferCountIrr >= D.P.Buffer){
						//write out buffer
						D.timestepBufferIrr[bufferCountIrr - 1] = D.timeStep;
						for(int st = 0; st < D.Nst; ++st){
							D.NBufferIrr[D.Nst * (bufferCountIrr - 1) + st].x = D.N_h[st];
							D.NBufferIrr[D.Nst * (bufferCountIrr - 1) + st].y = D.Nsmall_h[st];
						}
						D.CoordinateToBuffer(bufferCountIrr - 1, 1);
						D.CoordinateOutputBuffer(1);
					}
					else{
						//store in buffer
						D.timestepBufferIrr[bufferCountIrr - 1] = D.timeStep;
						for(int st = 0; st < D.Nst; ++st){
							D.NBufferIrr[D.Nst * (bufferCountIrr - 1) + st].x = D.N_h[st];
							D.NBufferIrr[D.Nst * (bufferCountIrr - 1) + st].y = D.Nsmall_h[st];
						}
						D.CoordinateToBuffer(bufferCountIrr - 1, 1);
					}

					D.IrregularStep(-dTau);
					for(int st = 0; st < D.Nst; ++st){
						D.time_h[st] -= dTau * D.idt_h[st];
					}
					if(D.Nst > 1){
						cudaMemcpy(D.time_d, D.time_h, D.Nst * sizeof(double), cudaMemcpyHostToDevice);
					}

					D.step();
					D.SymplecticP(1);

					++bufferCountIrr;
					if(bufferCountIrr > D.P.Buffer){
						bufferCountIrr = 1;
					}
					++D.irrTimeStep;
				
					dTau = -(D.time_h[0] - D.IrrOutputs[D.irrTimeStep]) / D.idt_h[0];
					if(dTau <= 0) ++ni;

					if(ni + D.irrTimeStep - 1 > D.NIrrOutputs) break;
				}
			}

			if(D.P.ci > 0 && ((D.timeStep - 1) % D.P.ci >= D.P.ci - D.P.nci)){
				++bufferCount;
			}
			if(bufferCount > D.P.Buffer){
				bufferCount = 1;
			}
#if USE_NAF == 1
			//compute the x and y arrays for the naf algorithm
			if(D.timeStep % D.P.NAFinterval == 0){
				D.naf.getnafvarsCall(D.x4_d, D.v4_d, D.x4small_d, D.v4small_d, D.index_d, D.indexsmall_d, D.NBS_d, D.vcom_d, D.U_d, D.test_d, D.P.NAFvars, D.naf.x_d, D.naf.y_d, D.Msun_d, D.Msun_h[0], D.NT, D.Nst, D.naf.n, NAFstep, D.NB[0], D.N_h[0], D.Nsmall_h[0], D.P.UseTestParticles);
				++NAFstep;
				if(NAFstep % D.P.NAFn0 == 0){
					er = D.naf.nafCall(D.NT, D.N_h, D.N_d, D.Nsmall_h, D.Nsmall_d, D.Nst, D.GSF, D.time_h, D.time_d, D.idt_h, D.P.NAFformat, D.P.NAFinterval, D.index_h, D.indexsmall_h, D.index_d, D.indexsmall_d, D.NBS_h);
					if(er == 0) return 0;
					NAFstep = 0;
				}
			}

#endif
	} // end of time step loop
	//write out the remaining buffer
	if(D.P.IrregularOutputs == 1){
		if(bufferCountIrr > 1){
			D.P.Buffer = bufferCountIrr - 1;
			D.CoordinateOutputBuffer(1);
		}
	}
	if(bufferCount > 1){
		D.P.Buffer = bufferCount - 1;
		D.CoordinateOutputBuffer(0);
	}

#if poincareFlag == 1
	fclose(D.poincarefile);
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

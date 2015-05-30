#include "Orbit2.h"

timeval tt1;				//start time
timeval tt2;				//start time of a output time intervall
timeval tt3;				//end time of a output time intervall
timeval tt4;				//end time//

long long times, timems;			//elapsed time in seconds and microseconds
// ********************************************3
//This function prints the initial Energy and Coordinate output
//If Restart is set, then it reads the corespondent initial conditions from the files and writes no output
//
//Authors: Simon Grimm, Joachim Stadel
//March 2014
// *************************************
__host__ int Data::firstoutput(){
	for(int st = 0; st < Nst; ++st){
		if(P.tRestart == 0){
			int NBS = NBS_h[st];
			int NsmallS = NsmallS_h[st];
			GSF[st].Energyfile = fopen(GSF[st].Energyfilename, "a");
			cudaMemcpy(Energy_h + NEnergy[st], Energy_d + NEnergy[st], sizeof(double)*8, cudaMemcpyDeviceToHost);
			fprintf(GSF[st].Energyfile,"%.16g %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g\n", ict_h[st], N_h[st] + Nsmall_h[st], Energy_h[0 + NEnergy[st]], Energy_h[1 + NEnergy[st]], Energy_h[2 + NEnergy[st]], Energy_h[3 + NEnergy[st]], Energy_h[4 + NEnergy[st]], Energy_h[5 + NEnergy[st]], Energy_h[6 + NEnergy[st]], Energy_h[7 + NEnergy[st]]);
			fclose(GSF[st].Energyfile);

			if(P.FormatP == 1){
				if(Nst == 1 || P.FormatS == 0){
					if(P.FormatT == 0) sprintf(GSF[st].outputfilename, "%sOut%s_%.12d.dat", GSF[st].path, GSF[st].X, 0);
					if(P.FormatT == 1) sprintf(GSF[st].outputfilename, "%sOut%s.dat", GSF[st].path, GSF[st].X);
					GSF[st].outputfile = fopen(GSF[st].outputfilename, "w");
				}
				else{
					if(P.FormatT == 0)sprintf(GSF[st].outputfilename, "%s../Out%s_%.12d.dat", GSF[st].path, GSF[st].X, 0);
					if(P.FormatT == 1)sprintf(GSF[st].outputfilename, "%s../Out%s.dat", GSF[st].path, GSF[st].X);
					if(st == 0) GSF[st].outputfile = fopen(GSF[st].outputfilename, "w");
					else GSF[st].outputfile = fopen(GSF[st].outputfilename, "a");
				}
			}

			printOutput(x4_h + NBS, v4_h + NBS, index_h + NBS, test_h + NBS, ict_h[st], 1, N_h[st], GSF[st].outputfile, Msun_h[st], spin_h + NBS, x4small_h + NsmallS, v4small_h + NsmallS, spinsmall_h + NsmallS, indexsmall_h + NsmallS, Nsmall_h[st], Nst, aelimits_h + NBS, aelimitssmall_h + NsmallS, aecount_h + NBS, aecountsmall_h + NsmallS, enccount_h + NBS, enccountsmall_h + NsmallS, aecountT_h + NBS, aecountsmallT_h + NsmallS, enccountT_h + NBS, enccountsmallT_h + NsmallS, P.ci);
			if(P.FormatP == 1) fclose(GSF[st].outputfile);
		}
		else if(N_h[st] + Nsmall_h[st] > 0){
			int tsign = 1;
			if(idt_h[st] < 0) tsign = -1;
			GSF[st].Energyfile = fopen(GSF[st].Energyfilename, "r");
			double skip;
			double Et;
			char Ets[160];
			sprintf(Ets, "%.16g", (P.tRestart * idt_h[st]) / 365.25 + ict_h[st]);
			fscanf (GSF[st].Energyfile, "%lf",&Et);
			fscanf (GSF[st].Energyfile, "%lf",&skip);
			fscanf (GSF[st].Energyfile, "%lf",&skip);
			fscanf (GSF[st].Energyfile, "%lf",&skip);
			fscanf (GSF[st].Energyfile, "%lf",&LI_h[st]);
			fscanf (GSF[st].Energyfile, "%lf",&U_h[st]);
			fscanf (GSF[st].Energyfile, "%lf",&Energy0_h[st]);
			fscanf (GSF[st].Energyfile, "%lf",&LI0_h[st]);
			fscanf (GSF[st].Energyfile, "%lf",&skip);
			fscanf (GSF[st].Energyfile, "%lf",&skip);
			int er = 0;
			while(Et * tsign <= atof(Ets) * tsign){
				fscanf (GSF[st].Energyfile, "%lf",&Et);
				fscanf (GSF[st].Energyfile, "%lf",&skip);
				fscanf (GSF[st].Energyfile, "%lf",&skip);
				fscanf (GSF[st].Energyfile, "%lf",&skip);
				fscanf (GSF[st].Energyfile, "%lf",&LI_h[st]);
				fscanf (GSF[st].Energyfile, "%lf",&U_h[st]);
				fscanf (GSF[st].Energyfile, "%lf",&skip);
				fscanf (GSF[st].Energyfile, "%lf",&skip);
				fscanf (GSF[st].Energyfile, "%lf",&skip);
				er = fscanf (GSF[st].Energyfile, "%lf",&skip);
				if(Et * tsign >= atof(Ets) * tsign) break;

				if(er <= 0){
					break;
				}				
			}		
			if(er <= 0){
				fprintf(masterfile, "Error: In Simulation %s: Restart time step not valid\n", GSF[st].path);
				printf("Error: In Simulation %s: Restart time step not valid\n", GSF[st].path);
				return 0;
			}

			U_h[st] /= Kg;

			fclose(GSF[st].Energyfile);

			cudaMemcpy(Energy0_d + st, Energy0_h + st, sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(U_d + st, U_h + st, sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(LI_d + st, LI_h + st, sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(LI0_d + st, LI0_h + st, sizeof(double), cudaMemcpyHostToDevice);
		}
	}
	return 1;
}


//**************************************
//This function prints the coordinate output
//Authors: Simon Grimm, Joachim Stadel
//March 2014
// ***************************************
__host__ void Data::printOutput(double4 *x4_h, double4 *v4_h, int *index_h, double *test_h, double t, long long ts, int N, FILE *outputfile, double Msun, double3 *spin_h, double4 *x4small_h, double4 *v4small_h, double3 *spinsmall_h, int *indexsmall_h, int Nsmall, int Nst, float4 *aelimits_h, float4 *aelimitssmall_h, int *aecount_h, int *aecountsmall_h, int *enccount_h, int *enccountsmall_h, long long *aecountT_h, long long *aecountsmallT_h, long long *enccountT_h, long long *enccountsmallT_h, int ci){

	DemoToHelio(x4_h, v4_h, Msun, N, x4small_h, v4small_h, Nsmall);

	int index;
	int st = 0;

	for(int j = 0; j < N; j+=1){
		if(Nst > 1) st = index_h[j] / 100;
		if(P.FormatP == 0){
			char outputfilename[160];
			if(Nst == 1){
				sprintf(outputfilename, "%sOut%s_p%.6d.dat", GSF[st].path, GSF[st].X, index_h[j]);
			}
			else sprintf(outputfilename, "%sOut%s_p%.6d.dat", GSF[st].path, GSF[st].X, index_h[j] % 100);
			if(t > ict_h[st]) outputfile = fopen(outputfilename, "a");
			else outputfile = fopen(outputfilename, "w");
		}

		if(Nst == 1 || P.FormatS == 1) index = index_h[j];
		else index = index_h[j] % 100;

		aecountT_h[j] += aecount_h[j];
		enccountT_h[j] += enccount_h[j];

		fprintf(outputfile,"%.16g %d %.40g %.40g %.40g %.40g %.40g %.40g %.40g %.40g %.40g %.40g %.40g %.8g %.8g %.8g %.8g %.8g %.8g %lld %.40g \n", t, index, x4_h[j].w, v4_h[j].w, x4_h[j].x, x4_h[j].y, x4_h[j].z, v4_h[j].x, v4_h[j].y, v4_h[j].z, spin_h[j].x, spin_h[j].y, spin_h[j].z, aelimits_h[j].x, aelimits_h[j].y, aelimits_h[j].z, aelimits_h[j].w, (double)(aecount_h[j])/ci, (double)(aecountT_h[j])/ts, enccountT_h[j], test_h[j]);
		if(P.FormatP == 0) fclose(outputfile);
	}
	for(int j = 0; j < Nsmall; j+=1){
		int st = 0;
		if(P.FormatP == 0){
			char outputfilename[160];
			sprintf(outputfilename, "%sOut%s_p%.6d.dat", GSF[st].path, GSF[st].X, indexsmall_h[j]);
			if(t > ict_h[st]) outputfile = fopen(outputfilename, "a");
			else outputfile = fopen(outputfilename, "w");
		}

		if(Nst == 1) index = indexsmall_h[j];
		else index = indexsmall_h[j] % 100;

		aecountsmallT_h[j] += aecountsmall_h[j];
		enccountsmallT_h[j] += enccountsmall_h[j];
		fprintf(outputfile,"%.16g %d %.40g %.40g %.40g %.40g %.40g %.40g %.40g %.40g %.40g %.40g %.40g %.8g %.8g %.8g %.8g %.8g %.8g %lld %.40g \n", t, index, x4small_h[j].w, v4small_h[j].w, x4small_h[j].x, x4small_h[j].y, x4small_h[j].z, v4small_h[j].x, v4small_h[j].y, v4small_h[j].z, spinsmall_h[j].x, spinsmall_h[j].y, spinsmall_h[j].z, aelimitssmall_h[j].x, aelimitssmall_h[j].y, aelimitssmall_h[j].z, aelimitssmall_h[j].w, (double)(aecountsmall_h[j])/ci, (double)(aecountsmallT_h[j])/ts, enccountsmallT_h[j], -1.0);
		if(P.FormatP == 0) fclose(outputfile);
	}
}

//this fucntion prints the first close encounter information to the info file
__host__ void Data::firstInfo(){
	cudaMemcpy(Nencpairs_h, Nencpairs_d, (Nst + 1) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(Nencpairssmall_h, Nencpairssmall_d, (Nst + 1) * sizeof(int), cudaMemcpyDeviceToHost);
	for(int st = 0; st < Nst; ++st){
		GSF[st].logfile = fopen(GSF[st].logfilename, "a");
		if(Nst == 1) fprintf(GSF[st].logfile, "Initial Precheck pairs: %d Test particles: %d\n", Nencpairs_h[0], Nencpairssmall_h[0]);
		else fprintf(GSF[st].logfile, "Initial Precheck pairs: %d Test particles: %d\n", Nencpairs_h[st + 1], Nencpairssmall_h[st + 1]);
		fclose(GSF[st].logfile);
	}
}

__host__ int Data::firstEnergy(){
	cudaStream_t hstream[16];

	for(int hst = 0; hst < 16; ++hst) cudaStreamCreate(&hstream[hst]);
	for(int st = 0; st < Nst; ++st){
		int NBS = NBS_h[st];
		EnergyCall(NB[st], x4_d + NBS, v4_d + NBS, spin_d + NBS, Msun_h[st], Energy_d + NEnergy[st], test_d + NBS, U_d, LI_d, Energy0_d, LI0_d, hstream[st%16], st, N_h[st], 0);
	}
	for(int hst = 0; hst < 16; ++hst) cudaStreamDestroy(hstream[hst]);
	error = cudaGetLastError();
	fprintf(masterfile,"Energy error = %d = %s\n",error, cudaGetErrorString(error));
	if(error != 0){
		printf("Energy error = %d = %s\n",error, cudaGetErrorString(error));
		return 0;
	}
	return 1;
}

//This function calls the Energy function and prints information
__host__ void Data::EnergyOutput(int bufferCount){
	cudaStream_t hstream[16];
	for(int hst = 0; hst < 16; ++hst){
		cudaStreamCreate(&hstream[hst]);
	}
	if(P.Usegas == 1){
		if(Nst == 1) gasEnergyCall(NB[0], Energy_d, test_d, U_d, hstream[0], 0, N_h[0]);
		else{
			for(int st = 0; st < Nst; ++st){
				int NBS = NBS_h[st];
				gasEnergyMCall(NB[st], Energy_d + NBS, test_d + NBS, U_d + st, hstream[st%16], st, N_h[st]);
			}
		}
	}
	for(int st = 0; st < Nst; ++st){
		int NBS = NBS_h[st];
		EnergyCall(NB[st], x4_d + NBS, v4_d + NBS, spin_d + NBS, Msun_h[st], Energy_d + NEnergy[st], test_d + NBS, U_d, LI_d, Energy0_d, LI0_d, hstream[st%16], st, N_h[st], 1);
	}
	for(int hst = 0; hst < 16; ++hst){
		cudaStreamDestroy(hstream[hst]);
	}

	if(Nst > 1) cudaMemcpy(time_h, time_d, Nst*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(Energy_h, Energy_d, sizeof(double) * NEnergyT, cudaMemcpyDeviceToHost);
	cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
	for(int st = 0; st < Nst; ++st){
		GSF[st].Energyfile = fopen(GSF[st].Energyfilename, "a");
		fprintf(GSF[st].Energyfile,"%.16g %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g\n", time_h[st]/365.25, N_h[st] + Nsmall_h[st], Energy_h[0 + NEnergy[st]], Energy_h[1 + NEnergy[st]], Energy_h[2 + NEnergy[st]], Energy_h[3 + NEnergy[st]], Energy_h[4 + NEnergy[st]], Energy_h[5 + NEnergy[st]], Energy_h[6 + NEnergy[st]], Energy_h[7 + NEnergy[st]]);
		fclose(GSF[st].Energyfile);

		GSF[st].logfile = fopen(GSF[st].logfilename, "a");
		cudaMemcpy(Nencpairs2_h + st + 1, Nencpairs2_d + st + 1, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(Nencpairs_h + st + 1, Nencpairs_d + st + 1, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(Nencpairssmall2_h + st, Nencpairssmall2_d + st, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(Nencpairssmall_h + st + 1, Nencpairssmall_d + st + 1, sizeof(int), cudaMemcpyDeviceToHost);

		if(Nst == 1){
			fprintf(GSF[st].logfile, "    CE:    %d; ", Nencpairs2_h[0]);
			fprintf(GSF[0].logfile, "groups: %d; 2: %d, 4: %d, 8: %d, 16: %d, 32: %d, 64: %d, 128: %d, 256: %d, 512: %d, 1024: %d, 2048: %d\n", Nenc_m[0], Nenc_m[1], Nenc_m[2], Nenc_m[3], Nenc_m[4], Nenc_m[5], Nenc_m[6], Nenc_m[7], Nenc_m[8], Nenc_m[9], Nenc_m[10], Nenc_m[11]);
			fprintf(GSF[st].logfile, "    CE TP: %d; ", Nencpairssmall2_h[0]);
			fprintf(GSF[0].logfile, "groups: %d; 2: %d, 4: %d, 8: %d, 16: %d, 32: %d, 64: %d, 128: %d, 256: %d, 512: %d, 1024: %d, 2048: %d\n", Nencsmall_m[0], Nencsmall_m[1], Nencsmall_m[2], Nencsmall_m[3], Nencsmall_m[4], Nencsmall_m[5], Nencsmall_m[6], Nencsmall_m[7], Nencsmall_m[8], Nencsmall_m[9], Nencsmall_m[10], Nencsmall_m[11]);

			fprintf(GSF[0].logfile, "    Precheck-pairs:    %d\n    TP Precheck-pairs: %d\n", Nencpairs_h[0], Nencpairssmall_h[0]);
		}
		else{
			fprintf(GSF[st].logfile, "    CE:    %d\n    CE TP: %d\n", Nencpairs2_h[st + 1], Nencpairssmall2_h[st + 1]);
			fprintf(GSF[st].logfile, "    Precheck-pairs:    %d\n    TP Precheck-pairs: %d\n", Nencpairs_h[st + 1], Nencpairssmall_h[st + 1]);
		}
		fclose(GSF[st].logfile);

	}
	cudaMemset(Energy_d, 0, NEnergyT * sizeof(double));
	
}


__global__ void CoordinateToBuffer_kernel(double4 *x4_d, double4 *v4_d, int *index_d, double3 *spin_d, float4 *aelimits_d, int* aecount_d, long long *aecountT_d, long long *enccountT_d, double *test_d, double4 *x4small_d, double4 *v4small_d, int *indexsmall_d, double3 *spinsmall_d, float4 *aelimitssmall_d, int* aecountsmall_d, long long *aecountsmallT_d, long long *enccountsmallT_d, double *coordinateBuffer_d, int NT, int NsmallT, int NconstT, int bufferCount){

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < NT){
		//time
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 1] = index_d[id];
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 2] = x4_d[id].w;
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 3] = v4_d[id].w;
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 4] = x4_d[id].x;
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 5] = x4_d[id].y;
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 6] = x4_d[id].z;
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 7] = v4_d[id].x;
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 8] = v4_d[id].y;
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 9] = v4_d[id].z;
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 10] = spin_d[id].x;
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 11] = spin_d[id].y;
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 12] = spin_d[id].z;
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 13] = aelimits_d[id].x;
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 14] = aelimits_d[id].y;
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 15] = aelimits_d[id].z;
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 16] = aelimits_d[id].w;
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 17] = aecount_d[id];
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 18] = aecountT_d[id];
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 19] = enccountT_d[id];
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 20] = test_d[id];
	}
	else if(id < NT + NsmallT){
		//time
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 1] = indexsmall_d[id - NT];
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 2] = x4small_d[id - NT].w;
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 3] = v4small_d[id - NT].w;
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 4] = x4small_d[id - NT].x;
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 5] = x4small_d[id - NT].y;
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 6] = x4small_d[id - NT].z;
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 7] = v4small_d[id - NT].x;
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 8] = v4small_d[id - NT].y;
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 9] = v4small_d[id - NT].z;
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 10] = spinsmall_d[id - NT].x;
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 11] = spinsmall_d[id - NT].y;
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 12] = spinsmall_d[id - NT].z;
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 13] = aelimitssmall_d[id - NT].x;
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 14] = aelimitssmall_d[id - NT].y;
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 15] = aelimitssmall_d[id - NT].z;
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 16] = aelimitssmall_d[id - NT].w;
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 17] = aecountsmall_d[id - NT];
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 18] = aecountsmallT_d[id - NT];
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 19] = enccountsmallT_d[id - NT];
		coordinateBuffer_d[21 * NconstT * bufferCount + 21 * id + 20] = 0.0;
	}
}

__host__ void Data::CoordinateToBuffer(int bufferCount){
	CoordinateToBuffer_kernel <<< (NT + 511) / 512, 512 >>> (x4_d, v4_d, index_d, spin_d, aelimits_d, aecount_d, aecountT_d, enccountT_d, test_d, x4small_d, v4small_d, indexsmall_d, spinsmall_d, aelimitssmall_d, aecountsmall_d, aecountsmallT_d, enccountsmallT_d, coordinateBuffer_d, NT, NsmallT, NconstT, bufferCount);
}

//This function copies the data from the device to host and calls the printoutput function
__host__ void Data::CoordinateOutput(){

	if(Nst > 1 && timeStep < P.deltaT){
		int s = 0;
		for(int st = 0; st < Nst; ++st){
			if(timeStep >= delta[st]){
				s = 1;
				N_h[st] = 0;
			}
		}
		if(s == 1) stopSimulations();
	}

	cudaMemcpy(x4_h, x4_d, sizeof(double4)*NT, cudaMemcpyDeviceToHost);
	cudaMemcpy(v4_h, v4_d, sizeof(double4)*NT, cudaMemcpyDeviceToHost);
	cudaMemcpy(index_h, index_d, sizeof(int)*NT, cudaMemcpyDeviceToHost);
	cudaMemcpy(test_h, test_d, sizeof(double)*NT, cudaMemcpyDeviceToHost);
	cudaMemcpy(spin_h, spin_d, sizeof(double3)*NT, cudaMemcpyDeviceToHost);
	cudaMemcpy(aelimits_h, aelimits_d, sizeof(float4)*NT, cudaMemcpyDeviceToHost);
	cudaMemcpy(aecount_h, aecount_d, sizeof(int)*NT, cudaMemcpyDeviceToHost);
	cudaMemcpy(enccount_h, enccount_d, sizeof(int)*NT, cudaMemcpyDeviceToHost);
	cudaMemcpy(aecountT_h, aecountT_d, sizeof(long long)*NT, cudaMemcpyDeviceToHost);
	cudaMemcpy(enccountT_h, enccountT_d, sizeof(long long)*NT, cudaMemcpyDeviceToHost);

	cudaMemcpy(x4small_h, x4small_d, sizeof(double4)*NsmallT, cudaMemcpyDeviceToHost);
	cudaMemcpy(v4small_h, v4small_d, sizeof(double4)*NsmallT, cudaMemcpyDeviceToHost);
	cudaMemcpy(indexsmall_h, indexsmall_d, sizeof(int)*NsmallT, cudaMemcpyDeviceToHost);
	cudaMemcpy(spinsmall_h, spinsmall_d, sizeof(double3)*NsmallT, cudaMemcpyDeviceToHost);
	cudaMemcpy(aelimitssmall_h, aelimitssmall_d, sizeof(float4)*NsmallT, cudaMemcpyDeviceToHost);
	cudaMemcpy(aecountsmall_h, aecountsmall_d, sizeof(int)*NsmallT, cudaMemcpyDeviceToHost);
	cudaMemcpy(enccountsmall_h, enccountsmall_d, sizeof(int)*NsmallT, cudaMemcpyDeviceToHost);
	cudaMemcpy(aecountsmallT_h, aecountsmallT_d, sizeof(long long)*NsmallT, cudaMemcpyDeviceToHost);
	cudaMemcpy(enccountsmallT_h, enccountsmallT_d, sizeof(long long)*NsmallT, cudaMemcpyDeviceToHost);

	if(Nst > 1) cudaMemcpy(time_h, time_d, Nst * sizeof(double), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	for(int st = 0; st < Nst; ++st){
		int NBS = NBS_h[st];
		int NsmallS = NsmallS_h[st];

		if(P.FormatP == 1){
			if(Nst == 1 || P.FormatS == 0){
				if(P.FormatT == 0){
					sprintf(GSF[st].outputfilename,"%sOut%s_%.12ld.dat", GSF[st].path, GSF[st].X, timeStep);
					GSF[st].outputfile = fopen(GSF[st].outputfilename, "w");
				}
				if(P.FormatT == 1){
					sprintf(GSF[st].outputfilename,"%sOut%s.dat", GSF[st].path, GSF[st].X);
					GSF[st].outputfile = fopen(GSF[st].outputfilename, "a");
					}
			}
			else{
				if(P.FormatT == 0){
					sprintf(GSF[st].outputfilename, "%s../Out%s_%.12d.dat", GSF[st].path, GSF[st].X, timeStep);
					if(st == 0) GSF[st].outputfile = fopen(GSF[st].outputfilename, "w");
					else GSF[st].outputfile = fopen(GSF[st].outputfilename, "a");
				}
				if(P.FormatT == 1){
					sprintf(GSF[st].outputfilename, "%s../Out%s.dat", GSF[st].path, GSF[st].X);
					GSF[st].outputfile = fopen(GSF[st].outputfilename, "a");
				}
			}
		}


		printOutput(x4_h + NBS, v4_h + NBS, index_h + NBS, test_h + NBS, time_h[st]/365.25, timeStep, N_h[st], GSF[st].outputfile, Msun_h[st], spin_h + NBS, x4small_h + NsmallS, v4small_h + NsmallS, spinsmall_h + NsmallS, indexsmall_h + NsmallS, Nsmall_h[st], Nst, aelimits_h + NBS, aelimitssmall_h + NsmallS, aecount_h + NBS, aecountsmall_h + NsmallS, enccount_h + NBS, enccountsmall_h + NsmallS, aecountT_h + NBS, aecountsmallT_h + NsmallS, enccountT_h + NBS, enccountsmallT_h + NsmallS, P.ci);

		if(P.FormatP == 1) fclose(GSF[st].outputfile);

	}
	cudaMemcpy(aecountT_d, aecountT_h, sizeof(long long)*NT, cudaMemcpyHostToDevice);
	cudaMemcpy(aecountsmallT_d, aecountsmallT_h, sizeof(long long)*NsmallT, cudaMemcpyHostToDevice);

	cudaMemset(aecount_d, 0, sizeof(int)*NT);
	cudaMemset(aecountsmall_d, 0, sizeof(int)*NsmallT);
}

//This function copies the data from the coordinate buffer and calls the printoutput function
__host__ void Data::CoordinateOutputBuffer(){

	cudaMemcpy(coordinateBuffer_h, coordinateBuffer_d, P.Buffer * 21 * NconstT * sizeof(double), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	for(int bf = 0; bf < P.Buffer; ++bf){
		for(int i = 0; i < NT; ++i){
			index_h[i] = coordinateBuffer_h[21 * NconstT * bf + 21 * i + 1];
			x4_h[i].w = coordinateBuffer_h[21 * NconstT * bf + 21 * i + 2];
			v4_h[i].w = coordinateBuffer_h[21 * NconstT * bf + 21 * i + 3];
			x4_h[i].x = coordinateBuffer_h[21 * NconstT * bf + 21 * i + 4];
			x4_h[i].y = coordinateBuffer_h[21 * NconstT * bf + 21 * i + 5];
			x4_h[i].z = coordinateBuffer_h[21 * NconstT * bf + 21 * i + 6];
			v4_h[i].x = coordinateBuffer_h[21 * NconstT * bf + 21 * i + 7];
			v4_h[i].y = coordinateBuffer_h[21 * NconstT * bf + 21 * i + 8];
			v4_h[i].z = coordinateBuffer_h[21 * NconstT * bf + 21 * i + 9];
			spin_h[i].x = coordinateBuffer_h[21 * NconstT * bf + 21 * i + 10];
			spin_h[i].y = coordinateBuffer_h[21 * NconstT * bf + 21 * i + 11];
			spin_h[i].z = coordinateBuffer_h[21 * NconstT * bf + 21 * i + 12];
			aelimits_h[i].x = coordinateBuffer_h[21 * NconstT * bf + 21 * i + 13];
			aelimits_h[i].y = coordinateBuffer_h[21 * NconstT * bf + 21 * i + 14];
			aelimits_h[i].z = coordinateBuffer_h[21 * NconstT * bf + 21 * i + 15];
			aelimits_h[i].w = coordinateBuffer_h[21 * NconstT * bf + 21 * i + 16];
			aecount_h[i] = coordinateBuffer_h[21 * NconstT * bf + 21 * i + 17];
			aecountT_h[i] = coordinateBuffer_h[21 * NconstT * bf + 21 * i + 18];
			enccountT_h[i] = coordinateBuffer_h[21 * NconstT * bf + 21 * i + 19];
			test_h[i] = coordinateBuffer_h[21 * NconstT * bf + 21 * i + 20];

		}
		for(int i = 0; i < NsmallT; ++i){
			int ii = i + NT;
			indexsmall_h[i] = coordinateBuffer_h[21 * NconstT * bf + 21 * ii + 1];
			x4small_h[i].w = coordinateBuffer_h[21 * NconstT * bf + 21 * ii + 2];
			v4small_h[i].w = coordinateBuffer_h[21 * NconstT * bf + 21 * ii + 3];
			x4small_h[i].x = coordinateBuffer_h[21 * NconstT * bf + 21 * ii + 4];
			x4small_h[i].y = coordinateBuffer_h[21 * NconstT * bf + 21 * ii + 5];
			x4small_h[i].z = coordinateBuffer_h[21 * NconstT * bf + 21 * ii + 6];
			v4small_h[i].x = coordinateBuffer_h[21 * NconstT * bf + 21 * ii + 7];
			v4small_h[i].y = coordinateBuffer_h[21 * NconstT * bf + 21 * ii + 8];
			v4small_h[i].z = coordinateBuffer_h[21 * NconstT * bf + 21 * ii + 9];
			spinsmall_h[i].x = coordinateBuffer_h[21 * NconstT * bf + 21 * ii + 10];
			spinsmall_h[i].y = coordinateBuffer_h[21 * NconstT * bf + 21 * ii + 11];
			spinsmall_h[i].z = coordinateBuffer_h[21 * NconstT * bf + 21 * ii + 12];
			aelimitssmall_h[i].x = coordinateBuffer_h[21 * NconstT * bf + 21 * ii + 13];
			aelimitssmall_h[i].y = coordinateBuffer_h[21 * NconstT * bf + 21 * ii + 14];
			aelimitssmall_h[i].z = coordinateBuffer_h[21 * NconstT * bf + 21 * ii + 15];
			aelimitssmall_h[i].w = coordinateBuffer_h[21 * NconstT * bf + 21 * ii + 16];
			aecountsmall_h[i] = coordinateBuffer_h[21 * NconstT * bf + 21 * ii + 17];
			aecountsmallT_h[i] = coordinateBuffer_h[21 * NconstT * bf + 21 * ii + 18];
			enccountsmallT_h[i] = coordinateBuffer_h[21 * NconstT * bf + 21 * ii + 19];

		}
		for(int st = 0; st < Nst; ++st){
			int NBS = NBS_h[st];
			int NsmallS = NsmallS_h[st];

			if(P.FormatP == 1){
				if(Nst == 1 || P.FormatS == 0){
					if(P.FormatT == 0){
						sprintf(GSF[st].outputfilename,"%sOut%s_%.12ld.dat", GSF[st].path, GSF[st].X, timestepBuffer[bf]);
						GSF[st].outputfile = fopen(GSF[st].outputfilename, "w");
					}
					if(P.FormatT == 1){
						sprintf(GSF[st].outputfilename,"%sOut%s.dat", GSF[st].path, GSF[st].X);
						GSF[st].outputfile = fopen(GSF[st].outputfilename, "a");
						}
				}
				else{
					if(P.FormatT == 0){
						sprintf(GSF[st].outputfilename, "%s../Out%s_%.12d.dat", GSF[st].path, GSF[st].X, timestepBuffer[bf]);
						if(st == 0) GSF[st].outputfile = fopen(GSF[st].outputfilename, "w");
						else GSF[st].outputfile = fopen(GSF[st].outputfilename, "a");
					}
					if(P.FormatT == 1){
						sprintf(GSF[st].outputfilename, "%s../Out%s.dat", GSF[st].path, GSF[st].X);
						GSF[st].outputfile = fopen(GSF[st].outputfilename, "a");
					}
				}
			}
	
			double time = timestepBuffer[bf] * idt_h[st] + ict_h[st] * 365.25;		

			printOutput(x4_h + NBS, v4_h + NBS, index_h + NBS, test_h + NBS, time/365.25, timestepBuffer[bf], N_h[st], GSF[st].outputfile, Msun_h[st], spin_h + NBS, x4small_h + NsmallS, v4small_h + NsmallS, spinsmall_h + NsmallS, indexsmall_h + NsmallS, Nsmall_h[st], Nst, aelimits_h + NBS, aelimitssmall_h + NsmallS, aecount_h + NBS, aecountsmall_h + NsmallS, enccount_h + NBS, enccountsmall_h + NsmallS, aecountT_h + NBS, aecountsmallT_h + NsmallS, enccountT_h + NBS, enccountsmallT_h + NsmallS, P.ci);

			if(P.FormatP == 1) fclose(GSF[st].outputfile);

		}
	}
	cudaMemcpy(aecountT_d, aecountT_h, sizeof(long long)*NT, cudaMemcpyHostToDevice);
	cudaMemcpy(aecountsmallT_d, aecountsmallT_h, sizeof(long long)*NsmallT, cudaMemcpyHostToDevice);

	cudaMemset(aecount_d, 0, sizeof(int)*NT);
	cudaMemset(aecountsmall_d, 0, sizeof(int)*NsmallT);

	if(timeStep < P.deltaT){
		int s = 0;
		for(int st = 0; st < Nst; ++st){
			if(timeStep >= delta[st]){
				s = 1;
				N_h[st] = 0;
			}
		}
		if(s == 1) stopSimulations();
	}
}


__host__ void Data::GridaeOutput(){
	int GridNae = Gridae.Na * Gridae.Ne;
	int GridNai = Gridae.Na * Gridae.Ni;
	sprintf(Gridae.filename, "aeCount%s_%.12ld.dat", Gridae.X, timeStep);
	Gridae.file = fopen(Gridae.filename, "w");
	cudaMemcpy(Gridaecount_h, Gridaecount_d, sizeof(int)*GridNae, cudaMemcpyDeviceToHost);
	cudaMemcpy(Gridaicount_h, Gridaicount_d, sizeof(int)*GridNai, cudaMemcpyDeviceToHost);
	//ae grid
	for(int i = 0; i < Gridae.Ne; ++i){
		for(int j = 0; j < Gridae.Na; ++j){
			if(timeStep > Gridae.Start){
				GridaecountS_h[i * Gridae.Na + j] += Gridaecount_h[i * Gridae.Na + j];
				GridaecountT_h[i * Gridae.Na + j] += Gridaecount_h[i * Gridae.Na + j];
			}
			fprintf(Gridae.file, "%lld ", GridaecountT_h[i * Gridae.Na + j]);
		}
		fprintf(Gridae.file, "\n");
	}
	fprintf(Gridae.file, "\n");
	fprintf(Gridae.file, "\n");
	for(int i = 0; i < Gridae.Ne; ++i){
		for(int j = 0; j < Gridae.Na; ++j){
			fprintf(Gridae.file, "%lld ", GridaecountS_h[i * Gridae.Na + j]);
			GridaecountS_h[i * Gridae.Na + j] = 0;
		}
		fprintf(Gridae.file, "\n");
	}
	fprintf(Gridae.file, "\n");
	fprintf(Gridae.file, "\n");
	//ai grid
	for(int i = 0; i < Gridae.Ni; ++i){
		for(int j = 0; j < Gridae.Na; ++j){
			if(timeStep > Gridae.Start){
				GridaicountS_h[i * Gridae.Na + j] += Gridaicount_h[i * Gridae.Na + j];
				GridaicountT_h[i * Gridae.Na + j] += Gridaicount_h[i * Gridae.Na + j];
			}
			fprintf(Gridae.file, "%lld ", GridaicountT_h[i * Gridae.Na + j]);
		}
		fprintf(Gridae.file, "\n");
	}
	fprintf(Gridae.file, "\n");
	fprintf(Gridae.file, "\n");
	for(int i = 0; i < Gridae.Ni; ++i){
		for(int j = 0; j < Gridae.Na; ++j){
			fprintf(Gridae.file, "%lld ", GridaicountS_h[i * Gridae.Na + j]);
			GridaicountS_h[i * Gridae.Na + j] = 0;
		}
		fprintf(Gridae.file, "\n");
	}

	fclose(Gridae.file);
	cudaMemset(Gridaecount_d, 0, sizeof(int)*GridNae);
	cudaMemset(Gridaicount_d, 0, sizeof(int)*GridNai);
}


//This function prints information if there are too many Collisions which fit not in allocated memory
//It stops the integration
__host__ void Data::printMaxColl(){
	printf("Error: Too many Collisions, MaxColl too small. Ncoll = %d. Integration Stopped at timestep = %lld\n", Ncoll_m[0], timeStep);
	if(Nst ==1){
		GSF[0].logfile = fopen(GSF[0].logfilename, "a");
		fprintf(GSF[0].logfile,"Error: MaxColl too small. Ncoll = %d. Integration Stopped at timestep = %lld\n", Ncoll_m[0], timeStep);
		fprintf(GSF[0].logfile, "Total number of groups when Error accured: %d, 2: %d, 4: %d, 8: %d, 16: %d, 32: %d, 64: %d, 128: %d, 256: %d, 512: %d, 1024: %d, 2048: %d\n", Nenc_m[0], Nenc_m[1], Nenc_m[2], Nenc_m[3], Nenc_m[4], Nenc_m[5], Nenc_m[6], Nenc_m[7], Nenc_m[8], Nenc_m[9], Nenc_m[10], Nenc_m[11]);
		cudaMemcpy(Coll_h, Coll_d, sizeof(double)*25*MaxColl, cudaMemcpyDeviceToHost);
		for(int nc = 0; nc < MaxColl; ++nc){
			for(int in = 0; in < 25; ++in){
				fprintf(GSF[0].logfile, "%d ", (int)(Coll_h[nc * 25 + in]));
			}
			fprintf(GSF[0].logfile, "\n");
		}

		fclose(GSF[0].logfile);
	}
	else{
		fprintf(masterfile,"Error: MaxColl too small. Ncoll = %d. Integration Stopped at timestep = %lld\n", Ncoll_m[0], timeStep);
	}
}


//This function prints information if a too big close encounter group occurs and stops the integrations
__host__ int Data::MaxGroups(){
	for(int nm = 7; nm < 12; ++nm){
		if(Nenc_m[nm] > 0){
			GSF[0].logfile = fopen(GSF[0].logfilename, "a");
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			fprintf(GSF[0].logfile, "Number of Close-Encounter-pairs: %d\n", *Nencpairs2_h);
			fprintf(GSF[0].logfile, "Total number of groups: %d, 2: %d, 4: %d, 8: %d, 16: %d, 32: %d, 64: %d, 128: %d, 256: %d, 512: %d, 1024: %d, 2048: %d\n", Nenc_m[0], Nenc_m[1], Nenc_m[2], Nenc_m[3], Nenc_m[4], Nenc_m[5], Nenc_m[6], Nenc_m[7], Nenc_m[8], Nenc_m[9], Nenc_m[10], Nenc_m[11]);
			fprintf(GSF[0].logfile, "Number of Precheck-pairs: %d\n", *Nencpairs_h);
			fprintf(GSF[0].logfile,"Output data when Error occured:\n");
			cudaMemcpy(xold_h, xold_d, sizeof(double4)*NB[0], cudaMemcpyDeviceToHost);
			cudaMemcpy(vold_h, vold_d, sizeof(double4)*NB[0], cudaMemcpyDeviceToHost);
			cudaMemcpy(index_h, index_d, sizeof(int)*NB[0], cudaMemcpyDeviceToHost);
			cudaMemcpy(test_h, test_d, sizeof(double)*NB[0], cudaMemcpyDeviceToHost);
			cudaMemcpy(spin_h, spin_d, sizeof(double3)*NB[0], cudaMemcpyDeviceToHost);
			cudaMemcpy(aelimits_h, aelimits_d, sizeof(float4)*NB[0], cudaMemcpyDeviceToHost);
			cudaMemcpy(aecount_h, aecount_d, sizeof(int)*NB[0], cudaMemcpyDeviceToHost);
			cudaMemcpy(enccount_h, enccount_d, sizeof(int)*NB[0], cudaMemcpyDeviceToHost);
			cudaMemcpy(aecountT_h, aecountT_d, sizeof(long long)*NB[0], cudaMemcpyDeviceToHost);
			cudaMemcpy(enccountT_h, enccountT_d, sizeof(long long)*NB[0], cudaMemcpyDeviceToHost);

			cudaMemcpy(xoldsmall_h, xoldsmall_d, sizeof(double4)*Nsmall_h[0], cudaMemcpyDeviceToHost);
			cudaMemcpy(voldsmall_h, voldsmall_d, sizeof(double4)*Nsmall_h[0], cudaMemcpyDeviceToHost);
			cudaMemcpy(indexsmall_h, indexsmall_d, sizeof(int)*Nsmall_h[0], cudaMemcpyDeviceToHost);
			cudaMemcpy(spinsmall_h, spinsmall_d, sizeof(double3)*Nsmall_h[0], cudaMemcpyDeviceToHost);
			cudaMemcpy(aelimitssmall_h, aelimitssmall_d, sizeof(float4)*Nsmall_h[0], cudaMemcpyDeviceToHost);
			cudaMemcpy(aecountsmall_h, aecountsmall_d, sizeof(int)*Nsmall_h[0], cudaMemcpyDeviceToHost);
			cudaMemcpy(enccountsmall_h, enccountsmall_d, sizeof(int)*Nsmall_h[0], cudaMemcpyDeviceToHost);
			cudaMemcpy(aecountsmallT_h, aecountsmallT_d, sizeof(long long)*Nsmall_h[0], cudaMemcpyDeviceToHost);				
			cudaMemcpy(enccountsmallT_h, enccountsmallT_d, sizeof(long long)*Nsmall_h[0], cudaMemcpyDeviceToHost);

			GSF[0].outputfile = fopen(GSF[0].outputfilename, "w");	
			printOutput(x4_h, v4_h, index_h, test_h, time_h[0]/365.25, timeStep, N_h[0], GSF[0].outputfile, Msun_h[0], spin_h, x4small_h, v4small_h, spinsmall_h, indexsmall_h, Nsmall_h[0], Nst, aelimits_h, aelimitssmall_h, aecount_h, aecountsmall_h, enccount_h, enccountsmall_h, aecountT_h, aecountsmallT_h, enccountT_h, enccountsmallT_h, P.ci);
			fclose(GSF[0].outputfile);

			fprintf(GSF[0].logfile,"Error: Too big group:%g. Integration Stopped at timestep = %lld\n", pow(2.0, nm), timeStep);
			printf("Error: Too big group:%g. Integration Stopped at timestep = %lld\n", pow(2.0, nm), timeStep);
			fclose(GSF[0].logfile);
			return 0;
		}
	}
	return 1;
}


//This functions set the starting rutime of the integrations
__host__ void Data::setStartTime(){
	gettimeofday(&tt1, NULL);
	gettimeofday(&tt2, NULL);
	times = 0.0;
	timems = 0.0;
}


//This function prints information how long the integration takes
__host__ void Data::printTime(){
	gettimeofday( &tt3, NULL );
	for(int st = 0; st < Nst; ++st){
		times = (tt3.tv_sec - tt2.tv_sec);
		timems = (tt3.tv_usec - tt2.tv_usec);

		GSF[st].timefile = fopen(GSF[st].timefilename, "a");
		fprintf(GSF[st].timefile, "%g\n", times + timems/1000000.0);
		GSF[st].logfile = fopen(GSF[st].logfilename, "a");
		fprintf(GSF[st].logfile,"Reached timestep %lld with %d bodies, %d test particles. Total Energy: %.20g\n", timeStep, N_h[st], Nsmall_h[st], Energy_h[4 + NEnergy[st]]);
		fclose(GSF[st].timefile);
		fclose(GSF[st].logfile);

		if(Nst == 1){
			printf("Reached timestep %lld with %d bodies, %d test particles. Total Energy: %.20g\n", timeStep, N_h[0], Nsmall_h[0], Energy_h[4]);
			fprintf(masterfile, "Reached timestep %lld with %d bodies, %d test particles. Total Energy: %.20g\n", timeStep, N_h[0], Nsmall_h[0], Energy_h[4]);
		}
		else if(st == 0) {
			printf("Reached timestep %lld with %d simulations\n", timeStep, Nst);
			fprintf(masterfile, "Reached timestep %lld with %d simulations\n", timeStep, Nst);
		}
	}
	gettimeofday( &tt2, NULL );
}

//This function prints the total integration runtime
__host__ void Data::printLastTime(){
        gettimeofday(&tt4, NULL );
        times = (tt4.tv_sec - tt1.tv_sec);
        timems = (tt4.tv_usec - tt1.tv_usec);
        for(int st = 0; st < Nst; ++st){
                GSF[st].timefile = fopen(GSF[st].timefilename, "a");
                fprintf(GSF[st].timefile, "\n\n%g\n", times + timems/1000000.0);
		if(st == 0) printf("Execution time: \n\n%g\n", times + timems/1000000.0);
                fclose(GSF[st].timefile);
        }
}


//This function prints the last information
__host__ void Data::LastInfo(){
	for(int st = 0; st < Nst; ++st){
	        GSF[st].logfile = fopen(GSF[st].logfilename, "a");
                fprintf(GSF[st].logfile,"Integration finished with %d bodies, %d test particles. Total Energy: %.20g\n", N_h[st], Nsmall_h[st], Energy_h[4 + NEnergy[st]]);
                fclose (GSF[st].logfile);       
        }
        if(Nst > 1) printf("Integration finished with %d simulations\n", Nst);
        else printf("Integration finished with %d bodies, %d test particles. Total Energy: %.20g\n", N_h[0], Nsmall_h[0], Energy_h[4]);
}

//This function prints details of the Collisions
__host__ void Data::printCollisions(){
  
	cudaMemcpy(Coll_h, Coll_d, sizeof(double)*25*Ncoll_m[0], cudaMemcpyDeviceToHost);
	FILE *collisionfile;
	FILE *logfile;
	for(int nc = 0; nc < Ncoll_m[0]; ++nc){
		int st;
		if(Nst == 1) st = 0;
		else st = (int)(Coll_h[nc * 25 + 1]) / 100;
		collisionfile = fopen(GSF[st].collisionfilename, "a");


		logfile = fopen(GSF[st].logfilename, "a");

		for(int in = 0; in < 25; ++in){
			if(in == 1 || in == 13){
				if(Nst == 1) fprintf(collisionfile, "%d ", (int)(Coll_h[nc * 25 + in]));
				else fprintf(collisionfile, "%d ", ((int)(Coll_h[nc * 25 + in])) % 100);
			}
			else fprintf(collisionfile, "%g ", Coll_h[nc * 25 + in]);
		}
		if(Nst == 1) fprintf(logfile, "Collision between body %d and %d\n", (int)(Coll_h[nc * 25 + 1]), (int)(Coll_h[nc * 25 + 13]));
		else{
			fprintf(logfile, "Collision between body %d and %d\n", (int)(Coll_h[nc * 25 + 1]) % 100 , (int)(Coll_h[nc * 25 + 13]) % 100);
			printf("In Simulation %s: Collision between body %d and %d\n", GSF[st].path, (int)(Coll_h[nc * 25 + 1]) % 100 , (int)(Coll_h[nc * 25 + 13]) % 100);
		}
		fprintf(collisionfile, "\n");
		fclose(collisionfile);
		fclose(logfile);
	}
}


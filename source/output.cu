#include "output.h"

timeval tt1;				//start time
timeval tt2;				//start time of a output time intervall
timeval tt3;				//end time of a output time intervall
timeval tt4;				//end time//

long times, timems;			//elapsed time in seconds and microseconds

//This function prints the initial Energy and Coordinate output
//If Restart is set, then it reads the corespondent initial conditions from the files and writes no output
__host__ int firstoutput(){
	for(int st = 0; st < Nst; ++st){
		if(P.tRestart == 0){
			int NBS = NBS_h[st];
			int NsmallS = NsmallS_h[st];
			GSF[st].Energyfile = fopen(GSF[st].Energyfilename, "a");
			cudaMemcpy(Energy_h + NEnergy[st], Energy_d + NEnergy[st], sizeof(double)*8, cudaMemcpyDeviceToHost);
			fprintf(GSF[st].Energyfile,"%g %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g\n", P.ict, N_h[st] + Nsmall_h[st], Energy_h[0 + NEnergy[st]], Energy_h[1 + NEnergy[st]], Energy_h[2 + NEnergy[st]], Energy_h[3 + NEnergy[st]], Energy_h[4 + NEnergy[st]], Energy_h[5 + NEnergy[st]], Energy_h[6 + NEnergy[st]], Energy_h[7 + NEnergy[st]]);
			fclose(GSF[st].Energyfile);

			if(FormatP == 1){
				if(Nst == 1 || FormatS == 0){
					if(FormatT == 0) sprintf(GSF[st].outputfilename, "%sOut%s_%.12d.dat", GSF[st].path, GSF[st].X, 0);
					if(FormatT == 1) sprintf(GSF[st].outputfilename, "%sOut%s.dat", GSF[st].path, GSF[st].X);
					GSF[st].outputfile = fopen(GSF[st].outputfilename, "w");
				}
				else{
					if(FormatT == 0)sprintf(GSF[st].outputfilename, "%s../Out%s_%.12d.dat", GSF[st].path, GSF[st].X, 0);
					if(FormatT == 1)sprintf(GSF[st].outputfilename, "%s../Out%s.dat", GSF[st].path, GSF[st].X);
					if(st == 0) GSF[st].outputfile = fopen(GSF[st].outputfilename, "w");
					else GSF[st].outputfile = fopen(GSF[st].outputfilename, "a");
				}
			}

			printOutput(x4_h + NBS, v4_h + NBS, index_h + NBS, test_h + NBS, P.ict, 1, N_h[st], GSF[st].outputfile, Msun_h[st], spin_h + NBS, x4small_h + NsmallS, v4small_h + NsmallS, spinsmall_h + NsmallS, indexsmall_h + NsmallS, Nsmall_h[st], Nst, aelimits_h + NBS, aelimitssmall_h + NsmallS, aecount_h + NBS, aecountsmall_h + NsmallS, enccount_h + NBS, enccountsmall_h + NsmallS, aecountT_h + NBS, aecountsmallT_h + NsmallS, enccountT_h + NBS, enccountsmallT_h + NsmallS, P.ci);
			if(FormatP == 1) fclose(GSF[st].outputfile);
		}
		else if(N_h[st] + Nsmall_h[st] > 0){
			GSF[st].Energyfile = fopen(GSF[st].Energyfilename, "r");
			double skip;
			double Et;
			char Ets[160];
			sprintf(Ets, "%.16g", (P.tRestart * P.idt) / 365.25);
			fscanf (GSF[st].Energyfile, "%lf",&Et);
			fscanf (GSF[st].Energyfile, "%lf",&skip);
			fscanf (GSF[st].Energyfile, "%lf",&skip);
			fscanf (GSF[st].Energyfile, "%lf",&skip);
			fscanf (GSF[st].Energyfile, "%lf",&skip);
			fscanf (GSF[st].Energyfile, "%lf",&U_h[st]);
			fscanf (GSF[st].Energyfile, "%lf",&Energy0_h[st]);
			fscanf (GSF[st].Energyfile, "%lf",&skip);
			fscanf (GSF[st].Energyfile, "%lf",&skip);
			fscanf (GSF[st].Energyfile, "%lf",&skip);
			int er = 0;
			while(Et <= atof(Ets)){
				fscanf (GSF[st].Energyfile, "%lf",&Et);
				fscanf (GSF[st].Energyfile, "%lf",&skip);
				fscanf (GSF[st].Energyfile, "%lf",&skip);
				fscanf (GSF[st].Energyfile, "%lf",&skip);
				fscanf (GSF[st].Energyfile, "%lf",&skip);
				fscanf(GSF[st].Energyfile, "%lf",&U_h[st]);
				fscanf (GSF[st].Energyfile, "%lf",&skip);
				fscanf (GSF[st].Energyfile, "%lf",&skip);
				fscanf (GSF[st].Energyfile, "%lf",&skip);
				er = fscanf (GSF[st].Energyfile, "%lf",&skip);
				if(Et >= atof(Ets)) break;

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
		}
	}
	return 1;
}


//**************************************
//This function prints the coordinate output
__host__ void printOutput(double4 *x4_h, double4 *v4_h, int *index_h, double *test_h, double t, long long ts, int N, FILE *outputfile, double Msun, double3 *spin_h, double4 *x4small_h, double4 *v4small_h, double3 *spinsmall_h, int *indexsmall_h, int Nsmall, int Nst, float4 *aelimits_h, float4 *aelimitssmall_h, int *aecount_h, int *aecountsmall_h, int *enccount_h, int *enccountsmall_h, long long *aecountT_h, long long *aecountsmallT_h, long long *enccountT_h, long long *enccountsmallT_h, int ci){

	DemoToHelio(x4_h, v4_h, Msun, N, x4small_h, v4small_h, Nsmall);

	int index;

	for(int j = 0; j < N; j+=1){
		int st = index_h[j] / 100;
		if(FormatP == 0){
			char outputfilename[160];
			if(Nst == 1) sprintf(outputfilename, "%sOut%s_p%.6d.dat", GSF[st].path, GSF[st].X, index_h[j]);
			else sprintf(outputfilename, "%sOut%s_p%.6d.dat", GSF[st].path, GSF[st].X, index_h[j] % 100);
			if(t > P.ict) outputfile = fopen(outputfilename, "a");
			else outputfile = fopen(outputfilename, "w");
		}

		if(Nst == 1 || FormatS == 1) index = index_h[j];
		else index = index_h[j] % 100;

		aecountT_h[j] += aecount_h[j];
		enccountT_h[j] += enccount_h[j];

		fprintf(outputfile,"%.16g %d %.40g %.40g %.40g %.40g %.40g %.40g %.40g %.40g %.40g %.40g %.40g %.8g %.8g %.8g %.8g %.8g %.8g %lld %.40g \n", t, index, x4_h[j].w, v4_h[j].w, x4_h[j].x, x4_h[j].y, x4_h[j].z, v4_h[j].x, v4_h[j].y, v4_h[j].z, spin_h[j].x, spin_h[j].y, spin_h[j].z, aelimits_h[j].x, aelimits_h[j].y, aelimits_h[j].z, aelimits_h[j].w, (double)(aecount_h[j])/ci, (double)(aecountT_h[j])/ts, enccountT_h[j], test_h[j]);
		if(FormatP == 0) fclose(outputfile);
	}
	for(int j = 0; j < Nsmall; j+=1){
		int st = 0;
		if(FormatP == 0){
			char outputfilename[160];
			sprintf(outputfilename, "%sOut%s_p%.6d.dat", GSF[st].path, GSF[st].X, indexsmall_h[j]);
			if(t > P.ict) outputfile = fopen(outputfilename, "a");
			else outputfile = fopen(outputfilename, "w");
		}

		if(Nst == 1) index = indexsmall_h[j];
		else index = indexsmall_h[j] % 100;

		aecountsmallT_h[j] += aecountsmall_h[j];
		enccountsmallT_h[j] += enccountsmall_h[j];

		fprintf(outputfile,"%.16g %d %.40g %.40g %.40g %.40g %.40g %.40g %.40g %.40g %.40g %.40g %.40g %.8g %.8g %.8g %.8g %.8g %.8g %lld %.40g \n", t, index, x4small_h[j].w, v4small_h[j].w, x4small_h[j].x, x4small_h[j].y, x4small_h[j].z, v4small_h[j].x, v4small_h[j].y, v4small_h[j].z, spinsmall_h[j].x, spinsmall_h[j].y, spinsmall_h[j].z, aelimitssmall_h[j].x, aelimitssmall_h[j].y, aelimitssmall_h[j].z, aelimitssmall_h[j].w, (double)(aecountsmall_h[j])/ci, (double)(aecountsmallT_h[j])/ts, enccountsmallT_h[j], -1.0);
		if(FormatP == 0) fclose(outputfile);
	}
}

__host__ void firstInfo(){
	cudaMemcpy(Nencpairs_h, Nencpairs_d, (Nst + 1) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(Nencpairssmall_h, Nencpairssmall_d, (Nst + 1) * sizeof(int), cudaMemcpyDeviceToHost);
	for(int st = 0; st < Nst; ++st){
		GSF[st].logfile = fopen(GSF[st].logfilename, "a");
		if(Nst == 1) fprintf(GSF[st].logfile, "Initial Precheck pairs: %d Test particles: %d\n", Nencpairs_h[0], Nencpairssmall_h[0]);
		else fprintf(GSF[st].logfile, "Initial Precheck pairs: %d Test particles: %d\n", Nencpairs_h[st + 1], Nencpairssmall_h[st + 1]);
		fclose(GSF[st].logfile);
	}
}


//This function calls the Energy function and prints informations
__host__ void EnergyOutput(long long ts, double t){
	cudaStream_t hstream[16];
	for(int hst = 0; hst < 16; ++hst){
		cudaStreamCreate(&hstream[hst]);
	}
#if useGas > 0
	if(Nst == 1) gasEnergy(NB[0], Energy_d, test_d, U_d, hstream[0], 0, N_h[0]);
	else{
		for(int st = 0; st < Nst; ++st){
			int NBS = NBS_h[st];
			gasEnergyM(NB[st], Energy_d + NBS, test_d + NBS, U_d + st, hstream[st%16], st, N_h[st]);
		}
	}
#endif
	for(int st = 0; st < Nst; ++st){
		int NBS = NBS_h[st];
		Energy(NB[st], x4_d + NBS, v4_d + NBS, Msun_h[st], Energy_d + NEnergy[st], test_d + NBS, U_d, Energy0_d, hstream[st%16], st, N_h[st], 1);
	}
	for(int hst = 0; hst < 16; ++hst){
		cudaStreamDestroy(hstream[hst]);
	}

	cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
	for(int st = 0; st < Nst; ++st){
		GSF[st].Energyfile = fopen(GSF[st].Energyfilename, "a");
		cudaMemcpy(Energy_h + NEnergy[st], Energy_d + NEnergy[st], sizeof(double)*8, cudaMemcpyDeviceToHost);
		fprintf(GSF[st].Energyfile,"%.16g %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g\n", t/365.25, N_h[st] + Nsmall_h[st], Energy_h[0 + NEnergy[st]], Energy_h[1 + NEnergy[st]], Energy_h[2 + NEnergy[st]], Energy_h[3 + NEnergy[st]], Energy_h[4 + NEnergy[st]], Energy_h[5 + NEnergy[st]], Energy_h[6 + NEnergy[st]], Energy_h[7 + NEnergy[st]]);
		fclose(GSF[st].Energyfile);
	
		GSF[st].logfile = fopen(GSF[st].logfilename, "a");
		cudaMemcpy(Nencpairs2_h + st + 1, Nencpairs2_d + st + 1, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(Nencpairs_h + st + 1, Nencpairs_d + st + 1, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(Nencpairssmall2_h + st, Nencpairssmall2_d + st, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(Nencpairssmall_h + st + 1, Nencpairssmall_d + st + 1, sizeof(int), cudaMemcpyDeviceToHost);

		if(Nst == 1){
			fprintf(GSF[st].logfile, "    CE: %d; ", Nencpairs2_h[0]);
			fprintf(GSF[0].logfile, "groups: %d, 2: %d, 4: %d, 8: %d, 16: %d, 32: %d, 64: %d, 128: %d, 256: %d, 512: %d, 1024: %d, 2048: %d\n", Nenc_m[0], Nenc_m[1], Nenc_m[2], Nenc_m[3], Nenc_m[4], Nenc_m[5], Nenc_m[6], Nenc_m[7], Nenc_m[8], Nenc_m[9], Nenc_m[10], Nenc_m[11]);
			fprintf(GSF[st].logfile, "    Test Particle CE: %d; ", Nencpairssmall2_h[0]);
			fprintf(GSF[0].logfile, "groups: %d, 2: %d, 4: %d, 8: %d, 16: %d, 32: %d, 64: %d, 128: %d, 256: %d, 512: %d, 1024: %d, 2048: %d\n", Nencsmall_m[0], Nencsmall_m[1], Nencsmall_m[2], Nencsmall_m[3], Nencsmall_m[4], Nencsmall_m[5], Nencsmall_m[6], Nencsmall_m[7], Nencsmall_m[8], Nencsmall_m[9], Nencsmall_m[10], Nencsmall_m[11]);

			fprintf(GSF[0].logfile, "    Precheck-pairs: %d\n    Test Particles Precheck-pairs: %d\n", Nencpairs_h[0], Nencpairssmall_h[0]);
		}
		else{
			fprintf(GSF[st].logfile, "    CE: %d\n    Test Particle CE: %d\n", Nencpairs2_h[st + 1], Nencpairssmall2_h[st + 1]);
			fprintf(GSF[st].logfile, "    Precheck-pairs: %d\n    Test Particles Precheck-pairs: %d\n", Nencpairs_h[st + 1], Nencpairssmall_h[st + 1]);
		}
		fclose(GSF[st].logfile);

	}
	cudaMemset(Energy_d, 0, NEnergyT*sizeof(double));
	
	
	
}

//This function copies the data from the device to host and calls the printoutput function
__host__ void CoordinateOutput(long long ts, double t){
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

	cudaDeviceSynchronize();

	for(int st = 0; st < Nst; ++st){
		int NBS = NBS_h[st];
		int NsmallS = NsmallS_h[st];

		if(FormatP == 1){
			if(Nst == 1 || FormatS == 0){
				if(FormatT == 0){
					sprintf(GSF[st].outputfilename,"%sOut%s_%.12ld.dat", GSF[st].path, GSF[st].X, ts);
					GSF[st].outputfile = fopen(GSF[st].outputfilename, "w");
				}
				if(FormatT == 1){
					sprintf(GSF[st].outputfilename,"%sOut%s.dat", GSF[st].path, GSF[st].X);
					GSF[st].outputfile = fopen(GSF[st].outputfilename, "a");
					}
			}
			else{
				if(FormatT == 0){
					sprintf(GSF[st].outputfilename, "%s../Out%s_%.12d.dat", GSF[st].path, GSF[st].X, ts);
					if(st == 0) GSF[st].outputfile = fopen(GSF[st].outputfilename, "w");
					else GSF[st].outputfile = fopen(GSF[st].outputfilename, "a");
				}
				if(FormatT == 1){
					sprintf(GSF[st].outputfilename, "%s../Out%s.dat", GSF[st].path, GSF[st].X);
					GSF[st].outputfile = fopen(GSF[st].outputfilename, "a");
				}
			}
		}


		printOutput(x4_h + NBS, v4_h + NBS, index_h + NBS, test_h + NBS, t/365.25, ts, N_h[st], GSF[st].outputfile, Msun_h[st], spin_h + NBS, x4small_h + NsmallS, v4small_h + NsmallS, spinsmall_h + NsmallS, indexsmall_h + NsmallS, Nsmall_h[st], Nst, aelimits_h + NBS, aelimitssmall_h + NsmallS, aecount_h + NBS, aecountsmall_h + NsmallS, enccount_h + NBS, enccountsmall_h + NsmallS, aecountT_h + NBS, aecountsmallT_h + NsmallS, enccountT_h + NBS, enccountsmallT_h + NsmallS, P.ci);

		if(FormatP == 1) fclose(GSF[st].outputfile);

	}
	cudaMemcpy(aecountT_d, aecountT_h, sizeof(long long)*NT, cudaMemcpyHostToDevice);
	cudaMemcpy(aecountsmallT_d, aecountsmallT_h, sizeof(long long)*NsmallT, cudaMemcpyHostToDevice);

	cudaMemset(aecount_d, 0, sizeof(int)*NT);
	cudaMemset(aecountsmall_d, 0, sizeof(int)*NsmallT);
}

__host__ void GridaeOutput(long long ts){
	int GridN = Gridae.Na * Gridae.Ne;
	sprintf(Gridae.filename, "aeCount%s_%.12ld.dat", Gridae.X, ts);
	Gridae.file = fopen(Gridae.filename, "w");
	cudaMemcpy(Gridaecount_h, Gridaecount_d, sizeof(int)*GridN, cudaMemcpyDeviceToHost);
	for(int i = 0; i < Gridae.Ne; ++i){
		for(int j = 0; j < Gridae.Na; ++j){
			if(ts > Gridae.Start){
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

	fclose(Gridae.file);
	cudaMemset(Gridaecount_d, 0, sizeof(int)*GridN);
}

//This function prints information if there are too many Collisions which fit not in allocated memory
//It stops the integration
__host__ void printfMaxColl(long long ts){
	printf("Error: Too many Collisions, MaxColl too small. Ncoll = %d. Integration Stopped at timestep = %lld\n", Ncoll_m[0], ts);
	if(Nst ==1){
		GSF[0].logfile = fopen(GSF[0].logfilename, "a");
		fprintf(GSF[0].logfile,"Error: MaxColl too small. Ncoll = %d. Integration Stopped at timestep = %lld\n", Ncoll_m[0], ts);
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
		fprintf(masterfile,"Error: MaxColl too small. Ncoll = %d. Integration Stopped at timestep = %lld\n", Ncoll_m[0], ts);
	}
}

//This function prints information if a too big close encounter group occurs and stops the integrations
__host__ int MaxGroups(long long ts, double t){
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
			printOutput(x4_h, v4_h, index_h, test_h, t/365.25, ts, N_h[0], GSF[0].outputfile, Msun_h[0], spin_h, x4small_h, v4small_h, spinsmall_h, indexsmall_h, Nsmall_h[0], Nst, aelimits_h, aelimitssmall_h, aecount_h, aecountsmall_h, enccount_h, enccountsmall_h, aecountT_h, aecountsmallT_h, enccountT_h, enccountsmallT_h, P.ci);
			fclose(GSF[0].outputfile);

			fprintf(GSF[0].logfile,"Error: Too big group:%g. Integration Stopped at timestep = %lld\n", pow(2.0, nm), ts);
			printf("Error: Too big group:%g. Integration Stopped at timestep = %lld\n", pow(2.0, nm), ts);
			fclose(GSF[0].logfile);
			return 0;
		}
	}
	return 1;
}

//This functions set the starting rutime of the integrations
__host__ void setStartTime(){
	gettimeofday(&tt1, NULL);
	gettimeofday(&tt2, NULL);
	times = 0.0;
	timems = 0.0;
}

//This function prints information how long the integration takes
__host__ void printTime(long long ts){
	gettimeofday( &tt3, NULL );
	for(int st = 0; st < Nst; ++st){
		times = (tt3.tv_sec - tt2.tv_sec);
		timems = (tt3.tv_usec - tt2.tv_usec);

		GSF[st].timefile = fopen(GSF[st].timefilename, "a");
		fprintf(GSF[st].timefile, "%g\n", times + timems/1000000.0);
		GSF[st].logfile = fopen(GSF[st].logfilename, "a");
		fprintf(GSF[st].logfile,"Reached timestep %lld with %d bodies, %d test particles. Total Energy: %.20g\n", ts, N_h[st], Nsmall_h[st], Energy_h[4 + NEnergy[st]]);
		fclose(GSF[st].timefile);
		fclose(GSF[st].logfile);

		if(Nst == 1){
			printf("Reached timestep %lld with %d bodies, %d test particles. Total Energy: %.20g\n", ts, N_h[0], Nsmall_h[0], Energy_h[4]);
			fprintf(masterfile, "Reached timestep %lld with %d bodies, %d test particles. Total Energy: %.20g\n", ts, N_h[0], Nsmall_h[0], Energy_h[4]);
		}
		else if(st == 0) {
			printf("Reached timestep %lld with %d simulations\n", ts, Nst);
			fprintf(masterfile, "Reached timestep %lld with %d simulations\n", ts, Nst);
		}
	}
	gettimeofday( &tt2, NULL );
}

//This function prints the total integration runtime
__host__ void printLastTime(){
        gettimeofday(&tt4, NULL );
        times = (tt4.tv_sec - tt1.tv_sec);
        timems = (tt4.tv_usec - tt1.tv_usec);
        for(int st = 0; st < Nst; ++st){
                GSF[st].timefile = fopen(GSF[st].timefilename, "a");
                fprintf(GSF[st].timefile, "\n\n%g\n", times + timems/1000000.0);
                fclose(GSF[st].timefile);
        }
}

//This function prints the last information
__host__ void LastInfo(){
	for(int st = 0; st < Nst; ++st){
	        GSF[st].logfile = fopen(GSF[st].logfilename, "a");
                fprintf(GSF[st].logfile,"Integration finished with %d bodies, %d test particles. Total Energy: %.20g\n", N_h[st], Nsmall_h[st], Energy_h[4 + NEnergy[st]]);
                fclose (GSF[st].logfile);       
        }
        if(Nst > 1) printf("Integration finished with %d simulations\n", Nst);
        else printf("Integration finished with %d bodies, %d test particles. Total Energy: %.20g\n", N_h[0], Nsmall_h[0], Energy_h[4]);
}

//This function prints details of the Collisions
__host__ void printCollisions(){
  
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

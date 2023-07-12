#include "Orbit2.h"
cudaEvent_t tt1;			//start time
cudaEvent_t tt2;			//start time of a output time intervall
cudaEvent_t tt3;			//end time of a output time intervall
cudaEvent_t tt4;			//end time//

float times;				//elapsed time in milliseconds

// ********************************************3
//This function prints the initial Energy and Coordinate output
//If Restart is set, then it reads the corespondent initial conditions from the files and writes no output
//
//Author: Simon Grimm
//June 2015
// *************************************
__host__ int Data::firstoutput(int irregular){

	char dat_bin[16];
	if(P.OutBinary == 0){
		sprintf(dat_bin, "%s", "dat");
	}
	else{
		sprintf(dat_bin, "%s", "bin");
	}

	for(int st = 0; st < Nst; ++st){
		FILE *Energyfile;

		//check if EnergyIrrfile already exists
		//This is needed for Gasolenga runs
		int readIrrEnergyFile = 0;
		if(irregular == 1){
			FILE *Efile;
			Efile = fopen(GSF[st].EnergyIrrfilename, "r");
			if(Efile != NULL){
				readIrrEnergyFile = 1;
				printf("read initial energy from %s file\n", GSF[st].EnergyIrrfilename);
			}

		}

		if(P.tRestart == 0 && readIrrEnergyFile == 0){
			int NBS = NBS_h[st];
			if(P.ei > 0 || irregular == 1){
				if(irregular == 0){
					Energyfile = fopen(GSF[st].Energyfilename, "a");
				}
				else{
					Energyfile = fopen(GSF[st].EnergyIrrfilename, "a");

				}
				if(Energyfile == NULL){
					printf("Error, Energyfile not valid %d %s\n", st, GSF[st].timefilename);
					return 0;
				}
				cudaMemcpy(Energy_h + NEnergy[st], Energy_d + NEnergy[st], sizeof(double)*8, cudaMemcpyDeviceToHost);
				fprintf(Energyfile,"%.16g %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g\n", ict_h[st], N_h[st] + Nsmall_h[st], Energy_h[0 + NEnergy[st]], Energy_h[1 + NEnergy[st]], Energy_h[2 + NEnergy[st]], Energy_h[3 + NEnergy[st]], Energy_h[4 + NEnergy[st]], Energy_h[5 + NEnergy[st]], Energy_h[6 + NEnergy[st]], Energy_h[7 + NEnergy[st]]);
				fclose(Energyfile);
			}
			if(P.ci > 0){
				if(P.FormatP == 1){
					if(Nst == 1 || P.FormatS == 0){
						//clear Irregular output files
						if(P.FormatT == 0) sprintf(GSF[st].outputfilename, "%sOutIrr%s_%.*d.%s", GSF[st].path, GSF[st].X, def_NFileNameDigits, 0, dat_bin);
						if(P.FormatT == 1) sprintf(GSF[st].outputfilename, "%sOutIrr%s.%s", GSF[st].path, GSF[st].X, dat_bin);
						FILE *file;
						if(P.OutBinary == 0){
							file = fopen(GSF[st].outputfilename, "r");
						}
						else{
							file = fopen(GSF[st].outputfilename, "rb");
						}
						if(file != NULL){
							fclose(file);
							if(P.OutBinary == 0){
								file = fopen(GSF[st].outputfilename, "w");
							}
							else{
								file = fopen(GSF[st].outputfilename, "wb");
							}
							fclose(file);
						}
			
		
						if(P.FormatT == 0) sprintf(GSF[st].outputfilename, "%sOut%s_%.*d.%s", GSF[st].path, GSF[st].X, def_NFileNameDigits, 0, dat_bin);
						if(P.FormatT == 1) sprintf(GSF[st].outputfilename, "%sOut%s.%s", GSF[st].path, GSF[st].X, dat_bin);
#if def_TTV == 0
						if(P.OutBinary == 0){
							GSF[st].outputfile = fopen(GSF[st].outputfilename, "w");
						}
						else{
							GSF[st].outputfile = fopen(GSF[st].outputfilename, "wb");
						}
#else
						if(P.OutBinary == 0){
							if(st == 0) GSF[st].outputfile = fopen(GSF[st].outputfilename, "w");
							else GSF[st].outputfile = fopen(GSF[st].outputfilename, "a");
						}
						else{
							if(st == 0) GSF[st].outputfile = fopen(GSF[st].outputfilename, "wb");
							else GSF[st].outputfile = fopen(GSF[st].outputfilename, "ab");
						}
#endif
					}
					else{
						//clear Irregular output files
						if(P.FormatT == 0)sprintf(GSF[st].outputfilename, "%s../OutIrr%s_%.*d.%s", GSF[st].path, GSF[st].X, def_NFileNameDigits, 0, dat_bin);
						if(P.FormatT == 1)sprintf(GSF[st].outputfilename, "%s../OutIrr%s.%s", GSF[st].path, GSF[st].X, dat_bin);
						FILE *file;
						if(P.OutBinary == 0){
							file = fopen(GSF[st].outputfilename, "r");
						}
						else{
							file = fopen(GSF[st].outputfilename, "rb");
						}
						if(file != NULL){
							fclose(file);
							if(P.OutBinary == 0){
								file = fopen(GSF[st].outputfilename, "w");
							}
							else{
								file = fopen(GSF[st].outputfilename, "wb");
							}
							fclose(file);
						}
				

						if(P.FormatT == 0)sprintf(GSF[st].outputfilename, "%s../Out%s_%.*d.%s", GSF[st].path, GSF[st].X, def_NFileNameDigits, 0, dat_bin);
						if(P.FormatT == 1)sprintf(GSF[st].outputfilename, "%s../Out%s.%s", GSF[st].path, GSF[st].X, dat_bin);
						if(P.OutBinary == 0){
							if(st == 0) GSF[st].outputfile = fopen(GSF[st].outputfilename, "w");
							else GSF[st].outputfile = fopen(GSF[st].outputfilename, "a");
						}
						else{
							if(st == 0) GSF[st].outputfile = fopen(GSF[st].outputfilename, "wb");
							else GSF[st].outputfile = fopen(GSF[st].outputfilename, "ab");
						}
					}
				}
				else{
					//clear Irregular output files
					if(Nst == 1 || P.FormatS == 0){
						for(int i = 0; i < N_h[st] + Nsmall_h[st]; ++i){
							char name[300];
							sprintf(name, "%sOutIrr%s_p%.6d.%s", GSF[st].path, GSF[st].X, i, dat_bin);
							FILE *file;
							if(P.OutBinary == 0){
								file = fopen(name, "r");
							}
							else{
								file = fopen(name, "rb");
							}
							if(file != NULL){
								fclose(file);
								if(P.OutBinary == 0){
									file = fopen(name, "w");
								}
								else{
									file = fopen(name, "wb");
								}
								fclose(file);
							}
						}
					}
					else{
						for(int i = 0; i < N_h[st] + Nsmall_h[st]; ++i){
							char name[300];
							sprintf(name, "%s../OutIrr%s_p%.6d.%s", GSF[st].path, GSF[st].X, i, dat_bin);
							FILE *file;
							if(P.OutBinary == 0){
								file = fopen(name, "r");
							}
							else{
								file = fopen(name, "rb");
							}
							if(file != NULL){
								fclose(file);
								if(P.OutBinary == 0){
									file = fopen(name, "w");
								}
								else{
									file = fopen(name, "wb");
								}
								fclose(file);
							}
						}

					}
				}

				printOutput(x4_h + NBS, v4_h + NBS, v4Helio_h + NBS, index_h + NBS, test_h + NBS, ict_h[st], 1, N_h[st], GSF[st].outputfile, Msun_h[st].x, spin_h + NBS, love_h + NBS, migration_h + NBS, rcrit_h + NBS, Nsmall_h[st], Nst, aelimits_h + NBS, aecount_h + NBS, enccount_h + NBS, aecountT_h + NBS, enccountT_h + NBS, P.ci, 0);
				if(P.FormatP == 1) fclose(GSF[st].outputfile);
			}
		}
		else if(N_h[st] + Nsmall_h[st] > 0){
			int tsign = 1;
			if(idt_h[st] < 0) tsign = -1;
			double skip;
			double Et;
			char Ets[160];
			int er = 0;
			if(readIrrEnergyFile == 0){
				Energyfile = fopen(GSF[st].Energyfilename, "r");
				sprintf(Ets, "%.16g", (P.tRestart * idt_h[st] + ict_h[st] * 365.25) / 365.25);

				er = fscanf (Energyfile, "%lf",&Et);
				er = fscanf (Energyfile, "%lf",&skip);
				er = fscanf (Energyfile, "%lf",&skip);
				er = fscanf (Energyfile, "%lf",&skip);
				er = fscanf (Energyfile, "%lf",&LI_h[st]);
				er = fscanf (Energyfile, "%lf",&U_h[st]);
				er = fscanf (Energyfile, "%lf",&Energy0_h[st]);
				er = fscanf (Energyfile, "%lf",&LI0_h[st]);
				er = fscanf (Energyfile, "%lf",&skip);
				er = fscanf (Energyfile, "%lf",&skip);
	
				U_h[st] /= def_Kg;
				LI_h[st] /= dayUnit;
			}
			else{
				//read only initial energy and angular momentum
				Energyfile = fopen(GSF[st].EnergyIrrfilename, "r");
				sprintf(Ets, "%.16g", (ict_h[st] * 365.25) / 365.25);

				er = fscanf (Energyfile, "%lf",&Et);
				er = fscanf (Energyfile, "%lf",&skip);
				er = fscanf (Energyfile, "%lf",&skip);
				er = fscanf (Energyfile, "%lf",&skip);
				er = fscanf (Energyfile, "%lf",&skip);
				er = fscanf (Energyfile, "%lf",&skip);
				er = fscanf (Energyfile, "%lf",&Energy0_h[st]);
				er = fscanf (Energyfile, "%lf",&LI0_h[st]);
				er = fscanf (Energyfile, "%lf",&skip);
				er = fscanf (Energyfile, "%lf",&skip);
			}

//printf("%.20g %.20g %d %d\n", Et, atof(Ets), tsign, er);
			while(Et * tsign < atof(Ets) * tsign){
				er = fscanf (Energyfile, "%lf",&Et);
				er = fscanf (Energyfile, "%lf",&skip);
				er = fscanf (Energyfile, "%lf",&skip);
				er = fscanf (Energyfile, "%lf",&skip);
				er = fscanf (Energyfile, "%lf",&LI_h[st]);
				er = fscanf (Energyfile, "%lf",&U_h[st]);
				er = fscanf (Energyfile, "%lf",&skip);
				er = fscanf (Energyfile, "%lf",&skip);
				er = fscanf (Energyfile, "%lf",&skip);
				er = fscanf (Energyfile, "%lf",&skip);
//printf("%.20g %.20g %d %d\n", Et, atof(Ets), tsign, er);
	
				U_h[st] /= def_Kg;
				LI_h[st] /= dayUnit;

				if(Et * tsign >= atof(Ets) * tsign) break;

				if(er <= 0){
					break;
				}				
			}		
			if(er <= 0){
				fprintf(masterfile, "Error: In Simulation %s: Restart time step not valid %g %g\n", GSF[st].path, atof(Ets), Et);
				printf("Error: In Simulation %s: Restart time step not valid %g %g\n", GSF[st].path, atof(Ets), Et);
				return 0;
			}
//printf("Energy %g %g %g %g\n", Energy0_h[0], U_h[0] * def_Kg, LI0_h[0], LI_h[0] * dayUnit);


			fclose(Energyfile);
			cudaMemcpy(Energy0_d + st, Energy0_h + st, sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(U_d + st, U_h + st, sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(LI_d + st, LI_h + st, sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(LI0_d + st, LI0_h + st, sizeof(double), cudaMemcpyHostToDevice);

			if(irregular == 0 && (P.UseTides > 0 || P.UseRotationalDeformation > 0)){
				//print star file
				FILE *starfile;
				int er = 0;
				starfile = fopen(GSF[st].starfilename, "r");

				er = fscanf (Energyfile, "%lf",&Et);
				er = fscanf (Energyfile, "%lf",&Msun_h[st].x);
				er = fscanf (Energyfile, "%lf",&Msun_h[st].y);
				er = fscanf (Energyfile, "%lf",&Spinsun_h[st].x);
				er = fscanf (Energyfile, "%lf",&Spinsun_h[st].y);
				er = fscanf (Energyfile, "%lf",&Spinsun_h[st].z);
				er = fscanf (Energyfile, "%lf",&Spinsun_h[st].w);
				er = fscanf (Energyfile, "%lf",&Lovesun_h[st].x);
				er = fscanf (Energyfile, "%lf",&Lovesun_h[st].y);
				er = fscanf (Energyfile, "%lf",&Lovesun_h[st].z);

//printf("%.20g %.20g %d %d\n", Et, atof(Ets), tsign, er);
				while(Et * tsign < atof(Ets) * tsign){
					er = fscanf (Energyfile, "%lf",&Et);
					er = fscanf (Energyfile, "%lf",&Msun_h[st].x);
					er = fscanf (Energyfile, "%lf",&Msun_h[st].y);
					er = fscanf (Energyfile, "%lf",&Spinsun_h[st].x);
					er = fscanf (Energyfile, "%lf",&Spinsun_h[st].y);
					er = fscanf (Energyfile, "%lf",&Spinsun_h[st].z);
					er = fscanf (Energyfile, "%lf",&Spinsun_h[st].w);
					er = fscanf (Energyfile, "%lf",&Lovesun_h[st].x);
					er = fscanf (Energyfile, "%lf",&Lovesun_h[st].y);
					er = fscanf (Energyfile, "%lf",&Lovesun_h[st].z);
//printf("%.20g %.20g %d %d\n", Et, atof(Ets), tsign, er);
					if(Et * tsign >= atof(Ets) * tsign) break;

					if(er <= 0){
						break;
					}				
				}
				if(er <= 0){
					fprintf(masterfile, "Error: In Simulation %s: Restart time step not valid for star file %g %g\n", GSF[st].path, atof(Ets), Et);
					printf("Error: In Simulation %s: Restart time step not valid for star file %g %g\n", GSF[st].path, atof(Ets), Et);
					return 0;
				}
	
				cudaMemcpy(Msun_d + st, Msun_h + st, sizeof(double2), cudaMemcpyHostToDevice);
				cudaMemcpy(Spinsun_d + st, Spinsun_h + st, sizeof(double4), cudaMemcpyHostToDevice);
				cudaMemcpy(Lovesun_d + st, Lovesun_h + st, sizeof(double3), cudaMemcpyHostToDevice);
//printf("Spin %g %g\n", Et, Spinsun_h[st].z);
				fclose(starfile);
			}
		}
	}
	return 1;
}


//**************************************
//This function prints the coordinate output
//Authors: Simon Grimm, Joachim Stadel
//March 2014
// ***************************************
__host__ void Data::printOutput(double4 *x4_h, double4 *v4_h, double4 *v4Helio_h, int *index_h, double *test_h, double time, long long timeStep, int N, FILE *outputfile, double Msun, double4 *spin_h, double3 *love_h, double3 *migration_h, double *rcrit_h, int Nsmall, int Nst, float4 *aelimits_h, unsigned int *aecount_h, unsigned int *enccount_h, unsigned long long *aecountT_h, unsigned long long *enccountT_h, int ci, int irregular){

	DemoToHelio(x4_h, v4_h, v4Helio_h, Msun, N + Nsmall);
	//BaryToHelio(x4_h, v4_h, Msun, N + Nsmall);

	int index;
	int st = 0;

	char dat_bin[16];
	if(P.OutBinary == 0){
		sprintf(dat_bin, "%s", "dat");
	}
	else{
		sprintf(dat_bin, "%s", "bin");
	}

	for(int j = 0; j < N + Nsmall; j+=1){
		if(Nst > 1) st = index_h[j] / def_MaxIndex;
		if(P.FormatP == 0){
			char outputfilename[300];
			if(Nst == 1){
				if(irregular == 0 || irregular == 3){
					sprintf(outputfilename, "%sOut%s_p%.6d.%s", GSF[st].path, GSF[st].X, index_h[j], dat_bin);
				}
				else{
					sprintf(outputfilename, "%sOutIrr%s_p%.6d.%s", GSF[st].path, GSF[st].X, index_h[j], dat_bin);
				}
			}
			else{
				if(irregular == 0 || irregular == 3){
					sprintf(outputfilename, "%sOut%s_p%.6d.%s", GSF[st].path, GSF[st].X, index_h[j] % def_MaxIndex, dat_bin);
				}
				else{
					sprintf(outputfilename, "%sOutIrr%s_p%.6d.%s", GSF[st].path, GSF[st].X, index_h[j] % def_MaxIndex, dat_bin);

				}
			}
			if((time > ict_h[st] && idt_h[st] > 0.0) || (time < ict_h[st] && idt_h[st] < 0.0)){
				if(P.OutBinary == 0){
					outputfile = fopen(outputfilename, "a");
				}
				else{
					outputfile = fopen(outputfilename, "ab");
				}
			}
			else{
				if(P.OutBinary == 0){
					outputfile = fopen(outputfilename, "w");
				}
				else{
					outputfile = fopen(outputfilename, "wb");
				}
			}
		}
#if def_TTV == 0
		if(Nst == 1 || P.FormatS == 1) index = index_h[j];
		else index = index_h[j] % def_MaxIndex;
#else
		index = index_h[j];
#endif

		aecountT_h[j] += aecount_h[j];
		enccountT_h[j] += enccount_h[j];

		if(x4_h[j].w >= 0.0){
			if(P.OutBinary == 0){
				//fprintf(outputfile,"%.16g %d %.40g %.40g %.40g %.40g %.40g %.40g %.40g %.40g %.40g %.40g %.40g %.8g %.8g %.8g %.8g %.8g %.8g %lld %.40g \n", time, index, x4_h[j].w, v4Helio_h[j].w, x4_h[j].x, x4_h[j].y, x4_h[j].z, v4Helio_h[j].x, v4Helio_h[j].y, v4Helio_h[j].z, spin_h[j].x, spin_h[j].y, spin_h[j].z, aelimits_h[j].x, aelimits_h[j].y, aelimits_h[j].z, aelimits_h[j].w, (double)(aecount_h[j])/ci, (double)(aecountT_h[j])/timeStep, enccountT_h[j], test_h[j]);
				for(int f = 0; f < def_Ninformat; ++f){
					if(GSF[st].outformat[f] == 19){
						fprintf(outputfile,"%.16g ", time);
					}
					if(GSF[st].outformat[f] == 13){
						fprintf(outputfile,"%d ", index);
					}
					if(GSF[st].outformat[f] == 4){
						fprintf(outputfile,"%.40g ", x4_h[j].w);
					}
					if(GSF[st].outformat[f] == 8){
						fprintf(outputfile,"%.40g ", v4Helio_h[j].w);
					}
					if(GSF[st].outformat[f] == 1){
						fprintf(outputfile,"%.40g ", x4_h[j].x);
					}
					if(GSF[st].outformat[f] == 2){
						fprintf(outputfile,"%.40g ", x4_h[j].y);
					}
					if(GSF[st].outformat[f] == 3){
						fprintf(outputfile,"%.40g ", x4_h[j].z);
					}
					if(GSF[st].outformat[f] == 5){
						fprintf(outputfile,"%.40g ", v4Helio_h[j].x);
					}
					if(GSF[st].outformat[f] == 6){
						fprintf(outputfile,"%.40g ", v4Helio_h[j].y);
					}
					if(GSF[st].outformat[f] == 7){
						fprintf(outputfile,"%.40g ", v4Helio_h[j].z);
					}
					if(GSF[st].outformat[f] == 10){
						fprintf(outputfile,"%.40g ", spin_h[j].x);
					}
					if(GSF[st].outformat[f] == 11){
						fprintf(outputfile,"%.40g ", spin_h[j].y);
					}
					if(GSF[st].outformat[f] == 12){
						fprintf(outputfile,"%.40g ", spin_h[j].z);
					}
					if(GSF[st].outformat[f] == 15){
						fprintf(outputfile,"%.8g ", aelimits_h[j].x);
					}
					if(GSF[st].outformat[f] == 16){
						fprintf(outputfile,"%.8g ", aelimits_h[j].y);
					}
					if(GSF[st].outformat[f] == 17){
						fprintf(outputfile,"%.8g ", aelimits_h[j].z);
					}
					if(GSF[st].outformat[f] == 18){
						fprintf(outputfile,"%.8g ", aelimits_h[j].w);
					}
					if(GSF[st].outformat[f] == 20){
						fprintf(outputfile,"%.40g ", love_h[j].x);
					}
					if(GSF[st].outformat[f] == 21){
						fprintf(outputfile,"%.40g ", love_h[j].y);
					}
					if(GSF[st].outformat[f] == 22){
						fprintf(outputfile,"%.40g ", love_h[j].z);
					}
					if(GSF[st].outformat[f] == 47){
						fprintf(outputfile,"%.8g ", (double)(aecount_h[j])/ci);
					}
					if(GSF[st].outformat[f] == 48){
						fprintf(outputfile,"%.8g ", (double)(aecountT_h[j])/timeStep);
					}
					if(GSF[st].outformat[f] == 46){
						fprintf(outputfile,"%llu ", enccountT_h[j]);
					}
					if(GSF[st].outformat[f] == 42){
						fprintf(outputfile,"%.40g ", rcrit_h[j]);
					}
					if(GSF[st].outformat[f] == 44){
						fprintf(outputfile,"%.40g ", spin_h[j].w);
					}
					if(GSF[st].outformat[f] == 45){
						fprintf(outputfile,"%.40g ", test_h[j]);
					}	
					if(P.UseMigrationForce > 0){
						if(GSF[st].outformat[f] == 49){
							fprintf(outputfile,"%.40g ", migration_h[j].x);
						}
						if(GSF[st].outformat[f] == 50){
							fprintf(outputfile,"%.40g ", migration_h[j].y);
						}
						if(GSF[st].outformat[f] == 51){
							fprintf(outputfile,"%.40g ", migration_h[j].z);
						}
					}
					else{
						if(GSF[st].outformat[f] == 49){
							fprintf(outputfile,"%.40g ", 0.0);
						}
						if(GSF[st].outformat[f] == 50){
							fprintf(outputfile,"%.40g ", 0.0);
						}
						if(GSF[st].outformat[f] == 51){
							fprintf(outputfile,"%.40g ", 0.0);
						}

					}
				}
				fprintf(outputfile,"\n");
			}
			else{
				float aecount = (double)(aecount_h[j])/ci;
				float aecountT = (double)(aecountT_h[j])/timeStep;

				for(int f = 0; f < def_Ninformat; ++f){
					if(GSF[st].outformat[f] == 19){
						fwrite(&time, sizeof(double), 1, outputfile);
					}
					if(GSF[st].outformat[f] == 13){
						fwrite(&index, sizeof(int), 1, outputfile);
					}
					if(GSF[st].outformat[f] == 4){
						fwrite(&x4_h[j].w, sizeof(double), 1, outputfile);
					}
					if(GSF[st].outformat[f] == 8){
						fwrite(&v4Helio_h[j].w, sizeof(double), 1, outputfile);
					}
					if(GSF[st].outformat[f] == 1){
						fwrite(&x4_h[j].x, sizeof(double), 1, outputfile);
					}
					if(GSF[st].outformat[f] == 2){
						fwrite(&x4_h[j].y, sizeof(double), 1, outputfile);
					}
					if(GSF[st].outformat[f] == 3){
						fwrite(&x4_h[j].z, sizeof(double), 1, outputfile);
					}
					if(GSF[st].outformat[f] == 5){
						fwrite(&v4Helio_h[j].x, sizeof(double), 1, outputfile);
					}
					if(GSF[st].outformat[f] == 6){
						fwrite(&v4Helio_h[j].y, sizeof(double), 1, outputfile);
					}
					if(GSF[st].outformat[f] == 7){
						fwrite(&v4Helio_h[j].z, sizeof(double), 1, outputfile);
					}
					if(GSF[st].outformat[f] == 10){
						fwrite(&spin_h[j].x, sizeof(double), 1, outputfile);
					}
					if(GSF[st].outformat[f] == 11){
						fwrite(&spin_h[j].y, sizeof(double), 1, outputfile);
					}
					if(GSF[st].outformat[f] == 12){
						fwrite(&spin_h[j].z, sizeof(double), 1, outputfile);
					}
					if(GSF[st].outformat[f] == 15){
						fwrite(&aelimits_h[j].x, sizeof(float), 1, outputfile);
					}
					if(GSF[st].outformat[f] == 16){
						fwrite(&aelimits_h[j].y, sizeof(float), 1, outputfile);
					}
					if(GSF[st].outformat[f] == 17){
						fwrite(&aelimits_h[j].z, sizeof(float), 1, outputfile);
					}
					if(GSF[st].outformat[f] == 18){
						fwrite(&aelimits_h[j].w, sizeof(float), 1, outputfile);
					}
					if(GSF[st].outformat[f] == 20){
						fwrite(&love_h[j].x, sizeof(double), 1, outputfile);
					}
					if(GSF[st].outformat[f] == 21){
						fwrite(&love_h[j].y, sizeof(double), 1, outputfile);
					}
					if(GSF[st].outformat[f] == 22){
						fwrite(&love_h[j].z, sizeof(double), 1, outputfile);
					}
					if(GSF[st].outformat[f] == 47){
						fwrite(&aecount, sizeof(float), 1, outputfile);
					}
					if(GSF[st].outformat[f] == 48){
						fwrite(&aecountT, sizeof(float), 1, outputfile);
					}
					if(GSF[st].outformat[f] == 46){
						fwrite(&enccountT_h[j], sizeof(unsigned long long), 1, outputfile);
					}
					if(GSF[st].outformat[f] == 42){
						fwrite(&rcrit_h[j], sizeof(double), 1, outputfile);
					}
					if(GSF[st].outformat[f] == 44){
						fwrite(&spin_h[j].w, sizeof(double), 1, outputfile);
					}
					if(GSF[st].outformat[f] == 45){
						fwrite(&test_h[j], sizeof(double), 1, outputfile);
					}
					if(P.UseMigrationForce > 0){
						if(GSF[st].outformat[f] == 49){
							fwrite(&migration_h[j].x, sizeof(double), 1, outputfile);
						}
						if(GSF[st].outformat[f] == 50){
							fwrite(&migration_h[j].y, sizeof(double), 1, outputfile);
						}
						if(GSF[st].outformat[f] == 51){
							fwrite(&migration_h[j].z, sizeof(double), 1, outputfile);
						}
					}
					else{
						double d = 0.0;
						if(GSF[st].outformat[f] == 49){
							fwrite(&d, sizeof(double), 1, outputfile);
						}
						if(GSF[st].outformat[f] == 50){
							fwrite(&d, sizeof(double), 1, outputfile);
						}
						if(GSF[st].outformat[f] == 51){
							fwrite(&d, sizeof(double), 1, outputfile);
						}
					}
				}
			}
		}
		if(P.FormatP == 0) fclose(outputfile);
	}

	if(P.UseTides > 0 || P.UseRotationalDeformation > 0){
		for(int st = 0; st < Nst; ++st){
			//print star file
			FILE *starfile;
			if(irregular == 0 || irregular == 3){ 
				starfile = fopen(GSF[st].starfilename, "a");
			}
			else{
				starfile = fopen(GSF[st].starIrrfilename, "a");
			}
			cudaMemcpy(Msun_h + st, Msun_d + st, sizeof(double2), cudaMemcpyDeviceToHost);
			cudaMemcpy(Spinsun_h + st, Spinsun_d + st, sizeof(double4), cudaMemcpyDeviceToHost);
			cudaMemcpy(Lovesun_h + st, Lovesun_d + st, sizeof(double3), cudaMemcpyDeviceToHost);
			fprintf(starfile, "%.16g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g\n", time, Msun_h[st].x, Msun_h[st].y, Spinsun_h[st].x, Spinsun_h[st].y, Spinsun_h[st].z, Spinsun_h[st].w, Lovesun_h[st].x, Lovesun_h[st].y, Lovesun_h[st].z);

			fclose(starfile);
		}
	}
}

//this function prints the first close encounter information to the info file, partA
__host__ void Data::firstInfo(){
	cudaMemcpy(Nencpairs_h, Nencpairs_d, (Nst + 1) * sizeof(int), cudaMemcpyDeviceToHost);
	for(int st = 0; st < Nst; ++st){
		GSF[st].logfile = fopen(GSF[st].logfilename, "a");
		if(Nst == 1) fprintf(GSF[st].logfile, "Initial Precheck pairs: %d\n", Nencpairs_h[0]);
		else fprintf(GSF[st].logfile, "Initial Precheck pairs: %d\n", Nencpairs_h[st + 1]);
		fclose(GSF[st].logfile);
		if(MTFlag == 1) break;
	}
}

//this function prints the first close encounter information to the info file, partB
__host__ void Data::firstInfoB(){
	for(int st = 0; st < Nst; ++st){
		GSF[st].logfile = fopen(GSF[st].logfilename, "a");
		cudaMemcpy(Nencpairs2_h + st + 1, Nencpairs2_d + st + 1, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(Nencpairs_h + st + 1, Nencpairs_d + st + 1, sizeof(int), cudaMemcpyDeviceToHost);

		if(Nst == 1){
			fprintf(GSF[0].logfile, "    CE:    %d; ", Nencpairs2_h[0]);
			fprintf(GSF[0].logfile, "groups: %d; ", Nenc_m[0]);
			int nn = 2;
			for(int st = 1; st < def_GMax; ++st){
				if(Nenc_m[st] > 0) fprintf(GSF[0].logfile, "%d: %d; ", nn, Nenc_m[st]);
				nn *= 2;
			}
			fprintf(GSF[0].logfile, "\n");

			fprintf(GSF[0].logfile, "    Precheck-pairs:    %d\n", Nencpairs_h[0]);
		}
		else{
			fprintf(GSF[st].logfile, "    CE:    %d\n", Nencpairs2_h[st + 1]);
			fprintf(GSF[st].logfile, "    Precheck-pairs:    %d\n", Nencpairs_h[st + 1]);
		}
		if(interrupt == 1){
			fprintf(GSF[st].logfile, "GENGA is terminated by SIGINT signal at time step %lld\n", timeStep);

		}
		fclose(GSF[st].logfile);
	}
}

__host__ int Data::firstEnergy(){

	for(int st = 0; st < Nst; ++st){
		EnergyCall(st, 0);
	}
	cudaDeviceSynchronize();
	error = cudaGetLastError();
	fprintf(masterfile,"Energy error = %d = %s\n",error, cudaGetErrorString(error));
	if(error != 0){
		printf("Energy error = %d = %s\n",error, cudaGetErrorString(error));
		return 0;
	}
	return 1;
}

//This function calls the Energy function and prints information
__host__ int Data::EnergyOutput(int irregular){
	FILE *Energyfile;
	for(int hst = 0; hst < 16; ++hst){
		error = cudaGetLastError();
		if(error != 0){
			printf("Energy Stream error = %d = %s %lld\n",error, cudaGetErrorString(error), timeStep);
			return 0;
		}
	}
	if(P.Usegas == 1){
		if(Nst == 1){
			gasEnergyCall();
		}
		else{
#if def_CPU == 0
			for(int st = 0; st < Nst; ++st){
				gasEnergyMCall(st);
			}
#endif
		}
	}
	for(int st = 0; st < Nst; ++st){
		EnergyCall(st, 1);
	}

	cudaDeviceSynchronize();
	error = cudaGetLastError();
	if(error != 0){
		printf("Energy error = %d = %s %lld\n",error, cudaGetErrorString(error), timeStep);
		return 0;
	}

	if(Nst > 1) cudaMemcpy(time_h, time_d, Nst*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(Energy_h, Energy_d, sizeof(double) * NEnergyT, cudaMemcpyDeviceToHost);
	cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
	for(int st = 0; st < Nst; ++st){

//printf("Print Energy | irregular: %d st: %d n1: %g\n", irregular, st, n1_h[st]);
		if(Nst > 1){
			int s = 0;
		
			if(irregular < 3) s = 1;	
			if(N_h[st] < Nmin[st].x) s = 1;
			if(Nsmall_h[st] < Nmin[st].y) s = 1;
			if(n1_h[st] < 0) s = 1;
			if(timeStep >= delta_h[st]) s = 1;
			//print only simulations which must be stopped by StopAtEncounter
			//or when the simulation reached the end
			if(s == 0){
				continue;
			}			
		}
//printf("Print Energy2 | irregular: %d st: %d n1: %g\n", irregular, st, n1_h[st]);

		if(irregular == 0 || irregular == 3){
			Energyfile = fopen(GSF[st].Energyfilename, "a");
		}
		else{
			Energyfile = fopen(GSF[st].EnergyIrrfilename, "a");
		}
		if(Energyfile == NULL){
			printf("Error, Energyfile not valid %d %s\n", st, GSF[st].Energyfilename);
			return 0;
		}
		int NE = NEnergy[st];
		fprintf(Energyfile,"%.16g %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g\n", time_h[st]/365.25, N_h[st] + Nsmall_h[st], Energy_h[0 + NE], Energy_h[1 + NE], Energy_h[2 + NE], Energy_h[3 + NE], Energy_h[4 + NE], Energy_h[5 + NE], Energy_h[6 + NE], Energy_h[7 + NE]);
		fclose(Energyfile);

		if(irregular == 0 || interrupt == 1){
			GSF[st].logfile = fopen(GSF[st].logfilename, "a");
			cudaMemcpy(Nencpairs2_h + st + 1, Nencpairs2_d + st + 1, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(Nencpairs_h + st + 1, Nencpairs_d + st + 1, sizeof(int), cudaMemcpyDeviceToHost);

			if(Nst == 1){
				fprintf(GSF[0].logfile, "    CE:    %d; ", Nencpairs2_h[0]);
				fprintf(GSF[0].logfile, "groups: %d; ", Nenc_m[0]);
				int nn = 2;
				for(int st = 1; st < def_GMax; ++st){
					if(Nenc_m[st] > 0) fprintf(GSF[0].logfile, "%d: %d; ", nn, Nenc_m[st]);
					nn *= 2;
				}
				fprintf(GSF[0].logfile, "\n");

				fprintf(GSF[0].logfile, "    Precheck-pairs:    %d\n", Nencpairs_h[0]);
			}
			else{
				fprintf(GSF[st].logfile, "    CE:    %d\n", Nencpairs2_h[st + 1]);
				fprintf(GSF[st].logfile, "    Precheck-pairs:    %d\n", Nencpairs_h[st + 1]);
			}
			if(interrupt == 1){
				fprintf(GSF[st].logfile, "GENGA is terminated by SIGINT signal at time step %lld\n", timeStep);

			}
			fclose(GSF[st].logfile);
		}

	}
	
	return 1;
}


__global__ void CoordinateToBuffer_kernel(double4 *x4_d, double4 *v4_d, int *index_d, double4 *spin_d, double3 *love_d, double3 *migration_d, double *rcrit_d, float4 *aelimits_d, unsigned int* aecount_d, unsigned long long *aecountT_d, unsigned long long *enccountT_d, double *test_d, double *coordinateBuffer_d, double *time_d, double *idt_d, const int Nst, const int NT, const int NsmallT, const int NconstT, const int bufferCount, const double dTau, const int UseMigrationForce){

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < NT + NsmallT){
		//time
		if(Nst == 1){
			coordinateBuffer_d[def_BufferSize * NconstT * bufferCount + def_BufferSize * id] = time_d[0] + dTau * idt_d[0];
		}
		else{
			int st = index_d[id] / def_MaxIndex;
			coordinateBuffer_d[def_BufferSize * NconstT * bufferCount + def_BufferSize * id] = time_d[st] + dTau * idt_d[st];
		}
		coordinateBuffer_d[def_BufferSize * NconstT * bufferCount + def_BufferSize * id + 1] = index_d[id];
		coordinateBuffer_d[def_BufferSize * NconstT * bufferCount + def_BufferSize * id + 2] = x4_d[id].w;
		coordinateBuffer_d[def_BufferSize * NconstT * bufferCount + def_BufferSize * id + 3] = v4_d[id].w;
		coordinateBuffer_d[def_BufferSize * NconstT * bufferCount + def_BufferSize * id + 4] = x4_d[id].x;
		coordinateBuffer_d[def_BufferSize * NconstT * bufferCount + def_BufferSize * id + 5] = x4_d[id].y;
		coordinateBuffer_d[def_BufferSize * NconstT * bufferCount + def_BufferSize * id + 6] = x4_d[id].z;
		coordinateBuffer_d[def_BufferSize * NconstT * bufferCount + def_BufferSize * id + 7] = v4_d[id].x;
		coordinateBuffer_d[def_BufferSize * NconstT * bufferCount + def_BufferSize * id + 8] = v4_d[id].y;
		coordinateBuffer_d[def_BufferSize * NconstT * bufferCount + def_BufferSize * id + 9] = v4_d[id].z;
		coordinateBuffer_d[def_BufferSize * NconstT * bufferCount + def_BufferSize * id + 10] = spin_d[id].x;
		coordinateBuffer_d[def_BufferSize * NconstT * bufferCount + def_BufferSize * id + 11] = spin_d[id].y;
		coordinateBuffer_d[def_BufferSize * NconstT * bufferCount + def_BufferSize * id + 12] = spin_d[id].z;
		coordinateBuffer_d[def_BufferSize * NconstT * bufferCount + def_BufferSize * id + 13] = aelimits_d[id].x;
		coordinateBuffer_d[def_BufferSize * NconstT * bufferCount + def_BufferSize * id + 14] = aelimits_d[id].y;
		coordinateBuffer_d[def_BufferSize * NconstT * bufferCount + def_BufferSize * id + 15] = aelimits_d[id].z;
		coordinateBuffer_d[def_BufferSize * NconstT * bufferCount + def_BufferSize * id + 16] = aelimits_d[id].w;
		coordinateBuffer_d[def_BufferSize * NconstT * bufferCount + def_BufferSize * id + 17] = aecount_d[id];
		coordinateBuffer_d[def_BufferSize * NconstT * bufferCount + def_BufferSize * id + 18] = aecountT_d[id];
		coordinateBuffer_d[def_BufferSize * NconstT * bufferCount + def_BufferSize * id + 19] = enccountT_d[id];
		coordinateBuffer_d[def_BufferSize * NconstT * bufferCount + def_BufferSize * id + 20] = test_d[id];
		coordinateBuffer_d[def_BufferSize * NconstT * bufferCount + def_BufferSize * id + 21] = spin_d[id].w;
		coordinateBuffer_d[def_BufferSize * NconstT * bufferCount + def_BufferSize * id + 22] = love_d[id].x;
		coordinateBuffer_d[def_BufferSize * NconstT * bufferCount + def_BufferSize * id + 23] = love_d[id].y;
		coordinateBuffer_d[def_BufferSize * NconstT * bufferCount + def_BufferSize * id + 24] = love_d[id].z;
		coordinateBuffer_d[def_BufferSize * NconstT * bufferCount + def_BufferSize * id + 25] = rcrit_d[id];
		if(UseMigrationForce > 0){
			coordinateBuffer_d[def_BufferSize * NconstT * bufferCount + def_BufferSize * id + 26] = migration_d[id].x;
			coordinateBuffer_d[def_BufferSize * NconstT * bufferCount + def_BufferSize * id + 27] = migration_d[id].y;
			coordinateBuffer_d[def_BufferSize * NconstT * bufferCount + def_BufferSize * id + 28] = migration_d[id].z;
		}
	}
}

__host__ void Data::CoordinateToBuffer(int bufferCount, int irregular, double dTau){
	if(NT + NsmallT > 0){
		if(irregular == 0){
			CoordinateToBuffer_kernel <<< (NT + NsmallT + 511) / 512, 512 >>> (x4_d, v4_d, index_d, spin_d, love_d, migration_d, rcrit_d, aelimits_d, aecount_d, aecountT_d, enccountT_d, test_d, coordinateBuffer_d, time_d, idt_d, Nst, NT, NsmallT, NconstT, bufferCount, dTau, P.UseMigrationForce);
		}
		else{
			CoordinateToBuffer_kernel <<< (NT + NsmallT + 511) / 512, 512 >>> (x4_d, v4_d, index_d, spin_d, love_d, migration_d, rcrit_d, aelimits_d, aecount_d, aecountT_d, enccountT_d, test_d, coordinateBufferIrr_d, time_d, idt_d, Nst, NT, NsmallT, NconstT, bufferCount, dTau, P.UseMigrationForce);

		}
	}
}

//This function copies the data from the device to host and calls the printoutput function
//irregular indicates irregular output intervals, which are read from a calendar file
//irregular = 2 means to print Coordinates at Collision time
//irregular = 3 means to print the last time step
//irregular = 4 means Step Error output
__host__ void Data::CoordinateOutput(int irregular){
	cudaMemcpy(x4_h, x4_d, sizeof(double4)*NconstT, cudaMemcpyDeviceToHost);
	cudaMemcpy(v4_h, v4_d, sizeof(double4)*NconstT, cudaMemcpyDeviceToHost);
	cudaMemcpy(index_h, index_d, sizeof(int)*NconstT, cudaMemcpyDeviceToHost);
	cudaMemcpy(test_h, test_d, sizeof(double)*NconstT, cudaMemcpyDeviceToHost);
	cudaMemcpy(spin_h, spin_d, sizeof(double4)*NconstT, cudaMemcpyDeviceToHost);
	cudaMemcpy(love_h, love_d, sizeof(double3)*NconstT, cudaMemcpyDeviceToHost);
	if(P.UseMigrationForce > 0){
		cudaMemcpy(migration_h, migration_d, sizeof(double3)*NconstT, cudaMemcpyDeviceToHost);
	}
	cudaMemcpy(aelimits_h, aelimits_d, sizeof(float4)*NconstT, cudaMemcpyDeviceToHost);
	cudaMemcpy(aecount_h, aecount_d, sizeof(unsigned int)*NconstT, cudaMemcpyDeviceToHost);
	cudaMemcpy(enccount_h, enccount_d, sizeof(unsigned int)*NconstT, cudaMemcpyDeviceToHost);
	cudaMemcpy(aecountT_h, aecountT_d, sizeof(unsigned long long)*NconstT, cudaMemcpyDeviceToHost);
	cudaMemcpy(enccountT_h, enccountT_d, sizeof(unsigned long long)*NconstT, cudaMemcpyDeviceToHost);
	cudaMemcpy(rcrit_h, rcrit_d, sizeof(double) * NconstT * P.SLevels, cudaMemcpyDeviceToHost);

	if(Nst > 1) cudaMemcpy(time_h, time_d, Nst * sizeof(double), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	char dat_bin[16];
	if(P.OutBinary == 0){
		sprintf(dat_bin, "%s", "dat");
	}
	else{
		sprintf(dat_bin, "%s", "bin");
	}

	for(int st = 0; st < Nst; ++st){


		int NBS = NBS_h[st];

//printf("Print Output | irregular: %d st: %d n1: %g\n", irregular, st, n1_h[st]);
		if(Nst > 1){
			int s = 0;
		
			if(irregular < 3) s = 1;	
			if(N_h[st] < Nmin[st].x) s = 1;
			if(Nsmall_h[st] < Nmin[st].y) s = 1;
			if(n1_h[st] < 0) s = 1;
			if(timeStep >= delta_h[st]) s = 1;
			//print only simulations which must be stopped by StopAtEncounter
			//or when the simulation reached the end
			if(s == 0){
				continue;
			}			
		}
//printf("Print Output2 | irregular: %d st: %d n1: %g\n", irregular, st, n1_h[st]);
		if(P.FormatP == 1){
			if(irregular == 2){
				sprintf(GSF[st].outputfilename,"OutCollision.%s", dat_bin);
				if(P.OutBinary == 0){
					GSF[st].outputfile = fopen(GSF[st].outputfilename, "w");
				}
				else{
					GSF[st].outputfile = fopen(GSF[st].outputfilename, "wb");
				}
			}
			else if(irregular == 4){
				sprintf(GSF[st].outputfilename,"OutError.%s", dat_bin);
				if(P.OutBinary == 0){
					GSF[st].outputfile = fopen(GSF[st].outputfilename, "w");
				}
				else{
					GSF[st].outputfile = fopen(GSF[st].outputfilename, "wb");
				}
			}
			else if(Nst == 1 || P.FormatS == 0){
				if(P.FormatT == 0){
					if(irregular == 0 || irregular == 3){
						long long scale = 1ll;
						if(P.FormatO == 1){
							scale = (long long)(P.ci);
							if(P.ci == -1) scale = (long long)(delta_h[st]);
						}
						sprintf(GSF[st].outputfilename,"%sOut%s_%.*lld.%s", GSF[st].path, GSF[st].X, def_NFileNameDigits, timeStep / scale, dat_bin);
						if(P.FormatO == 1 && interrupt == 1){
							sprintf(GSF[st].outputfilename,"%sOutbackup%s_%.20lld.%s", GSF[st].path, GSF[st].X, timeStep, dat_bin);
						}
					}
					else if(irregular == 1){
						sprintf(GSF[st].outputfilename,"%sOutIrr%s_%.*lld.%s", GSF[st].path, GSF[st].X, def_NFileNameDigits, irrTimeStep, dat_bin);
					}
#if def_TTV == 0
					if(P.OutBinary == 0){
						GSF[st].outputfile = fopen(GSF[st].outputfilename, "w");
					}
					else{
						GSF[st].outputfile = fopen(GSF[st].outputfilename, "wb");
					}
#else
					if(P.OutBinary == 0){
						if(st == 0) GSF[st].outputfile = fopen(GSF[st].outputfilename, "w");
						else GSF[st].outputfile = fopen(GSF[st].outputfilename, "a");
					}
					else{
						if(st == 0) GSF[st].outputfile = fopen(GSF[st].outputfilename, "wb");
						else GSF[st].outputfile = fopen(GSF[st].outputfilename, "ab");
					}
#endif	
				}
				if(P.FormatT == 1){
					if(irregular == 0 || irregular == 3){
						sprintf(GSF[st].outputfilename,"%sOut%s.%s", GSF[st].path, GSF[st].X, dat_bin);
					}
					else if(irregular == 1){
						sprintf(GSF[st].outputfilename,"%sOutIrr%s.%s", GSF[st].path, GSF[st].X, dat_bin);
					}
					if(P.OutBinary == 0){
						GSF[st].outputfile = fopen(GSF[st].outputfilename, "a");
					}
					else{
						GSF[st].outputfile = fopen(GSF[st].outputfilename, "ab");
					}
				}
			}
			else{
				if(P.FormatT == 0){
					if(irregular == 0 || irregular == 3){
						long long scale = 1ll;
						if(P.FormatO == 1){
							scale = (long long)(P.ci);
							if(P.ci == -1) scale = (long long)(delta_h[st]);
						}
						sprintf(GSF[st].outputfilename, "%s../Out%s_%.*lld.%s", GSF[st].path, GSF[st].X, def_NFileNameDigits, timeStep / scale, dat_bin);
						if(P.FormatO == 1 && interrupt == 1){
							sprintf(GSF[st].outputfilename, "%s../Outbackup%s_%.20lld.%s", GSF[st].path, GSF[st].X, timeStep, dat_bin);
						}
					}
					else if(irregular == 1){
						sprintf(GSF[st].outputfilename, "%s../OutIrr%s_%.*lld.%s", GSF[st].path, GSF[st].X, def_NFileNameDigits, irrTimeStep, dat_bin);
					}
					if(P.OutBinary == 0){
						if(st == 0) GSF[st].outputfile = fopen(GSF[st].outputfilename, "w");
						else GSF[st].outputfile = fopen(GSF[st].outputfilename, "a");
					}
					else{
						if(st == 0) GSF[st].outputfile = fopen(GSF[st].outputfilename, "wb");
						else GSF[st].outputfile = fopen(GSF[st].outputfilename, "ab");
					}
				}
				if(P.FormatT == 1){
					if(irregular == 0 || irregular == 3){
						sprintf(GSF[st].outputfilename, "%s../Out%s.%s", GSF[st].path, GSF[st].X, dat_bin);
					}
					else if(irregular == 1){
						sprintf(GSF[st].outputfilename, "%s../OutIrr%s.%s", GSF[st].path, GSF[st].X, dat_bin);
					}
					if(P.OutBinary == 0){
						GSF[st].outputfile = fopen(GSF[st].outputfilename, "a");
					}
					else{
						GSF[st].outputfile = fopen(GSF[st].outputfilename, "ab");
					}
				}
			}
		}
		//if(irregular < 3 || timeStep == delta_h[st] || irregular == 4){
			printOutput(x4_h + NBS, v4_h + NBS, v4Helio_h + NBS, index_h + NBS, test_h + NBS, time_h[st]/365.25, timeStep, N_h[st], GSF[st].outputfile, Msun_h[st].x, spin_h + NBS, love_h + NBS, migration_h + NBS, rcrit_h + NBS, Nsmall_h[st], Nst, aelimits_h + NBS, aecount_h + NBS, enccount_h + NBS, aecountT_h + NBS, enccountT_h + NBS, P.ci, irregular);

			if(P.FormatP == 1) fclose(GSF[st].outputfile);
		//}

	}
	cudaMemcpy(aecountT_d, aecountT_h, sizeof(unsigned long long)*NconstT, cudaMemcpyHostToDevice);
	cudaMemcpy(enccountT_d, enccountT_h, sizeof(unsigned long long)*NconstT, cudaMemcpyHostToDevice);

	cudaMemset(aecount_d, 0, sizeof(unsigned int)*NconstT);
	cudaMemset(enccount_d, 0, sizeof(unsigned int)*NconstT);
}

//This function copies the data from the coordinate buffer and calls the printoutput function
__host__ void Data::CoordinateOutputBuffer(int irregular){

	double *buffer_h;

	if(irregular == 0){
		cudaMemcpy(coordinateBuffer_h, coordinateBuffer_d, P.Buffer * def_BufferSize * NconstT * sizeof(double), cudaMemcpyDeviceToHost);
		buffer_h = coordinateBuffer_h;
	}
	else{
		cudaMemcpy(coordinateBufferIrr_h, coordinateBufferIrr_d, P.Buffer * def_BufferSize * NconstT * sizeof(double), cudaMemcpyDeviceToHost);
		buffer_h = coordinateBufferIrr_h;
	}
	cudaDeviceSynchronize();

	char dat_bin[16];
	if(P.OutBinary == 0){
		sprintf(dat_bin, "%s", "dat");
	}
	else{
		sprintf(dat_bin, "%s", "bin");
	}

	int Nbf = bufferCount;
	if(irregular == 1) Nbf = bufferCountIrr;
	for(int bf = 0; bf < Nbf; ++bf){
		for(int i = 0; i < NT + NsmallT; ++i){
			index_h[i] =		buffer_h[def_BufferSize * NconstT * bf + def_BufferSize * i + 1];
			x4_h[i].w =		buffer_h[def_BufferSize * NconstT * bf + def_BufferSize * i + 2];
			v4_h[i].w =		buffer_h[def_BufferSize * NconstT * bf + def_BufferSize * i + 3];
			x4_h[i].x =		buffer_h[def_BufferSize * NconstT * bf + def_BufferSize * i + 4];
			x4_h[i].y =		buffer_h[def_BufferSize * NconstT * bf + def_BufferSize * i + 5];
			x4_h[i].z =		buffer_h[def_BufferSize * NconstT * bf + def_BufferSize * i + 6];
			v4_h[i].x =		buffer_h[def_BufferSize * NconstT * bf + def_BufferSize * i + 7];
			v4_h[i].y =		buffer_h[def_BufferSize * NconstT * bf + def_BufferSize * i + 8];
			v4_h[i].z =		buffer_h[def_BufferSize * NconstT * bf + def_BufferSize * i + 9];
			spin_h[i].x =		buffer_h[def_BufferSize * NconstT * bf + def_BufferSize * i + 10];
			spin_h[i].y =		buffer_h[def_BufferSize * NconstT * bf + def_BufferSize * i + 11];
			spin_h[i].z =		buffer_h[def_BufferSize * NconstT * bf + def_BufferSize * i + 12];
			aelimits_h[i].x =	buffer_h[def_BufferSize * NconstT * bf + def_BufferSize * i + 13];
			aelimits_h[i].y =	buffer_h[def_BufferSize * NconstT * bf + def_BufferSize * i + 14];
			aelimits_h[i].z =	buffer_h[def_BufferSize * NconstT * bf + def_BufferSize * i + 15];
			aelimits_h[i].w =	buffer_h[def_BufferSize * NconstT * bf + def_BufferSize * i + 16];
			aecount_h[i] =		buffer_h[def_BufferSize * NconstT * bf + def_BufferSize * i + 17];
			aecountT_h[i] =		buffer_h[def_BufferSize * NconstT * bf + def_BufferSize * i + 18];
			enccountT_h[i] =	buffer_h[def_BufferSize * NconstT * bf + def_BufferSize * i + 19];
			test_h[i] =		buffer_h[def_BufferSize * NconstT * bf + def_BufferSize * i + 20];
			spin_h[i].w =		buffer_h[def_BufferSize * NconstT * bf + def_BufferSize * i + 21];
			love_h[i].x =		buffer_h[def_BufferSize * NconstT * bf + def_BufferSize * i + 22];
			love_h[i].y =		buffer_h[def_BufferSize * NconstT * bf + def_BufferSize * i + 23];
			love_h[i].z =		buffer_h[def_BufferSize * NconstT * bf + def_BufferSize * i + 24];
			rcrit_h[i] =		buffer_h[def_BufferSize * NconstT * bf + def_BufferSize * i + 25];
			if(P.UseMigrationForce > 0){
				migration_h[i].x =		buffer_h[def_BufferSize * NconstT * bf + def_BufferSize * i + 26];
				migration_h[i].y =		buffer_h[def_BufferSize * NconstT * bf + def_BufferSize * i + 27];
				migration_h[i].z =		buffer_h[def_BufferSize * NconstT * bf + def_BufferSize * i + 28];
			}

		}
		for(int st = 0; st < Nst; ++st){
			int NBS = NBS_h[st];

//printf("Print Output Buffer %d %d %g\n", irregular, st, n1_h[st]);
		if(Nst > 1){
			int s = 0;

			if(irregular < 3) s = 1;	
			if(N_h[st] < Nmin[st].x) s = 1;
			if(Nsmall_h[st] < Nmin[st].y) s = 1;
			if(n1_h[st] < 0) s = 1;
			if(timeStep >= delta_h[st]) s = 1;
			//print only simulations which must be stopped by StopAtEncounter
			//or when the simulation reached the end
			if(s == 0){
				continue;
			}			
		}

			if(P.FormatP == 1){
				if(Nst == 1 || P.FormatS == 0){
					if(P.FormatT == 0){
						if(irregular == 0){
							sprintf(GSF[st].outputfilename,"%sOut%s_%.*lld.%s", GSF[st].path, GSF[st].X, def_NFileNameDigits, timestepBuffer[bf], dat_bin);
						}
						else{
							sprintf(GSF[st].outputfilename,"%sOutIrr%s_%.*lld.%s", GSF[st].path, GSF[st].X, def_NFileNameDigits, irrTimeStepOut + bf, dat_bin);
						}
						if(P.OutBinary == 0){
							GSF[st].outputfile = fopen(GSF[st].outputfilename, "w");
						}
						else{
							GSF[st].outputfile = fopen(GSF[st].outputfilename, "wb");
						}
					}
					if(P.FormatT == 1){
						if(irregular == 0){
							sprintf(GSF[st].outputfilename,"%sOut%s.%s", GSF[st].path, GSF[st].X, dat_bin);
						}
						else{
							sprintf(GSF[st].outputfilename,"%sOutIrr%s.%s", GSF[st].path, GSF[st].X, dat_bin);
						}
						if(P.OutBinary == 0){
							GSF[st].outputfile = fopen(GSF[st].outputfilename, "a");
						}
						else{
							GSF[st].outputfile = fopen(GSF[st].outputfilename, "ab");
						}
					}
				}
				else{
					if(P.FormatT == 0){
						if(irregular == 0){
							sprintf(GSF[st].outputfilename, "%s../Out%s_%.*lld.%s", GSF[st].path, GSF[st].X, def_NFileNameDigits, timestepBuffer[bf], dat_bin);
						}
						else{
							sprintf(GSF[st].outputfilename, "%s../OutIrr%s_%.*lld.%s", GSF[st].path, GSF[st].X, def_NFileNameDigits, irrTimeStepOut + bf, dat_bin);
						}
						if(P.OutBinary == 0){
							if(st == 0) GSF[st].outputfile = fopen(GSF[st].outputfilename, "w");
							else GSF[st].outputfile = fopen(GSF[st].outputfilename, "a");
						}
						else{
							if(st == 0) GSF[st].outputfile = fopen(GSF[st].outputfilename, "wb");
							else GSF[st].outputfile = fopen(GSF[st].outputfilename, "ab");
						}
					}
					if(P.FormatT == 1){
						if(irregular == 0){
							sprintf(GSF[st].outputfilename, "%s../Out%s.%s", GSF[st].path, GSF[st].X, dat_bin);
						}
						else{
							sprintf(GSF[st].outputfilename, "%s../OutIrr%s.%s", GSF[st].path, GSF[st].X, dat_bin);
						}
						if(P.OutBinary == 0){
							GSF[st].outputfile = fopen(GSF[st].outputfilename, "a");
						}
						else{
							GSF[st].outputfile = fopen(GSF[st].outputfilename, "ab");
						}
					}
				}
			}
	
			double time;
			if(irregular == 0){
				time = timestepBuffer[bf] * idt_h[st] + ict_h[st] * 365.25;
				int N = NBuffer[Nst * bf + st].x;		
				int Nsmall = NBuffer[Nst * bf + st].y;
				printOutput(x4_h + NBS, v4_h + NBS, v4Helio_h + NBS, index_h + NBS, test_h + NBS, time/365.25, timestepBuffer[bf], N, GSF[st].outputfile, Msun_h[st].x, spin_h + NBS, love_h + NBS, migration_h + NBS, rcrit_h + NBS, Nsmall, Nst, aelimits_h + NBS, aecount_h + NBS, enccount_h + NBS, aecountT_h + NBS, enccountT_h + NBS, P.ci, irregular);
			}
			else{
				int N = NBufferIrr[Nst * bf + st].x;		
				int Nsmall = NBufferIrr[Nst * bf + st].y;		
				time = buffer_h[def_BufferSize * NconstT * bf + def_BufferSize * NBS];
				printOutput(x4_h + NBS, v4_h + NBS, v4Helio_h + NBS, index_h + NBS, test_h + NBS, time/365.25, timestepBufferIrr[bf], N, GSF[st].outputfile, Msun_h[st].x, spin_h + NBS, love_h + NBS, migration_h + NBS, rcrit_h + NBS, Nsmall, Nst, aelimits_h + NBS, aecount_h + NBS, enccount_h + NBS, aecountT_h + NBS, enccountT_h + NBS, P.ci, irregular);
			}

			if(P.FormatP == 1) fclose(GSF[st].outputfile);

		}
	}
	cudaMemcpy(aecountT_d, aecountT_h, sizeof(unsigned long long)*NconstT, cudaMemcpyHostToDevice);
	cudaMemcpy(enccountT_d, enccountT_h, sizeof(unsigned long long)*NconstT, cudaMemcpyHostToDevice);

	cudaMemset(aecount_d, 0, sizeof(unsigned int)*NconstT);
	cudaMemset(enccount_d, 0, sizeof(unsigned int)*NconstT);
}


__host__ void Data::GridaeOutput(){
	int GridNae = Gridae.Na * Gridae.Ne;
	int GridNai = Gridae.Na * Gridae.Ni;
	sprintf(Gridae.filename, "aeCount%s_%.*lld.dat", Gridae.X, def_NFileNameDigits, timeStep);
	Gridae.file = fopen(Gridae.filename, "w");
	cudaMemcpy(Gridaecount_h, Gridaecount_d, sizeof(unsigned int)*GridNae, cudaMemcpyDeviceToHost);
	cudaMemcpy(Gridaicount_h, Gridaicount_d, sizeof(unsigned int)*GridNai, cudaMemcpyDeviceToHost);
	//ae grid
	for(int i = 0; i < Gridae.Ne; ++i){
		for(int j = 0; j < Gridae.Na; ++j){
			if(timeStep > Gridae.Start){
				GridaecountS_h[i * Gridae.Na + j] += Gridaecount_h[i * Gridae.Na + j];
				GridaecountT_h[i * Gridae.Na + j] += Gridaecount_h[i * Gridae.Na + j];
			}
			fprintf(Gridae.file, "%llu ", GridaecountT_h[i * Gridae.Na + j]);
		}
		fprintf(Gridae.file, "\n");
	}
	fprintf(Gridae.file, "\n");
	fprintf(Gridae.file, "\n");
	for(int i = 0; i < Gridae.Ne; ++i){
		for(int j = 0; j < Gridae.Na; ++j){
			fprintf(Gridae.file, "%llu ", GridaecountS_h[i * Gridae.Na + j]);
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
			fprintf(Gridae.file, "%llu ", GridaicountT_h[i * Gridae.Na + j]);
		}
		fprintf(Gridae.file, "\n");
	}
	fprintf(Gridae.file, "\n");
	fprintf(Gridae.file, "\n");
	for(int i = 0; i < Gridae.Ni; ++i){
		for(int j = 0; j < Gridae.Na; ++j){
			fprintf(Gridae.file, "%llu ", GridaicountS_h[i * Gridae.Na + j]);
			GridaicountS_h[i * Gridae.Na + j] = 0;
		}
		fprintf(Gridae.file, "\n");
	}

	fclose(Gridae.file);
	cudaMemset(Gridaecount_d, 0, sizeof(int)*GridNae);
	cudaMemset(Gridaicount_d, 0, sizeof(int)*GridNai);
}


//This function prints information if a too big close encounter group occurs and stops the integrations
__host__ int Data::MaxGroups(){
	for(int nm = def_GMax - 1; nm < def_GMax; ++nm){
	//for(int nm = 12; nm < def_GMax; ++nm){
		if(Nenc_m[nm] > 0){
			GSF[0].logfile = fopen(GSF[0].logfilename, "a");
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			fprintf(GSF[0].logfile, "Number of Close-Encounter-pairs: %d\n", *Nencpairs2_h);
			fprintf(GSF[0].logfile, "Total number of groups: %d; ", Nenc_m[0]);
			int nn = 2;
			for(int st = 1; st < def_GMax; ++st){
				if(Nenc_m[st] > 0) fprintf(GSF[0].logfile, "%d: %d; ", nn, Nenc_m[st]);
				nn *= 2;
			}
			fprintf(GSF[0].logfile, "\n");

			fprintf(GSF[0].logfile, "Number of Precheck-pairs: %d\n", *Nencpairs_h);
			fprintf(GSF[0].logfile,"Output data when Error occured:\n");
			cudaMemcpy(index_h, index_d, sizeof(int)*NconstT, cudaMemcpyDeviceToHost);
			cudaMemcpy(test_h, test_d, sizeof(double)*NB[0], cudaMemcpyDeviceToHost);
			cudaMemcpy(spin_h, spin_d, sizeof(double4)*NconstT, cudaMemcpyDeviceToHost);
			cudaMemcpy(love_h, love_d, sizeof(double3)*NconstT, cudaMemcpyDeviceToHost);
			if(P.UseMigrationForce > 0){
				cudaMemcpy(migration_h, migration_d, sizeof(double3)*NconstT, cudaMemcpyDeviceToHost);
			}
			cudaMemcpy(aelimits_h, aelimits_d, sizeof(float4)*NconstT, cudaMemcpyDeviceToHost);
			cudaMemcpy(aecount_h, aecount_d, sizeof(unsigned int)*NconstT, cudaMemcpyDeviceToHost);
			cudaMemcpy(enccount_h, enccount_d, sizeof(unsigned int)*NconstT, cudaMemcpyDeviceToHost);
			cudaMemcpy(aecountT_h, aecountT_d, sizeof(unsigned long long)*NconstT, cudaMemcpyDeviceToHost);
			cudaMemcpy(enccountT_h, enccountT_d, sizeof(unsigned long long)*NconstT, cudaMemcpyDeviceToHost);
			cudaMemcpy(rcrit_h, rcrit_d, sizeof(double) * NconstT * P.SLevels, cudaMemcpyDeviceToHost);


			if(P.OutBinary == 0){
				GSF[0].outputfile = fopen(GSF[0].outputfilename, "w");	
			}
			else{
				GSF[0].outputfile = fopen(GSF[0].outputfilename, "wb");	
			}
			printOutput(x4_h, v4_h, v4Helio_h, index_h, test_h, time_h[0]/365.25, timeStep, N_h[0], GSF[0].outputfile, Msun_h[0].x, spin_h, love_h, migration_h, rcrit_h, Nsmall_h[0], Nst, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, P.ci, 0);
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
	cudaEventCreate(&tt1);
	cudaEventCreate(&tt2);
	cudaEventCreate(&tt3);
	cudaEventCreate(&tt4);

	cudaEventRecord(tt1, 0);
	cudaEventRecord(tt2, 0);

	times = 0.0f;
}


//This function prints information how long the integration takes
__host__ int Data::printTime(int irregular){
	
	cudaEventRecord(tt3, 0);
	cudaEventSynchronize(tt3);
	cudaEventElapsedTime(&times, tt2, tt3);
	FILE *timefile;
	for(int st = 0; st < Nst; ++st){

//printf("Print time | irregular: %d st: %d n1: %g\n", irregular, st, n1_h[st]);
		if(Nst > 1){
			int s = 0;
		
			if(irregular < 3) s = 1;	
			if(N_h[st] < Nmin[st].x) s = 1;
			if(Nsmall_h[st] < Nmin[st].y) s = 1;
			if(n1_h[st] < 0) s = 1;
			if(timeStep >= delta_h[st]) s = 1;
			//print only simulations which must be stopped by StopAtEncounter
			//or when the simulation reached the end
			if(s == 0){
				continue;
			}			
		}
//printf("Print time2 | irregular: %d st: %d n1: %g\n", irregular, st, n1_h[st]);

		timefile = fopen(GSF[st].timefilename, "a");
		if(timefile == NULL){
			printf("Error, timefile not valid %d %s\n", st, GSF[st].timefilename);
			return 0;
		}

		fprintf(timefile, "%lld %.20g\n", timeStep, times * 0.001);
		fclose(timefile);
		GSF[st].logfile = fopen(GSF[st].logfilename, "a");
		if(GSF[st].logfile == NULL){
			printf("Error, infofile not valid %d %s\n", st, GSF[st].logfilename);
			return 0;
		}
		fprintf(GSF[st].logfile,"Reached timestep %lld with %d bodies, %d test particles. Total Energy: %.20g\n", timeStep, N_h[st], Nsmall_h[st], Energy_h[4 + NEnergy[st]]);
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
	if(irregular == 0){
		cudaEventRecord(tt2, 0);
	}
	return 1;
}

//This function prints the total integration runtime
__host__ void Data::printLastTime(int irregular){
	cudaEventRecord(tt4, 0);
	cudaEventSynchronize(tt4);
	cudaEventElapsedTime(&times, tt1, tt4);
	FILE *timefile;
	for(int st = 0; st < Nst; ++st){

//printf("Print last time | irregular: %d st: %d n1: %g\n", irregular, st, n1_h[st]);
		if(Nst > 1){
			int s = 0;
		
			if(irregular < 3) s = 1;	
			if(N_h[st] < Nmin[st].x) s = 1;
			if(Nsmall_h[st] < Nmin[st].y) s = 1;
			if(n1_h[st] < 0) s = 1;
			if(timeStep >= delta_h[st]) s = 1;
			//print only simulations which must be stopped by StopAtEncounter
			//or when the simulation reached the end
			if(s == 0){
				continue;
			}			
		}
//printf("Print last time2 | irregular: %d st: %d n1: %g\n", irregular, st, n1_h[st]);
		timefile = fopen(GSF[st].timefilename, "a");
		if(irregular == 0){
			fprintf(timefile, "\n\n%lld %.20g\n", timeStep -1, times * 0.001);
		}
		else{
			fprintf(timefile, "\n\n%lld %.20g\n", timeStep, times * 0.001);
		}
		if(st == 0) printf("Execution time: \n\n%g\n", times * 0.001);
		fclose(timefile);
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
//stopAtCollision checks if one of the 2 colliding bodies is large enough to resolve the collision externally.
__host__ int Data::printCollisions(){
  
	cudaMemcpy(Coll_h, Coll_d, sizeof(double) * def_NColl * Ncoll_m[0], cudaMemcpyDeviceToHost);
	FILE *collisionfile;
	FILE *logfile;
	int stopAtCollision = 0;
	for(int nc = 0; nc < Ncoll_m[0]; ++nc){
		int st;
		if(Nst == 1) st = 0;
		else st = (int)(Coll_h[nc * def_NColl + 1]) / def_MaxIndex;
		collisionfile = fopen(GSF[st].collisionfilename, "a");

		logfile = fopen(GSF[st].logfilename, "a");

		for(int in = 0; in < def_NColl; ++in){
			if(in == 1 || in == 13){
				if(Nst == 1) fprintf(collisionfile, "%d ", (int)(Coll_h[nc * def_NColl + in]));
				else fprintf(collisionfile, "%d ", ((int)(Coll_h[nc * def_NColl + in])) % def_MaxIndex);
			}
			else fprintf(collisionfile, "%.20g ", Coll_h[nc * def_NColl + in]);
		}
		if(Nst == 1){
			fprintf(logfile, "Collision between body %d and %d\n", (int)(Coll_h[nc * def_NColl + 1]), (int)(Coll_h[nc * def_NColl + 13]));
			printf("Collision between body %d and %d\n", (int)(Coll_h[nc * def_NColl + 1]), (int)(Coll_h[nc * def_NColl + 13]));
		}
		else{
			fprintf(logfile, "Collision between body %d and %d\n", (int)(Coll_h[nc * def_NColl + 1]) % def_MaxIndex , (int)(Coll_h[nc * def_NColl + 13]) % def_MaxIndex);
			printf("In Simulation %s: Collision between body %d and %d\n", GSF[st].path, (int)(Coll_h[nc * def_NColl + 1]) % def_MaxIndex , (int)(Coll_h[nc * def_NColl + 13]) % def_MaxIndex);
		}
	
		if(Coll_h[nc * def_NColl + 2] >= P.StopMinMass && Coll_h[nc * def_NColl + 14] >= P.StopMinMass){
			stopAtCollision = 1;
		}

		fprintf(collisionfile, "\n");
		fclose(collisionfile);
		fclose(logfile);
	}
	return stopAtCollision;
}

//This function prints details of the Collisions
__host__ void Data::printCollisionsTshift(){
  
	FILE *collisionfile;
	for(int nc = Ncoll_m[0] / 2; nc < Ncoll_m[0]; ++nc){
		int st;
		if(Nst == 1) st = 0;
		else st = (int)(Coll_h[nc * def_NColl + 1]) / def_MaxIndex;
		collisionfile = fopen(GSF[st].collisionTshiftfilename, "a");

		for(int in = 0; in < def_NColl; ++in){
			if(in == 1 || in == 13){
				if(Nst == 1) fprintf(collisionfile, "%d ", (int)(Coll_h[nc * def_NColl + in]));
				else fprintf(collisionfile, "%d ", ((int)(Coll_h[nc * def_NColl + in])) % def_MaxIndex);
			}
			else fprintf(collisionfile, "%.20g ", Coll_h[nc * def_NColl + in]);
		}
		fprintf(collisionfile, "\n");
		fclose(collisionfile);
	}
}

//This function prints details of the Encounters
__host__ int Data::printEncounters(){
 
	if(NWriteEnc_m[0] >= def_MaxWriteEnc){
		for(int st = 0; st < Nst; ++st){ 
			GSF[st].logfile = fopen(GSF[st].logfilename, "a");
			fprintf(GSF[st].logfile, "Error: Too many Encounters to write %d, allowed are %d\n", NWriteEnc_m[0], def_MaxWriteEnc);
			printf("Error: Too many Encounters to write %d, allowed are %d\n", NWriteEnc_m[0], def_MaxWriteEnc);
			fclose(GSF[st].logfile);
		}
		return 0;
	}
 
	cudaMemcpy(writeEnc_h, writeEnc_d, sizeof(double) * def_NColl * NWriteEnc_m[0], cudaMemcpyDeviceToHost);

	FILE *encounterfile;
	for(int nc = 0; nc < NWriteEnc_m[0]; ++nc){
		int st;
		if(Nst == 1) st = 0;
		else st = (int)(writeEnc_h[nc * def_NColl + 1]) / def_MaxIndex;
		encounterfile = fopen(GSF[st].encounterfilename, "a");

		for(int in = 0; in < def_NColl; ++in){
			if(in == 1 || in == 13){
				if(Nst == 1) fprintf(encounterfile, "%d ", (int)(writeEnc_h[nc * def_NColl + in]));
				else fprintf(encounterfile, "%d ", ((int)(writeEnc_h[nc * def_NColl + in])) % def_MaxIndex);
			}
			else fprintf(encounterfile, "%.20g ", writeEnc_h[nc * def_NColl + in]);
		}
		fprintf(encounterfile, "\n");
		fclose(encounterfile);
	}
	return 1;
}

//This function prints details of fragmentations
__host__ int Data::printFragments(int nf){

	int st = 0; 
	GSF[st].logfile = fopen(GSF[st].logfilename, "a");
	fprintf(GSF[st].logfile, "Created %d fragments\n", nf);
	printf("Created %d fragments\n", nf);
	fclose(GSF[st].logfile);

	if(nf > P.Nfragments){
		GSF[st].logfile = fopen(GSF[st].logfilename, "a");
		fprintf(GSF[st].logfile, "Error: More particles created than Nfragments: %d %d\n", nf, P.Nfragments);
		printf("Error: Error: More particles created than Nfragments: %d %d\n", nf, P.Nfragments);
		fclose(GSF[st].logfile);

		return 0;
	}

	if(N_h[0] + Nsmall_h[0] >= NconstT){
		GSF[st].logfile = fopen(GSF[st].logfilename, "a");
		fprintf(GSF[st].logfile, "Error: Too many particles created\n");
		printf("Error: Too many particles created\n");
		fclose(GSF[st].logfile);

		return 0;
	}
 
	cudaMemcpy(Fragments_h, Fragments_d, sizeof(double) * 25 * P.Nfragments, cudaMemcpyDeviceToHost);

	FILE *fragmentfile;
	for(int nc = 0; nc < nf + 1; ++nc){
		int st;
		if(Nst == 1) st = 0;
		else st = (int)(Fragments_h[nc * 25 + 1]) / def_MaxIndex;
		fragmentfile = fopen(GSF[st].fragmentfilename, "a");

		for(int in = 0; in < 13; ++in){
			if(in == 1 || in == 13){
				if(Nst == 1) fprintf(fragmentfile, "%d ", (int)(Fragments_h[nc * 25 + in]));
				else fprintf(fragmentfile, "%d ", ((int)(Fragments_h[nc * 25 + in])) % def_MaxIndex);
			}
			else fprintf(fragmentfile, "%.20g ", Fragments_h[nc * 25 + in]);
		}
		if(nc == 0) fprintf(fragmentfile, " -1\n");	//particle is destroyed
		else{
			if(Fragments_h[nc * 25 + 3] * def_AU < P.Asteroid_rdel) fprintf(fragmentfile, " 2\n");	//new particle but too small
			else fprintf(fragmentfile, " 1\n");							//new particle
		}
		fclose(fragmentfile);
	}
	return 1;
}
//This function prints details of rotation resets
__host__ int Data::printRotation(){

	int st = 0; 
	GSF[st].logfile = fopen(GSF[st].logfilename, "a");
	fprintf(GSF[st].logfile, "Rotation reset\n");
	printf("Rotation reset\n");
	fclose(GSF[st].logfile);

	cudaMemcpy(Fragments_h, Fragments_d, sizeof(double) * 25, cudaMemcpyDeviceToHost);

	FILE *fragmentfile;
	if(Nst == 1) st = 0;
	else st = (int)(Fragments_h[1]) / def_MaxIndex;
	fragmentfile = fopen(GSF[st].fragmentfilename, "a");

	for(int in = 0; in < 13; ++in){
		if(in == 1 || in == 13){
			if(Nst == 1) fprintf(fragmentfile, "%d ", (int)(Fragments_h[in]));
			else fprintf(fragmentfile, "%d ", ((int)(Fragments_h[in])) % def_MaxIndex);
		}
		else fprintf(fragmentfile, "%.20g ", Fragments_h[in]);
	}
	fprintf(fragmentfile, " 0\n");
	fclose(fragmentfile);

	return 1;
}

//This function prints details of particle creation events
__host__ int Data::printCreateparticle(int nf){

	int st = 0; 

	cudaMemcpy(Fragments_h, Fragments_d, sizeof(double) * 25 * nf, cudaMemcpyDeviceToHost);

	FILE *fragmentfile;
	for(int nc = 0; nc < nf; ++nc){
		if(Nst == 1) st = 0;
		else st = (int)(Fragments_h[nc * 25 + 1]) / def_MaxIndex;
		fragmentfile = fopen(GSF[st].fragmentfilename, "a");
		GSF[st].logfile = fopen(GSF[st].logfilename, "a");

		int id = -1;
		for(int in = 0; in < 13; ++in){
			if(in == 1 || in == 13){
				if(Nst == 1){
					id = (int)(Fragments_h[nc * 25 + in]);
					fprintf(fragmentfile, "%d ", id);
				}
				else{
					id = ((int)(Fragments_h[nc * 25 + in])) % def_MaxIndex;
					fprintf(fragmentfile, "%d ", id);
				}
			}
			else fprintf(fragmentfile, "%.20g ", Fragments_h[nc * 25 + in]);
		}
		fprintf(GSF[st].logfile, "Create particle %d\n", id);
		fprintf(fragmentfile, " 10\n");
		fclose(fragmentfile);
		fclose(GSF[st].logfile);
	}

	return 1;
}

//This function prints the transit times
__host__ int Data::printTransits(){
	cudaMemcpy(TransitTime_h, TransitTime_d, def_NtransitTimeMax * NconstT * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(elementsC_h, elementsC_d, (Nst + MCMC_NT) * sizeof(int2), cudaMemcpyDeviceToHost);

	FILE *Transitfile;
	Transitfile = fopen("Transits.dat", "a");
	for(int i = 0; i < NconstT; ++i){
		int si = i / def_MaxIndex;
		if(elementsC_h[si + MCMC_NT].x >= 0){
			int Epoch = 0;
			int setEpoch = 0;
			for(int EpochObs = 0; EpochObs <= NtransitsT_h[i].x; ++EpochObs){
				double T = TransitTime_h[i * def_NtransitTimeMax + Epoch + 1]; 
				double T1 = TransitTime_h[i * def_NtransitTimeMax + Epoch + 2];
				double2 TObs;
				if(EpochObs <= NtransitsTObs_h[i % N_h[0]]){
					TObs = TransitTimeObs_h[(i % N_h[0]) * def_NtransitTimeMax + EpochObs + 1];
				}
				else{
					TObs.x = 0.0;
					TObs.y = 0.0;
				}

//printf("---- %d %.20g %.20g %d %d\n", i, T, TObs.x, Epoch, EpochObs);

				if(fabs(TObs.x - T) < fabs(TObs.x - T1) && T != 0.0 && TObs.x != 0.0){
					setEpoch = 1;
				}


				if(setEpoch == 0 && T != 0 && TObs.x != 0 && fabs(TObs.x - T) < fabs(TObs.x - T1)){
//printf("***** %d %.20g %.20g %d %d\n", i, T, TObs.x, Epoch, EpochObs);
					++EpochObs;
					TObs = TransitTimeObs_h[(i % N_h[0]) * def_NtransitTimeMax + EpochObs + 1];
				}

				if(fabs(TObs.x - T) < fabs(TObs.x - T1) && T != 0.0 && TObs.x != 0.0){
					setEpoch = 1;
				}

				if(P.PrintTransits == 2){				
					 if(setEpoch == 0) fprintf(Transitfile, "%d %d %25.20g %25.20g %25.20g\n", i, Epoch, T, 0.0, 0.0);
				}

				if(setEpoch == 0 && T != 0 && TObs.x != 0 && fabs(TObs.x - T) >= fabs(TObs.x - T1)){
//printf("#####  %d %.20g %.20g %d %d\n", i, T, TObs.x, Epoch, EpochObs);
					++Epoch;
					--EpochObs;
					continue;
				}

				if(P.PrintTransits == 1){				
					if(setEpoch == 1) fprintf(Transitfile, "%d %d %25.20g %25.20g %25.20g\n", i, EpochObs, T, TObs.x, TObs.y);
				}
				if(P.PrintTransits == 2){				
					 if(setEpoch == 1)fprintf(Transitfile, "%d %d %25.20g %25.20g %25.20g\n", i, Epoch, T, TObs.x, TObs.y);
				}

				++Epoch;
				if(NtransitsTObs_h[i % N_h[0]] >= def_NtransitTimeMax -1){
					printf("Error: more transits than def_NtransitTimeMax for object %d: %d %d\n", i, NtransitsTObs_h[i % N_h[0]], def_NtransitTimeMax);
					return 0;
				}
			}
		}
	}

	fclose(Transitfile);
	return 1;
}

//This function prints the RV data at the obervation times
__host__ int Data::printRV(){
	cudaMemcpy(RV_h, RV_d, def_NRVMax * Nst * sizeof(double2), cudaMemcpyDeviceToHost);

	FILE *RVfile;
	RVfile = fopen("RVs.dat", "a");
	for(int si = 0; si < Nst; ++si){
//printf("NVRT %d %d %d\n", si, NRVT_h[si].x, NRVTObs_h[si]);
		for(int i = 0; i < NRVT_h[si].x; ++i){
			double2 T = RV_h[si * def_NRVMax + i]; 
			double3 TObs;
			if(i <= NRVTObs_h[si]){
				TObs = RVObs_h[si * def_NRVMax + i];
			}
			else{
				TObs.x = 0.0;
				TObs.y = 0.0;
				TObs.z = 1.0;
			}

			
			fprintf(RVfile, "%d %d %.25g %25.20g %25.20g %25.20g\n", si, i, T.x, T.y, TObs.y, TObs.z);

			if(NRVTObs_h[si] >= def_NRVMax -1){
				printf("Error: more RV data than def_NRVMax: %d %d\n", NRVTObs_h[si], def_NRVMax);
				return 0;
			}
		}
	}

	fclose(RVfile);
	return 1;
}
//This function prints the RV data at all time steps and no observation data
__host__ int Data::printRV2(){
	cudaMemcpy(RV_h, RV_d, def_NRVMax * Nst * sizeof(double2), cudaMemcpyDeviceToHost);

	FILE *RVfile;
	RVfile = fopen("RVall.dat", "a");
	for(int si = 0; si < Nst; ++si){
//printf("NVRT %d %d %d\n", si, NRVT_h[si].x, NRVTObs_h[si]);
		for(int i = 0; i < NRVT_h[si].x; ++i){
			double2 T = RV_h[si * def_NRVMax + i]; 
			
			fprintf(RVfile, "%d %d %.25g %25.20g\n", si, i, T.x, T.y);

			if(NRVTObs_h[si] >= def_NRVMax -1){
				printf("Error: more RV data than def_NRVMax: %d %d\n", NRVTObs_h[si], def_NRVMax);
				return 0;
			}
		}
	}

	fclose(RVfile);
	return 1;
}

__host__ void Data::printMCMC(int E){
	FILE *MCMCfile;
	MCMCfile = fopen("MCMC.dat", "a");

	if(P.PrintMCMC == 3){
	//print all, reprint old values for not accepted steps
		cudaMemcpy(elementsA_h, elementsAOld_d, NconstT * sizeof(double4), cudaMemcpyDeviceToHost);
		cudaMemcpy(elementsB_h, elementsBOld_d, NconstT * sizeof(double4), cudaMemcpyDeviceToHost);
		cudaMemcpy(elementsT_h, elementsTOld_d, NconstT * sizeof(double4), cudaMemcpyDeviceToHost);
		cudaMemcpy(elementsSpin_h, elementsSpinOld_d, NconstT * sizeof(double4), cudaMemcpyDeviceToHost);
	}
	if(P.PrintMCMC == 2){
	//print all, also not accepted steps
		cudaMemcpy(elementsA_h, elementsA_d, NconstT * sizeof(double4), cudaMemcpyDeviceToHost);
		cudaMemcpy(elementsB_h, elementsB_d, NconstT * sizeof(double4), cudaMemcpyDeviceToHost);
		cudaMemcpy(elementsT_h, elementsT_d, NconstT * sizeof(double4), cudaMemcpyDeviceToHost);
		cudaMemcpy(elementsSpin_h, elementsSpin_d, NconstT * sizeof(double4), cudaMemcpyDeviceToHost);
	}
	if(P.PrintMCMC == 1){
	//print only accepted
		cudaMemcpy(elementsA_h, elementsAOld_d, NconstT * sizeof(double4), cudaMemcpyDeviceToHost);
		cudaMemcpy(elementsB_h, elementsBOld_d, NconstT * sizeof(double4), cudaMemcpyDeviceToHost);
		cudaMemcpy(elementsT_h, elementsTOld_d, NconstT * sizeof(double4), cudaMemcpyDeviceToHost);
		cudaMemcpy(elementsSpin_h, elementsSpinOld_d, NconstT * sizeof(double4), cudaMemcpyDeviceToHost);
	}
	cudaMemcpy(elementsL_h, elementsL_d, NconstT * sizeof(elements10), cudaMemcpyDeviceToHost);
	cudaMemcpy(elementsP_h, elementsP_d, Nst * sizeof(double4), cudaMemcpyDeviceToHost);
	cudaMemcpy(elementsSA_h, elementsSA_d, Nst * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(elementsC_h, elementsC_d, (Nst + MCMC_NT) * sizeof(int2), cudaMemcpyDeviceToHost);

	if(E == 0){

#if MCMC_Q == 0
		for(int id = 0; id < NconstT; ++id){
#elif MCMC_Q == 2
		for(int id = 0; id < NconstT; ++id){
#else
		for(int id = 0; id < NconstT / 3; ++id){

#endif
			int si = 0;
			if(Nst > 1) si = index_h[id] / def_MaxIndex;

			int p = 0;
			double pp = elementsP_h[si].z;
			if(P.PrintMCMC == 1){

				if(elementsC_h[si + MCMC_NT].x >= 0) p = 1;
			}
			if(P.PrintMCMC == 2){
				pp = elementsP_h[si].x;
				if(pp >= 1.0e299) pp = elementsP_h[si].z;
				if(elementsP_h[si].z < 1.0e299){
					p = 1;
				}
			}
			if(P.PrintMCMC == 3){
				if(pp >= 1.0e299) pp = elementsP_h[si].z;
				if(elementsP_h[si].z < 1.0e299){
					p = 1;
				}
			}

			if(p == 1){
				double f = 1.0;
				double time = ict_h[0];
		
				int ii = id;
				fprintf(MCMCfile, "%#15.10g %d %#25.20g %#25.20g %#25.20g %#25.20g %#25.20g %#25.20g %#25.20g %#25.20g %#15.10g %#15.10g %#15.10g %#15.10g %#15.10g %#15.10g %#15.10g %#15.10g %#15.10g %#15.10g %#15.10g %#15.10g\n", time, id % N_h[0], elementsA_h[ii].w, elementsB_h[ii].w, elementsT_h[ii].z, elementsA_h[ii].y, elementsA_h[ii].z, elementsB_h[ii].x, elementsB_h[ii].y, elementsT_h[ii].x, f * elementsL_h[ii].m, f * elementsL_h[ii].r, f * elementsL_h[ii].P, f * elementsL_h[ii].e, f * elementsL_h[ii].inc, f * elementsL_h[ii].O, f * elementsL_h[ii].w, f * elementsL_h[ii].T, pp * 2.0, elementsP_h[si].w, elementsSA_h[si], Msun_h[si].x);
				//fprintf(MCMCfile, "%#15.10g %d %#25.20g %#25.20g %#25.20g %#25.20g %#25.20g %#25.20g %#25.20g %#25.20g %#15.10g %#15.10g %#15.10g %#15.10g %#15.10g %#15.10g %#15.10g %#15.10g %#15.10g %#15.10g %#15.10g %#15.10g\n", time, id % N_h[0], elementsA_h[ii].w, elementsSpin_h[ii].y, elementsT_h[ii].z, elementsA_h[ii].y, elementsA_h[ii].z, elementsB_h[ii].x, elementsB_h[ii].y, elementsT_h[ii].x, f * elementsL_h[ii].m, f * elementsL_h[ii].r, f * elementsL_h[ii].P, f * elementsL_h[ii].e, f * elementsL_h[ii].inc, f * elementsL_h[ii].O, f * elementsL_h[ii].w, f * elementsL_h[ii].T, pp * 2.0, elementsP_h[si].w, elementsSA_h[si], Msun_h[si].x);
			}
		}
		fclose(MCMCfile);
	}
	else{
		MCMCfile = fopen("MCMC_bak.dat", "w");
		for(int id = 0; id < NconstT; ++id){
			int si = 0;
			if(Nst > 1) si = index_h[id] / def_MaxIndex;
			double f = 1.0;
			double time = ict_h[0];
		
			int ii = id;
			fprintf(MCMCfile, "%#15.10g %d %#25.20g %#25.20g %#25.20g %#25.20g %#25.20g %#25.20g %#25.20g %#25.20g %#15.10g %#15.10g %#15.10g %#15.10g %#15.10g %#15.10g %#15.10g %#15.10g %#15.10g %#15.10g %#15.10g %#15.10g\n", time, id % N_h[0], elementsA_h[ii].w, elementsB_h[ii].w, elementsT_h[ii].z, elementsA_h[ii].y, elementsA_h[ii].z, elementsB_h[ii].x, elementsB_h[ii].y, elementsT_h[ii].x, f * elementsL_h[ii].m, f * elementsL_h[ii].r, f * elementsL_h[ii].P, f * elementsL_h[ii].e, f * elementsL_h[ii].inc, f * elementsL_h[ii].O, f * elementsL_h[ii].w, f * elementsL_h[ii].T, elementsP_h[si].x * 2.0, elementsP_h[si].w, elementsSA_h[si], Msun_h[si].x);
		}
		fclose(MCMCfile);
	}

}


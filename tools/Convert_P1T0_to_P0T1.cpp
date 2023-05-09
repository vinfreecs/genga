#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// ***********************************
// This tool converts GENGA outputs from FormatP = 1 & FormatT = 0, to FormatP = 0 & FormatT = 1
// Date: May 2023
// Author: Simon Grimm
// ***********************************


int main(int argc, char*argv[]){

	long long int kmin = 0;
	long long int kmax = 100;
	long long int step = 1;
	char X[160];
	char inputfilename[260];
	char outputfilename[260];
	FILE *inputfile;

	for(int i = 1; i < argc; i += 2){

		if(strcmp(argv[i], "-tmin") == 0){
			kmin = atoll(argv[i + 1]);
		}
		else if(strcmp(argv[i], "-tmax") == 0){
			kmax = atoll(argv[i + 1]);
		}
		else if(strcmp(argv[i], "-step") == 0){
			step = atoll(argv[i + 1]);
		}
		else if(strcmp(argv[i], "-in") == 0){
			sprintf(X, "%s", argv[i + 1]);
		}
		else{
		printf("Error, console argument not valid.\n");
		}
	}
	printf("tmin: %lld, tmax: %lld, step: %lld, Name: %s\n", kmin, kmax, step, X);

	
	double time;
	int index;
	double m, r;
	double x, y, z;
	double vx, vy, vz;
	double spinx, spiny, spinz;
	double a1, a2, a3, a4, a5, a6, a7, a8;

	FILE *outputfile;
	int er = 0;

	for(long long int k = kmin; k <= kmax; k += step){

		sprintf(inputfilename, "Out%s_%.12lld.dat", X, k);
		inputfile = fopen(inputfilename, "r");
		if(inputfile == NULL){
printf("%s skipped %lld\n", inputfilename, k);
			continue;
		}
printf("%s\n", inputfilename);

		for(int i = 0; i < 1e7; ++i){
			fscanf (inputfile, "%lf",&time);
			fscanf (inputfile, "%d",&index);
			fscanf (inputfile, "%lf",&m);
			fscanf (inputfile, "%lf",&r);
			fscanf (inputfile, "%lf",&x);
			fscanf (inputfile, "%lf",&y);
			fscanf (inputfile, "%lf",&z);
			fscanf (inputfile, "%lf",&vx);
			fscanf (inputfile, "%lf",&vy);
			fscanf (inputfile, "%lf",&vz);
			fscanf (inputfile, "%lf",&spinx);
			fscanf (inputfile, "%lf",&spiny);
			fscanf (inputfile, "%lf",&spinz);
			fscanf (inputfile, "%lf",&a1);
			fscanf (inputfile, "%lf",&a2);
			fscanf (inputfile, "%lf",&a3);
			fscanf (inputfile, "%lf",&a4);
			fscanf (inputfile, "%lf",&a5);
			fscanf (inputfile, "%lf",&a6);
			fscanf (inputfile, "%lf",&a7);
			er = fscanf (inputfile, "%lf",&a8);
			if(er <= 0){
				break;
			}

			sprintf(outputfilename, "Out%s_p%.6d.dat", X, index);
			outputfile = fopen(outputfilename, "a");

			fprintf(outputfile,"%.16g %d %.40g %.40g %.40g %.40g %.40g %.40g %.40g %.40g %.40g %.40g %.40g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g\n", time, index, m, r, x, y, z, vx, vy, vz, spinx, spiny, spinz, a1, a2, a3, a4, a5, a6, a7, a8);

			fclose(outputfile);
		}
		fclose(inputfile);

	}
}

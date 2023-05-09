#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// ***********************************
// This tool converts GENGA outputs from FormatP = 0 & FormatT = 1, to FormatP = 1 & FormatT = 0
// Date: May 2023
// Author: Simon Grimm
// ***********************************
  
int main(int argc, char*argv[]){

	int pmin = 0;			//used for FormatP = 0
	int pmax = 0;
	char X[160];
	char inputfilename[260];
	char outputfilename[260];
	FILE *inputfile;

	double dt = 6.0;
	double ict = 0.0;

	for(int i = 1; i < argc; i += 2){

		if(strcmp(argv[i], "-pmin") == 0){
			pmin = atoi(argv[i + 1]);
		}
		else if(strcmp(argv[i], "-pmax") == 0){
			pmax = atoi(argv[i + 1]);
		}
		else if(strcmp(argv[i], "-dt") == 0){
			dt = atof(argv[i + 1]);
		}
		else if(strcmp(argv[i], "-t") == 0){
			ict = atof(argv[i + 1]);
		}
		else if(strcmp(argv[i], "-in") == 0){
			sprintf(X, "%s", argv[i + 1]);
		}
		else{
			printf("Error, console argument not valid.\n");
		}
	}

	printf("pmin: %d, pmax: %d, Name: %s, dt: %g, ict: %g\n", pmin, pmax, X, dt, ict);

	double time;
	int index;
	double m, r;
	double x, y, z;
	double vx, vy, vz;
	double spinx, spiny, spinz;
	double a1, a2, a3, a4, a5, a6, a7, a8;

	FILE *outputfile;
	int er = 0;

	//FormatT = 1, FormatP = 0
	for(int p = pmin; p <= pmax; ++p){
		sprintf(inputfilename, "Out%s_p%.6d.dat", X, p);	
		inputfile = fopen(inputfilename, "r");
		if(inputfile == NULL){
printf("%s skipped %d\n", inputfilename, p);
			continue;
		}
printf("%s\n", inputfilename);

		for(long long int tt = 0ll; tt < 1e12; ++tt){
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
			long long int k = (((time - ict) * 365.25) + 0.5 * dt) / dt;
			//printf("%lld %d %g %lld\n", tt, index, time, k);

			sprintf(outputfilename, "Out%s_%.12lld.dat", X, k);
			outputfile = fopen(outputfilename, "a");

			fprintf(outputfile,"%.16g %d %.40g %.40g %.40g %.40g %.40g %.40g %.40g %.40g %.40g %.40g %.40g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g\n", time, index, m, r, x, y, z, vx, vy, vz, spinx, spiny, spinz, a1, a2, a3, a4, a5, a6, a7, a8);

			fclose(outputfile);
		}
		fclose(inputfile);

	}
}

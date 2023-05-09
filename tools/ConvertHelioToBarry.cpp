#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


void HelioToBary(double *m, double *x, double*y, double *z, double *vx, double *vy, double *vz, double &xsun, double &ysun, double &zsun, double &vxsun, double &vysun, double &vzsun, double &mtot, double Msun, int N){

	mtot = Msun;
	double xcomx = 0.0;
	double xcomy = 0.0;
	double xcomz = 0.0;
	double vcomx = 0.0;
	double vcomy = 0.0;
	double vcomz = 0.0;

	for(int i = 0; i < N; ++i){
		if(m[i] > 0.0){
			mtot += m[i];
			xcomx += m[i] * x[i];
			xcomy += m[i] * y[i];
			xcomz += m[i] * z[i];
			vcomx += m[i] * vx[i];
			vcomy += m[i] * vy[i];
			vcomz += m[i] * vz[i];
		}
	}
	xcomx /= mtot;
	xcomy /= mtot;
	xcomz /= mtot;
	vcomx /= mtot;
	vcomy /= mtot;
	vcomz /= mtot;

	for(int i = 0; i < N; ++i){
		x[i] -= xcomx;
		y[i] -= xcomy;
		z[i] -= xcomz;
		vx[i] -= vcomx;
		vy[i] -= vcomy;
		vz[i] -= vcomz;
	}
	xsun = -xcomx;
	ysun = -xcomy;
	zsun = -xcomz;
	vxsun = -vcomx;
	vysun = -vcomy;
	vzsun = -vcomz;
}


int main(int argc, char*argv[]){

	long long int kmin = 0;
	long long int kmax = 100;
	long long int step = 1;
	char X[160];
	char inputfilename[260];
	char outputfilename[260];
	FILE *inputfile;

	double Msun = 1.0;

	for(int i = 1; i < argc; i += 2){

		if(strcmp(argv[i], "-tmin") == 0){
			kmin = atoll(argv[i + 1]);
		}
		else if(strcmp(argv[i], "-tmax") == 0){
			kmax = atoll(argv[i + 1]);
		}
		else if(strcmp(argv[i], "-in") == 0){
			sprintf(X, "%s", argv[i + 1]);
		}
		else if(strcmp(argv[i], "-step") == 0){
			step = atoll(argv[i + 1]);
		}
		else if(strcmp(argv[i], "-Msun") == 0){
			Msun = atof(argv[i + 1]);
		}
		else{
			printf("Error, console argument not valid. This tool works only in FormatT = 0, FormatP = 1 format\n");
			printf("Use the Convert_P0T1_to_P1T0.cpp tool first\n");
		}
	}

	printf("tmin: %lld, tmax: %lld, step: %lld, Name: %s\n", kmin, kmax, step, X);

	const int NN = 500000;
	int N = 0;
	
	double *time;
	int *index;
	double *m, *r;
	double *x, *y, *z;
	double *vx, *vy, *vz;
	double *spinx, *spiny, *spinz;
	double *a1, *a2, *a3, *a4, *a5, *a6, *a7, *a8;


	time = (double*)malloc(NN * sizeof(double));
	index = (int*)malloc(NN * sizeof(int));
	m = (double*)malloc(NN * sizeof(double));
	r = (double*)malloc(NN * sizeof(double));
	x = (double*)malloc(NN * sizeof(double));
	y = (double*)malloc(NN * sizeof(double));
	z = (double*)malloc(NN * sizeof(double));
	vx = (double*)malloc(NN * sizeof(double));
	vy = (double*)malloc(NN * sizeof(double));
	vz = (double*)malloc(NN * sizeof(double));
	spinx = (double*)malloc(NN * sizeof(double));
	spiny = (double*)malloc(NN * sizeof(double));
	spinz = (double*)malloc(NN * sizeof(double));
	a1 = (double*)malloc(NN * sizeof(double));
	a2 = (double*)malloc(NN * sizeof(double));
	a3 = (double*)malloc(NN * sizeof(double));
	a4 = (double*)malloc(NN * sizeof(double));
	a5 = (double*)malloc(NN * sizeof(double));
	a6 = (double*)malloc(NN * sizeof(double));
	a7 = (double*)malloc(NN * sizeof(double));
	a8 = (double*)malloc(NN * sizeof(double));


	for(long long int k = kmin; k <= kmax; k += step){
		sprintf(outputfilename, "OutBary%s_%.12lld.dat", X, k);
		sprintf(inputfilename, "Out%s_%.12lld.dat", X, k);	

		inputfile = fopen(inputfilename, "r");
		if(inputfile == NULL){
			printf("%s skipped %lld\n", inputfilename, k);
			//continue;
			break;
		}
		printf("%s\n", inputfilename);
		FILE *outputfile;
		outputfile = fopen(outputfilename, "w");
		int er = 0;
		for(int i = 0; i < NN; ++i){
			fscanf (inputfile, "%lf",&time[i]);
			fscanf (inputfile, "%d",&index[i]);
//printf("%d %g %d\n", i, t, index);
			fscanf (inputfile, "%lf",&m[i]);
			fscanf (inputfile, "%lf",&r[i]);
			fscanf (inputfile, "%lf",&x[i]);
			fscanf (inputfile, "%lf",&y[i]);
			fscanf (inputfile, "%lf",&z[i]);
			fscanf (inputfile, "%lf",&vx[i]);
			fscanf (inputfile, "%lf",&vy[i]);
			fscanf (inputfile, "%lf",&vz[i]);
			fscanf (inputfile, "%lf",&spinx[i]);
			fscanf (inputfile, "%lf",&spiny[i]);
			fscanf (inputfile, "%lf",&spinz[i]);
			fscanf (inputfile, "%lf",&a1[i]);
			fscanf (inputfile, "%lf",&a2[i]);
			fscanf (inputfile, "%lf",&a3[i]);
			fscanf (inputfile, "%lf",&a4[i]);
			fscanf (inputfile, "%lf",&a5[i]);
			fscanf (inputfile, "%lf",&a6[i]);
			fscanf (inputfile, "%lf",&a7[i]);
			er = fscanf (inputfile, "%lf",&a8[i]);
			if(er <= 0){
				N = i;
printf("%d\n", N);
				break;
			}
		}


		double xsun, ysun, zsun;
		double vxsun, vysun, vzsun;
		double mtot;

		HelioToBary(m, x, y, z, vx, vy, vz, xsun, ysun, zsun, vxsun, vysun, vzsun, mtot, Msun, N);

		fprintf(outputfile,"%.20g %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g\n", time[0], -1, mtot, 0.0, xsun, ysun, zsun, vxsun, vysun, vzsun, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

		for(int i = 0; i < N; ++i){
			fprintf(outputfile,"%.20g %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g\n", time[i], index[i], m[i], r[i], x[i], y[i], z[i], vx[i], vy[i], vz[i], spinx[i], spiny[i], spinz[i], a1[i], a2[i], a3[i], a4[i], a5[i], a6[i], a7[i], a8[i]);
		}
		fclose(outputfile);
		fclose(inputfile);
	}
}

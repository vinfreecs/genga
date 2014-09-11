#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>


void aei(double3 x4i, double3 v4i, double mu, double &a, double &e, double &inc){

	double rsq, vsq, u, ir, ia;
	double t1, ria, ien2, ec, es2;
	double3 h3;
	double h2, h, t;

        rsq = x4i.x*x4i.x + x4i.y*x4i.y + x4i.z*x4i.z;
        vsq = v4i.x*v4i.x + v4i.y*v4i.y + v4i.z*v4i.z;
        u =  x4i.x*v4i.x + x4i.y*v4i.y + x4i.z*v4i.z;
        ir = rsqrt(rsq);
        ia = 2.0*ir-vsq/mu;

	a = 1.0 / ia;

        t1 = ia*ia;
        ria = rsq*ir*ia;
        ien2 = 1.0 / (mu*t1*ia);
        ec = 1.0-ria;
        es2 = u * u * t1 * t1 * ien2;
        e = sqrt(ec*ec + es2);


	h3.x = x4i.y * v4i.z - x4i.z * v4i.y;
	h3.y = -x4i.x * v4i.z + x4i.z * v4i.x;
	h3.z = x4i.x * v4i.y - x4i.y * v4i.x;

	h2 = h3.x * h3.x + h3.y * h3.y + h3.z * h3.z;
	h = sqrt(h2);

	t = h3.z / h;

	if(t <= -1){
		inc = M_PI;
	}
	else{
		if(t < 1){
			inc = acos(t);
		}
		else inc = 0.0;
	}
}

int main(int argc, char*argv[]){

	long long int kmin = 0;
	long long int kmax = 100;
	int step = 1;
	char X[160];
        char inputfilename[160];
        char outputfilename[160];
        FILE *inputfile;


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
                        step = atoi(argv[i + 1]);
                }

	}

	printf("tmin: %ld, tmax: %ld, step: %d, Name: %s\n", kmin, kmax, step, X);

	int N = 110000;
	int NN = 0;
	double Msun = 1.0;

	double3 x, v, spin;
	double xOld;
	double m, r, a, e, inc;
	double s, t;
	int index;

	for(long long int k = kmin; k <= kmax; k += step){	    
                sprintf(outputfilename, "aei%s_%.12d.dat", X, k);

        	index = -1;
        	t = 1.0e8;

		sprintf(inputfilename, "Out%s_%.12d.dat", X, k);	
		inputfile = fopen(inputfilename, "r");
		if(inputfile == NULL){
printf("%s skipped\n", inputfilename);
			continue;
		}
printf("%s\n", inputfilename);
                FILE *outputfile;
                outputfile = fopen(outputfilename, "a");
		index = -1;
		x.x = 0.0;
		for(int i = 0; i < N; ++i){
			xOld = x.x;
                        fscanf (inputfile, "%lf",&t);
                        fscanf (inputfile, "%d",&index);
//printf("%d %g %d\n", i, t, index);
			fscanf (inputfile, "%lf",&m);
			fscanf (inputfile, "%lf",&r);
			fscanf (inputfile, "%lf",&x.x);
			fscanf (inputfile, "%lf",&x.y);
			fscanf (inputfile, "%lf",&x.z);
			fscanf (inputfile, "%lf",&v.x);
			fscanf (inputfile, "%lf",&v.y);
			fscanf (inputfile, "%lf",&v.z);
                        fscanf (inputfile, "%lf",&spin.x);
                        fscanf (inputfile, "%lf",&spin.y);
                        fscanf (inputfile, "%lf",&spin.z);
			fscanf (inputfile, "%lf",&s);
			fscanf (inputfile, "%lf",&s);
			fscanf (inputfile, "%lf",&s);
			fscanf (inputfile, "%lf",&s);
			fscanf (inputfile, "%lf",&s);
			fscanf (inputfile, "%lf",&s);
			fscanf (inputfile, "%lf",&s);
			fscanf (inputfile, "%lf",&s);
                        if(xOld == x.x){
                                NN = i;
printf("%d\n", NN);
                                break;
                        }
			aei(x, v, Msun + m, a, e, inc);
			fprintf(outputfile,"%.20g %d %.20g %.20g %.20g %g %g\n", t, index, a, e, inc, m, r);
		}
		fclose(outputfile);
	}

}

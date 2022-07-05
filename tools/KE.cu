#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

//in parabolic orbits, a is used as the periapsis distance q

void aei(double3 x4i, double3 v4i, double mu, double &a, double &e, double &inc, double &Omega, double &w, double &Theta, double &E, double &M){

	double rsq = x4i.x * x4i.x + x4i.y * x4i.y + x4i.z * x4i.z;
	double vsq = v4i.x * v4i.x + v4i.y * v4i.y + v4i.z * v4i.z;
	double u =  x4i.x * v4i.x + x4i.y * v4i.y + x4i.z * v4i.z;
	double ir = 1.0 / sqrt(rsq);
	double ia = 2.0 * ir - vsq / mu;

	a = 1.0 / ia;

	//inclination
	double3 h3;
	double h2, h, t;
	h3.x = ( x4i.y * v4i.z) - (x4i.z * v4i.y);
	h3.y = (-x4i.x * v4i.z) + (x4i.z * v4i.x);
	h3.z = ( x4i.x * v4i.y) - (x4i.y * v4i.x);

	h2 = h3.x * h3.x + h3.y * h3.y + h3.z * h3.z;
	h = sqrt(h2);

	t = h3.z / h;
	if(t < -1.0) t = -1.0;
	if(t > 1.0) t = 1.0;

	inc = acos(t);

	//longitude of ascending node
	double n = sqrt(h3.x * h3.x + h3.y * h3.y);
	Omega = acos(-h3.y / n);
	if(h3.x < 0.0){
		Omega = 2.0 * M_PI - Omega;
	}

	if(inc < 1.0e-10 || n == 0) Omega = 0.0;

	//argument of periapsis
	double3 e3;
	e3.x = ( v4i.y * h3.z - v4i.z * h3.y) / mu - x4i.x * ir;
	e3.y = (-v4i.x * h3.z + v4i.z * h3.x) / mu - x4i.y * ir;
	e3.z = ( v4i.x * h3.y - v4i.y * h3.x) / mu - x4i.z * ir;


	e = sqrt(e3.x * e3.x + e3.y * e3.y + e3.z * e3.z);

	t = (-h3.y * e3.x + h3.x * e3.y) / (n * e);
	if(t < -1.0) t = -1.0;
	if(t > 1.0) t = 1.0;
	w = acos(t);
	if(e3.z < 0.0) w = 2.0 * M_PI - w;
	if(n == 0) w = 0.0;

	//True Anomaly
	t = (e3.x * x4i.x + e3.y * x4i.y + e3.z * x4i.z) / e * ir;
	if(t < -1.0) t = -1.0;
	if(t > 1.0) t = 1.0;
	Theta = acos(t);
	if(u < 0.0){
		if(e < 1.0 - 1.0e-10){
			//elliptic
			Theta = 2.0 * M_PI - Theta;
		}
		else if(e > 1.0 + 1.0e-10){
			//hyperbolic
			Theta = -Theta;
		}
		else{
			//parabolic
			Theta = - Theta;
		}
	}

	//Non circular, equatorial orbit
	if(e > 1.0e-10 && inc < 1.0e-10){
		Omega = 0.0;
		w = acos(e3.x / e);
		if(e3.y < 0.0) w = 2.0 * M_PI - w;
	}

	//circular, inclinded orbit
		if(e < 1.0e-10 && inc > 1.0e-11){
		w = 0.0;
	}

	//circular, equatorial orbit
	if(e < 1.0e-10 && inc < 1.0e-11){
		w = 0.0;
		Omega = 0.0;
	}

	if(w == 0 && Omega != 0.0){
		t = (-h3.y * x4i.x + h3.x * x4i.y) / n * ir;
		if(t < -1.0) t = -1.0;
		if(t > 1.0) t = 1.0;
		Theta = acos(t);
		if(x4i.z < 0.0){
			if(e < 1.0 - 1.0e-10){
				//elliptic
				Theta = 2.0 * M_PI - Theta;
			}
			else if(e > 1.0 + 1.0e-10){
				//hyperbolic
				Theta = -Theta;
			}
			else{
				//parabolic
				Theta = -Theta;
			}
		}
	}
	if(w == 0 && Omega == 0.0){
		Theta = acos(x4i.x * ir);
		if(x4i.y < 0.0){
			if(e < 1.0 - 1.0e-10){
				//elliptic
				Theta = 2.0 * M_PI - Theta;
			}
			else if(e > 1.0 + 1.0e-10){
				//hyperbolic
				Theta = -Theta;
			}
			else{
				//parabolic
				Theta = -Theta;
			}
		}
	}

	if(e < 1.0 - 1.0e-10){
		//Eccentric Anomaly
		E = acos((e + cos(Theta)) / (1.0 + e * cos(Theta)));
		if(M_PI < Theta && Theta < 2.0 * M_PI) E = 2.0 * M_PI - E;

		//Mean Anomaly
		M = E - e * sin(E);
	}
	else if(e > 1.0 + 1.0e-10){
		//Hyperbolic Anomaly
		//named still E instead of H or F
		E = acosh((e + t) / (1.0 + e * t));
		if(Theta < 0.0) E = - E;

		M = e * sinh(E) - E;
	}
	else{
		//Parabolic Anomaly
		E = tan(Theta * 0.5);

		if(E > M_PI) E = E - 2.0 * M_PI;

		M = E + E * E * E / 3.0;

		//use a to store q
		a = h * h / mu * 0.5;
	}
}

int main(int argc, char*argv[]){

	long long int kmin = 0;
	long long int kmax = 100;
	long long int step = 1;
	int pmin = 0;			//used for FormatP = 0
	int pmax = 0;
	char X[160];
	char inputfilename[260];
	char outputfilename[260];
	FILE *inputfile;
	int useCollfile = 0;		//reads Collisionfile and transforms into aei
	double Msun = 1.0;

	for(int i = 1; i < argc; i += 2){

		if(strcmp(argv[i], "-tmin") == 0){
			kmin = atoll(argv[i + 1]);
		}
		else if(strcmp(argv[i], "-tmax") == 0){
			kmax = atoll(argv[i + 1]);
		}
		else if(strcmp(argv[i], "-pmin") == 0){
			pmin = atoi(argv[i + 1]);
		}
		else if(strcmp(argv[i], "-pmax") == 0){
			pmax = atoi(argv[i + 1]);
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
		else if(strcmp(argv[i], "-Coll") == 0){
			useCollfile = atoi(argv[i + 1]);
		}
	}

	if(useCollfile == 1){
		kmin = 0;
		kmax = 1;
		step = 1;
	}

	if(pmax > 0){
		printf("pmin: %d, pmax: %d, Name: %s, Msun: %lf\n", pmin, pmax, X, Msun);
	}
	else{
		printf("tmin: %lld, tmax: %lld, step: %lld, Name: %s, Msun: %lf\n", kmin, kmax, step, X, Msun);
	}

	int N = 500000;
	int NN = 0;

	double3 x, v, spin;
	double xOld;
	double m, r, a, e, inc, Omega, w, Theta, E, M;
	double s;
	double t = 0.0;
	int index;
	int indexOld;
	double timeOld;

	//FormatT = 1, FormatP = 0
	if(pmax > 0){
		printf("Format T 0, Format P 1\n");
		for(int p = pmin; p <= pmax; ++p){
			sprintf(outputfilename, "aei%s_p%.6d.dat", X, p);
			sprintf(inputfilename, "Out%s_p%.6d.dat", X, p);	
			inputfile = fopen(inputfilename, "r");
			if(inputfile == NULL){
printf("%s skipped %d\n", inputfilename, p);
				continue;
			}
printf("%s\n", inputfilename);
			FILE *outputfile;
			outputfile = fopen(outputfilename, "w");

			for(long long int tt = 0ll; tt < 1e12; ++tt){
				xOld = x.x;
				timeOld = t;
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
				if(xOld == x.x && t == timeOld){
					break;
				}
				aei(x, v, Msun + m, a, e, inc, Omega, w, Theta, E, M);
				fprintf(outputfile,"%.20g %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %g %g\n", t, index, a, e, inc, Omega, w, Theta, E, M, m, r);
			}


		}

	}
	else{	


		for(long long int k = kmin; k <= kmax; k += step){
			if(useCollfile == 0){
				sprintf(outputfilename, "aei%s_%.12lld.dat", X, k);
				//sprintf(outputfilename, "aei%s.dat", X);
			}
			else{
				sprintf(outputfilename, "Collisions_aei%s.dat", X);
			}

			if(useCollfile == 0){
				sprintf(inputfilename, "Out%s_%.12lld.dat", X, k);	
				//sprintf(inputfilename, "Out%s.dat", X);
			}
			else{
				sprintf(inputfilename, "Collisions%s.dat", X);
			}

			inputfile = fopen(inputfilename, "r");
			if(inputfile == NULL){
				printf("%s skipped %lld\n", inputfilename, k);
				//continue;
				break;
			}
			printf("%s\n", inputfilename);
			FILE *outputfile;
			outputfile = fopen(outputfilename, "w");
			index = -1;
			x.x = 1.0e300;
			t = 1.0e8;
			if(useCollfile == 0){
				for(int i = 0; i < N; ++i){
					xOld = x.x;
					indexOld = index;
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
					if(xOld == x.x && indexOld == index){
						NN = i;
	printf("%d\n", NN);
						break;
					}
					aei(x, v, Msun + m, a, e, inc, Omega, w, Theta, E, M);
					fprintf(outputfile,"%.20g %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %g %g\n", t, index, a, e, inc, Omega, w, Theta, E, M, m, r);
				}
			}
			else{
				for(int i = 0; i < N; ++i){
					int er = 0;
					xOld = x.x;
					indexOld = index;
					//planet i
					fscanf (inputfile, "%lf",&t);
					fscanf (inputfile, "%d",&index);
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
					er = fscanf (inputfile, "%lf",&spin.z);
					if(er < 0){
						NN = i;
	printf("%d\n", NN);
						break;
					}
					aei(x, v, Msun + m, a, e, inc, Omega, w, Theta, E, M);
					fprintf(outputfile,"%.20g %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %g %g ", t, index, a, e, inc, Omega, w, Theta, E, M, m, r);
					
					//planet j
					fscanf (inputfile, "%d",&index);
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
					er = fscanf (inputfile, "%lf",&spin.z);
					if(er < 0){
						NN = i;
	printf("%d\n", NN);
						break;
					}
					aei(x, v, Msun + m, a, e, inc, Omega, w, Theta, E, M);
					fprintf(outputfile,"%d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %g %g\n", index, a, e, inc, Omega, w, Theta, E, M, m, r);
				}

			}
			fclose(outputfile);
			fclose(inputfile);
		}
	}
}

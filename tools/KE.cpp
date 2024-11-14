#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


//default values
#define def_Ninformat 55                //number of entries in informat array
#define def_OutputFileFormat "<< t i m r x y z vx vy vz Sx Sy Sz amin amax emin emax aec aecT encc test >>"

struct double3{
        double x;
        double y;
        double z;
};

struct double4{
        double x;
        double y;
        double z;
        double w;
};

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



int assignInformat(char *ff, char fileFormat[][5], int &format){
	int check = 0;  

	for(int i = 0; i < def_Ninformat; ++i){
		if(strcmp(ff, fileFormat[i]) == 0){
			format = i;
			check = 1;
		}               
	}                       

	if(check == 0){
		if(strcmp(ff, ">>") == 0){
			return 2;
		}               
		else if(strcmp(ff, "<<") == 0){
		}       
		else {  
			printf("Error: Input or output format not valid! Maybe the spaces in << ... >> have been forgotten\n");
			return 1;
		}

	}                       
	return 0;
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



	int outformat[def_Ninformat];
	char fileFormat[def_Ninformat][5];

	for(int i = 0; i < def_Ninformat; ++i){
		sprintf(fileFormat[i], "%s", "_");
	}
	//parameters must be less than 5 characters long
	sprintf(fileFormat[ 1], "%s", "x");
	sprintf(fileFormat[ 2], "%s", "y");
	sprintf(fileFormat[ 3], "%s", "z");
	sprintf(fileFormat[ 4], "%s", "m");
	sprintf(fileFormat[ 5], "%s", "vx");
	sprintf(fileFormat[ 6], "%s", "vy");
	sprintf(fileFormat[ 7], "%s", "vz");
	sprintf(fileFormat[ 8], "%s", "r");
	sprintf(fileFormat[ 9], "%s", "rho"); 
	sprintf(fileFormat[10], "%s", "Sx");
	sprintf(fileFormat[11], "%s", "Sy");
	sprintf(fileFormat[12], "%s", "Sz");
	sprintf(fileFormat[13], "%s", "i");
	sprintf(fileFormat[14], "%s", "-");
	sprintf(fileFormat[15], "%s", "amin");  //aelimits
	sprintf(fileFormat[16], "%s", "amax");
	sprintf(fileFormat[17], "%s", "emin");
	sprintf(fileFormat[18], "%s", "emax");
	sprintf(fileFormat[19], "%s", "t"); 
	sprintf(fileFormat[20], "%s", "k2");
	sprintf(fileFormat[21], "%s", "k2f");
	sprintf(fileFormat[22], "%s", "tau");
	sprintf(fileFormat[23], "%s", "a");
	sprintf(fileFormat[24], "%s", "e");
	sprintf(fileFormat[25], "%s", "inc");
	sprintf(fileFormat[26], "%s", "O");
	sprintf(fileFormat[27], "%s", "w");
	sprintf(fileFormat[28], "%s", "M");
	sprintf(fileFormat[29], "%s", "aL");
	sprintf(fileFormat[30], "%s", "eL");
	sprintf(fileFormat[31], "%s", "incL");
	sprintf(fileFormat[32], "%s", "mL");
	sprintf(fileFormat[33], "%s", "OL");
	sprintf(fileFormat[34], "%s", "wL");
	sprintf(fileFormat[35], "%s", "ML");
	sprintf(fileFormat[36], "%s", "rL");
	sprintf(fileFormat[37], "%s", "saT");
	sprintf(fileFormat[38], "%s", "P");
	sprintf(fileFormat[39], "%s", "PL");
	sprintf(fileFormat[40], "%s", "T");
	sprintf(fileFormat[41], "%s", "TL");
	sprintf(fileFormat[42], "%s", "Rc");    //Rcrit
	sprintf(fileFormat[43], "%s", "gw");    //gamma w
	sprintf(fileFormat[44], "%s", "Ic");    //Moment of Inertia
	sprintf(fileFormat[45], "%s", "test");
	sprintf(fileFormat[46], "%s", "encc");  //enccountT
	sprintf(fileFormat[47], "%s", "aec");   //aecount
	sprintf(fileFormat[48], "%s", "aecT");  //aecountT
	sprintf(fileFormat[49], "%s", "mig");   //artificial migration time scale
	sprintf(fileFormat[50], "%s", "mige");  //artificial migration time scale e
	sprintf(fileFormat[51], "%s", "migi");  //artifitial migration time scale i



	char oformat[def_Ninformat * 5];
	sprintf(oformat, def_OutputFileFormat); 

	for(int i = 0; i < def_Ninformat; ++i){
		outformat[i] = 0;
	}

	int pos = 0;
	for(int f = -1; f < def_Ninformat; ++f){
		char ff[5];
		int n = 0;
		int er = sscanf(oformat + pos, "%s%n", ff, &n);
		if(er <= 0) break;

		pos += n;

		er = assignInformat(ff, fileFormat, outformat[f]);
		if(er == 2) break;

	}

	//for(int f = 0; f < def_Ninformat; ++f){
	//	printf("%d %d\n", f, outformat[f]);
	//}


	//-------------------------------------------
	//Read paramKE.dat file
	//-------------------------------------------
	FILE *paramfile;
	paramfile = fopen("paramKE.dat", "r");



	char sp[160];           
	int er;                 
	char *str;      //Needed for return value of fgest, otherwise a compiler warning is generated



	if(paramfile == NULL){
		printf("****************************************************************\n");
		printf("Warning, paramKE.dat file does not exist, use default parameters\n");
		printf("****************************************************************\n");
	}
	else{       
		for(int j = 0; j < 1000; ++j){ //loop around all lines in the paramKE.dat file
			int c;
			for(int i = 0; i < 50; ++i){
				c = fgetc(paramfile);
				if(c == EOF){
					break; 
				}
				sp[i] = char(c);
				if(c == '=' || c == ':'){
					sp[i + 1] = '\0';
					break;
				}
			}               
			if(c == EOF) break;
			if(strcmp(sp, "Output name =") == 0){
				er = fscanf (paramfile, "%s", X);
				if(er <= 0){
					printf("Error: Output name is not valid!\n");
					return 0;
				}
				str = fgets(sp, 3, paramfile);
			}
			else if(strcmp(sp, "step =") == 0){
				er = fscanf (paramfile, "%lld", &step);
				if(er <= 0){
					printf("Error: step value is not valid!\n");
					return 0;
				}
				str = fgets(sp, 3, paramfile);
			}
			else if(strcmp(sp, "tmin =") == 0){
				er = fscanf (paramfile, "%lld", &kmin);
				if(er <= 0){
					printf("Error: tmin value is not valid!\n");
					return 0;
				}
				str = fgets(sp, 3, paramfile);
			}
			else if(strcmp(sp, "tmax =") == 0){
				er = fscanf (paramfile, "%lld", &kmax);
				if(er <= 0){
					printf("Error: tmax value is not valid!\n");
					return 0;
				}
				str = fgets(sp, 3, paramfile);
			}
			else if(strcmp(sp, "pmin =") == 0){
				er = fscanf (paramfile, "%d", &pmin);
				if(er <= 0){
					printf("Error: pmin value is not valid!\n");
					return 0;
				}
				str = fgets(sp, 3, paramfile);
			}
			else if(strcmp(sp, "pmax =") == 0){
				er = fscanf (paramfile, "%d", &pmax);
				if(er <= 0){
					printf("Error: pmax value is not valid!\n");
					return 0;
				}
				str = fgets(sp, 3, paramfile);
			}
			else if(strcmp(sp, "Central Mass =") == 0){
				er = fscanf (paramfile, "%lf", &Msun);
				if(er <= 0){
					printf("Error: Central Mass value is not valid!\n");
					return 0;
				}
				str = fgets(sp, 3, paramfile);
			}
			else if(strcmp(sp, "Output file Format:") == 0){
				for(int i = 0; i < def_Ninformat; ++i){
					outformat[i] = 0;
				}
				//Read output file Format
				int f;
				for(f = -1; f < def_Ninformat; ++f){
					er = fscanf (paramfile, "%s", sp);


					int er2 = assignInformat(sp, fileFormat, outformat[f]);
					if(er2 == 2) break;
					if(er2 == 1) return 0;

				}       
				if(er <= 0){ 
					printf("Error: Output file format is not valid!\n");
					return 0;
				}
				str = fgets(sp, 3, paramfile);
			} 
			else{
				printf("Error: paramKE.dat file is not valid! %s\n", sp);
				return 0;
			}
		}
		fclose(paramfile);
	}

	//-------------------------------------------



	//-------------------------------------------
	//Read console arguments
	//-------------------------------------------
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
		else{
			printf("Error, console argument not valid.\n");
		}
	}
	//-------------------------------------------

	if(useCollfile == 1){
		kmin = 0;
		kmax = 1;
		step = 1;
	}

	printf("Name: %s\n", X);
	printf("Msun: %g\n", Msun);
	if(pmax > 0){
		printf("pmin: %d, pmax: %d\n", pmin, pmax);
	}
	else{
		printf("tmin: %lld, tmax: %lld, step: %lld\n", kmin, kmax, step);
	}


	//for(int f = 0; f < def_Ninformat; ++f){
	//	printf("%d %d\n", f, outformat[f]);
	//}


	int N = 500000;
	int NN = 0;

	double3 x, v;
	double m, r, a, e, inc, Omega, w, Theta, E, M;
	double s;
	double t = 0.0;
	int index;

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
				for(int f = 0; f < def_Ninformat; ++f){
					if(outformat[f] == 0){
						break;
					}
					else if(outformat[f] == 19){
						er = fscanf (inputfile, "%lf",&t);
					}
					else if(outformat[f] == 13){
						er = fscanf (inputfile, "%d",&index);
					}
					else if(outformat[f] == 4){
						er = fscanf (inputfile, "%lf",&m);
					}
					else if(outformat[f] == 8){
						er = fscanf (inputfile, "%lf",&r);
					}
					else if(outformat[f] == 1){
						er = fscanf (inputfile, "%lf",&x.x);
					}
					else if(outformat[f] == 2){
						er = fscanf (inputfile, "%lf",&x.y);
					}
					else if(outformat[f] == 3){
						er = fscanf (inputfile, "%lf",&x.z);
					}
					else if(outformat[f] == 5){
						er = fscanf (inputfile, "%lf",&v.x);
					}
					else if(outformat[f] == 6){
						er = fscanf (inputfile, "%lf",&v.y);
					}
					else if(outformat[f] == 7){
						er = fscanf (inputfile, "%lf",&v.z);
					}
					else{
						er = fscanf (inputfile, "%lf",&s);
					}
					if(f != 0 && er <= 0){
						printf("Error, file format does not match paramKE.dat file or default values\n");
						return 0;

					}

					if(f == 0 && er <= 0){
printf("T = %lld\n", tt);
						break;

					}
				}
				if(er <= 0){
					break;

				}

				aei(x, v, Msun + m, a, e, inc, Omega, w, Theta, E, M);

				fprintf(outputfile,"%.20g %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %g %g\n", t, index, a, e, inc, Omega, w, Theta, E, M, m, r);
			}
			fclose(inputfile);
			fclose(outputfile);


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
			int er;
			if(useCollfile == 0){
				for(int i = 0; i < N; ++i){
					for(int f = 0; f < def_Ninformat; ++f){
						if(outformat[f] == 0){
							break;
						}
						else if(outformat[f] == 19){
							er = fscanf (inputfile, "%lf",&t);
						}
						else if(outformat[f] == 13){
							er = fscanf (inputfile, "%d",&index);
						}
						else if(outformat[f] == 4){
							er = fscanf (inputfile, "%lf",&m);
						}
						else if(outformat[f] == 8){
							er = fscanf (inputfile, "%lf",&r);
						}
						else if(outformat[f] == 1){
							er = fscanf (inputfile, "%lf",&x.x);
						}
						else if(outformat[f] == 2){
							er = fscanf (inputfile, "%lf",&x.y);
						}
						else if(outformat[f] == 3){
							er = fscanf (inputfile, "%lf",&x.z);
						}
						else if(outformat[f] == 5){
							er = fscanf (inputfile, "%lf",&v.x);
						}
						else if(outformat[f] == 6){
							er = fscanf (inputfile, "%lf",&v.y);
						}
						else if(outformat[f] == 7){
							er = fscanf (inputfile, "%lf",&v.z);
						}
						else{
							er = fscanf (inputfile, "%lf",&s);
						}
						if(f != 0 && er <= 0){
							printf("Error, file format does not match paramKE.dat file or default values\n");
							return 0;

						}

						if(f == 0 && er <= 0){
							NN = i;
printf("N = %d\n", NN);
							break;

						}
					}
					if(er <= 0){
						break;

					}
					aei(x, v, Msun + m, a, e, inc, Omega, w, Theta, E, M);
					fprintf(outputfile,"%.20g %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %g %g\n", t, index, a, e, inc, Omega, w, Theta, E, M, m, r);
				}
			}
			else{
				for(int i = 0; i < N; ++i){
					int er = 0;
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
					fscanf (inputfile, "%lf",&s);
					fscanf (inputfile, "%lf",&s);
					er = fscanf (inputfile, "%lf",&s);
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
					fscanf (inputfile, "%lf",&s);
					fscanf (inputfile, "%lf",&s);
					er = fscanf (inputfile, "%lf",&s);
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

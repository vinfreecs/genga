#include "Host2.h"
// *******************************************************
// This is a template function for additional forces
// The velocities in this kernel are already converted to heliocentric coordinates
//
//June 2015
//Authors: Simon Grimm
// **********************************************************
__global__ void force(double4 *x4_d, double4 *v4_d, int *index_d, double *Msun_d, double *dt_d, double Ct, double *time_d, int N, int Nst, int UseForce){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	if(id < N){
/*	
		int st = 0;

		if(Nst > 1 && id < N) st = index_d[id] / 100;	//st is the sub simulation index

		double4 x4 = x4_d[id];
		double4 v4 = v4_d[id];
		int index = index_d[id];
		double Msun = Msun_d[st];			//This is the mass of the central star
		double dt = dt_d[st] * Ct;			//This is the time step to do
		double time = time_d[st] / 365.25;		//This is the time in years

		double3 a3;
		a3.x = 0.0; 	
		a3.y = 0.0;
		a3.z = 0.0;
		
		if(UseForce == 1){
			//Insert here the force 	
			
		}

		//apply the Kick
		v4.x += a3.x * dt;
		v4.y += a3.y * dt;
		v4.z += a3.z * dt;

		v4_d[id] = v4;
*/
	}
}

__constant__ int setElementsNumbers_c[4];
__constant__ int setElements_c[12];
//**************************************
// This function copies the setElements parameters to constant memor. This functions must be in
// the same file as the use of the constant memory
//
//Authors: Simon Grimm
//June 2015
//
//***************************************/
__host__ void Host::constantCopy3(int *Elements, int nelements, int nbodies, int nlines, int ncolumns){
	int setElementsNumbers[4] = {nelements, nbodies, nlines, ncolumns};	
        cudaMemcpyToSymbol(setElements_c, Elements, 12 * sizeof(int), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(setElementsNumbers_c, setElementsNumbers, 4 * sizeof(int), 0, cudaMemcpyHostToDevice);
}


__global__ void setElements(double4 *x4_d, double4 *v4_d, int *index_d, double *setElementsData_d, int *setElementsLine_d, double *Msun_d, double *dt_d, double *time_d, int N, int Nst){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	int line = setElementsLine_d[0];
	int nelements = setElementsNumbers_c[0];
	int nbodies = setElementsNumbers_c[1];
	int nlines = setElementsNumbers_c[2];
	int ncolumns = setElementsNumbers_c[3];

	if(id < nbodies){


		//Compute the Kepler Elements
		int st = 0;

		if(Nst > 1 && id < N) st = index_d[id] / 100;	//st is the sub simulation index


		//check if one of the Keplerian elements will be modified
		int doConversion = 0;
		for(int i = 0; i < nelements; ++i){
			if(setElements_c[i] == 3){
				doConversion = 1;
				break;
			}
			if(setElements_c[i] == 4){
				doConversion = 1;
				break;
			}
			if(setElements_c[i] == 5){
				doConversion = 1;
				break;
			}
			if(setElements_c[i] == 6){
				doConversion = 1;
				break;
			}
			if(setElements_c[i] == 7){
				doConversion = 1;
				break;
			}
		}


		double4 x4i = x4_d[id];
		double4 v4i = v4_d[id];

		if(x4i.w >= 0.0){

			//int index = index_d[id];
			double Msun = Msun_d[st];
			//double dt = dt_d[st];
			double time = time_d[st] / 365.25;		//time in years
			double mu = ksq * (Msun + x4i.w);

			double a, e, inc, Omega, w, Theta, E;
		
			if(doConversion == 1){

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
				if(u < 0.0) Theta = 2.0 * M_PI - Theta;


				//Non circular, equatorial orbit
				if(e > 1.0e-10 && inc < 1.0e-10){
					Omega = 0.0;
					w = acos(e3.x / e);
					if(e3.y < 0.0) w = 2.0 * M_PI - w;
				}
				
				//circular, inclindes orbit
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
					if(x4i.z < 0.0) Theta = 2.0 * M_PI - Theta;
				}
				if(w == 0 && Omega == 0.0){
					Theta = acos(x4i.x * ir);
					if(x4i.y < 0.0) Theta = 2.0 * M_PI - Theta;

				}

				//Eccentric Anomaly
				E = acos((e + cos(Theta)) / (1.0 + e * cos(Theta)));
				if(M_PI < Theta && Theta < 2.0 * M_PI) E = 2.0 * M_PI - E;

	//printf("K %g %g %g %g %g %g %g\n", a, e, inc, Omega, w, E, Theta);
			}
			//modify Elements

			int line1 = line + 1;
			if(line1 >= nlines - 1) line1 = nlines - 1;
			double time0, time1;
			double m0, m1;
			double r0, r1;
			double a0, a1;
			double e0, e1;
			double i0, i1;
			double Omega0, Omega1;
			double w0, w1;
			double M0, M1, M;

			for(int i = 0; i < nelements; ++i){
				if(setElements_c[i] == 1){
					time0 = setElementsData_d[line * ncolumns + i];
					time1 = setElementsData_d[line1 * ncolumns + i];
	//printf("t %g %g %g %d %d\n", time0, time1, time, line, line1);
					if(time >= time1 && line < nlines - 1){
						++line;
						if(line1 < nlines - 1){
							++line1;
						}
						i = -1;
					}
				}
				if(setElements_c[i] == 3){
					a0 = setElementsData_d[line * ncolumns + 1 + (i - 1) * nbodies + id];
					a1 = setElementsData_d[line1 * ncolumns + 1 + (i - 1) * nbodies + id];
	//printf("a %g %g\n", a0, a1);
				}
				if(setElements_c[i] == 4){
					e0 = setElementsData_d[line * ncolumns + 1 + (i - 1) * nbodies + id];
					e1 = setElementsData_d[line1 * ncolumns + 1 + (i - 1) * nbodies + id];
				}
				if(setElements_c[i] == 5){
					i0 = setElementsData_d[line * ncolumns + 1 + (i - 1) * nbodies + id];
					i1 = setElementsData_d[line1 * ncolumns + 1 + (i - 1) * nbodies + id];
				}
				if(setElements_c[i] == 6){
					Omega0 = setElementsData_d[line * ncolumns + 1 + (i - 1) * nbodies + id];
					Omega1 = setElementsData_d[line1 * ncolumns + 1 + (i - 1) * nbodies + id];
				}
				if(setElements_c[i] == 7){
					w0 = setElementsData_d[line * ncolumns + 1 + (i - 1) * nbodies + id];
					w1 = setElementsData_d[line1 * ncolumns + 1 + (i - 1) * nbodies + id];
				}
				if(setElements_c[i] == 8){
					m0 = setElementsData_d[line * ncolumns + 1 + (i - 1) * nbodies + id];
					m1 = setElementsData_d[line1 * ncolumns + 1 + (i - 1) * nbodies + id];
	//printf("m %g %g\n", m0, m1);
				}
				if(setElements_c[i] == 9){
					r0 = setElementsData_d[line * ncolumns + 1 + (i - 1) * nbodies + id];
					r1 = setElementsData_d[line1 * ncolumns + 1 + (i - 1) * nbodies + id];
	//printf("r %g %g\n", r0, r1);
				}
				if(setElements_c[i] == 10){
					M0 = setElementsData_d[line * ncolumns + 1 + (i - 1) * nbodies + id];
					M1 = setElementsData_d[line1 * ncolumns + 1 + (i - 1) * nbodies + id];
	//printf("r %g %g\n", r0, r1);
				}

			}
			setElementsLine_d[0] = line;
			
			double x = (time - time0) / (time1 - time0);
			if(time1 - time0 == 0 || time < time0) x = 0.0;

			for(int i = 0; i < nelements; ++i){
				if(setElements_c[i] == 3){
					a = (a0 + (a1 - a0) * x);
				}
				if(setElements_c[i] == 4){
					e = (e0 + (e1 - e0) * x);
				}
				if(setElements_c[i] == 5){
					inc = (i0 + (i1 - i0) * x);
				}
				if(setElements_c[i] == 6){
					Omega = (Omega0 + (Omega1 - Omega0) * x);
				}
				if(setElements_c[i] == 7){
					w = (w0 + (w1 - w0) * x);
				}
				if(setElements_c[i] == 8){
					x4i.w = (m0 + (m1 - m0) * x) * 3.0024584e-6; //convert Earth masses to Solar masses
				}
				if(setElements_c[i] == 9){
					v4i.w = (r0 + (r1 - r0) * x) * 6.68458712e-14; //convert cm in AU
				}
				if(setElements_c[i] == 10){
					M = (M0 + (M1 - M0) * x) * dayUnit;	//t0 epoch time day to day'
					M *= sqrt(mu / (a * a * a));		//Mean anomaly
	//if(id == 0) printf("%g ", M);
					M = fmod(M, 2.0*M_PI);
	//if(id == 0) printf("%g\n", M);
				}
			}
			for(int i = 0; i < nelements; ++i){
				if(setElements_c[i] == 10){
					E = M + e * 0.5;
					double Eold = E;
					for(int j = 0; j < 32; ++j){
						E = E - (E - e * sin(E) - M) / (1.0 - e * cos(E));
						if(fabs(E - Eold) < 1.0e-15) break;
						Eold = E;
					}
				}
			}

			if(doConversion == 1){
				//Convert to Cartesian Coordinates

				double cw = cos(w);
				double sw = sin(w);
				double cOmega = cos(Omega);
				double sOmega = sin(Omega);
				double ci = cos(inc);
				double si = sin(inc);

				double Px = cw * cOmega - sw * ci * sOmega;
				double Py = cw * sOmega + sw * ci * cOmega;
				double Pz = sw * si;

				double Qx = -sw * cOmega - cw * ci * sOmega;
				double Qy = -sw * sOmega + cw * ci * cOmega;
				double Qz = cw * si;

				double cE = cos(E);
				double sE = sin(E);
				double t1 = a * (cE - e);
				double t2 = a * sqrt(1.0 - e * e) * sE;

				x4i.x =  t1 * Px + t2 * Qx;
				x4i.y =  t1 * Py + t2 * Qy;
				x4i.z =  t1 * Pz + t2 * Qz;

				double t0 = 1.0 / (1.0 - e * cE) * sqrt(mu / a);
				t1 = -sE;
				t2 = sqrt(1.0 - e * e) * cE;

				v4i.x = t0 * (t1 * Px + t2 * Qx);
				v4i.y = t0 * (t1 * Py + t2 * Qy);
				v4i.z = t0 * (t1 * Pz + t2 * Qz);
			}

			x4_d[id] = x4i;
			v4_d[id] = v4i;
	//printf("%g %g %g %g %g %g %g %g\n", x4i.x, x4i.y, x4i.z, x4i.w, v4i.x, v4i.y, v4i.z, v4i.w);
		}	
	}
}

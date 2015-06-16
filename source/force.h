#include "Host2.h"
// *******************************************************
// This is a template function for additional forces
// The velocities in this kernel are already converted to heliocentric coordinates
//
//June 2015
//Authors: Simon Grimm
// **********************************************************
__global__ void force(double4 *x4_d, double4 *v4_d, int *index_d, double *Msun_d, double *dt_d, double Ct, double *time_d, int N, int Nst, int UseForce){
/*
	int idy = threadIdx.x;
	int id = blockIdx.x * blockIdx.x + idy;

	if(id < N){
		
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
	}
*/
}


__constant__ int setElementsNumbers_c[3];
__constant__ int setElements_c[10];
//**************************************
// This function copies the setElements parameters to constant memor. This functions must be iny
// the same file as the use of the constant memory
//
//Authors: Simon Grimm
//June 2015
//
//***************************************/
__host__ void Host::constantCopy3(int *Elements, int nelements, int nbodies, int nlines){
	int setElementsNumbers[3] = {nelements, nbodies, nlines};	
        cudaMemcpyToSymbol(setElements_c, Elements, 10 * sizeof(int), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(setElementsNumbers_c, setElementsNumbers, 3 * sizeof(int), 0, cudaMemcpyHostToDevice);
}


__global__ void setElements(double4 *x4_d, double4 *v4_d, int *index_d, double *setElementsData_d, int *setElementsLine_d, double *Msun_d, double *dt_d, double *time_d, int N, int Nst){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockIdx.x + idy;

	if(id < 1 /*N*/){


		int line = setElementsLine_d[0];
		int nelements = setElementsNumbers_c[0];
		int nlines = setElementsNumbers_c[2];


		//Compute the Kepler Elements
		int st = 0;

		if(Nst > 1 && id < N) st = index_d[id] / 100;	//st is the sub simulation index


		double4 x4i = x4_d[id];
		double4 v4i = v4_d[id];
//		int index = index_d[id];
		double Msun = Msun_d[st];
//		double dt = dt_d[st];
		double time = time_d[st] / 365.25;		//time in years
		double mu = ksq * (Msun + x4i.w);

		double rsq = x4i.x * x4i.x + x4i.y * x4i.y + x4i.z * x4i.z;
		double vsq = v4i.x * v4i.x + v4i.y * v4i.y + v4i.z * v4i.z;
		double u =  x4i.x * v4i.x + x4i.y * v4i.y + x4i.z * v4i.z;
		double ir = 1.0 / sqrtf(rsq);
		double ia = 2.0 * ir - vsq / mu;

		double a = 1.0 / ia;

		double t1 = ia * ia;
		double ria = rsq * ir * ia;
		double ien2 = 1.0 / (mu * t1 * ia);
		double ec = 1.0 - ria;
		double es2 = u * u * t1 * t1 * ien2;
		double e = sqrtf(ec * ec + es2);


		//inclination
		double3 h3;
		double h2, h, t;
		h3.x =  x4i.y * v4i.z - x4i.z * v4i.y;
		h3.y = -x4i.x * v4i.z + x4i.z * v4i.x;
		h3.z =  x4i.x * v4i.y - x4i.y * v4i.x;

		h2 = h3.x * h3.x + h3.y * h3.y + h3.z * h3.z;
		h = sqrtf(h2);

		t = h3.z / h;
	
		double inc;
		if(t <= -1){
			inc = M_PI;
		}
		else{
			if(t < 1){
				inc = acos(t);
			}
			else inc = 0.0;
		}

		//longitude of ascending node
		double n = sqrtf(h3.x * h3.x + h3.y * h3.y);
		double Omega;
		if(h3.x >= 0.0){
			Omega = acos(-h3.y / n);
		}
		else{
			Omega = 2.0 * M_PI - acos(-h3.y / n);
		}
		if(inc == 0.0) Omega = 0.0;

		//argument of periapsis
		double3 e3;
		e3.x = ( v4i.y * h3.z - v4i.z * h3.y) / mu - x4i.x * ir;
		e3.y = (-v4i.x * h3.z + v4i.z * h3.x) / mu - x4i.y * ir;
		e3.z = ( v4i.x * h3.y - v4i.y * h3.x) / mu - x4i.z * ir;

		t = (-h3.y * e3.x + h3.x * e3.y) / (n * e);
		double w = acos(t);
		if(e3.z < 0.0) w = 2.0 * M_PI - w;

		//True Anomaly
		t = (e3.x * x4i.x + e3.y * x4i.y + e3.z * x4i.z) / e * ir;
		double Theta = acos(t);
		if(u < 0.0) Theta = 2.0 * M_PI - Theta;

		//Eccentric Anomaly
		double E = acos((e + cos(Theta)) / (1.0 + e * cos(Theta)));
		if(M_PI < Theta && Theta < 2.0 * M_PI) E = 2.0 * M_PI - E;

		//modify Elements

		int line1 = line + 1;
		if(line1 >= nlines - 1) line1 = nlines - 1;
		double time0, time1;
		double m0, m1;
		double r0, r1;
		for(int i = 0; i < nelements; ++i){
			double t0 = setElementsData_d[line * nelements + i];
			double t1 = setElementsData_d[line1 * nelements + i];
			if(setElements_c[i] == 1){
				time0 = t0;
				time1 = t1;
//printf("t %g %g %g %d %d\n", t0, t1, time, line, line1);
				if(time >= t1 && line < nlines - 1){
					++line;
					if(line1 < nlines - 1){
						++line1;
					}
					i = -1;
				}
			}
			if(setElements_c[i] == 8){
				m0 = t0;
				m1 = t1;
//printf("m %g %g\n", t0, t1);
			}
			if(setElements_c[i] == 9){
				r0 = t0;
				r1 = t1;
//printf("r %g %g\n", t0, t1);
			}

		}
		setElementsLine_d[0] = line;
		
		double x = (time - time0) / (time1 - time0);
		if(time1 - time0 == 0 || time < time0) x = 0.0;
//		double timei = time0 + (time1 - time0) * x;

		for(int i = 0; i < nelements; ++i){
			if(setElements_c[i] == 8){
				x4i.w = (m0 + (m1 - m0) * x) * 3.0024584e-6; //convert Earth masses to Solar masses
			}
			if(setElements_c[i] == 9){
				v4i.w = (r0 + (r1 - r0) * x) * 6.68458712e-14; //convert cm in AU
			}
		}


//printf("%g %g %g %g\n", x, timei, x4i.w, v4i.w);

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
		t1 = a * (cE - e);
		double t2 = a * sqrtf(1.0 - e * e) * sE;

		x4i.x =  t1 * Px + t2 * Qx;
		x4i.y =  t1 * Py + t2 * Qy;
		x4i.z =  t1 * Pz + t2 * Qz;

		double t0 = 1.0 / (1.0 - e * cE) * sqrtf(mu / a);
		t1 = -sE;
		t2 = sqrtf(1.0 - e * e) * cE;

		v4i.x = t0 * (t1 * Px + t2 * Qx);
		v4i.y = t0 * (t1 * Py + t2 * Qy);
		v4i.z = t0 * (t1 * Pz + t2 * Qz);

		x4_d[id] = x4i;
		v4_d[id] = v4i;
	
	}
}

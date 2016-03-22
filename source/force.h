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
				h3.x = ( x4i.y * v4i.z) - (x4i.z * v4i.y);
				h3.y = (-x4i.x * v4i.z) + (x4i.z * v4i.x);
				h3.z = ( x4i.x * v4i.y) - (x4i.y * v4i.x);

				double h = sqrt(h3.x * h3.x + h3.y * h3.y + h3.z * h3.z);

				double t = h3.z / h;
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


__global__ void rotation_kernel(curandState *random_d, double4 *x4_d, double4 *v4_d, double3 *spin_d, int *index_d, double *dt_d, int N, int Nst, double time){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	int st = 0;

	if(Nst > 1 && id < N) st = index_d[id] / 100;	//st is the sub simulation index

	if(id < N){

		double4 x4i = x4_d[id];
		double4 v4i = v4_d[id];
		double3 spin = spin_d[id];
		curandState random = random_d[id];

		if(x4i.w >= 0.0){

			double rd = curand_uniform(&random);
//			random_d[id] = random;
//printf("%d %g\n", id, x);

			double RR = v4i.w * def_AU;	//covert radius in m 
			double V = 5000.0;		//collisional velocity in m / s
			double rho = 3500.0; 		//density of body in kg/m^3, /this is needed because test particles have zero mass

			//int index = index_d[id];
			double dt = dt_d[st];
			double m = x4i.w;
			if(x4i.w == 0.0){
				m = rho * 4.0 / 3.0 * M_PI * v4i.w * v4i.w * v4i.w * def_AU * def_AU * def_AU; 	//mass in Kg;
				m /= def_Solarmass;						//mass im Solar masses
			}

			//compute rotation vector from spin vector
			double iI = 5.0 / (2.0 * m * v4i.w * v4i.w); // inverse Moment of inertia of a solid sphere in 1/ (Solar Masses AU^2)
			double3 omega3;
			omega3.x = spin.x * iI;
			omega3.y = spin.y * iI;
			omega3.z = spin.z * iI;

			double omega = sqrt(omega3.x * omega3.x + omega3.y * omega3.y + omega3.z * omega3.z);   //angular velocity in 1 / day * 0.017
			omega *= 2.0 * M_PI * dayUnit / (24.0 * 3600.0); 					//in 1 / s


			//compute probability of rotation reset
			double t1 = 2.0 * sqrt(2.0) * omega / (5.0 * V);
			double p = 1.0e-18 / sqrt(sqrt(RR * RR * RR)) * pow(t1, -5.0/6.0); //probability per year
			p = p / 365.25 * dt / dayUnit;	//probability per time step

			if(rd < p){
				//reset the rotation rate an spin vector
				rd = curand_uniform(&random);
				double omin = 1.0 / (36.0 * 2.0 * RR);
				double omax = 1.0 / (2.0 * RR);
				omega = rd * (omax - omin) + omin;        //rotations per s
printf("%g %d %g\n", time, id, omega);
				omega = omega / dayUnit * 24.0 * 3600.0;  //rotation in 1 / day'

				double S = 2.0 / 5.0 * m * v4i.w * v4i.w * omega;
				//printf("%g %g %g %g\n", m, r, omega, Sz);
				double u = curand_uniform(&random);
				double theta = curand_uniform(&random) * 2.0 * M_PI;
				//sign
				double s = curand_uniform(&random);;

				double t2 = S * sqrt(1.0 - u * u);
				spin.x = t2 * cos(theta);
				spin.y = t2 * sin(theta);
				spin.z = S * u;

				if( s > 0.5){
					spin.z *= -1.0;
				}
//				spin_d[id] = spin;
			}
		}
	}
}


__global__ void CallYarkovsky2(double4 *x4_d, double4 *v4_d, double3 *spin_d, int *index_d, double *Msun_d, double *dt_d, double Ct, int N, int Nst){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	int st = 0;

	if(Nst > 1 && id < N) st = index_d[id] / 100;	//st is the sub simulation index

	if(id < N){

		double4 x4i = x4_d[id];
		double4 v4i = v4_d[id];
		double3 spin = spin_d[id];

		if(x4i.w >= 0.0){

			//material constants
			double K = 2.65; 		//Thermal conductivity in W/mK
			double rho = 3500.0; 		//density of body in kg/m^3, /this is needed because test particles have zero mass
			double C = 680.0;		//Specific Heat Capacity in J/kgK
			double eps = 0.95;		//Emissivity
			double sigma = 5.670373e-8;	//Stefan Boltzmann constant J m^-2 s^-1 K^-4
			double A = 0.2;			//Bond albedo
			double S = 1367.0;		//Solar Constant at 1 AU in W /m^2
//A = 0.0;
//eps = 1.0;

			double c = 299792458.0;		//speed of light im m/s
			double Gamma = sqrt(K * rho * C);	//surface thermal intertia 
			double RR = v4i.w * def_AU;		//covert radius in m 

			//int index = index_d[id];
			double Msun = Msun_d[st];
			double dt = dt_d[st] * Ct;
			double mu = ksq * (Msun + x4i.w);
			double m = x4i.w;
			if(x4i.w == 0.0){
				m = rho * 4.0 / 3.0 * M_PI * RR * RR * RR; 	//mass in Kg;
				m /= def_Solarmass;						//mass im Solar masses
				mu = ksq * (Msun + m);
			}
	
			double rsq = x4i.x * x4i.x + x4i.y * x4i.y + x4i.z * x4i.z;
			double vsq = v4i.x * v4i.x + v4i.y * v4i.y + v4i.z * v4i.z;
			double u =  x4i.x * v4i.x + x4i.y * v4i.y + x4i.z * v4i.z;
			double ir = 1.0 / sqrt(rsq);
			double ia = 2.0 * ir - vsq / mu;

			double a = fabs(1.0 / ia);

			double3 h3;
			h3.x = ( x4i.y * v4i.z) - (x4i.z * v4i.y);
			h3.y = (-x4i.x * v4i.z) + (x4i.z * v4i.x);
			h3.z = ( x4i.x * v4i.y) - (x4i.y * v4i.x);

			double h = sqrt(h3.x * h3.x + h3.y * h3.y + h3.z * h3.z);
		
			double n = sqrt(mu / (a * a * a)); //mean motion in 1 / day * 0.017 
			n *= dayUnit / (24.0 * 3600.0);  //mean motion  in 1 / s;
	
			//longitude of ascending node
			double nn = sqrt(h3.x * h3.x + h3.y * h3.y);

			//argument of periapsis
			double3 e3;
			e3.x = ( v4i.y * h3.z - v4i.z * h3.y) / mu - x4i.x * ir;
			e3.y = (-v4i.x * h3.z + v4i.z * h3.x) / mu - x4i.y * ir;
			e3.z = ( v4i.x * h3.y - v4i.y * h3.x) / mu - x4i.z * ir;
		
			double e = sqrt(e3.x * e3.x + e3.y * e3.y + e3.z * e3.z); 

			//compute rotation vetor from spin vector
			double iI = 5.0 / (2.0 * m * v4i.w * v4i.w); // inverse Moment of inertia of a solid sphere in 1/ (Solar Masses AU^2)
			double3 omega3;
			omega3.x = spin.x * iI;
			omega3.y = spin.y * iI;
			omega3.z = spin.z * iI;

			double omega = sqrt(omega3.x * omega3.x + omega3.y * omega3.y + omega3.z * omega3.z); 	//angular velocity in 1 / day * 0.017
	
		//Normalize spin vector
			omega3.x /= omega;
			omega3.y /= omega;
			omega3.z /= omega;

			double sp, sq;
			//True Anomaly
			double Theta;
			double t;
			if(e > 1.0e-10){
				t = (e3.x * x4i.x + e3.y * x4i.y + e3.z * x4i.z) / e * ir;
				if(t < -1.0) t = -1.0;
				if(t > 1.0) t = 1.0;
				Theta = acos(t);
				if(u < 0.0) Theta = 2.0 * M_PI - Theta;
	
				sp = (omega3.x * e3.x + omega3.y * e3.y + omega3.z * e3.z) / e;
				double3 q3;
				q3.x = ( h3.y * e3.z) - (h3.z * e3.y);
				q3.y = (-h3.x * e3.z) + (h3.z * e3.x);
				q3.z = ( h3.x * e3.y) - (h3.y * e3.x);
				sq = (omega3.x * q3.x + omega3.y * q3.y + omega3.z * q3.z) / (e * h);
			}
			else{
			//circular inclined orbit
				if(h3.z < h * (1.0 - 1.0e-11)){
					t = (-h3.y * x4i.x + h3.x * x4i.y) / nn * ir;
					if(t < -1.0) t = -1.0;
					if(t > 1.0) t = 1.0;
					Theta = acos(t);
					if(x4i.z < 0.0) Theta = 2.0 * M_PI - Theta;
		
					sp = (omega3.x * -h3.y + omega3.y * h3.x) / nn;
					double3 q3;
					q3.x = 0.0;
					q3.y = 0.0;
					q3.z = ( h3.x * h3.x) - (h3.y * -h3.y);
					sq = (omega3.x * q3.x + omega3.y * q3.y + omega3.z * q3.z) / (e * nn);
				}
			//circular equatorial orbit
				else{
					t = x4i.x * ir;
					Theta = acos(t);
					if(x4i.y < 0.0) Theta = 2.0 * M_PI - Theta;
		
					sp = (omega3.x);
					double3 q3;
					q3.x = 0.0;
					q3.y = h3.z;
					q3.z = h3.y;
					sq = (omega3.x * q3.x + omega3.y * q3.y + omega3.z * q3.z) / h;
				}
			}

			if(omega == 0){
				sp = 0.0;
				sq = 0.0;
			}
			//Eccentric Anomaly
			double E = acos((e + t) / (1.0 + e * t));
			if(M_PI < Theta && Theta < 2.0 * M_PI) E = 2.0 * M_PI - E;

			//Mean Anomaly
			double M = E - e * sin(E);

			if(e >= 1){
				E = acosh((e + t) / (1.0 + e * t));
				if(M_PI < Theta && Theta < 2.0 * M_PI) E = 2.0 * M_PI - E;
				M = E - e * sinh(E);
				

			}
	
//printf("a %d %g %g %g %g %g %g %g %g %g %g %g\n", id, a, e, m, RR, omega, v4i.x, v4i.y, v4i.z,  Theta, E, M);

			double3 rs3;
			rs3.x = (( x4i.y * omega3.z) - (x4i.z * omega3.y)) * ir;
			rs3.y = ((-x4i.x * omega3.z) + (x4i.z * omega3.x)) * ir;
			rs3.z = (( x4i.x * omega3.y) - (x4i.y * omega3.x)) * ir;

			double3 srs3;
			srs3.x = (( omega3.y * rs3.z) - (omega3.z * rs3.y));
			srs3.y = ((-omega3.x * rs3.z) + (omega3.z * rs3.x));
			srs3.z = (( omega3.x * rs3.y) - (omega3.y * rs3.x));


			omega *= 2.0 * M_PI * dayUnit / (24.0 * 3600.0); 						//in 1 / s

			double d = a * (1.0 + e*e * 0.5);//time averaged heliocentric distance in AU
			double F = S / (d * d);		//scaled heliocentric distance, F = SEarth * (aEarth/a)^2

			double Ts4 = (1.0 - A) * F / (eps * sigma);
			double Ts = sqrt(sqrt(Ts4));

			double t1 = Gamma / (eps * sigma * Ts * Ts * Ts);
			double t2 = (1.0 - A) * 3.0 * F / (9.0 * rho * RR * c);

			double s2 = sqrt(2.0);

			double3 a3;
			a3.x = 0.0;
			a3.y = 0.0;
			a3.z = 0.0;

			//Diurnal 
			{
			double ilD = sqrt(rho * C * omega / K);
			double ThetaD = t1 * sqrt(omega);
			double X = s2 * RR * ilD;
			double lamda = ThetaD / X;
			double L = lamda / (1.0 + lamda);

			double cX = cos(X);
			double sX = sin(X);

			double X2cX = (X - 2.0) * cX;
			double X2sX = (X - 2.0) * sX;
			double eX = exp(-X);

			double Ax = -eX * (X + 2.0) - (X2cX - X * sX);
			double Bx = -eX * X - (X * cX + X2sX);
			double Cx = Ax + L * (eX * 3.0 * (X + 2.0) + (3.0 * X2cX + X * (X - 3.0) * sX));
			double Dx = Bx + L * (eX * X * (X + 3.0) - (X * (X - 3.0) * cX - 3.0 * X2sX));

			double iC2D2 = 1.0 / (Cx * Cx + Dx * Dx);

			double Gcd = (Ax * Cx + Bx * Dx) * iC2D2;
			double Gsd = (Bx * Cx - Ax * Dx) * iC2D2;

			double WD = t2 / (1.0 + lamda);
			a3.x += WD * (Gsd * rs3.x + Gcd * srs3.x);
			a3.y += WD * (Gsd * rs3.y + Gcd * srs3.y);
			a3.z += WD * (Gsd * rs3.z + Gcd * srs3.z);
//		printf("D %d %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g\n", id, a3.x, a3.y, a3.z, srs3.x, srs3.y, srs3.z, Gcd, Gsd, WD, lamda);
			}
			
			//seasonal
			{
			double ilS = sqrt(rho * C * n / K);
			double ThetaS = t1 * sqrt(n);
			double eta = sqrt(1.0 - e * e);
			if(e >= 1.0) eta = 1.0;
		int k = 1;
	
			double e2 = e * e;
			double e3 = e * e2;
			double e4 = e2 * e2;
			//double e5 = e2 * e3;
			double e6 = e3 * e3;

			double alpha = 1.0 - 0.375 * e2 + 5.0 / 6.0 * 0.25 * e4 - 7.0 / 72.0 / 128.0;
			double beta = 1.0 - e2 / 8.0 + e4 / 192.0 - e6 / 9216.0;
	
			double X = s2 * RR * ilS;
			double lamda = ThetaS / X * sqrt(sqrt(eta * eta * eta));
			double L = lamda / (1.0 + lamda);

			double cX = cos(X);
			double sX = sin(X);

			double X2cX = (X - 2.0) * cX;
			double X2sX = (X - 2.0) * sX;

			double eX = exp(-X);
			double Ax = -eX * (X + 2.0) - (X2cX - X * sX);
			double Bx = -eX * X - (X * cX + X2sX);
			double Cx = Ax + L * (eX * 3.0 * (X + 2.0) + (3.0 * X2cX + X * (X - 3.0) * sX));
			double Dx = Bx + L * (eX * X * (X + 3.0) - (X * (X - 3.0) * cX - 3.0 * X2sX));

			double iC2D2 = 1.0 / (Cx * Cx + Dx * Dx);

			double Gcd = (Ax * Cx + Bx * Dx) * iC2D2;
			double Gsd = (Bx * Cx - Ax * Dx) * iC2D2;
			double cM = cos(k * M); 
			double sM = sin(k * M); 

			double WS = (sp * alpha * (cM * Gcd - sM * Gsd) + sq * beta * (sM * Gcd + cM * Gsd)) / (1.0 + lamda);
			double aS = t2 * WS;
			a3.x += aS * omega3.x;
			a3.y += aS * omega3.y;
			a3.z += aS * omega3.z;
//		printf("S %d %g %g %g %.10g %.10g %.10g %.10g %.10g %g %g %g %g %g\n", id, a3.x, a3.y, a3.z, sp, sq, sp * sp + sq * sq, RR, Gcd, Gsd, n, lamda, sM, cM);
			}

			a3.x *= 24.0 * 3600.0 * 24.0 * 3600.0 / (def_AU * dayUnit * dayUnit); //in AU /day^2 * 0.017^2
			a3.y *= 24.0 * 3600.0 * 24.0 * 3600.0 / (def_AU * dayUnit * dayUnit);
			a3.z *= 24.0 * 3600.0 * 24.0 * 3600.0 / (def_AU * dayUnit * dayUnit);

		//printf("%g %g %g %g %g %g %g\n", RR, a, omega, n, a3.x, a3.y, a3.z);

			v4i.x += a3.x * dt;
			v4i.y += a3.y * dt;
			v4i.z += a3.z * dt;

			v4_d[id] = v4i;

	//printf("%g %g %g %g %g %g %g %g\n", x4i.x, x4i.y, x4i.z, x4i.w, v4i.x, v4i.y, v4i.z, v4i.w);
		}	

	}
}



// Yarkovski
/*
__device__ void alpha(double e){
	
	double e2 = e * e;
	double e3 = e * e2;
	double e4 = e2 * e2;
	double e5 = e2 * e3;
	double e6 = e3 * e3;

	double alpha1 = 1.0 - 0.375 * e2 + 5.0 / 6.0 * 0.25 * e4 - 7.0 / 72.0 / 128.0;
	double alpha2 = 4.0 * (0.5 * e - 3.0 *  e3 + 0.0625 * e5);
	double alpha3 = 9.0 * (0.375 * e2 - 11.25 / 32.0 * e4 + 567 / 5120.0 * e6);
	double alpha4 = 16.0 * (1.0 / 3.0 * e3 - 0.4 * e5);
	double alpha5 = 25.0 * (125.0 / 384.0 * e4 - 4375.0 / 9216.0 * e6);
	double alpha6 = 36.0 * (108.0 / 320.0 * e5);
	double alpha7 = 49.0 * (16807.0 / 46080.0 * e6);

	double beta1 = 1.0 - e2 / 8.0 + e4 / 192.0 - e6 / 9216.0;
	double beta2 = 2.0 * e * (1.0 - e2 / 3.0 + e4 / 24.0);
	double beta3 = 27.0 / 8.0 * e2 * (1.0 - 9.0 / 16.0 * e2 + 81.0 / 640.0 * e4);
	double beta4 = 16.0 / 3.0 * e3 * (1.0 - 0.8 * e2);
	double beta5 = 25.0 * 125.0 / 384.0 * e4 * (1.0 - 25.0 / 24.0 * e2);
	double beta6 = 972.0 / 80.0 * e5;
	double beta7 = 823543.0 / 46080.0 * e6;  

}

*/

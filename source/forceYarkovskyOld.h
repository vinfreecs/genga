#include "Host2.h"

#define PVERSION 3
//1 Farinella 1998
//2 Vokroulicky et Al 2015
//3 Vokroulicky et Al 2000

//see Vokrouhlicky et al 2001
//a semi major axis
//e eccentricity
//mu gravitational parameter
//m mass in Solar Mass
//Radius in AU
//h3 angular momentum vector
//h norm of angular momentum vector |r x v|
//dt time step in day / 0.017
__device__ void Yarkovski(double &a, const double e, double m, const double mu, const double R, const double4 spin, const double3 h3, const double h, const double dt){

	//material constants
//A = 0.0;
//eps = 1.0;

	double Gamma = sqrt(Asteroid_K_c[0] * Asteroid_rho_c[0] * Asteroid_C_c[0]);	//surface thermal intertia 
	double RR = R * def_AU;		//covert radius in m 

//comppute mass through density
if(m == 0.0) m = Asteroid_rho_c[0] * 4.0 / 3.0 * M_PI * RR * RR * RR; 	//mass in kg;
	m /= 1.98855e30;						//mass im Solar masses

	double d = a * (1.0 + e*e * 0.5);//time averaged heliocentric distance in AU
	double F = SolarConstant_c[0] / (d * d);		//scaled heliocentric distance, F = SEarth * (aEarth/a)^2
	
	double n;
	if(e < 1.0 - 1.0e-10){
		//Elliptic
		n = sqrt(mu / (a * a * a)); //mean motion in 1 / day * 0.017 
	}
	else if(e > 1.0 + 1.0e-10){
		//hyperbolic
		n = sqrt(mu / (-a * a * a)); //mean motion in 1 / day * 0.017 
	}
	else{
		//parabolic
		n = sqrt(mu); //mean motion in 1 / day * 0.017 
	}
	n *= dayUnit / (24.0 * 3600.0);  //mean motion  in 1 / s;

	double Ts4 = (1.0 - Asteroid_A_c[0]) * F / (Asteroid_eps_c[0] * def_sigma);
#if PVERSION == 1
	Ts4 *= 0.25;
#endif

	double Ts = sqrt(sqrt(Ts4));

	double t1 = Gamma / (Asteroid_eps_c[0] * def_sigma * Ts * Ts * Ts);
	double t2 = (1.0 - Asteroid_A_c[0]) * 3.0 * F / (9.0 * n * Asteroid_rho_c[0] * RR * def_c); // a factor of 4 is cancelled with da/dt, n = nu / (2pi)

	//compute rotation vetor from spin vector
	double Ic = spin.w;
	double iI = 1.0 / (Ic * m * R * R); // inverse Moment of inertia of a solid sphere in 1/ (Solar Masses AU^2)
	double3 omega3;
	omega3.x = spin.x * iI;
	omega3.y = spin.y * iI;
	omega3.z = spin.z * iI;

	double omega = sqrt(omega3.x * omega3.x + omega3.y * omega3.y + omega3.z * omega3.z); 	//angular velocity in 1 / day * 0.017
	double cgamma = (h3.x * omega3.x + h3.y * omega3.y + h3.z * omega3.z) / (h * omega);	//h X omega = |h|*|omega|*cos(gamma)  
	cgamma = fmax(cgamma, -1.0);
	cgamma = fmin(cgamma, 1.0);
	
	omega *= 2.0 * M_PI * dayUnit / (24.0 * 3600.0); 						//in 1 / s
#if PVERSION == 1
	t1 /= (2.0 * M_PI);
#endif
//printf("%g %g %g %g %g %g %g %g %g %g %g %.20g %g\n", m, RR, n, omega, a, h3.x, h3.y, h3.z, omega3.x, omega3.y, omega3.z, cgamma, acos(cgamma));
	//Diurnal 
	double ThetaD = t1 * sqrt(omega);
#if PVERSION == 1
	//Farinella 1998
	double WD = 0.667 * ThetaD / (1.0 + 2.03 * ThetaD + 2.04 * ThetaD * ThetaD);
	double dadtD = t2 / 3.0 * 9.0 * WD * cgamma;
#else
	//Vokroulicky 2015
	double WD = -0.5 * ThetaD / (1.0 + ThetaD + 0.5 * ThetaD * ThetaD);
	double dadtD = -2.0 * t2 * WD * cgamma;	// in m/s	
#endif
	
	//seasonal
	RR *= sqrt(Asteroid_rho_c[0] * Asteroid_C_c[0] * n / Asteroid_K_c[0]); 

	double dadtS = 0.0;

#if PVERSION == 1
	//This is not Farinella 1998
	double ilS = sqrt(Asteroid_rho_c[0] * Asteroid_C_c[0] * n / Asteroid_K_c[0]);
	double ThetaS = t1 * sqrt(n);

	double X = RR * ilS;
	double lamda = ThetaS / X;
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

	//double Gcd = (Ax * Cx + Bx * Dx) * iC2D2;
	double Gsd = (Bx * Cx - Ax * Dx) * iC2D2;

	double WS = Gsd / (3.0 * (1.0 + lamda));
	double sgamma2 = 1.0 - cgamma * cgamma;	//sin^2 + cos^2 = 1.0

	dadtS = - t2 * 3.0 * WS * sgamma2; // in m/s
#endif
#if PVERSION == 2
	double s2 = sqrt(2.0);

	double kappa1 = s2 / (2.0 * RR);
	double kappa2 = 0.5 / (RR * RR);
	double kappa3 = s2 * 0.1 * RR; 

	if(RR >= s2) kappa1 = 0.5;
	if(RR >= 1.0) kappa2 = 0.5;
	if(RR >= 5.0 / s2) kappa3 = 0.5;

	double ThetaS = t1 * sqrt(n);

	double WS = kappa3 * ThetaS / (1.0 + 2.0 * kappa1 * ThetaS + kappa2 * ThetaS * ThetaS);
	double sgamma2 = 1.0 - cgamma * cgamma;	//sin^2 + cos^2 = 1.0

	dadtS = t2 * WS * sgamma2; // in m/s
#endif
//printf("%g %g %g %g %g %g %g\n", R * AU, a, acos(cgamma), omega, n, dadtS / AU * 3600 * 24 * 365.25 * 1000000.0, dadtD / AU * 3600 * 24 * 365.25 * 1000000.0);

	double dadt = (dadtD + dadtS) / (def_AU * dayUnit) * 24.0 * 3600.0; //in AU / day * 0.017

	a += dadt * dt;
	
}	

// ***************************************************************
// This kernel computes the seasonal and diurnal Yarkovsky effect.
// it computes the yarkovsky drift rate da/dt and modifies
// the Keplerian elements.

// See Vokroulicky Milani Chesley 2000

// March 2017
// Authors: Simon Grimm, Matthias Meier
// *****************************************************************
__device__ void Yarkovski2(double &a, const double e, double m, const double Msun, const double R, const double4 spin, const double3 h3, const double h, const double dt){

	//material constants
//A = 0.0;
//eps = 1.0;

	double Gamma = sqrt(Asteroid_K_c[0] * Asteroid_rho_c[0] * Asteroid_C_c[0]);	//surface thermal intertia 
	double RR = R * def_AU;		//covert radius in m 

	double mu = def_ksq * (Msun + m);
//comppute mass through density
	if(m == 0.0){
		m = Asteroid_rho_c[0] * 4.0 / 3.0 * M_PI * RR * RR * RR; 	//mass in kg;
		m /= 1.98855e30;						//mass im Solar masses
		mu = def_ksq * (Msun + m);
	}

	double d = a * (1.0 + e*e * 0.5);//time averaged heliocentric distance in AU
	double F = SolarConstant_c[0] / (d * d);		//scaled heliocentric distance, F = SEarth * (aEarth/a)^2
	
	double n;
	if(e < 1.0 - 1.0e-10){
		//Elliptic
		n = sqrt(mu / (a * a * a)); //mean motion in 1 / day * 0.017 
	}
	else if(e > 1.0 + 1.0e-10){
		//hyperbolic
		n = sqrt(mu / (-a * a * a)); //mean motion in 1 / day * 0.017 
	}
	else{
		//parabolic
		n = sqrt(mu); //mean motion in 1 / day * 0.017 
	}
	n *= dayUnit / (24.0 * 3600.0);  //mean motion  in 1 / s;

	double Ts4 = (1.0 - Asteroid_A_c[0]) * F / (Asteroid_eps_c[0] * def_sigma);
	double Ts = sqrt(sqrt(Ts4));

	double t1 = Gamma / (Asteroid_eps_c[0] * def_sigma * Ts * Ts * Ts);
	double t2 = (1.0 - Asteroid_A_c[0]) * 3.0 * F / (9.0 * n * Asteroid_rho_c[0] * RR * def_c); // a factor of 4 is cancelled with da/dt, n = nu / (2pi)

	//compute rotation vector from spin vector
	double Ic = spin.w;
	double iI = 1.0 / (Ic * m * R * R); // inverse Moment of inertia of a solid sphere in 1/ (Solar Masses AU^2)
	double3 omega3;
	omega3.x = spin.x * iI;
	omega3.y = spin.y * iI;
	omega3.z = spin.z * iI;

	double omega = sqrt(omega3.x * omega3.x + omega3.y * omega3.y + omega3.z * omega3.z); 	//angular velocity in 1 / day * 0.017
	double cgamma = (h3.x * omega3.x + h3.y * omega3.y + h3.z * omega3.z) / (h * omega);	//h X omega = |h|*|omega|*cos(gamma)  
	cgamma = fmax(cgamma, -1.0);
	cgamma = fmin(cgamma, 1.0);

	double sgamma2 = 1.0 - cgamma * cgamma;	//sin^2 + cos^2 = 1.0
	
	omega *= 2.0 * M_PI * dayUnit / (24.0 * 3600.0); 						//in 1 / s
//printf("%g %g %g %g %g %g %g %g %g %g %g %.20g %.20g %g\n", m, RR, n, omega, a, h3.x, h3.y, h3.z, omega3.x, omega3.y, omega3.z, cgamma, sgamma2, acos(cgamma));

	double s2 = sqrt(2.0);

	double dadtD, dadtS;

	//Diurnal 
	{
	double ilD = sqrt(Asteroid_rho_c[0] * Asteroid_C_c[0] * omega / Asteroid_K_c[0]);
	double ThetaD = t1 * sqrt(omega);
	double X = s2 * RR * ilD;
	double lamda = ThetaD / X;

	//double lamda = Asteroid_K_c[0] / (Asteroid_eps_c[0] * def_sigma * Ts * Ts * Ts * s2 * RR);

	double L = lamda / (1.0 + lamda);
//printf("D %g %g %g %g\n", ilD, X, lamda, L);

	double cX = cos(X);
	double sX = sin(X);

	double X2cX = (X - 2.0) * cX;
	double X2sX = (X - 2.0) * sX;
	double eX = exp(-X);
	// A B C D are multiplied by e^-X, which cancelles out later
	double Ax = -eX * (X + 2.0) - (X2cX - X * sX);
	double Bx = -eX * X - (X * cX + X2sX);
	double Cx = Ax + L * (eX * 3.0 * (X + 2.0) + (3.0 * X2cX + X * (X - 3.0) * sX));
	double Dx = Bx + L * (eX * X * (X + 3.0) - (X * (X - 3.0) * cX - 3.0 * X2sX));

	double iC2D2 = 1.0 / (Cx * Cx + Dx * Dx);

	//double Gcd = (Ax * Cx + Bx * Dx) * iC2D2;
	double Gsd = (Bx * Cx - Ax * Dx) * iC2D2;

	double WD = Gsd / (1.0 + lamda);
	dadtD = -2.0 * t2 * WD * cgamma;	// in m/s	
//printf("dadtD %g %g %g\n", t2, WD, cgamma);
	}
	
	//seasonal
	{
	double ilS = sqrt(Asteroid_rho_c[0] * Asteroid_C_c[0] * n / Asteroid_K_c[0]);
	double ThetaS = t1 * sqrt(n);
	double eta = sqrt(1.0 - e * e);
	if(e > 1.0) eta = sqrt(e * e - 1.0);



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

	//double Gcd = (Ax * Cx + Bx * Dx) * iC2D2;
	double Gsd = (Bx * Cx - Ax * Dx) * iC2D2;

	double WS = Gsd / (1.0 + lamda);
	dadtS = t2 * WS * sgamma2; // in m/s
//printf("dadtS %g %g %g\n", t2, WS, sgamma2);
	}

	double dadt = (dadtD + dadtS) / (def_AU * dayUnit) * 24.0 * 3600.0; //in AU / day * 0.017
	if(omega == 0.0) dadt = 0.0;
//printf("%g %g %g %g %g %g %g %g %g %g\n", RR, a, acos(cgamma), omega, n, dadtS / def_AU * 3600 * 24 * 365.25 * 1000000.0, dadtD / def_AU * 3600 * 24 * 365.25 * 1000000.0, cgamma, sgamma2, dadt);

	a += dadt * dt;
	
}	

__global__ void CallYarkovsky_averaged_kernel(double4 *x4_d, double4 *v4_d, double4 *spin_d, int *index_d, double2 *Msun_d, double *dt_d, const double Kt, const int N, const int Nst, const int Nstart){

	int id = blockIdx.x * blockDim.x + threadIdx.x + Nstart;

	//Compute the Kepler Elements


	if(id < N + Nstart){

		double4 x4i = x4_d[id];
		double4 v4i = v4_d[id];
		double4 spini = spin_d[id];

		int st = 0;

		if(Nst > 1) st = index_d[id] / def_MaxIndex;	//st is the sub simulation index

		if(x4i.w >= 0.0){

			double RR = v4i.w * def_AU;

			//int index = index_d[id];
			double Msun = Msun_d[st].x;
			double dt = dt_d[st] * Kt;
			double mu = def_ksq * (Msun + x4i.w);
			if(x4i.w == 0.0){
				double m = Asteroid_rho_c[0] * 4.0 / 3.0 * M_PI * RR * RR * RR;; 	//mass in kg;
				m /= 1.98855e30;						//mass im Solar masses
				mu = def_ksq * (Msun + m);
			}
			double a, e, inc, Omega, w, Theta, E;
		
//printf("K0 %d %g %g %g %g %g %g %g %g\n", id, x4i.x, x4i.y, x4i.z, x4i.w, v4i.x, v4i.y, v4i.z, v4i.w);

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
				//double M = E - e * sin(E);
			}
			else if(e > 1.0 + 1.0e-10){
				//Hyperbolic Anomaly
				//named still E instead of H or F
				E = acosh((e + t) / (1.0 + e * t));
				if(Theta < 0.0) E = - E;

				//M = e * sinh(E) - E;
			}
			else{
				//Parabolic Anomaly
				E = tan(Theta * 0.5);
				if(E > M_PI) E = E - 2.0 * M_PI;

				//M = E + E * E * E / 3.0;

				//use a to store q
				a = h * h / mu * 0.5;
			}


//printf("K1 %d %.20g %.20g %g %g %g %g %g\n", id, a, e, inc, Omega, w, E, Theta);

			if(e < 1.0){

				//modify elements
#if PVERSION == 1
				Yarkovski(a, e, x4i.w, mu, v4i.w, spini, h3, h, dt);
#endif
#if PVERSION == 2
				Yarkovski(a, e, x4i.w, mu, v4i.w, spini, h3, h, dt);
#endif
#if PVERSION == 3
				Yarkovski2(a, e, x4i.w, Msun, v4i.w, spini, h3, h, dt);
#endif
				//Convert to Cartesian Coordinates

//printf("K2 %d %.20g %.20g %g %g %g %g %g\n", id, a, e, inc, Omega, w, E, Theta);
				x4i.x = a;
				x4i.y = e;
				x4i.z = inc;
				v4i.x = Omega;
				v4i.y = w;
				v4i.z = E;
				KepToCart_E(x4i, v4i, Msun);

				x4_d[id] = x4i;
				v4_d[id] = v4i;
			}
//printf("K3 %d %g %g %g %g %g %g %g %g\n", id, x4i.x, x4i.y, x4i.z, x4i.w, v4i.x, v4i.y, v4i.z, v4i.w);
		}	
	}
}


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

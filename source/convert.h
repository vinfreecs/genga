#ifndef CONVERT_H
#define CONVERT_H

#include "define.h"


__device__ void EccentricAnomaly(double M, double e, double &E){

	if(e < 1.0 - 1.0e-10){	
		//Eccentric Anomaly
		E = M + e * 0.5;
		double Eold = E;
		for(int j = 0; j < 32; ++j){
			E = E - (E - e * sin(E) - M) / (1.0 - e * cos(E));
			if(fabs(E - Eold) < 1.0e-15) break;
			Eold = E;
		}
	}
	else if(e > 1.0 + 1.0e-10){
		//hyperbolic
		//E is assumed to be the hyperbolic eccentricity 
		E = M;
		double Eold = E;
		for(int j = 0; j < 32; ++j){
			E = E + (E - e * sinh(E) + M) / (e * cosh(E) - 1.0);
			if(fabs(E - Eold) < 1.0e-15) break;
			Eold = E;
		}

	}
	else{
		//parabolic, solve Barkers equation 
		// M = D + D^3 / 3, 
		// use cot(s) = 1.5 * M  -> s = pi / 2 - atan(1.5 * M)

		//double s = M_PI * 0.5 - atan(1.5 * M);
		E = M;
		double Eold = E;
		for(int j = 0; j < 32; ++j){
			E = E - (E + E * E * E / 3.0 - M) / (1.0 + E * E);
			if(fabs(E - Eold) < 1.0e-15) break;
			Eold = E;
		}

	}
}

// *************************************
//This is a copy of the KepToCart function in Orbit2.cu, can be removed by separated compilation
//This function converts Keplerian Elements into Cartesian Coordinates

//input a e inc Omega w M
__device__ void KepToCart_M(double4 &x, double4 &v, double Msun){

	double a = x.x;
	double e = x.y;
	double inc = x.z;
	double Omega = v.x;
	double w = v.y;
	double M = v.z;
//printf("A KtoC m:%g r:%g a:%g e:%g i:%g O:%g w:%g M:%g\n", x.w, v.w, x.x, x.y, x.z, v.x ,v.y, v.z);

	double mu = def_ksq * (Msun + x.w);

	double E;
	if(e < 1.0 - 1.0e-10){	
		//Eccentric Anomaly
		E = M + e * 0.5;
		double Eold = E;
		for(int j = 0; j < 32; ++j){
			E = E - (E - e * sin(E) - M) / (1.0 - e * cos(E));
			if(fabs(E - Eold) < 1.0e-15) break;
			Eold = E;
		}
	}
	else if(e > 1.0 + 1.0e-10){
		//hyperbolic
		//E is assumed to be the hyperbolic eccentricity 
		E = M;
		double Eold = E;
		for(int j = 0; j < 32; ++j){
			E = E + (E - e * sinh(E) + M) / (e * cosh(E) - 1.0);
			if(fabs(E - Eold) < 1.0e-15) break;
			Eold = E;
		}

	}
	else{
		//parabolic, solve Barkers equation 
		// M = D + D^3 / 3, 
		// use cot(s) = 1.5 * M  -> s = pi / 2 - atan(1.5 * M)

		//double s = M_PI * 0.5 - atan(1.5 * M);
		E = M;
		double Eold = E;
		for(int j = 0; j < 32; ++j){
			E = E - (E + E * E * E / 3.0 - M) / (1.0 + E * E);
			if(fabs(E - Eold) < 1.0e-15) break;
			Eold = E;
		}

	}


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

	double t0, t1, t2;

	if(e < 1.0 - 1.0e-10){
		//elliptic

		//double r = a * ( 1.0 - e * cE);
		//double r = a * (1.0 - e*e)/(1.0 + e *cos(Theta));
		//double t1 = r * cos(Theta); 
		//double t2 = r * sin(Theta); 
		t1 = a * (cE - e);
		t2 = a * sqrt(1.0 - e * e) * sE;
	}
	else if(e > 1.0 + 1.0e-10){
		//hyperbolic
		//double r = a * (1.0 - e*e)/(1.0 + e *cos(Theta));
		//or
		//double r = a * ( 1.0 - e * cosh(E));
		//t1 = r * cos(Theta); 
		//t2 = r * sin(Theta); 
		t1 = a * (cosh(E) - e);
		t2 = -a * sqrt(e * e - 1.0) * sinh(E);
	}
	else{
		//parabolic
		// a is assumed to be q, p = 2q, p = h^2/mu
		double Theta = 2.0 * atan(E);
		double r = 2 * a /(1.0 + cos(Theta));
		t1 = r * cos(Theta);
		t2 = r * sin(Theta);
	}

	x.x =  t1 * Px + t2 * Qx;
	x.y =  t1 * Py + t2 * Qy;
	x.z =  t1 * Pz + t2 * Qz;

	if(e < 1.0 - 1.0e-10){
		//elliptic
		t0 = 1.0 / (1.0 - e * cE) * sqrt(mu / a);
		t1 = -sE;
		t2 = sqrt(1.0 - e * e) * cE;
	}
	else if(e > 1.0 + 1.0e-10){
		//hyperbolic
		//double r = a * (1.0 - e*e)/(1.0 + e *cos(Theta));
		double r = a * ( 1.0 - e * cosh(E));
		t0 = sqrt(-mu * a) / r;
		t1 = -sinh(E);
		t2 = sqrt(e * e - 1.0) * cosh(E);
	}
	else{
		//parabolic
		double Theta = 2.0 * atan(E);
		t0 = mu / sqrt(2.0 * a * mu);
		t1 = -sin(Theta);
		t2 = 1.0 +  cos(Theta);
	}

	v.x = t0 * (t1 * Px + t2 * Qx);
	v.y = t0 * (t1 * Py + t2 * Qy);
	v.z = t0 * (t1 * Pz + t2 * Qz);
//printf("B KtoC m:%g r:%g x:%g y:%g z:%g vx:%g vy:%g vz:%g\n", x.w, v.w, x.x, x.y, x.z, v.x ,v.y, v.z);
}

// *************************************
//This is a copy of the KepToCart function in Orbit2.cu, can be removed by separated compilation
//This function converts Keplerian Elements into Cartesian Coordinates

//input a e inc Omega w E
__device__ void KepToCart_E(double4 &x, double4 &v, double Msun){

	double a = x.x;
	double e = x.y;
	double inc = x.z;
	double Omega = v.x;
	double w = v.y;
	double E = v.z;
//printf("A KtoC m:%g r:%g a:%g e:%g i:%g O:%g w:%g E:%g\n", x.w, v.w, x.x, x.y, x.z, v.x ,v.y, v.z);

	double mu = def_ksq * (Msun + x.w);

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

	double t0, t1, t2;

	if(e < 1.0 - 1.0e-10){
		//elliptic

		//double r = a * ( 1.0 - e * cE);
		//double r = a * (1.0 - e*e)/(1.0 + e *cos(Theta));
		//double t1 = r * cos(Theta); 
		//double t2 = r * sin(Theta); 
		t1 = a * (cE - e);
		t2 = a * sqrt(1.0 - e * e) * sE;
	}
	else if(e > 1.0 + 1.0e-10){
		//hyperbolic
		//double r = a * (1.0 - e*e)/(1.0 + e *cos(Theta));
		//or
		//double r = a * ( 1.0 - e * cosh(E));
		//t1 = r * cos(Theta); 
		//t2 = r * sin(Theta); 
		t1 = a * (cosh(E) - e);
		t2 = -a * sqrt(e * e - 1.0) * sinh(E);
	}
	else{
		//parabolic
		// a is assumed to be q, p = 2q, p = h^2/mu
		double Theta = 2.0 * atan(E);
		double r = 2 * a /(1.0 + cos(Theta));
		t1 = r * cos(Theta);
		t2 = r * sin(Theta);
	}

	x.x =  t1 * Px + t2 * Qx;
	x.y =  t1 * Py + t2 * Qy;
	x.z =  t1 * Pz + t2 * Qz;

	if(e < 1.0 - 1.0e-10){
		//elliptic
		t0 = 1.0 / (1.0 - e * cE) * sqrt(mu / a);
		t1 = -sE;
		t2 = sqrt(1.0 - e * e) * cE;
	}
	else if(e > 1.0 + 1.0e-10){
		//hyperbolic
		//double r = a * (1.0 - e*e)/(1.0 + e *cos(Theta));
		double r = a * ( 1.0 - e * cosh(E));
		t0 = sqrt(-mu * a) / r;
		t1 = -sinh(E);
		t2 = sqrt(e * e - 1.0) * cosh(E);
	}
	else{
		//parabolic
		double Theta = 2.0 * atan(E);
		t0 = mu / sqrt(2.0 * a * mu);
		t1 = -sin(Theta);
		t2 = 1.0 +  cos(Theta);
	}

	v.x = t0 * (t1 * Px + t2 * Qx);
	v.y = t0 * (t1 * Py + t2 * Qy);
	v.z = t0 * (t1 * Pz + t2 * Qz);
//printf("B KtoC m:%g r:%g x:%g y:%g z:%g vx:%g vy:%g vz:%g\n", x.w, v.w, x.x, x.y, x.z, v.x ,v.y, v.z);
}

__device__ void CartToKep(double4 x, double4 v, double Msun, double &a, double &e, double &inc, double &Omega, double &w, double &Theta, double &M, double &E){

	double mu = def_ksq * (Msun + x.w);

	double rsq = x.x * x.x + x.y * x.y + x.z * x.z;
	double vsq = v.x * v.x + v.y * v.y + v.z * v.z;
	double u =  x.x * v.x + x.y * v.y + x.z * v.z;
	double ir = 1.0 / sqrt(rsq);
	double ia = 2.0 * ir - vsq / mu;

	a = 1.0 / ia;

	//inclination
	double3 h3;

	h3.x = ( x.y * v.z) - (x.z * v.y);
	h3.y = (-x.x * v.z) + (x.z * v.x);
	h3.z = ( x.x * v.y) - (x.y * v.x);

	double h2 = h3.x * h3.x + h3.y * h3.y + h3.z * h3.z;
	double h = sqrt(h2);

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
	e3.x = ( v.y * h3.z - v.z * h3.y) / mu - x.x * ir;
	e3.y = (-v.x * h3.z + v.z * h3.x) / mu - x.y * ir;
	e3.z = ( v.x * h3.y - v.y * h3.x) / mu - x.z * ir;


	e = sqrt(e3.x * e3.x + e3.y * e3.y + e3.z * e3.z);

	t = (-h3.y * e3.x + h3.x * e3.y) / (n * e);
	if(t < -1.0) t = -1.0;
	if(t > 1.0) t = 1.0;
	w = acos(t);
	if(e3.z < 0.0) w = 2.0 * M_PI - w;
	if(n == 0) w = 0.0;

	//True Anomaly
	t = (e3.x * x.x + e3.y * x.y + e3.z * x.z) / e * ir;
	if(t < -1.0) t = -1.0;
	if(t > 1.0) t = 1.0;
	Theta = acos(t);

//printf("t %g %g %g %.20g %d %d\n", t, u, Theta, e, e < 1.0 - 1.0e-10, e > 1.0 + 1.0e-10); 
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
//printf("t %g %g %g %.20g %d %d\n", t, u, Theta, e, e < 1.0 - 1.0e-10, e > 1.0 + 1.0e-10); 

	//Non circular, equatorial orbit
	if(e > 1.0e-10 && inc < 1.0e-10){
		Omega = 0.0;
		w = acos(e3.x / e);
		if(e3.y < 0.0) w = 2.0 * M_PI - w;
	//printf("%.20g %.20g %.20g %.20g %d\n", w, e, e3.y, e3.x / e, w > 2.0 * M_PI);
	}

	//circular, inclinded orbit
	if(e <= 1.0e-10 && inc > 1.0e-11){
		w = 0.0;
	}

	//circular, equatorial orbit
	if(e <= 1.0e-10 && inc <= 1.0e-11){
		w = 0.0;
		Omega = 0.0;
	}

	if(w == 0 && Omega != 0.0){
		t = (-h3.y * x.x + h3.x * x.y) / n * ir;
		if(t < -1.0) t = -1.0;
		if(t > 1.0) t = 1.0;
		Theta = acos(t);
		if(x.z < 0.0){
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
		Theta = acos(x.x * ir);
		if(x.y < 0.0){
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
//printf("%g %g %g %g\n", Theta, E, M, w);
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


#endif

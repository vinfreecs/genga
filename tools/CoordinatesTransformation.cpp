#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <random>

#define def_ksq 1.0

void CartesianToKepler(double x, double y, double z, double vx, double vy, double vz, double mu, double &a, double &e, double &inc, double &Omega, double &w, double &Theta, double &E, double &M){

	double rsq = x * x + y * y + z * z;
	double vsq = vx * vx + vy * vy + vz * vz;
	double u =  x * vx + y * vy + z * vz;
	double ir = 1.0 / sqrt(rsq);
	double ia = 2.0 * ir - vsq / mu;

	a = 1.0 / ia;

	//inclination
	double hx, hy, hz;

	double h2, h, t;
	hx = ( y * vz) - (z * vy);
	hy = (-x * vz) + (z * vx);
	hz = ( x * vy) - (y * vx);

	h2 = hx * hx + hy * hy + hz * hz;
	h = sqrt(h2);

	t = hz / h;
	if(t < -1.0) t = -1.0;
	if(t > 1.0) t = 1.0;

	inc = acos(t);

	//longitude of ascending node
	double n = sqrt(hx * hx + hy * hy);
	Omega = acos(-hy / n);
	if(hx < 0.0){
		Omega = 2.0 * M_PI - Omega;
	}

	if(inc < 1.0e-10 || n == 0) Omega = 0.0;

	//argument of periapsis
	double ex, ey, ez;

	ex = ( vy * hz - vz * hy) / mu - x * ir;
	ey = (-vx * hz + vz * hx) / mu - y * ir;
	ez = ( vx * hy - vy * hx) / mu - z * ir;


	e = sqrt(ex * ex + ey * ey + ez * ez);

	t = (-hy * ex + hx * ey) / (n * e);
	if(t < -1.0) t = -1.0;
	if(t > 1.0) t = 1.0;
	w = acos(t);
	if(ez < 0.0) w = 2.0 * M_PI - w;
	if(n == 0) w = 0.0;

	//True Anomaly
	t = (ex * x + ey * y + ez * z) / e * ir;
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
		w = acos(ex / e);
		if(ey < 0.0) w = 2.0 * M_PI - w;
//printf("%.20g %.20g %.20g %.20g %d\n", w, e, ey, ex / e, w > 2.0 * M_PI);
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
		t = (-hy * x + hx * y) / n * ir;
		if(t < -1.0) t = -1.0;
		if(t > 1.0) t = 1.0;
		Theta = acos(t);
		if(z < 0.0){
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
		Theta = acos(x * ir);
		if(y < 0.0){
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

void KeplerToCartesian(double &x, double &y, double &z, double &vx, double &vy, double &vz, double mu, double a, double e, double inc, double Omega, double w, double M){


	double E;
	double Theta;
	//Eccentric Anomaly
	if(e < 1.0 - 1.0e-10){
		//elliptic
		E = M + e * 0.5;
		double Eold = E;
		for(int j = 0; j < 32; ++j){
			E = E - (E - e * sin(E) - M) / (1.0 - e * cos(E));
			if(fabs(E - Eold) < 1.0e-15) break;
			Eold = E;
		}
		//True Anonamly
		double tx = sqrt(1.0 + e) * sin(E * 0.5);
		double ty = sqrt(1.0 - e) * cos(E * 0.5);
		Theta = 2.0 * atan2(tx, ty);

		//Circular orbit
		if(e < 1.0e-10){
			Theta = E;
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

		double tx = sqrt(e + 1.0) * sinh(E * 0.5);
		double ty = sqrt(e - 1.0) * cosh(E * 0.5);
		Theta = 2.0 * atan2(tx, ty);
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

		Theta = 2.0 * atan(E);
	}


//printf("%g %g %g\n", M, E, Theta);

	double cw = cos(w);
	double sw = sin(w);
	double cOmega = cos(Omega);
	double sOmega = sin(Omega);
	double ci = cos(inc);
	double si = sin(inc);

	double Px, Py, Pz;
	Px = cw * cOmega - sw * ci * sOmega;
	Py = cw * sOmega + sw * ci * cOmega;
	Pz = sw * si;

	double Qx, Qy, Qz;
	Qx = -sw * cOmega - cw * ci * sOmega;
	Qy = -sw * sOmega + cw * ci * cOmega;
	Qz = cw * si;

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
		double r = 2 * a /(1.0 + cos(Theta));
		t1 = r * cos(Theta); 
		t2 = r * sin(Theta); 

//printf("r  %g %g %g %g\n", r, t1, t2, cos(Theta) );
	}


	x =  t1 * Px + t2 * Qx;
	y =  t1 * Py + t2 * Qy;
	z =  t1 * Pz + t2 * Qz;

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
		t0 = mu / sqrt(2.0 * a * mu);
		t1 = -sin(Theta);
		t2 = 1.0 +  cos(Theta);
	}


	vx = t0 * (t1 * Px + t2 * Qx);
	vy = t0 * (t1 * Py + t2 * Qy);
	vz = t0 * (t1 * Pz + t2 * Qz);
}


int main(int argc, char*argv[]){

	double Msun = 1.0;

	double x, y, z, vx, vy, vz;
	double a, e, inc, Omega, w, M;
	double Theta, E;

	std::random_device rd;
	std::default_random_engine eng(rd());

	std::uniform_real_distribution<double> distr(0.0, 1.0);

	for(int i = 0; i < 1000; ++i){
		double m = 1.0e-5;

		//For parabolic orbits use a = q = rx
		a = 0.05;//distr(eng) * 1.5 + 0.5;

		e = 1.0; //distr(eng) * 0.99;

		inc = 0.1;//distr(eng) * 1.0 * M_PI;

		Omega = 0.0;//distr(eng) * 2.0 * M_PI;

		w = 0.0;//distr(eng) * 2.0 * M_PI;

		//double E = distr(eng) * 2.0 * M_PI;
		M = i / 500.0 * 2.0 * M_PI - 2.0 * M_PI;

		//printf("%g\n", M);

		double mu = def_ksq * (Msun + m);

		double a0 = a;
		double e0 = e;
		double inc0 = inc;
		double Omega0 = Omega;
		double w0 = w;
		double M0 = M;

		KeplerToCartesian(x, y, z, vx, vy, vz, Msun + m, a, e, inc, Omega, w, M);

		printf("%g %g %g %g %g %g\n", x, y, z, vx, vy, vz);

		//CartesianToKepler(x, y, z, vx, vy, vz, Msun + m, a, e, inc, Omega, w, Theta, E, M);
		//printf("%g %g %g %g %g %g\n", a, e, inc, Omega, w, M);
		//printf("%g %g %g %g %g %g\n", a-a0, e-e0, inc-inc0, Omega-Omega0, w-w0, M-M0);
		

		/*
		if(i == 0){	
			x = 0.00499753;
			y = 0.000222699;
			z = 0.0;
			vx = -0.44407;
			vy = 20.0401;
			vz = 0.0;	
		}
		
		if(i == 1){
			x = 0.00499753;
			y = -0.000222699;
			z = 0.0;
			vx = 0.44407;
			vy = 20.0401;
			vz = 0.0;	
		}

		printf("%g %g %g %g %g %g\n", x, y, z, vx, vy, vz);

		CartesianToKepler(x, y, z, vx, vy, vz, Msun + m, a, e, inc, Omega, w, Theta, E, M);
		printf("%g %g %g %g %g %.20g | %g %g\n", a, e, inc, Omega, w, E, Theta, M);

		KeplerToCartesian(x, y, z, vx, vy, vz, Msun + m, a, e, inc, Omega, w, M);
		printf("%g %g %g %g %g %g\n", x, y, z, vx, vy, vz);
		*/
	}


}

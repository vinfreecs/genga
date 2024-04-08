#include "define.h"

//This function is needed for the pseudovelocity conversion
//It is the right hand side of equation 32 from Saha & Tremaine 1994
//vv is pseudovelocity
void FPseudoV(double mu, double x, double y, double z, double vvx, double vvy, double vvz, double &fx, double &fy, double &fz){

	double c2 = def_cm * def_cm;

	double vsq = vvx * vvx + vvy * vvy + vvz * vvz;
	double rsq = x * x + y * y + z * z;
	double r = sqrt(rsq);

	double t = 1.0 - 1.0/c2 * (vsq * 0.5 + 3.0 * mu / r);

	fx = vvx * t;
	fy = vvy * t;
	fz = vvz * t;
}


//This function converts pseudovelocities to true velocities
//See Saha & Tremaine 1994
void convertPseudovToV(double4 *x4_h, double4 *v4_h, double Msun, int N){

	#pragma omp parallel for schedule(static)
	for(int id = 0; id < N; ++id){

		double c2 = def_cm * def_cm;

		double mu = def_ksq * (Msun + x4_h[id].w);
		//use here Jacoby masses from Saha Tremaine

		double vsq = v4_h[id].x * v4_h[id].x + v4_h[id].y * v4_h[id].y + v4_h[id].z * v4_h[id].z;
		double rsq = x4_h[id].x * x4_h[id].x + x4_h[id].y * x4_h[id].y + x4_h[id].z * x4_h[id].z;
		double r = sqrt(rsq);

		double t = 1.0 - 1.0/c2 * (vsq * 0.5 + 3.0 * mu / r);

		//printf("%d %.20g %.20g %.20g | %.20g %.20g %.20g\n", i, vx[i], vy[i], vz[i], vx[i] * t, vy[i] * t, vz[i] * t);

		v4_h[id].x *= t;
		v4_h[id].y *= t;
		v4_h[id].z *= t;
	}
}


//This function converts velocities to pseudovelocities
//See Saha & Tremaine 1994
void convertVToPseidov(double4 *x4_h, double4 *v4_h, int *ErrorFlag_m, double Msun, int N){

	
	#pragma omp parallel for schedule(static)
	for(int id = 0; id < N; ++id){

		double mu = def_ksq * (Msun + x4_h[id].w);
		//use here Jacoby masses from Saha Tremaine

		double xi = x4_h[id].x;
		double yi = x4_h[id].y;
		double zi = x4_h[id].z;

		double vxi = v4_h[id].x;
		double vyi = v4_h[id].y;
		double vzi = v4_h[id].z;

		//first guess of pseudovelocity
		double vvx0 = vxi;
		double vvy0 = vyi;
		double vvz0 = vzi;

		//second guess of pseudovelocity
		double vvx1 = vvx0 * 0.01;
		double vvy1 = vvy0 * 0.01;
		double vvz1 = vvz0 * 0.01;

		double fx0;
		double fy0;
		double fz0;

		FPseudoV(mu, xi, yi, zi, vvx0, vvy0, vvz0, fx0, fy0, fz0);
		fx0 -= vxi;
		fy0 -= vyi;
		fz0 -= vzi;

		double fx1;
		double fy1;
		double fz1;

		FPseudoV(mu, xi, yi, zi, vvx1, vvy1, vvz1, fx1, fy1, fz1);
		fx1 -= vxi;
		fy1 -= vyi;
		fz1 -= vzi;

		//Newton Method
		int k;
		for(k = 0; k < 30; ++k){

			double tx = vvx1 - (vvx1 - vvx0) / (fx1 - fx0) * fx1;
			double ty = vvy1 - (vvy1 - vvy0) / (fy1 - fy0) * fy1;
			double tz = vvz1 - (vvz1 - vvz0) / (fz1 - fz0) * fz1;

			int Stop = 0;
			if(fabs(fx1 - fx0) < 1.0e-18){
				tx = vvx1;
				++Stop;
			}
			if(fabs(fy1 - fy0) < 1.0e-18){
				ty = vvy1;
				++Stop;
			}
			if(fabs(fz1 - fz0) < 1.0e-18){
				tz = vvz1;
				++Stop;
			}

			vvx0 = vvx1;
			vvy0 = vvy1;
			vvz0 = vvz1;

			fx0 = fx1;
			fy0 = fy1;
			fz0 = fz1;

			vvx1 = tx;
			vvy1 = ty;
			vvz1 = tz;

			if(Stop == 3){
				break;
			}
			FPseudoV(mu, xi, yi, zi, vvx1, vvy1, vvz1, fx1, fy1, fz1);
			fx1 -= vxi;
			fy1 -= vyi;
			fz1 -= vzi;
//if(k > 4) printf("%d %d %.20g %.20g %.20g | %.20g %.20g %.20g | %g %g %g\n", id, k, vxi, vyi, vzi, vvx1, vvy1, vvz1, fx1, fy1, fz1);

		}

		if(k >= 29){
			ErrorFlag_m[0] = 1;
			printf("Warning: Newton Method in 'convertVToPseidov' did not convert. %d\n", id);
		}

		v4_h[id].x = vvx1;
		v4_h[id].y = vvy1;
		v4_h[id].z = vvz1;
	}
}


//First call f = 1;
//Second call f = -1;
//serial version
void Data::HCCall_1(const double Ct, const int f){

	int N = N_h[0] + Nsmall_h[0];

	if(P.UseGR == 1 && f == 1){
		convertVToPseidov(x4_h, v4_h, ErrorFlag_m, Msun_h[0].x, N);
	}
	//HC

	double dt = dt_h[0] * Ct;
	double dtiMsun = dt / Msun_h[0].x;

	double ax = 0.0;
	double ay = 0.0;
	double az = 0.0;

	for(int i = 0; i < N; ++i){
		double m = x4_h[i].w;
		if(m > 0.0){
			ax += m * v4_h[i].x;
			ay += m * v4_h[i].y;
			az += m * v4_h[i].z;
//printf("%d %d %.20g | %.20g\n", i, k, ax, v4_h[i].x * m );
		}
	}
	ax *= dtiMsun;
	ay *= dtiMsun;
	az *= dtiMsun;
//printf("%.20g %.20g %.20g | %.20g %.20g %.20g\n", ax, ay, az, v4_h[0].x, v4_h[0].y, v4_h[0].z);

	for(int i = 0; i < N; ++i){
		x4_h[i].x += ax;
		x4_h[i].y += ay;
		x4_h[i].z += az;
	}
	if(P.UseGR == 1){
		double c2 = def_cm * def_cm;
		for(int i = 0; i < N; ++i){
			double vsq = v4_h[i].x * v4_h[i].x + v4_h[i].y * v4_h[i].y + v4_h[i].z * v4_h[i].z;
			double vcdt = 2.0 * vsq / c2 * dt;
			x4_h[i].x -= (v4_h[i].x * vcdt);
			x4_h[i].y -= (v4_h[i].y * vcdt);
			x4_h[i].z -= (v4_h[i].z * vcdt);
		}
	}


	if(P.UseGR == 1 && f == -1){
		convertPseudovToV(x4_h, v4_h, Msun_h[0].x, N);
	}

}
//First call f = 1;
//Second call f = -1;
//parallel version
void Data::HCCall(const double Ct, const int f){

	int N = N_h[0] + Nsmall_h[0];

	if(P.UseGR == 1 && f == 1){
		convertVToPseidov(x4_h, v4_h, ErrorFlag_m, Msun_h[0].x, N);
	}
	//HC

	double dt = dt_h[0] * Ct;
	double dtiMsun = dt / Msun_h[0].x;

	double ax[Nomp];
	double ay[Nomp];
	double az[Nomp];

	for(int k = 0; k < Nomp; ++k){
		ax[k] = 0.0;
		ay[k] = 0.0;
		az[k] = 0.0;
	}
	#pragma omp parallel for schedule(static) if(HCX > 0)
	for(int i = 0; i < N; ++i){
		double m = x4_h[i].w;
		if(m > 0.0){
			int k = omp_get_thread_num();
			ax[k] += m * v4_h[i].x;
			ay[k] += m * v4_h[i].y;
			az[k] += m * v4_h[i].z;
//printf("%d %d %.20g | %.20g\n", i, k, ax[k], v4_h[i].x * m );
		}
	}
	for(int k = 1; k < Nomp; ++k){
		ax[0] += ax[k];
		ay[0] += ay[k];
		az[0] += az[k];

	}
	ax[0] *= dtiMsun;
	ay[0] *= dtiMsun;
	az[0] *= dtiMsun;
//printf("%.20g %.20g %.20g | %.20g %.20g %.20g\n", ax[0], ay[0], az[0], v4_h[0].x, v4_h[0].y, v4_h[0].z);

	#pragma omp parallel for schedule(static) if(HCX > 0)
	for(int i = 0; i < N; ++i){
		x4_h[i].x += ax[0];
		x4_h[i].y += ay[0];
		x4_h[i].z += az[0];
	}
	if(P.UseGR == 1){
		double c2 = def_cm * def_cm;
		#pragma omp parallel for schedule(static) if(HCX > 0)
		for(int i = 0; i < N; ++i){
			double vsq = v4_h[i].x * v4_h[i].x + v4_h[i].y * v4_h[i].y + v4_h[i].z * v4_h[i].z;
			double vcdt = 2.0 * vsq / c2 * dt;
			x4_h[i].x -= (v4_h[i].x * vcdt);
			x4_h[i].y -= (v4_h[i].y * vcdt);
			x4_h[i].z -= (v4_h[i].z * vcdt);
		}
	}


	if(P.UseGR == 1 && f == -1){
		convertPseudovToV(x4_h, v4_h, Msun_h[0].x, N);
	}

}


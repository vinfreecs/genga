#ifndef BSSINGLE_H
#define BSSINGLE_H


#include "directAcc.h"

// **************************************
//This function  intergrates one body using a Bulirsh Stoer method.
//The implementation is based on the mercury code from Chambers.
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
// ****************************************
__device__ void BSSinglestep(double4 &x4, double4 &v4, const double Msun, const double dt, int id){

	double3 a;
	double3 a0;
	double4 xp;
	double4 vp;
	double4 xt;
	double4 vt;

	double3 dx[8]; 
	double3 dv[8];

	double errorx;
	double errorv;
	double3 scalex;
	double3 scalev;

	double t0;
	double t1;
	double t2;

	double dt1 =  dt;
	double dt2;
	double dt22;
	double error = 10.0;
	volatile int f;
	double t = 0.0;

	const double ksqMsun = def_ksq * Msun;
//printf("BSSingle %d %g %g %g %g %g %g %g %g\n", id, x4.w, v4.w, x4.x, x4.y, x4.z, v4.x, v4.y, v4.z);

	xt.x = 0.0;
	xt.y = 0.0;
	xt.z = 0.0;

	vt.x = 0.0;
	vt.y = 0.0;
	vt.z = 0.0;

	int sgnt = 1;
	if(dt < 0) sgnt = -1;

	int mm = -1;

	while(sgnt * t < sgnt * dt && mm < 1e8){
		++mm;

		scalex.x = 1.0 / (x4.x * x4.x + 1.0e-50);
		scalex.y = 1.0 / (x4.y * x4.y + 1.0e-50);
		scalex.z = 1.0 / (x4.z * x4.z + 1.0e-50);

		scalev.x = 1.0 / (v4.x * v4.x + 1.0e-50);
		scalev.y = 1.0 / (v4.y * v4.y + 1.0e-50);
		scalev.z = 1.0 / (v4.z * v4.z + 1.0e-50);

		a0.x = 0.0;
		a0.y = 0.0;
		a0.z = 0.0;

		accEncSun(x4, a0, ksqMsun);

		f = 1;
		int fc = -1;
		while(f == 1 && fc < 1e3){

			++fc;
			for(int n = 1; n <= 8; ++n){
				dt2 = dt1 / (2.0 * n);	//
				dt22 = dt2 * 2.0;

				xt.w = x4.w;
				vt.w = v4.w;
				xp.w = x4.w;
				vp.w = v4.w;

				xp.x = x4.x + dt2 * v4.x;
				xp.y = x4.y + dt2 * v4.y;
				xp.z = x4.z + dt2 * v4.z;

				vp.x = v4.x + dt2 * a0.x;
				vp.y = v4.y + dt2 * a0.y;
				vp.z = v4.z + dt2 * a0.z;

				a.x = 0.0;
				a.y = 0.0;
				a.z = 0.0;

				accEncSun(xp, a, ksqMsun);
				xt.x = x4.x + dt22 * vp.x;
				xt.y = x4.y + dt22 * vp.y;
				xt.z = x4.z + dt22 * vp.z;	
			
				vt.x = v4.x + dt22 * a.x;
				vt.y = v4.y + dt22 * a.y;
				vt.z = v4.z + dt22 * a.z;

				for(int m = 2; m <= n; ++m){
					a.x = 0.0;
					a.y = 0.0;
					a.z = 0.0;

					accEncSun(xt, a, ksqMsun);

					xp.x = xp.x + dt22 * vt.x;
					xp.y = xp.y + dt22 * vt.y;
					xp.z = xp.z + dt22 * vt.z;

					vp.x = vp.x + dt22 * a.x;
					vp.y = vp.y + dt22 * a.y;
					vp.z = vp.z + dt22 * a.z;

					a.x = 0.0;
					a.y = 0.0;
					a.z = 0.0;

					accEncSun(xp, a, ksqMsun);

					xt.x = xt.x + dt22 * vp.x;
					xt.y = xt.y + dt22 * vp.y;
					xt.z = xt.z + dt22 * vp.z;

					vt.x = vt.x + dt22 * a.x;
					vt.y = vt.y + dt22 * a.y;
					vt.z = vt.z + dt22 * a.z;
				}
				a.x = 0.0;
				a.y = 0.0;
				a.z = 0.0;
				accEncSun(xt, a, ksqMsun);

				dx[n-1].x = 0.5 * (xt.x + xp.x + dt2 * vt.x);
				dx[n-1].y = 0.5 * (xt.y + xp.y + dt2 * vt.y);
				dx[n-1].z = 0.5 * (xt.z + xp.z + dt2 * vt.z);

				dv[n-1].x = 0.5 * (vt.x + vp.x + dt2 * a.x);
				dv[n-1].y = 0.5 * (vt.y + vp.y + dt2 * a.y);
				dv[n-1].z = 0.5 * (vt.z + vp.z + dt2 * a.z);
				for(int j = n-1; j >=1; --j){
					t0 = BSt0_c[(n-1) * 8 + (j-1)];
					t1 = t0 * BSddt_c[j];
					t2 = t0 * BSddt_c[n-1];

					dx[j-1].x = t1 * dx[j].x - t2 * dx[j-1].x;
					dx[j-1].y = t1 * dx[j].y - t2 * dx[j-1].y;
					dx[j-1].z = t1 * dx[j].z - t2 * dx[j-1].z;

					dv[j-1].x = t1 * dv[j].x - t2 * dv[j-1].x;
					dv[j-1].y = t1 * dv[j].y - t2 * dv[j-1].y;
					dv[j-1].z = t1 * dv[j].z - t2 * dv[j-1].z;
				}

				if(n > 3){  


					error = 0.0;
					errorx = dx[0].x * dx[0].x * scalex.x;
					errorv = dv[0].x * dv[0].x * scalev.x;

					errorx = fmax(errorx, dx[0].y * dx[0].y * scalex.y);
					errorx = fmax(errorx, dx[0].z * dx[0].z * scalex.z);

					errorv = fmax(errorv, dv[0].y * dv[0].y * scalev.y);
					errorv = fmax(errorv, dv[0].z * dv[0].z * scalev.z);

					error = fmax(errorx, errorv);	

					if(error <= def_tol * def_tol || sgnt * dt1 < def_dtmin){

						xt.x = dx[0].x; 
						xt.y = dx[0].y;
						xt.z = dx[0].z;

						vt.x = dv[0].x;
						vt.y = dv[0].y;
						vt.z = dv[0].z;

						for(int j = 1; j < n; ++j){ 
							xt.x += dx[j].x;
							xt.y += dx[j].y;
							xt.z += dx[j].z;

							vt.x += dv[j].x;
							vt.y += dv[j].y;
							vt.z += dv[j].z;
						}

						t += dt1;

						if(n >= 8) dt1 *= 0.55;
						if(n < 7) dt1 *= 1.3;
						if(sgnt * dt1 > sgnt * dt) dt1 = dt;
						if(sgnt * (t + dt1) > sgnt *dt) dt1 = dt - t;
						if(sgnt * dt1 < def_dtmin) dt1 = sgnt * def_dtmin;
							
						x4 = xt;
						v4 = vt;

						f = 0;
						break;
					}
				}
			}
			if(f ==1) dt1 *= 0.5;
		}
	}

}
#endif

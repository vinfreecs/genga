#ifndef BSBSMALL_H
#define BSBSMALL_H

#include "define.h"
#include "directAcc.h"
#include "Encounter3.h"
#endif

// **************************************
//This Kernel intergrates the independent groups of close encunters for a time step
//using a Bulirsh Stoer method with nb threds. Where n is the minimum of n^2 and 256
//The implementation of the Bulirsh Stoer method is based on the mercury code from Chambers.
//  ****************************************
template< int NN, int nb>
__global__ void BSBStepsmall_kernel(double4 *x4small_d, double4 *v4small_d, double4 *xold_d, double4 *vold_d, double4 *xoldsmall_d, double4 *voldsmall_d, double *rcrit_d, double *rcritv_d, int2 *Encpairssmall_d, int2 *Encpairssmall2_d, double dt, double Msun, double *U_d, int st, int *index_d, int *indexsmall_d, int *Ncoll_d, double *Coll_d, double time, double3 *spin_d, double3 *spinsmall_d, const int Nconst){
	int idy = threadIdx.x;
	int idx = blockIdx.x;

	int ii = idy / nb;
	int jj = idy % nb;
	double dt1 = dt; 
	double dt2, dt22;
	double t = 0.0;

	const double tol = 1.0e-12;

	__shared__ double4 x4_s[NN];
	__shared__ double4 v4_s[NN];
	__shared__ double rcritv_s[NN];
	__shared__ double3 a0_s[nb * NN +NN];
	__shared__ double3 a_s[nb * NN + NN];
	__shared__ double4 xp_s[NN];
	__shared__ double4 vp_s[NN];
	__shared__ double4 xt_s[NN];
	__shared__ double4 vt_s[NN];

	__shared__ double3 dx_s[NN][8];
	__shared__ double3 dv_s[NN][8];

	__shared__ int Ncol[1];
	__shared__ int2 Colpairs_s[MaxColl];
	__shared__ int N2; 
	__shared__ int sgnt;

	double3 scalex;
	double3 scalev;

	__shared__ double error_s[2*NN];
	double test;
	int idi;
	int si = Encpairssmall2_d[Nconst * idx + st].y; 
	N2 = Encpairssmall_d[si * Nconst].y;
	if(idy < N2){
		idi = Encpairssmall_d[si * Nconst + idy].x;
//printf("BS2 %d %d %d \n", idx, st, idi);
	}
	else idi = 0;

        if(dt < 0.0){
                sgnt = -1;
        }
        else sgnt = 1;

 	__syncthreads();


	if(idy == 0){
                x4_s[idy] = xoldsmall_d[idi];  
                v4_s[idy] = voldsmall_d[idi];  
                rcritv_s[idy] = 0.0;
//printf("BS2 %d %d %d %d\n", idx, st, idi, indexsmall_d[idi]);
	}
	else if(idy< N2){
                x4_s[idy] = xold_d[idi];  
                v4_s[idy] = vold_d[idi];  
		rcritv_s[idy] = rcritv_d[idi];
//printf("BS2 %d %d %d %d\n", idx, st, idi, index_d[idi]);
	}
	else if(idy < NN){
		x4_s[idy].x = 0.0;
		x4_s[idy].y = 0.0;
		x4_s[idy].z = 0.0;
		x4_s[idy].w = 0.0;
                v4_s[idy].x = 0.0;
                v4_s[idy].y = 0.0;
                v4_s[idy].z = 0.0;
                v4_s[idy].w = 0.0;
		rcritv_s[idy] = 0.0;
	}
	if(idy < NN){
		a0_s[idy + nb*NN].x = 0.0;
		a0_s[idy + nb*NN].y = 0.0;
		a0_s[idy + nb*NN].z = 0.0;

		a_s[idy + nb*NN].x = 0.0;
                a_s[idy + nb*NN].y = 0.0;
                a_s[idy + nb*NN].z = 0.0;
	}
	if(idy < MaxColl){
		Colpairs_s[idy].x = 0;
		Colpairs_s[idy].y = 0;
	}
	if(idy < 2 * NN) error_s[idy] = 0.0;
	__syncthreads();

	for(int tt = 0; tt < 1000; ++tt){
	__syncthreads();

		if(idy < N2){

			scalex.x = 1.0 / (x4_s[idy].x * x4_s[idy].x + 1.0e-20);
			scalex.y = 1.0 / (x4_s[idy].y * x4_s[idy].y + 1.0e-20);
			scalex.z = 1.0 / (x4_s[idy].z * x4_s[idy].z + 1.0e-20);

			scalev.x = 1.0 / (v4_s[idy].x * v4_s[idy].x + 1.0e-20);
			scalev.y = 1.0 / (v4_s[idy].y * v4_s[idy].y + 1.0e-20);
			scalev.z = 1.0 / (v4_s[idy].z * v4_s[idy].z + 1.0e-20);

		}

		a0_s[idy].x = 0.0;
        	a0_s[idy].y = 0.0;
        	a0_s[idy].z = 0.0;
		__syncthreads();
		for(int l = 0; l < NN; l += nb){
			accEnc(x4_s[ii], x4_s[jj + l], a0_s[idy], rcritv_s[ii], rcritv_s[jj + l], test, ii, jj + l);
		}
		__syncthreads();
		{
			volatile double3 *a = a0_s;
			if(nb >= 16) a[idy].x += a[idy + 8].x;
			if(nb >= 8) a[idy].x += a[idy + 4].x;
			if(nb >= 4) a[idy].x += a[idy + 2].x;
			if(nb >= 2) a[idy].x += a[idy + 1].x;

			if(nb >= 16) a[idy].y += a[idy + 8].y;
			if(nb >= 8) a[idy].y += a[idy + 4].y;
			if(nb >= 4) a[idy].y += a[idy + 2].y;
			if(nb >= 2) a[idy].y += a[idy + 1].y;

			if(nb >= 16) a[idy].z += a[idy + 8].z;
			if(nb >= 8) a[idy].z += a[idy + 4].z;
			if(nb >= 4) a[idy].z += a[idy + 2].z;
			if(nb >= 2) a[idy].z += a[idy + 1].z;
		}
		__syncthreads();

		if(idy < N2){
			accEncSun(x4_s[idy], a0_s[idy * nb], ksq * Msun);
		}

		volatile int f = 1;
		__syncthreads();
		for(int ff = 0; ff < 1e6; ++ff){
			__syncthreads();
			for(int n = 1; n <= 8; ++n){
				dt2 = dt1 / (2.0 * n);
				dt22 = dt2 * 2.0;

				if(idy < NN){
					xp_s[idy].x = x4_s[idy].x + dt2 * v4_s[idy].x;
					xp_s[idy].y = x4_s[idy].y + dt2 * v4_s[idy].y;
					xp_s[idy].z = x4_s[idy].z + dt2 * v4_s[idy].z;
					xp_s[idy].w = x4_s[idy].w;

					vp_s[idy].x = v4_s[idy].x + dt2 * a0_s[idy * nb].x;  
					vp_s[idy].y = v4_s[idy].y + dt2 * a0_s[idy * nb].y;  
					vp_s[idy].z = v4_s[idy].z + dt2 * a0_s[idy * nb].z; 
					vp_s[idy].w = v4_s[idy].w;
				}

				a_s[idy].x = 0.0;
				a_s[idy].y = 0.0;
				a_s[idy].z = 0.0;

				__syncthreads();
				for(int l = 0; l < NN; l += nb){
					accEnc(xp_s[ii], xp_s[jj + l], a_s[idy], rcritv_s[ii], rcritv_s[jj + l], test, ii, jj + l);
				}
				__syncthreads();
				{
					volatile double3 *a = a_s;
					if(nb >= 16) a[idy].x += a[idy + 8].x;
					if(nb >= 8) a[idy].x += a[idy + 4].x;
					if(nb >= 4) a[idy].x += a[idy + 2].x;
					if(nb >= 2) a[idy].x += a[idy + 1].x;

					if(nb >= 16) a[idy].y += a[idy + 8].y;
					if(nb >= 8) a[idy].y += a[idy + 4].y;
					if(nb >= 4) a[idy].y += a[idy + 2].y;
					if(nb >= 2) a[idy].y += a[idy + 1].y;

					if(nb >= 16) a[idy].z += a[idy + 8].z;
					if(nb >= 8) a[idy].z += a[idy + 4].z;
					if(nb >= 4) a[idy].z += a[idy + 2].z;
					if(nb >= 2) a[idy].z += a[idy + 1].z;
				}
				__syncthreads();
				if(idy < NN){
					accEncSun(xp_s[idy], a_s[idy * nb], ksq * Msun);

					xt_s[idy].x = x4_s[idy].x + dt22 * vp_s[idy].x;
					xt_s[idy].y = x4_s[idy].y + dt22 * vp_s[idy].y;
					xt_s[idy].z = x4_s[idy].z + dt22 * vp_s[idy].z;
					xt_s[idy].w = x4_s[idy].w;

					vt_s[idy].x = v4_s[idy].x + dt22 * a_s[idy * nb].x;
					vt_s[idy].y = v4_s[idy].y + dt22 * a_s[idy * nb].y;
					vt_s[idy].z = v4_s[idy].z + dt22 * a_s[idy * nb].z;
					vt_s[idy].w = v4_s[idy].w;
				}
				__syncthreads();
				
				for(int m = 2; m <= n; ++m){
					a_s[idy].x = 0.0;
					a_s[idy].y = 0.0;
					a_s[idy].z = 0.0;

					__syncthreads();
					for(int l = 0; l < NN; l += nb){
						accEnc(xt_s[ii], xt_s[jj + l], a_s[idy], rcritv_s[ii], rcritv_s[jj + l], test, ii, jj + l);
					}
					__syncthreads();
					{
						volatile double3 *a = a_s;
						if(nb >= 16) a[idy].x += a[idy + 8].x;
						if(nb >= 8) a[idy].x += a[idy + 4].x;
						if(nb >= 4) a[idy].x += a[idy + 2].x;
						if(nb >= 2) a[idy].x += a[idy + 1].x;

						if(nb >= 16) a[idy].y += a[idy + 8].y;
						if(nb >= 8) a[idy].y += a[idy + 4].y;
						if(nb >= 4) a[idy].y += a[idy + 2].y;
						if(nb >= 2) a[idy].y += a[idy + 1].y;

						if(nb >= 16) a[idy].z += a[idy + 8].z;
						if(nb >= 8) a[idy].z += a[idy + 4].z;
						if(nb >= 4) a[idy].z += a[idy + 2].z;
						if(nb >= 2) a[idy].z += a[idy + 1].z; 
					}
					__syncthreads();

					if(idy < N2){
						accEncSun(xt_s[idy], a_s[idy * nb], ksq * Msun);

						xp_s[idy].x += dt22 * vt_s[idy].x;
						xp_s[idy].y += dt22 * vt_s[idy].y;
						xp_s[idy].z += dt22 * vt_s[idy].z;

						vp_s[idy].x += dt22 * a_s[idy * nb].x;
						vp_s[idy].y += dt22 * a_s[idy * nb].y;
						vp_s[idy].z += dt22 * a_s[idy * nb].z;
					}
					__syncthreads();
					a_s[idy].x = 0.0;
					a_s[idy].y = 0.0;
					a_s[idy].z = 0.0;

					__syncthreads();
					for(int l = 0; l < NN; l += nb){
						accEnc(xp_s[ii], xp_s[jj + l], a_s[idy], rcritv_s[ii], rcritv_s[jj + l], test, ii, jj + l);
					}
					__syncthreads();
					{
						volatile double3 *a = a_s;
						if(nb >= 16) a[idy].x += a[idy + 8].x;
						if(nb >= 8) a[idy].x += a[idy + 4].x;
						if(nb >= 4) a[idy].x += a[idy + 2].x;
						if(nb >= 2) a[idy].x += a[idy + 1].x;

						if(nb >= 16) a[idy].y += a[idy + 8].y;
						if(nb >= 8) a[idy].y += a[idy + 4].y;
						if(nb >= 4) a[idy].y += a[idy + 2].y;
						if(nb >= 2) a[idy].y += a[idy + 1].y;

						if(nb >= 16) a[idy].z += a[idy + 8].z;
						if(nb >= 8) a[idy].z += a[idy + 4].z;
						if(nb >= 4) a[idy].z += a[idy + 2].z;
						if(nb >= 2) a[idy].z += a[idy + 1].z;
					}
					__syncthreads();

					if(idy < N2){
						accEncSun(xp_s[idy], a_s[idy * nb], ksq * Msun);

						xt_s[idy].x += dt22 * vp_s[idy].x;
						xt_s[idy].y += dt22 * vp_s[idy].y;
						xt_s[idy].z += dt22 * vp_s[idy].z;

						vt_s[idy].x += dt22 * a_s[idy * nb].x;
						vt_s[idy].y += dt22 * a_s[idy * nb].y;
						vt_s[idy].z += dt22 * a_s[idy * nb].z;
					}
					__syncthreads();
				}
				a_s[idy].x = 0.0;
				a_s[idy].y = 0.0;
				a_s[idy].z = 0.0;

				__syncthreads();
				for(int l = 0; l < NN; l += nb){
					accEnc(xt_s[ii], xt_s[jj + l], a_s[idy], rcritv_s[ii], rcritv_s[jj + l], test, ii, jj + l);
				}
				__syncthreads();
				{
					volatile double3 *a = a_s;
					if(nb >= 16) a[idy].x += a[idy + 8].x;
					if(nb >= 8) a[idy].x += a[idy + 4].x;
					if(nb >= 4) a[idy].x += a[idy + 2].x;
					if(nb >= 2) a[idy].x += a[idy + 1].x;

					if(nb >= 16) a[idy].y += a[idy + 8].y;
					if(nb >= 8) a[idy].y += a[idy + 4].y;
					if(nb >= 4) a[idy].y += a[idy + 2].y;
					if(nb >= 2) a[idy].y += a[idy + 1].y;

					if(nb >= 16) a[idy].z += a[idy + 8].z;
					if(nb >= 8) a[idy].z += a[idy + 4].z;
					if(nb >= 4) a[idy].z += a[idy + 2].z;
					if(nb >= 2) a[idy].z += a[idy + 1].z; 
				}
				__syncthreads();
				if(idy < N2){
					accEncSun(xt_s[idy], a_s[idy * nb], ksq * Msun);

					dx_s[idy][n-1].x = 0.5 * (xt_s[idy].x + xp_s[idy].x + dt2 * vt_s[idy].x);
					dx_s[idy][n-1].y = 0.5 * (xt_s[idy].y + xp_s[idy].y + dt2 * vt_s[idy].y);
					dx_s[idy][n-1].z = 0.5 * (xt_s[idy].z + xp_s[idy].z + dt2 * vt_s[idy].z);

					dv_s[idy][n-1].x = 0.5 * (vt_s[idy].x + vp_s[idy].x + dt2 * a_s[idy * nb].x);
					dv_s[idy][n-1].y = 0.5 * (vt_s[idy].y + vp_s[idy].y + dt2 * a_s[idy * nb].y);
					dv_s[idy][n-1].z = 0.5 * (vt_s[idy].z + vp_s[idy].z + dt2 * a_s[idy * nb].z);	
				}
				
				if(idy < N2){
					double ddt0 = 0.25 / (n*n);
					for(int j = n-1; j >= 1; --j){
						double ddt1 = 0.25 / (j*j);
						double t0 = 1.0 / (ddt1 - ddt0);
						double t1 = t0 * 0.25 / ((j+1)*(j+1));
						double t2 = t0 * ddt0;
						
						dx_s[idy][j-1].x = t1 * dx_s[idy][j].x - t2 * dx_s[idy][j-1].x;	
						dx_s[idy][j-1].y = t1 * dx_s[idy][j].y - t2 * dx_s[idy][j-1].y;
						dx_s[idy][j-1].z = t1 * dx_s[idy][j].z - t2 * dx_s[idy][j-1].z;

						dv_s[idy][j-1].x = t1 * dv_s[idy][j].x - t2 * dv_s[idy][j-1].x;
						dv_s[idy][j-1].y = t1 * dv_s[idy][j].y - t2 * dv_s[idy][j-1].y;
						dv_s[idy][j-1].z = t1 * dv_s[idy][j].z - t2 * dv_s[idy][j-1].z;
					}
					double errorx = dx_s[idy][0].x * dx_s[idy][0].x * scalex.x;
					double errorv = dv_s[idy][0].x * dv_s[idy][0].x * scalev.x;

					errorx = fmax(errorx, dx_s[idy][0].y * dx_s[idy][0].y * scalex.y);
					errorv = fmax(errorv, dv_s[idy][0].y * dv_s[idy][0].y * scalev.y);

					errorx = fmax(errorx, dx_s[idy][0].z * dx_s[idy][0].z * scalex.z);
					errorv = fmax(errorv, dv_s[idy][0].z * dv_s[idy][0].z * scalev.z);

					error_s[idy] = fmax(errorx, errorv);
	
					Ncol[0] = 0;

				}
				__syncthreads();
				if(idy < N2){
					volatile  double *error = error_s;
					if(NN >= 32) error[idy] = fmax(error[idy], error[idy + 16]);
					if(NN >= 16) error[idy] = fmax(error[idy], error[idy + 8]); 
					if(NN >= 8) error[idy] = fmax(error[idy], error[idy + 4]);
					if(NN >= 4) error[idy] = fmax(error[idy], error[idy + 2]);
					if(NN >= 2) error[idy] = fmax(error[idy], error[idy + 1]);
				}
				__syncthreads();
				if(error_s[0] < tol * tol || sgnt * dt1 < 1.0e-6){

					if(idy < N2){
						xt_s[idy].x = dx_s[idy][0].x;
						xt_s[idy].y = dx_s[idy][0].y;
						xt_s[idy].z = dx_s[idy][0].z;

						vt_s[idy].x = dv_s[idy][0].x;
						vt_s[idy].y = dv_s[idy][0].y;
						vt_s[idy].z = dv_s[idy][0].z;		

						for(int j = 1; j < n; ++j){
							xt_s[idy].x += dx_s[idy][j].x;
							xt_s[idy].y += dx_s[idy][j].y;
							xt_s[idy].z += dx_s[idy][j].z;

							vt_s[idy].x += dv_s[idy][j].x;
							vt_s[idy].y += dv_s[idy][j].y;
							vt_s[idy].z += dv_s[idy][j].z;
						}
					}
					__syncthreads();
					for(int l = 0; l < NN; l += nb){
                                        	encouter<1>(xt_s[ii], vt_s[ii], x4_s[ii], v4_s[ii], xt_s[jj + l], vt_s[jj + l], x4_s[jj + l], v4_s[jj + l], v4_s[ii].w, v4_s[jj + l].w, 0.0, 0.0, dt1, ii, jj + l, &test, Colpairs_s, Ncol[0], 0);
					}
                                        __syncthreads();
                                        if(idy == 0) {
                                               for(int c = 0; c < Ncol[0]; ++c){
							int i = Colpairs_s[c].x;
							int j = Colpairs_s[c].y;
							if(xt_s[i].w >= 0 && xt_s[j].w >= 0){
								int nc = atomicAdd(Ncoll_d, 1);
								if(nc >= MaxColl -1) nc = MaxColl -1;
                                                        	collidesmall(xt_s, vt_s, i, j, Encpairssmall_d[si * Nconst + i].x, Encpairssmall_d[si * Nconst + j].x, index_d, indexsmall_d, nc, Coll_d, time + t/0.01720209895, spin_d, spinsmall_d);
							}
                                                }
                                        }
                                        __syncthreads();
					t += dt1;				
					if(n >= 8) dt1 *= 0.55;
					if(n < 7) dt1 *= 1.3;
					if(sgnt * dt1 > sgnt * dt) dt1 = dt;
					if(sgnt * (t+dt1) > sgnt * dt) dt1 = dt - t;
					if(sgnt * dt1 < 1.0e-7) dt1 = sgnt * 1.0e-7;

					if(idy < N2){
						x4_s[idy] = xt_s[idy];
						v4_s[idy] = vt_s[idy];
					}
					f = 0;

					__syncthreads();
					break;
				}
			}
			if(f == 0) break;
			__syncthreads();
			dt1 *= 0.5;
		}
		if(sgnt * t >= sgnt * dt) break;
		__syncthreads();
	}
	if(idy == 0){ 
		x4small_d[idi] = x4_s[idy]; 
		v4small_d[idi] = v4_s[idy];
	}

}
extern template __global__ void BSBStepsmall_kernel < 2, 2 > (double4 *, double4 *, double4 *, double4 *, double4 *, double4 *, double *, double *, int2 *, int2 *, double, double, double *, int, int *, int *, int *, double *, double, double3 *, double3 *, const int);
extern template __global__ void BSBStepsmall_kernel < 4, 4 > (double4 *, double4 *, double4 *, double4 *, double4 *, double4 *, double *, double *, int2 *, int2 *, double, double, double *, int, int *, int *, int *, double *, double, double3 *, double3 *, const int);
extern template __global__ void BSBStepsmall_kernel < 8, 8 > (double4 *, double4 *, double4 *, double4 *, double4 *, double4 *, double *, double *, int2 *, int2 *, double, double, double *, int, int *, int *, int *, double *, double, double3 *, double3 *, const int);
extern template __global__ void BSBStepsmall_kernel < 16, 16 > (double4 *, double4 *, double4 *, double4 *, double4 *, double4 *, double *, double *, int2 *, int2 *, double, double, double *, int, int *, int *, int *, double *, double, double3 *, double3 *, const int);

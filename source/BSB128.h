#include "directAcc.h"
#include "Encounter3.h"

template< int NN, int nb>
__global__ void BSBStep128_kernel(double4 *x4_d, double4 *v4_d, double4 *xold_d, double4 *vold_d, double *rcrit_d, double *rcritv_d, int2 *Encpairs2_d, double dt, double Msun, double *U_d, int st, int *index_d, int *Ncoll_d, double *Coll_d, double time, double3 *spin_d, int Nst, float4 *aelimits_d, int *aecount_d, int *enccount_d, long long *aecountT_d, long long *enccountT_d, int NB, double *K_d, double *Kold_d, int *groupIndex_d){
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
	double3 a;
	__shared__ double4 xp_s[NN];
	double4 vp;
	__shared__ double4 xt_s[NN];
	__shared__ double4 vt_s[NN];

	double3 dx[8];
	double3 dv[8];

	__shared__ int Ncol[1];
	__shared__ int2 Colpairs_s[MaxColl];
	__shared__ int N2; 
	__shared__ int sgnt;

	double3 scalex;
	double3 scalev;

	__shared__ double error_s[NN];
	double test;
	int idi;
	int si = Encpairs2_d[ (st+1) * NB + idx].y; 
		
	N2 = Encpairs2_d[si].y;
	if(idy < N2){
		idi = Encpairs2_d[si * NB + idy].x;

	}
	else idi = 0;

        if(dt < 0.0){
                sgnt = -1;
        }
        else sgnt = 1;

 	__syncthreads();
	
	if(idy < N2){
                x4_s[idy] = xold_d[idi];  
                v4_s[idy] = vold_d[idi];  
		rcritv_s[idy] = rcritv_d[idi]; 
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
		a.x = 0.0;
                a.y = 0.0;
                a.z = 0.0;
	}
	if(idy < MaxColl){
		Colpairs_s[idy].x = 0;
		Colpairs_s[idy].y = 0;

	}
	
	if(idy < NN) error_s[idy] = 0.0;
	__syncthreads();

	for(int tt = 0; tt < 1000; ++ tt){
	__syncthreads();

		if(idy < N2){
			scalex.x = 1.0 / (x4_s[idy].x * x4_s[idy].x + 1.0e-20);
			scalex.y = 1.0 / (x4_s[idy].y * x4_s[idy].y + 1.0e-20);
			scalex.z = 1.0 / (x4_s[idy].z * x4_s[idy].z + 1.0e-20);

			scalev.x = 1.0 / (v4_s[idy].x * v4_s[idy].x + 1.0e-20);
			scalev.y = 1.0 / (v4_s[idy].y * v4_s[idy].y + 1.0e-20);
			scalev.z = 1.0 / (v4_s[idy].z * v4_s[idy].z + 1.0e-20);

		}

		volatile int f = 1;
		__syncthreads();
		for(int ff = 0; ff < 1e6; ++ff){
			__syncthreads();
			for(int n = 1; n <= 8; ++n){ 
				
				dt2 = dt1 / (2.0 * n);
				dt22 = dt2 * 2.0;

				a.x = 0.0;
				a.y = 0.0;
				a.z = 0.0;
				__syncthreads();
				for(int l = 0; l < NN; l += nb){
					accEnc(x4_s[ii], x4_s[jj + l], a, rcritv_s[ii], rcritv_s[jj + l], test, ii, jj + l);
				}
				__syncthreads();

				if(idy < N2){
					accEncSun(x4_s[idy], a, ksq * Msun);
				}

				if(idy < NN){
					xp_s[idy].x = x4_s[idy].x + dt2 * v4_s[idy].x;
					xp_s[idy].y = x4_s[idy].y + dt2 * v4_s[idy].y;
					xp_s[idy].z = x4_s[idy].z + dt2 * v4_s[idy].z;
					xp_s[idy].w = x4_s[idy].w;

					vp.x = v4_s[idy].x + dt2 * a.x;  
					vp.y = v4_s[idy].y + dt2 * a.y;  
					vp.z = v4_s[idy].z + dt2 * a.z; 
					vp.w = v4_s[idy].w;
				}
				__syncthreads();
				a.x = 0.0;
				a.y = 0.0;
				a.z = 0.0;

				__syncthreads();
				for(int l = 0; l < NN; l += nb){
					accEnc(xp_s[ii], xp_s[jj + l], a, rcritv_s[ii], rcritv_s[jj + l], test, ii, jj + l);
				}
				__syncthreads();
				if(idy < NN){
					accEncSun(xp_s[idy], a, ksq * Msun);

					xt_s[idy].x = x4_s[idy].x + dt22 * vp.x;
					xt_s[idy].y = x4_s[idy].y + dt22 * vp.y;
					xt_s[idy].z = x4_s[idy].z + dt22 * vp.z;
					xt_s[idy].w = x4_s[idy].w;

					vt_s[idy].x = v4_s[idy].x + dt22 * a.x;
					vt_s[idy].y = v4_s[idy].y + dt22 * a.y;
					vt_s[idy].z = v4_s[idy].z + dt22 * a.z;
					vt_s[idy].w = v4_s[idy].w;
				}
				__syncthreads();
				
				for(int m = 2; m <= n; ++m){
					a.x = 0.0;
					a.y = 0.0;
					a.z = 0.0;

					__syncthreads();
					for(int l = 0; l < NN; l += nb){
						accEnc(xt_s[ii], xt_s[jj + l], a, rcritv_s[ii], rcritv_s[jj + l], test, ii, jj + l);
					}
					__syncthreads();

					if(idy < N2){
						accEncSun(xt_s[idy], a, ksq * Msun);

						xp_s[idy].x += dt22 * vt_s[idy].x;
						xp_s[idy].y += dt22 * vt_s[idy].y;
						xp_s[idy].z += dt22 * vt_s[idy].z;

						vp.x += dt22 * a.x;
						vp.y += dt22 * a.y;
						vp.z += dt22 * a.z;
					}
					__syncthreads();
					a.x = 0.0;
					a.y = 0.0;
					a.z = 0.0;

					__syncthreads();
					for(int l = 0; l < NN; l += nb){
						accEnc(xp_s[ii], xp_s[jj + l], a, rcritv_s[ii], rcritv_s[jj + l], test, ii, jj + l);
					}
					__syncthreads();

					if(idy < N2){
						accEncSun(xp_s[idy], a, ksq * Msun);

						xt_s[idy].x += dt22 * vp.x;
						xt_s[idy].y += dt22 * vp.y;
						xt_s[idy].z += dt22 * vp.z;

						vt_s[idy].x += dt22 * a.x;
						vt_s[idy].y += dt22 * a.y;
						vt_s[idy].z += dt22 * a.z;
					}
					__syncthreads();
				}
				a.x = 0.0;
				a.y = 0.0;
				a.z = 0.0;

				__syncthreads();
				for(int l = 0; l < NN; l += nb){
					accEnc(xt_s[ii], xt_s[jj + l], a, rcritv_s[ii], rcritv_s[jj + l], test, ii, jj + l);
				}
				__syncthreads();
				if(idy < N2){
					accEncSun(xt_s[idy], a, ksq * Msun);

					dx[n-1].x = 0.5 * (xt_s[idy].x + xp_s[idy].x + dt2 * vt_s[idy].x);
					dx[n-1].y = 0.5 * (xt_s[idy].y + xp_s[idy].y + dt2 * vt_s[idy].y);
					dx[n-1].z = 0.5 * (xt_s[idy].z + xp_s[idy].z + dt2 * vt_s[idy].z);

					dv[n-1].x = 0.5 * (vt_s[idy].x + vp.x + dt2 * a.x);
					dv[n-1].y = 0.5 * (vt_s[idy].y + vp.y + dt2 * a.y);
					dv[n-1].z = 0.5 * (vt_s[idy].z + vp.z + dt2 * a.z);	
				}
				
				if(idy < N2){
					double ddt0 = 0.25 / (n*n);
					for(int j = n-1; j >= 1; --j){
						double ddt1 = 0.25 / (j*j);
						double t0 = 1.0 / (ddt1 - ddt0);
						double t1 = t0 * 0.25 / ((j+1)*(j+1));
						double t2 = t0 * ddt0;
						
						dx[j-1].x = t1 * dx[j].x - t2 * dx[j-1].x;	
						dx[j-1].y = t1 * dx[j].y - t2 * dx[j-1].y;
						dx[j-1].z = t1 * dx[j].z - t2 * dx[j-1].z;

						dv[j-1].x = t1 * dv[j].x - t2 * dv[j-1].x;
						dv[j-1].y = t1 * dv[j].y - t2 * dv[j-1].y;
						dv[j-1].z = t1 * dv[j].z - t2 * dv[j-1].z;
					}
					double errorx = dx[0].x * dx[0].x * scalex.x;
					double errorv = dv[0].x * dv[0].x * scalev.x;

					errorx = fmax(errorx, dx[0].y * dx[0].y * scalex.y);
					errorv = fmax(errorv, dv[0].y * dv[0].y * scalev.y);

					errorx = fmax(errorx, dx[0].z * dx[0].z * scalex.z);
					errorv = fmax(errorv, dv[0].z * dv[0].z * scalev.z);

					error_s[idy] = fmax(errorx, errorv);
	
					Ncol[0] = 0;
				}
                                __syncthreads();

				if(NN >= 256){
                                	if(idy < 128) error_s[idy] = fmax(error_s[idy], error_s[idy + 128]);
					__syncthreads();
				}

				if(idy < 64) error_s[idy] = fmax(error_s[idy], error_s[idy + 64]);
				__syncthreads();
	
				if(idy < 32){
					volatile  double *error = error_s;
					error[idy] = fmax(error[idy], error[idy + 32]);
					error[idy] = fmax(error[idy], error[idy + 16]);
					error[idy] = fmax(error[idy], error[idy + 8]);
					error[idy] = fmax(error[idy], error[idy + 4]);
					error[idy] = fmax(error[idy], error[idy + 2]);
					error[idy] = fmax(error[idy], error[idy + 1]);
				}
				__syncthreads();


				if(error_s[0] < tol * tol || sgnt * dt1 < 1.0e-6){
					if(idy < N2){
						xt_s[idy].x = dx[0].x;
						xt_s[idy].y = dx[0].y;
						xt_s[idy].z = dx[0].z;

						vt_s[idy].x = dv[0].x;
						vt_s[idy].y = dv[0].y;
						vt_s[idy].z = dv[0].z;		

						for(int j = 1; j < n; ++j){
							xt_s[idy].x += dx[j].x;
							xt_s[idy].y += dx[j].y;
							xt_s[idy].z += dx[j].z;

							vt_s[idy].x += dv[j].x;
							vt_s[idy].y += dv[j].y;
							vt_s[idy].z += dv[j].z;
						}
					}
					__syncthreads();
					for(int l = 0; l < NN; l += nb){
                                        	encounter<1>(xt_s[ii], vt_s[ii], x4_s[ii], v4_s[ii], xt_s[jj + l], vt_s[jj + l], x4_s[jj + l], v4_s[jj + l], v4_s[ii].w, v4_s[jj + l].w, 0.0, 0.0, dt1, ii, jj + l, &test, Colpairs_s, Ncol[0], 0);
					}
                                        __syncthreads();

                                        if(idy == 0) {
                                               for(int c = 0; c < Ncol[0]; ++c){
							int i = Colpairs_s[c].x;
							int j = Colpairs_s[c].y;
							if(xt_s[i].w >= 0 && xt_s[j].w >= 0){
                                                                int nc = atomicAdd(Ncoll_d, 1);
								if(nc >= MaxColl -1) nc = MaxColl -1;
                                                        	collide(xt_s, vt_s, i, j, Encpairs2_d[si * NB + i].x, Encpairs2_d[si * NB + j].x, Msun, U_d, test, index_d, nc, Coll_d, time + t/0.01720209895, spin_d, rcritv_s, rcrit_d, Nst, 0, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d);
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
						v4_s = vt_s;
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
	if(idy < N2){
		x4_d[idi] = x4_s[idy]; 
		v4_d[idi] = v4_s[idy]; 
		rcritv_d[idi] = rcritv_s[idy];
	}
}

#include "directAcc.h"
#include "Encounter3.h"

// **************************************
// For less than 64 bodies
//This Kernel intergrates full simulations for one time step 
//using a Bulirsh Stoer method with nb threads.
//The implementation of the Bulirsh Stoer method is based on the mercury code from Chambers.
//
//Authors: Simon Grimm
//November 2016
// ****************************************

template< int NN, int nb>
__global__ void BSTTVStep_kernel(double4 *xold_d, double4 *vold_d, int *Transit_d, int *N_d, double *dt_d, double2 *Msun_d, int *index_d, double *time_d, int *NBS_d, const int UseGR, const double MinMass, const int UseTestParticles, int Nst, double *TransitTime_d, int2 *NtransitsT_d){
	int idy = threadIdx.x;
	int idx = blockIdx.x;

	int ii = idy / nb;
	int jj = idy % nb;

	__shared__ double4 x4_s[NN];
	__shared__ double4 v4_s[NN];
	__shared__ double3 a0_s[NN];
	__shared__ double3 a_s[NN];
	__shared__ double4 xp_s[NN];
	__shared__ double4 vp_s[NN];
	__shared__ double4 xt_s[NN];
	__shared__ double4 vt_s[NN];
	__shared__ double3 dx_s[NN][8];
	__shared__ double3 dv_s[NN][8];

	__shared__ int N2; 

	double3 a0, a;

	double3 scalex;
	double3 scalev;

	__shared__ double error_s[1];
	double error = 0.0;
	double test;
	volatile int idi;
	volatile int si = 0;

	int itransit = Transit_d[idx]; //index of the transiting object
	if(Nst > 0){
		si = index_d[itransit] / def_MaxIndex;
		itransit -= NBS_d[si];
	}
	N2 = N_d[si]; //Number of bodies in current BS simulation
//printf("BS %d %d %d %d\n", idx, si, N2, NBS_d[si]);
	if(idy < N2){
		idi = NBS_d[si] + idy;
//printf("BS2 %d %d %d %d %d %d %d %g\n", idx, idy, si, idi, index_d[idi], itransit, N2, time_d[si]);
	}
	else idi = 0;
	
	double Msun = Msun_d[si].x;
	double Rsun = Msun_d[si].y;
	double dt = dt_d[si];
	double time = time_d[si] - dt_d[si] / dayUnit;
	volatile double dt1 = dt; 
	volatile double dt2, dt22;
	volatile double t = 0.0;
	volatile double dtgr = 1.0;

	volatile double g = 1.0;
	volatile double gd = 1.0;

 	__syncthreads();
	if(idy < N2){
		x4_s[idy] = xold_d[idi];  
		v4_s[idy] = vold_d[idi];  
//if(itransit == 0) printf("BSold %d %.40g %.40g %.40g %.40g %.40g %.40g\n", idi, xold_d[idi].x, xold_d[idi].y, xold_d[idi].z, vold_d[idi].x, vold_d[idi].y, vold_d[idi].z);
		if(UseGR == 1){// GR time rescale (Saha & Tremaine 1994)
			double c2 = def_cm * def_cm;
			double mu = def_ksq * Msun;
			double rsq = x4_s[idy].x * x4_s[idy].x + x4_s[idy].y * x4_s[idy].y + x4_s[idy].z * x4_s[idy].z;
			double vsq = v4_s[idy].x * v4_s[idy].x + v4_s[idy].y * v4_s[idy].y + v4_s[idy].z * v4_s[idy].z;
			double ir = 1.0/sqrt(rsq);
			double ia = 2.0*ir-vsq/mu;
			dtgr = 1.0 - 1.5 * mu * ia / c2;
		}
	}
	else if(idy < NN){
		x4_s[idy].x = 0.0;
		x4_s[idy].y = 0.0;
		x4_s[idy].z = 0.0;
		x4_s[idy].w = -1.0e-12;
		v4_s[idy].x = 0.0;
		v4_s[idy].y = 0.0;
		v4_s[idy].z = 0.0;
		v4_s[idy].w = 0.0;
	}
	if(idy == 0) error_s[0] = 0.0;

	__syncthreads();
	for(int tt = 0; tt < 10000; ++tt){
		__syncthreads();

		if(idy < N2){
			scalex.x = 1.0 / (x4_s[idy].x * x4_s[idy].x + 1.0e-20);
			scalex.y = 1.0 / (x4_s[idy].y * x4_s[idy].y + 1.0e-20);
			scalex.z = 1.0 / (x4_s[idy].z * x4_s[idy].z + 1.0e-20);

			scalev.x = 1.0 / (v4_s[idy].x * v4_s[idy].x + 1.0e-20);
			scalev.y = 1.0 / (v4_s[idy].y * v4_s[idy].y + 1.0e-20);
			scalev.z = 1.0 / (v4_s[idy].z * v4_s[idy].z + 1.0e-20);
		}

		if(idy < NN){
			a0_s[idy].x = 0.0;
			a0_s[idy].y = 0.0;
			a0_s[idy].z = 0.0;
		}
		a0.x = 0.0;
		a0.y = 0.0;
		a0.z = 0.0;

		__syncthreads();
		for(int l = 0; l < NN; l += nb){
			accEncFull(x4_s[ii], x4_s[jj + l], a0, test, ii, jj + l, MinMass, UseTestParticles);
		}
		__syncthreads();
		{
#if def_OldShuffle == 0
			if(nb >= 16){
				a0.x += __shfl_down_sync(0xffffffff, a0.x, 8, warpSize);
				a0.y += __shfl_down_sync(0xffffffff, a0.y, 8, warpSize);
				a0.z += __shfl_down_sync(0xffffffff, a0.z, 8, warpSize);
			}
			if(nb >= 8){
				a0.x += __shfl_down_sync(0xffffffff, a0.x, 4, warpSize);
				a0.y += __shfl_down_sync(0xffffffff, a0.y, 4, warpSize);
				a0.z += __shfl_down_sync(0xffffffff, a0.z, 4, warpSize);
			}
			if(nb >= 4){
				a0.x += __shfl_down_sync(0xffffffff, a0.x, 2, warpSize);
				a0.y += __shfl_down_sync(0xffffffff, a0.y, 2, warpSize);
				a0.z += __shfl_down_sync(0xffffffff, a0.z, 2, warpSize);
			}
			if(nb >= 2){
				a0.x += __shfl_down_sync(0xffffffff, a0.x, 1, warpSize);
				a0.y += __shfl_down_sync(0xffffffff, a0.y, 1, warpSize);
				a0.z += __shfl_down_sync(0xffffffff, a0.z, 1, warpSize);
			}
#else
			if(nb >= 16){
				a0.x += __shfld_down(a0.x, 8);
				a0.y += __shfld_down(a0.y, 8);
				a0.z += __shfld_down(a0.z, 8);
			}
			if(nb >= 8){
				a0.x += __shfld_down(a0.x, 4);
				a0.y += __shfld_down(a0.y, 4);
				a0.z += __shfld_down(a0.z, 4);
			}
			if(nb >= 4){
				a0.x += __shfld_down(a0.x, 2);
				a0.y += __shfld_down(a0.y, 2);
				a0.z += __shfld_down(a0.z, 2);
			}
			if(nb >= 2){
				a0.x += __shfld_down(a0.x, 1);
				a0.y += __shfld_down(a0.y, 1);
				a0.z += __shfld_down(a0.z, 1);
			}
#endif
			if(jj == 0){
				a0_s[ii] = a0;
			}
		}
		__syncthreads();
		if(idy < N2){
			accEncSun(x4_s[idy], a0_s[idy], def_ksq * Msun * dtgr);
		}

		volatile int f = 1;
		__syncthreads();
		for(int ff = 0; ff < 1e6; ++ff){
			__syncthreads();

			g = x4_s[itransit].x * v4_s[itransit].x + x4_s[itransit].y * v4_s[itransit].y;
			gd = v4_s[itransit].x * v4_s[itransit].x + v4_s[itransit].y * v4_s[itransit].y + x4_s[itransit].x * a0_s[itransit].x + x4_s[itransit].y * a0_s[itransit].y;
//if(itransit == 0 && idy == 0) printf("A id %d %d %d  x %.20g y %.20g g %.20g gd %.20g -g/gd %g %.20g %g %g time %.20g\n", (idy + si * N_d[0]), tt, ff, x4_s[itransit].x, x4_s[itransit].y, g, gd, -g / gd, dt1 / dayUnit, dt / dayUnit, t / dayUnit, time);
			if(-g / (gd * dt1) < 0) dt1 = -dt1; 
			if(fabs(g / gd) < fabs(dt1)) dt1 = -g /gd;
//if(itransit == 0 && idy == 0) printf("B id %d %d %d  x %.20g y %.20g g %.20g gd %.20g -g/gd %g %.20g %g %g time %.20g\n", (idy + si * N_d[0]), tt, ff, x4_s[itransit].x, x4_s[itransit].y, g, gd, -g / gd, dt1 / dayUnit, dt / dayUnit, t / dayUnit, time);
			if(fabs(g / gd) < def_TransitTol){
				if(idy == itransit){
					double rsky = sqrt(x4_s[idy].x * x4_s[idy].x +  x4_s[idy].y * x4_s[idy].y);
					double R = Rsun + v4_s[idy].w;
					if(rsky < R){
						int ii = (idy + si * N_d[0]) * def_NtransitTimeMax;

						double T = time + t / dayUnit;
						int Epoch;
						if(dt > 0){
							Epoch = atomicAdd(&NtransitsT_d[idy + si * N_d[0]].x, 1);
							if(Epoch > 0){
								//check if the same transition is detected twice
								double TOld = TransitTime_d[ii + Epoch];
								if(fabs(TOld - T) < fabs(dt * 0.1)){
									NtransitsT_d[idy + si * N_d[0]].x = Epoch;
									--Epoch;
								}
							}
						}
						else{
							Epoch = atomicAdd(&NtransitsT_d[idy + si * N_d[0]].x, -1);
							if(Epoch > 0){
								double TOld = TransitTime_d[ii + Epoch + 2];
								if(fabs(TOld - T) < fabs(dt * 0.1)){
									NtransitsT_d[idy + si * N_d[0]].x = Epoch;
									++Epoch;
								}
							}

						}
//if(itransit == 2 && si == 0)  printf("TRANSIT id %d %d R %g rsky %g T %.20g Epoch %d %d\n", itransit, (idy + si * N_d[0]), R, rsky, T, Epoch, ii + Epoch + 1);
// /*if(itransit % N_d[0] == 2)*/  printf("TRANSIT id %d %d R %g rsky %g T %.20g Epoch %d %d\n", itransit, (idy + si * N_d[0]), R, rsky, T, Epoch, ii + Epoch + 1);
						TransitTime_d[ii + Epoch + 1] = T;
					}

				}
				__syncthreads();
				break;
			}
			for(int n = 1; n <= 8; ++n){

				dt2 = dt1 / (2.0 * n);
				dt22 = dt2 * 2.0;

				if(idy < NN){
					xp_s[idy].x = x4_s[idy].x + __dmul_rn(dt2 * dtgr, v4_s[idy].x);
					xp_s[idy].y = x4_s[idy].y + __dmul_rn(dt2 * dtgr, v4_s[idy].y);
					xp_s[idy].z = x4_s[idy].z + __dmul_rn(dt2 * dtgr, v4_s[idy].z);
					xp_s[idy].w = x4_s[idy].w;

					vp_s[idy].x = v4_s[idy].x + __dmul_rn(dt2, a0_s[idy].x);  
					vp_s[idy].y = v4_s[idy].y + __dmul_rn(dt2, a0_s[idy].y);  
					vp_s[idy].z = v4_s[idy].z + __dmul_rn(dt2, a0_s[idy].z); 
					vp_s[idy].w = v4_s[idy].w;
//printf("xp0 %d %d %.20g %.20g %.20g %.20g %.20g %.20g | %.20g %.20g %.20g\n", idx, idy, xp_s[idy].x, xp_s[idy].y, xp_s[idy].z, vp_s[idy].x, vp_s[idy].y, vp_s[idy].z, dt2, a0_s[idy].x, a0_s[idy].x);
				}

				if(idy < NN){
					a_s[idy].x = 0.0;
					a_s[idy].y = 0.0;
					a_s[idy].z = 0.0;
				}
				a.x = 0.0;
				a.y = 0.0;
				a.z = 0.0;

				__syncthreads();
				for(int l = 0; l < NN; l += nb){
					accEncFull(xp_s[ii], xp_s[jj + l], a, test, ii, jj + l, MinMass, UseTestParticles);
				}
				__syncthreads();
				{
#if def_OldShuffle == 0
					if(nb >= 16){
						a.x += __shfl_down_sync(0xffffffff, a.x, 8, warpSize);
						a.y += __shfl_down_sync(0xffffffff, a.y, 8, warpSize);
						a.z += __shfl_down_sync(0xffffffff, a.z, 8, warpSize);
					}
					if(nb >= 8){
						a.x += __shfl_down_sync(0xffffffff, a.x, 4, warpSize);
						a.y += __shfl_down_sync(0xffffffff, a.y, 4, warpSize);
						a.z += __shfl_down_sync(0xffffffff, a.z, 4, warpSize);
					}
					if(nb >= 4){
						a.x += __shfl_down_sync(0xffffffff, a.x, 2, warpSize);
						a.y += __shfl_down_sync(0xffffffff, a.y, 2, warpSize);
						a.z += __shfl_down_sync(0xffffffff, a.z, 2, warpSize);
					}
					if(nb >= 2){
						a.x += __shfl_down_sync(0xffffffff, a.x, 1, warpSize);
						a.y += __shfl_down_sync(0xffffffff, a.y, 1, warpSize);
						a.z += __shfl_down_sync(0xffffffff, a.z, 1, warpSize);
					}
#else
					if(nb >= 16){
						a.x += __shfld_down(a.x, 8);
						a.y += __shfld_down(a.y, 8);
						a.z += __shfld_down(a.z, 8);
					}
					if(nb >= 8){
						a.x += __shfld_down(a.x, 4);
						a.y += __shfld_down(a.y, 4);
						a.z += __shfld_down(a.z, 4);
					}
					if(nb >= 4){
						a.x += __shfld_down(a.x, 2);
						a.y += __shfld_down(a.y, 2);
						a.z += __shfld_down(a.z, 2);
					}
					if(nb >= 2){
						a.x += __shfld_down(a.x, 1);
						a.y += __shfld_down(a.y, 1);
						a.z += __shfld_down(a.z, 1);
					}
#endif
					if(jj == 0){
						a_s[ii] = a;
					}
				}
				__syncthreads();
				if(idy < NN){
					accEncSun(xp_s[idy], a_s[idy], def_ksq * Msun * dtgr);
					xt_s[idy].x = x4_s[idy].x + __dmul_rn(dt22 * dtgr, vp_s[idy].x);
					xt_s[idy].y = x4_s[idy].y + __dmul_rn(dt22 * dtgr, vp_s[idy].y);
					xt_s[idy].z = x4_s[idy].z + __dmul_rn(dt22 * dtgr, vp_s[idy].z);
					xt_s[idy].w = x4_s[idy].w;

					vt_s[idy].x = v4_s[idy].x + __dmul_rn(dt22, a_s[idy].x);
					vt_s[idy].y = v4_s[idy].y + __dmul_rn(dt22, a_s[idy].y);
					vt_s[idy].z = v4_s[idy].z + __dmul_rn(dt22, a_s[idy].z);
					vt_s[idy].w = v4_s[idy].w;
//printf("xt0 %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", idx, idy, xt_s[idy].x, xt_s[idy].y, xt_s[idy].z, vt_s[idy].x, vt_s[idy].y, vt_s[idy].z);
				}
				__syncthreads();
				
				for(int m = 2; m <= n; ++m){
					if(idy < NN){
						a_s[idy].x = 0.0;
						a_s[idy].y = 0.0;
						a_s[idy].z = 0.0;
					}
					a.x = 0.0;
					a.y = 0.0;
					a.z = 0.0;

					__syncthreads();
					for(int l = 0; l < NN; l += nb){
						accEncFull(xt_s[ii], xt_s[jj + l], a, test, ii, jj + l, MinMass, UseTestParticles);
					}
					__syncthreads();
					{
#if def_OldShuffle == 0
						if(nb >= 16){
							a.x += __shfl_down_sync(0xffffffff, a.x, 8, warpSize);
							a.y += __shfl_down_sync(0xffffffff, a.y, 8, warpSize);
							a.z += __shfl_down_sync(0xffffffff, a.z, 8, warpSize);
						}
						if(nb >= 8){
							a.x += __shfl_down_sync(0xffffffff, a.x, 4, warpSize);
							a.y += __shfl_down_sync(0xffffffff, a.y, 4, warpSize);
							a.z += __shfl_down_sync(0xffffffff, a.z, 4, warpSize);
						}
						if(nb >= 4){
							a.x += __shfl_down_sync(0xffffffff, a.x, 2, warpSize);
							a.y += __shfl_down_sync(0xffffffff, a.y, 2, warpSize);
							a.z += __shfl_down_sync(0xffffffff, a.z, 2, warpSize);
						}
						if(nb >= 2){
							a.x += __shfl_down_sync(0xffffffff, a.x, 1, warpSize);
							a.y += __shfl_down_sync(0xffffffff, a.y, 1, warpSize);
							a.z += __shfl_down_sync(0xffffffff, a.z, 1, warpSize);
						}
#else
						if(nb >= 16){
							a.x += __shfld_down(a.x, 8);
							a.y += __shfld_down(a.y, 8);
							a.z += __shfld_down(a.z, 8);
						}
						if(nb >= 8){
							a.x += __shfld_down(a.x, 4);
							a.y += __shfld_down(a.y, 4);
							a.z += __shfld_down(a.z, 4);
						}
						if(nb >= 4){
							a.x += __shfld_down(a.x, 2);
							a.y += __shfld_down(a.y, 2);
							a.z += __shfld_down(a.z, 2);
						}
						if(nb >= 2){
							a.x += __shfld_down(a.x, 1);
							a.y += __shfld_down(a.y, 1);
							a.z += __shfld_down(a.z, 1);
						}
#endif
						if(jj == 0){
							a_s[ii] = a;
						}
					}
					__syncthreads();
					if(idy < N2){
						accEncSun(xt_s[idy], a_s[idy], def_ksq * Msun * dtgr);

						xp_s[idy].x += __dmul_rn(dt22, dtgr * vt_s[idy].x);
						xp_s[idy].y += __dmul_rn(dt22, dtgr * vt_s[idy].y);
						xp_s[idy].z += __dmul_rn(dt22, dtgr * vt_s[idy].z);

						vp_s[idy].x += __dmul_rn(dt22, a_s[idy].x);
						vp_s[idy].y += __dmul_rn(dt22, a_s[idy].y);
						vp_s[idy].z += __dmul_rn(dt22, a_s[idy].z);
					}
					__syncthreads();

					if(idy < NN){
						a_s[idy].x = 0.0;
						a_s[idy].y = 0.0;
						a_s[idy].z = 0.0;
					}
					a.x = 0.0;
					a.y = 0.0;
					a.z = 0.0;

					__syncthreads();
					for(int l = 0; l < NN; l += nb){
						accEncFull(xp_s[ii], xp_s[jj + l], a, test, ii, jj + l, MinMass, UseTestParticles);
					}
					__syncthreads();
					{
#if def_OldShuffle == 0
						if(nb >= 16){
							a.x += __shfl_down_sync(0xffffffff, a.x, 8, warpSize);
							a.y += __shfl_down_sync(0xffffffff, a.y, 8, warpSize);
							a.z += __shfl_down_sync(0xffffffff, a.z, 8, warpSize);
						}
						if(nb >= 8){
							a.x += __shfl_down_sync(0xffffffff, a.x, 4, warpSize);
							a.y += __shfl_down_sync(0xffffffff, a.y, 4, warpSize);
							a.z += __shfl_down_sync(0xffffffff, a.z, 4, warpSize);
						}
						if(nb >= 4){
							a.x += __shfl_down_sync(0xffffffff, a.x, 2, warpSize);
							a.y += __shfl_down_sync(0xffffffff, a.y, 2, warpSize);
							a.z += __shfl_down_sync(0xffffffff, a.z, 2, warpSize);
						}
						if(nb >= 2){
							a.x += __shfl_down_sync(0xffffffff, a.x, 1, warpSize);
							a.y += __shfl_down_sync(0xffffffff, a.y, 1, warpSize);
							a.z += __shfl_down_sync(0xffffffff, a.z, 1, warpSize);
						}
#else
						if(nb >= 16){
							a.x += __shfld_down(a.x, 8);
							a.y += __shfld_down(a.y, 8);
							a.z += __shfld_down(a.z, 8);
						}
						if(nb >= 8){
							a.x += __shfld_down(a.x, 4);
							a.y += __shfld_down(a.y, 4);
							a.z += __shfld_down(a.z, 4);
						}
						if(nb >= 4){
							a.x += __shfld_down(a.x, 2);
							a.y += __shfld_down(a.y, 2);
							a.z += __shfld_down(a.z, 2);
						}
						if(nb >= 2){
							a.x += __shfld_down(a.x, 1);
							a.y += __shfld_down(a.y, 1);
							a.z += __shfld_down(a.z, 1);
						}
#endif
						if(jj == 0){
							a_s[ii] = a;
						}
					}
					__syncthreads();
					if(idy < N2){
						accEncSun(xp_s[idy], a_s[idy], def_ksq * Msun * dtgr);

						xt_s[idy].x += __dmul_rn(dt22, dtgr * vp_s[idy].x);
						xt_s[idy].y += __dmul_rn(dt22, dtgr * vp_s[idy].y);
						xt_s[idy].z += __dmul_rn(dt22, dtgr * vp_s[idy].z);

						vt_s[idy].x += __dmul_rn(dt22, a_s[idy].x);
						vt_s[idy].y += __dmul_rn(dt22, a_s[idy].y);
						vt_s[idy].z += __dmul_rn(dt22, a_s[idy].z);
//printf("xt %d %.20g %.20g %.20g %.20g %.20g %.20g\n", idy, xt_s[idy].x, xt_s[idy].y, xt_s[idy].z, vt_s[idy].x, vt_s[idy].y, vt_s[idy].z);
					}
					__syncthreads();
				}//end of m loop

				if(idy < NN){
					a_s[idy].x = 0.0;
					a_s[idy].y = 0.0;
					a_s[idy].z = 0.0;
				}
				a.x = 0.0;
				a.y = 0.0;
				a.z = 0.0;

				__syncthreads();
				for(int l = 0; l < NN; l += nb){
					accEncFull(xt_s[ii], xt_s[jj + l], a, test, ii, jj + l, MinMass, UseTestParticles);
				}
				__syncthreads();
				{
#if def_OldShuffle == 0
					if(nb >= 16){
						a.x += __shfl_down_sync(0xffffffff, a.x, 8, warpSize);
						a.y += __shfl_down_sync(0xffffffff, a.y, 8, warpSize);
						a.z += __shfl_down_sync(0xffffffff, a.z, 8, warpSize);
					}
					if(nb >= 8){
						a.x += __shfl_down_sync(0xffffffff, a.x, 4, warpSize);
						a.y += __shfl_down_sync(0xffffffff, a.y, 4, warpSize);
						a.z += __shfl_down_sync(0xffffffff, a.z, 4, warpSize);
					}
					if(nb >= 4){
						a.x += __shfl_down_sync(0xffffffff, a.x, 2, warpSize);
						a.y += __shfl_down_sync(0xffffffff, a.y, 2, warpSize);
						a.z += __shfl_down_sync(0xffffffff, a.z, 2, warpSize);
					}
					if(nb >= 2){
						a.x += __shfl_down_sync(0xffffffff, a.x, 1, warpSize);
						a.y += __shfl_down_sync(0xffffffff, a.y, 1, warpSize);
						a.z += __shfl_down_sync(0xffffffff, a.z, 1, warpSize);
					}
#else
					if(nb >= 16){
						a.x += __shfld_down(a.x, 8);
						a.y += __shfld_down(a.y, 8);
						a.z += __shfld_down(a.z, 8);
					}
					if(nb >= 8){
						a.x += __shfld_down(a.x, 4);
						a.y += __shfld_down(a.y, 4);
						a.z += __shfld_down(a.z, 4);
					}
					if(nb >= 4){
						a.x += __shfld_down(a.x, 2);
						a.y += __shfld_down(a.y, 2);
						a.z += __shfld_down(a.z, 2);
					}
					if(nb >= 2){
						a.x += __shfld_down(a.x, 1);
						a.y += __shfld_down(a.y, 1);
						a.z += __shfld_down(a.z, 1);
					}
#endif
					if(jj == 0){
						a_s[ii] = a;
					}
				}
				__syncthreads();

				if(idy < N2){
					accEncSun(xt_s[idy], a_s[idy], def_ksq * Msun * dtgr);

					dx_s[idy][n-1].x = 0.5 * (xt_s[idy].x + (xp_s[idy].x + (dt2 * dtgr * vt_s[idy].x)));
					dx_s[idy][n-1].y = 0.5 * (xt_s[idy].y + (xp_s[idy].y + (dt2 * dtgr * vt_s[idy].y)));
					dx_s[idy][n-1].z = 0.5 * (xt_s[idy].z + (xp_s[idy].z + (dt2 * dtgr * vt_s[idy].z)));

					dv_s[idy][n-1].x = 0.5 * (vt_s[idy].x + (vp_s[idy].x + (dt2 * a_s[idy].x)));
					dv_s[idy][n-1].y = 0.5 * (vt_s[idy].y + (vp_s[idy].y + (dt2 * a_s[idy].y)));
					dv_s[idy][n-1].z = 0.5 * (vt_s[idy].z + (vp_s[idy].z + (dt2 * a_s[idy].z)));	
				}
				
				if(idy < N2){
					for(int j = n-1; j >= 1; --j){
						double t0 = BSt0_c[(n-1) * 8 + (j-1)];
						double t1 = t0 * BSddt_c[j];
						double t2 = t0 * BSddt_c[n-1];
						
						dx_s[idy][j-1].x = t1 * dx_s[idy][j].x - t2 * dx_s[idy][j-1].x;	
						dx_s[idy][j-1].y = t1 * dx_s[idy][j].y - t2 * dx_s[idy][j-1].y;
						dx_s[idy][j-1].z = t1 * dx_s[idy][j].z - t2 * dx_s[idy][j-1].z;

						dv_s[idy][j-1].x = t1 * dv_s[idy][j].x - t2 * dv_s[idy][j-1].x;
						dv_s[idy][j-1].y = t1 * dv_s[idy][j].y - t2 * dv_s[idy][j-1].y;
						dv_s[idy][j-1].z = t1 * dv_s[idy][j].z - t2 * dv_s[idy][j-1].z;
					}
					double errorx = dx_s[idy][0].x * dx_s[idy][0].x * scalex.x;
					double errorv = dv_s[idy][0].x * dv_s[idy][0].x * scalev.x;
//printf("dx %d %d %d %d %g %g %g %g %g %g | %g %g \n", idy, tt, ff, n, dx_s[idy][0].x, dx_s[idy][0].y, dx_s[idy][0].z, dv_s[idy][0].x, dv_s[idy][0].y, dv_s[idy][0].z, t, dt1); 
					errorx = fmax(errorx, dx_s[idy][0].y * dx_s[idy][0].y * scalex.y);
					errorv = fmax(errorv, dv_s[idy][0].y * dv_s[idy][0].y * scalev.y);

					errorx = fmax(errorx, dx_s[idy][0].z * dx_s[idy][0].z * scalex.z);
					errorv = fmax(errorv, dv_s[idy][0].z * dv_s[idy][0].z * scalev.z);

					error = fmax(errorx, errorv);
				}
				else{
					error = 0.0;
				}
				__syncthreads();
				{
#if def_OldShuffle == 0
					if(NN >= 32) error = fmax(error, __shfl_down_sync(0xffffffff, error, 16, warpSize));
					if(NN >= 16) error = fmax(error, __shfl_down_sync(0xffffffff, error,  8, warpSize));
					if(NN >= 8)  error = fmax(error, __shfl_down_sync(0xffffffff, error,  4, warpSize));
					if(NN >= 4)  error = fmax(error, __shfl_down_sync(0xffffffff, error,  2, warpSize));
					if(NN >= 2)  error = fmax(error, __shfl_down_sync(0xffffffff, error,  1, warpSize));
#else
					if(NN >= 32) error = fmax(error, __shfld_down(error, 16));
					if(NN >= 16) error = fmax(error, __shfld_down(error,  8));
					if(NN >= 8)  error = fmax(error, __shfld_down(error,  4));
					if(NN >= 4)  error = fmax(error, __shfld_down(error,  2));
					if(NN >= 2)  error = fmax(error, __shfld_down(error,  1));
#endif
					if(idy == 0) error_s[0] = error;
				}
				__syncthreads();

				if(error_s[0] < def_tol * def_tol || fabs(dt1) < def_dtmin){

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
					t += dt1;
					if(n >= 8) dt1 *= 0.55;
					if(n < 7) dt1 *= 1.3;
					if(dt1 < def_dtmin && dt1 > 0.0) dt1 = def_dtmin;
					if(dt1 > def_dtmin && dt1 <= 0.0) dt1 = -def_dtmin;

					if(idy < N2){
						x4_s[idy] = xt_s[idy];
						v4_s[idy] = vt_s[idy];
//if(itransit == 1) printf("update %d %d %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %g %g %d %d %d\n", idy, idx, index_d[idi], x4_s[idy].x, x4_s[idy].y, x4_s[idy].z, v4_s[idy].x, v4_s[idy].y, v4_s[idy].z, x4_s[idy].w, v4_s[idy].w, t, dt1, tt, ff, n);
					}
					f = 0;

					__syncthreads();
					break;
				}
			}//end of n loop
			if(f == 0) break;
			__syncthreads();
			dt1 *= 0.5;
		}//end of ff loop
		if(fabs(g / gd) < def_TransitTol) break;

		__syncthreads();
	}//end of tt loop
}

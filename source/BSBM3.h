#include "directAcc.h"
#include "Encounter3.h"

// **************************************
//
//Authors: Simon Grimm
//March 2023
// ****************************************

template< int NN, int nb>
__global__ void BSBM3Step_kernel(curandState *random_d, double4 *x4_d, double4 *v4_d, double4 *xold_d, double4 *vold_d, double *rcrit_d, double *rcritv_d, int2 *Encpairs2_d, int2 *groupIndex_d, double *dt_d, const double FGt, double2 *Msun_d, double *U_d, const int st, int *index_d, int *BSstop_d, int *Ncoll_d, double *Coll_d, double *time_d, double4 *spin_d, double3 *love_d, int *createFlag_d, float4 *aelimits_d, unsigned int *aecount_d, unsigned int *enccount_d, unsigned long long *aecountT_d, unsigned long long *enccountT_d, const int NT, const int NconstT, int *NWriteEnc_d, double *writeEnc_d, const int UseGR, const double MinMass, const int UseTestParticles, const int SLevels, int noColl){


	int idy = threadIdx.x;
	int idx = blockIdx.x;

#if USE_RANDOM == 1
	curandState random = random_d[idx];
#else
	int random = 0;
#endif

	if((noColl == 1 || noColl == -1) && BSstop_d[0] == 3){
//if(idy == 0 && idx == 0)      printf("Stop BSB b\n");
		return;
	}
//printf("BSB %d %d %g %g %d\n", idy, idx, StopMinMass_c[0], CollisionPrecision_c[0], noColl);

	int ii = idy / nb;
	int jj = idy % nb;
	volatile double dt1;
	volatile double dt2, dt22;
	volatile double t = 0.0;
	volatile double dtgr = 1.0;

	__shared__ double4 x4_s[NN];
	__shared__ double4 v4_s[NN];
	__shared__ double rcritv_s[NN * def_SLevelsMax];
	__shared__ double3 a0_s[NN];
	__shared__ double3 a_s[NN];
	__shared__ double4 xp_s[NN];
	__shared__ double4 vp_s[NN];
	__shared__ double4 xt_s[NN];
	__shared__ double4 vt_s[NN];
	__shared__ double3 dx_s[NN][8];
	__shared__ double3 dv_s[NN][8];

	double3 a0, a;

	__shared__ int Ncol_s[1];
	__shared__ int2 Colpairs_s[def_MaxColl];
	__shared__ double Coltime_s[def_MaxColl];
	__shared__ int sstt;

	volatile int sgnt;

	double3 scalex;
	double3 scalev;

	__shared__ double error_s[1];
	double error = 0.0;
	double test;
	volatile int idi;
	volatile int si = Encpairs2_d[ (st+2) * NT + idx].y;	//group index 
	volatile int N2 = groupIndex_d[si + 1].y; //Number of bodies in current BS simulation
	volatile int start = Encpairs2_d[NT + si].y;
//printf("BS %d %d %d %d %d %d\n", idx, st, si, N2, NT, start);
	if(idy < N2){
		idi = Encpairs2_d[start + idy].x;
//printf("BS2 %d %d %d %d %d %d %d\n", idx, st, si, idi, index_d[idi], N2, NN);
	}
	else idi = 0;

	if(idy == 0){
		sstt = index_d[idi] / def_MaxIndex;
	}
	__syncthreads();

	double Msun = Msun_d[sstt].x;
	double time = time_d[sstt] - dt_d[sstt] / dayUnit;
	double dt = dt_d[sstt] * FGt;

	dt1 = dt;

	if(dt < 0.0){
		sgnt = -1;
	}
	else sgnt = 1;

 	__syncthreads();
	if(idy < N2){
		x4_s[idy] = xold_d[idi];
		v4_s[idy] = vold_d[idi];
		for(int l = 0; l < SLevels; ++l){
			rcritv_s[idy + l * NN] = rcritv_d[idi + l * NconstT];
		}
//printf("BSold %d %.40g %.40g %.40g %.40g %.40g %.40g\n", idi, xold_d[idi].x, xold_d[idi].y, xold_d[idi].z, vold_d[idi].x, vold_d[idi].y, vold_d[idi].z);
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
		for(int l = 0; l < SLevels; ++l){
			rcritv_s[idy + l * NN] = 0.0;
		}
	}
	if(idy < def_MaxColl){
		Colpairs_s[idy].x = 0;
		Colpairs_s[idy].y = 0;
		Coltime_s[idy] = 10.0;
	}
	if(idy == 0){
		error_s[0] = 0.0;
	}
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
			accEnc(x4_s[ii], x4_s[jj + l], a0, rcritv_s, test, ii, jj + l, NN, MinMass, UseTestParticles, SLevels);
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
			for(int n = 1; n <= 8; ++n){
				dt2 = dt1 / (2.0 * n);
				dt22 = dt2 * 2.0;

				if(idy < NN){
					xp_s[idy].x = x4_s[idy].x + (dt2 * dtgr * v4_s[idy].x);
					xp_s[idy].y = x4_s[idy].y + (dt2 * dtgr * v4_s[idy].y);
					xp_s[idy].z = x4_s[idy].z + (dt2 * dtgr * v4_s[idy].z);
					xp_s[idy].w = x4_s[idy].w;

					vp_s[idy].x = v4_s[idy].x + (dt2 * a0_s[idy].x);
					vp_s[idy].y = v4_s[idy].y + (dt2 * a0_s[idy].y);
					vp_s[idy].z = v4_s[idy].z + (dt2 * a0_s[idy].z);
					vp_s[idy].w = v4_s[idy].w;
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
					accEnc(xp_s[ii], xp_s[jj + l], a, rcritv_s, test, ii, jj + l, NN, MinMass, UseTestParticles, SLevels);
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

					xt_s[idy].x = x4_s[idy].x + (dt22 * dtgr * vp_s[idy].x);
					xt_s[idy].y = x4_s[idy].y + (dt22 * dtgr * vp_s[idy].y);
					xt_s[idy].z = x4_s[idy].z + (dt22 * dtgr * vp_s[idy].z);
					xt_s[idy].w = x4_s[idy].w;

					vt_s[idy].x = v4_s[idy].x + (dt22 * a_s[idy].x);
					vt_s[idy].y = v4_s[idy].y + (dt22 * a_s[idy].y);
					vt_s[idy].z = v4_s[idy].z + (dt22 * a_s[idy].z);
					vt_s[idy].w = v4_s[idy].w;
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
						accEnc(xt_s[ii], xt_s[jj + l], a, rcritv_s, test, ii, jj + l, NN, MinMass, UseTestParticles, SLevels);
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

						xp_s[idy].x += (dt22 * dtgr * vt_s[idy].x);
						xp_s[idy].y += (dt22 * dtgr * vt_s[idy].y);
						xp_s[idy].z += (dt22 * dtgr * vt_s[idy].z);

						vp_s[idy].x += (dt22 * a_s[idy].x);
						vp_s[idy].y += (dt22 * a_s[idy].y);
						vp_s[idy].z += (dt22 * a_s[idy].z);
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
						accEnc(xp_s[ii], xp_s[jj + l], a, rcritv_s, test, ii, jj + l, NN, MinMass, UseTestParticles, SLevels);
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

						xt_s[idy].x += (dt22 * dtgr * vp_s[idy].x);
						xt_s[idy].y += (dt22 * dtgr * vp_s[idy].y);
						xt_s[idy].z += (dt22 * dtgr * vp_s[idy].z);

						vt_s[idy].x += (dt22 * a_s[idy].x);
						vt_s[idy].y += (dt22 * a_s[idy].y);
						vt_s[idy].z += (dt22 * a_s[idy].z);
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
					accEnc(xt_s[ii], xt_s[jj + l], a, rcritv_s, test, ii, jj + l, NN, MinMass, UseTestParticles, SLevels);
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
//printf("dx %d %d %d %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", tt, ff, n, idy, idi, dx_s[idy][n-1].x, dx_s[idy][n-1].y, dx_s[idy][n-1].z, dv_s[idy][n-1].x, dv_s[idy][n-1].y, dv_s[idy][n-1].z);
				}
				
				if(idy < N2){
					for(int j = n-1; j >= 1; --j){
						double t0 = BSt0_c[(n-1) * 8 + (j-1)];
						double t1 = t0 * BSddt_c[j];
						double t2 = t0 * BSddt_c[n-1];
						
						dx_s[idy][j-1].x = (t1 * dx_s[idy][j].x) - (t2 * dx_s[idy][j-1].x);	
						dx_s[idy][j-1].y = (t1 * dx_s[idy][j].y) - (t2 * dx_s[idy][j-1].y);
						dx_s[idy][j-1].z = (t1 * dx_s[idy][j].z) - (t2 * dx_s[idy][j-1].z);

						dv_s[idy][j-1].x = (t1 * dv_s[idy][j].x) - (t2 * dv_s[idy][j-1].x);
						dv_s[idy][j-1].y = (t1 * dv_s[idy][j].y) - (t2 * dv_s[idy][j-1].y);
						dv_s[idy][j-1].z = (t1 * dv_s[idy][j].z) - (t2 * dv_s[idy][j-1].z);
					}
					double errorx = dx_s[idy][0].x * dx_s[idy][0].x * scalex.x;
					double errorv = dv_s[idy][0].x * dv_s[idy][0].x * scalev.x;
					errorx = fmax(errorx, dx_s[idy][0].y * dx_s[idy][0].y * scalex.y);
					errorv = fmax(errorv, dv_s[idy][0].y * dv_s[idy][0].y * scalev.y);

					errorx = fmax(errorx, dx_s[idy][0].z * dx_s[idy][0].z * scalex.z);
					errorv = fmax(errorv, dv_s[idy][0].z * dv_s[idy][0].z * scalev.z);

					error = fmax(errorx, errorv);
//printf("%d %d %d %d %d %.20g %g %g %g %g %g %g | %g %g \n", idy, tt, ff, n, idi, error, dx_s[idy][0].x, dx_s[idy][0].y, dx_s[idy][0].z, dv_s[idy][0].x, dv_s[idy][0].y, dv_s[idy][0].z, t, dt1); 
	
					Ncol_s[0] = 0;
					Coltime_s[0] = 10.0;

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

				if(error_s[0] < def_tol * def_tol || sgnt * dt1 < def_dtmin){

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
//printf("xt %d %d %d %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", tt, ff, n, idy, idi, xt_s[idy].x, xt_s[idy].y, xt_s[idy].z, vt_s[idy].x, vt_s[idy].y, vt_s[idy].z);
					}
					__syncthreads();
					for(int l = 0; l < NN; l += nb){
						if(ii < N2 && jj + l < N2){
							double delta = 1000.0;
							double enct = 100.0;
							double colt = 100.0;
							double rcrit = v4_s[ii].w + v4_s[jj + l].w;
							if((noColl == 1 || noColl == -1) && index_d[Encpairs2_d[start + ii].x] == CollTshiftpairs_c[0].x && index_d[Encpairs2_d[start + jj + l].x] == CollTshiftpairs_c[0].y){
								rcrit = v4_s[ii].w * CollTshift_c[0] + v4_s[jj + l].w * CollTshift_c[0];
							}
							if((noColl == 1 || noColl == -1) && index_d[Encpairs2_d[start + ii].x] == CollTshiftpairs_c[0].y && index_d[Encpairs2_d[start + jj + l].x] == CollTshiftpairs_c[0].x){
								rcrit = v4_s[ii].w * CollTshift_c[0] + v4_s[jj + l].w * CollTshift_c[0];
							}

							if(CollisionPrecision_c[0] < 0.0){
								//do not overlap bodies when collision, increase therefore radius slightly, R + R * precision
								rcrit *= (1.0 - CollisionPrecision_c[0]);	
							}
							if(Encpairs2_d[start + ii].x > Encpairs2_d[start + jj + l].x){
								delta = encounter1(xt_s[ii], vt_s[ii], x4_s[ii], v4_s[ii], xt_s[jj + l], vt_s[jj + l], x4_s[jj + l], v4_s[jj + l], rcrit, dt1 * dtgr, ii, jj + l, enct, colt, MinMass, noColl);
							}
							if((noColl == 1 || noColl == -1) && colt == 100.0){
								delta = 100.0;
							}
							if((noColl == 1 || noColl == -1) && colt == 200.0){
								noColl = 2;
								BSstop_d[0] = 3;
							}

//if( Encpairs2_d[start + ii].x == 0 &&  Encpairs2_d[start + jj + l].x == 1) printf("BSBEE %d %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %g %d %d %d\n", Encpairs2_d[start + ii].x, Encpairs2_d[start + jj + l].x, xt_s[ii].w, xt_s[jj + l].w, xt_s[ii].x, xt_s[ii].y, xt_s[ii].z, xt_s[jj + l].x, xt_s[jj + l].y, xt_s[jj + l].z, delta, rcrit*rcrit, colt, tt, ff, n);
							if(delta < rcrit*rcrit){
								int Ni = atomicAdd(&Ncol_s[0], 1);
								if(Ncol_s[0] >= def_MaxColl) Ni = def_MaxColl - 1;
//printf("EE1 %d %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %g %d\n", Encpairs2_d[start + ii].x, Encpairs2_d[start + jj + l].x, xt_s[ii].w, xt_s[jj + l].w, xt_s[ii].x, xt_s[ii].y, xt_s[ii].z, xt_s[jj + l].x, xt_s[jj + l].y, xt_s[jj + l].z, delta, rcrit*rcrit, colt, Ni);
								if(xt_s[ii].w >= xt_s[jj + l].w){
									Colpairs_s[Ni].x = ii;
									Colpairs_s[Ni].y = jj + l;
								}
								else{
									Colpairs_s[Ni].x = jj + l;
									Colpairs_s[Ni].y = ii;
								}
								Coltime_s[Ni] = colt;

								// *****************
								//dont group test particles
/*								if(xt_s[ii].w == 0.0){
									Colpairs_s[Ni].x = ii;
									Colpairs_s[Ni].y = ii;
								}
								if(xt_s[jj + l].w == 0.0){
									Colpairs_s[Ni].x = jj + l;
									Colpairs_s[Ni].y = jj + l;
								}
*/
								// *****************
							}

							if(WriteEncounters_c[0] > 0 && noColl == 0){
								double writeRadius = 0.0;
								//in scales of planetary Radius
								writeRadius = WriteEncountersRadius_c[0] * fmax(vt_s[ii].w, vt_s[jj + l].w);
//printf("Enc %g %g %g %g %g %d %d\n", (t + dt1) / dayUnit, writeRadius, sqrt(delta), enct, colt, ii, jj + l);
								if(delta < writeRadius * writeRadius){

									if(enct > 0.0 && enct < 1.0){
										//ingnore encounters within the same particle cloud
										int indexi = Encpairs2_d[start + ii].x;
										int indexj = Encpairs2_d[start + jj + l].x;
										if(index_d[indexi] / WriteEncountersCloudSize_c[0] != index_d[indexj] / WriteEncountersCloudSize_c[0]){

//printf("Write Enc %g %g %g %g %g %d %d\n", (t + dt1) / dayUnit, writeRadius, sqrt(delta), enct, colt, ii, jj + l);
											int ne = atomicAdd(NWriteEnc_d, 1);
											if(ne >= def_MaxWriteEnc - 1) ne = def_MaxWriteEnc - 1;
											storeEncounters(xt_s, vt_s, ii, jj + l, indexi, indexj, index_d, ne, writeEnc_d, time + (t + dt1) / dayUnit, spin_d);
										}
									}
								}
							}
						}
					}
					__syncthreads();
					if(idy == 0){
						double Coltime = 10.0;
						for(int c = 0; c < min(Ncol_s[0], def_MaxColl); ++c){
							int i = Colpairs_s[c].x;
							int j = Colpairs_s[c].y;

							//Calculate real separation at the end of the time step
							double dx = xt_s[i].x - xt_s[j].x;
							double dy = xt_s[i].y - xt_s[j].y;
							double dz = xt_s[i].z - xt_s[j].z;
							double d = sqrt(dx * dx + dy * dy + dz * dz);
							double R = vt_s[i].w + vt_s[j].w;

							if((noColl == 1 || noColl == -1) && index_d[Encpairs2_d[start + i].x] == CollTshiftpairs_c[0].x && index_d[Encpairs2_d[start + j].x] == CollTshiftpairs_c[0].y){
								R = vt_s[i].w * CollTshift_c[0] + vt_s[j].w * CollTshift_c[0];
							}

							if(CollisionPrecision_c[0] < 0.0){
								//do not overlap bodies when collision, increase therefore radius slightly, R + R * precision
								R *= (1.0 - CollisionPrecision_c[0]);	
							}

							double dR = (R - d) / R;
							if(noColl == -1) dR = -dR;

//printf("dR %d %d %.20g %.20g %.20g %g\n", i, j, d, R, dR, Coltime_s[c]);
							if(dR > fabs(CollisionPrecision_c[0]) && d != 0.0){
								//bodies are already overlapping
								Coltime = fmin(Coltime_s[c], Coltime);
							}

						}
						Coltime_s[0] = Coltime;
//printf("ColtimeT %.20g %g %g %g %d %d %d %d %d\n", Coltime_s[0], t / dayUnit, dt1 / dayUnit, (1.0 - Coltime) * dt1, tt, ff, n, Ncol_s[0], Ncoll_d[0]);
					}
					__syncthreads();
					if(Coltime_s[0] == 10.0){
						if(idy == 0) {
							for(int c = 0; c < min(Ncol_s[0], def_MaxColl); ++c){
								int i = Colpairs_s[c].x;
								int j = Colpairs_s[c].y;
								if(xt_s[i].w >= 0 && xt_s[j].w >= 0){
									int nc = 0;
									if(noColl == 0 || ((noColl == 1 || noColl == -1) && index_d[Encpairs2_d[start + i].x] == CollTshiftpairs_c[0].x && index_d[Encpairs2_d[start + j].x] == CollTshiftpairs_c[0].y)){
										nc = atomicAdd(Ncoll_d, 1);
										if(nc >= def_MaxColl) nc = def_MaxColl - 1;
										if(noColl == 1 || noColl == -1){
											noColl = 2;
											BSstop_d[0] = 3;
										}
									}
//printf("cTime coll BSB %g %g %g %.20g %d %d %d\n", time, t / dayUnit, dt /dayUnit, time + (t + dt1) / dayUnit, index_d[Encpairs2_d[start + i].x], index_d[Encpairs2_d[start + j].x], nc);
									collide(random, xt_s, vt_s, i, j, Encpairs2_d[start + i].x, Encpairs2_d[start + j].x, Msun, U_d, test, index_d, nc, Coll_d, time + (t + dt1) / dayUnit, spin_d, love_d, createFlag_d, rcritv_s, rcrit_d, NN, NconstT, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, SLevels, noColl);
								}
							}
						}
						__syncthreads();
						t += dt1;
						if(n >= 8) dt1 *= 0.55;
						if(n < 7) dt1 *= 1.3;
						if(sgnt * dt1 > sgnt * dt) dt1 = dt;
						if(sgnt * (t+dt1) > sgnt * dt) dt1 = dt - t;
						if(sgnt * dt1 < def_dtmin) dt1 = sgnt * def_dtmin;

						if(idy < N2){
							x4_s[idy] = xt_s[idy];
							v4_s[idy] = vt_s[idy];
//printf("update %d %d %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.10g %g %g %d %d %d\n", idy, idx, idi, x4_s[idy].x, x4_s[idy].y, x4_s[idy].z, v4_s[idy].x, v4_s[idy].y, v4_s[idy].z, x4_s[idy].w, v4_s[idy].w, rcritv_s[idy], t / dayUnit, dt1 / dayUnit, tt, ff, n);
						}
					}
					else{
						dt1 *= Coltime_s[0];
//if(idy == 0) printf("New Coltime %g %g\n", dt1, Coltime_s[0]);
					}
					f = 0;

					__syncthreads();
					break;
				}
				__syncthreads();
				if(BSstop_d[0] == 3){
//if(idy == 0) printf("Stop BSB\n");
					__syncthreads();
					return;
				}
			}//end of n loop
			if(f == 0) break;
			__syncthreads();
			dt1 *= 0.5;
		}//end of ff loop
		if(sgnt * t >= sgnt * dt){
			break;
		}

		__syncthreads();
	}//end of tt loop
	__syncthreads();
	if(idy < N2){
//if(x4_s[idy].w <= 0){
		x4_d[idi] = x4_s[idy]; 
		v4_d[idi] = v4_s[idy];
		for(int l = 0; l < SLevels; ++l){  
			rcritv_d[idi + l * NconstT] = rcritv_s[idy + l * NN];
		}
//}
//printf("BS %d %.40g %.40g %.40g %.40g %.40g %.40g %.20g\n", idi, x4_s[idy].x, x4_s[idy].y, x4_s[idy].z, v4_s[idy].x, v4_s[idy].y, v4_s[idy].z, time + t/dayUnit);
	}
#if USE_RANDOM == 1
	random_d[idx] = random;
#endif

}

#include "directAcc.h"
#include "Encounter3.h"

// **************************************
// For multi simulation mode
//This Kernel intergrates the independent groups of close encunters for a time step
//using a Bulirsh Stoer method with nb threds. Where n is the minimum of n^2 and 256
//The implementation of the Bulirsh Stoer method is based on the mercury code from Chambers.
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
//  ****************************************
template< int NN, int nb>
__global__ void BSBMStep_kernel(double4 *x4_d, double4 *v4_d, double4 *xold_d, double4 *vold_d, double *rcrit_d, double *rcritv_d, int2 *Encpairs_d, int2 *Encpairs2_d, double *dt_d, double FGt, double4 *Msun_d, double *U_d, int st, int *index_d, int *Ncoll_d, double *Coll_d, double *time_d, double3 *spin_d, int Nst, float4 *aelimits_d, int *aecount_d, int *enccount_d, long long *aecountT_d, long long *enccountT_d, int writeEncounters_d, double writeEncountersRadius_d, int *NWriteEnc_d, double *writeEnc_d, int UseForce, double MinMass){

	int idy = threadIdx.x;
	int idx = blockIdx.x;

	int ii = idy / nb;
	int jj = idy % nb;
	double dt1; 
	double dt2, dt22;
	double t = 0.0;

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

	__shared__ int Ncol_s[1];
	__shared__ int2 Colpairs_s[MaxColl];
	__shared__ double Coltime_s[MaxColl];
	__shared__ int N2; 
	__shared__ int sgnt;
	
	__shared__ int sstt;

	double3 scalex;
	double3 scalev;

	__shared__ double error_s[2*NN];
	double test;
	int writeEncounters = writeEncounters_d;
	double writeEncountersRadius = writeEncountersRadius_d;
	int idi;
	int si = Encpairs2_d[ (st+1) + NmaxM * idx].y;
	N2 = Encpairs_d[si + Nst].y;
	if(idy < N2){
		idi = Encpairs_d[(si * NmaxM) + idy].x;
//printf("BS %d %d %d %d %d %d\n", idx, st, si, idi, index_d[idi], N2);
	}
	else idi = 0;

	if(idy == 0){
		sstt = index_d[idi] / 100;
	}
	__syncthreads();

        double Msun = Msun_d[sstt].x;
        double time = time_d[sstt];
	double dt = dt_d[sstt] * FGt;
	double dtgr = 1.0;
	dt1 = dt;

        if(dt < 0.0){
                sgnt = -1;
        }
        else sgnt = 1;

 	__syncthreads();

	if(idy < N2){
                x4_s[idy] = xold_d[idi];  
                v4_s[idy] = vold_d[idi];  
		rcritv_s[idy] = rcritv_d[idi];
//printf("BSold %d %.40g %.40g %.40g %.40g %.40g %.40g\n", idi, xold_d[idi].x, xold_d[idi].y, xold_d[idi].z, vold_d[idi].x, vold_d[idi].y, vold_d[idi].z);
//printf("BSB %d %d %d %.40g %.40g %.40g %.40g %.40g %.40g\n", sstt, idy, idi, x4_s[idy].x, x4_s[idy].y, x4_s[idy].z, v4_s[idy].x, v4_s[idy].y, v4_s[idy].z);
		if(UseForce & 1){// GR time rescale (Saha & Tremaine 1994)
			double c2 = def_cm * def_cm;
			double mu = ksq * Msun;
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
		Coltime_s[idy] = 0.0;
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
			accEnc(x4_s[ii], x4_s[jj + l], a0_s[idy], rcritv_s[ii], rcritv_s[jj + l], test, ii, jj + l, MinMass);
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
			accEncSun(x4_s[idy], a0_s[idy * nb], ksq * Msun * dtgr);
		}

		volatile int f = 1;
		__syncthreads();
		for(int ff = 0; ff < 1e6; ++ff){
			__syncthreads();
			for(int n = 1; n <= 8; ++n){
				dt2 = dt1 / (2.0 * n);
				dt22 = dt2 * 2.0;

				if(idy < NN){
					xp_s[idy].x = x4_s[idy].x + dt2 * dtgr * v4_s[idy].x;
					xp_s[idy].y = x4_s[idy].y + dt2 * dtgr * v4_s[idy].y;
					xp_s[idy].z = x4_s[idy].z + dt2 * dtgr * v4_s[idy].z;
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
					accEnc(xp_s[ii], xp_s[jj + l], a_s[idy], rcritv_s[ii], rcritv_s[jj + l], test, ii, jj + l, MinMass);
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
					accEncSun(xp_s[idy], a_s[idy * nb], ksq * Msun * dtgr);

					xt_s[idy].x = x4_s[idy].x + dt22 * dtgr * vp_s[idy].x;
					xt_s[idy].y = x4_s[idy].y + dt22 * dtgr * vp_s[idy].y;
					xt_s[idy].z = x4_s[idy].z + dt22 * dtgr * vp_s[idy].z;
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
						accEnc(xt_s[ii], xt_s[jj + l], a_s[idy], rcritv_s[ii], rcritv_s[jj + l], test, ii, jj + l, MinMass);
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
						accEncSun(xt_s[idy], a_s[idy * nb], ksq * Msun * dtgr);

						xp_s[idy].x += dt22 * dtgr * vt_s[idy].x;
						xp_s[idy].y += dt22 * dtgr * vt_s[idy].y;
						xp_s[idy].z += dt22 * dtgr * vt_s[idy].z;

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
						accEnc(xp_s[ii], xp_s[jj + l], a_s[idy], rcritv_s[ii], rcritv_s[jj + l], test, ii, jj + l, MinMass);
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
						accEncSun(xp_s[idy], a_s[idy * nb], ksq * Msun * dtgr);

						xt_s[idy].x += dt22 * dtgr * vp_s[idy].x;
						xt_s[idy].y += dt22 * dtgr * vp_s[idy].y;
						xt_s[idy].z += dt22 * dtgr * vp_s[idy].z;

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
					accEnc(xt_s[ii], xt_s[jj + l], a_s[idy], rcritv_s[ii], rcritv_s[jj + l], test, ii, jj + l, MinMass);
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
					accEncSun(xt_s[idy], a_s[idy * nb], ksq * Msun * dtgr);

					dx_s[idy][n-1].x = 0.5 * (xt_s[idy].x + (xp_s[idy].x + (dt2 * dtgr * vt_s[idy].x)));
					dx_s[idy][n-1].y = 0.5 * (xt_s[idy].y + (xp_s[idy].y + (dt2 * dtgr * vt_s[idy].y)));
					dx_s[idy][n-1].z = 0.5 * (xt_s[idy].z + (xp_s[idy].z + (dt2 * dtgr * vt_s[idy].z)));

					dv_s[idy][n-1].x = 0.5 * (vt_s[idy].x + (vp_s[idy].x + (dt2 * a_s[idy * nb].x)));
					dv_s[idy][n-1].y = 0.5 * (vt_s[idy].y + (vp_s[idy].y + (dt2 * a_s[idy * nb].y)));
					dv_s[idy][n-1].z = 0.5 * (vt_s[idy].z + (vp_s[idy].z + (dt2 * a_s[idy * nb].z)));	
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
	
					Ncol_s[0] = 0;

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
//printf("Error %d %d  ff: %d %d %g\n", sstt, n, ff, idy, error_s[0]);
				if(error_s[0] < def_tol * def_tol || sgnt * dt1 < 1.0e-6){

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
						double enct = 0.0;
				//		double colltime = 0.0;
						if(Encpairs_d[(si * NmaxM) + ii].x > Encpairs_d[(si * NmaxM) + jj + l].x){
							encounter<1>(xt_s[ii], vt_s[ii], x4_s[ii], v4_s[ii], xt_s[jj + l], vt_s[jj + l], x4_s[jj + l], v4_s[jj + l], v4_s[ii].w, v4_s[jj + l].w, 0.0, 0.0, dt1 * dtgr, ii, jj + l, &test, Colpairs_s, Coltime_s, Ncol_s[0], 0, enct, writeEncounters, writeEncountersRadius, 0.0);
						}
						//write Encounters to file
						if(enct > 0.0){
							int ne = atomicAdd(NWriteEnc_d, 1);
							if(ne >= MaxWriteEnc -1) ne = MaxWriteEnc -1;
							storeEncounters(xt_s, vt_s, ii, jj + l, Encpairs_d[(si * NmaxM) + ii].x, Encpairs_d[(si * NmaxM) + jj + l].x, index_d, ne, writeEnc_d, time + (t - dt)/dayUnit, spin_d);
						}
					}
                                        __syncthreads();
					if(idy == 0) {
						for(int c = 0; c < min(Ncol_s[0], MaxColl); ++c){
							int i = Colpairs_s[c].x;
							int j = Colpairs_s[c].y;
							if(xt_s[i].w >= 0 && xt_s[j].w >= 0){
								int nc = atomicAdd(Ncoll_d, 1);
								if(nc >= MaxColl -1) nc = MaxColl -1;
								collide(xt_s, vt_s, i, j, Encpairs_d[(si * NmaxM) + i].x, Encpairs_d[(si * NmaxM) + j].x, Msun, U_d + sstt, test, index_d, nc, Coll_d, time + (t - dt)/dayUnit, spin_d, rcritv_s, rcrit_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d);
							}
						}
						if(Ncol_s[0] >= MaxColl - 1) Ncoll_d[0] = MaxColl - 1;
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
//printf("update %d %.20g %.20g %.20g %.20g %.20g %.20g %g %g\n", index_d[idi], x4_s[idy].x, x4_s[idy].y, x4_s[idy].z, v4_s[idy].x, v4_s[idy].y, v4_s[idy].z, t, dt1);
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
//printf("BS %d %.40g %.40g %.40g %.40g %.40g %.40g\n", idi, x4_d[idi].x, x4_d[idi].y, x4_d[idi].z, v4_d[idi].x, v4_d[idi].y, v4_d[idi].z);

	}
}


/**************************************
* This Kernel intergrates the independent groups of close encunters for a time step
* using a Bulirsh Stoer method.
* The implementation is based on the mercury code from Chambers.
*
*
* Authors: Simon Grimm, Joachmin Stadel
* December 2011
*
****************************************/

template <int NN, int NB>
__global__ void BSstep_kernel(double4 *x4_d, double4 *v4_d, double4 *xold_d, double4 *vold_d, double3 *a_d, int2 *Encpairs2_d, const double Msun, const double dt, double *test_d, double *test2_d, double *U_d, double *rcrit_d, double *rcritv_d, int *index_d, int *Ncoll_d, double *Coll_d, double time, double3 *spin_d, int Nst, float4 *aelimits_d, int *aecount_d, int *enccount_d, long long *aecountT_d, long long *enccountT_d, double *K_d, double *Kold_d, int *groupIndex_d, int *groupIndexOld_d){
	int idy = threadIdx.x;
	int idx = blockIdx.x;


	const double ksqMsun = ksq * Msun;
	const double tol = 1.0e-12;
	__shared__ double4 x4_s[NN];
	__shared__ double4 v4_s[NN];
	__shared__ double3 a_s[NN];
	__shared__ double3 a0_s[NN];
	__shared__ double4 xp_s[NN];
	__shared__ double4 vp_s[NN];
	__shared__ double4 xt_s[NN];
	__shared__ double4 vt_s[NN];

	__shared__ double3 dx_s[NN][8]; 
	__shared__ double3 dv_s[NN][8]; 
	__shared__ double errorx[NN];
	__shared__ double errorv[NN];
	__shared__ double3 scalex[NN];
	__shared__ double3 scalev[NN];

	__shared__ int Encpairs2_s[NN];

	__shared__ double rcritv_s[NN];
	double dt1; dt1 = dt;
	double dt2;
	double dt22;
	__shared__ volatile double error; error = 0.0;
	double t; t = 0.0;
	__shared__ int Ncol[1];
	__shared__ int2 Colpairs_s[MaxColl];
	__shared__ int N2; N2 = Encpairs2_d[idx].y;
	double test;
	int ii;
//if(N2 > 32) test_d[idx] = N2;
//if(N2 > 32) N2 = 32; //***************
if(idy < N2){
	Encpairs2_s[idy] = Encpairs2_d[idx * NB + idy].x;


	if(idy < N2){
		ii = Encpairs2_d[idx * NB + idy].x;
//printf("%d %d\n ",idx, ii);
	}
	else ii = 0;

	if(idy < N2){
		x4_s[idy] = xold_d[ii];
		v4_s[idy] = vold_d[ii];
		rcritv_s[idy] = rcritv_d[ii];
	}
	else{
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

	Colpairs_s[idy].x = 0;
	Colpairs_s[idy].y = 0;

	xt_s[idy].x = 0.0;
	xt_s[idy].y = 0.0;
	xt_s[idy].z = 0.0;

	vt_s[idy].x = 0.0;
	vt_s[idy].y = 0.0;
	vt_s[idy].z = 0.0;

	__syncthreads();

int sgnt = 1;
if(dt < 0) sgnt = -1;

	int mm = -1;
	while(sgnt * t < sgnt * dt && mm < 1e8){
		++mm;
		scalex[idy].x = 1.0 / (x4_s[idy].x * x4_s[idy].x + 1.0e-20);
		scalex[idy].y = 1.0 / (x4_s[idy].y * x4_s[idy].y + 1.0e-20);
		scalex[idy].z = 1.0 / (x4_s[idy].z * x4_s[idy].z + 1.0e-20);

		scalev[idy].x = 1.0 / (v4_s[idy].x * v4_s[idy].x + 1.0e-20);
		scalev[idy].y = 1.0 / (v4_s[idy].y * v4_s[idy].y + 1.0e-20);
		scalev[idy].z = 1.0 / (v4_s[idy].z * v4_s[idy].z + 1.0e-20);

		a0_s[idy].x = 0.0;
		a0_s[idy].y = 0.0;
		a0_s[idy].z = 0.0;

		__syncthreads();
		for(int j = 0; j < N2; ++j){
			int jj = Encpairs2_s[j];
			accEnc(x4_s[idy], x4_s[j], a0_s[idy], rcritv_s[idy], rcritv_s[j], test, ii, jj);
		}
		accEncSun(x4_s[idy], a0_s[idy], ksqMsun);

		volatile int f = 1;
		int fc = -1;
		__syncthreads();
		while(f == 1 && fc < 1e3){
			__syncthreads();
			++fc;
			for(int n = 1; n <= 8; ++n){ 
				dt2 = dt1 / (2.0 * n);	
				dt22 = dt2 * 2.0;

				xt_s[idy].w = x4_s[idy].w;
				vt_s[idy].w = v4_s[idy].w;
				xp_s[idy].w = x4_s[idy].w;
				vp_s[idy].w = v4_s[idy].w;

				xp_s[idy].x = x4_s[idy].x + dt2 * v4_s[idy].x;
				xp_s[idy].y = x4_s[idy].y + dt2 * v4_s[idy].y;
				xp_s[idy].z = x4_s[idy].z + dt2 * v4_s[idy].z;

				vp_s[idy].x = v4_s[idy].x + dt2 * a0_s[idy].x;
				vp_s[idy].y = v4_s[idy].y + dt2 * a0_s[idy].y;
				vp_s[idy].z = v4_s[idy].z + dt2 * a0_s[idy].z;

				a_s[idy].x = 0.0;
				a_s[idy].y = 0.0;
				a_s[idy].z = 0.0;

				__syncthreads();
				for(int j = 0; j < N2; ++j){
					int jj = Encpairs2_s[j];
					accEnc(xp_s[idy], xp_s[j], a_s[idy], rcritv_s[idy], rcritv_s[j], test, ii, jj);
				}
				accEncSun(xp_s[idy], a_s[idy], ksqMsun);

				xt_s[idy].x = x4_s[idy].x + dt22 * vp_s[idy].x;
				xt_s[idy].y = x4_s[idy].y + dt22 * vp_s[idy].y;
				xt_s[idy].z = x4_s[idy].z + dt22 * vp_s[idy].z;	
			
				vt_s[idy].x = v4_s[idy].x + dt22 * a_s[idy].x;
				vt_s[idy].y = v4_s[idy].y + dt22 * a_s[idy].y;
				vt_s[idy].z = v4_s[idy].z + dt22 * a_s[idy].z;

				for(int m = 2; m <= n; ++m){
					a_s[idy].x = 0.0;
					a_s[idy].y = 0.0;
					a_s[idy].z = 0.0;
					
					__syncthreads();
					for(int j = 0; j < N2; ++j){
						int jj = Encpairs2_s[j];
						accEnc(xt_s[idy], xt_s[j], a_s[idy], rcritv_s[idy], rcritv_s[j], test, ii, jj);
					}
					accEncSun(xt_s[idy], a_s[idy], ksqMsun);

					xp_s[idy].x = xp_s[idy].x + dt22 * vt_s[idy].x;
					xp_s[idy].y = xp_s[idy].y + dt22 * vt_s[idy].y;
					xp_s[idy].z = xp_s[idy].z + dt22 * vt_s[idy].z;

					vp_s[idy].x = vp_s[idy].x + dt22 * a_s[idy].x;
					vp_s[idy].y = vp_s[idy].y + dt22 * a_s[idy].y;
					vp_s[idy].z = vp_s[idy].z + dt22 * a_s[idy].z;

					a_s[idy].x = 0.0;
					a_s[idy].y = 0.0;
					a_s[idy].z = 0.0;

					__syncthreads();
					for(int j = 0; j < N2; ++j){
						int jj = Encpairs2_s[j];
						accEnc(xp_s[idy], xp_s[j], a_s[idy], rcritv_s[idy], rcritv_s[j], test, ii, jj);
					}
					accEncSun(xp_s[idy], a_s[idy], ksqMsun);

					xt_s[idy].x = xt_s[idy].x + dt22 * vp_s[idy].x;
					xt_s[idy].y = xt_s[idy].y + dt22 * vp_s[idy].y;
					xt_s[idy].z = xt_s[idy].z + dt22 * vp_s[idy].z;

					vt_s[idy].x = vt_s[idy].x + dt22 * a_s[idy].x;
					vt_s[idy].y = vt_s[idy].y + dt22 * a_s[idy].y;
					vt_s[idy].z = vt_s[idy].z + dt22 * a_s[idy].z;
				}

				a_s[idy].x = 0.0;
				a_s[idy].y = 0.0;
				a_s[idy].z = 0.0;

				__syncthreads();
				for(int j = 0; j < N2; ++j){
					int jj = Encpairs2_s[j];
					accEnc(xt_s[idy], xt_s[j], a_s[idy], rcritv_s[idy], rcritv_s[j], test, ii, jj);
				}
				accEncSun(xt_s[idy], a_s[idy], ksqMsun);

				dx_s[idy][n-1].x = 0.5 * (xt_s[idy].x + xp_s[idy].x + dt2 * vt_s[idy].x);
				dx_s[idy][n-1].y = 0.5 * (xt_s[idy].y + xp_s[idy].y + dt2 * vt_s[idy].y);
				dx_s[idy][n-1].z = 0.5 * (xt_s[idy].z + xp_s[idy].z + dt2 * vt_s[idy].z);

				dv_s[idy][n-1].x = 0.5 * (vt_s[idy].x + vp_s[idy].x + dt2 * a_s[idy].x);
				dv_s[idy][n-1].y = 0.5 * (vt_s[idy].y + vp_s[idy].y + dt2 * a_s[idy].y);
				dv_s[idy][n-1].z = 0.5 * (vt_s[idy].z + vp_s[idy].z + dt2 * a_s[idy].z);

				double ddt0 = 0.25 / (n*n);
				for(int j = n-1; j >=1; --j){
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

				if(n > 2){   // 

					error = 0.0;
					errorx[idy] = dx_s[idy][0].x * dx_s[idy][0].x * scalex[idy].x;
					errorv[idy] = dv_s[idy][0].x * dv_s[idy][0].x * scalev[idy].x;

					errorx[idy] = fmax(errorx[idy], dx_s[idy][0].y * dx_s[idy][0].y * scalex[idy].y); 
					errorx[idy] = fmax(errorx[idy], dx_s[idy][0].z * dx_s[idy][0].z * scalex[idy].z); 

					errorv[idy] = fmax(errorv[idy], dv_s[idy][0].y * dv_s[idy][0].y * scalev[idy].y); 
					errorv[idy] = fmax(errorv[idy], dv_s[idy][0].z * dv_s[idy][0].z * scalev[idy].z); 

					__syncthreads();
		
					if(idy == 0){
						for(int i = 0; i < N2; ++i){
							error = fmax(error, errorx[i]);
							error = fmax(error, errorv[i]);
						}
						Ncol[0] = 0;
					}
	
					__syncthreads();

					if(error <= tol * tol || sgnt * dt1 < 0.000001){

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
						__syncthreads();

						for(int j = 0; j < N2; ++j){
							encounter<1>(xt_s[idy], vt_s[idy], x4_s[idy], v4_s[idy], xt_s[j], vt_s[j], x4_s[j], v4_s[j], v4_s[idy].w, v4_s[j].w, 0.0, 0.0, dt1, idy, j, &test, Colpairs_s, Ncol[0], 0); 
						}
						__syncthreads();
						if(idy == 0){
							for(int c = 0; c < Ncol[0]; ++c){
								int i = Colpairs_s[c].x;
								int j = Colpairs_s[c].y;
								if(xt_s[i].w >= 0 && xt_s[j].w >= 0){
                                                                	int nc = atomicAdd(Ncoll_d, 1);
									collide(xt_s, vt_s, i, j, Encpairs2_d[idx * NB + i].x, Encpairs2_d[idx * NB + j].x, Msun, U_d, test, index_d, nc, Coll_d, time + t/0.01720209895, spin_d, rcritv_s, rcrit_d, Nst, 0, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d);
								}
							}
						}
						__syncthreads();
						t += dt1;

						if(n >= 8) dt1 *= 0.55;
						if(n < 7) dt1 *= 1.3;
						if(sgnt * dt1 > sgnt * dt) dt1 = dt;
						if(sgnt * (t + dt1) > sgnt *dt) dt1 = dt - t;
						if(sgnt *dt1 < 0.0000001) dt1 = sgnt *0.0000001;


						x4_s[idy] = xt_s[idy];
						v4_s[idy] = vt_s[idy];

						f = 0;
						__syncthreads();
						break;
					}
				}
			}
			if(f ==1) dt1 *= 0.5;
		}
	}

	if(idy < N2){
		x4_d[ii] = x4_s[idy];
		v4_d[ii] = v4_s[idy];
		rcritv_d[ii] = rcritv_s[idy];
	}
}
}

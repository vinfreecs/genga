template <int NN>
__global__ void BSA_kernel(double4 *x4_d, double4 *v4_d, double4 *xold_d, double4 *vold_d, double *rcrit_d, double *rcritv_d, int *index_d, double3 *spin_d, int2 *Encpairs_d, int2 *Encpairs2_d, double dt, double Msun, double *U_d, int st, int NB, int NencMax, int *Ncoll_d, double *Coll_d, double time, int Nst, float4 *aelimits_d, int *aecount_d, int *enccount_d, long long *aecountT_d, long long *enccountT_d, int writeEncounters_d, double writeEncountersRadius_d, int *NWriteEnc_d, double *writeEnc_d){
	int idy = threadIdx.x;
	int idx = blockIdx.x;

	double dt1 = dt;
	double dt2, dt22;
	double t = 0.0;

	const double tol = 1.0e-12;

	__shared__ double4 x4_s[NN];
	__shared__ double4 xp_s[NN];
	__shared__ double4 xt_s[NN];
	__shared__ double4 v4_s[NN];
	__shared__ double4 vt_s[NN];
	__shared__ double rcritv_s[NN];
	__shared__ int stop_s[1];
	__shared__ int Ncol[1];
	__shared__ int2 Colpairs_s[MaxColl];
	
	double4 vp;
	double4 vt;
	double3 a;
	double3 a0;

	double3 dx0, dv0;
	double3 dx1, dv1;
	double3 dx2, dv2;
	double3 dx3, dv3;
	double3 dx4, dv4;
	double3 dx5, dv5;
	double3 dx6, dv6;
	double3 dx7, dv7;

	int sgnt = 1;

	double3 scalex;
	double3 scalev;

	double errorx, errorv;
	double test;
	int writeEncounters = writeEncounters_d;
	double writeEncountersRadius = writeEncountersRadius_d;

	int idi;
	int si = Encpairs2_d[ (st+2) * NB + idx].y;
	int N2 = Encpairs2_d[si].y; //Number of bodies in  current BS simulation
	int start = Encpairs2_d[NB + si].y;
//printf("BS %d %d %d %d %d\n", idx, st, si, N2, NB);


	int Ne; //number of pairs
	int j0, j1;

	if(idy < N2){
		idi = Encpairs2_d[start + idy].x;
		x4_s[idy] = xold_d[idi];
		v4_s[idy] = vold_d[idi];
		rcritv_s[idy] = rcritv_d[idi];
		Ne = Encpairs_d[idi].y;
		
		int j0g = Encpairs_d[idi * NencMax + 0].x; //index of j in global memory
		int j1g = Encpairs_d[idi * NencMax + 1].x; //index of j in global memory
		
		j0 = Encpairs_d[NB + j0g].y;
		j1 = Encpairs_d[NB + j1g].y;
//printf("BS2 %d %d %d %d %d %d %d %d %d %d %d\n", idx, idy, st, idi, index_d[idi], j0g, j0, j1g, j1, N2, Ne);
	}	
	else{
		idi = 0;
		x4_s[idy].x = 0.0;
		x4_s[idy].y = 0.0;
		x4_s[idy].z = 0.0;
		x4_s[idy].w = -1.0e-12;//0.0;
		v4_s[idy].x = 0.0;
		v4_s[idy].y = 0.0;
		v4_s[idy].z = 0.0;
		v4_s[idy].w = 0.0;
		rcritv_s[idy] = 0.0;
		Ne = 0;
		j0 = 0;
		j1 = 0;
	}

	if(dt < 0.0){
		sgnt = -1;
	}

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
		a0.x = 0.0;
		a0.y = 0.0;
		a0.z = 0.0;

		if(Ne > 0) accEnc(x4_s[idy], x4_s[j0], a0, rcritv_s[idy], rcritv_s[j0], test, idy, j0);
		if(Ne > 1) accEnc(x4_s[idy], x4_s[j1], a0, rcritv_s[idy], rcritv_s[j1], test, idy, j1);
		for(int i = 2; i < Ne; ++i){
			int jg = Encpairs_d[idi * NencMax + i].x;
			int j = Encpairs_d[NB + jg].y;
			accEnc(x4_s[idy], x4_s[j], a0, rcritv_s[idy], rcritv_s[j], test, idy, j);
		}
		__syncthreads();
		if(Ne > 0){
			accEncSun(x4_s[idy], a0, ksq * Msun);
		}

		volatile int f = 1;
		__syncthreads();
		for(int ff = 0; ff < 1e6; ++ff){
			for(int n = 1; n <= 8; ++n){
				if(idy == 0) stop_s[0] = 1;
				__syncthreads();
				dt2 = dt1 / (2.0 * n);
				dt22 = dt2 * 2.0;

				if(Ne > 0){
					xp_s[idy].x = x4_s[idy].x + dt2 * v4_s[idy].x;
					xp_s[idy].y = x4_s[idy].y + dt2 * v4_s[idy].y;
					xp_s[idy].z = x4_s[idy].z + dt2 * v4_s[idy].z;
					xp_s[idy].w = x4_s[idy].w;

					vp.x = v4_s[idy].x + dt2 * a0.x;
					vp.y = v4_s[idy].y + dt2 * a0.y;
					vp.z = v4_s[idy].z + dt2 * a0.z;
					vp.w = v4_s[idy].w;
//printf("a0p %d %d %d %.20g %.20g %.20g %.20g %d\n", idy, idx, idi, xp_s[idy].x, vp.x, x4_s[idy].x, v4_s[idy].x, n);
				}
				__syncthreads();

				a.x = 0.0;
				a.y = 0.0;
				a.z = 0.0;

				if(Ne > 0) accEnc(xp_s[idy], xp_s[j0], a, rcritv_s[idy], rcritv_s[j0], test, idy, j0);
				if(Ne > 1) accEnc(xp_s[idy], xp_s[j1], a, rcritv_s[idy], rcritv_s[j1], test, idy, j1);
				for(int i = 2; i < Ne; ++i){
					int jg = Encpairs_d[idi * NencMax + i].x;
					int j = Encpairs_d[NB + jg].y;
					accEnc(xp_s[idy], xp_s[j], a, rcritv_s[idy], rcritv_s[j], test, idy, j);
				}
				__syncthreads();
				if(Ne > 0){
					accEncSun(xp_s[idy], a, ksq * Msun);

					xt_s[idy].x = x4_s[idy].x + dt22 * vp.x;
					xt_s[idy].y = x4_s[idy].y + dt22 * vp.y;
					xt_s[idy].z = x4_s[idy].z + dt22 * vp.z;
					xt_s[idy].w = x4_s[idy].w;

					vt.x = v4_s[idy].x + dt22 * a.x;
					vt.y = v4_s[idy].y + dt22 * a.y;
					vt.z = v4_s[idy].z + dt22 * a.z;
					vt.w = v4_s[idy].w;

//printf("a0t %d %d %d %.20g %.20g %.20g %.20g %d\n", idy, idx, idi, xt_s[idy].x, vt.x, x4_s[idy].x, vp.x, n);
				}
				__syncthreads();

				for(int m = 2; m <= n; ++m){

					a.x = 0.0;
					a.y = 0.0;
					a.z = 0.0;

					if(Ne > 0) accEnc(xt_s[idy], xt_s[j0], a, rcritv_s[idy], rcritv_s[j0], test, idy, j0);
					if(Ne > 1) accEnc(xt_s[idy], xt_s[j1], a, rcritv_s[idy], rcritv_s[j1], test, idy, j1);
					for(int i = 2; i < Ne; ++i){
						int jg = Encpairs_d[idi * NencMax + i].x;
						int j = Encpairs_d[NB + jg].y;
						accEnc(xt_s[idy], xt_s[j], a, rcritv_s[idy], rcritv_s[j], test, idy, j);
					}
					__syncthreads();
					if(Ne > 0){
						accEncSun(xt_s[idy], a, ksq * Msun);

						xp_s[idy].x += dt22 * vt.x;
						xp_s[idy].y += dt22 * vt.y;
						xp_s[idy].z += dt22 * vt.z;

						vp.x += dt22 * a.x;
						vp.y += dt22 * a.y;
						vp.z += dt22 * a.z;
//printf("amp %d %d %d %.20g %.20g %.20g %d\n", idy, idx, idi, xp_s[idy].x, vp.x, vt.x, n);
					}
					__syncthreads();

					a.x = 0.0;
					a.y = 0.0;
					a.z = 0.0;

					if(Ne > 0) accEnc(xp_s[idy], xp_s[j0], a, rcritv_s[idy], rcritv_s[j0], test, idy, j0);
					if(Ne > 1) accEnc(xp_s[idy], xp_s[j1], a, rcritv_s[idy], rcritv_s[j1], test, idy, j1);	
					for(int i = 2; i < Ne; ++i){
						int jg = Encpairs_d[idi * NencMax + i].x;
						int j = Encpairs_d[NB + jg].y;
						accEnc(xp_s[idy], xp_s[j], a, rcritv_s[idy], rcritv_s[j], test, idy, j);
					}
					__syncthreads();
					if(Ne > 0){
						accEncSun(xp_s[idy], a, ksq * Msun);

						xt_s[idy].x += dt22 * vp.x;
						xt_s[idy].y += dt22 * vp.y;
						xt_s[idy].z += dt22 * vp.z;

						vt.x += dt22 * a.x;
						vt.y += dt22 * a.y;
						vt.z += dt22 * a.z;
//printf("amt %d %d %d %.20g %.20g %.20g %d\n", idy, idx, idi, xt_s[idy].x, vt.x, vp.x, n);
					}
					__syncthreads();
				}//end of m loop
				a.x = 0.0;
				a.y = 0.0;
				a.z = 0.0;

				if(Ne > 0) accEnc(xt_s[idy], xt_s[j0], a, rcritv_s[idy], rcritv_s[j0], test, idy, j0);
				if(Ne > 1) accEnc(xt_s[idy], xt_s[j1], a, rcritv_s[idy], rcritv_s[j1], test, idy, j1);
				for(int i = 2; i < Ne; ++i){
					int jg = Encpairs_d[idi * NencMax + i].x;
					int j = Encpairs_d[NB + jg].y;
					accEnc(xt_s[idy], xt_s[j], a, rcritv_s[idy], rcritv_s[j], test, idy, j);
				}
				__syncthreads();
				if(Ne > 0){
					accEncSun(xt_s[idy], a, ksq * Msun);

					xp_s[idy].x += dt2 * vt.x;
					xp_s[idy].y += dt2 * vt.y;
					xp_s[idy].z += dt2 * vt.z;

					vp.x += dt2 * a.x;
					vp.y += dt2 * a.y;
					vp.z += dt2 * a.z;
//printf("xt %d %.20g %.20g %.20g %.20g %.20g %.20g %d\n", idi, xt_s[idy].x, xt_s[idy].y, xt_s[idy].z, vt.x, vt.y, vt.z, n);
//printf("xp %d %.20g %.20g %.20g %.20g %.20g %.20g %d\n", idi, xp_s[idy].x, xp_s[idy].y, xp_s[idy].z, vp.x, vp.y, vp.z, n);
					if(n == 8){				
						dx7.x = 0.5 * (xt_s[idy].x + xp_s[idy].x);
						dx7.y = 0.5 * (xt_s[idy].y + xp_s[idy].y);
						dx7.z = 0.5 * (xt_s[idy].z + xp_s[idy].z);

						dv7.x = 0.5 * (vt.x + vp.x);
						dv7.y = 0.5 * (vt.y + vp.y);
						dv7.z = 0.5 * (vt.z + vp.z);
//printf("A %d %d %d %.20g %.20g %.20g\n", idy, idi, n - 1, dx7.x, xt_s[idy].x, xp_s[idy].x); 
					}
					if(n == 7){				
						dx6.x = 0.5 * (xt_s[idy].x + xp_s[idy].x);
						dx6.y = 0.5 * (xt_s[idy].y + xp_s[idy].y);
						dx6.z = 0.5 * (xt_s[idy].z + xp_s[idy].z);

						dv6.x = 0.5 * (vt.x + vp.x);
						dv6.y = 0.5 * (vt.y + vp.y);
						dv6.z = 0.5 * (vt.z + vp.z);
//printf("A %d %d %d %.20g %.20g %.20g\n", idy, idi, n - 1, dx6.x, xt_s[idy].x, xp_s[idy].x); 
					}
					if(n == 6){				
						dx5.x = 0.5 * (xt_s[idy].x + xp_s[idy].x);
						dx5.y = 0.5 * (xt_s[idy].y + xp_s[idy].y);
						dx5.z = 0.5 * (xt_s[idy].z + xp_s[idy].z);

						dv5.x = 0.5 * (vt.x + vp.x);
						dv5.y = 0.5 * (vt.y + vp.y);
						dv5.z = 0.5 * (vt.z + vp.z);
//printf("A %d %d %d %.20g %.20g %.20g\n", idy, idi, n - 1, dx5.x, xt_s[idy].x, xp_s[idy].x); 
					}
					if(n == 5){				
						dx4.x = 0.5 * (xt_s[idy].x + xp_s[idy].x);
						dx4.y = 0.5 * (xt_s[idy].y + xp_s[idy].y);
						dx4.z = 0.5 * (xt_s[idy].z + xp_s[idy].z);

						dv4.x = 0.5 * (vt.x + vp.x);
						dv4.y = 0.5 * (vt.y + vp.y);
						dv4.z = 0.5 * (vt.z + vp.z);
//printf("A %d %d %d %.20g %.20g %.20g\n", idy, idi, n - 1, dx4.x, xt_s[idy].x, xp_s[idy].x); 
					}
					if(n == 4){				
						dx3.x = 0.5 * (xt_s[idy].x + xp_s[idy].x);
						dx3.y = 0.5 * (xt_s[idy].y + xp_s[idy].y);
						dx3.z = 0.5 * (xt_s[idy].z + xp_s[idy].z);

						dv3.x = 0.5 * (vt.x + vp.x);
						dv3.y = 0.5 * (vt.y + vp.y);
						dv3.z = 0.5 * (vt.z + vp.z);
//printf("A %d %d %d %.20g %.20g %.20g\n", idy, idi, n - 1, dx3.x, xt_s[idy].x, xp_s[idy].x); 
					}
					if(n == 3){				
						dx2.x = 0.5 * (xt_s[idy].x + xp_s[idy].x);
						dx2.y = 0.5 * (xt_s[idy].y + xp_s[idy].y);
						dx2.z = 0.5 * (xt_s[idy].z + xp_s[idy].z);

						dv2.x = 0.5 * (vt.x + vp.x);
						dv2.y = 0.5 * (vt.y + vp.y);
						dv2.z = 0.5 * (vt.z + vp.z);
//printf("A %d %d %d %.20g %.20g %.20g\n", idy, idi, n - 1, dx2.x, xt_s[idy].x, xp_s[idy].x); 
					}
					if(n == 2){				
						dx1.x = 0.5 * (xt_s[idy].x + xp_s[idy].x);
						dx1.y = 0.5 * (xt_s[idy].y + xp_s[idy].y);
						dx1.z = 0.5 * (xt_s[idy].z + xp_s[idy].z);

						dv1.x = 0.5 * (vt.x + vp.x);
						dv1.y = 0.5 * (vt.y + vp.y);
						dv1.z = 0.5 * (vt.z + vp.z);
//printf("A %d %d %d %.20g %.20g %.20g\n", idy, idi, n - 1, dx1.x, xt_s[idy].x, xp_s[idy].x); 
					}
					if(n == 1){				
						dx0.x = 0.5 * (xt_s[idy].x + xp_s[idy].x);
						dx0.y = 0.5 * (xt_s[idy].y + xp_s[idy].y);
						dx0.z = 0.5 * (xt_s[idy].z + xp_s[idy].z);

						dv0.x = 0.5 * (vt.x + vp.x);
						dv0.y = 0.5 * (vt.y + vp.y);
						dv0.z = 0.5 * (vt.z + vp.z);
//printf("A %d %d %d %.20g %.20g %.20g\n", idy, idi, n - 1, dx0.x, xt_s[idy].x, xp_s[idy].x); 
					}

					double ddt0 = 0.25 / (n*n);
					for(int j = n-1; j >= 1; --j){
						double ddt1 = 0.25 / (j*j);
						double t0 = 1.0 / (ddt1 - ddt0);
						double t1 = t0 * 0.25 / ((j+1)*(j+1));
						double t2 = t0 * ddt0;
				
						if(j == 7){
							dx6.x = t1 * dx7.x - t2 * dx6.x;
							dx6.y = t1 * dx7.y - t2 * dx6.y;
							dx6.z = t1 * dx7.z - t2 * dx6.z;

							dv6.x = t1 * dv7.x - t2 * dv6.x;
							dv6.y = t1 * dv7.y - t2 * dv6.y;
							dv6.z = t1 * dv7.z - t2 * dv6.z;
//printf("A %d %d %d %.20g %.20g\n", idy, idi, j - 1, dx6.x, dv6.x); 
						}
						if(j == 6){
							dx5.x = t1 * dx6.x - t2 * dx5.x;
							dx5.y = t1 * dx6.y - t2 * dx5.y;
							dx5.z = t1 * dx6.z - t2 * dx5.z;

							dv5.x = t1 * dv6.x - t2 * dv5.x;
							dv5.y = t1 * dv6.y - t2 * dv5.y;
							dv5.z = t1 * dv6.z - t2 * dv5.z;
//printf("A %d %d %d %.20g %.20g\n", idy, idi, j - 1, dx5.x, dv5.x); 
						}
						if(j == 5){
							dx4.x = t1 * dx5.x - t2 * dx4.x;
							dx4.y = t1 * dx5.y - t2 * dx4.y;
							dx4.z = t1 * dx5.z - t2 * dx4.z;

							dv4.x = t1 * dv5.x - t2 * dv4.x;
							dv4.y = t1 * dv5.y - t2 * dv4.y;
							dv4.z = t1 * dv5.z - t2 * dv4.z;
//printf("A %d %d %d %.20g %.20g\n", idy, idi, j - 1, dx4.x, dv4.x); 
						}
						if(j == 4){
							dx3.x = t1 * dx4.x - t2 * dx3.x;
							dx3.y = t1 * dx4.y - t2 * dx3.y;
							dx3.z = t1 * dx4.z - t2 * dx3.z;

							dv3.x = t1 * dv4.x - t2 * dv3.x;
							dv3.y = t1 * dv4.y - t2 * dv3.y;
							dv3.z = t1 * dv4.z - t2 * dv3.z;
//printf("A %d %d %d %.20g %.20g\n", idy, idi, j - 1, dx3.x, dv3.x); 
						}
						if(j == 3){
							dx2.x = t1 * dx3.x - t2 * dx2.x;
							dx2.y = t1 * dx3.y - t2 * dx2.y;
							dx2.z = t1 * dx3.z - t2 * dx2.z;

							dv2.x = t1 * dv3.x - t2 * dv2.x;
							dv2.y = t1 * dv3.y - t2 * dv2.y;
							dv2.z = t1 * dv3.z - t2 * dv2.z;
//printf("A %d %d %d %.20g %.20g\n", idy, idi, j - 1, dx2.x, dv2.x); 
						}
						if(j == 2){
							dx1.x = t1 * dx2.x - t2 * dx1.x;
							dx1.y = t1 * dx2.y - t2 * dx1.y;
							dx1.z = t1 * dx2.z - t2 * dx1.z;

							dv1.x = t1 * dv2.x - t2 * dv1.x;
							dv1.y = t1 * dv2.y - t2 * dv1.y;
							dv1.z = t1 * dv2.z - t2 * dv1.z;
//printf("A %d %d %d %.20g %.20g\n", idy, idi, j - 1, dx1.x, dv1.x); 
						}
						if(j == 1){
							dx0.x = t1 * dx1.x - t2 * dx0.x;
							dx0.y = t1 * dx1.y - t2 * dx0.y;
							dx0.z = t1 * dx1.z - t2 * dx0.z;

							dv0.x = t1 * dv1.x - t2 * dv0.x;
							dv0.y = t1 * dv1.y - t2 * dv0.y;
							dv0.z = t1 * dv1.z - t2 * dv0.z;
//printf("A %d %d %d %.20g %.20g\n", idy, idi, j - 1, dx0.x, dv0.x); 
						}
					}
					errorx = dx0.x * dx0.x * scalex.x;
					errorv = dv0.x * dv0.x * scalev.x;
					errorx = fmax(errorx, dx0.y * dx0.y * scalex.y);
					errorv = fmax(errorv, dv0.y * dv0.y * scalev.y);
					errorx = fmax(errorx, dx0.z * dx0.z * scalex.z);
					errorv = fmax(errorv, dv0.z * dv0.z * scalev.z);

//printf("dx %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", idy, idi, dx0.x, dx0.y, dx0.z, dv0.x, dv0.y, dv0.z);
//printf("scale %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", idy, idi, scalex.x, scalex.y, scalex.z, scalev.x, scalev.y, scalev.z);
					errorx = fmax(errorx, errorv);
//printf("error %d %d %.20g %d\n", idy, idi, errorx, n);
					if(errorx >= tol * tol){
						stop_s[0] = 0;
					}
					Ncol[0] = 0;
				}
				__syncthreads();
				if(stop_s[0] == 1 || sgnt * dt1 < 1.0e-6){
					if(Ne > 0){
						xt_s[idy].x = dx0.x;
						xt_s[idy].y = dx0.y;
						xt_s[idy].z = dx0.z;

						vt.x = dv0.x;
						vt.y = dv0.y;
						vt.z = dv0.z;
						if(n >= 2){
							xt_s[idy].x += dx1.x;
							xt_s[idy].y += dx1.y;
							xt_s[idy].z += dx1.z;
							vt.x += dv1.x;
							vt.y += dv1.y;
							vt.z += dv1.z;
						}
						if(n >= 3){
							xt_s[idy].x += dx2.x;
							xt_s[idy].y += dx2.y;
							xt_s[idy].z += dx2.z;
							vt.x += dv2.x;
							vt.y += dv2.y;
							vt.z += dv2.z;
						}
						if(n >= 4){
							xt_s[idy].x += dx3.x;
							xt_s[idy].y += dx3.y;
							xt_s[idy].z += dx3.z;
							vt.x += dv3.x;
							vt.y += dv3.y;
							vt.z += dv3.z;
						}
						if(n >= 5){
							xt_s[idy].x += dx4.x;
							xt_s[idy].y += dx4.y;
							xt_s[idy].z += dx4.z;
							vt.x += dv4.x;
							vt.y += dv4.y;
							vt.z += dv4.z;
						}
						if(n >= 6){
							xt_s[idy].x += dx5.x;
							xt_s[idy].y += dx5.y;
							xt_s[idy].z += dx5.z;
							vt.x += dv5.x;
							vt.y += dv5.y;
							vt.z += dv5.z;
						}
						if(n >= 7){
							xt_s[idy].x += dx6.x;
							xt_s[idy].y += dx6.y;
							xt_s[idy].z += dx6.z;
							vt.x += dv6.x;
							vt.y += dv6.y;
							vt.z += dv6.z;
						}
						if(n >= 8){
							xt_s[idy].x += dx7.x;
							xt_s[idy].y += dx7.y;
							xt_s[idy].z += dx7.z;
							vt.x += dv7.x;
							vt.y += dv7.y;
							vt.z += dv7.z;
						}
					}
					vt_s[idy] = vt;
					__syncthreads();
					for(int i = 0; i < Ne; ++i){
						double enct = 0.0;
						int jg = Encpairs_d[idi * NencMax + i].x;
						int j = Encpairs_d[NB + jg].y;

						encounter<1>(xt_s[idy], vt_s[idy], x4_s[idy], v4_s[idy], xt_s[j], vt_s[j], x4_s[j], v4_s[j], v4_s[idy].w, v4_s[j].w, 0.0, 0.0, dt1, idy, j, &test, Colpairs_s, Ncol[0], 0, enct, writeEncounters, writeEncountersRadius);
						//write Encounters to file
						if(enct > 0.0){
							int ne = atomicAdd(NWriteEnc_d, 1);
							if(ne >= MaxWriteEnc -1) ne = MaxWriteEnc -1;
							storeEncounters(xt_s, vt_s, idy, j, idi, jg, index_d, ne, writeEnc_d, time + t/0.01720209895, spin_d);
						}
					}
					__syncthreads();
					if(idy == 0) {
						for(int c = 0; c < Ncol[0]; ++c){
							int i = Colpairs_s[c].x;
							int j = Colpairs_s[c].y;
							if(xt_s[i].w >= 0 && xt_s[j].w >= 0){
								int nc = atomicAdd(Ncoll_d, 1);
								if(nc >= MaxColl -1) nc = MaxColl -1;
								collide(xt_s, vt_s, i, j, Encpairs2_d[start + i].x, Encpairs2_d[start + j].x, Msun, U_d, test, index_d, nc, Coll_d, time + t/0.01720209895, spin_d, rcritv_s, rcrit_d, Nst, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d);
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

					if(Ne > 0){
						x4_s[idy] = xt_s[idy];
						v4_s[idy] = vt;
//printf("update %d %d %d %.20g %.20g %.20g %.20g %.20g %.20g %g %g %d %d %d\n", idy, idx, idi, x4_s[idy].x, x4_s[idy].y, x4_s[idy].z, v4_s[idy].x, v4_s[idy].y, v4_s[idy].z, t, dt1, tt, ff, n);
					}
					f = 0;

					syncthreads();
					break;
				}
			} //end of n loop
			if(f == 0) break;
			__syncthreads();	
			dt1 *= 0.5;
		}//end of ff loop
		if(sgnt * t >= sgnt * dt){
			break;
		}

		__syncthreads();
	}//end of tt loop
	if(idy < N2){
		x4_d[idi] = xt_s[idy];
		v4_d[idi] = vt;
	}
}

/*
template <int NN, int Bl>
__global__ void BSA512_kernel(double4 *x4_d, double4 *v4_d, double4 *xold_d, double4 *vold_d, double4 *xp_d, double4 *vp_d, double4 *xt_d, double4 *vt_d, double *rcrit_d, double *rcritv_d, int *index_d, double3 *spin_d, int2 *Encpairs_d, int2 *Encpairs2_d, double3 *dx_d, double3 *dv_d, double dt, double Msun, double *U_d, int st, int NB, int NencMax, int *Ncoll_d, double *Coll_d, double time, int Nst, float4 *aelimits_d, int *aecount_d, int *enccount_d, long long *aecountT_d, long long *enccountT_d, int writeEncounters_d, double writeEncountersRadius_d, int *NWriteEnc_d, double *writeEnc_d){
	int idy = threadIdx.x;
	int idx = blockIdx.x;

	double dt1 = dt;
	double dt2, dt22;
	double t = 0.0;

	const double tol = 1.0e-12;

	__shared__ int stop_s[1];
	__shared__ int Ncol[1];
	__shared__ int2 Colpairs_s[MaxColl];

	double3 a;

	double3 dxj;
	double3 dvj;

	int sgnt = 1;

	int Ne; //number of pairs

	double3 scalex;
	double3  scalev;

	double errorx, errorv;
	double test;
	int writeEncounters = writeEncounters_d;
	double writeEncountersRadius = writeEncountersRadius_d;

	int idi;
	int si = Encpairs2_d[ (st+2) * NB + idx].y;
	int N2 = Encpairs2_d[si].y; //Number of bodies in  current BS simulation
	int start = Encpairs2_d[NB + si].y;
//printf("BS %d %d %d %d %d\n", idx, st, si, N2, NB);
	if(dt < 0.0) sgnt = -1;

	for(int tt = 0; tt < 1000; ++tt){
		volatile int f = 1;
		__syncthreads();
		for(int ff = 0; ff < 1e6; ++ff){
			for(int n = 1; n <= 8; ++n){
				if(idy == 0) stop_s[0] = 1;
				__syncthreads();
				dt2 = dt1 / (2.0 * n);
				dt22 = dt2 * 2.0;

				for(int i = 0; i < NN; i += Bl){
					if(idy + i < N2){
						idi = Encpairs2_d[start + idy + i].x;
						Ne = Encpairs_d[idi].y;
					}
					else{
						idi = 0;
						Ne = 0;
					}
					a.x = 0.0;
					a.y = 0.0;
					a.z = 0.0;
					for(int ii = 0; ii < Ne; ++ii){
						int j = Encpairs_d[idi * NencMax + ii].x;
//printf("BS2 %d %d %d %d %d %d %d %d %d %.20g\n", idx, idy, st, idi, index_d[idi], j, Ne, N2, ff, xold_d[idi].x);
						accEnc(xold_d[idi], xold_d[j], a, rcritv_d[idi], rcritv_d[j], test, idi, j);
					}
					if(Ne > 0){
						accEncSun(xold_d[idi], a, ksq * Msun);

						xp_d[idi].x = xold_d[idi].x + dt2 * vold_d[idi].x;
						xp_d[idi].y = xold_d[idi].y + dt2 * vold_d[idi].y;
						xp_d[idi].z = xold_d[idi].z + dt2 * vold_d[idi].z;
						xp_d[idi].w = xold_d[idi].w;

						vp_d[idi].x = vold_d[idi].x + dt2 * a.x;
						vp_d[idi].y = vold_d[idi].y + dt2 * a.y;
						vp_d[idi].z = vold_d[idi].z + dt2 * a.z;
						vp_d[idi].w = vold_d[idi].w;
					}
				}
				__syncthreads();
				for(int i = 0; i < NN; i += Bl){
					if(idy + i < N2){
						idi = Encpairs2_d[start + idy + i].x;
						Ne = Encpairs_d[idi].y;
					}
					else{
						idi = 0;
						Ne = 0;
					}
					a.x = 0.0;
					a.y = 0.0;
					a.z = 0.0;
					for(int ii = 0; ii < Ne; ++ii){
						int j = Encpairs_d[idi * NencMax + ii].x;
						accEnc(xp_d[idi], xp_d[j], a, rcritv_d[idi], rcritv_d[j], test, idi, j);
					}
					if(Ne > 0){
						accEncSun(xp_d[idi], a, ksq * Msun);

						xt_d[idi].x = xold_d[idi].x + dt22 * vp_d[idi].x;
						xt_d[idi].y = xold_d[idi].y + dt22 * vp_d[idi].y;
						xt_d[idi].z = xold_d[idi].z + dt22 * vp_d[idi].z;
						xt_d[idi].w = xold_d[idi].w;

						vt_d[idi].x = vold_d[idi].x + dt22 * a.x;
						vt_d[idi].y = vold_d[idi].y + dt22 * a.y;
						vt_d[idi].z = vold_d[idi].z + dt22 * a.z;
						vt_d[idi].w = vold_d[idi].w;
		//printf("a0 %d %.20g %.20g %.20g %.20g %d\n", idi, xt_d[idi].x, vt_d[idi].x, xold_d[idi].x, vp_d[idi].x, n);
					}
				}
				__syncthreads();

				for(int m = 2; m <= n; ++m){
					for(int i = 0; i < NN; i += Bl){
						if(idy + i < N2){
							idi = Encpairs2_d[start + idy + i].x;
							Ne = Encpairs_d[idi].y;
						}
						else{
							idi = 0;
							Ne = 0;
						}
						a.x = 0.0;
						a.y = 0.0;
						a.z = 0.0;
						for(int ii = 0; ii < Ne; ++ii){
							int j = Encpairs_d[idi * NencMax + ii].x;
							accEnc(xt_d[idi], xt_d[j], a, rcritv_d[idi], rcritv_d[j], test, idi, j);
						}
						if(Ne > 0){
							accEncSun(xt_d[idi], a, ksq * Msun);

							xp_d[idi].x += dt22 * vt_d[idi].x;
							xp_d[idi].y += dt22 * vt_d[idi].y;
							xp_d[idi].z += dt22 * vt_d[idi].z;

							vp_d[idi].x += dt22 * a.x;
							vp_d[idi].y += dt22 * a.y;
							vp_d[idi].z += dt22 * a.z;
						}
					}
					__syncthreads();
					for(int i = 0; i < NN; i += Bl){
						if(idy + i < N2){
							idi = Encpairs2_d[start + idy + i].x;
							Ne = Encpairs_d[idi].y;
						}
						else{
							idi = 0;
							Ne = 0;
						}
						a.x = 0.0;
						a.y = 0.0;
						a.z = 0.0;
						for(int ii = 0; ii < Ne; ++ii){
							int j = Encpairs_d[idi * NencMax + ii].x;
							accEnc(xp_d[idi], xp_d[j], a, rcritv_d[idi], rcritv_d[j], test, idi, j);
						}
						if(Ne > 0){
							accEncSun(xp_d[idi], a, ksq * Msun);

							xt_d[idi].x += dt22 * vp_d[idi].x;
							xt_d[idi].y += dt22 * vp_d[idi].y;
							xt_d[idi].z += dt22 * vp_d[idi].z;

							vt_d[idi].x += dt22 * a.x;
							vt_d[idi].y += dt22 * a.y;
							vt_d[idi].z += dt22 * a.z;
		//printf("am %d %.20g %.20g %.20g %d\n", idi, xt_d[idi].x, vt_d[idi].x, vp_d[idi].x, n);
						}
					}
					__syncthreads();
				}//end of m loop
				for(int i = 0; i < NN; i += Bl){
					if(idy + i < N2){
						idi = Encpairs2_d[start + idy + i].x;
						Ne = Encpairs_d[idi].y;
					}
					else{
						idi = 0;
						Ne = 0;
					}
					a.x = 0.0;
					a.y = 0.0;
					a.z = 0.0;
					for(int ii = 0; ii < Ne; ++ii){
						int j = Encpairs_d[idi * NencMax + ii].x;
						accEnc(xt_d[idi], xt_d[j], a, rcritv_d[idi], rcritv_d[j], test, idi, j);
					}
					if(Ne > 0){
						accEncSun(xt_d[idi], a, ksq * Msun);

						xp_d[idi].x += dt2 * vt_d[idi].x;
						xp_d[idi].y += dt2 * vt_d[idi].y;
						xp_d[idi].z += dt2 * vt_d[idi].z;

						vp_d[idi].x += dt2 * a.x;
						vp_d[idi].y += dt2 * a.y;
						vp_d[idi].z += dt2 * a.z;

						dxj.x = 0.5 * (xt_d[idi].x + xp_d[idi].x);
						dxj.y = 0.5 * (xt_d[idi].y + xp_d[idi].y);
						dxj.z = 0.5 * (xt_d[idi].z + xp_d[idi].z);

						dvj.x = 0.5 * (vt_d[idi].x + vp_d[idi].x);
						dvj.y = 0.5 * (vt_d[idi].y + vp_d[idi].y);
						dvj.z = 0.5 * (vt_d[idi].z + vp_d[idi].z);

						dx_d[(n - 1) * NB + idi] = dxj;
						dv_d[(n - 1) * NB + idi] = dvj;

	//printf("A %d %d %.20g %.20g %.20g\n", idi, n - 1, dxj.x, xt_d[idi].x, xp_d[idi].x); 

						double ddt0 = 0.25 / (n*n);
						for(int j = n-1; j >= 1; --j){
							double ddt1 = 0.25 / (j*j);
							double t0 = 1.0 / (ddt1 - ddt0);
							double t1 = t0 * 0.25 / ((j+1)*(j+1));
							double t2 = t0 * ddt0;

							dxj.x = t1 * dxj.x - t2 * dx_d[(j - 1) * NB + idi].x;
							dxj.y = t1 * dxj.y - t2 * dx_d[(j - 1) * NB + idi].y;
							dxj.z = t1 * dxj.z - t2 * dx_d[(j - 1) * NB + idi].z;

							dvj.x = t1 * dvj.x - t2 * dv_d[(j - 1) * NB + idi].x;
							dvj.y = t1 * dvj.y - t2 * dv_d[(j - 1) * NB + idi].y;
							dvj.z = t1 * dvj.z - t2 * dv_d[(j - 1) * NB + idi].z;


							dx_d[(j - 1) * NB + idi] = dxj;
							dv_d[(j - 1) * NB + idi] = dvj;
					
	//printf("A %d %d %.20g %.20g\n", idi, j - 1, dx_d[(j - 1) * NB + idi].x, dv_d[(j - 1) * NB + idi].x); 
						}

						dxj = dx_d[0 * NB + idi];
						dvj = dv_d[0 * NB + idi];

						scalex.x = 1.0 / (xold_d[idi].x * xold_d[idi].x + 1.0e-20);
						scalex.y = 1.0 / (xold_d[idi].y * xold_d[idi].y + 1.0e-20);
						scalex.z = 1.0 / (xold_d[idi].z * xold_d[idi].z + 1.0e-20);

						scalev.x = 1.0 / (vold_d[idi].x * vold_d[idi].x + 1.0e-20);
						scalev.y = 1.0 / (vold_d[idi].y * vold_d[idi].y + 1.0e-20);
						scalev.z = 1.0 / (vold_d[idi].z * vold_d[idi].z + 1.0e-20);

						errorx = dxj.x * dxj.x * scalex.x;
						errorv = dvj.x * dvj.x * scalev.x;
						errorx = fmax(errorx, dxj.y * dxj.y * scalex.y);
						errorv = fmax(errorv, dvj.y * dvj.y * scalev.y);
						errorx = fmax(errorx, dxj.z * dxj.z * scalex.z);
						errorv = fmax(errorv, dvj.z * dvj.z * scalev.z);

//printf("dx %d %.20g %.20g %.20g %.20g %.20g %.20g\n", idi, dxj.x, dxj.y, dxj.z, dvj.x, dvj.y, dvj.z);
//printf("scale %d %.20g %.20g %.20g %.20g %.20g %.20g\n", idi, scalex.x, scalex.y, scalex.z, scalev.x, scalev.y, scalev.z);
						errorx = fmax(errorx, errorv);
//printf("error %d %.20g %d\n", idi, errorx, n);
						if(errorx >= tol * tol){
							stop_s[0] = 0;
						}
						Ncol[0] = 0;
					}
				}
				__syncthreads();
				if(stop_s[0] == 1 || sgnt * dt1 < 1.0e-6){	
					for(int i = 0; i < NN; i += Bl){
						if(idy + i < N2){
							idi = Encpairs2_d[start + idy + i].x;
							Ne = Encpairs_d[idi].y;
						}
						else{
							idi = 0;
							Ne = 0;
						}
						if(Ne > 0){
							double4 xt, vt;

							xt.x = dx_d[0 * NB + idi].x;
							xt.y = dx_d[0 * NB + idi].y;
							xt.z = dx_d[0 * NB + idi].z;
							xt.w = xold_d[idi].w;

							vt.x = dv_d[0 * NB + idi].x;
							vt.y = dv_d[0 * NB + idi].y;
							vt.z = dv_d[0 * NB + idi].z;
							vt.w = vold_d[idi].w;
		//printf("xt %d %d %.20g %.20g %d\n", idi, 0, xt.x, vt.x, n);
							for(int j = 1; j < n; ++j){
								xt.x += dx_d[j * NB + idi].x;
								xt.y += dx_d[j * NB + idi].y;
								xt.z += dx_d[j * NB + idi].z;

								vt.x += dv_d[j * NB + idi].x;
								vt.y += dv_d[j * NB + idi].y;
								vt.z += dv_d[j * NB + idi].z;
		//printf("xt %d %d %.20g %.20g\n", idi, j, xt.x, vt.x);
							}
							xt_d[idi] = xt;
							vt_d[idi] = vt;
						}
					}
					__syncthreads();
					for(int i = 0; i < NN; i += Bl){
						if(idy + i < N2){
							idi = Encpairs2_d[start + idy + i].x;
							Ne = Encpairs_d[idi].y;
						}
						else{
							idi = 0;
							Ne = 0;
						}
						for(int ii = 0; ii < Ne; ++ii){
							double enct = 0.0;
							int j = Encpairs_d[idi * NencMax + ii].x;
							encounter<1>(xt_d[idi], vt_d[idi], xold_d[idi], vold_d[idi], xt_d[j], vt_d[j], xold_d[j], vold_d[j], vold_d[idi].w, vold_d[j].w, 0.0, 0.0, dt1, idi, j, &test, Colpairs_s, Ncol[0], 0, enct, writeEncounters, writeEncountersRadius);
							//write Encounters to file
							if(enct > 0.0){
								int ne = atomicAdd(NWriteEnc_d, 1);
								if(ne >= MaxWriteEnc -1) ne = MaxWriteEnc -1;
								storeEncounters(xt_d, vt_d, idi, j, idi, j, index_d, ne, writeEnc_d, time + t/0.01720209895, spin_d);
							}
						}
					}
					__syncthreads();
					if(idy == 0) {
						for(int c = 0; c < Ncol[0]; ++c){
							int i = Colpairs_s[c].x;
							int j = Colpairs_s[c].y;
							if(xt_d[i].w >= 0 && xt_d[j].w >= 0){
								int nc = atomicAdd(Ncoll_d, 1);
								if(nc >= MaxColl -1) nc = MaxColl -1;
								collide(xt_d, vt_d, i, j, Encpairs2_d[start + i].x, Encpairs2_d[start + j].x, Msun, U_d, test, index_d, nc, Coll_d, time + t/0.01720209895, spin_d, rcritv_d, rcrit_d, Nst, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d);
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

					for(int i = 0; i < NN; i += Bl){
						if(idy + i < N2){
							idi = Encpairs2_d[start + idy + i].x;
							Ne = Encpairs_d[idi].y;
						}
						else{
							idi = 0;
							Ne = 0;
						}
						if(Ne > 0){
							xold_d[idi] = xt_d[idi];
							vold_d[idi] = vt_d[idi];
//printf("update %d %.20g %.20g %.20g %.20g %.20g %.20g %g %g\n", idi, xold_d[idi].x, xold_d[idi].y, xold_d[idi].z, vold_d[idi].x, vold_d[idi].y, vold_d[idi].z, t, dt1);
						}
					}

					f = 0;
					__syncthreads();
					break;
				}
			}//end of n loop
			if(f == 0) break;
			__syncthreads();
			dt1 *= 0.5;
		} //end of ff loop	
		if(sgnt * t >= sgnt * dt){
			break;
		}

		__syncthreads();
	}//end if tt loop
	for(int i = 0; i < NN; i += Bl){
		if(idy + i < N2){
			idi = Encpairs2_d[start + idy + i].x;
			Ne = Encpairs_d[idi].y;
		}
		else{
			idi = 0;
			Ne = 0;
		}
		if(Ne > 0){
			x4_d[idi] = xold_d[idi];
			v4_d[idi] = vold_d[idi];
		}
	}
}


template <int E>
__global__ void BSAcc_kernel(double4 *x4_d, double4 *v4_d, double4 *xA_d, double4 *vA_d, double4 *xB_d, double4 *vB_d, double *rcritv_d, int2 *Encpairs_d, int2 *Encpairs2_d, double *dt1_d, double Msun, int st, int NB, int NencMax, int *BSAstop_d, int n){

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = blockIdx.y;

	int si = Encpairs2_d[ (st+2) * NB + idx].y;
	int N2 = Encpairs2_d[si].y; //Number of bodies in  current BS simulation
	int start = Encpairs2_d[NB + si].y;
	if(id < N2){
		int idi = Encpairs2_d[start + id].x;
		int Ne;
		if(E == 0 && n == 1){
			Ne = Encpairs_d[idi].y; //number of pairs
			Encpairs_d[idi + 6 * NB].y = Ne;
		}
		else Ne = Encpairs_d[idi + 6 * NB].y; //number of pairs
		if(Ne > 0){
			double dt1 = dt1_d[idi];
			double dt2 = dt1 / (2.0 * n);
			if(E == 1 || E == 2) dt2 *= 2.0;

			double4 xAi = xA_d[idi];
			double4 vAi = vA_d[idi];
			double rcritvi = rcritv_d[idi];

			double3 a = {0.0, 0.0, 0.0};
			double test;

			for(int i = 0; i < Ne; ++i){	
				int j = Encpairs_d[idi * NencMax + i].x;
//if(n == 1 && E == 0)  printf("%d %d %d %d %d %g\n", idx, idi, Ne, j, n, dt1);
				double4 xAj = xA_d[j];
				double rcritvj = rcritv_d[j];

				accEnc(xAi, xAj, a, rcritvi, rcritvj, test, idi, j);		
		
			}	
			accEncSun(xAi, a, ksq * Msun);
			double4 x4B;
			double4 v4B;

			if(E == 0){
				//here xA = x4, and xB = xp
				x4B.x = xAi.x + dt2 * vAi.x;
				x4B.y = xAi.y + dt2 * vAi.y;
				x4B.z = xAi.z + dt2 * vAi.z;
				x4B.w = xAi.w;
				v4B.x = vAi.x + dt2 * a.x;
				v4B.y = vAi.y + dt2 * a.y;
				v4B.z = vAi.z + dt2 * a.z;
				v4B.w = vAi.w;
//if(Ne == 1) printf("a %d %.20g %.20g %.20g %.20g %d\n", idi, x4B.x, v4B.x, xAi.x, vAi.x, n);
				Encpairs_d[idi + 5 * NB].y = 0; //set accept condition
			}
			if(E == 1){
				//here xA = xp, and xB = xt
				double4 x4i = x4_d[idi];
				double4 v4i = v4_d[idi];
				x4B.x = x4i.x + dt2 * vAi.x;
				x4B.y = x4i.y + dt2 * vAi.y;
				x4B.z = x4i.z + dt2 * vAi.z;
				x4B.w = x4i.w;

				v4B.x = v4i.x + dt2 * a.x;
				v4B.y = v4i.y + dt2 * a.y;
				v4B.z = v4i.z + dt2 * a.z;
				v4B.w = v4i.w;
			}
			if(E == 2 || E == 3){
				x4B = xB_d[idi];
				v4B = vB_d[idi];
				x4B.x += dt2 * vAi.x;
				x4B.y += dt2 * vAi.y;
				x4B.z += dt2 * vAi.z;

				v4B.x += dt2 * a.x;
				v4B.y += dt2 * a.y;
				v4B.z += dt2 * a.z;
			}

			xB_d[idi] = x4B;
			vB_d[idi] = v4B;
		}		
	}
	if(E == 0 && n == 1 && id == 0) BSAstop_d[0] = 1;
}

__global__ void BSError_kernel(double4 *x4_d, double4 *v4_d, double4 *xp_d, double4 *vp_d, double4 *xt_d, double4 *vt_d, double3 *dx_d, double3 *dv_d, int2 *Encpairs_d, int2 *Encpairs2_d, int st, int NB, int n){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = blockIdx.y;

	const double tol = 1.0e-12;

	int si = Encpairs2_d[ (st+2) * NB + idx].y;
	int N2 = Encpairs2_d[si].y; //Number of bodies in  current BS simulation
	int start = Encpairs2_d[NB + si].y;

	if(id < N2){
		int idi = Encpairs2_d[start + id].x;
		int Ne = Encpairs_d[idi + 6 * NB].y; //number of pairs
		if(Ne > 0){
			int gi = Encpairs_d[idi + NB].y; //group index

			double4 x;
			double4 v;
			double3 dxj; //this is dx[n-1]
			double3 dvj;
			x = x4_d[idi];
			v = v4_d[idi];
	
			double3 scalex;
			double3 scalev;
			scalex.x = 1.0 / (x.x * x.x + 1.0e-20);
			scalex.y = 1.0 / (x.y * x.y + 1.0e-20);
			scalex.z = 1.0 / (x.z * x.z + 1.0e-20);
			scalev.x = 1.0 / (v.x * v.x + 1.0e-20);
			scalev.y = 1.0 / (v.y * v.y + 1.0e-20);
			scalev.z = 1.0 / (v.z * v.z + 1.0e-20);


			dxj.x = 0.5 * (xt_d[idi].x + xp_d[idi].x);
			dx_d[(n - 1) * NB + idi].x = dxj.x;
			dxj.y = 0.5 * (xt_d[idi].y + xp_d[idi].y);
			dx_d[(n - 1) * NB + idi].y = dxj.y;
			dxj.z = 0.5 * (xt_d[idi].z + xp_d[idi].z);
			dx_d[(n - 1) * NB + idi].z = dxj.z;
			dvj.x = 0.5 * (vt_d[idi].x + vp_d[idi].x);
			dv_d[(n - 1) * NB + idi].x = dvj.x;
			dvj.y = 0.5 * (vt_d[idi].y + vp_d[idi].y);
			dv_d[(n - 1) * NB + idi].y = dvj.y;
			dvj.z = 0.5 * (vt_d[idi].z + vp_d[idi].z);
			dv_d[(n - 1) * NB + idi].z = dvj.z;
//printf("xp %d %.20g %.20g %.20g %.20g %.20g %.20g %d\n", idi, xp_d[idi].x, xp_d[idi].y, xp_d[idi].z, vp_d[idi].x, vp_d[idi].y, vp_d[idi].z, n);
//printf("xt %d %.20g %.20g %.20g %.20g %.20g %.20g %d\n", idi, xt_d[idi].x, xt_d[idi].y, xt_d[idi].z, vt_d[idi].x, vt_d[idi].y, vt_d[idi].z, n);

			double ddt0 = 0.25 / (n * n);
			double dj1;

			for(int j = n - 1; j >= 1; --j){
				double ddt1 = 0.25 / (j * j);
				double t0 = 1.0 / (ddt1 - ddt0);
				double t1 = t0 * 0.25 / ((j+1) * (j+1));
				double t2 = t0 * ddt0;
				
				dj1 = dx_d[(j - 1) * NB + idi].x;
				dxj.x = t1 * dxj.x - t2 * dj1;
				dx_d[(j - 1) * NB + idi].x = dxj.x;
				dj1 = dx_d[(j - 1) * NB + idi].y;
				dxj.y = t1 * dxj.y - t2 * dj1;
				dx_d[(j - 1) * NB + idi].y = dxj.y;
				dj1 = dx_d[(j - 1) * NB + idi].z;
				dxj.z = t1 * dxj.z - t2 * dj1;
				dx_d[(j - 1) * NB + idi].z = dxj.z;
				dj1 = dv_d[(j - 1) * NB + idi].x;
				dvj.x = t1 * dvj.x - t2 * dj1;
				dv_d[(j - 1) * NB + idi].x = dvj.x;
				dj1 = dv_d[(j - 1) * NB + idi].y;
				dvj.y = t1 * dvj.y - t2 * dj1;
				dv_d[(j - 1) * NB + idi].y = dvj.y;
				dj1 = dv_d[(j - 1) * NB + idi].z;
				dvj.z = t1 * dvj.z - t2 * dj1;
				dv_d[(j - 1) * NB + idi].z = dvj.z;
			}
			double error = dxj.x * dxj.x * scalex.x;
			error = fmax(error, dxj.y * dxj.y * scalex.y);
			error = fmax(error, dxj.z * dxj.z * scalex.z);
			error = fmax(error, dvj.x * dvj.x * scalev.x);
			error = fmax(error, dvj.y * dvj.y * scalev.y);
			error = fmax(error, dvj.z * dvj.z * scalev.z);
//printf("dx %d %d %.20g %.20g %.20g %.20g %.20g %.20g %d\n", idx, idi, dxj.x, dxj.y, dxj.z, dvj.x, dvj.y, dvj.z, n);
//printf("scale %d %d %.20g %.20g %.20g %.20g %.20g %.20g %d\n", idx, idi, scalex.x, scalex.y, scalex.z, scalev.x, scalev.y, scalev.z, n);
//printf("error %d %d %d %.20g\n", idx, idi, gi, error);
			if(error >= tol * tol){
				//dont accept BS step
				Encpairs_d[gi + 5 * NB].y = 1; //Accept
			}
		}
	}
}


__global__ void BSAccept_kernel(double4 *xt_d, double4 *vt_d, double3 *dx_d, double3 *dv_d, int2 *Encpairs_d, int2 *Encpairs2_d, double *dt1_d, int st, int NB, int n){

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = blockIdx.y;

	int si = Encpairs2_d[ (st+2) * NB + idx].y;
	int N2 = Encpairs2_d[si].y; //Number of bodies in  current BS simulation
	int start = Encpairs2_d[NB + si].y;

	if(id < N2){
		int idi = Encpairs2_d[start + id].x;
		int Ne = Encpairs_d[idi + 6 * NB].y; //group index
		if(Ne > 0){
			int gi = Encpairs_d[idi + NB].y; //group index
			int accept = Encpairs_d[gi + 5 * NB].y;
			double3 xt;
			double3 vt;
			double dt1 = dt1_d[idi];
			int sgnt = 1;
			if(dt1 < 0.0) sgnt = -1;
			if(accept == 0 || sgnt * dt1 < 1.0e-6){
				xt.x = dx_d[idi].x;
				xt.y = dx_d[idi].y;
				xt.z = dx_d[idi].z;
				vt.x = dv_d[idi].x;
				vt.y = dv_d[idi].y;
				vt.z = dv_d[idi].z;

				for(int j = 1; j < n; ++j){
					xt.x += dx_d[j * NB + idi].x;
					xt.y += dx_d[j * NB + idi].y;
					xt.z += dx_d[j * NB + idi].z;
					vt.x += dv_d[j * NB + idi].x;
					vt.y += dv_d[j * NB + idi].y;
					vt.z += dv_d[j * NB + idi].z;
				}
				xt_d[idi].x = xt.x;
				xt_d[idi].y = xt.y;
				xt_d[idi].z = xt.z;
				vt_d[idi].x = vt.x;
				vt_d[idi].y = vt.y;
				vt_d[idi].z = vt.z;
			}
		}
	}
}


__global__ void BSUpdate_kernel(double4 *xold_d, double4 *vold_d, double4 *x4_d, double4 *v4_d, double4 *xt_d, double4 *vt_d, int2 *Encpairs_d, int2 *Encpairs2_d, int *BSAstop_d, double *dt1_d, double *t1_d, double dt, int st, int NB, int n){

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = blockIdx.y;	

	int si = Encpairs2_d[ (st+2) * NB + idx].y;
	int N2 = Encpairs2_d[si].y; //Number of bodies in  current BS simulation
	int start = Encpairs2_d[NB + si].y;

	if(id < N2){
		int idi = Encpairs2_d[start + id].x;
		int Ne = Encpairs_d[idi].y; //number of pairs
		int Ne1 = Encpairs_d[idi + 6 * NB].y; //number of pairs
		if(Ne > 0){
			int gi = Encpairs_d[idi + NB].y; //group index
			int accept = Encpairs_d[gi + 5 * NB].y;
			double dt1 = dt1_d[idi];
			double t1 = t1_d[idi];
			int sgnt = 1;
			if(dt < 0.0) sgnt= -1;
//printf("accept %d %d %d %d %.20g %.20g\n", idx, idi, gi, accept, dt1, t1);

			if((accept == 0 || sgnt * dt1 < 1.0e-6) && Ne1 > 0){
				
				t1 += dt1;
				if(n >= 8) dt1 *= 0.55;
				if(n < 7) dt1 *= 1.3;
				if(sgnt * dt1 > sgnt * dt) dt1 = dt;
				if(sgnt * (t1 + dt1) > sgnt * dt) dt1 = dt - t1;
				if(sgnt * dt1 < 1.0e-7) dt1 = sgnt * 1.0e-7;

				xold_d[idi] = xt_d[idi];
				vold_d[idi] = vt_d[idi];
//printf("update %d %d %.20g %.20g %.20g %.20g %.20g %.20g %g %g %d\n", idx, idi, xold_d[idi].x, xold_d[idi].y, xold_d[idi].z, vold_d[idi].x, vold_d[idi].y, vold_d[idi].z, t1, dt1, n);
				dt1_d[idi] = dt1;
				t1_d[idi] = t1;
				Encpairs_d[idi + 6 * NB].y = 0;
			}
			else{
				if(n == 8 && Ne1 > 0){
					dt1_d[idi] = 0.5 * dt1;
//printf("continue %d %d %g %g\n", idx, idi, t1_d[idi], dt1_d[idi]);
				}
			}
			if(sgnt * t1 >= sgnt * dt){
				//BS step finished
				Encpairs_d[idi].y = 0;
				x4_d[idi] = xt_d[idi];
				v4_d[idi] = vt_d[idi];
//printf("finished %d %.20g %.20g %.20g %.20g %.20g %.20g\n", idi, x4_d[idi].x, x4_d[idi].y, x4_d[idi].z, v4_d[idi].x, v4_d[idi].y, v4_d[idi].z);
			}
			else{
//printf("not finished %d %d\n", idx, idi);
				BSAstop_d[0] = 0;
			}
		}
	}
}

__global__ void BSA_setdt_kernel(double *dt1_d, double *t1_d, double dt, int N){

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < N){
		dt1_d[id] = dt;
		t1_d[id] = 0.0;
	}
}

__host__ void Data::BSACall(int st, int b, int Nm){

	int Nt = 32;
	int Nb = (b + Nt - 1) / Nt;
	BSA_setdt_kernel <<< (N_h[0] + 255) / 256, 256 >>> (dt1_d, t1_d, dt_h[0], N_h[0]);
	for(int f = 0; f < 10000; ++f){
		for(int n = 1; n <= 8; ++n){
			BSAcc_kernel < 0 > <<< dim3(Nb, Nm, 1), dim3(Nt, 1, 1) >>> (xold_d, vold_d, xold_d, vold_d, xp_d, vp_d, rcritv_d, Encpairs_d, Encpairs2_d, dt1_d, Msun_h[0], st, NB[0], P.NencMax, BSAstop_d, n);
			BSAcc_kernel < 1 > <<< dim3(Nb, Nm, 1), dim3(Nt, 1, 1) >>> (xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, rcritv_d, Encpairs_d, Encpairs2_d, dt1_d, Msun_h[0], st, NB[0], P.NencMax, BSAstop_d, n);
			for(int m = 2; m <= n; ++m){
				BSAcc_kernel < 2 > <<< dim3(Nb, Nm, 1), dim3(Nt, 1, 1) >>> (xold_d, vold_d, xt_d, vt_d, xp_d, vp_d, rcritv_d, Encpairs_d, Encpairs2_d, dt1_d, Msun_h[0], st, NB[0], P.NencMax, BSAstop_d, n);
				BSAcc_kernel < 2 > <<< dim3(Nb, Nm, 1), dim3(Nt, 1, 1) >>> (xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, rcritv_d, Encpairs_d, Encpairs2_d, dt1_d, Msun_h[0], st, NB[0], P.NencMax, BSAstop_d, n);
			}
			BSAcc_kernel < 3 > <<< dim3(Nb, Nm, 1), dim3(Nt, 1, 1) >>> (xold_d, vold_d, xt_d, vt_d, xp_d, vp_d, rcritv_d, Encpairs_d, Encpairs2_d, dt1_d, Msun_h[0], st, NB[0], P.NencMax, BSAstop_d, n);
			BSError_kernel <<< dim3(Nb, Nm, 1), dim3(Nt, 1, 1) >>> (xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, dx_d, dv_d, Encpairs_d, Encpairs2_d, st, NB[0], n);
			BSAccept_kernel <<< dim3(Nb, Nm, 1), dim3(Nt, 1, 1) >>> (xt_d, vt_d, dx_d, dv_d, Encpairs_d, Encpairs2_d, dt1_d, st, NB[0], n);
			BSUpdate_kernel <<< dim3(Nb, Nm, 1), dim3(Nt, 1, 1) >>> (xold_d, vold_d, x4_d, v4_d, xt_d, vt_d, Encpairs_d, Encpairs2_d, BSAstop_d, dt1_d, t1_d, dt_h[0], st, NB[0], n);
			cudaMemcpy(BSAstop_h, BSAstop_d, sizeof(int), cudaMemcpyDeviceToHost);
			if(BSAstop_h[0] == 1) break; 
		}
		if(BSAstop_h[0] == 1) break; 
	}
}
*/
/*
__global__ void BSACall_kernel(double4 *xold_d, double4 *vold_d, double4 *x4_d, double4 *v4_d, double4 *xp_d, double4 *vp_d,double4 *xt_d, double4 *vt_d, double3 *dx_d, double3 *dv_d, double *rcritv_d, int2 *Encpairs_d, double *dt1_d, double *t1_d, int *BSAstop_d, double dt, double Msun, int NB, int N){

	int Nt = 32;
	int Nb = (N + Nt - 1) / Nt;

	BSA_setdt_kernel <<< Nb, Nt >>> (dt1_d, t1_d, dt, N);
	for(int f = 0; f < 10000; ++f){
//printf("f %d\n", f);
		for(int n = 1; n <= 8; ++n){
			BSAcc_kernel < 0 > <<< Nb, Nt >>> (xold_d, vold_d, xold_d, vold_d, xp_d, vp_d, rcritv_d, Encpairs_d, dt1_d, Msun, N, NB, BSAstop_d, n);
			BSAcc_kernel < 1 > <<< Nb, Nt >>> (xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, rcritv_d, Encpairs_d, dt1_d, Msun, N, NB, BSAstop_d, n);
			for(int m = 2; m <= n; ++m){
				BSAcc_kernel < 2 > <<< Nb, Nt >>> (xold_d, vold_d, xt_d, vt_d, xp_d, vp_d, rcritv_d, Encpairs_d, dt1_d, Msun, N, NB, BSAstop_d, n);
				BSAcc_kernel < 2 > <<< Nb, Nt >>> (xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, rcritv_d, Encpairs_d, dt1_d, Msun, N, NB, BSAstop_d, n);
			}
			BSAcc_kernel < 3 > <<< Nt, Nb >>> (xold_d, vold_d, xt_d, vt_d, xp_d, vp_d, rcritv_d, Encpairs_d, dt1_d, Msun, N, NB, BSAstop_d, n);
			BSError_kernel <<< dim3(Nt, 6, 1), dim3(Nb, 1, 1) >>> (xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, dx_d, dv_d, Encpairs_d, N, NB, n);
			BSAccept_kernel <<< Nt, Nb >>> (xt_d, vt_d, dx_d, dv_d, Encpairs_d, dt1_d, N, NB, n);
			BSUpdate_kernel <<< Nt, Nb >>> (xold_d, vold_d, x4_d, v4_d, xt_d, vt_d, Encpairs_d, BSAstop_d, dt1_d, t1_d, dt, N, NB, n);
			cudaDeviceSynchronize();
			if(BSAstop_d[0] == 1) break; 
		}
		if(BSAstop_d[0] == 1) break; 
	}
}
*/

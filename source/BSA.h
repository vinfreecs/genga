template <int NN>
__global__ void BSA_kernel(curandState *random_d, double4 *x4_d, double4 *v4_d, double4 *xold_d, double4 *vold_d, double *rcrit_d, double *rcritv_d, int *index_d, double4 *spin_d, double3 *love_d, int *createFlag_d, int2 *Encpairs_d, int2 *Encpairs2_d, const double dt, const double Msun, double *U_d, const int st, const int NT, const int NconstT, const int NencMax, int *BSstop_d, int *Ncoll_d, double *Coll_d, const double time, float4 *aelimits_d, unsigned int *aecount_d, unsigned int *enccount_d, unsigned long long *aecountT_d, unsigned long long *enccountT_d, int *NWriteEnc_d, double *writeEnc_d, const int UseGR, const double MinMass, const int UseTestParticles, const int SLevels, int noColl){

	int idy = threadIdx.x;
	int idx = blockIdx.x;

#if USE_RANDOM == 1
	curandState random = random_d[idx];
#else
	int random = 0;
#endif
//printf("BSA %d %d %g\n", idy, idx, StopMinMass_c[0]);

	if((noColl == 1 || noColl == -1) && BSstop_d[0] == 3){
//if(idy == 0 && idx == 0)	printf("Stop BSA b\n");
		return;
	}

	volatile double dt1 = dt;
	volatile double dt2, dt22;
	volatile double t = 0.0;
	volatile double dtgr = 1.0;

	__shared__ double4 x4_s[NN];
	__shared__ double4 xp_s[NN];
	__shared__ double4 xt_s[NN];
	__shared__ double4 v4_s[NN];
	__shared__ double4 vt_s[NN];
	__shared__ double rcritv_s[NN * def_SLevelsMax];
	__shared__ volatile int stop_s[1];
	__shared__ int Ncol_s[1];
	__shared__ int2 Colpairs_s[def_MaxColl];
	__shared__ double Coltime_s[def_MaxColl];
	
	volatile double4 vp;
	volatile double4 vt;
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

	volatile int sgnt = 1;

	double3 scalex;
	double3 scalev;

	double errorx, errorv;
	double test;

	volatile int idi;
	volatile int si = Encpairs2_d[ (st+2) * NT + idx].y;
	volatile int N2 = Encpairs2_d[si].y; //Number of bodies in  current BS simulation
	volatile int start = Encpairs2_d[NT + si].y;
//printf("BS %d %d %d %d %d\n", idx, st, si, N2, NT);


	volatile int Ne = -1; //number of pairs
	volatile int j0 = 0;
	volatile int j1 = 0;

	if(idy < N2){
		idi = Encpairs2_d[start + idy].x;
		x4_s[idy].x = xold_d[idi].x;
		x4_s[idy].y = xold_d[idi].y;
		x4_s[idy].z = xold_d[idi].z;
		x4_s[idy].w = xold_d[idi].w;
		v4_s[idy].x = vold_d[idi].x;
		v4_s[idy].y = vold_d[idi].y;
		v4_s[idy].z = vold_d[idi].z;
		v4_s[idy].w = vold_d[idi].w;
		for(int l = 0; l < SLevels; ++l){
			rcritv_s[idy + l * NN] = rcritv_d[idi + l * NconstT];
		}
		Ne = Encpairs_d[idi].y;
		volatile int j0g = 0;
		volatile int j1g = 0;

		if(Ne > 0){
			j0g = Encpairs_d[idi * NencMax + 0].x; //index of j in global memory
			j0 = Encpairs_d[NT + j0g].y;
		}
		if(Ne > 1){
			j1g = Encpairs_d[idi * NencMax + 1].x; //index of j in global memory
			j1 = Encpairs_d[NT + j1g].y;
		}
//printf("BSA2 %d %d %d %d %d %d %d %d %d %d %d\n", idx, idy, st, idi, index_d[idi], j0g, j0, j1g, j1, N2, Ne);
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
		for(int l = 0; l < SLevels; ++l){
			rcritv_s[idy + l * NN] = 0.0;
		}
	}

	if(dt < 0.0){
		sgnt = -1;
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
		a0.x = 0.0;
		a0.y = 0.0;
		a0.z = 0.0;

		if(Ne > 0) accEnc(x4_s[idy], x4_s[j0], a0, rcritv_s, test, idy, j0, NN, MinMass, UseTestParticles, SLevels);
		if(Ne > 1) accEnc(x4_s[idy], x4_s[j1], a0, rcritv_s, test, idy, j1, NN, MinMass, UseTestParticles, SLevels);
		for(int i = 2; i < Ne; ++i){
			volatile int jg = Encpairs_d[idi * NencMax + i].x;
			volatile int j = Encpairs_d[NT + jg].y;
			accEnc(x4_s[idy], x4_s[j], a0, rcritv_s, test, idy, j, NN, MinMass, UseTestParticles, SLevels);
		}
		__syncthreads();
		if(Ne >= 0){
			accEncSun(x4_s[idy], a0, def_ksq * Msun * dtgr);
		}

		volatile int f = 1;
		__syncthreads();
		for(int ff = 0; ff < 1e6; ++ff){
			__syncthreads();
			for(int n = 1; n <= 8; ++n){
				if(idy == 0) stop_s[0] = 1;
				__syncthreads();
				dt2 = dt1 / (2.0 * n);
				dt22 = dt2 * 2.0;

				if(Ne >= 0){
					xp_s[idy].x = x4_s[idy].x + (dt2 * dtgr * v4_s[idy].x);
					xp_s[idy].y = x4_s[idy].y + (dt2 * dtgr * v4_s[idy].y);
					xp_s[idy].z = x4_s[idy].z + (dt2 * dtgr * v4_s[idy].z);
					xp_s[idy].w = x4_s[idy].w;

					vp.x = v4_s[idy].x + (dt2 * a0.x);
					vp.y = v4_s[idy].y + (dt2 * a0.y);
					vp.z = v4_s[idy].z + (dt2 * a0.z);
					vp.w = v4_s[idy].w;
				}
				__syncthreads();

				a.x = 0.0;
				a.y = 0.0;
				a.z = 0.0;

				if(Ne > 0) accEnc(xp_s[idy], xp_s[j0], a, rcritv_s, test, idy, j0, NN, MinMass, UseTestParticles, SLevels);
				if(Ne > 1) accEnc(xp_s[idy], xp_s[j1], a, rcritv_s, test, idy, j1, NN, MinMass, UseTestParticles, SLevels);
				for(int i = 2; i < Ne; ++i){
					volatile int jg = Encpairs_d[idi * NencMax + i].x;
					volatile int j = Encpairs_d[NT + jg].y;
					accEnc(xp_s[idy], xp_s[j], a, rcritv_s, test, idy, j, NN, MinMass, UseTestParticles, SLevels);
				}
				__syncthreads();
				if(Ne >= 0){
					accEncSun(xp_s[idy], a, def_ksq * Msun * dtgr);

					xt_s[idy].x = x4_s[idy].x + (dt22 * dtgr * vp.x);
					xt_s[idy].y = x4_s[idy].y + (dt22 * dtgr * vp.y);
					xt_s[idy].z = x4_s[idy].z + (dt22 * dtgr * vp.z);
					xt_s[idy].w = x4_s[idy].w;

					vt.x = v4_s[idy].x + (dt22 * a.x);
					vt.y = v4_s[idy].y + (dt22 * a.y);
					vt.z = v4_s[idy].z + (dt22 * a.z);
					vt.w = v4_s[idy].w;
				}
				__syncthreads();

				for(int m = 2; m <= n; ++m){

					a.x = 0.0;
					a.y = 0.0;
					a.z = 0.0;

					if(Ne > 0) accEnc(xt_s[idy], xt_s[j0], a, rcritv_s, test, idy, j0, NN, MinMass, UseTestParticles, SLevels);
					if(Ne > 1) accEnc(xt_s[idy], xt_s[j1], a, rcritv_s, test, idy, j1, NN, MinMass, UseTestParticles, SLevels);
					for(int i = 2; i < Ne; ++i){
						volatile int jg = Encpairs_d[idi * NencMax + i].x;
						volatile int j = Encpairs_d[NT + jg].y;
						accEnc(xt_s[idy], xt_s[j], a, rcritv_s, test, idy, j, NN, MinMass, UseTestParticles, SLevels);
					}
					__syncthreads();
					if(Ne >= 0){
						accEncSun(xt_s[idy], a, def_ksq * Msun * dtgr);

						xp_s[idy].x += (dt22 * dtgr * vt.x);
						xp_s[idy].y += (dt22 * dtgr * vt.y);
						xp_s[idy].z += (dt22 * dtgr * vt.z);

						vp.x += (dt22 * a.x);
						vp.y += (dt22 * a.y);
						vp.z += (dt22 * a.z);
					}
					__syncthreads();

					a.x = 0.0;
					a.y = 0.0;
					a.z = 0.0;

					if(Ne > 0) accEnc(xp_s[idy], xp_s[j0], a, rcritv_s, test, idy, j0, NN, MinMass, UseTestParticles, SLevels);
					if(Ne > 1) accEnc(xp_s[idy], xp_s[j1], a, rcritv_s, test, idy, j1, NN, MinMass, UseTestParticles, SLevels);	
					for(int i = 2; i < Ne; ++i){
						volatile int jg = Encpairs_d[idi * NencMax + i].x;
						volatile int j = Encpairs_d[NT + jg].y;
						accEnc(xp_s[idy], xp_s[j], a, rcritv_s, test, idy, j, NN, MinMass, UseTestParticles, SLevels);
					}
					__syncthreads();
					if(Ne >= 0){
						accEncSun(xp_s[idy], a, def_ksq * Msun * dtgr);

						xt_s[idy].x += (dt22 * dtgr * vp.x);
						xt_s[idy].y += (dt22 * dtgr * vp.y);
						xt_s[idy].z += (dt22 * dtgr * vp.z);

						vt.x += (dt22 * a.x);
						vt.y += (dt22 * a.y);
						vt.z += (dt22 * a.z);
					}
					__syncthreads();
				}//end of m loop
				a.x = 0.0;
				a.y = 0.0;
				a.z = 0.0;

				if(Ne > 0) accEnc(xt_s[idy], xt_s[j0], a, rcritv_s, test, idy, j0, NN, MinMass, UseTestParticles, SLevels);
				if(Ne > 1) accEnc(xt_s[idy], xt_s[j1], a, rcritv_s, test, idy, j1, NN, MinMass, UseTestParticles, SLevels);
				for(int i = 2; i < Ne; ++i){
					volatile int jg = Encpairs_d[idi * NencMax + i].x;
					volatile int j = Encpairs_d[NT + jg].y;
					accEnc(xt_s[idy], xt_s[j], a, rcritv_s, test, idy, j, NN, MinMass, UseTestParticles, SLevels);
				}
				__syncthreads();
				if(Ne >= 0){
					accEncSun(xt_s[idy], a, def_ksq * Msun * dtgr);

					xp_s[idy].x += (dt2 * dtgr * vt.x);
					xp_s[idy].y += (dt2 * dtgr * vt.y);
					xp_s[idy].z += (dt2 * dtgr * vt.z);

					vp.x += (dt2 * a.x);
					vp.y += (dt2 * a.y);
					vp.z += (dt2 * a.z);
					if(n == 8){				
						dx7.x = 0.5 * (xt_s[idy].x + xp_s[idy].x);
						dx7.y = 0.5 * (xt_s[idy].y + xp_s[idy].y);
						dx7.z = 0.5 * (xt_s[idy].z + xp_s[idy].z);

						dv7.x = 0.5 * (vt.x + vp.x);
						dv7.y = 0.5 * (vt.y + vp.y);
						dv7.z = 0.5 * (vt.z + vp.z);
//printf("dx8 %d %d %d %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", tt, ff, n, idy, idi, dx7.x, dx7.y, dx7.z, dv7.x, dv7.y, dv7.z);
					}
					if(n == 7){				
						dx6.x = 0.5 * (xt_s[idy].x + xp_s[idy].x);
						dx6.y = 0.5 * (xt_s[idy].y + xp_s[idy].y);
						dx6.z = 0.5 * (xt_s[idy].z + xp_s[idy].z);

						dv6.x = 0.5 * (vt.x + vp.x);
						dv6.y = 0.5 * (vt.y + vp.y);
						dv6.z = 0.5 * (vt.z + vp.z);
//printf("dx7 %d %d %d %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", tt, ff, n, idy, idi, dx6.x, dx6.y, dx6.z, dv6.x, dv6.y, dv6.z);
					}
					if(n == 6){				
						dx5.x = 0.5 * (xt_s[idy].x + xp_s[idy].x);
						dx5.y = 0.5 * (xt_s[idy].y + xp_s[idy].y);
						dx5.z = 0.5 * (xt_s[idy].z + xp_s[idy].z);

						dv5.x = 0.5 * (vt.x + vp.x);
						dv5.y = 0.5 * (vt.y + vp.y);
						dv5.z = 0.5 * (vt.z + vp.z);
//printf("dx6 %d %d %d %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", tt, ff, n, idy, idi, dx5.x, dx5.y, dx5.z, dv5.x, dv5.y, dv5.z);
					}
					if(n == 5){				
						dx4.x = 0.5 * (xt_s[idy].x + xp_s[idy].x);
						dx4.y = 0.5 * (xt_s[idy].y + xp_s[idy].y);
						dx4.z = 0.5 * (xt_s[idy].z + xp_s[idy].z);

						dv4.x = 0.5 * (vt.x + vp.x);
						dv4.y = 0.5 * (vt.y + vp.y);
						dv4.z = 0.5 * (vt.z + vp.z);
//printf("dx5 %d %d %d %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", tt, ff, n, idy, idi, dx4.x, dx4.y, dx4.z, dv4.x, dv4.y, dv4.z);
					}
					if(n == 4){				
						dx3.x = 0.5 * (xt_s[idy].x + xp_s[idy].x);
						dx3.y = 0.5 * (xt_s[idy].y + xp_s[idy].y);
						dx3.z = 0.5 * (xt_s[idy].z + xp_s[idy].z);

						dv3.x = 0.5 * (vt.x + vp.x);
						dv3.y = 0.5 * (vt.y + vp.y);
						dv3.z = 0.5 * (vt.z + vp.z);
//printf("dx4 %d %d %d %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", tt, ff, n, idy, idi, dx3.x, dx3.y, dx3.z, dv3.x, dv3.y, dv3.z);
					}
					if(n == 3){				
						dx2.x = 0.5 * (xt_s[idy].x + xp_s[idy].x);
						dx2.y = 0.5 * (xt_s[idy].y + xp_s[idy].y);
						dx2.z = 0.5 * (xt_s[idy].z + xp_s[idy].z);

						dv2.x = 0.5 * (vt.x + vp.x);
						dv2.y = 0.5 * (vt.y + vp.y);
						dv2.z = 0.5 * (vt.z + vp.z);
//printf("dx3 %d %d %d %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", tt, ff, n, idy, idi, dx2.x, dx2.y, dx2.z, dv2.x, dv2.y, dv2.z);
					}
					if(n == 2){				
						dx1.x = 0.5 * (xt_s[idy].x + xp_s[idy].x);
						dx1.y = 0.5 * (xt_s[idy].y + xp_s[idy].y);
						dx1.z = 0.5 * (xt_s[idy].z + xp_s[idy].z);

						dv1.x = 0.5 * (vt.x + vp.x);
						dv1.y = 0.5 * (vt.y + vp.y);
						dv1.z = 0.5 * (vt.z + vp.z);
//printf("dx2 %d %d %d %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", tt, ff, n, idy, idi, dx1.x, dx1.y, dx1.z, dv1.x, dv1.y, dv1.z);
					}
					if(n == 1){				
						dx0.x = 0.5 * (xt_s[idy].x + xp_s[idy].x);
						dx0.y = 0.5 * (xt_s[idy].y + xp_s[idy].y);
						dx0.z = 0.5 * (xt_s[idy].z + xp_s[idy].z);

						dv0.x = 0.5 * (vt.x + vp.x);
						dv0.y = 0.5 * (vt.y + vp.y);
						dv0.z = 0.5 * (vt.z + vp.z);
//printf("dx1 %d %d %d %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", tt, ff, n, idy, idi, dx0.x, dx0.y, dx0.z, dv0.x, dv0.y, dv0.z);
					}

					for(int j = n-1; j >= 1; --j){
						double t0 = BSt0_c[(n-1) * 8 + (j-1)];
						double t1 = t0 * BSddt_c[j];
						double t2 = t0 * BSddt_c[n-1];
				
						if(j == 7){
							dx6.x = (t1 * dx7.x) - (t2 * dx6.x);
							dx6.y = (t1 * dx7.y) - (t2 * dx6.y);
							dx6.z = (t1 * dx7.z) - (t2 * dx6.z);

							dv6.x = (t1 * dv7.x) - (t2 * dv6.x);
							dv6.y = (t1 * dv7.y) - (t2 * dv6.y);
							dv6.z = (t1 * dv7.z) - (t2 * dv6.z);
						}
						if(j == 6){
							dx5.x = (t1 * dx6.x) - (t2 * dx5.x);
							dx5.y = (t1 * dx6.y) - (t2 * dx5.y);
							dx5.z = (t1 * dx6.z) - (t2 * dx5.z);

							dv5.x = (t1 * dv6.x) - (t2 * dv5.x);
							dv5.y = (t1 * dv6.y) - (t2 * dv5.y);
							dv5.z = (t1 * dv6.z) - (t2 * dv5.z);
						}
						if(j == 5){
							dx4.x = (t1 * dx5.x) - (t2 * dx4.x);
							dx4.y = (t1 * dx5.y) - (t2 * dx4.y);
							dx4.z = (t1 * dx5.z) - (t2 * dx4.z);

							dv4.x = (t1 * dv5.x) - (t2 * dv4.x);
							dv4.y = (t1 * dv5.y) - (t2 * dv4.y);
							dv4.z = (t1 * dv5.z) - (t2 * dv4.z);
						}
						if(j == 4){
							dx3.x = (t1 * dx4.x) - (t2 * dx3.x);
							dx3.y = (t1 * dx4.y) - (t2 * dx3.y);
							dx3.z = (t1 * dx4.z) - (t2 * dx3.z);

							dv3.x = (t1 * dv4.x) - (t2 * dv3.x);
							dv3.y = (t1 * dv4.y) - (t2 * dv3.y);
							dv3.z = (t1 * dv4.z) - (t2 * dv3.z);
						}
						if(j == 3){
							dx2.x = (t1 * dx3.x) - (t2 * dx2.x);
							dx2.y = (t1 * dx3.y) - (t2 * dx2.y);
							dx2.z = (t1 * dx3.z) - (t2 * dx2.z);

							dv2.x = (t1 * dv3.x) - (t2 * dv2.x);
							dv2.y = (t1 * dv3.y) - (t2 * dv2.y);
							dv2.z = (t1 * dv3.z) - (t2 * dv2.z);
						}
						if(j == 2){
							dx1.x = (t1 * dx2.x) - (t2 * dx1.x);
							dx1.y = (t1 * dx2.y) - (t2 * dx1.y);
							dx1.z = (t1 * dx2.z) - (t2 * dx1.z);

							dv1.x = (t1 * dv2.x) - (t2 * dv1.x);
							dv1.y = (t1 * dv2.y) - (t2 * dv1.y);
							dv1.z = (t1 * dv2.z) - (t2 * dv1.z);
						}
						if(j == 1){
							dx0.x = (t1 * dx1.x) - (t2 * dx0.x);
							dx0.y = (t1 * dx1.y) - (t2 * dx0.y);
							dx0.z = (t1 * dx1.z) - (t2 * dx0.z);

							dv0.x = (t1 * dv1.x) - (t2 * dv0.x);
							dv0.y = (t1 * dv1.y) - (t2 * dv0.y);
							dv0.z = (t1 * dv1.z) - (t2 * dv0.z);
						}
					}
					errorx = dx0.x * dx0.x * scalex.x;
					errorv = dv0.x * dv0.x * scalev.x;
					errorx = fmax(errorx, dx0.y * dx0.y * scalex.y);
					errorv = fmax(errorv, dv0.y * dv0.y * scalev.y);
					errorx = fmax(errorx, dx0.z * dx0.z * scalex.z);
					errorv = fmax(errorv, dv0.z * dv0.z * scalev.z);

//if(idi == 3534) printf("dx %d %d %d %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", tt, ff, n, idy, idi, dx0.x, dx0.y, dx0.z, dv0.x, dv0.y, dv0.z);
//if(idi == 3534) printf("scale %d %d %d %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", tt, ff, n, idy, idi, scalex.x, scalex.y, scalex.z, scalev.x, scalev.y, scalev.z);
					errorx = fmax(errorx, errorv);
					if(errorx >= def_tol * def_tol){
						stop_s[0] = 0;
//if(tt == 0 && ff >= 13 && ff < 15) printf("error %d %d %d %d %d %.20g %.20g %.20g %.20g\n", tt, ff, n, idy, idi, errorx, dx0.x, dx0.y, dx0.z);
//if(idi == 3534) printf("error %d %d %d %d %d %.20g %.20g %.20g %.20g\n", tt, ff, n, idy, idi, errorx, dx0.x, dx0.y, dx0.z);
					}
					Ncol_s[0] = 0;
					Coltime_s[0] = 10.0;
				}
				__syncthreads();
				if(stop_s[0] == 1 || sgnt * dt1 < def_dtmin){
//if(idy == 0) printf("tt %d %d %d %g\n", tt, ff, n, Coltime_s[0]);
					if(Ne >= 0){
						xt_s[idy].x = dx0.x;
						xt_s[idy].y = dx0.y;
						xt_s[idy].z = dx0.z;

						vt.x = dv0.x;
						vt.y = dv0.y;
						vt.z = dv0.z;
///*if(Ne >= 0 && tt == 8 && n == 7)* / printf("xt %d %d %d %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", tt, ff, n, idy, idi, xt_s[idy].x, xt_s[idy].y, xt_s[idy].z, vt.x, vt.y, vt.z);
						if(n >= 2){
							xt_s[idy].x += dx1.x;
							xt_s[idy].y += dx1.y;
							xt_s[idy].z += dx1.z;
							vt.x += dv1.x;
							vt.y += dv1.y;
							vt.z += dv1.z;
///*if(Ne >= 0 && tt == 8 && n == 7)* / printf("xt %d %d %d %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", tt, ff, n, idy, idi, xt_s[idy].x, xt_s[idy].y, xt_s[idy].z, vt.x, vt.y, vt.z);
						}
						if(n >= 3){
							xt_s[idy].x += dx2.x;
							xt_s[idy].y += dx2.y;
							xt_s[idy].z += dx2.z;
							vt.x += dv2.x;
							vt.y += dv2.y;
							vt.z += dv2.z;
///*if(Ne >= 0 && tt == 8 && n == 7)* / printf("xt %d %d %d %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", tt, ff, n, idy, idi, xt_s[idy].x, xt_s[idy].y, xt_s[idy].z, vt.x, vt.y, vt.z);
						}
						if(n >= 4){
							xt_s[idy].x += dx3.x;
							xt_s[idy].y += dx3.y;
							xt_s[idy].z += dx3.z;
							vt.x += dv3.x;
							vt.y += dv3.y;
							vt.z += dv3.z;
///*if(Ne >= 0 && tt == 8 && n == 7)* / printf("xt %d %d %d %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", tt, ff, n, idy, idi, xt_s[idy].x, xt_s[idy].y, xt_s[idy].z, vt.x, vt.y, vt.z);
						}
						if(n >= 5){
							xt_s[idy].x += dx4.x;
							xt_s[idy].y += dx4.y;
							xt_s[idy].z += dx4.z;
							vt.x += dv4.x;
							vt.y += dv4.y;
							vt.z += dv4.z;
///*if(Ne >= 0 && tt == 8 && n == 7)* / printf("xt %d %d %d %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", tt, ff, n, idy, idi, xt_s[idy].x, xt_s[idy].y, xt_s[idy].z, vt.x, vt.y, vt.z);
						}
						if(n >= 6){
							xt_s[idy].x += dx5.x;
							xt_s[idy].y += dx5.y;
							xt_s[idy].z += dx5.z;
							vt.x += dv5.x;
							vt.y += dv5.y;
							vt.z += dv5.z;
///*if(Ne >= 0 && tt == 8 && n == 7)* / printf("xt %d %d %d %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", tt, ff, n, idy, idi, xt_s[idy].x, xt_s[idy].y, xt_s[idy].z, vt.x, vt.y, vt.z);
						}
						if(n >= 7){
							xt_s[idy].x += dx6.x;
							xt_s[idy].y += dx6.y;
							xt_s[idy].z += dx6.z;
							vt.x += dv6.x;
							vt.y += dv6.y;
							vt.z += dv6.z;
///*if(Ne >= 0 && tt == 8 && n == 7)* / printf("xt %d %d %d %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", tt, ff, n, idy, idi, xt_s[idy].x, xt_s[idy].y, xt_s[idy].z, vt.x, vt.y, vt.z);
						}
						if(n >= 8){
							xt_s[idy].x += dx7.x;
							xt_s[idy].y += dx7.y;
							xt_s[idy].z += dx7.z;
							vt.x += dv7.x;
							vt.y += dv7.y;
							vt.z += dv7.z;
///*if(Ne >= 0 && tt == 8 && n == 7)* / printf("xt %d %d %d %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", tt, ff, n, idy, idi, xt_s[idy].x, xt_s[idy].y, xt_s[idy].z, vt.x, vt.y, vt.z);
						}
					}
					vt_s[idy].x = vt.x;
					vt_s[idy].y = vt.y;
					vt_s[idy].z = vt.z;
					vt_s[idy].w = vt.w;
///*if(Ne >= 0 && tt == 8 && n == 7)* / printf("x %d %d %d %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", tt, ff, n, idy, idi, xt_s[idy].x, xt_s[idy].y, xt_s[idy].z, vt.x, vt.y, vt.z);
					__syncthreads();
					for(int i = 0; i < Ne; ++i){
						double delta = 100.0;
						double enct = 100.0;
						double colt = 100.0;
						volatile int jg = Encpairs_d[idi * NencMax + i].x;
						volatile int j = Encpairs_d[NT + jg].y;
						double rcrit = v4_s[idy].w + v4_s[j].w;
						if((noColl == 1 || noColl == -1) && index_d[idi] == CollTshiftpairs_c[0].x && index_d[jg] == CollTshiftpairs_c[0].y){
							rcrit = v4_s[idy].w * CollTshift_c[0] + v4_s[j].w * CollTshift_c[0];
						}
						if((noColl == 1 || noColl == -1) && index_d[idi] == CollTshiftpairs_c[0].y && index_d[jg] == CollTshiftpairs_c[0].x){
							rcrit = v4_s[idy].w * CollTshift_c[0] + v4_s[j].w * CollTshift_c[0];
						}
						if(CollisionPrecision_c[0] < 0.0){
							//do not overlap bodies when collision, increase therefore radius slightly, R + R * precision
							rcrit *= (1.0 - CollisionPrecision_c[0]);	
						}
//printf("%d %d %d %d %d\n", Encpairs2_d[start + idy].x, Encpairs2_d[start + j].x, jg, j, idi);
						if(idi > jg){
							delta = encounter1(xt_s[idy], vt_s[idy], x4_s[idy], v4_s[idy], xt_s[j], vt_s[j], x4_s[j], v4_s[j], rcrit, dt1 * dtgr, idy, j, enct, colt, MinMass, noColl);
						}
						if((noColl == 1 || noColl == -1) && colt == 100.0){
							delta = 100.0;
						}
						if((noColl == 1 || noColl == -1) && colt == 200.0){
							noColl = 2;
							BSstop_d[0] = 3;
						}
						if(delta < rcrit*rcrit){
							int Ni = atomicAdd(&Ncol_s[0], 1);
							if(Ncol_s[0] >= def_MaxColl) Ni = def_MaxColl - 1;
//printf("EE1 %d %d %d %d | %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %d %d %g %d\n", idi, jg, index_d[idi], index_d[jg], xt_s[idy].w, xt_s[j].w, xt_s[idy].x, xt_s[idy].y, xt_s[idy].z, xt_s[j].x, xt_s[j].y, xt_s[j].z, delta, rcrit*rcrit, f, n, colt, Ni);

							if(xt_s[idy].w >= xt_s[j].w){
								Colpairs_s[Ni].x = idy;
								Colpairs_s[Ni].y = j;
							}
							else{
								Colpairs_s[Ni].x = j;
								Colpairs_s[Ni].y = idy;
							}
							Coltime_s[Ni] = colt;

							// *****************
							//dont group test particles
/*							if(xt_s[idy].w == 0.0){
								Colpairs_s[Ni].x = idy;
								Colpairs_s[Ni].y = idy;
							}
							if(xt_s[j].w == 0.0){
								Colpairs_s[Ni].x = j;
								Colpairs_s[Ni].y = j;
							}
*/
							// *****************
						}

						if(WriteEncounters_c[0] > 0 && noColl == 0){
							double writeRadius = 0.0;
							//in scales of planetary Radius
							writeRadius = WriteEncountersRadius_c[0] * fmax(vt_s[idy].w, vt_s[j].w);
							if(delta < writeRadius * writeRadius){

								if(enct > 0.0 && enct < 1.0){
									//ingnore encounters within the same particle cloud
									if(index_d[idi] / WriteEncountersCloudSize_c[0] != index_d[jg] / WriteEncountersCloudSize_c[0]){

//printf("Write Enc %g %g %g %g %g %d %d\n", (t + dt1) / dayUnit, writeRadius, sqrt(delta), enct, colt, ii, jg);
										int ne = atomicAdd(NWriteEnc_d, 1);
										if(ne >= def_MaxWriteEnc - 1) ne = def_MaxWriteEnc - 1;
										storeEncounters(xt_s, vt_s, idy, j, idi, jg, index_d, ne, writeEnc_d, time + (t + dt1) / dayUnit, spin_d);
									}
								}
							}
						}
					}
					__syncthreads();
					if(idy == 0) {
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

//printf("dRA %d %d %.20g %.20g %.20g\n", i, j, d, R, dR);
							if(dR > fabs(CollisionPrecision_c[0]) && d != 0.0){
							//bodies are already overlapping
								Coltime = fmin(Coltime_s[c], Coltime);
							}


						}
						Coltime_s[0] = Coltime;
//printf("ColtimeT %.20g %g %g %g %d %d %d %d %d\n", Coltime, t / dayUnit, dt1 / dayUnit, (1.0 - Coltime) * dt1, tt, ff, n, Ncol_s[0], Ncoll_d[0]);
					}
					__syncthreads();
					if(Coltime_s[0] == 10.0){
						if(idy == 0){
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
//printf("cTime coll BSA %g %g %g %.20g %d %d %d\n", time, t / dayUnit, dt / dayUnit, time + (t + dt1) / dayUnit, index_d[Encpairs2_d[start + i].x], index_d[Encpairs2_d[start + j].x], nc);
									collide(random, xt_s, vt_s, i, j, Encpairs2_d[start + i].x, Encpairs2_d[start + j].x, Msun, U_d, test, index_d, nc, Coll_d, time + (t + dt1) / dayUnit, spin_d, love_d, createFlag_d, rcritv_s, rcrit_d, NN, NconstT, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, SLevels, noColl);
								}
							}
						}
						__syncthreads();
						vt.x = vt_s[idy].x;
						vt.y = vt_s[idy].y;
						vt.z = vt_s[idy].z;
						vt.w = vt_s[idy].w;

						t += dt1;
						if(n >= 8) dt1 *= 0.55;
						if(n < 7) dt1 *= 1.3;
						if(sgnt * dt1 > sgnt * dt) dt1 = dt;
						if(sgnt * (t+dt1) > sgnt * dt) dt1 = dt - t;
						if(sgnt * dt1 < def_dtmin) dt1 = sgnt * def_dtmin;

						if(Ne >= 0){
							x4_s[idy].x = xt_s[idy].x;
							x4_s[idy].y = xt_s[idy].y;
							x4_s[idy].z = xt_s[idy].z;
							x4_s[idy].w = xt_s[idy].w;
							v4_s[idy].x = vt.x;
							v4_s[idy].y = vt.y;
							v4_s[idy].z = vt.z;
							v4_s[idy].w = vt.w;
//printf("update %d %d %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.10g %g %g %d %d %d\n", idy, idx, idi, x4_s[idy].x, x4_s[idy].y, x4_s[idy].z, v4_s[idy].x, v4_s[idy].y, v4_s[idy].z, x4_s[idy].w, v4_s[idy].w, rcritv_s[idy], t / dayUnit, dt1 / dayUnit, tt, ff, n);
						}
					}
					else{
						dt1 *= Coltime_s[0];
					}
					f = 0;

					__syncthreads();
					break;
				}
				__syncthreads();
				if(BSstop_d[0] == 3){
//if(idy == 0) printf("Stop BSA\n");
					__syncthreads();
					return;
				}
			} //end of n loop
			if(f == 0) break;
			__syncthreads();	
			dt1 *= 0.5;
//if(idy == 0) printf("continue %d %d %g %g %d %d\n", idy, idx, t, dt1, tt, ff);
		}//end of ff loop
		if(sgnt * t >= sgnt * dt){
			break;
		}
//if(idy == 0) printf("not finished %d %d\n", idy, idx);

		__syncthreads();
	}//end of tt loop
	if(idy < N2){
//if(xt_s[idy].w <= 0){
		x4_d[idi].x = xt_s[idy].x;
		x4_d[idi].y = xt_s[idy].y;
		x4_d[idi].z = xt_s[idy].z;
		x4_d[idi].w = xt_s[idy].w;
		v4_d[idi].x = vt.x;
		v4_d[idi].y = vt.y;
		v4_d[idi].z = vt.z;
		v4_d[idi].w = vt.w;
		for(int l = 0; l < SLevels; ++l){  
			rcritv_d[idi + l * NconstT] = rcritv_s[idy + l * NN];
		}
//}
///*if(idi == 4077)* / printf("final %g %d %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g\n", time, idi, NN, x4_d[idi].x, x4_d[idi].y, x4_d[idi].z, v4_d[idi].x, v4_d[idi].y, v4_d[idi].z, x4_d[idi].w, v4_d[idi].w);
	}
#if USE_RANDOM == 1
	random_d[idx] = random;
#endif
}


template <int NN, int Bl>
__global__ void BSA512_kernel(curandState *random_d, double4 *x4_d, double4 *v4_d, double4 *xold_d, double4 *vold_d, double4 *xp_d, double4 *vp_d, double4 *xt_d, double4 *vt_d, double *rcrit_d, double *rcritv_d, int *index_d, double4 *spin_d, double3 *love_d, int *createFlag_d, int2 *Encpairs_d, int2 *Encpairs2_d, double3 *dx_d, double3 *dv_d, const double dt, const double Msun, double *U_d, const int st, const int NT, const int NconstT, const int NencMax, int *BSstop_d, int *Ncoll_d, double *Coll_d, const double time, float4 *aelimits_d, unsigned int *aecount_d, unsigned int *enccount_d, unsigned long long *aecountT_d, unsigned long long *enccountT_d, int *NWriteEnc_d, double *writeEnc_d, double *dtgr_d, const int UseGR, const double MinMass, const int UseTestParticles, const int SLevels, int noColl){

	int idy = threadIdx.x;
	int idx = blockIdx.x;

	if((noColl == 1 || noColl == -1) && BSstop_d[0] == 3){
//if(idy == 0 && idx == 0)	printf("Stop BSA512 b\n");
		return;
	}

#if USE_RANDOM == 1
	curandState random = random_d[idx];
#else
	int random = 0;
#endif
	volatile double dt1 = dt;
	volatile double dt2, dt22;
	volatile double t = 0.0;

	__shared__ volatile int stop_s[1];
	__shared__ int Ncol_s[1];
	__shared__ int2 Colpairs_s[def_MaxColl];
	__shared__ double Coltime_s[def_MaxColl];

	double3 dxj;
	double3 dvj;

	volatile int sgnt = 1;

	double3 scalex;
	double3 scalev;

	double errorx, errorv;

	volatile int si = Encpairs2_d[ (st+2) * NT + idx].y;
	volatile int N2 = Encpairs2_d[si].y; //Number of bodies in  current BS simulation
	volatile int start = Encpairs2_d[NT + si].y;
//if(idy == 0) printf("BS %d %d %d %d %d\n", idx, st, si, N2, NT);
	if(dt < 0.0) sgnt = -1;

	for(int i = 0; i < NN; i += Bl){
		volatile int Ne = -1; //number of pairs
		volatile int idi = 0;
		if(idy + i < N2){
			idi = Encpairs2_d[start + idy + i].x;
			Ne = Encpairs_d[idi].y;
		}
		if(Ne >= 0){
			if(UseGR == 1){// GR time rescale (Saha & Tremaine 1994)
				double c2 = def_cm * def_cm;
				double mu = def_ksq * Msun;
				double rsq = xold_d[idi].x * xold_d[idi].x + xold_d[idi].y * xold_d[idi].y + xold_d[idi].z * xold_d[idi].z;
				double vsq = vold_d[idi].x * vold_d[idi].x + vold_d[idi].y * vold_d[idi].y + vold_d[idi].z * vold_d[idi].z;
				double ir = 1.0/sqrt(rsq);
				double ia = 2.0*ir-vsq/mu;
				dtgr_d[idi] = 1.0 - 1.5 * mu * ia / c2;
			}
			else dtgr_d[idi] = 1.0;
		}
	}


	for(int tt = 0; tt < 10000; ++tt){
		volatile int f = 1;
		__syncthreads();
		for(int ff = 0; ff < 1e6; ++ff){
			for(int n = 1; n <= 8; ++n){
				if(idy == 0) stop_s[0] = 1;
				__syncthreads();
				dt2 = dt1 / (2.0 * n);
				dt22 = dt2 * 2.0;

				for(int i = 0; i < NN; i += Bl){
					volatile int Ne = -1; //number of pairs
					volatile int idi = 0;
					if(idy + i < N2){
						idi = Encpairs2_d[start + idy + i].x;
						Ne = Encpairs_d[idi].y;
					}
					double3 a;
					a.x = 0.0;
					a.y = 0.0;
					a.z = 0.0;
					for(int ii = 0; ii < Ne; ++ii){
						volatile int j = Encpairs_d[idi * NencMax + ii].x;
						double test;
//if(tt == 0 && ff == 0 && n == 1) printf("BS2 %d %d %d %d %d %d %d %d | %d %d %d %.40g %g %g\n", idx, idy, st, idi, index_d[idi], j, N2, Ne, tt, ff, n, xold_d[idi].x, t, dt1);
						accEnc(xold_d[idi], xold_d[j], a, rcritv_d, test, idi, j, NconstT, MinMass, UseTestParticles, SLevels);
					}
					if(Ne >= 0){
						volatile double dtgr;
						if(UseGR == 1){
							dtgr = dtgr_d[idi];
						}
						else dtgr = 1.0;
						
						accEncSun(xold_d[idi], a, def_ksq * Msun * dtgr);
						
						xp_d[idi].x = xold_d[idi].x + (dt2 * dtgr * vold_d[idi].x);
						xp_d[idi].y = xold_d[idi].y + (dt2 * dtgr * vold_d[idi].y);
						xp_d[idi].z = xold_d[idi].z + (dt2 * dtgr * vold_d[idi].z);
						xp_d[idi].w = xold_d[idi].w;

						vp_d[idi].x = vold_d[idi].x + (dt2 * a.x);
						vp_d[idi].y = vold_d[idi].y + (dt2 * a.y);
						vp_d[idi].z = vold_d[idi].z + (dt2 * a.z);
						vp_d[idi].w = vold_d[idi].w;
					}
				}
				__syncthreads();
				for(int i = 0; i < NN; i += Bl){
					volatile int Ne = -1; //number of pairs
					volatile int idi = 0;
					if(idy + i < N2){
						idi = Encpairs2_d[start + idy + i].x;
						Ne = Encpairs_d[idi].y;
					}
					double3 a;
					a.x = 0.0;
					a.y = 0.0;
					a.z = 0.0;
					for(int ii = 0; ii < Ne; ++ii){
						volatile int j = Encpairs_d[idi * NencMax + ii].x;
						double test;
						accEnc(xp_d[idi], xp_d[j], a, rcritv_d, test, idi, j, NconstT, MinMass, UseTestParticles, SLevels);
					}
					if(Ne >= 0){
						volatile double dtgr;
						if(UseGR == 1){
							dtgr = dtgr_d[idi];
						}
						else dtgr = 1.0;

						accEncSun(xp_d[idi], a, def_ksq * Msun * dtgr);
						
						xt_d[idi].x = xold_d[idi].x + (dt22 * dtgr * vp_d[idi].x);
						xt_d[idi].y = xold_d[idi].y + (dt22 * dtgr * vp_d[idi].y);
						xt_d[idi].z = xold_d[idi].z + (dt22 * dtgr * vp_d[idi].z);
						xt_d[idi].w = xold_d[idi].w;

						vt_d[idi].x = vold_d[idi].x + (dt22 * a.x);
						vt_d[idi].y = vold_d[idi].y + (dt22 * a.y);
						vt_d[idi].z = vold_d[idi].z + (dt22 * a.z);
						vt_d[idi].w = vold_d[idi].w;
//if(idy == 0) printf("xp %d %.20g %.20g %.20g %.20g %.20g %.20g %d %d %d\n", idi, xp_d[idi].x, xp_d[idi].y, xp_d[idi].z, vp_d[idi].x, vp_d[idi].y, vp_d[idi].z, tt, ff, n);
//if(idi == 3159) printf("xt %d %.20g %.20g %.20g %.20g %.20g %.20g %d %d %d\n", idi, xt_d[idi].x, xt_d[idi].y, xt_d[idi].z, vt_d[idi].x, vt_d[idi].y, vt_d[idi].z, tt, ff, n);
					}
				}
				__syncthreads();

				for(int m = 2; m <= n; ++m){
					for(int i = 0; i < NN; i += Bl){
						volatile int Ne = -1; //number of pairs
						volatile int idi = 0;
						if(idy + i < N2){
							idi = Encpairs2_d[start + idy + i].x;
							Ne = Encpairs_d[idi].y;
						}
						double3 a;
						a.x = 0.0;
						a.y = 0.0;
						a.z = 0.0;
						for(int ii = 0; ii < Ne; ++ii){
							volatile int j = Encpairs_d[idi * NencMax + ii].x;
							double test;
							accEnc(xt_d[idi], xt_d[j], a, rcritv_d, test, idi, j, NconstT, MinMass, UseTestParticles, SLevels);
						}
						if(Ne >= 0){
							volatile double dtgr;
							if(UseGR == 1){
								dtgr = dtgr_d[idi];
							}
							else dtgr = 1.0;

							accEncSun(xt_d[idi], a, def_ksq * Msun * dtgr);
							
							xp_d[idi].x += (dt22 * dtgr * vt_d[idi].x);
							xp_d[idi].y += (dt22 * dtgr * vt_d[idi].y);
							xp_d[idi].z += (dt22 * dtgr * vt_d[idi].z);

							vp_d[idi].x += (dt22 * a.x);
							vp_d[idi].y += (dt22 * a.y);
							vp_d[idi].z += (dt22 * a.z);
						}
					}
					__syncthreads();
					for(int i = 0; i < NN; i += Bl){
						volatile int Ne = -1; //number of pairs
						volatile int idi = 0;
						if(idy + i < N2){
							idi = Encpairs2_d[start + idy + i].x;
							Ne = Encpairs_d[idi].y;
						}
						double3 a;
						a.x = 0.0;
						a.y = 0.0;
						a.z = 0.0;
						for(int ii = 0; ii < Ne; ++ii){
							volatile int j = Encpairs_d[idi * NencMax + ii].x;
							double test;
							accEnc(xp_d[idi], xp_d[j], a, rcritv_d, test, idi, j, NconstT, MinMass, UseTestParticles, SLevels);
						}
						if(Ne >= 0){
							volatile double dtgr;
							if(UseGR == 1){
								dtgr = dtgr_d[idi];
							}
							else dtgr = 1.0;

							accEncSun(xp_d[idi], a, def_ksq * Msun * dtgr);
							
							xt_d[idi].x += (dt22 * dtgr * vp_d[idi].x);
							xt_d[idi].y += (dt22 * dtgr * vp_d[idi].y);
							xt_d[idi].z += (dt22 * dtgr * vp_d[idi].z);

							vt_d[idi].x += (dt22 * a.x);
							vt_d[idi].y += (dt22 * a.y);
							vt_d[idi].z += (dt22 * a.z);
						}
					}
					__syncthreads();
				}//end of m loop
				for(int i = 0; i < NN; i += Bl){
					volatile int Ne = -1; //number of pairs
					volatile int idi = 0;
					if(idy + i < N2){
						idi = Encpairs2_d[start + idy + i].x;
						Ne = Encpairs_d[idi].y;
					}
					double3 a;
					a.x = 0.0;
					a.y = 0.0;
					a.z = 0.0;
					for(int ii = 0; ii < Ne; ++ii){
						volatile int j = Encpairs_d[idi * NencMax + ii].x;
						double test;
						accEnc(xt_d[idi], xt_d[j], a, rcritv_d, test, idi, j, NconstT, MinMass, UseTestParticles, SLevels);
					}
					if(Ne >= 0){
						volatile double dtgr;
						if(UseGR == 1){
							dtgr = dtgr_d[idi];
						}
						else dtgr = 1.0;

						accEncSun(xt_d[idi], a, def_ksq * Msun * dtgr);
						
						xp_d[idi].x += (dt2 * dtgr * vt_d[idi].x);
						xp_d[idi].y += (dt2 * dtgr * vt_d[idi].y);
						xp_d[idi].z += (dt2 * dtgr * vt_d[idi].z);

						vp_d[idi].x += (dt2 * a.x);
						vp_d[idi].y += (dt2 * a.y);
						vp_d[idi].z += (dt2 * a.z);

//if(idi == 1) printf("xp %d %.20g %.20g %.20g %.20g %.20g %.20g %d %d %d\n", idi, xp_d[idi].x, xp_d[idi].y, xp_d[idi].z, vp_d[idi].x, vp_d[idi].y, vp_d[idi].z, tt, ff, n);
//if(idi == 1) printf("xt %d %.20g %.20g %.20g %.20g %.20g %.20g %d %d %d\n", idi, xt_d[idi].x, xt_d[idi].y, xt_d[idi].z, vt_d[idi].x, vt_d[idi].y, vt_d[idi].z, tt, ff, n);

						dxj.x = 0.5 * (xt_d[idi].x + xp_d[idi].x);
						dxj.y = 0.5 * (xt_d[idi].y + xp_d[idi].y);
						dxj.z = 0.5 * (xt_d[idi].z + xp_d[idi].z);

						dvj.x = 0.5 * (vt_d[idi].x + vp_d[idi].x);
						dvj.y = 0.5 * (vt_d[idi].y + vp_d[idi].y);
						dvj.z = 0.5 * (vt_d[idi].z + vp_d[idi].z);

						dx_d[(n - 1) * NT + idi] = dxj;
						dv_d[(n - 1) * NT + idi] = dvj;
//printf("dxj %d %.20g %.20g %.20g %.20g %.20g %.20g\n", idi, dxj.x, dxj.y, dxj.z, dvj.x, dvj.y, dvj.z);

						for(int j = n-1; j >= 1; --j){
							double t0 = BSt0_c[(n-1) * 8 + (j-1)];
							double t1 = t0 * BSddt_c[j];
							double t2 = t0 * BSddt_c[n-1];

							dxj.x = (t1 * dxj.x) - (t2 * dx_d[(j - 1) * NT + idi].x);
							dxj.y = (t1 * dxj.y) - (t2 * dx_d[(j - 1) * NT + idi].y);
							dxj.z = (t1 * dxj.z) - (t2 * dx_d[(j - 1) * NT + idi].z);

							dvj.x = (t1 * dvj.x) - (t2 * dv_d[(j - 1) * NT + idi].x);
							dvj.y = (t1 * dvj.y) - (t2 * dv_d[(j - 1) * NT + idi].y);
							dvj.z = (t1 * dvj.z) - (t2 * dv_d[(j - 1) * NT + idi].z);


							dx_d[(j - 1) * NT + idi] = dxj;
							dv_d[(j - 1) * NT + idi] = dvj;
					
						}

						dxj = dx_d[0 * NT + idi];
						dvj = dv_d[0 * NT + idi];
//if(idy == 0 || n > 10 || n < 0) printf("d %d %d %g %g %d %d %d %g %g\n", idi, N2, dxj, dvj, n, tt, ff, t, dt1);

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

//if(idi == 1) printf("dx %d %.20g %.20g %.20g %.20g %.20g %.20g\n", idi, dxj.x, dxj.y, dxj.z, dvj.x, dvj.y, dvj.z);
//if(idi == 1) printf("scale %d %.20g %.20g %.20g %.20g %.20g %.20g\n", idi, scalex.x, scalex.y, scalex.z, scalev.x, scalev.y, scalev.z);
						errorx = fmax(errorx, errorv);
//if(idy == 0) printf("error %d %.20g %d %d %d %g %g\n", idi, errorx, n, tt, ff, t, dt1);
						if(errorx >= def_tol * def_tol){
							stop_s[0] = 0;
						}
						Ncol_s[0] = 0;
						Coltime_s[0] = 10.0;
					}
				}
				__syncthreads();
				if(stop_s[0] == 1 || sgnt * dt1 < def_dtmin){	
//if(idy == 0) printf("acceptA %d %d %d %.20g %.20g %.20g\n", tt, ff, n, dt, t, dt1);
					for(int i = 0; i < NN; i += Bl){
						volatile int Ne = -1; //number of pairs
						volatile int idi = 0;
						if(idy + i < N2){
							idi = Encpairs2_d[start + idy + i].x;
							Ne = Encpairs_d[idi].y;
						}
						if(Ne >= 0){
							double4 xt, vt;

							xt.x = dx_d[0 * NT + idi].x;
							xt.y = dx_d[0 * NT + idi].y;
							xt.z = dx_d[0 * NT + idi].z;
							xt.w = xold_d[idi].w;

							vt.x = dv_d[0 * NT + idi].x;
							vt.y = dv_d[0 * NT + idi].y;
							vt.z = dv_d[0 * NT + idi].z;
							vt.w = vold_d[idi].w;
///*if(idi == 3159) */printf("xt %d %d %.20g %.20g %.20g %.20g %.20g %.20g %d\n", idi, 0, xt.x, xt.y, xt.z, vt.x, vt.y, vt.z, n);
							for(int j = 1; j < n; ++j){
								xt.x += dx_d[j * NT + idi].x;
								xt.y += dx_d[j * NT + idi].y;
								xt.z += dx_d[j * NT + idi].z;

								vt.x += dv_d[j * NT + idi].x;
								vt.y += dv_d[j * NT + idi].y;
								vt.z += dv_d[j * NT + idi].z;
///*if(idi == 3159)*/ printf("xt %d %d %.20g %.20g %.20g %.20g %.20g %.20g %d\n", idi, j, xt.x, xt.y, xt.z, vt.x, vt.y, vt.z, n);
							}
							xt_d[idi] = xt;
							vt_d[idi] = vt;
						}
					}
					__syncthreads();
					for(int i = 0; i < NN; i += Bl){
						volatile int Ne = -1; //number of pairs
						volatile int idi = 0;
						if(idy + i < N2){
							idi = Encpairs2_d[start + idy + i].x;
							Ne = Encpairs_d[idi].y;
						}
						volatile double dtgr;
						if(UseGR == 1){
							dtgr = dtgr_d[idi];
						}
						else dtgr = 1.0;
						for(int ii = 0; ii < Ne; ++ii){
							double delta = 100.0;
							double enct = 100.0;
							double colt = 100.0;
							volatile int j = Encpairs_d[idi * NencMax + ii].x;
							double rcrit = vold_d[idi].w + vold_d[j].w;
							if((noColl == 1 || noColl == -1) && index_d[idi] == CollTshiftpairs_c[0].x && index_d[j] == CollTshiftpairs_c[0].y){
								rcrit = vold_d[idi].w * CollTshift_c[0] + vold_d[j].w * CollTshift_c[0];
							}
							if((noColl == 1 || noColl == -1) && index_d[idi] == CollTshiftpairs_c[0].y && index_d[j] == CollTshiftpairs_c[0].x){
								rcrit = vold_d[idi].w * CollTshift_c[0] + vold_d[j].w * CollTshift_c[0];
							}
							if(CollisionPrecision_c[0] < 0.0){
								//do not overlap bodies when collision, increase therefore radius slightly, R + R * precision
								rcrit *= (1.0 - CollisionPrecision_c[0]);	
							}

							if(idi > j){
								delta = encounter1(xt_d[idi], vt_d[idi], xold_d[idi], vold_d[idi], xt_d[j], vt_d[j], xold_d[j], vold_d[j], rcrit, dt1 * dtgr, idi, j, enct, colt, MinMass, noColl);
							}

							if((noColl == 1 || noColl == -1) && colt == 100.0){
								delta = 100.0;
							}
							if((noColl == 1 || noColl == -1) && colt == 200.0){
								noColl = 2;
								BSstop_d[0] = 3;
							}
							if(delta < rcrit*rcrit){
								int Ni = atomicAdd(&Ncol_s[0], 1);
								if(Ncol_s[0] >= def_MaxColl) Ni = def_MaxColl - 1;
//printf("EE1 %d %d %d %d | %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %d %d %g %d\n", idi, j, index_d[idi], index_d[j], xt_d[idi].w, xt_d[j].w, xt_d[idi].x, xt_d[idi].y, xt_d[idi].z, xt_d[j].x, xt_d[j].y, xt_d[j].z, delta, rcrit*rcrit, f, n, colt, Ni);
								if(xt_d[idi].w >= xt_d[j].w){
									Colpairs_s[Ni].x = idi;
									Colpairs_s[Ni].y = j;
								}
								else{
									Colpairs_s[Ni].x = j;
									Colpairs_s[Ni].y = idi;
								}
								Coltime_s[Ni] = colt;
					
					// *****************
					//dont group test particles
					/*			if(xt_d[idi].w == 0.0){
									Colpairs_s[Ni].x = idi;
									Colpairs_s[Ni].y = idi;
								}
								if(xt_d[j].w == 0.0){
									Colpairs_s[Ni].x = j;
									Colpairs_s[Ni].y = j;
								}
					*/
					// *****************
							}

							if(WriteEncounters_c[0] > 0 && noColl == 0){
								double writeRadius = 0.0;
								//in scales of planetary Radius
								writeRadius = WriteEncountersRadius_c[0] * fmax(vt_d[idi].w, vt_d[j].w);
								if(delta < writeRadius * writeRadius){

									if(enct > 0.0 && enct < 1.0){
										//ingnore encounters within the same particle cloud
										if(index_d[idi] / WriteEncountersCloudSize_c[0] != index_d[j] / WriteEncountersCloudSize_c[0]){

//printf("Write Enc %g %g %g %g %g %d %d\n", (t + dt1) / dayUnit, writeRadius, sqrt(delta), enct, colt, idi, j);
											int ne = atomicAdd(NWriteEnc_d, 1);
											if(ne >= def_MaxWriteEnc - 1) ne = def_MaxWriteEnc - 1;
											storeEncounters(xt_d, vt_d, idi, j, idi, j, index_d, ne, writeEnc_d, time + (t + dt1) / dayUnit, spin_d);
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
							double dx = xt_d[i].x - xt_d[j].x;
							double dy = xt_d[i].y - xt_d[j].y;
							double dz = xt_d[i].z - xt_d[j].z;
							double d = sqrt(dx * dx + dy * dy + dz * dz);
							double R = vt_d[i].w + vt_d[j].w;
				
							if((noColl == 1 || noColl == -1) && index_d[i] == CollTshiftpairs_c[0].x && index_d[j] == CollTshiftpairs_c[0].y){
								R = vt_d[i].w * CollTshift_c[0] + vt_d[j].w * CollTshift_c[0];
							}
							if(CollisionPrecision_c[0] < 0.0){
								//do not overlap bodies when collision, increase therefore radius slightly, R + R * precision
								R *= (1.0 - CollisionPrecision_c[0]);	
							}

							double dR = (R - d) / R;

							if(noColl == -1) dR = -dR;

//printf("dR512 %d %d %.20g %.20g %.20g\n", i, j, d, R, dR);
							if(dR > fabs(CollisionPrecision_c[0]) && d != 0.0){
								//bodies are already overlapping
								Coltime = fmin(Coltime_s[c], Coltime);
							}

						}
						Coltime_s[0] = Coltime;
//printf("ColtimeT %.20g %g %g %g %d %d %d %d %d\n", Coltime, t / dayUnit, dt1 / dayUnit, (1.0 - Coltime) * dt1, tt, ff, n, Ncol_s[0], Ncoll_d[0]);
					}
					__syncthreads();
					if(Coltime_s[0] == 10.0){
						if(idy == 0) {
							for(int c = 0; c < min(Ncol_s[0], def_MaxColl); ++c){
								int i = Colpairs_s[c].x;
								int j = Colpairs_s[c].y;
								if(xt_d[i].w >= 0 && xt_d[j].w >= 0){
									int nc = 0;
									if(noColl == 0 || ((noColl == 1 || noColl == -1) && index_d[i] == CollTshiftpairs_c[0].x && index_d[j] == CollTshiftpairs_c[0].y)){
										nc = atomicAdd(Ncoll_d, 1);
										if(nc >= def_MaxColl) nc = def_MaxColl - 1;
										if(noColl == 1 || noColl == -1){
											BSstop_d[0] = 3;
											noColl = 2;
										}
									}
									double test;
//printf("cTime coll BSA512%g %g %g %.20g %d %d %d\n", time, t / dayUnit, dt / dayUnit, time + (t + dt1) / dayUnit, index_d[i], index_d[j], nc);
									collide(random, xt_d, vt_d, i, j, i, j, Msun, U_d, test, index_d, nc, Coll_d, time + (t + dt1) / dayUnit, spin_d, love_d, createFlag_d, rcritv_d, rcrit_d, NconstT, NconstT, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, SLevels, noColl);
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

						for(int i = 0; i < NN; i += Bl){
							volatile int Ne = -1; //number of pairs
							volatile int idi = 0;
							if(idy + i < N2){
								idi = Encpairs2_d[start + idy + i].x;
								Ne = Encpairs_d[idi].y;
							}
							if(Ne >= 0){
								xold_d[idi] = xt_d[idi];
								vold_d[idi] = vt_d[idi];
//if(idy == 0) printf("update %d %d %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %g %g %d %d %d\n", idy, idx, idi, xold_d[idi].x, xold_d[idi].y, xold_d[idi].z, vold_d[idi].x, vold_d[idi].y, vold_d[idi].z, xold_d[idi].w, vold_d[idi].w, t, dt1, tt, ff, n);
							}
						}
					}
					else{
						dt1 *= Coltime_s[0];
//if(idy == 0) printf("reduce time step %d %g Coltime: %g %d %d noColl %d\n", idx, dt1 / dayUnit,  Coltime_s[0], f, n, noColl);
					}
					f = 0;
					__syncthreads();
					break;
				}
				__syncthreads();
				if(BSstop_d[0] == 3){
//if(idy == 0) printf("Stop BSA512 %d\n", idx);
					__syncthreads();
					return;
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
		volatile int Ne = -1; //number of pairs
		volatile int idi = 0;
		if(idy + i < N2){
			idi = Encpairs2_d[start + idy + i].x;
			Ne = Encpairs_d[idi].y;
		}
		if(Ne >= 0){
			x4_d[idi] = xold_d[idi];
			v4_d[idi] = vold_d[idi];
//if(idi == 0) printf("final %g %d %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.10g\n", time, idi, NN, xold_d[idi].x, xold_d[idi].y, xold_d[idi].z, vold_d[idi].x, vold_d[idi].y, vold_d[idi].z, xold_d[idi].w, vold_d[idi].w, rcritv_d[idi]);
		}
	}

#if USE_RANDOM == 1
	random_d[idx] = random;
#endif
}


template <int E>
__global__ void BSAcc_kernel(double4 *x4_d, double4 *v4_d, double4 *xA_d, double4 *vA_d, double4 *xB_d, double4 *vB_d, double *rcritv_d, int2 *Encpairs_d, int2 *Encpairs2_d, double *dt1_d, double *dtgr_d, const double Msun, const int st, const int NT, const int NconstT, const int NencMax, int *BSAstop_d, double *Coltime_d, const int n, const int f, const double MinMass, const int UseTestParticles, const double dt, const int SLevels, int noColl){

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = blockIdx.y;

	int si = Encpairs2_d[ (st+2) * NT + idx].y;
	int N2 = Encpairs2_d[si].y; //Number of bodies in  current BS simulation
	int start = Encpairs2_d[NT + si].y;
//if(n == 1 && E == 0) printf("BS %d %d %d %d %d\n", idx, st, si, N2, NT);
	if(id < N2){
		volatile int idi = Encpairs2_d[start + id].x;
		volatile int Ne;
		if(E == 0 && n == 1){
			if(f == 0){
				Ne = Encpairs_d[idi].y; //number of pairs
				Encpairs_d[idi + 7 * NT].y = Ne;
				Encpairs_d[idi + 6 * NT].y = Ne;
			}
			else{
				Ne = Encpairs_d[idi + 7 * NT].y; //number of pairs
				Encpairs_d[idi + 6 * NT].y = Ne;
			}
		}
		else Ne = Encpairs_d[idi + 6 * NT].y; //number of pairs
		if(Ne >= 0){
			double dt1 = dt1_d[idi];
			double dt2 = dt1 / (2.0 * n);
			double dtgr = dtgr_d[idi];
			if(E == 1 || E == 2) dt2 *= 2.0;

			double4 xAi = xA_d[idi];
			double4 vAi = vA_d[idi];

			double3 a = {0.0, 0.0, 0.0};
			double test;

			for(int i = 0; i < Ne; ++i){	
				int j = Encpairs_d[idi * NencMax + i].x;
//if(n == 1 && E == 0)  printf("%d %d %d %d %d %d %d %d %g\n", idx, id, idi, Ne, N2, start, j, n, dt1);
				double4 xAj = xA_d[j];

				accEnc(xAi, xAj, a, rcritv_d, test, idi, j, NconstT, MinMass, UseTestParticles, SLevels);		
		
			}	
			accEncSun(xAi, a, def_ksq * Msun * dtgr);
			double4 x4B;
			double4 v4B;

			if(E == 0){
				//here xA = x4, and xB = xp
				x4B.x = xAi.x + (dt2 * dtgr * vAi.x);
				x4B.y = xAi.y + (dt2 * dtgr * vAi.y);
				x4B.z = xAi.z + (dt2 * dtgr * vAi.z);
				x4B.w = xAi.w;
				v4B.x = vAi.x + (dt2 * a.x);
				v4B.y = vAi.y + (dt2 * a.y);
				v4B.z = vAi.z + (dt2 * a.z);
				v4B.w = vAi.w;
				Encpairs_d[si + 5 * NT].y = 0; //set accept condition
			}
			if(E == 1){
				//here xA = xp, and xB = xt
				double4 x4i = x4_d[idi];
				double4 v4i = v4_d[idi];
				x4B.x = x4i.x + (dt2 * dtgr * vAi.x);
				x4B.y = x4i.y + (dt2 * dtgr * vAi.y);
				x4B.z = x4i.z + (dt2 * dtgr * vAi.z);
				x4B.w = x4i.w;

				v4B.x = v4i.x + (dt2 * a.x);
				v4B.y = v4i.y + (dt2 * a.y);
				v4B.z = v4i.z + (dt2 * a.z);
				v4B.w = v4i.w;
			}
			if(E == 2 || E == 3){
				x4B = xB_d[idi];
				v4B = vB_d[idi];
				x4B.x += (dt2 * dtgr * vAi.x);
				x4B.y += (dt2 * dtgr * vAi.y);
				x4B.z += (dt2 * dtgr * vAi.z);

				v4B.x += (dt2 * a.x);
				v4B.y += (dt2 * a.y);
				v4B.z += (dt2 * a.z);
			}

			xB_d[idi] = x4B;
			vB_d[idi] = v4B;
//if(idi == 1 && E == 0 && n == 1) printf("xB %d %.20g %.20g %.20g %.20g %.20g %.20g %d\n", idi, xB_d[idi].x, xB_d[idi].y, xB_d[idi].z, vB_d[idi].x, vB_d[idi].y, vB_d[idi].z, n);
		}		
	}
	if(E == 0 && id == 0){
		BSAstop_d[0] = 1;
		Coltime_d[0] = 10.0;
	}
}

__global__ void BSError_kernel(double4 *x4_d, double4 *v4_d, double4 *xp_d, double4 *vp_d, double4 *xt_d, double4 *vt_d, double3 *dx_d, double3 *dv_d, int2 *Encpairs_d, int2 *Encpairs2_d, const int st, const int NT, const int n, const int f){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = blockIdx.y;

	int si = Encpairs2_d[ (st+2) * NT + idx].y;
	int N2 = Encpairs2_d[si].y; //Number of bodies in  current BS simulation
	int start = Encpairs2_d[NT + si].y;

	if(id < N2){
		volatile int idi = Encpairs2_d[start + id].x;
		volatile int Ne = Encpairs_d[idi + 6 * NT].y; //number of pairs
		if(Ne >= 0){
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

//if(idi == 1) printf("xp %d %.20g %.20g %.20g %.20g %.20g %.20g %d\n", idi, xp_d[idi].x, xp_d[idi].y, xp_d[idi].z, vp_d[idi].x, vp_d[idi].y, vp_d[idi].z, n);
//if(idi == 1) printf("xt %d %.20g %.20g %.20g %.20g %.20g %.20g %d\n", idi, xt_d[idi].x, xt_d[idi].y, xt_d[idi].z, vt_d[idi].x, vt_d[idi].y, vt_d[idi].z, n);


			dxj.x = 0.5 * (xt_d[idi].x + xp_d[idi].x);
			dx_d[(n - 1) * NT + idi].x = dxj.x;
			dxj.y = 0.5 * (xt_d[idi].y + xp_d[idi].y);
			dx_d[(n - 1) * NT + idi].y = dxj.y;
			dxj.z = 0.5 * (xt_d[idi].z + xp_d[idi].z);
			dx_d[(n - 1) * NT + idi].z = dxj.z;
			dvj.x = 0.5 * (vt_d[idi].x + vp_d[idi].x);
			dv_d[(n - 1) * NT + idi].x = dvj.x;
			dvj.y = 0.5 * (vt_d[idi].y + vp_d[idi].y);
			dv_d[(n - 1) * NT + idi].y = dvj.y;
			dvj.z = 0.5 * (vt_d[idi].z + vp_d[idi].z);
			dv_d[(n - 1) * NT + idi].z = dvj.z;
//if(idi == 1) printf("dxj %d %d %.20g %.20g %.20g %.20g %.20g %.20g %d\n", idx, idi, dxj.x, dxj.y, dxj.z, dvj.x, dvj.y, dvj.z, n);

			double dj1;

			for(int j = n - 1; j >= 1; --j){

				double t0 = BSt0_c[(n-1) * 8 + (j-1)];
				double t1 = t0 * BSddt_c[j];
				double t2 = t0 * BSddt_c[n-1];
				
				dj1 = dx_d[(j - 1) * NT + idi].x;
				dxj.x = (t1 * dxj.x) - (t2 * dj1);
				dx_d[(j - 1) * NT + idi].x = dxj.x;
				dj1 = dx_d[(j - 1) * NT + idi].y;
				dxj.y = (t1 * dxj.y) - (t2 * dj1);
				dx_d[(j - 1) * NT + idi].y = dxj.y;
				dj1 = dx_d[(j - 1) * NT + idi].z;
				dxj.z = (t1 * dxj.z) - (t2 * dj1);
				dx_d[(j - 1) * NT + idi].z = dxj.z;
				dj1 = dv_d[(j - 1) * NT + idi].x;
				dvj.x = (t1 * dvj.x) - (t2 * dj1);
				dv_d[(j - 1) * NT + idi].x = dvj.x;
				dj1 = dv_d[(j - 1) * NT + idi].y;
				dvj.y = (t1 * dvj.y) - (t2 * dj1);
				dv_d[(j - 1) * NT + idi].y = dvj.y;
				dj1 = dv_d[(j - 1) * NT + idi].z;
				dvj.z = (t1 * dvj.z) - (t2 * dj1);
				dv_d[(j - 1) * NT + idi].z = dvj.z;
			}
			double error = dxj.x * dxj.x * scalex.x;
			error = fmax(error, dxj.y * dxj.y * scalex.y);
			error = fmax(error, dxj.z * dxj.z * scalex.z);
			error = fmax(error, dvj.x * dvj.x * scalev.x);
			error = fmax(error, dvj.y * dvj.y * scalev.y);
			error = fmax(error, dvj.z * dvj.z * scalev.z);
//if(idi == 1) printf("dx %d %d %.20g %.20g %.20g %.20g %.20g %.20g %d\n", idx, idi, dxj.x, dxj.y, dxj.z, dvj.x, dvj.y, dvj.z, n);
//if(idi == 1) printf("scale %d %d %.20g %.20g %.20g %.20g %.20g %.20g %d\n", idx, idi, scalex.x, scalex.y, scalex.z, scalev.x, scalev.y, scalev.z, n);
			if(error >= def_tol * def_tol){
				//dont accept BS step
				Encpairs_d[si + 5 * NT].y = 1; //Accept
//if(idi == 3534) printf("error %d %d %d %d %d %.20g %.20g %.20g %.20g\n", 0, f, n, id, idi, error, dxj.x, dxj.y, dxj.z);
			}
		}
	}
}


__global__ void BSAccept_kernel(double4 *xt_d, double4 *vt_d, double3 *dx_d, double3 *dv_d, int2 *Encpairs_d, int2 *Encpairs2_d, double *dt1_d, const int st, const int NT, const int n){

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = blockIdx.y;

	int si = Encpairs2_d[ (st+2) * NT + idx].y;
	int N2 = Encpairs2_d[si].y; //Number of bodies in  current BS simulation
	int start = Encpairs2_d[NT + si].y;

	if(id < N2){
		volatile int idi = Encpairs2_d[start + id].x;
		volatile int Ne = Encpairs_d[idi + 6 * NT].y; //group index
		if(Ne >= 0){
			int accept = Encpairs_d[si + 5 * NT].y;
			double3 xt;
			double3 vt;
			double dt1 = dt1_d[idi];
			int sgnt = 1;
			if(dt1 < 0.0) sgnt = -1;
			if(accept == 0 || sgnt * dt1 < def_dtmin){
				xt.x = dx_d[idi].x;
				xt.y = dx_d[idi].y;
				xt.z = dx_d[idi].z;
				vt.x = dv_d[idi].x;
				vt.y = dv_d[idi].y;
				vt.z = dv_d[idi].z;
//if(idi == 3159) printf("xt %d %d %.20g %.20g %.20g %.20g %.20g %.20g %d\n", idi, 0, xt.x, xt.y, xt.z, vt.x, vt.y, vt.z, n);
//if(idi == 566) printf("xt %d %d %.20g %.20g %.20g %.20g %.20g %.20g %d\n", idi, 0, xt.x, xt.y, xt.z, vt.x, vt.y, vt.z, n);

				for(int j = 1; j < n; ++j){
					xt.x += dx_d[j * NT + idi].x;
					xt.y += dx_d[j * NT + idi].y;
					xt.z += dx_d[j * NT + idi].z;
					vt.x += dv_d[j * NT + idi].x;
					vt.y += dv_d[j * NT + idi].y;
					vt.z += dv_d[j * NT + idi].z;
//if(idi == 3159) printf("xt %d %d %.20g %.20g %.20g %.20g %.20g %.20g %d\n", idi, j, xt.x, xt.y, xt.z, vt.x, vt.y, vt.z, n);
//if(idi == 566) printf("xt %d %d %.20g %.20g %.20g %.20g %.20g %.20g %d\n", idi, j, xt.x, xt.y, xt.z, vt.x, vt.y, vt.z, n);
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


__global__ void BSUpdate_kernel(curandState *random_d, double4 *xold_d, double4 *vold_d, double4 *x4_d, double4 *v4_d, double4 *xt_d, double4 *vt_d, double *rcrit_d, double *rcritv_d, int *index_d, double4 *spin_d, double3 *love_d, int *createFlag_d, int2 *Encpairs_d, int2 *Encpairs2_d, int *BSAstop_d, double *dt1_d, double *t1_d, const double dt, const double Msun, double *U_d, const int st, const int NT, const int NconstT, const int f, const int n, const int NencMax, int *Ncoll_d, double *Coll_d, const double time, float4 *aelimits_d, unsigned int *aecount_d, unsigned int *enccount_d, unsigned long long *aecountT_d, unsigned long long *enccountT_d, int *NWriteEnc_d, double *writeEnc_d, double *dtgr_d, double *Coltime_d, const double MinMass, const int UseTestParticles, const int SLevels, int noColl){

	int idx = blockIdx.y;	

#if USE_RANDOM == 1
	curandState random = random_d[idx];
#else
	int random = 0;
#endif

	int si = Encpairs2_d[ (st+2) * NT + idx].y;
	int N2 = Encpairs2_d[si].y; //Number of bodies in  current BS simulation
	int start = Encpairs2_d[NT + si].y;

	__shared__ int Ncol_s[1];
	__shared__ int2 Colpairs_s[def_MaxColl];
	__shared__ double Coltime_s[def_MaxColl];
	double test;
	Ncol_s[0] = 0;
	__syncthreads();
	for(int ii = 0; ii < N2; ii += blockDim.x){
		int id = ii + threadIdx.x;
		if(id < N2){
			volatile int idi = Encpairs2_d[start + id].x;
			volatile int Ne = Encpairs_d[idi + 6 * NT].y; //number of pairs
			if(Ne >= 0){
				int accept = Encpairs_d[si + 5 * NT].y;
				volatile double dt1 = dt1_d[idi];
				volatile double t1 = t1_d[idi];
				volatile double dtgr = dtgr_d[idi];
				int sgnt = 1;
				if(dt < 0.0) sgnt= -1;

				if(accept == 0 || sgnt * dt1 < def_dtmin){
//if(id == 0) printf("acceptA %d %d %d %d %d %d %.20g %.20g %.20g\n", idx, idi, si, n, accept, sgnt, dt, t1, dt1);

					for(int i = 0; i < Ne; ++i){
						double delta = 100.0;
						double enct = 100.0;
						double colt = 100.0;
						volatile int j = Encpairs_d[idi * NencMax + i].x;
						double rcrit = vold_d[idi].w + vold_d[j].w;
						if((noColl == 1 || noColl == -1) && index_d[idi] == CollTshiftpairs_c[0].x && index_d[j] == CollTshiftpairs_c[0].y){
							rcrit = vold_d[idi].w * CollTshift_c[0] + vold_d[j].w * CollTshift_c[0];
						}
						if((noColl == 1 || noColl == -1) && index_d[idi] == CollTshiftpairs_c[0].y && index_d[j] == CollTshiftpairs_c[0].x){
							rcrit = vold_d[idi].w * CollTshift_c[0] + vold_d[j].w * CollTshift_c[0];
						}
						if(CollisionPrecision_c[0] < 0.0){
							//do not overlap bodies when collision, increase therefore radius slightly, R + R * precision
							rcrit *= (1.0 - CollisionPrecision_c[0]);	
						}

						if(idi > j){
							delta = encounter1(xt_d[idi], vt_d[idi], xold_d[idi], vold_d[idi], xt_d[j], vt_d[j], xold_d[j], vold_d[j], rcrit, dt1 * dtgr, idi, j, enct, colt, MinMass, noColl);
						}
//if(idi == 12888 && j == 11191) printf("delta %g %g %g\n", enct, colt, delta);
						if((noColl == 1 || noColl == -1) && colt == 100.0){
							delta = 100.0;
						}
						if((noColl == 1 || noColl == -1) && colt == 200.0){
							noColl = 2;
							BSAstop_d[0] = 3;
						}
						if(delta < rcrit*rcrit){
							int Ni = atomicAdd(&Ncol_s[0], 1);
							if(Ncol_s[0] >= def_MaxColl) Ni = def_MaxColl - 1;
//printf("EE1 %d %d %d %d | %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %d %d %g %d\n", idi, j, index_d[idi], index_d[j], xt_d[idi].w, xt_d[j].w, xt_d[idi].x, xt_d[idi].y, xt_d[idi].z, xt_d[j].x, xt_d[j].y, xt_d[j].z, delta, rcrit*rcrit, f, n, colt, Ni);
							if(xt_d[idi].w >= xt_d[j].w){
								Colpairs_s[Ni].x = idi;
								Colpairs_s[Ni].y = j;
							}
							else{
								Colpairs_s[Ni].x = j;
								Colpairs_s[Ni].y = idi;
							}
							Coltime_s[Ni] = colt;
				
							// *****************
							//dont group test particles
/*								if(xt_d[idi].w == 0.0){
								Colpairs_s[Ni].x = idi;
								Colpairs_s[Ni].y = idi;
							}
							if(xt_d[j].w == 0.0){
								Colpairs_s[Ni].x = j;
								Colpairs_s[Ni].y = j;
							}
*/
							// *****************
						}

						if(WriteEncounters_c[0] > 0 && noColl == 0){
							double writeRadius = 0.0;
							//in scales of planetary Radius
							writeRadius = WriteEncountersRadius_c[0] * fmax(vt_d[idi].w, vt_d[j].w);
							if(delta < writeRadius * writeRadius){

								if(enct > 0.0 && enct < 1.0){
									//ingnore encounters within the same particle cloud
									if(index_d[idi] / WriteEncountersCloudSize_c[0] != index_d[j] / WriteEncountersCloudSize_c[0]){
//printf("Write Enc %g %g %g %g %g %d %d\n", (t + dt1) / dayUnit, writeRadius, sqrt(delta), enct, colt, idi, j);
										int ne = atomicAdd(NWriteEnc_d, 1);
										if(ne >= def_MaxWriteEnc - 1) ne = def_MaxWriteEnc - 1;
										storeEncounters(xt_d, vt_d, idi, j, idi, j, index_d, ne, writeEnc_d, time + (t1 + dt1) / dayUnit, spin_d);
									}
								}
							}
						}
					}
				}
			}
		}
	}	
	__syncthreads();
	if(threadIdx.x == 0){
		double Coltime = 10.0;
		for(int c = 0; c < min(Ncol_s[0], def_MaxColl); ++c){
			int i = Colpairs_s[c].x;
			int j = Colpairs_s[c].y;

			//Calculate real separation at the end of the time step
			double dx = xt_d[i].x - xt_d[j].x;
			double dy = xt_d[i].y - xt_d[j].y;
			double dz = xt_d[i].z - xt_d[j].z;
			double d = sqrt(dx * dx + dy * dy + dz * dz);
			double R = vt_d[i].w + vt_d[j].w;

			if((noColl == 1 || noColl == -1) && index_d[i] == CollTshiftpairs_c[0].x && index_d[j] == CollTshiftpairs_c[0].y){
				R = vt_d[i].w * CollTshift_c[0] + vt_d[j].w * CollTshift_c[0];
			}
			if(CollisionPrecision_c[0] < 0.0){
				//do not overlap bodies when collision, increase therefore radius slightly, R + R * precision
				R *= (1.0 - CollisionPrecision_c[0]);	
			}

			double dR = (R - d) / R;

			if(noColl == -1) dR = -dR;

//printf("dRm %d %d %.20g %.20g %.20g | noColl %d\n", i, j, d, R, dR, noColl);
			if(dR > fabs(CollisionPrecision_c[0]) && d != 0.0){
				//bodies are already overlapping
				Coltime = fmin(Coltime_s[c], Coltime);
			}
		}
		Coltime_d[0] = Coltime;
//printf("ColtimeT BSAm %.20g %g %g %g %d %d %d\n", Coltime, t1_d[0] / dayUnit, dt1_d[0] / dayUnit, (1.0 - Coltime) * dt1_d[0], n, Ncol_s[0], Ncoll_d[0]);

	}

	__syncthreads();
	if(threadIdx.x == 0) {
		for(int c = 0; c < min(Ncol_s[0], def_MaxColl); ++c){
			int i = Colpairs_s[c].x;
			int j = Colpairs_s[c].y;
			volatile double t1 = t1_d[i];
			if(Coltime_d[0] == 10.0){
				if(xt_d[i].w >= 0 && xt_d[j].w >= 0){
					int nc = 0;
					if(noColl == 0 || ((noColl == 1 || noColl == -1) && index_d[i] == CollTshiftpairs_c[0].x && index_d[j] == CollTshiftpairs_c[0].y)){
						nc = atomicAdd(Ncoll_d, 1);
						if(nc >= def_MaxColl) nc = def_MaxColl - 1;
						if(noColl == 1 || noColl == -1){
							noColl = 2;
							BSAstop_d[0] = 3;
						}
					}
//printf("cTime coll BSAm %g %g %g %.20g %d %d %d | noColl %d\n", time, t1_d[i] / dayUnit, dt /dayUnit, time + (t1_d[i] + dt1_d[i]) / dayUnit, index_d[i], index_d[j], nc, noColl);
					collide(random, xt_d, vt_d, i, j, i, j, Msun, U_d, test, index_d, nc, Coll_d, time + (t1 + dt1_d[i]) / dayUnit, spin_d, love_d, createFlag_d, rcritv_d, rcrit_d, NconstT, NconstT, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, SLevels, noColl);
				}
			}
		}
	}
	__syncthreads();
	for(int ii = 0; ii < N2; ii += blockDim.x){
		int id = ii + threadIdx.x;
		if(id < N2){
			volatile int idi = Encpairs2_d[start + id].x;
			volatile int Ne = Encpairs_d[idi + 7 * NT].y; //number of pairs
			volatile int Ne1 = Encpairs_d[idi + 6 * NT].y; //number of pairs
//if(idi == 5942 || idi == 3472) printf("B %d %d %d %d\n", id, idi, Ne, Ne1);
			if(Ne >= 0){
				int accept = Encpairs_d[si + 5 * NT].y;
				volatile double dt1 = dt1_d[idi];
				volatile double t1 = t1_d[idi];
				int sgnt = 1;
				if(dt < 0.0) sgnt= -1;
//if(idi == 5942 || idi == 3472) printf("C %d %d %g %g %g %g %d %d %d\n", id, idi, Coltime_d[0], t1_d[idi], dt1_d[idi], 1.0 - CollisionPrecision_c[0]/dt1_d[idi], f, n, ii);
				if(Coltime_d[0] == 10.0) {
					if((accept == 0 || sgnt * dt1 < def_dtmin) && Ne1 >= 0){

						t1 += dt1;
						if(n >= 8) dt1 *= 0.55;
						if(n < 7) dt1 *= 1.3;
						if(sgnt * dt1 > sgnt * dt) dt1 = dt;
						if(sgnt * (t1 + dt1) > sgnt * dt) dt1 = dt - t1;
						if(sgnt * dt1 < def_dtmin) dt1 = sgnt * def_dtmin;

						xold_d[idi] = xt_d[idi];
						vold_d[idi] = vt_d[idi];
//if(idi == 12888 || idi == 11191) printf("update %d %d %.20g %.20g %.20g %.20g %.20g %.20g %g %g %g %d %d\n", idx, idi, xold_d[idi].x, xold_d[idi].y, xold_d[idi].z, vold_d[idi].x, vold_d[idi].y, vold_d[idi].z, t1 / dayUnit, dt1 / dayUnit, dt / dayUnit, f, n);
						dt1_d[idi] = dt1;
						t1_d[idi] = t1;
						Encpairs_d[idi + 6 * NT].y = -1;
					}
					else{
						if(n == 8 && Ne1 >= 0){
							dt1_d[idi] = 0.5 * dt1;
//if(id == 0) printf("continue %d %d %g %g %d %d\n", idx, idi, t1_d[idi] / dayUnit, dt1_d[idi] / dayUnit, f, n);
						}
					}
					if(sgnt * t1 >= sgnt * dt){
						//BS step finished
						x4_d[idi] = xt_d[idi];
						v4_d[idi] = vt_d[idi];
						Encpairs_d[idi + 7 * NT].y = -1;
//if(id == 0) printf("finished %d %d %g %.20g %.20g %.20g %.20g %.20g %.20g %d %d\n", idi, index_d[idi], x4_d[idi].w, x4_d[idi].x, x4_d[idi].y, x4_d[idi].z, v4_d[idi].x, v4_d[idi].y, v4_d[idi].z, f, n);
					}
					else{
//if(id == 0) printf("not finished %d %d %d %d %d\n", idx, idi, f, n, BSAstop_d[0]);
						if(BSAstop_d[0] != 3) BSAstop_d[0] = 0;
					}
				}
				else{
					dt1_d[idi] *= Coltime_d[0];
					BSAstop_d[0] = 2;
//if(id == 0) printf("reduce time step %d %g Coltime: %g %d %d noColl %d\n", idx, dt1_d[idi] / dayUnit,  Coltime_d[0], f, n, noColl);
				}
			}
		}
	}
#if USE_RANDOM == 1
	random_d[idx] = random;
#endif
}

__global__ void BSA_setdt_kernel(double *dt1_d, double *t1_d, const double dt, const int N, double ksqMsun, double4 *x4_d, double4 *v4_d, double *dtgr_d, const int UseGR){

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < N){
		if(UseGR == 1){// GR time rescale (Saha & Tremaine 1994)
			double c2 = def_cm * def_cm;
			double mu = ksqMsun;
			double rsq = x4_d[id].x * x4_d[id].x + x4_d[id].y * x4_d[id].y + x4_d[id].z * x4_d[id].z;
			double vsq = v4_d[id].x * v4_d[id].x + v4_d[id].y * v4_d[id].y + v4_d[id].z * v4_d[id].z;
			double ir = 1.0/sqrt(rsq);
			double ia = 2.0*ir-vsq/mu;
			dtgr_d[id] = 1.0 - 1.5 * mu * ia / c2;
		}
		else{
			dtgr_d[id] = 1.0;
		}
		dt1_d[id] = dt;
		t1_d[id] = 0.0;
	}
}

__host__ void Data::BSACall(const int st, const int b, const int Nm, const int si, const double t, const double FGt, int noColl){
	int Nt = 32;
	int Nb = (b + Nt - 1) / Nt;
	int N = N_h[0] + Nsmall_h[0];
	BSA_setdt_kernel <<< (N + 255) / 256, 256 >>> (dt1_d, t1_d, dt_h[0] * FGt, N, def_ksq * Msun_h[0].x, xold_d, vold_d, dtgr_d, P.UseGR);
	BSAstop_h[0] = 0;
	for(int f = 0; f < 100000; ++f){
		for(int n = 1; n <= 8; ++n){
//printf("%d %d %d\n", f, n, BSAstop_h[0]);
			if(BSAstop_h[0] == 1) break; 
			if(BSAstop_h[0] == 3) break;
			if(Ncoll_m[0] > def_MaxColl) break; 
			BSAcc_kernel < 0 > <<< dim3(Nb, Nm, 1), dim3(Nt, 1, 1) >>> (xold_d, vold_d, xold_d, vold_d, xp_d, vp_d, rcritv_d, Encpairs_d, Encpairs2_d, dt1_d, dtgr_d, Msun_h[0].x, st, N, NconstT, P.NencMax, BSAstop_d, Coltime_d, n, f, P.MinMass, P.UseTestParticles, dt_h[0], P.SLevels, noColl);
			BSAcc_kernel < 1 > <<< dim3(Nb, Nm, 1), dim3(Nt, 1, 1) >>> (xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, rcritv_d, Encpairs_d, Encpairs2_d, dt1_d, dtgr_d, Msun_h[0].x, st, N, NconstT, P.NencMax, BSAstop_d, Coltime_d, n, f, P.MinMass, P.UseTestParticles, dt_h[0], P.SLevels, noColl);
			for(int m = 2; m <= n; ++m){
				BSAcc_kernel < 2 > <<< dim3(Nb, Nm, 1), dim3(Nt, 1, 1) >>> (xold_d, vold_d, xt_d, vt_d, xp_d, vp_d, rcritv_d, Encpairs_d, Encpairs2_d, dt1_d, dtgr_d, Msun_h[0].x, st, N, NconstT, P.NencMax, BSAstop_d, Coltime_d, n, f, P.MinMass, P.UseTestParticles, dt_h[0], P.SLevels, noColl);
				BSAcc_kernel < 2 > <<< dim3(Nb, Nm, 1), dim3(Nt, 1, 1) >>> (xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, rcritv_d, Encpairs_d, Encpairs2_d, dt1_d, dtgr_d, Msun_h[0].x, st, N, NconstT, P.NencMax, BSAstop_d, Coltime_d, n, f, P.MinMass, P.UseTestParticles, dt_h[0], P.SLevels, noColl);
			}
			BSAcc_kernel < 3 > <<< dim3(Nb, Nm, 1), dim3(Nt, 1, 1) >>> (xold_d, vold_d, xt_d, vt_d, xp_d, vp_d, rcritv_d, Encpairs_d, Encpairs2_d, dt1_d, dtgr_d, Msun_h[0].x, st, N, NconstT, P.NencMax, BSAstop_d, Coltime_d, n, f, P.MinMass, P.UseTestParticles, dt_h[0], P.SLevels, noColl);
			BSError_kernel <<< dim3(Nb, Nm, 1), dim3(Nt, 1, 1) >>> (xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, dx_d, dv_d, Encpairs_d, Encpairs2_d, st, N, n, f);
			BSAccept_kernel <<< dim3(Nb, Nm, 1), dim3(Nt, 1, 1) >>> (xt_d, vt_d, dx_d, dv_d, Encpairs_d, Encpairs2_d, dt1_d, st, N, n);
			BSUpdate_kernel <<< dim3(1, Nm, 1), dim3(Nt, 1, 1) >>> (random_d, xold_d, vold_d, x4_d, v4_d, xt_d, vt_d, rcrit_d, rcritv_d, index_d, spin_d, love_d, createFlag_d, Encpairs_d, Encpairs2_d, BSAstop_d, dt1_d, t1_d, dt_h[0] * FGt, Msun_h[0].x, U_d, st, N, NconstT, f, n, P.NencMax, Ncoll_d, Coll_d, t, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NWriteEnc_d, writeEnc_d, dtgr_d, Coltime_d, P.MinMass, P.UseTestParticles, P.SLevels, noColl);
			cudaMemcpy(BSAstop_h, BSAstop_d, sizeof(int), cudaMemcpyDeviceToHost);
//printf("A %d %d\n", BSAstop_h[0], Ncoll_m[0]);
			if(BSAstop_h[0] == 1) break;
			if(BSAstop_h[0] == 2) break;
			if(BSAstop_h[0] == 3) break;
			if(Ncoll_m[0] > def_MaxColl) break; 
		}
//printf("B %d %d\n", BSAstop_h[0], Ncoll_m[0]);
		if(BSAstop_h[0] == 1) break; 
		if(BSAstop_h[0] == 3) break;
		if(Ncoll_m[0] > def_MaxColl) break; 
	}
}

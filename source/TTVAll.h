
__device__ void  acc_ttv(double3 &ac, double4 &x4i, double4 &x4j, int j, int i){
	double rsq, ir, ir3, s;
	double3 r3ij;


	r3ij.x = x4j.x - x4i.x;
	r3ij.y = x4j.y - x4i.y;
	r3ij.z = x4j.z - x4i.z;

	rsq = (r3ij.x*r3ij.x) + (r3ij.y*r3ij.y) + (r3ij.z*r3ij.z);

	ir = 1.0/sqrt(rsq);
	ir3 = ir*ir*ir;

	s = (x4j.w * ir3) * (i != j);

	ac.x += __dmul_rn(r3ij.x, s);
	ac.y += __dmul_rn(r3ij.y, s);
	ac.z += __dmul_rn(r3ij.z, s);
}

//ny = 4
template <const int ny>
__global__ void ttv_step_kernel(double4 *x4_d, double4 *v4_d, double4 *xold_d, double4 *vold_d, double *dt_d, const double dt0, double2 *Msun_d, const int N, const int Nst, const int nsteps, double *time_d, double *timeold_d, double *lastTransitTime_d, int *transitIndex_d, int2 *NtransitsT_d, double *TransitTime_d, double2 *TransitTimeObs_d, int2 *EpochCount_d, double *TTV_d, const int PrintTransits){

	int idx = threadIdx.x;

	int ity = threadIdx.y;
	int si = blockIdx.x * blockDim.y + ity; //simulation index

	__shared__ int store_s[ny];
	__shared__ int read_s[ny];
	__shared__ double dt_s[ny];


	if(idx < N && si < Nst){

		double Msun = Msun_d[si].x;
		double Rsun = Msun_d[si].y;

		double4 x4 = x4_d[si * N + idx];
		double4 v4 = v4_d[si * N + idx];
	
		if(idx == 0){
			dt_s[ity] = dt_d[si];
		}
		__syncthreads();

		double time = time_d[si];
		//double time = time_d[si] - dt_d[si] / dayUnit;

		double4 xold = x4;
		double4 vold = v4;
		double timeold = time;

		
		for(int t = 0; t < nsteps; ++t){

			if(idx == 0){
				store_s[ity] = 0;
				read_s[ity] = 0;
			}
			__syncthreads();

			//Kick

			double3 a = {0.0, 0.0, 0.0};
			for(int j = 0; j < N; ++j){
				double4 x4j;
				x4j.x = __shfl_sync(0xffffffff, x4.x, ity * blockDim.x + j, warpSize);
				x4j.y = __shfl_sync(0xffffffff, x4.y, ity * blockDim.x + j, warpSize);
				x4j.z = __shfl_sync(0xffffffff, x4.z, ity * blockDim.x + j, warpSize);
				x4j.w = __shfl_sync(0xffffffff, x4.w, ity * blockDim.x + j, warpSize);
//printf("Kick %d %d %d %.20g %.20g\n", si, idx, j, x4.w, x4j.w);
				acc_ttv(a, x4, x4j, j, idx);
			}
			
			
			v4.x += __dmul_rn(a.x, dt_s[ity] * 0.5);
			v4.y += __dmul_rn(a.y, dt_s[ity] * 0.5);
			v4.z += __dmul_rn(a.z, dt_s[ity] * 0.5);

			//HC
			a = {0.0, 0.0, 0.0};
			for(int j = 0; j < N; ++j){
				double4 v4j;
				double mj;
				//by using with = warpSize, all sources lines in the warp start at 0, therefore ity * blockDim.x sets the new starting point
				//using a width of 8 would also work, but is not as flexible
				v4j.x = __shfl_sync(0xffffffff, v4.x, ity * blockDim.x + j, warpSize);	
				v4j.y = __shfl_sync(0xffffffff, v4.y, ity * blockDim.x + j, warpSize);
				v4j.z = __shfl_sync(0xffffffff, v4.z, ity * blockDim.x + j, warpSize);
				mj =    __shfl_sync(0xffffffff, x4.w, ity * blockDim.x + j, warpSize);
				a.x += mj * v4j.x;
				a.y += mj * v4j.y;
				a.z += mj * v4j.z;
//printf("HCA %d %d %d %.20g %.20g %.20g %.20g\n", si, idx, j, v4.x, v4j.x, mj, a.x);
			}
		
			double dt05Msun = dt_s[ity] * 0.5 / Msun;
			x4.x += __dmul_rn(a.x, dt05Msun);
			x4.y += __dmul_rn(a.y, dt05Msun);
			x4.z += __dmul_rn(a.z, dt05Msun);


			//FG
			//fgcfull(x4, v4, dt_s[ity], def_ksq * Msun, GR);
			double test;
			float4 aelimits = {0.0f, 0.0f, 0.0f, 0.0f};
			unsigned int aecount = 0u;
			int UseGR = 0;
			int index = 0;
			int id = 0;
//printf("fg %d %d %g %g %.10g | %.10g %.10g %.10g\n", si, idx, time, dt_s[ity], x4.w, x4.x, x4.y, x4.z); 
			fgfull(x4, v4, dt_s[ity], def_ksq * Msun, test, test, Msun, aelimits, aecount, NULL, NULL, si, id, index, UseGR);

			//HC
			a = {0.0, 0.0, 0.0};
			for(int j = 0; j < N; ++j){
				double4 v4j;
				double mj;
				v4j.x = __shfl_sync(0xffffffff, v4.x, ity * blockDim.x + j, warpSize);
				v4j.y = __shfl_sync(0xffffffff, v4.y, ity * blockDim.x + j, warpSize);
				v4j.z = __shfl_sync(0xffffffff, v4.z, ity * blockDim.x + j, warpSize);
				mj =    __shfl_sync(0xffffffff, x4.w, ity * blockDim.x + j, warpSize);
				a.x += mj * v4j.x;
				a.y += mj * v4j.y;
				a.z += mj * v4j.z;
//printf("HCA %d %d %g %g %g %g\n", idx, j, v4.x, v4j.x, mj, a.x);
			}

			x4.x += __dmul_rn(a.x, dt05Msun);
			x4.y += __dmul_rn(a.y, dt05Msun);
			x4.z += __dmul_rn(a.z, dt05Msun);

			//Kick
			a = {0.0, 0.0, 0.0};
			for(int j = 0; j < N; ++j){
				double4 x4j;
				x4j.x = __shfl_sync(0xffffffff, x4.x, ity * blockDim.x + j, warpSize);
				x4j.y = __shfl_sync(0xffffffff, x4.y, ity * blockDim.x + j, warpSize);
				x4j.z = __shfl_sync(0xffffffff, x4.z, ity * blockDim.x + j, warpSize);
				x4j.w = __shfl_sync(0xffffffff, x4.w, ity * blockDim.x + j, warpSize);
//printf("Kick %d %d %g %g\n", idx, j, x4.x, x4j.x);
				acc_ttv(a, x4, x4j, j, idx);
			}

			v4.x += __dmul_rn(a.x, dt_s[ity] * 0.5);
			v4.y += __dmul_rn(a.y, dt_s[ity] * 0.5);
			v4.z += __dmul_rn(a.z, dt_s[ity] * 0.5);
	
			time += dt_s[ity] / dayUnit;


// **************** TTV
			//calculate acceleration from the central star
			double rsq = x4.x*x4.x + x4.y*x4.y + x4.z*x4.z + 1.0e-30;
			double ir = 1.0 / sqrt(rsq);
			double ir3 = ir * ir * ir;
			double s = - def_ksq * Msun * ir3;

			a.x += s * x4.x;
			a.y += s * x4.y;
			a.z += s * x4.z;

			double g = x4.x * v4.x + x4.y * v4.y;
			double gd = v4.x * v4.x + v4.y * v4.y + x4.x * a.x + x4.y * a.y;
			double rsky = sqrt(x4.x * x4.x + x4.y * x4.y);
			double R = Rsun + v4.w;


			double dt1 = dt_s[ity];


			if(-g / (gd * dt1) < 0.0) dt1 = -dt1;
			if(fabs(g / gd) < fabs(dt1)) dt1 = -g /gd;


//if(idx == 0) printf("%g %g %g %g | %g %g | %.15g %.15g\n", g/gd, g, gd, x4.z, 1.2 * dt0, 1.2 * fabs(dt_s[ity]), lastTransitTime_d[si * N + idx], time);

			//first check
			//if(dt_s[ity] == dt0 && lastTransitTime_d[si * N + idx] < time - 2.0 * dt0 / dayUnit) {
			if(transitIndex_d[si] > N && lastTransitTime_d[si * N + idx] < time - 2.0 * dt0 / dayUnit) {

				if(x4.z > 0.0 && gd > 0.0 && fabs(g / gd) < 1.2 * fabs(dt0)){

					if((dt_s[ity] > 0.0 && g <= 0.0) || (dt_s[ity] <= 0.0 && g >= 0.0)){

						store_s[ity] = 1;
						atomicMin(&transitIndex_d[si], idx);
						__syncthreads();
//printf("check0 %d %d %g %g %.20g\n", idx, transitIndex_d[si], dt1, dt_s[ity], time);
					}
				}
			}
			else if(idx == transitIndex_d[si]){

//printf("check1 %d %g %g %.20g\n", idx, dt1, dt_s[ity], time);

				if(fabs(g / gd) < def_TransitTol){

					lastTransitTime_d[si * N + idx] = time;
					read_s[ity] = 1;
	
					int Epoch;
					int EpochObs;


					Epoch = NtransitsT_d[si * N + idx].x++;
					EpochObs = EpochCount_d[si * N + idx].x++;

					if(PrintTransits > 0){
						TransitTime_d[(si * N + idx) * def_NtransitTimeMax + Epoch + 1] = time;
					}

					double P = TransitTimeObs_d[idx * def_NtransitTimeMax].y; //period
					

					double2 TObs0 = TransitTimeObs_d[idx * def_NtransitTimeMax + EpochObs + 1];
					//double2 TObs1 = TransitTimeObs_d[idx * def_NtransitTimeMax + EpochObs + 2];

//if(idx == 2) printf("Transit %d %.10g %d | %.10g %d\n", idx, time, Epoch, TObs0.x, EpochObs);

					if(TObs0.x != 0.0){

						if(fabs(time - TObs0.x) < fabs(time + P - TObs0.x)){ 

							double p = 0;
							if(rsky < R){
								p = (time - TObs0.x) / TObs0.y;
								TTV_d[si * N + idx] += p * p * 0.5;
								++EpochCount_d[si * N + idx].y;
							}
//if(idx == 2) printf("    match %.10g %.10g %.10g %.10g\n", time, TObs0.x, p * p * 0.5, TTV_d[si * N + idx]);

						}
						else{
//if(idx == 2) printf("    wait %.10g %.10g\n", time, TObs0.x);
							--EpochCount_d[si * N + idx].x;
						}
					}


				}
			}

			__syncthreads();

			if(idx == transitIndex_d[si]){
				dt_s[ity] = dt1;
			}
			__syncthreads();
			if(store_s[ity] == 1){
//printf("store old coordinates %d %.10g\n", idx, timeold);
				xold_d[si * N + idx] = xold;
				vold_d[si * N + idx] = vold;
				timeold_d[si] = timeold;
			}

			if(read_s[ity] == 1){
//printf("read old coordinates %d %.10g\n", idx, timeold_d[si]);
				dt_s[ity] = dt0;
				x4 = xold_d[si * N + idx];
				v4 = vold_d[si * N + idx];
				time = timeold_d[si];
				transitIndex_d[si] = 10 * N; //large number
			}


// *****************

		} // end of t loop
		x4_d[si * N + idx].x = x4.x;
		x4_d[si * N + idx].y = x4.y;
		x4_d[si * N + idx].z = x4.z;
		x4_d[si * N + idx].w = x4.w;
		v4_d[si * N + idx].x = v4.x;
		v4_d[si * N + idx].y = v4.y;
		v4_d[si * N + idx].z = v4.z;

		if(idx == 0){
			dt_d[si] = dt_s[ity];
			time_d[si] = time;
		}
	}
}


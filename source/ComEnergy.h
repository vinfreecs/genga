#include "define.h"

#ifndef COMENERGY_H
#define COMENERGY_H


// *********************************************************
// This kernel computes the kinetic energy of the center of mass.
// It converts the velocities between heliocentric and democratic coordinates
// Must be followed by comd2 and comd3
//
// using vold as temporary storage
//
// Author: Simon Grimm
// May 2019
// ***********************************************************

__global__ void comd1_kernel(double4 *x4_d, double4 *v4_d, double4 *vold_d, const int N){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	double4 p = {0.0, 0.0, 0.0, 0.0};

	extern __shared__ double4 comd1_s[];
	double4 *p_s = comd1_s;


	int lane = threadIdx.x % warpSize;
	int warp = threadIdx.x / warpSize;

	if(warp == 0){
		p_s[threadIdx.x].x = 0.0;
		p_s[threadIdx.x].y = 0.0;
		p_s[threadIdx.x].z = 0.0;
		p_s[threadIdx.x].w = 0.0;
	}


	for(int i = 0; i < N; i += blockDim.x * gridDim.x){	//gridDim.x is for multiple block reduce
		if(id + i < N){
			double m = x4_d[id + i].w;
			double4 v4i = v4_d[id + i];
			if(m > 0.0){
				p.x += m * v4i.x;
				p.y += m * v4i.y;
				p.z += m * v4i.z;
				p.w += m;
			}
		}
	}

	__syncthreads();

	for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
		p.x += __shfl_xor_sync(0xffffffff, p.x, i, warpSize);
		p.y += __shfl_xor_sync(0xffffffff, p.y, i, warpSize);
		p.z += __shfl_xor_sync(0xffffffff, p.z, i, warpSize);
		p.w += __shfl_xor_sync(0xffffffff, p.w, i, warpSize);
#else
		p.x += __shfld_xor(p.x, i);
		p.y += __shfld_xor(p.y, i);
		p.z += __shfld_xor(p.z, i);
		p.w += __shfld_xor(p.w, i);
#endif
//if(i >= 16) printf("d1A %d %d %.20g\n", idy, i, p.x);
 	}
	__syncthreads();

	if(blockDim.x > warpSize){
		//reduce across warps
		if(lane == 0){
			p_s[warp] = p;
		}
		__syncthreads();
		//reduce previous warp results in the first warp
		if(warp == 0){
			p = p_s[threadIdx.x];
			for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
				p.x += __shfl_xor_sync(0xffffffff, p.x, i, warpSize);
				p.y += __shfl_xor_sync(0xffffffff, p.y, i, warpSize);
				p.z += __shfl_xor_sync(0xffffffff, p.z, i, warpSize);
				p.w += __shfl_xor_sync(0xffffffff, p.w, i, warpSize);
#else
				p.x += __shfld_xor(p.x, i);
				p.y += __shfld_xor(p.y, i);
				p.z += __shfld_xor(p.z, i);
				p.w += __shfld_xor(p.w, i);
#endif
//if(i >= 16) printf("d1B %d %d %.20g\n", idy, i, p.x);
			}
		}
	}
	__syncthreads();
	if(threadIdx.x == 0){
		vold_d[blockIdx.x] = p;
	}
}

// *********************************************************
// This kernel reads the result from the multiple thread block kernel comd1
// and performs the last summation step in
// --a single thread block --
//
// using vold as temporary storage
//
// Must be followed by comd3
// Author: Simon Grimm
// May 2019
// ***********************************************************
__global__ void comd2_kernel(double4 *vold_d, const int N){

	int idy = threadIdx.x;

	double4 p = {0.0, 0.0, 0.0, 0.0};

	extern __shared__ double4 comd2_s[];
	double4 *p_s = comd2_s;

	int lane = threadIdx.x % warpSize;
	int warp = threadIdx.x / warpSize;

	if(warp == 0){
		p_s[threadIdx.x].x = 0.0;
		p_s[threadIdx.x].y = 0.0;
		p_s[threadIdx.x].z = 0.0;
		p_s[threadIdx.x].w = 0.0;
	}


	if(idy < N){
		p = vold_d[idy];
	}

	__syncthreads();

	for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
		p.x += __shfl_xor_sync(0xffffffff, p.x, i, warpSize);
		p.y += __shfl_xor_sync(0xffffffff, p.y, i, warpSize);
		p.z += __shfl_xor_sync(0xffffffff, p.z, i, warpSize);
		p.w += __shfl_xor_sync(0xffffffff, p.w, i, warpSize);
#else
		p.x += __shfld_xor(p.x, i);
		p.y += __shfld_xor(p.y, i);
		p.z += __shfld_xor(p.z, i);
		p.w += __shfld_xor(p.w, i);
#endif
//if(i >= 16) printf("d2A %d %d %.20g\n", idy, i, p.x);
	}
	__syncthreads();

	if(blockDim.x > warpSize){
		//reduce across warps
		if(lane == 0){
			p_s[warp] = p;
		}
		__syncthreads();
		//reduce previous warp results in the first warp
		if(warp == 0){
			p = p_s[threadIdx.x];
			for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
				p.x += __shfl_xor_sync(0xffffffff, p.x, i, warpSize);
				p.y += __shfl_xor_sync(0xffffffff, p.y, i, warpSize);
				p.z += __shfl_xor_sync(0xffffffff, p.z, i, warpSize);
				p.w += __shfl_xor_sync(0xffffffff, p.w, i, warpSize);
#else
				p.x += __shfld_xor(p.x, i);
				p.y += __shfld_xor(p.y, i);
				p.z += __shfld_xor(p.z, i);
				p.w += __shfld_xor(p.w, i);
#endif
//if(i >= 16) printf("d2B %d %d %.20g\n", idy, i, p.x);
			}
		}
	}
	__syncthreads();
	if(threadIdx.x == 0){
		vold_d[0] = p;
	}
}

// *********************************************************
// This kernel distributes the result from the multiple thread block kernel
// comd1 and comd2.
//
// Must be followed by comd3
//
// using vold as temporary storage
// 
// Author: Simon Grimm
// May 2019
// ***********************************************************
__global__ void comd3_kernel(double4 *x4_d, double4 *v4_d, double4 *vold_d, double3 *vcom_d, const double Msun, const int N, const int f){

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	double4 p = vold_d[0];
	double iMsun = 1.0 / Msun;


	if(id == 0 && f == 0){
		vcom_d[0].x = p.x;
		vcom_d[0].y = p.y;
		vcom_d[0].z = p.z;
	}
//if(id == 0) printf("comd3 %d %.20g %.20g %.20g %.20g\n", id, p.x, p.y, p.z, p.w);

	if(id < N){
		double m = x4_d[id].w;
		if(m >= 0.0 && f == 1){
			//Convert to Heliocentric coordinates
			v4_d[id].x += p.x * iMsun;
			v4_d[id].y += p.y * iMsun;
			v4_d[id].z += p.z * iMsun;
		}
		if(m >= 0.0 && f == -1){
			//Convert to Democratic coordinates
			double iMsunp = 1.0 / (Msun + p.w);
			v4_d[id].x -= p.x * iMsunp;
			v4_d[id].y -= p.y * iMsunp;
			v4_d[id].z -= p.z * iMsunp;
		}
	}
}


//multiple warps, but only 1 thread block
__global__ void comB_kernel(double4 *x4_d, double4 *v4_d, double3 *vcom_d, const double Msun, const int N, const int f){

	int idy = threadIdx.x;
	double4 p = {0.0, 0.0, 0.0, 0.0};

	for(int i = 0; i < N; i += blockDim.x){
		if(idy + i < N){
			double m = x4_d[idy + i].w;
			if(m > 0.0){
				p.x += m * v4_d[idy + i].x;
				p.y += m * v4_d[idy + i].y;
				p.z += m * v4_d[idy + i].z;
				p.w += m;
			}
		}
	}
	__syncthreads();

	for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
		p.x += __shfl_xor_sync(0xffffffff, p.x, i, warpSize);
		p.y += __shfl_xor_sync(0xffffffff, p.y, i, warpSize);
		p.z += __shfl_xor_sync(0xffffffff, p.z, i, warpSize);
		p.w += __shfl_xor_sync(0xffffffff, p.w, i, warpSize);
#else
		p.x += __shfld_xor(p.x, i);
		p.y += __shfld_xor(p.y, i);
		p.z += __shfld_xor(p.z, i);
		p.w += __shfld_xor(p.w, i);
#endif
//if(i >= 16) printf("BA %d %d %.20g\n", idy, i, p.x);
	}

	__syncthreads();
	if(blockDim.x > warpSize){
		//reduce across warps
		extern __shared__ double4 comB_s[];
		double4 *p_s = comB_s;

		int lane = threadIdx.x % warpSize;
		int warp = threadIdx.x / warpSize;
		if(warp == 0){
			p_s[threadIdx.x].x = 0.0;
			p_s[threadIdx.x].y = 0.0;
			p_s[threadIdx.x].z = 0.0;
			p_s[threadIdx.x].w = 0.0;
		}
		__syncthreads();

		if(lane == 0){
			p_s[warp] = p;
		}

		__syncthreads();
		//reduce previous warp results in the first warp
		if(warp == 0){
			p = p_s[threadIdx.x];
			for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
				p.x += __shfl_xor_sync(0xffffffff, p.x, i, warpSize);
				p.y += __shfl_xor_sync(0xffffffff, p.y, i, warpSize);
				p.z += __shfl_xor_sync(0xffffffff, p.z, i, warpSize);
				p.w += __shfl_xor_sync(0xffffffff, p.w, i, warpSize);
#else
				p.x += __shfld_xor(p.x, i);
				p.y += __shfld_xor(p.y, i);
				p.z += __shfld_xor(p.z, i);
				p.w += __shfld_xor(p.w, i);
#endif
//if(i >= 16) printf("BB %d %d %.20g\n", idy, i, p.x);
			}
			if(lane == 0){
				p_s[0] = p;
			}
		}
		__syncthreads();

		p = p_s[0];
	}
	__syncthreads();

	double iMsun = 1.0 / Msun;

	if(idy == 0){
		if(f == 0){
			vcom_d[0].x = p.x;
			vcom_d[0].y = p.y;
			vcom_d[0].z = p.z;
		}
	}
//if(idy == 0) printf("comB %d %.20g %.20g %.20g %.20g\n", idy, p.x, p.y, p.z, p.w);
	for(int i = 0; i < N; i += blockDim.x){
		if(idy + i < N){
			double m = x4_d[idy + i].w;
			if(m >= 0.0 && f == 1){
				//Convert to Heliocentric coordinates
				v4_d[idy + i].x += p.x * iMsun;
				v4_d[idy + i].y += p.y * iMsun;
				v4_d[idy + i].z += p.z * iMsun;
			}
			if(m >= 0.0 && f == -1){
				//Convert to Democratic coordinates
				double iMsunp = 1.0 / (Msun + p.w);
				v4_d[idy + i].x -= p.x * iMsunp;
				v4_d[idy + i].y -= p.y * iMsunp;
				v4_d[idy + i].z -= p.z * iMsunp;
			}
		}
	}
}

//multiple warps, but only 1 thread block
__global__ void comBM_kernel(double4 *x4_d, double4 *v4_d, double3 *vcom_d, double2 *Msun_d, int *N_d, int *NBS_d, const int Nst, const int f){

	int st = blockIdx.x;
	int idy = threadIdx.x;

	if(st < Nst){
		double4 p = {0.0, 0.0, 0.0, 0.0};
		double m;
		double4 v4i;

		int Ni = N_d[st];
		int NBS = NBS_d[st];

		double Msun = Msun_d[st].x;

		for(int i = 0; i < Ni; i += blockDim.x){
			if(idy + i < Ni){
				m = x4_d[NBS + idy + i].w;
				v4i = v4_d[NBS + idy + i];
				if(m > 0.0){
					p.x += m * v4i.x;
					p.y += m * v4i.y;
					p.z += m * v4i.z;
					p.w += m;
				}
			}
		}
		__syncthreads();

		for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
			p.x += __shfl_xor_sync(0xffffffff, p.x, i, warpSize);
			p.y += __shfl_xor_sync(0xffffffff, p.y, i, warpSize);
			p.z += __shfl_xor_sync(0xffffffff, p.z, i, warpSize);
			p.w += __shfl_xor_sync(0xffffffff, p.w, i, warpSize);
#else
			p.x += __shfld_xor(p.x, i);
			p.y += __shfld_xor(p.y, i);
			p.z += __shfld_xor(p.z, i);
			p.w += __shfld_xor(p.w, i);
#endif
	//if(i >= 16) printf("BA %d %d %.20g\n", idy, i, p.x);
		}

		__syncthreads();
		if(blockDim.x > warpSize){
			//reduce across warps
			extern __shared__ double4 comBM_s[];
			double4 *p_s = comBM_s;

			int lane = threadIdx.x % warpSize;
			int warp = threadIdx.x / warpSize;
			if(warp == 0){
				p_s[threadIdx.x].x = 0.0;
				p_s[threadIdx.x].y = 0.0;
				p_s[threadIdx.x].z = 0.0;
				p_s[threadIdx.x].w = 0.0;
			}
			__syncthreads();

			if(lane == 0){
				p_s[warp] = p;
			}

			__syncthreads();
			//reduce previous warp results in the first warp
			if(warp == 0){
				p = p_s[threadIdx.x];
				for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
					p.x += __shfl_xor_sync(0xffffffff, p.x, i, warpSize);
					p.y += __shfl_xor_sync(0xffffffff, p.y, i, warpSize);
					p.z += __shfl_xor_sync(0xffffffff, p.z, i, warpSize);
					p.w += __shfl_xor_sync(0xffffffff, p.w, i, warpSize);
#else
					p.x += __shfld_xor(p.x, i);
					p.y += __shfld_xor(p.y, i);
					p.z += __shfld_xor(p.z, i);
					p.w += __shfld_xor(p.w, i);
#endif
	//if(i >= 16) printf("BB %d %d %.20g\n", idy, i, p.x);
				}
				if(lane == 0){
					p_s[0] = p;
				}
			}
			__syncthreads();

			p = p_s[0];
		}
		__syncthreads();

		double iMsun = 1.0 / Msun;

		if(idy == 0){
			if(f == 0){
				vcom_d[st].x = p.x;
				vcom_d[st].y = p.y;
				vcom_d[st].z = p.z;
			}
		}
	//if(idy == 0) printf("comB %d %.20g %.20g %.20g %.20g\n", idy, p.x, p.y, p.z, p.w);
		for(int i = 0; i < Ni; i += blockDim.x){
			if(idy + i < Ni){
				m = x4_d[NBS + idy + i].w;
				if(m >= 0.0 && f == 1){
					//Convert to Heliocentric coordinates
					v4_d[NBS + idy + i].x += p.x * iMsun;
					v4_d[NBS + idy + i].y += p.y * iMsun;
					v4_d[NBS + idy + i].z += p.z * iMsun;
				}
				if(m >= 0.0 && f == -1){
					//Convert to Democratic coordinates
					double iMsunp = 1.0 / (Msun + p.w);
					v4_d[NBS + idy + i].x -= p.x * iMsunp;
					v4_d[NBS + idy + i].y -= p.y * iMsunp;
					v4_d[NBS + idy + i].z -= p.z * iMsunp;
				}
			}
		}
	}
}

//only 1 single warp
__global__ void comC_kernel(double4 *x4_d, double4 *v4_d, double3 *vcom_d, const double Msun, const int N, const int f){

	int idy = threadIdx.x;
	double4 p = {0.0, 0.0, 0.0, 0.0};

	for(int i = 0; i < N; i += blockDim.x){
		if(idy + i < N){
			double m = x4_d[idy + i].w;
			if(m > 0.0){
				p.x += m * v4_d[idy + i].x;
				p.y += m * v4_d[idy + i].y;
				p.z += m * v4_d[idy + i].z;
				p.w += m;
			}
		}
	}
	__syncthreads();

	for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
		p.x += __shfl_xor_sync(0xffffffff, p.x, i, warpSize);
		p.y += __shfl_xor_sync(0xffffffff, p.y, i, warpSize);
		p.z += __shfl_xor_sync(0xffffffff, p.z, i, warpSize);
		p.w += __shfl_xor_sync(0xffffffff, p.w, i, warpSize);
#else
		p.x += __shfld_xor(p.x, i);
		p.y += __shfld_xor(p.y, i);
		p.z += __shfld_xor(p.z, i);
		p.w += __shfld_xor(p.w, i);
#endif
//if(i >= 16) printf("C %d %d %.20g\n", idy, i, p.x);
	}

	__syncthreads();

	double iMsun = 1.0 / Msun;

	if(idy == 0){
		if(f == 0){
			vcom_d[0].x = p.x;
			vcom_d[0].y = p.y;
			vcom_d[0].z = p.z;
		}
	}
//if(idy == 0) printf("comC %d %.20g %.20g %.20g %.20g\n", idy, p.x, p.y, p.z, p.w);
	for(int i = 0; i < N; i += blockDim.x){
		if(idy + i < N){
			double m = x4_d[idy + i].w;
			if(m >= 0.0 && f == 1){
				//Convert to Heliocentric coordinates
				v4_d[idy + i].x += p.x * iMsun;
				v4_d[idy + i].y += p.y * iMsun;
				v4_d[idy + i].z += p.z * iMsun;
//printf("comC +1 %d %.20g %.20g %.20g %.20g %g | %.20g\n", idy + i, p.x, p.y, p.z, p.w, iMsun, p.x * iMsun);
			}
			if(m >= 0.0 && f == -1){
				//Convert to Democratic coordinates
				double iMsunp = 1.0 / (Msun + p.w);
				v4_d[idy + i].x -= p.x * iMsunp;
				v4_d[idy + i].y -= p.y * iMsunp;
				v4_d[idy + i].z -= p.z * iMsunp;
//printf("comC -1 %d %.20g %.20g %.20g %.20g %g | %.20g\n", idy + i, p.x, p.y, p.z, p.w, iMsunp, p.x * iMsunp);
			}
		}
	}
}

__host__ void Data::comCall(const int f){
	if(N_h[0] + Nsmall_h[0] <= WarpSize){
		comC_kernel <<< 1, WarpSize >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, N_h[0] + Nsmall_h[0], f);
	}
	else if(N_h[0] + Nsmall_h[0] <= 512){
		int nn = (N_h[0] + Nsmall_h[0] + WarpSize - 1) / WarpSize;
		comB_kernel <<< 1, nn * WarpSize, WarpSize * sizeof(double4) >>> (x4_d, v4_d, vcom_d, Msun_h[0].x, N_h[0] + Nsmall_h[0], f);
	}
	else{
		int nct = 512;
		int ncb = min((N_h[0] + Nsmall_h[0] + nct - 1) / nct, 1024);
		comd1_kernel <<< dim3(ncb, 1, 1), dim3(nct, 1, 1), WarpSize * sizeof(double4) >>> (x4_d, v4_d, vold_d, N_h[0] + Nsmall_h[0]);
		comd2_kernel <<< 1, ((ncb + WarpSize - 1) / WarpSize) * WarpSize, WarpSize * sizeof(double4)  >>> (vold_d, ncb);
		comd3_kernel <<<(N_h[0] + Nsmall_h[0] + FTX - 1)/FTX, FTX >>> (x4_d, v4_d, vold_d, vcom_d, Msun_h[0].x,  N_h[0] + Nsmall_h[0], f);
	}

}


// *********************************************************
// This kernel computes the kinetic energy of the center of mass
//
// Authors: Simon Grimm, Joachim Stadel
// October  2015
// ***********************************************************
template <int Bl, int Bl2, int Nmax >
__global__ void comM_kernel(double4 *x4_d, double4 *v4_d, double3 *vcom_d, const double2 *Msun_d, int *index_d, int *NBS_d, int NT, int ff, int Nstart){

	int idy = threadIdx.x;
	int id = blockIdx.x * Bl2 + idy - Nmax + Nstart;
	__shared__ volatile double4 p_s[Bl + Nmax / 2];
	__shared__ int st_s[Bl + Nmax / 2];
	volatile double Msun;
	int NBS;

	if(id < NT + Nstart && id >= Nstart){
		st_s[idy] = index_d[id] / def_MaxIndex;
		volatile double m = x4_d[id].w;
		if(m > 0.0){
			p_s[idy].x = m * v4_d[id].x;
			p_s[idy].y = m * v4_d[id].y;
			p_s[idy].z = m * v4_d[id].z;
			p_s[idy].w = m;
		}
		else{
			p_s[idy].x = 0.0;
			p_s[idy].y = 0.0;
			p_s[idy].z = 0.0;
			p_s[idy].w = 0.0;
		}
		Msun = Msun_d[st_s[idy]].x;
		NBS = NBS_d[st_s[idy]];
//printf("ComA1 %d %d %g %g %g %g\n", id, idy, p_s[idy].x, p_s[idy].y, p_s[idy].z, p_s[idy].w);

	}
	else{
		st_s[idy] = -idy-1;
		p_s[idy].x = 0.0;
		p_s[idy].y = 0.0;
		p_s[idy].z = 0.0;
		p_s[idy].w = 0.0;
		Msun = 0.0;
		NBS = -1;
	}
	//halo

	if(idy < Nmax / 2){
		//right
		if(id + Bl < NT + Nstart){
			st_s[idy + Bl] = index_d[id + Bl] / def_MaxIndex;
			volatile double m = x4_d[id + Bl].w;
			p_s[idy + Bl].x = m * v4_d[id + Bl].x;
			p_s[idy + Bl].y = m * v4_d[id + Bl].y;
			p_s[idy + Bl].z = m * v4_d[id + Bl].z;
			p_s[idy + Bl].w = m;
		}
		else{
			st_s[idy + Bl] = -idy-Bl-1;
			p_s[idy + Bl].x = 0.0;
			p_s[idy + Bl].y = 0.0;
			p_s[idy + Bl].z = 0.0;
			p_s[idy + Bl].w = 0.0;
		}
//printf("ComA2 %d %d %g %g %g %g\n", id, idy, p_s[idy + Bl].x, p_s[idy + Bl].y, p_s[idy + Bl].z, p_s[idy + Bl].w);
	}
	volatile int f;
	volatile double px;
	volatile double py;
	volatile double pz;
	volatile double pw;	
	if(Nmax >= 64){
		__syncthreads();
		f = ((st_s[idy] - st_s[idy + 32]) == 0);		//one if sti == stj, zero else
		px = (p_s[idy + 32].x) * f;	
		py = (p_s[idy + 32].y) * f;
		pz = (p_s[idy + 32].z) * f;
		pw = (p_s[idy + 32].w) * f;

		__syncthreads();
	
		p_s[idy].x += px;
		p_s[idy].y += py;
		p_s[idy].z += pz;
		p_s[idy].w += pw;
//printf("ComA3 %d %d %g %g %g %g\n", id, idy, p_s[idy].x, p_s[idy].y, p_s[idy].z, p_s[idy].w);
	}

	if(Nmax >= 32){
		__syncthreads();
		f = ((st_s[idy] - st_s[idy + 16]) == 0);		//one if sti == stj, zero else
		px = (p_s[idy + 16].x) * f;	
		py = (p_s[idy + 16].y) * f;
		pz = (p_s[idy + 16].z) * f;
		pw = (p_s[idy + 16].w) * f;

		__syncthreads();
	
		p_s[idy].x += px;
		p_s[idy].y += py;
		p_s[idy].z += pz;
		p_s[idy].w += pw;
	}
	if(Nmax >= 16){
		__syncthreads();
		f = ((st_s[idy] - st_s[idy + 8]) == 0);		//one if sti == stj, zero else
		px = (p_s[idy + 8].x) * f;	
		py = (p_s[idy + 8].y) * f;
		pz = (p_s[idy + 8].z) * f;
		pw = (p_s[idy + 8].w) * f;

		__syncthreads();
	
		p_s[idy].x += px;
		p_s[idy].y += py;
		p_s[idy].z += pz;
		p_s[idy].w += pw;
	}

	if(Nmax >= 8){
		__syncthreads();
		f = ((st_s[idy] - st_s[idy + 4]) == 0);		//one if sti == stj, zero else
		px = (p_s[idy + 4].x) * f;
		py = (p_s[idy + 4].y) * f;
		pz = (p_s[idy + 4].z) * f;
		pw = (p_s[idy + 4].w) * f;

		__syncthreads();

		p_s[idy].x += px;
		p_s[idy].y += py;
		p_s[idy].z += pz;
		p_s[idy].w += pw;
	}

	__syncthreads();

	f = ((st_s[idy] - st_s[idy + 2]) == 0);			//one if sti == stj, zero else
	px = (p_s[idy + 2].x) * f;
	py = (p_s[idy + 2].y) * f;
	pz = (p_s[idy + 2].z) * f;
	pw = (p_s[idy + 2].w) * f;

	__syncthreads();

	p_s[idy].x += px;
	p_s[idy].y += py;
	p_s[idy].z += pz;
	p_s[idy].w += pw;

	__syncthreads();

	f = ((st_s[idy] - st_s[idy + 1]) == 0);			//one if sti == stj, zero else
	px = (p_s[idy + 1].x) * f;
	py = (p_s[idy + 1].y) * f;
	pz = (p_s[idy + 1].z) * f;
	pw = (p_s[idy + 1].w) * f;

	__syncthreads();

	p_s[idy].x += px;
	p_s[idy].y += py;
	p_s[idy].z += pz;
	p_s[idy].w += pw;

	__syncthreads();
	//sum is complete, now distribute solution
	f = ((st_s[idy] - st_s[idy + 1]) == 0);
	px = (p_s[idy].x) * f + (1 - f) * p_s[idy + 1].x;
	py = (p_s[idy].y) * f + (1 - f) * p_s[idy + 1].y;
	pz = (p_s[idy].z) * f + (1 - f) * p_s[idy + 1].z;
	pw = (p_s[idy].w) * f + (1 - f) * p_s[idy + 1].w;

	__syncthreads();
	p_s[idy + 1].x = px;
	p_s[idy + 1].y = py;
	p_s[idy + 1].z = pz;
	p_s[idy + 1].w = pw;
	__syncthreads();

	f = ((st_s[idy] - st_s[idy + 2]) == 0);
	px = (p_s[idy].x) * f + (1 - f) * p_s[idy + 2].x;
	py = (p_s[idy].y) * f + (1 - f) * p_s[idy + 2].y;
	pz = (p_s[idy].z) * f + (1 - f) * p_s[idy + 2].z;
	pw = (p_s[idy].w) * f + (1 - f) * p_s[idy + 2].w;

	__syncthreads();
	p_s[idy + 2].x = px;
	p_s[idy + 2].y = py;
	p_s[idy + 2].z = pz;
	p_s[idy + 2].w = pw;
	__syncthreads();

	if(Nmax >= 8){
		f = ((st_s[idy] - st_s[idy + 4]) == 0);
		px = (p_s[idy].x) * f + (1 - f) * p_s[idy + 4].x;
		py = (p_s[idy].y) * f + (1 - f) * p_s[idy + 4].y;
		pz = (p_s[idy].z) * f + (1 - f) * p_s[idy + 4].z;
		pw = (p_s[idy].w) * f + (1 - f) * p_s[idy + 4].w;

		__syncthreads();
		p_s[idy + 4].x = px;
		p_s[idy + 4].y = py;
		p_s[idy + 4].z = pz;
		p_s[idy + 4].w = pw;
		__syncthreads();
	}

	if(Nmax >= 16){
		f = ((st_s[idy] - st_s[idy + 8]) == 0);
		px = (p_s[idy].x) * f + (1 - f) * p_s[idy + 8].x;
		py = (p_s[idy].y) * f + (1 - f) * p_s[idy + 8].y;
		pz = (p_s[idy].z) * f + (1 - f) * p_s[idy + 8].z;
		pw = (p_s[idy].w) * f + (1 - f) * p_s[idy + 8].w;

		__syncthreads();
		p_s[idy + 8].x = px;
		p_s[idy + 8].y = py;
		p_s[idy + 8].z = pz;
		p_s[idy + 8].w = pw;
		__syncthreads();
	}

	if(Nmax >= 32){
		f = ((st_s[idy] - st_s[idy + 16]) == 0);
		px = (p_s[idy].x) * f + (1 - f) * p_s[idy + 16].x;
		py = (p_s[idy].y) * f + (1 - f) * p_s[idy + 16].y;
		pz = (p_s[idy].z) * f + (1 - f) * p_s[idy + 16].z;
		pw = (p_s[idy].w) * f + (1 - f) * p_s[idy + 16].w;

		__syncthreads();
		p_s[idy + 16].x = px;
		p_s[idy + 16].y = py;
		p_s[idy + 16].z = pz;
		p_s[idy + 16].w = pw;
		__syncthreads();
	}

	if(Nmax >= 64){
		f = ((st_s[idy] - st_s[idy + 32]) == 0);
		px = (p_s[idy].x) * f + (1 - f) * p_s[idy + 32].x;
		py = (p_s[idy].y) * f + (1 - f) * p_s[idy + 32].y;
		pz = (p_s[idy].z) * f + (1 - f) * p_s[idy + 32].z;
		pw = (p_s[idy].w) * f + (1 - f) * p_s[idy + 32].w;

		__syncthreads();
		p_s[idy + 32].x = px;
		p_s[idy + 32].y = py;
		p_s[idy + 32].z = pz;
		p_s[idy + 32].w = pw;
		__syncthreads();
	}
//if(p_s[idy].x != p_s[idy].x) printf("CoM %d %g %g %g %d %d\n", id, p_s[idy].x, p_s[idy].y, p_s[idy].z, Nmax, ff);

	double iMsun = 1.0 / Msun;
	//now the sum is complete
	if(id == NBS && NBS >= Nstart && idy >= Nmax && idy < Bl - Nmax / 2){
		if(ff == 0){
			vcom_d[st_s[idy]].x = p_s[idy].x;
			vcom_d[st_s[idy]].y = p_s[idy].y;
			vcom_d[st_s[idy]].z = p_s[idy].z;
		}
	}
	if(id < NT + Nstart && id >= Nstart && idy >= Nmax && idy < Bl - Nmax / 2 && x4_d[id].w >= 0.0 && ff == 1){
		v4_d[id].x += p_s[idy].x * iMsun;
		v4_d[id].y += p_s[idy].y * iMsun;
		v4_d[id].z += p_s[idy].z * iMsun;
	}
	if(id < NT + Nstart && id >= Nstart && idy >= Nmax && idy < Bl - Nmax / 2 && x4_d[id].w >= 0.0 && ff == -1){
		double iMsunp = 1.0 / (Msun + p_s[idy].w);
		v4_d[id].x -= p_s[idy].x * iMsunp;
		v4_d[id].y -= p_s[idy].y * iMsunp;
		v4_d[id].z -= p_s[idy].z * iMsunp;
	}
}


#endif

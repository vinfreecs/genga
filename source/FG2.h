#ifndef FG_H
#define FG_H

#include "Host2.h"
#include "Orbit2.h"
#include "BSSingle.h"
#endif

extern __device__ __noinline__ void fgfull(double4 &, double4 &, double, double, double &, double &, const double, float4, int &, int *, int, int);

// **************************************
//The fg_kernel does a copy of the coordinates and calls the FG function to perform the Kepler drift.
//There are 2 different FG, and one Burlish Stoer function, fastest one is fastfg.
//This Kernel is launched wich NB/Bl blocks with Bl threads. NB is the next bigger number of N
//which is a power of two.
// *****************************************
template <int Bl>
__global__ void fg_kernel(double4 *x4_d, double4 *v4_d, double4 *xold_d, double4 *vold_d, double dt, const double Msun, double *test_d, int N, float4 *aelimits_d, int *aecount_d, int *Gridaecount_d, int si){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	__shared__ double4 x4_s[Bl];
	__shared__ double4 v4_s[Bl];
	if(id < N){
		int aecount = 0;
		x4_s[idy] = x4_d[id];
		v4_s[idy] = v4_d[id];
		__syncthreads();

		xold_d[id] = x4_s[idy];
		vold_d[id] = v4_s[idy];
		double test;
		float4 aelimits = aelimits_d[idy];
		//fastfg(x4_s[idy], v4_s[idy], dt, ksq * Msun, test, Msun, aelimits, aecount, Gridaecount_d, si, id);
		fgfull(x4_s[idy], v4_s[idy], dt, ksq * Msun, test, test, Msun, aelimits, aecount, Gridaecount_d, si, id);
		//BSSinglestep(x4_s[idy], v4_s[idy], Msun, dt, test, test);
		__syncthreads();
		x4_d[id] = x4_s[idy];
		v4_d[id] = v4_s[idy];
		if(si == 0){
		aecount_d[id] += aecount;
		}
	}
}
extern template __global__ void fg_kernel < 16 >(double4 *, double4 *, double4 *x, double4 *, double, const double, double *, int, float4 *, int *, int *, int);
extern template __global__ void fg_kernel < 32 >(double4 *, double4 *, double4 *x, double4 *, double, const double, double *, int, float4 *, int *, int *, int);




template <int Bl>
__global__ void fgsmall_kernel(double4 *x4_d, double4 *v4_d, double4 *xold_d, double4 *vold_d, double dt, const double Msun, double *test_d, double4 *x4small_d, double4 *v4small_d, double4 *xoldsmall_d, double4 *voldsmall_d, int Nsmall, int N, float4 *aelimits_d, float4 *aelimitssmall_d, int *aecount_d, int *aecountsmall_d, int *Gridaecount_d, int si){

        int idy = threadIdx.x;
        int id = blockIdx.x * blockDim.x + idy;

        __shared__ double4 x4_s[Bl];
        __shared__ double4 v4_s[Bl];

	float4 aelimits;
	int aecount = 0;

	if(id < N){
        	x4_s[idy] = x4_d[id];
        	v4_s[idy] = v4_d[id];
		aelimits = aelimits_d[id];
	        xold_d[id] = x4_s[idy];
        	vold_d[id] = v4_s[idy];
	}
	else if(id < N + Nsmall){
		x4_s[idy] = x4small_d[id - N];
		v4_s[idy] = v4small_d[id - N];
		aelimits = aelimitssmall_d[id - N];
	        xoldsmall_d[id - N] = x4_s[idy];
        	voldsmall_d[id - N] = v4_s[idy];
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
		aelimits.x = 0.0f;
		aelimits.y = 0.0f;
		aelimits.z = 0.0f;
		aelimits.w = 0.0f;
		
	}
        __syncthreads();
        
	double test;

	if(id < N + Nsmall){
        	//fastfg(x4_s[idy], v4_s[idy], dt, ksq * Msun, test, Msun, aelimits, aecount, Gridaecount_d, si, id);
        	fgfull(x4_s[idy], v4_s[idy], dt, ksq * Msun, test, test, Msun, aelimits, aecount, Gridaecount_d, si, id);
        	//BSSinglestep(x4_s[idy], v4_s[idy], Msun, dt, test, id);
        }
        __syncthreads();

	if(id < N){
        	x4_d[id] = x4_s[idy];
        	v4_d[id] = v4_s[idy];
		if(si == 0){
			aecount_d[id] += aecount;
		}
	}
	else if (id < N + Nsmall){
                x4small_d[id - N] = x4_s[idy];
                v4small_d[id - N] = v4_s[idy];
		if(si == 0){
			aecountsmall_d[id - N] += aecount;
		}
	}

}
extern template __global__ void fgsmall_kernel < 128 > (double4 *, double4 *, double4 *, double4 *, double, const double, double *, double4 *, double4 *, double4 *, double4 *, int, int, float4 *, float4 *, int *, int *, int *, int);

template <int Bl>
__global__ void fgM_kernel(double4 *x4_d, double4 *v4_d, double4 *xold_d, double4 *vold_d, double dt, const double *Msun_d, double *test_d, int *index_d, int NT, float4 *aelimits_d, int *aecount_d, int *Gridaecount_d, double FGt, int si){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;
	int st = index_d[id] / 100;

	__shared__ double4 x4_s[Bl];
	__shared__ double4 v4_s[Bl]; 

	if(id < NT){
		int aecount = 0;
		x4_s[idy] = x4_d[id];
		v4_s[idy] = v4_d[id];
		__syncthreads();

		xold_d[id] = x4_s[idy];
		vold_d[id] = v4_s[idy];
		double test;
		double Msun = Msun_d[st];
		float4 aelimits = aelimits_d[id];
		//fastfg(x4_s[idy], v4_s[idy], dt * FGt, ksq * Msun, test, Msun, aelimits, aecount, Gridaecount_d, si, id);
		fgfull(x4_s[idy], v4_s[idy], dt * FGt, ksq * Msun, test, test, Msun, aelimits, aecount, Gridaecount_d, si, id);
		//BSSinglestep(x4_s[idy], v4_s[idy], Msun, dt * FGt, test, test);
		__syncthreads();
		x4_d[id] = x4_s[idy];
		v4_d[id] = v4_s[idy];

		if(si == 0){
			aecount_d[id] += aecount;
		}
	}
}
extern template __global__ void fgM_kernel < 128 > (double4 *, double4 *, double4 *, double4 *, double, const double *, double *, int *, int, float4 *, int *, int *, double, int);

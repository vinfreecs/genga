#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include "Host2.h"
#include "Orbit2.h"
#include "output.h"
#include "Kick3.h"
#include "HC.h"
#include "FG2.h"
#include "Encounter3.h"
#include "BSB.h"
#include "BSB64.h"
#include "BSBsmall.h"
#include "BSBM.h"
#endif

extern __host__ void  SymplecticP();
extern __global__ void test_kernel(double4 *, double3 *, int *, int);

extern __global__ void initialsmall_kernel(int2 *, int2 *, double3 *, int, int, int, int2 *, int2 *, double3 *, int , const int);
extern __host__ void firstKick_16();
extern __host__ void firstKick_32();
extern __host__ void firstKick_64();
extern __host__ void firstKick_128();
extern __host__ void firstKick_256();
extern __host__ void firstKick_512();
extern __host__ void firstKick_1024();
extern __host__ void firstKick_2048();
extern __host__ void firstKick_small();
extern __host__ void firstKick_M();

extern __host__ int step_16(double);
extern __host__ int step_32(double);
extern __host__ int step_64(double);
extern __host__ int step_128(double);
extern __host__ int step_256(double);
extern __host__ int step_512(double);
extern __host__ int step_1024(double);
extern __host__ int step_2048(double);
extern __host__ int step_small(double);
extern __host__ int step_M(double);


// **************************************
// This kernel sets initial values for the Encouter pair arrays
template <int Bl, int NB>
__global__ void initial_kernel(int2 *Encpairs_d, int2 *Encpairs2_d, double3 *acck_d){
	int idy = threadIdx.x;
	int idx = blockIdx.x;

	for(int i = 0; i < NB; i += Bl){
		Encpairs_d[(idy +i)* NB + idx].x = -1;
		Encpairs_d[(idy +i)* NB + idx].y = -1;

		Encpairs2_d[(idy +i)* NB + idx].x = -1;
		Encpairs2_d[(idy +i)* NB + idx].y = -1;
		
		if(idx == 0){
			acck_d[idy + i].x = 0.0;
			acck_d[idy + i].y = 0.0;
			acck_d[idy + i].z = 0.0;
		}
	}
}
extern template __global__ void initial_kernel <16, 16> (int2 *, int2 *, double3 *);
extern template __global__ void initial_kernel <32, 32> (int2 *, int2 *, double3 *);
extern template __global__ void initial_kernel <64, 64> (int2 *, int2 *, double3 *);
extern template __global__ void initial_kernel <128, 128> (int2 *, int2 *, double3 *);
extern template __global__ void initial_kernel <256, 256> (int2 *, int2 *, double3 *);
extern template __global__ void initial_kernel <512, 512> (int2 *, int2 *, double3 *);
extern template __global__ void initial_kernel <1024, 1024> (int2 *, int2 *, double3 *);
extern template __global__ void initial_kernel <2048, 2048> (int2 *, int2 *, double3 *);

template <int Bl>
__global__ void initialM_kernel(int2 *Encpairs_d, int2 *Encpairs2_d, double3 *acck_d, int NT){
	int idy = threadIdx.x;
	int idx = blockIdx.x;
	int id = idx * blockDim.x + idy;

	if(id < NT){
		for(int i = 0; i < Bl; ++i){
			Encpairs_d[id * Bl + i].x = -1;
			Encpairs_d[id * Bl + i].y = -1;

			Encpairs2_d[id * Bl + i].x = -1;
			Encpairs2_d[id * Bl + i].y = -1;
		}
		acck_d[id].x = 0.0;
		acck_d[id].y = 0.0;
		acck_d[id].z = 0.0;
	}
}
extern template __global__ void initialM_kernel < 16 > (int2 *, int2 *, double3 *, int);

/**************************************
* This kernel calculates the critical radius rcrit = max(n1 * Rh, n2 * dt * v), with the 
* Hill radius Rh = r * (m/(3Msun))^1/3, the velocity v and two constants n1 and  n2.
* rcritv is used for the the prechecker.
* In Rh we use the radius instead of the semi major axis.
* It searches also for ejections.
* This Kernel is launched wich NB/Bl blocks with Bl threads. NB is the next bigger number of N
* which is a power of two.
****************************************/
template <int Bl>
__global__ void Rcrit_kernel(double4 *x4_d, double4 *v4_d, double Msun, double *rcrit_d, double *rcritv_d, double dt, double *test_d, double n1, double n2, int *EjectionFlag_d, int N){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	__shared__ double4 x4_s[Bl];
	__shared__ double4 v4_s[Bl];

	double rcrit, rcritv ;
	double rsq, vsq, r, v;
	if(id < N){

       		x4_s[idy] = x4_d[id];
	        v4_s[idy] = v4_d[id];

		rsq = x4_s[idy].x*x4_s[idy].x + x4_s[idy].y*x4_s[idy].y + x4_s[idy].z*x4_s[idy].z + 1.0e-30;
		vsq = v4_s[idy].x*v4_s[idy].x + v4_s[idy].y*v4_s[idy].y + v4_s[idy].z*v4_s[idy].z + 1.0e-30;

		r = sqrt(rsq);
		v = sqrt(vsq);

		rcrit = n1 * r * cbrt(x4_s[idy].w  / ( Msun * 3.0));
		rcritv = fmax(rcrit, n2 * dt * v);

		rcrit_d[id] = fmax(rcrit, rcrit_d[id]);
		rcritv_d[id] = fmax(rcritv, rcritv_d[id]);
		

		//Check for Ejections or to small distances to the Sun
		if((rsq > Rcut * Rcut || rsq < RcutSun * RcutSun) && x4_d[id].w >= 0){
			 EjectionFlag_d[0] = 1;
		}
	}
}
extern template __global__ void Rcrit_kernel < 16 > (double4 *, double4 *, double, double *, double *, double, double *, double, double, int *, int);
extern template __global__ void Rcrit_kernel < 32 > (double4 *, double4 *, double, double *, double *, double, double *, double, double, int *, int);


template <int Bl>
__global__ void Rcritsmall_kernel(double4 *x4_d, double4 *v4_d, double Msun, double *rcrit_d, double *rcritv_d, double dt, double *test_d, double n1, double n2, int *EjectionFlag_d, double4 *x4small_d, double4 *v4small_d, int Nsmall, int N){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	__shared__ double4 x4_s[Bl];
	__shared__ double4 v4_s[Bl];

	double rcrit, rcritv ;
	double rsq, vsq, r, v;

	if(id < N){
		x4_s[idy] = x4_d[id];
		v4_s[idy] = v4_d[id];
	}
	else if (id < N + Nsmall){
                x4_s[idy] = x4small_d[id - N];
                v4_s[idy] = v4small_d[id - N];
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
	}
	__syncthreads();

	rsq = x4_s[idy].x*x4_s[idy].x + x4_s[idy].y*x4_s[idy].y + x4_s[idy].z*x4_s[idy].z + 1.0e-30;

	if(id < N){
		vsq = v4_s[idy].x*v4_s[idy].x + v4_s[idy].y*v4_s[idy].y + v4_s[idy].z*v4_s[idy].z + 1.0e-30;
		r = sqrt(rsq);
		v = sqrt(vsq);

		rcrit = n1 * r * cbrt(x4_s[idy].w  / ( Msun * 3.0));
		rcritv = fmax(rcrit, n2 * dt * v);
		rcrit_d[id] = rcrit;
		rcritv_d[id] = rcritv;
	}
	if(id < N + Nsmall){
        	//Check for Ejections or to small distances to the Sun
        	if((rsq > Rcut * Rcut || rsq < RcutSun * RcutSun) && x4_s[idy].w >= 0.0){
		 	EjectionFlag_d[0] = 1;
		}
	}

}
extern template __global__ void Rcritsmall_kernel < 128 > (double4 *, double4 *, double, double *, double *, double, double *, double, double, int *, double4 *, double4 *, int, int);

// **************************************
//This kernel calculates the critical radius rcrit = max(n1 * Rh, n2 * dt * v), with the 
//Hill radius Rh = r * (m/(3Msun))^1/3, the velocity v and two constants n1 and  n2.
//critv is used for the the prechecker.
//In Rh we use the radius instead of the semi major axis.
//It searches also for ejections.
//This Kernel is launched wich NB/Bl blocks with Bl threads. NB is the next bigger number of N
//which is a power of two.
// ****************************************
template <int Bl>
__global__ void RcritM_kernel(double4 *x4_d, double4 *v4_d, double *Msun_d, double *rcrit_d, double *rcritv_d, double dt, double *test_d, double *n1_d, double *n2_d, int *EjectionFlag_d, int *index_d, int Nst, int NT){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;
	int st = 0;

	if(id < NT) st = index_d[id] / 100;

	__shared__ double4 x4_s[Bl];
	__shared__ double4 v4_s[Bl];

	double rcrit, rcritv ;
	double rsq, vsq, r, v;

	if(id < NT){
		double Msun = Msun_d[st];
		double n1 = n1_d[st];
		double n2 = n2_d[st];
		x4_s[idy] = x4_d[id];
		v4_s[idy] = v4_d[id];

		__syncthreads();

		rsq = x4_s[idy].x*x4_s[idy].x + x4_s[idy].y*x4_s[idy].y + x4_s[idy].z*x4_s[idy].z + 1.0e-30;
		vsq = v4_s[idy].x*v4_s[idy].x + v4_s[idy].y*v4_s[idy].y + v4_s[idy].z*v4_s[idy].z + 1.0e-30;
		r = sqrt(rsq);
		v = sqrt(vsq);

		rcrit = n1 * r * cbrt(x4_s[idy].w  / ( Msun * 3.0));
		rcritv = fmax(rcrit, n2 * dt * v);

		rcrit_d[id] = rcrit;
		rcritv_d[id] = rcritv;
		

		//Check for Ejections or to small distances to the Sun
		if((rsq > Rcut * Rcut || rsq < RcutSun * RcutSun) && x4_d[id].w >= 0){
			EjectionFlag_d[st + 1] = 1;
			EjectionFlag_d[0] = 1;
		}
	}
}
extern template __global__ void RcritM_kernel < 32 > (double4 *, double4 *, double *, double *, double *, double, double *, double *, double *, int *, int *, int, int);
extern template __global__ void RcritM_kernel < 128 > (double4 *, double4 *, double *, double *, double *, double, double *, double *, double *, int *, int *, int, int);

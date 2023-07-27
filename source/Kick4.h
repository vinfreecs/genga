#ifndef KICK4_H
#define KICK4_H
#include "define.h"


//**************************************
//This function computes the terms a = mi/rij^3 * Kij.
//This function also finds the pairs of bodies which are separated less than pc * rcritv^2. 
//The function writes the encounter pairs into a list.

//Authors: Simon Grimm
//March 2019
//****************************************
__device__ void acc_e(volatile double3 &ac, double4 &x4i, double4 &x4j, volatile double rcritvi, volatile double rcritvj, int2 *Encpairs_d, int2 *Encpairs2_d, int *Nencpairs_d, int *EncFlag_d, const int j, const int i, const int NencMax, const int EE){

	if(i != j && x4i.w >= 0.0 && x4j.w >= 0.0){
		double3 r3ij;

		r3ij.x = x4j.x - x4i.x;
		r3ij.y = x4j.y - x4i.y;
		r3ij.z = x4j.z - x4i.z;

		double rsq = r3ij.x*r3ij.x + r3ij.y*r3ij.y + r3ij.z*r3ij.z;
		double rcritv = fmax(rcritvi, rcritvj);

		double ir = 1.0/sqrt(rsq);
		double ir3 = ir*ir*ir;

		double s = x4j.w * ir3;

		if(rsq < def_pc * rcritv * rcritv && (x4i.w > 0.0 || x4j.w > 0.0)){

			int Ni = atomicAdd(&Encpairs2_d[i * NencMax].x, 1);
//printf("enc1 %d %d %d %d\n", i, j, Ni, EE);
			if(Ni >= NencMax){
				atomicMax(&EncFlag_d[0], Ni);
			}
			else{
				Encpairs2_d[i * NencMax + Ni].y = j;
			}

			if(EE == 0){
				if(i < j && Ni < NencMax){
					int Ne = atomicAdd(Nencpairs_d, 1);
					Encpairs_d[Ne].x = i;
					Encpairs_d[Ne].y = j;
//printf("Precheck %d %d %d %d\n", i, j, Ne, EE);
				}
			}
			else{
				if(i > j && Ni < NencMax){
					int Ne = atomicAdd(Nencpairs_d, 1);
					Encpairs_d[Ne].x = i;
					Encpairs_d[Ne].y = j;
//printf("Precheck %d %d %d %d\n", i, j, Ne, EE);
				}
			}

			s = 0.0;
		}
		else if(rsq == 0.0) s = 0.0;

		ac.x += __dmul_rn(r3ij.x, s);
		ac.y += __dmul_rn(r3ij.y, s);
		ac.z += __dmul_rn(r3ij.z, s);
//if(i == 50) printf("acce %d %d %.20g %.20g %.20g\n", i, j, r3ij.x, s, rsq);
	}
}

//float version
__device__ void acc_ef(volatile float3 &ac, float4 &x4i, float4 &x4j, volatile float rcritvi, volatile float rcritvj, int2 *Encpairs_d, int2 *Encpairs2_d, int *Nencpairs_d, int *EncFlag_d, const int j, const int i, const int NencMax, const int EE){

	if(i != j && x4i.w >= 0.0f && x4j.w >= 0.0f){
		float3 r3ij;

		r3ij.x = x4j.x - x4i.x;
		r3ij.y = x4j.y - x4i.y;
		r3ij.z = x4j.z - x4i.z;

		float rsq = r3ij.x*r3ij.x + r3ij.y*r3ij.y + r3ij.z*r3ij.z;
		float rcritv = fmaxf(rcritvi, rcritvj);

		float ir = 1.0f/sqrtf(rsq);
		float ir3 = ir*ir*ir;

		float s = x4j.w * ir3;

		if(rsq < def_pcf * rcritv * rcritv && (x4i.w > 0.0f || x4j.w > 0.0f)){

			int Ni = atomicAdd(&Encpairs2_d[i * NencMax].x, 1);
//printf("enc1 %d %d %d\n", i, j, Ni);
			if(Ni >= NencMax){
				atomicMax(&EncFlag_d[0], Ni);
			}
			else{
				Encpairs2_d[i * NencMax + Ni].y = j;
			}

			if(EE == 0){
				if(i < j && Ni < NencMax){
					int Ne = atomicAdd(Nencpairs_d, 1);
					Encpairs_d[Ne].x = i;
					Encpairs_d[Ne].y = j;
				}
			}
			else{
				if(i > j && Ni < NencMax){
					int Ne = atomicAdd(Nencpairs_d, 1);
					Encpairs_d[Ne].x = i;
					Encpairs_d[Ne].y = j;
				}
			}

			s = 0.0f;
		}
		else if(rsq == 0.0f) s = 0.0f;

		ac.x += __fmul_rn(r3ij.x, s);
		ac.y += __fmul_rn(r3ij.y, s);
		ac.z += __fmul_rn(r3ij.z, s);
	}
}

// ********************************************************************************************
// This kernel sets all close Encounter lists to zero. It also sets the acceleration to zero.
// It is needed in the tunig loop for the kick kernel parameters
//
//Date: March 2019
//Author: Simon Grimm
// *******************************************************************************************
__global__ void EncpairsZeroC_kernel(int2 *Encpairs2_d, double3 *a_d, int *Nencpairs_d, int *Nencpairs2_d, const int NencMax, const int N){

	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if(id == 0){
		Nencpairs_d[0] = 0;
		Nencpairs2_d[0] = 0;
	}

	if(id < N){
		Encpairs2_d[NencMax * id].x = 0;

		a_d[id].x = 0.0;
		a_d[id].y = 0.0;
		a_d[id].z = 0.0;
	}
}

__global__ void compare_a_kernel(double3 *a_d, double3 *ab_d, const int KickFloat, const int N, const int f){

	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if(id < N){
		if(f == 1){
			double dx = fabs(a_d[id].x - ab_d[id].x);
			double dy = fabs(a_d[id].y - ab_d[id].y);
			double dz = fabs(a_d[id].z - ab_d[id].z);
//printf("compare a %d %.20g %.20g %.20g | %.20g %.20g %.20g\n", id, a_d[id].x, a_d[id].y, a_d[id].z, ab_d[id].x, ab_d[id].y, ab_d[id].z);
			if(KickFloat == 0){
				if(dx + dy + dz > 1.0e-8){
					printf("Comparison of acc from different kick kernel tuning parameters failed %d %.20g %.20g %.20g %.20g %.20g %.20g\n", id, a_d[id].x, ab_d[id].x, a_d[id].y, ab_d[id].y, a_d[id].z, ab_d[id].z);
				}
			}
			else{
				if(dx + dy + dz > 1.0e-6){
					printf("Comparison of acc from different kick kernel tuning parameters failed %d %.20g %.20g %.20g %.20g %.20g %.20g\n", id, a_d[id].x, ab_d[id].x, a_d[id].y, ab_d[id].y, a_d[id].z, ab_d[id].z);
				}
			}
		}
		ab_d[id] = a_d[id];
	}
}

// **********************************************************
// This kernel calculates the acceleration between all particles 
// the parallelization can be different according to the GPU one uses
// kernel parameters can be determinded beforehand in a tuning step

// The kernel writes a list of all close encounter candidates

// EE = 0: used for normal particles
// EE = 1: used for normal test particles
// EE = 2: used for semi test particles
// Date: April 2019
// Author: Simon Grimm
__global__ void acc4C_kernel(double4 *x4_d, double3 *acck_d, double *rcritv_d, int2 *Encpairs_d, int2 *Encpairs2_d, int *Nencpairs_d, int *EncFlag_d, const int Nstart, const int N, const int N0, const int N1, const int NencMax, const int p, const int EE){

	int idy = threadIdx.y;
	int ix = threadIdx.x;
	int idx = (blockIdx.x * blockDim.x + ix) * p + Nstart;
	int Bl = blockDim.y;
	int Bll = Bl * blockDim.x;
//if(idy == 0) printf("idx %d %d %d %d\n", idx, blockIdx.x, blockDim.x, threadIdx.x);

	extern volatile __shared__ double3 a_s[];

	double4 x4i1, x4i2, x4i3, x4i4;
	double rcritvi1, rcritvi2, rcritvi3, rcritvi4;

	if(idx + 0 < N){
		x4i1 = x4_d[idx + 0];
		rcritvi1 = rcritv_d[idx + 0];
		if(idy == 0 && EE < 2){
			Encpairs2_d[(idx + 0) * NencMax].x = 0;
		}
	}
	if(idx + 1 < N && p > 1){
		x4i2 = x4_d[idx + 1];
		rcritvi2 = rcritv_d[idx + 1];
		if(idy == 0 && EE < 2){
			Encpairs2_d[(idx + 1) * NencMax].x = 0;
		}
	}
	if(idx + 2 < N && p > 2){
		x4i3 = x4_d[idx + 2];
		rcritvi3 = rcritv_d[idx + 2];
		if(idy == 0 && EE < 2){
			Encpairs2_d[(idx + 2) * NencMax].x = 0;
		}
	}
	if(idx + 3 < N && p > 3){
		x4i4 = x4_d[idx + 3];
		rcritvi4 = rcritv_d[idx + 3];
		if(idy == 0 && EE < 2){
			Encpairs2_d[(idx + 3) * NencMax].x = 0;
		}
	}

	for(int j = 0; j < p; ++j){
		a_s[idy + ix * Bl + j * Bll].x = 0.0;
		a_s[idy + ix * Bl + j * Bll].y = 0.0;
		a_s[idy + ix * Bl + j * Bll].z = 0.0;
	}

	__syncthreads();
	for(int i = N0; i < N1; i += Bl){
		if(idy + i < N1){
			double4 x4j = x4_d[idy + i];
			double rcritvj = rcritv_d[idy + i];

			if(idx + 0 < N)          acc_e(a_s[idy + ix * Bl + 0 * Bll], x4i1, x4j, rcritvi1, rcritvj, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, idy + i, idx + 0, NencMax, EE);
			if(idx + 1 < N && p > 1) acc_e(a_s[idy + ix * Bl + 1 * Bll], x4i2, x4j, rcritvi2, rcritvj, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, idy + i, idx + 1, NencMax, EE);
			if(idx + 2 < N && p > 2) acc_e(a_s[idy + ix * Bl + 2 * Bll], x4i3, x4j, rcritvi3, rcritvj, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, idy + i, idx + 2, NencMax, EE);
			if(idx + 3 < N && p > 3) acc_e(a_s[idy + ix * Bl + 3 * Bll], x4i4, x4j, rcritvi4, rcritvj, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, idy + i, idx + 3, NencMax, EE);

		}
	}
	__syncthreads();
//if(idx == 0)	printf("A %d %d %d %.20g %.20g %.20g\n", idx, ix, idy, a_s[ix * Bl + 0 * Bll].x, a_s[ix * Bl + 0 * Bll].y, a_s[ix * Bl + 0 * Bll].z); 

	int s = Bl/2;

	for(int i = 6; i < log2f(Bl); ++i){
		if( idy < s ) {
			for(int j = 0; j < p; ++j){
				a_s[idy + ix * Bl + j * Bll].x += a_s[idy + ix * Bl + j * Bll + s].x;
				a_s[idy + ix * Bl + j * Bll].y += a_s[idy + ix * Bl + j * Bll + s].y;
				a_s[idy + ix * Bl + j * Bll].z += a_s[idy + ix * Bl + j * Bll + s].z;
			}
		}
		__syncthreads();
		s /= 2;
	}


	for(int j = 0; j < p; ++j){

		if(Bl > 32 && idy < 32){
			a_s[idy + ix * Bl + j * Bll].x += a_s[idy + ix * Bl + j * Bll + 32].x;
			a_s[idy + ix * Bl + j * Bll].y += a_s[idy + ix * Bl + j * Bll + 32].y;
			a_s[idy + ix * Bl + j * Bll].z += a_s[idy + ix * Bl + j * Bll + 32].z;
		}
		__syncthreads();        //this is needed here because idy are not neccessary in the same warp
		if(Bl > 16 && idy < 16){
			a_s[idy + ix * Bl + j * Bll].x += a_s[idy + ix * Bl + j * Bll + 16].x;
			a_s[idy + ix * Bl + j * Bll].y += a_s[idy + ix * Bl + j * Bll + 16].y;
			a_s[idy + ix * Bl + j * Bll].z += a_s[idy + ix * Bl + j * Bll + 16].z;
		}
		__syncthreads();
		if(Bl >  8 && idy < 8){
			a_s[idy + ix * Bl + j * Bll].x += a_s[idy + ix * Bl + j * Bll + 8].x;
			a_s[idy + ix * Bl + j * Bll].y += a_s[idy + ix * Bl + j * Bll + 8].y;
			a_s[idy + ix * Bl + j * Bll].z += a_s[idy + ix * Bl + j * Bll + 8].z;
		}
		__syncthreads();
		if(Bl >  4 && idy < 4){
			a_s[idy + ix * Bl + j * Bll].x += a_s[idy + ix * Bl + j * Bll + 4].x;
			a_s[idy + ix * Bl + j * Bll].y += a_s[idy + ix * Bl + j * Bll + 4].y;
			a_s[idy + ix * Bl + j * Bll].z += a_s[idy + ix * Bl + j * Bll + 4].z;
		}
		__syncthreads();
		if(Bl >  2 && idy < 2){
			a_s[idy + ix * Bl + j * Bll].x += a_s[idy + ix * Bl + j * Bll + 2].x;
			a_s[idy + ix * Bl + j * Bll].y += a_s[idy + ix * Bl + j * Bll + 2].y;
			a_s[idy + ix * Bl + j * Bll].z += a_s[idy + ix * Bl + j * Bll + 2].z;
		}
		__syncthreads();
		if(Bl >  1 && idy < 1){
			a_s[idy + ix * Bl + j * Bll].x += a_s[idy + ix * Bl + j * Bll + 1].x;
			a_s[idy + ix * Bl + j * Bll].y += a_s[idy + ix * Bl + j * Bll + 1].y;
			a_s[idy + ix * Bl + j * Bll].z += a_s[idy + ix * Bl + j * Bll + 1].z;
		}
		__syncthreads();
	}

//if(idy == 0 && idx < 10)	printf("Kick %d %d %d %.20g %.20g %.20g\n", idx, ix, idy, a_s[ix * Bl + 0 * Bll].x, a_s[ix * Bl + 0 * Bll].y, a_s[ix * Bl + 0 * Bll].z); 


	if(EE < 2){
		if(idy == 0){
			if(idx + 0 < N){
				acck_d[idx + 0].x = a_s[ix * Bl + 0 * Bll].x;
				acck_d[idx + 0].y = a_s[ix * Bl + 0 * Bll].y;
				acck_d[idx + 0].z = a_s[ix * Bl + 0 * Bll].z;
			}
			if(idx + 1 < N && p > 1){
				acck_d[idx + 1].x = a_s[ix * Bl + 1 * Bll].x;
				acck_d[idx + 1].y = a_s[ix * Bl + 1 * Bll].y;
				acck_d[idx + 1].z = a_s[ix * Bl + 1 * Bll].z;
			}
			if(idx + 2 < N && p > 2){
				acck_d[idx + 2].x = a_s[ix * Bl + 2 * Bll].x;
				acck_d[idx + 2].y = a_s[ix * Bl + 2 * Bll].y;
				acck_d[idx + 2].z = a_s[ix * Bl + 2 * Bll].z;
			}
			if(idx + 3 < N && p > 3){
				acck_d[idx + 3].x = a_s[ix * Bl + 3 * Bll].x;
				acck_d[idx + 3].y = a_s[ix * Bl + 3 * Bll].y;
				acck_d[idx + 3].z = a_s[ix * Bl + 3 * Bll].z;
			}
		}
	}
	if(EE == 2){
		if(idy == 0){
			if(idx + 0 < N){
				acck_d[idx + 0].x += a_s[ix * Bl + 0 * Bll].x;
				acck_d[idx + 0].y += a_s[ix * Bl + 0 * Bll].y;
				acck_d[idx + 0].z += a_s[ix * Bl + 0 * Bll].z;
			}
			if(idx + 1 < N && p > 1){
				acck_d[idx + 1].x += a_s[ix * Bl + 1 * Bll].x;
				acck_d[idx + 1].y += a_s[ix * Bl + 1 * Bll].y;
				acck_d[idx + 1].z += a_s[ix * Bl + 1 * Bll].z;
			}
			if(idx + 2 < N && p > 2){
				acck_d[idx + 2].x += a_s[ix * Bl + 2 * Bll].x;
				acck_d[idx + 2].y += a_s[ix * Bl + 2 * Bll].y;
				acck_d[idx + 2].z += a_s[ix * Bl + 2 * Bll].z;
			}
			if(idx + 3 < N && p > 3){
				acck_d[idx + 3].x += a_s[ix * Bl + 3 * Bll].x;
				acck_d[idx + 3].y += a_s[ix * Bl + 3 * Bll].y;
				acck_d[idx + 3].z += a_s[ix * Bl + 3 * Bll].z;
			}
		}
	}
}

//float version
__global__ void acc4Cf_kernel(double4 *x4_d, double3 *acck_d, double *rcritv_d, int2 *Encpairs_d, int2 *Encpairs2_d, int *Nencpairs_d, int *EncFlag_d, const int Nstart, const int N, const int N0, const int N1, const int NencMax, const int p, const int EE){

	int idy = threadIdx.y;
	int ix = threadIdx.x;
	int idx = (blockIdx.x * blockDim.x + ix) * p + Nstart;
	int Bl = blockDim.y;
	int Bll = Bl * blockDim.x;
//if(idy == 0) printf("idx %d %d %d %d\n", idx, blockIdx.x, blockDim.x, threadIdx.x);

	extern volatile __shared__ float3 af_s[];

//        extern __shared__ double s_s[];
//        double *K_s = s_s;                              //size NBx * NBy
//        double *x_s = (double*)&K_s[NBx * NBy];         //size NBx


	float4 x4i1, x4i2, x4i3, x4i4;
	float rcritvi1, rcritvi2, rcritvi3, rcritvi4;

	if(idx + 0 < N){
		x4i1.x = float(x4_d[idx + 0].x);
		x4i1.y = float(x4_d[idx + 0].y);
		x4i1.z = float(x4_d[idx + 0].z);
		x4i1.w = float(x4_d[idx + 0].w);
		rcritvi1 = float(rcritv_d[idx + 0]);
		if(idy == 0 && EE < 2){
			Encpairs2_d[(idx + 0) * NencMax].x = 0;
		}
	}
	if(idx + 1 < N && p > 1){
		x4i2.x = float(x4_d[idx + 1].x);
		x4i2.y = float(x4_d[idx + 1].y);
		x4i2.z = float(x4_d[idx + 1].z);
		x4i2.w = float(x4_d[idx + 1].w);
		rcritvi2 = float(rcritv_d[idx + 1]);
		if(idy == 0 && EE < 2){
			Encpairs2_d[(idx + 1) * NencMax].x = 0;
		}
	}
	if(idx + 2 < N && p > 2){
		x4i3.x = float(x4_d[idx + 2].x);
		x4i3.y = float(x4_d[idx + 2].y);
		x4i3.z = float(x4_d[idx + 2].z);
		x4i3.w = float(x4_d[idx + 2].w);
		rcritvi3 =float( rcritv_d[idx + 2]);
		if(idy == 0 && EE < 2){
			Encpairs2_d[(idx + 2) * NencMax].x = 0;
		}
	}
	if(idx + 3 < N && p > 3){
		x4i4.x = float(x4_d[idx + 3].x);
		x4i4.y = float(x4_d[idx + 3].y);
		x4i4.z = float(x4_d[idx + 3].z);
		x4i4.w = float(x4_d[idx + 3].w);
		rcritvi4 = float(rcritv_d[idx + 3]);
		if(idy == 0 && EE < 2){
			Encpairs2_d[(idx + 3) * NencMax].x = 0;
		}
	}

	for(int j = 0; j < p; ++j){
		af_s[idy + ix * Bl + j * Bll].x = 0.0f;
		af_s[idy + ix * Bl + j * Bll].y = 0.0f;
		af_s[idy + ix * Bl + j * Bll].z = 0.0f;
	}

	__syncthreads();
	for(int i = N0; i < N1; i += Bl){
		if(idy + i < N1){
			float4 x4j;
			x4j.x = float(x4_d[idy + i].x);
			x4j.y = float(x4_d[idy + i].y);
			x4j.z = float(x4_d[idy + i].z);
			x4j.w = float(x4_d[idy + i].w);
			float rcritvj = float(rcritv_d[idy + i]);

			if(idx + 0 < N)          acc_ef(af_s[idy + ix * Bl + 0 * Bll], x4i1, x4j, rcritvi1, rcritvj, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, idy + i, idx + 0, NencMax, EE);
			if(idx + 1 < N && p > 1) acc_ef(af_s[idy + ix * Bl + 1 * Bll], x4i2, x4j, rcritvi2, rcritvj, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, idy + i, idx + 1, NencMax, EE);
			if(idx + 2 < N && p > 2) acc_ef(af_s[idy + ix * Bl + 2 * Bll], x4i3, x4j, rcritvi3, rcritvj, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, idy + i, idx + 2, NencMax, EE);
			if(idx + 3 < N && p > 3) acc_ef(af_s[idy + ix * Bl + 3 * Bll], x4i4, x4j, rcritvi4, rcritvj, Encpairs_d, Encpairs2_d, Nencpairs_d, EncFlag_d, idy + i, idx + 3, NencMax, EE);

		}
	}
	__syncthreads();
//	if(idx > 980)	printf("A %d %d %d %.20g %.20g %.20g\n", idx, ix, idy, af_s[ix * Bl + 0 * Bll].x, af_s[ix * Bl + 0 * Bll].y, af_s[ix * Bl + 0 * Bll].z); 

	int s = Bl/2;

	for(int i = 6; i < log2f(Bl); ++i){
		if( idy < s ) {
			for(int j = 0; j < p; ++j){
				af_s[idy + ix * Bl + j * Bll].x += af_s[idy + ix * Bl + j * Bll + s].x;
				af_s[idy + ix * Bl + j * Bll].y += af_s[idy + ix * Bl + j * Bll + s].y;
				af_s[idy + ix * Bl + j * Bll].z += af_s[idy + ix * Bl + j * Bll + s].z;
			}
		}
		__syncthreads();
		s /= 2;
	}


	for(int j = 0; j < p; ++j){

		if(Bl > 32 && idy < 32){
			af_s[idy + ix * Bl + j * Bll].x += af_s[idy + ix * Bl + j * Bll + 32].x;
			af_s[idy + ix * Bl + j * Bll].y += af_s[idy + ix * Bl + j * Bll + 32].y;
			af_s[idy + ix * Bl + j * Bll].z += af_s[idy + ix * Bl + j * Bll + 32].z;
		}
		__syncthreads();        //this is needed here because idy are not neccessary in the same warp
		if(Bl > 16 && idy < 16){
			af_s[idy + ix * Bl + j * Bll].x += af_s[idy + ix * Bl + j * Bll + 16].x;
			af_s[idy + ix * Bl + j * Bll].y += af_s[idy + ix * Bl + j * Bll + 16].y;
			af_s[idy + ix * Bl + j * Bll].z += af_s[idy + ix * Bl + j * Bll + 16].z;
		}
		__syncthreads();
		if(Bl >  8 && idy < 8){
			af_s[idy + ix * Bl + j * Bll].x += af_s[idy + ix * Bl + j * Bll + 8].x;
			af_s[idy + ix * Bl + j * Bll].y += af_s[idy + ix * Bl + j * Bll + 8].y;
			af_s[idy + ix * Bl + j * Bll].z += af_s[idy + ix * Bl + j * Bll + 8].z;
		}
		__syncthreads();
		if(Bl >  4 && idy < 4){
			af_s[idy + ix * Bl + j * Bll].x += af_s[idy + ix * Bl + j * Bll + 4].x;
			af_s[idy + ix * Bl + j * Bll].y += af_s[idy + ix * Bl + j * Bll + 4].y;
			af_s[idy + ix * Bl + j * Bll].z += af_s[idy + ix * Bl + j * Bll + 4].z;
		}
		__syncthreads();
		if(Bl >  2 && idy < 2){
			af_s[idy + ix * Bl + j * Bll].x += af_s[idy + ix * Bl + j * Bll + 2].x;
			af_s[idy + ix * Bl + j * Bll].y += af_s[idy + ix * Bl + j * Bll + 2].y;
			af_s[idy + ix * Bl + j * Bll].z += af_s[idy + ix * Bl + j * Bll + 2].z;
		}
		__syncthreads();
		if(Bl >  1 && idy < 1){
			af_s[idy + ix * Bl + j * Bll].x += af_s[idy + ix * Bl + j * Bll + 1].x;
			af_s[idy + ix * Bl + j * Bll].y += af_s[idy + ix * Bl + j * Bll + 1].y;
			af_s[idy + ix * Bl + j * Bll].z += af_s[idy + ix * Bl + j * Bll + 1].z;
		}
		__syncthreads();
	}

//	if(idy == 0 && idx > 980)	printf("%d %d %d %.20g %.20g %.20g\n", idx, ix, idy, af_s[ix * Bl + 0 * Bll].x, af_s[ix * Bl + 0 * Bll].y, af_s[ix * Bl + 0 * Bll].z); 


	if(EE < 2){
		if(idy == 0){
			if(idx + 0 < N){
				acck_d[idx + 0].x = af_s[ix * Bl + 0 * Bll].x;
				acck_d[idx + 0].y = af_s[ix * Bl + 0 * Bll].y;
				acck_d[idx + 0].z = af_s[ix * Bl + 0 * Bll].z;
			}
			if(idx + 1 < N && p > 1){
				acck_d[idx + 1].x = af_s[ix * Bl + 1 * Bll].x;
				acck_d[idx + 1].y = af_s[ix * Bl + 1 * Bll].y;
				acck_d[idx + 1].z = af_s[ix * Bl + 1 * Bll].z;
			}
			if(idx + 2 < N && p > 2){
				acck_d[idx + 2].x = af_s[ix * Bl + 2 * Bll].x;
				acck_d[idx + 2].y = af_s[ix * Bl + 2 * Bll].y;
				acck_d[idx + 2].z = af_s[ix * Bl + 2 * Bll].z;
			}
			if(idx + 3 < N && p > 3){
				acck_d[idx + 3].x = af_s[ix * Bl + 3 * Bll].x;
				acck_d[idx + 3].y = af_s[ix * Bl + 3 * Bll].y;
				acck_d[idx + 3].z = af_s[ix * Bl + 3 * Bll].z;
			}
		}
	}
	if(EE == 2){
		if(idy == 0){
			if(idx + 0 < N){
				acck_d[idx + 0].x += af_s[ix * Bl + 0 * Bll].x;
				acck_d[idx + 0].y += af_s[ix * Bl + 0 * Bll].y;
				acck_d[idx + 0].z += af_s[ix * Bl + 0 * Bll].z;
			}
			if(idx + 1 < N && p > 1){
				acck_d[idx + 1].x += af_s[ix * Bl + 1 * Bll].x;
				acck_d[idx + 1].y += af_s[ix * Bl + 1 * Bll].y;
				acck_d[idx + 1].z += af_s[ix * Bl + 1 * Bll].z;
			}
			if(idx + 2 < N && p > 2){
				acck_d[idx + 2].x += af_s[ix * Bl + 2 * Bll].x;
				acck_d[idx + 2].y += af_s[ix * Bl + 2 * Bll].y;
				acck_d[idx + 2].z += af_s[ix * Bl + 2 * Bll].z;
			}
			if(idx + 3 < N && p > 3){
				acck_d[idx + 3].x += af_s[ix * Bl + 3 * Bll].x;
				acck_d[idx + 3].y += af_s[ix * Bl + 3 * Bll].y;
				acck_d[idx + 3].z += af_s[ix * Bl + 3 * Bll].z;
			}
		}
	}
}
#if def_CPU == 1

// Can be applied only to massive particles
// Here EE is always 0
// Serial version
void Data::acc4E_cpu(){
	
	//if E == 0
	for(int i = 0; i < N_h[0]; ++i){
		a_h[i] = {0.0, 0.0, 0.0};
		Encpairs2_h[i * P.NencMax].x = 0;
	}
	
	for(int i = 0; i < N_h[0]; ++i){
		double ir, ir3;
		double3 r3ij;

		for(int j = i + 1; j < N_h[0]; ++j){
			
			if(x4_h[i].w >= 0.0 && x4_h[j].w >= 0.0){
				
//printf("%d %d %d\n", i, j);
				
				r3ij.x = x4_h[j].x - x4_h[i].x;
				r3ij.y = x4_h[j].y - x4_h[i].y;
				r3ij.z = x4_h[j].z - x4_h[i].z;
				
				double rsq = (r3ij.x*r3ij.x) + (r3ij.y*r3ij.y) + (r3ij.z*r3ij.z);
				
				double rcritv = fmax(rcritv_h[i], rcritv_h[j]);
				
				ir = 1.0/sqrt(rsq);
				ir3 = ir*ir*ir;
				
				
//				if(rsq < 3.0 * rcritv * rcritv){
				if(rsq < def_pc * rcritv * rcritv && (x4_h[i].w > 0.0 || x4_h[j].w > 0.0)){
					
					int Ni = Encpairs2_h[i * P.NencMax].x++;
					int Nj = Encpairs2_h[j * P.NencMax].x++;
//printf("enc1 %d %d %d %d\n", i, j, Ni, Nj);
					if(Ni >= P.NencMax){
						EncFlag_m[0] = max(EncFlag_m[0], Ni);
					}
					else{
						Encpairs2_h[i * P.NencMax + Ni].y = j;
					}

					if(Nj >= P.NencMax){
						EncFlag_m[0] = max(EncFlag_m[0], Nj);
					}
					else{
						Encpairs2_h[j * P.NencMax + Nj].y = i;
					}
					
					if(Ni < P.NencMax && Nj < P.NencMax){
						// i < j is always true
						int Ne = Nencpairs_h[0]++;
						Encpairs_h[Ne].x = i;
						Encpairs_h[Ne].y = j;
//printf("Precheck %d %d %d\n", i, j, Ne);
					}

					ir3 = 0.0;
				}
				else if(rsq == 0.0){
					ir3 = 0.0;
				}
				
				double si = (x4_h[i].w * ir3);
				double sj = (x4_h[j].w * ir3);
				
				a_h[i].x += r3ij.x * sj;
				a_h[i].y += r3ij.y * sj;
				a_h[i].z += r3ij.z * sj;
				
				a_h[j].x -= r3ij.x * si;
				a_h[j].y -= r3ij.y * si;
				a_h[j].z -= r3ij.z * si;
				
//if(i == 10) printf("acci %d %d %g %g\n", i, j, sj * r3ij.x, b_h[k * NconstT + i].x);
//if(j == 10) printf("accj %d %d %g %g\n", j, i, -si * r3ij.x, b_h[k * NconstT + j].x);
				
			}
		}
	}
}

// Is applied only to test particles particles
// Serial Version
void Data::acc4Esmall_cpu(){
	
	for(int i = N_h[0]; i < N_h[0] + Nsmall_h[0]; ++i){
		a_h[i] = {0.0, 0.0, 0.0};
		Encpairs2_h[i * P.NencMax].x = 0;
	}
	
	for(int i = N_h[0]; i < N_h[0] + Nsmall_h[0]; ++i){
		double ir, ir3;
		double3 r3ij;

		for(int j = 0; j < N_h[0]; ++j){
			
			if(x4_h[i].w >= 0.0 && x4_h[j].w >= 0.0){
				
				
				r3ij.x = x4_h[j].x - x4_h[i].x;
				r3ij.y = x4_h[j].y - x4_h[i].y;
				r3ij.z = x4_h[j].z - x4_h[i].z;
				
				double rsq = (r3ij.x*r3ij.x) + (r3ij.y*r3ij.y) + (r3ij.z*r3ij.z);
				
				double rcritv = fmax(rcritv_h[i], rcritv_h[j]);
				
				ir = 1.0/sqrt(rsq);
				ir3 = ir*ir*ir;
				
				
//				if(rsq < 3.0 * rcritv * rcritv){
				if(rsq < def_pc * rcritv * rcritv && (x4_h[i].w > 0.0 || x4_h[j].w > 0.0)){
					
					int Ni = Encpairs2_h[i * P.NencMax].x++;
					int Nj = 0;
					if(Ni >= P.NencMax){
						EncFlag_m[0] = max(EncFlag_m[0], Ni);
					}
					else{
						Encpairs2_h[i * P.NencMax + Ni].y = j;
					}

					if(P.UseTestParticles == 2){
						Nj = Encpairs2_h[j * P.NencMax].x++;

						if(Nj >= P.NencMax){
							EncFlag_m[0] = max(EncFlag_m[0], Nj);
						}
						else{
							Encpairs2_h[j * P.NencMax + Nj].y = i;
						}
					}					
//printf("enc1 small %d %d %d\n", i, j, Ni);
//printf("enc1 %d %d %d %d\n", i, j, Ni, Nj);

					if(Ni < P.NencMax && Nj < P.NencMax){
						// j < i is always true
						int Ne = Nencpairs_h[0]++;
						Encpairs_h[Ne].x = i;
						Encpairs_h[Ne].y = j;
//printf("Precheck small %d %d %d\n", i, j, Ne);
					}

					ir3 = 0.0;
				}
				else if(rsq == 0.0){
					ir3 = 0.0;
				}
				
				double sj = (x4_h[j].w * ir3);
				
				a_h[i].x += r3ij.x * sj;
				a_h[i].y += r3ij.y * sj;
				a_h[i].z += r3ij.z * sj;
	
				if(P.UseTestParticles == 2){
					double si = (x4_h[i].w * ir3);
				
					a_h[j].x -= r3ij.x * si;
					a_h[j].y -= r3ij.y * si;
					a_h[j].z -= r3ij.z * si;
				
				}
//if(i == 10) printf("acci %d %d %g %g\n", i, j, sj * r3ij.x, b_h[k * NconstT + i].x);
//if(j == 10) printf("accj %d %d %g %g\n", j, i, -si * r3ij.x, b_h[k * NconstT + j].x);
				
			}
		}
	}
}

//float version
void Data::acc4Ef_cpu(){
	
	//if E == 0
	for(int i = 0; i < N_h[0]; ++i){
		a_h[i] = {0.0, 0.0, 0.0};
		Encpairs2_h[i * P.NencMax].x = 0;
	}
	
	for(int i = 0; i < N_h[0]; ++i){
		float ir, ir3;
		float3 r3ij;

		for(int j = i + 1; j < N_h[0]; ++j){
			
			if(x4_h[i].w >= 0.0 && x4_h[j].w >= 0.0){
				
//printf("%d %d %d %d %d\n", 0, ii, jj, i, j);
				
				r3ij.x = x4_h[j].x - x4_h[i].x;
				r3ij.y = x4_h[j].y - x4_h[i].y;
				r3ij.z = x4_h[j].z - x4_h[i].z;
				
				float rsq = (r3ij.x*r3ij.x) + (r3ij.y*r3ij.y) + (r3ij.z*r3ij.z);
				
				float rcritv = fmax(rcritv_h[i], rcritv_h[j]);
				
				ir = 1.0f/sqrt(rsq);
				ir3 = ir*ir*ir;
				
				
//				if(rsq < 3.0 * rcritv * rcritv){
				if(rsq < def_pc * rcritv * rcritv && (x4_h[i].w > 0.0 || x4_h[j].w > 0.0)){
					
					int Ni = Encpairs2_h[i * P.NencMax].x++;
					int Nj = Encpairs2_h[j * P.NencMax].x++;
//printf("enc1 %d %d %d %d\n", i, j, Ni, Nj);
					if(Ni >= P.NencMax){
						EncFlag_m[0] = max(EncFlag_m[0], Ni);
					}
					else{
						Encpairs2_h[i * P.NencMax + Ni].y = j;
					}

					if(Nj >= P.NencMax){
						EncFlag_m[0] = max(EncFlag_m[0], Nj);
					}
					else{
						Encpairs2_h[j * P.NencMax + Nj].y = i;
					}
					
					if(Ni < P.NencMax && Nj < P.NencMax){
						// i < j is always true
						int Ne = Nencpairs_h[0]++;
						Encpairs_h[Ne].x = i;
						Encpairs_h[Ne].y = j;
//printf("Precheck %d %d %d\n", i, j, Ne);
					}

					ir3 = 0.0f;
				}
				else if(rsq == 0.0f){
					ir3 = 0.0f;
				}
				
				double si = (x4_h[i].w * ir3);
				double sj = (x4_h[j].w * ir3);
				
				a_h[i].x += r3ij.x * sj;
				a_h[i].y += r3ij.y * sj;
				a_h[i].z += r3ij.z * sj;
				
				a_h[j].x -= r3ij.x * si;
				a_h[j].y -= r3ij.y * si;
				a_h[j].z -= r3ij.z * si;
				
//if(i == 10) printf("acci %d %d %g %g\n", i, j, sj * r3ij.x, b_h[k * NconstT + i].x);
//if(j == 10) printf("accj %d %d %g %g\n", j, i, -si * r3ij.x, b_h[k * NconstT + j].x);
				
			}
		}
	}
}
// Is applied only to test particles particles
// Serial Version
void Data::acc4Efsmall_cpu(){
	
	for(int i = N_h[0]; i < N_h[0] + Nsmall_h[0]; ++i){
		a_h[i] = {0.0, 0.0, 0.0};
		Encpairs2_h[i * P.NencMax].x = 0;
	}
	
	for(int i = N_h[0]; i < N_h[0] + Nsmall_h[0]; ++i){
		float ir, ir3;
		float3 r3ij;

		for(int j = 0; j < N_h[0]; ++j){
			
			if(x4_h[i].w >= 0.0 && x4_h[j].w >= 0.0){
				
				
				r3ij.x = x4_h[j].x - x4_h[i].x;
				r3ij.y = x4_h[j].y - x4_h[i].y;
				r3ij.z = x4_h[j].z - x4_h[i].z;
				
				float rsq = (r3ij.x*r3ij.x) + (r3ij.y*r3ij.y) + (r3ij.z*r3ij.z);
				
				float rcritv = fmax(rcritv_h[i], rcritv_h[j]);
				
				ir = 1.0f/sqrt(rsq);
				ir3 = ir*ir*ir;
				
				
//				if(rsq < 3.0 * rcritv * rcritv){
				if(rsq < def_pc * rcritv * rcritv && (x4_h[i].w > 0.0 || x4_h[j].w > 0.0)){
					
					int Ni = Encpairs2_h[i * P.NencMax].x++;
					int Nj = 0;
					if(Ni >= P.NencMax){
						EncFlag_m[0] = max(EncFlag_m[0], Ni);
					}
					else{
						Encpairs2_h[i * P.NencMax + Ni].y = j;
					}

					if(P.UseTestParticles == 2){
						Nj = Encpairs2_h[j * P.NencMax].x++;

						if(Nj >= P.NencMax){
							EncFlag_m[0] = max(EncFlag_m[0], Nj);
						}
						else{
							Encpairs2_h[j * P.NencMax + Nj].y = i;
						}
					}					
//printf("enc1 small %d %d %d\n", i, j, Ni);
//printf("enc1 %d %d %d %d\n", i, j, Ni, Nj);

					if(Ni < P.NencMax && Nj < P.NencMax){
						// j < i is always true
						int Ne = Nencpairs_h[0]++;
						Encpairs_h[Ne].x = i;
						Encpairs_h[Ne].y = j;
//printf("Precheck small %d %d %d\n", i, j, Ne);
					}

					ir3 = 0.0f;
				}
				else if(rsq == 0.0){
					ir3 = 0.0f;
				}
				
				double sj = (x4_h[j].w * ir3);
				
				a_h[i].x += r3ij.x * sj;
				a_h[i].y += r3ij.y * sj;
				a_h[i].z += r3ij.z * sj;

				if(P.UseTestParticles == 2){
					double si = (x4_h[i].w * ir3);
				
					a_h[j].x -= r3ij.x * si;
					a_h[j].y -= r3ij.y * si;
					a_h[j].z -= r3ij.z * si;
				
				}
//if(i == 10) printf("acci %d %d %g %g\n", i, j, sj * r3ij.x, b_h[k * NconstT + i].x);
//if(j == 10) printf("accj %d %d %g %g\n", j, i, -si * r3ij.x, b_h[k * NconstT + j].x);
				
			}
		}
	}
}

// Can be applied only to massive particles
// Here EE is always 0
// paralle version with OpenMP
void Data::acc4D_cpu(){
	
	for(int i = 0; i < N_h[0]; ++i){
		for(int k = 0; k < Nomp; ++k){
			b_h[k * NconstT + i] = {0.0, 0.0, 0.0};
		}
		Encpairs2_h[i * P.NencMax].x = 0;
	}
	
	#pragma omp parallel for // proc_bind(close)
	for(int ii = 0; ii < N_h[0] / 2; ++ii){
		double ir, ir3;
		double3 r3ij;

		int k = omp_get_thread_num();
		//int k = 0;
//int cpuid = sched_getcpu();
//if(ii == 0 || ii == N_h[0] / 2 - 1) printf("acc4D %d %d %d %d\n", ii, k, cpuid, 0);

		for(int jj = 0; jj < N_h[0]; ++jj){
	
			if(ii == (N_h[0] + 1) / 2 - 1 && jj >= N_h[0] / 2) break;
			
			int j = jj;
			int i = ii;

			if(jj <= ii){
				i = N_h[0] - 2 - ii;
				j = N_h[0] - 1 - jj;
			}
			
			if(x4_h[i].w >= 0.0 && x4_h[j].w >= 0.0){
				
				//printf("%d %d %d %d %d\n", k, ii, jj, i, j);
				
				r3ij.x = x4_h[j].x - x4_h[i].x;
				r3ij.y = x4_h[j].y - x4_h[i].y;
				r3ij.z = x4_h[j].z - x4_h[i].z;
				
				double rsq = (r3ij.x*r3ij.x) + (r3ij.y*r3ij.y) + (r3ij.z*r3ij.z);
				
				double rcritv = fmax(rcritv_h[i], rcritv_h[j]);
				
				ir = 1.0/sqrt(rsq);
				ir3 = ir*ir*ir;
				
				
//				if(rsq < 3.0 * rcritv * rcritv){
				if(rsq < def_pc * rcritv * rcritv && (x4_h[i].w > 0.0 || x4_h[j].w > 0.0)){
					
					int Ni, Nj;
					#pragma omp atomic capture
					Ni = Encpairs2_h[i * P.NencMax].x++;
					#pragma omp atomic capture
					Nj = Encpairs2_h[j * P.NencMax].x++;

//printf("enc1 %d | %d %d %d %d\n", k, i, j, Ni, Nj);
					if(Ni >= P.NencMax){
						EncFlag_m[0] = max(EncFlag_m[0], Ni);
					}
					else{
						Encpairs2_h[i * P.NencMax + Ni].y = j;
					}
	
					if(Nj >= P.NencMax){
						EncFlag_m[0] = max(EncFlag_m[0], Nj);
					}
					else{
						Encpairs2_h[j * P.NencMax + Nj].y = i;
					}
					
					if(Ni < P.NencMax && Nj < P.NencMax){
						// i < j is always true
						int Ne;
						#pragma omp atomic capture
						Ne = Nencpairs_h[0]++;

						Encpairs_h[Ne].x = i;
						Encpairs_h[Ne].y = j;
//printf("Precheck %d | %d %d %d\n", k, i, j, Ne);
					}
					ir3 = 0.0;
				}
				else if(rsq == 0.0){
					ir3 = 0.0;
				}
				
				double si = (x4_h[i].w * ir3);
				double sj = (x4_h[j].w * ir3);
				
				int ki = k * NconstT + i; 
				int kj = k * NconstT + j; 
				
				b_h[ki].x += r3ij.x * sj;
				b_h[ki].y += r3ij.y * sj;
				b_h[ki].z += r3ij.z * sj;
				
				b_h[kj].x -= r3ij.x * si;
				b_h[kj].y -= r3ij.y * si;
				b_h[kj].z -= r3ij.z * si;
				
//if(i == 10) printf("acci %d %d %g %g\n", i, j, sj * r3ij.x, b_h[k * NconstT + i].x);
//if(j == 10) printf("accj %d %d %g %g\n", j, i, -si * r3ij.x, b_h[k * NconstT + j].x);
				
			}
		}
	}
	
	for(int i = 0; i < N_h[0]; ++i){
		double3 a = {0.0, 0.0, 0.0};
		for(int k = 0; k < Nomp; ++k){
			a.x += b_h[k * NconstT + i].x;
			a.y += b_h[k * NconstT + i].y;
			a.z += b_h[k * NconstT + i].z;
//if(i == 10) printf("%d %g %g\n", k, b_h[k * NconstT + 10].x, a.x);
			
		}
		
		a_h[i].x = a.x;
		a_h[i].y = a.y;
		a_h[i].z = a.z;
	}
	
}

// Is applied only to test particles
// parallel version with OpenMP
void Data::acc4Dsmall_cpu(){
	
	for(int i = N_h[0]; i < N_h[0] + Nsmall_h[0]; ++i){
		for(int k = 0; k < Nomp; ++k){
			b_h[k * NconstT + i] = {0.0, 0.0, 0.0};
		}
		Encpairs2_h[i * P.NencMax].x = 0;
	}
	
	#pragma omp parallel for // proc_bind(close)
	for(int i = N_h[0]; i < N_h[0] + Nsmall_h[0]; ++i){
		double ir, ir3;
		double3 r3ij;

		int k = omp_get_thread_num();
		//int k = 0;

//int cpuid = sched_getcpu();
//if(i == N_h[0] || i == N_h[0] + Nsmall_h[0] - 1) printf("acc4Dsmall %d %d %d %d\n", i, k, cpuid, 1);


		for(int j = 0; j < N_h[0]; ++j){
	
			if(x4_h[i].w >= 0.0 && x4_h[j].w >= 0.0){
				
				//printf("%d %d %d %d %d\n", k, ii, jj, i, j);
				
				r3ij.x = x4_h[j].x - x4_h[i].x;
				r3ij.y = x4_h[j].y - x4_h[i].y;
				r3ij.z = x4_h[j].z - x4_h[i].z;
				
				double rsq = (r3ij.x*r3ij.x) + (r3ij.y*r3ij.y) + (r3ij.z*r3ij.z);
				
				double rcritv = fmax(rcritv_h[i], rcritv_h[j]);
				
				ir = 1.0/sqrt(rsq);
				ir3 = ir*ir*ir;
				
				
//				if(rsq < 3.0 * rcritv * rcritv){
				if(rsq < def_pc * rcritv * rcritv && (x4_h[i].w > 0.0 || x4_h[j].w > 0.0)){
					
					int Ni = 0;
					int Nj = 0;
					#pragma omp atomic capture
					Ni = Encpairs2_h[i * P.NencMax].x++;

//printf("enc1 %d | %d %d %d %d\n", k, i, j, Ni, Nj);
					if(Ni >= P.NencMax){
						EncFlag_m[0] = max(EncFlag_m[0], Ni);
					}
					else{
						Encpairs2_h[i * P.NencMax + Ni].y = j;
					}
	
					if(P.UseTestParticles == 2){
						#pragma omp atomic capture
						Nj = Encpairs2_h[j * P.NencMax].x++;

						if(Nj >= P.NencMax){
							EncFlag_m[0] = max(EncFlag_m[0], Nj);
						}
						else{
							Encpairs2_h[j * P.NencMax + Nj].y = i;
						}
					}	
				
					if(Ni < P.NencMax && Nj < P.NencMax){
						// i < j is always true
						int Ne;
						#pragma omp atomic capture
						Ne = Nencpairs_h[0]++;

						Encpairs_h[Ne].x = i;
						Encpairs_h[Ne].y = j;
//printf("Precheck small %d | %d %d %d\n", k, i, j, Ne);
					}
					ir3 = 0.0;
				}
				else if(rsq == 0.0){
					ir3 = 0.0;
				}
				
				double sj = (x4_h[j].w * ir3);
				
				int ki = k * NconstT + i; 
				
				b_h[ki].x += r3ij.x * sj;
				b_h[ki].y += r3ij.y * sj;
				b_h[ki].z += r3ij.z * sj;

				if(P.UseTestParticles == 2){
					double si = (x4_h[i].w * ir3);
					int kj = k * NconstT + j; 
				
					b_h[kj].x -= r3ij.x * si;
					b_h[kj].y -= r3ij.y * si;
					b_h[kj].z -= r3ij.z * si;
				}
				
//if(i == 10) printf("acci %d %d %g %g\n", i, j, sj * r3ij.x, b_h[k * NconstT + i].x);
//if(j == 10) printf("accj %d %d %g %g\n", j, i, -si * r3ij.x, b_h[k * NconstT + j].x);
				
			}
		}
	}
	
	for(int i = 0; i < N_h[0]; ++i){
		double3 a = {0.0, 0.0, 0.0};
		for(int k = 0; k < Nomp; ++k){
			a.x += b_h[k * NconstT + i].x;
			a.y += b_h[k * NconstT + i].y;
			a.z += b_h[k * NconstT + i].z;
//if(i == 10) printf("%d %g %g\n", k, b_h[k * NconstT + 10].x, a.x);
			
		}
		
		a_h[i].x = a.x;
		a_h[i].y = a.y;
		a_h[i].z = a.z;
	}
	
}

//float version
void Data::acc4Df_cpu(){
	
	for(int i = 0; i < N_h[0]; ++i){
		for(int k = 0; k < Nomp; ++k){
			b_h[k * NconstT + i] = {0.0, 0.0, 0.0};
		}
		Encpairs2_h[i * P.NencMax].x = 0;
	}
	
	#pragma omp parallel for
	for(int ii = 0; ii < N_h[0] / 2; ++ii){
		float ir, ir3;
		float3 r3ij;

		int k = omp_get_thread_num();
		//int k = 0;

		for(int jj = 0; jj < N_h[0]; ++jj){
	
			if(ii == (N_h[0] + 1) / 2 - 1 && jj >= N_h[0] / 2) break;
			
			int j = jj;
			int i = ii;

			if(jj <= ii){
				i = N_h[0] - 2 - ii;
				j = N_h[0] - 1 - jj;
			}
			
			if(x4_h[i].w >= 0.0 && x4_h[j].w >= 0.0){
				
				//printf("%d %d %d %d %d\n", k, ii, jj, i, j);
				
				r3ij.x = x4_h[j].x - x4_h[i].x;
				r3ij.y = x4_h[j].y - x4_h[i].y;
				r3ij.z = x4_h[j].z - x4_h[i].z;
				
				float rsq = (r3ij.x*r3ij.x) + (r3ij.y*r3ij.y) + (r3ij.z*r3ij.z);
				
				float rcritv = fmax(rcritv_h[i], rcritv_h[j]);
				
				ir = 1.0f/sqrt(rsq);
				ir3 = ir*ir*ir;
				
				
//				if(rsq < 3.0 * rcritv * rcritv){
				if(rsq < def_pc * rcritv * rcritv && (x4_h[i].w > 0.0 || x4_h[j].w > 0.0)){
					
					int Ni, Nj;
					#pragma omp atomic capture
					Ni = Encpairs2_h[i * P.NencMax].x++;
					#pragma omp atomic capture
					Nj = Encpairs2_h[j * P.NencMax].x++;

//printf("enc1 %d | %d %d %d %d\n", k, i, j, Ni, Nj);
					if(Ni >= P.NencMax){
						EncFlag_m[0] = max(EncFlag_m[0], Ni);
					}
					else{
						Encpairs2_h[i * P.NencMax + Ni].y = j;
					}
	
					if(Nj >= P.NencMax){
						EncFlag_m[0] = max(EncFlag_m[0], Nj);
					}
					else{
						Encpairs2_h[j * P.NencMax + Nj].y = i;
					}
					
					if(Ni < P.NencMax && Nj < P.NencMax){
						// i < j is always true
						int Ne;
						#pragma omp atomic capture
						Ne = Nencpairs_h[0]++;

						Encpairs_h[Ne].x = i;
						Encpairs_h[Ne].y = j;
//printf("Precheck %d | %d %d %d\n", k, i, j, Ne);
					}
					ir3 = 0.0f;
				}
				else if(rsq == 0.0f){
					ir3 = 0.0f;
				}
				
				float si = (x4_h[i].w * ir3);
				float sj = (x4_h[j].w * ir3);
				
				int ki = k * NconstT + i; 
				int kj = k * NconstT + j; 
				
				b_h[ki].x += r3ij.x * sj;
				b_h[ki].y += r3ij.y * sj;
				b_h[ki].z += r3ij.z * sj;
				
				b_h[kj].x -= r3ij.x * si;
				b_h[kj].y -= r3ij.y * si;
				b_h[kj].z -= r3ij.z * si;
				
//if(i == 10) printf("acci %d %d %g %g\n", i, j, sj * r3ij.x, b_h[k * NconstT + i].x);
//if(j == 10) printf("accj %d %d %g %g\n", j, i, -si * r3ij.x, b_h[k * NconstT + j].x);
				
			}
		}
	}
	
	for(int i = 0; i < N_h[0]; ++i){
		double3 a = {0.0, 0.0, 0.0};
		for(int k = 0; k < Nomp; ++k){
			a.x += b_h[k * NconstT + i].x;
			a.y += b_h[k * NconstT + i].y;
			a.z += b_h[k * NconstT + i].z;
//if(i == 10) printf("%d %g %g\n", k, b_h[k * NconstT + 10].x, a.x);
			
		}
		
		a_h[i].x = a.x;
		a_h[i].y = a.y;
		a_h[i].z = a.z;
	}
	
}

// Is applied only to test particles
// parallel version with OpenMP
void Data::acc4Dfsmall_cpu(){
	
	for(int i = N_h[0]; i < N_h[0] + Nsmall_h[0]; ++i){
		for(int k = 0; k < Nomp; ++k){
			b_h[k * NconstT + i] = {0.0, 0.0, 0.0};
		}
		Encpairs2_h[i * P.NencMax].x = 0;
	}
	
	#pragma omp parallel for
	for(int i = N_h[0]; i < N_h[0] + Nsmall_h[0]; ++i){
		float ir, ir3;
		float3 r3ij;

		int k = omp_get_thread_num();
		//int k = 0;

		for(int j = 0; j < N_h[0]; ++j){
	
			if(x4_h[i].w >= 0.0 && x4_h[j].w >= 0.0){
				
				//printf("%d %d %d %d %d\n", k, ii, jj, i, j);
				
				r3ij.x = x4_h[j].x - x4_h[i].x;
				r3ij.y = x4_h[j].y - x4_h[i].y;
				r3ij.z = x4_h[j].z - x4_h[i].z;
				
				float rsq = (r3ij.x*r3ij.x) + (r3ij.y*r3ij.y) + (r3ij.z*r3ij.z);
				
				float rcritv = fmax(rcritv_h[i], rcritv_h[j]);
				
				ir = 1.0f/sqrt(rsq);
				ir3 = ir*ir*ir;
				
				
//				if(rsq < 3.0 * rcritv * rcritv){
				if(rsq < def_pc * rcritv * rcritv && (x4_h[i].w > 0.0 || x4_h[j].w > 0.0)){
					
					int Ni = 0;
					int Nj = 0;
					#pragma omp atomic capture
					Ni = Encpairs2_h[i * P.NencMax].x++;

//printf("enc1 %d | %d %d %d %d\n", k, i, j, Ni, Nj);
					if(Ni >= P.NencMax){
						EncFlag_m[0] = max(EncFlag_m[0], Ni);
					}
					else{
						Encpairs2_h[i * P.NencMax + Ni].y = j;
					}
	
					if(P.UseTestParticles == 2){
						#pragma omp atomic capture
						Nj = Encpairs2_h[j * P.NencMax].x++;

						if(Nj >= P.NencMax){
							EncFlag_m[0] = max(EncFlag_m[0], Nj);
						}
						else{
							Encpairs2_h[j * P.NencMax + Nj].y = i;
						}
					}	
				
					if(Ni < P.NencMax && Nj < P.NencMax){
						// i < j is always true
						int Ne;
						#pragma omp atomic capture
						Ne = Nencpairs_h[0]++;

						Encpairs_h[Ne].x = i;
						Encpairs_h[Ne].y = j;
//printf("Precheck %d | %d %d %d\n", k, i, j, Ne);
					}
					ir3 = 0.0f;
				}
				else if(rsq == 0.0f){
					ir3 = 0.0f;
				}
				
				float sj = (x4_h[j].w * ir3);
				
				int ki = k * NconstT + i; 
				
				b_h[ki].x += r3ij.x * sj;
				b_h[ki].y += r3ij.y * sj;
				b_h[ki].z += r3ij.z * sj;

				if(P.UseTestParticles == 2){
					float si = (x4_h[i].w * ir3);
					int kj = k * NconstT + j; 
				
					b_h[kj].x -= r3ij.x * si;
					b_h[kj].y -= r3ij.y * si;
					b_h[kj].z -= r3ij.z * si;
				}
				
//if(i == 10) printf("acci %d %d %g %g\n", i, j, sj * r3ij.x, b_h[k * NconstT + i].x);
//if(j == 10) printf("accj %d %d %g %g\n", j, i, -si * r3ij.x, b_h[k * NconstT + j].x);
				
			}
		}
	}
	
	for(int i = 0; i < N_h[0]; ++i){
		double3 a = {0.0, 0.0, 0.0};
		for(int k = 0; k < Nomp; ++k){
			a.x += b_h[k * NconstT + i].x;
			a.y += b_h[k * NconstT + i].y;
			a.z += b_h[k * NconstT + i].z;
//if(i == 10) printf("%d %g %g\n", k, b_h[k * NconstT + 10].x, a.x);
			
		}
		
		a_h[i].x = a.x;
		a_h[i].y = a.y;
		a_h[i].z = a.z;
	}
	
}
#endif


//******************************************************
// This function calculates the force between body i and j
// it must be called n^2/2 times

//Author: Simon Grimm, Joachim Stadel
// January 2015
//********************************************************
__device__  void forceij(double4 x4i, double4 x4j, double4 &fi, double4 &fj, int j, int i, const int NconstT){

	double3 r3ij;

	r3ij.x = x4j.x - x4i.x;
	r3ij.y = x4j.y - x4i.y;
	r3ij.z = x4j.z - x4i.z;

	double rsq = r3ij.x*r3ij.x + r3ij.y*r3ij.y + r3ij.z*r3ij.z;

	double rcritv = fmax(fi.w, fj.w);
	bool cl = (rsq < def_pc * rcritv * rcritv && (x4i.w > 0.0 || x4j.w > 0.0)) ? true : false;
	//Encpairsb_d[NconstT * i + j] = cl; 
	
	double ir = 1.0 / sqrt(rsq);

	double ir3 = ir * ir * ir;
	double s;

	s = x4j.w * ir3 * (!cl);

	r3ij.x *= s;
	r3ij.y *= s;
	r3ij.z *= s;

	fi.x += r3ij.x;	
	fi.y += r3ij.y;
	fi.z += r3ij.z;
	fj.x -= r3ij.x;
	fj.y -= r3ij.y;
	fj.z -= r3ij.z;
}


__device__ void accc(double3 &ac, double4 &x4i, double4 &x4j, double rcritvi, double rcritvj, int j, int i, const int NconstT){
	if( i != j && x4i.w >= 0.0 && x4j.w >= 0.0){
		double rsq, ir, ir3, s;
		double3 r3ij;
		double rcritv;

		r3ij.x = x4j.x - x4i.x;
		r3ij.y = x4j.y - x4i.y;
		r3ij.z = x4j.z - x4i.z;

		rsq = r3ij.x*r3ij.x + r3ij.y*r3ij.y + r3ij.z*r3ij.z;
		rcritv = fmax(rcritvi, rcritvj);

		bool cl = (rsq < def_pc * rcritv * rcritv && (x4i.w > 0.0 || x4j.w > 0.0)) ? true : false;
		//Encpairsb_d[NconstT * i + j] = cl;


		ir = 1.0/sqrt(rsq);
		ir3 = ir*ir*ir;

		s = x4j.w * ir3 * (!cl);

		ac.x += r3ij.x * s;
		ac.y += r3ij.y * s;
		ac.z += r3ij.z * s;
	}
}


//******************************************************
// This kernel perfomes a Kick operation on the triangle part
// of the interaction matrix
// the two indexes I and II must come from a driver routine

//The template arguments are
//p: number of threads per block, it is set in the driver routine
//nb:number of threadsblock, it is set in the driver routine

//Author: Simon Grimm, Joachim Stadel
// January 2015
//********************************************************
template <int p>
__global__ void ForceTri_kernel(double4 *x4_d, double3 *f_d, double *rcritv_d, int NconstT, int I, int II, int nb){

	int idy = threadIdx.x;
	int T = blockIdx.x;

	int J = (T ^ I);
	int F = (T & II) != 0;
	int FF = (F * (2 * nb - 1));
	int TT = T ^ FF;
	int JJ = J ^ FF;	
	__shared__ double4 x4_s[p];
	__shared__ double4 fj_s[p];

	int iii = idy + TT * p;

	x4_s[idy] = x4_d[idy + JJ * p];
	double4 x4i = x4_d[iii];

	double4 fi = {0.0, 0.0, 0.0, rcritv_d[iii]};
	fj_s[idy].x = 0.0;
	fj_s[idy].y = 0.0;
	fj_s[idy].z = 0.0;
	fj_s[idy].w = rcritv_d[idy + JJ * p];	
	__syncthreads();

	for(int i = 0; i < p; i += 32){
		for(int ii = 0; ii < 32; ++ii){
			int j = idy ^ (i + ii);
			int jjj = j + JJ * p;
			forceij(x4i, x4_s[j], fi, fj_s[j], jjj, iii, NconstT);
		}
		__syncthreads();
//printf("%d %d %d %d\n", TT, JJ, TT * p + i, JJ * p + j);
	}

	f_d[idy + TT * p].x += fi.x;
	f_d[idy + TT * p].y += fi.y;
	f_d[idy + TT * p].z += fi.z;
	f_d[idy + JJ * p].x += fj_s[idy].x;
	f_d[idy + JJ * p].y += fj_s[idy].y;
	f_d[idy + JJ * p].z += fj_s[idy].z;
}


//******************************************************
// This kernel perfomes a Kick operation on blocks on the diagonal part
// of the interaction matrix in single precision

//The template arguments are
//p: number of threads per block, it is set in the driver routine

//Author: Simon Grimm, Joachim Stadel
// January 2015
//********************************************************
template <int p>
__global__ void ForceDiag_kernel(double4 *x4_d, double3 *f_d, double *rcritv_d, int NconstT){

	int idy = threadIdx.x;
	int T = blockIdx.x;

	int J = T;
	__shared__ double4 x4_s[p];
	__shared__ double rcritv_s[p];

	int iii = idy + T * p;

	x4_s[idy] = x4_d[idy + J * p];
	rcritv_s[idy] = rcritv_d[idy + J * p];

	double4 x4i = x4_d[iii];
	double rcritvi = rcritv_d[iii];

	double3 ai = {0.0, 0.0, 0.0};
	__syncthreads();

	for(int i = 1; i < p; ++i){
		int j = idy ^ i;
		int jjj = j + J * p;
		accc(ai, x4i, x4_s[j], rcritvi, rcritv_s[j], jjj, iii, NconstT);
		__syncthreads();
	}

	double3 fi = f_d[idy + T * p];

	fi.x += ai.x * x4_s[idy].w;
	fi.y += ai.y * x4_s[idy].w;
	fi.z += ai.z * x4_s[idy].w;

	f_d[idy + T * p] = fi;
}

//******************************************************
// This kernel perfomes a Kick operation on the lower left square part
// of the interaction matrix
// the index I must come from a driver routine

//The template arguments are
//p: number of threads per block, it is set in the driver routine
//nb:number of threadsblock, it is set in the driver routine

//Author: Simon Grimm, Joachim Stadel
// January 2015
//********************************************************
template <int p>
__global__ void ForceSq_kernel(double4 *x4_d, double3 *f_d, double *rcritv_d, int NconstT, int I, int nb){

	int idy = threadIdx.x;
	int T = blockIdx.x;

	int J = (blockIdx.x ^ I) + nb;
	__shared__ double4 x4_s[p];
	__shared__ double4 fj_s[p];

	int iii = idy + T * p;

	x4_s[idy] = x4_d[idy + J * p];
	double4 x4i = x4_d[iii];

	double4 fi = {0.0, 0.0, 0.0, rcritv_d[iii]};
	fj_s[idy].x = 0.0;
	fj_s[idy].y = 0.0;
	fj_s[idy].z = 0.0;
	fj_s[idy].w = rcritv_d[idy + J * p];
	
	__syncthreads();

	for(int i = 0; i < p; i += 32){
		for(int ii = 0; ii < 32; ++ii){
			int j = idy ^ (i + ii);
			int jjj = j + J * p;
			forceij(x4i, x4_s[j], fi, fj_s[j], jjj, iii, NconstT);
		}
		__syncthreads();
	}
	f_d[idy + T * p].x += fi.x;
	f_d[idy + T * p].y += fi.y;
	f_d[idy + T * p].z += fi.z;
	f_d[idy + J * p].x += fj_s[idy].x;
	f_d[idy + J * p].y += fj_s[idy].y;
	f_d[idy + J * p].z += fj_s[idy].z;
}

__global__ void EncpairsZero_kernel(int2 *Encpairs2_d, double3 *a_d, const int NencMax, const int N){

	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if(id < N){
		Encpairs2_d[NencMax * id].x = 0;
		Encpairs2_d[NencMax * id + 1].x = 0;

		a_d[id].x = 0.0;
		a_d[id].y = 0.0;
		a_d[id].z = 0.0;
	}
}


__global__ void acclargeN_kernel(double4 *x4_d, double3 *f_d, double dtksq, const int N){

	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if(id < N){

		double im = 1.0 / x4_d[id].w;

		f_d[id].x *= im * dtksq;
		f_d[id].y *= im * dtksq;
		f_d[id].z *= im * dtksq;
	}
}

//******************************************************
// this function is a driver for the Kick kernels
// it splits thes N * N matrix into smaller blocks of lenght p * p
// this blocks are devided into 3 sets:
// set 1: blocks on the diagonal
// set 2: upper left triangle and lowet right triangle
// set 3: lower left square

// p sets the size of the blocks and the number of threads per block

//Author: Simon Grimm, Joachim Stadel
// January 2015
//********************************************************
__host__ void ForceDriver(double4 *x4_d, double *rcritv_d, double3 *f_d, int2 *Encpairs2_d, double dtksq, int NconstT, int NencMax, int NB, int N){

	const int p = 256;
	const int nb = NB / (2 * p);

	//set NencpairsI and NencpairsJ to zero
	EncpairsZero_kernel <<< (NB + p - 1) / p, p >>> (Encpairs2_d, f_d, NencMax, NB);
	//Blocks on the Diagonal
	ForceDiag_kernel < p > <<< NB / p, p>>> (x4_d, f_d, rcritv_d, NconstT);

	//Combine upper left quarter triangle with lower right quarter triangle
	for(int ii = 1; ii < nb; ii *= 2){
		for(int k = 0; k < ii; ++k){
			int i = ii + k;
			ForceTri_kernel < p > <<< nb, p>>> (x4_d, f_d, rcritv_d, NconstT, i, ii, nb);
		}
	}

	//Lower left quarter
	for(int i = 0; i < nb; ++i){
		ForceSq_kernel < p > <<< nb, p >>> (x4_d, f_d, rcritv_d, NconstT, i, nb);
	}

	acclargeN_kernel <<< (N + p - 1) / p, p >>> (x4_d, f_d, dtksq, N);

}




#endif

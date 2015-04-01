#ifndef KICK4_H
#define KICK4_H
#include "define.h"


//**************************************
//This function computes the terms a = mi/rij^3 * Kij and b = mi/rij.
//This function also finds the pairs of bodies which are separated less than pc * rcritv^2. The index of those 
//pairs are stored in the array Encpairs_d in two different ways. This indexes are then used
//in the KickA32 kernel and in the Encounter kernel.

//Authors: Simon Grimm, Joachim Stadel
//March 2015

//****************************************
__device__ void  accb(double3 &ac, double4 &x4i, double4 &x4j, double rcritvi, double rcritvj, int *NencpairsI, int *NencpairsJ, int2 *Encpairs_d, int j, int i, int icNB){
	if( i != j && x4i.w >= 0.0 && x4j.w >= 0.0){
		double rsq, ir, ir3, s;
		double3 r3ij;
		double rcritv;
		int Ni, Nj;

		r3ij.x = x4j.x - x4i.x;
		r3ij.y = x4j.y - x4i.y;
		r3ij.z = x4j.z - x4i.z;

		rsq = r3ij.x*r3ij.x + r3ij.y*r3ij.y + r3ij.z*r3ij.z;
		rcritv = fmax(rcritvi, rcritvj);

		int cl = (rsq < pc * rcritv * rcritv) ? 1 : 0;

		if(cl && (x4i.w > 0.0 || x4j.w > 0.0)){  //prechecker
//printf("Precheck %d %d\n", i, j);
			if( i < j){
				Ni = atomicAdd(NencpairsI, 1);
				Encpairs_d[icNB * i + Ni].x = i;
				Encpairs_d[icNB * i + Ni].y = j;
			}
			else{
				Nj = atomicAdd(NencpairsJ, 1);
				Encpairs_d[icNB * i + icNB - 1 - Nj].y = j;
			}

		}

		ir = 1.0/sqrt(rsq);
		ir3 = ir*ir*ir;

		if(cl){
			s = 0.0;
		}
		else{
			s = x4j.w * ir3;
		}

		ac.x += r3ij.x * s;
		ac.y += r3ij.y * s;
		ac.z += r3ij.z * s;
	}
}

// **************************************
//This kernel performs the seccond kick of the time step, in the case NB = 16. NB is the next bigger number of N
//which is a power of two.
//It calculates the acceleration between all bodies with respect to the changeover function K.
//It also calculates all accelerations from bodies not beeing in a close encounter and store it in accK_d. This values will then be used 
//it the next time step.
//It performs also a precheck for close encouter candidates. This pairs are stored in the array Encpairs_d.
//The number of close encounter candidates is stored in Nencpairs_d.
//
//The Kernel is launched with N blocks a NB theads.
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
//****************************************
template <const int Bl, int Bl2>
__global__ void acc16_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double *rcrit_d, double *rcritv_d, int *groupIndex_d, const double dtksq, int *Nencpairs_d, int2 *Encpairs_d, int2 *Encpairs2_d, double *test_d, int icNB, double t){
	int idy = threadIdx.x;
	int idx = blockIdx.x;

	__shared__ double3 ab1_s[Bl2]; 		//the b1_s array is here stored in ab1_s[idy + 16]
	__shared__ int NencpairsI_s;
	__shared__ int NencpairsJ_s;

	double4 x4i = x4_d[idx];
	double rcritvi = rcritv_d[idx];

	double4 x4j;
	double rcritvj;

	if(idy < Bl){
		x4j = x4_d[idy];
		rcritvj = rcritv_d[idy];
	}
	else{
		x4j.x = 0.0;
		x4j.y = 0.0;
		x4j.z = 0.0;
		x4j.w = 0.0;
		rcritvj = 0.0;
	}

	if(idy == 0){
		NencpairsI_s = 0;
		NencpairsJ_s = 0;
	}

	__syncthreads();

	ab1_s[idy].x = 0.0;
	ab1_s[idy].y = 0.0;
	ab1_s[idy].z = 0.0;

	ab1_s[idy + 8].x = 0.0;
	ab1_s[idy + 8].y = 0.0;
	ab1_s[idy + 8].z = 0.0;

	__syncthreads();

	if(idy < Bl){
		accb(ab1_s[idy], x4i, x4j, rcritvi, rcritvj, &NencpairsI_s, &NencpairsJ_s, Encpairs2_d, idy, idx, icNB); 
	}

	__syncthreads();
	volatile double3 *ab1 = ab1_s;

	ab1[idy].x += ab1[idy + 8].x;
	ab1[idy].x += ab1[idy + 4].x;
	ab1[idy].x += ab1[idy + 2].x;
	ab1[idy].x += ab1[idy + 1].x;

	ab1[idy].y += ab1[idy + 8].y;
	ab1[idy].y += ab1[idy + 4].y;
	ab1[idy].y += ab1[idy + 2].y;
	ab1[idy].y += ab1[idy + 1].y;

	ab1[idy].z += ab1[idy + 8].z;
	ab1[idy].z += ab1[idy + 4].z;
	ab1[idy].z += ab1[idy + 2].z;
	ab1[idy].z += ab1[idy + 1].z;


	__shared__ int Ne;
	__syncthreads();

	if(idy == 0){
		acck_d[idx].x = ab1[0].x * dtksq;
		acck_d[idx].y = ab1[0].y * dtksq;
		acck_d[idx].z = ab1[0].z * dtksq;
		Ne = atomicAdd(Nencpairs_d, NencpairsI_s);
	}
	__syncthreads();
	if(idy < Bl){
		for(int i = 0; i < NencpairsI_s; i += Bl){
			if(idy + i < NencpairsI_s){
				Encpairs_d[idy + i + Ne] = Encpairs2_d[icNB * idx + idy + i];
			}
		}
	}
	__syncthreads();
	if(idy == 0){
		Encpairs2_d[icNB * idx].x = NencpairsI_s;
		Encpairs2_d[icNB * idx + 1].x = NencpairsJ_s;

	}
}

// **************************************
//This kernel performs the seccond kick of the time step, in the case 32 <= NB < 128. NB is the next bigger number of N
//which is a power of two.
//It calculates the acceleration between all bodies with respect to the changeover function K.
//It also calculates all accelerations from bodies not beeing in a close encounter. This values will then be used 
//it the next time step.
//It performs also a precheck for close encouter candidates. This pairs are stored in the array Encpairs_d.
//The number of close encounter candidates is stored in Nencpairs_d.
//
//The Kernel is launched with N blocks a NB theads.

//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
//****************************************
template <const int Bl, int Bl2>
__global__ void acc32_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double *rcrit_d, double *rcritv_d, int *groupIndex_d, const double dtksq, int *Nencpairs_d, int2 *Encpairs_d, int2 *Encpairs2_d, double *test_d, int icNB, double t){
	int idy = threadIdx.x;
	int idx = blockIdx.x;

	__shared__ double3 a1_s[Bl2];
	__shared__ int NencpairsI_s;
	__shared__ int NencpairsJ_s;

	double4 x4i = x4_d[idx];
	double rcritvi = rcritv_d[idx];
#if G3 == 1
	int groupIndexi = groupIndex_d[idx];
#endif

	double4 x4j = x4_d[idy];
	double rcritvj = rcritv_d[idy];
#if G3 == 1
	int groupIndexj = groupIndex_d[idy];
#endif

	if(idy == 0){
		NencpairsI_s = 0;
		NencpairsJ_s = 0;
	}
	__syncthreads();
	a1_s[idy].x = 0.0;
	a1_s[idy].y = 0.0;
	a1_s[idy].z = 0.0;

	__syncthreads();
	if(idy < Bl){
		accb(a1_s[idy], x4i, x4j, rcritvi, rcritvj, &NencpairsI_s, &NencpairsJ_s, Encpairs2_d, idy, idx, icNB); 
	}

	__syncthreads();
	volatile double3 *a1 = a1_s;

	if(idy < 32){
		if(Bl >= 64) a1[idy].x += a1[idy + 32].x;
		if(Bl >= 32) a1[idy].x += a1[idy + 16].x;
		a1[idy].x += a1[idy + 8].x;
		a1[idy].x += a1[idy + 4].x;
		a1[idy].x += a1[idy + 2].x;
		a1[idy].x += a1[idy + 1].x;

		if(Bl >= 64) a1[idy].y += a1[idy + 32].y;
		if(Bl >= 32) a1[idy].y += a1[idy + 16].y;
		a1[idy].y += a1[idy + 8].y;
		a1[idy].y += a1[idy + 4].y;
		a1[idy].y += a1[idy + 2].y;
		a1[idy].y += a1[idy + 1].y;

		if(Bl >= 64) a1[idy].z += a1[idy + 32].z;
		if(Bl >= 32) a1[idy].z += a1[idy + 16].z;
		a1[idy].z += a1[idy + 8].z;
		a1[idy].z += a1[idy + 4].z;
		a1[idy].z += a1[idy + 2].z;
		a1[idy].z += a1[idy + 1].z;
	}

	__shared__ int Ne;
	__syncthreads();

	if(idy == 0){
		acck_d[idx].x = a1[0].x * dtksq;
		acck_d[idx].y = a1[0].y * dtksq;
		acck_d[idx].z = a1[0].z * dtksq;
		Ne = atomicAdd(Nencpairs_d, NencpairsI_s);
	}
	__syncthreads();
	if(idy < Bl){
		for(int i = 0; i < NencpairsI_s; i += Bl){
			if(idy + i < NencpairsI_s){
				Encpairs_d[idy + i + Ne] = Encpairs2_d[icNB * idx + idy + i];
			}
		}
	}
	__syncthreads();
	if(idy == 0){
		Encpairs2_d[icNB * idx].x = NencpairsI_s;
		Encpairs2_d[icNB * idx + 1].x = NencpairsJ_s;

	}
}

// **************************************
//This kernel performs the seccond kick of the time step, in the case NB = 128. NB is the next bigger number of N
//which is a power of two.
//It calculates the acceleration between all bodies with respect to the changeover function K.
//It also calculates all accelerations from bodies not beeing in a close encounter. This values will then be used 
//it the next time step.
//It performs also a precheck for close encouter candidates. This pairs are stored in the array Encpairs_d.
//The number of close encounter candidates is stored in Nencpairs_d.
//
//The Kernel is launched with N/2 blocks a 128 theads.
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
// ****************************************
template <int Bl>
__global__ void acc128_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double *rcrit_d, double *rcritv_d, int *groupIndex_d, const double dtksq, int *Nencpairs_d, int2 *Encpairs_d, int2 *Encpairs2_d, double *test_d, int N2, int icNB, double t){

	int idy = threadIdx.x;
	int idx = blockIdx.x;

	__shared__ double3 a1_s[Bl];
	__shared__ double3 a2_s[Bl];

	__shared__ int NencpairsI_s;
	__shared__ int NencpairsI2_s;
	__shared__ int NencpairsJ_s;
	__shared__ int NencpairsJ2_s;


	double4 x4i = x4_d[idx];
	double4 x4i2 = x4_d[idx + N2];
	double4 x4j = x4_d[idy];

	double rcritvj = rcritv_d[idy];
#if G3 == 1
	int groupIndexj = groupIndex_d[idy];
#endif

	double rcritvi = rcritv_d[idx];
	double rcritvi2 = rcritv_d[idx + N2];
#if G3 == 1
	int groupIndexi = groupIndex_d[idx];
	int groupIndexi2 = groupIndex_d[idx + N2];
#endif

	a1_s[idy].x = 0.0;
	a1_s[idy].y = 0.0;
	a1_s[idy].z = 0.0;
	
	a2_s[idy].x = 0.0;
	a2_s[idy].y = 0.0;
	a2_s[idy].z = 0.0;

	if(idy == 0){
		NencpairsI_s = 0;
		NencpairsJ_s = 0;
	}
	if(idy == 32){
		NencpairsI2_s = 0;
		NencpairsJ2_s = 0;
	}

	__syncthreads();
	accb(a1_s[idy], x4i, x4j, rcritvi, rcritvj, &NencpairsI_s, &NencpairsJ_s, Encpairs2_d, idy, idx, icNB);
	accb(a2_s[idy], x4i2, x4j, rcritvi2, rcritvj, &NencpairsI2_s, &NencpairsJ2_s, Encpairs2_d, idy, idx + N2, icNB);

	__syncthreads();
	volatile double3 *a1 = a1_s;
	volatile double3 *a2 = a2_s;

	if(idy < 64){
		a1[idy].x += a1[idy + 64].x;
		a1[idy].y += a1[idy + 64].y;
		a1[idy].z += a1[idy + 64].z;

		a2[idy].x += a2[idy + 64].x;
		a2[idy].y += a2[idy + 64].y;
		a2[idy].z += a2[idy + 64].z;
	}
	__syncthreads();

	if(idy < 32){
		a1[idy].x += a1[idy + 32].x;
		a1[idy].x += a1[idy + 16].x;
		a1[idy].x += a1[idy + 8].x;
		a1[idy].x += a1[idy + 4].x;
		a1[idy].x += a1[idy + 2].x;
		a1[idy].x += a1[idy + 1].x;

		a1[idy].y += a1[idy + 32].y;
		a1[idy].y += a1[idy + 16].y;
		a1[idy].y += a1[idy + 8].y;
		a1[idy].y += a1[idy + 4].y;
		a1[idy].y += a1[idy + 2].y;
		a1[idy].y += a1[idy + 1].y;

		a1[idy].z += a1[idy + 32].z;
		a1[idy].z += a1[idy + 16].z;
		a1[idy].z += a1[idy + 8].z;
		a1[idy].z += a1[idy + 4].z;
		a1[idy].z += a1[idy + 2].z;
		a1[idy].z += a1[idy + 1].z;
	}

	else{
		if(idy < 64){
			
			a2[idy-32].x += a2[idy + 32-32].x;
			a2[idy-32].x += a2[idy + 16-32].x;
			a2[idy-32].x += a2[idy + 8-32].x;
			a2[idy-32].x += a2[idy + 4-32].x;
			a2[idy-32].x += a2[idy + 2-32].x;
			a2[idy-32].x += a2[idy + 1-32].x;

			a2[idy-32].y += a2[idy + 32-32].y;
			a2[idy-32].y += a2[idy + 16-32].y;
			a2[idy-32].y += a2[idy + 8-32].y;
			a2[idy-32].y += a2[idy + 4-32].y;
			a2[idy-32].y += a2[idy + 2-32].y;
			a2[idy-32].y += a2[idy + 1-32].y;
	
			a2[idy-32].z += a2[idy + 32-32].z;
			a2[idy-32].z += a2[idy + 16-32].z;
			a2[idy-32].z += a2[idy + 8-32].z;
			a2[idy-32].z += a2[idy + 4-32].z;
			a2[idy-32].z += a2[idy + 2-32].z;
			a2[idy-32].z += a2[idy + 1-32].z;	
		}
	}
	__shared__ int Ne1;
	__shared__ int Ne2;

	__syncthreads();

	if(idy == 0){
		acck_d[idx].x = a1[0].x * dtksq;
		acck_d[idx].y = a1[0].y * dtksq;
		acck_d[idx].z = a1[0].z * dtksq;
		Ne1 = atomicAdd(Nencpairs_d, NencpairsI_s);
	}

	if(idy == 32){
		acck_d[idx + N2].x = a2[0].x * dtksq;
		acck_d[idx + N2].y = a2[0].y * dtksq;
		acck_d[idx + N2].z = a2[0].z * dtksq;
		Ne2 = atomicAdd(Nencpairs_d, NencpairsI2_s);
	}

	__syncthreads();


	if(idy < NencpairsI_s){
		Encpairs_d[idy + Ne1] = Encpairs2_d[icNB * idx + idy];
	}
	__syncthreads();
	if(idy < NencpairsI2_s){
		Encpairs_d[idy + Ne2] = Encpairs2_d[icNB * (idx + N2) + idy];

	}
	__syncthreads();


	if(idy == 0){
		Encpairs2_d[icNB * idx].x = NencpairsI_s;
		Encpairs2_d[icNB * idx + 1].x = NencpairsJ_s;

	}
	if(idy == 32){
		Encpairs2_d[icNB * (idx + N2)].x = NencpairsI2_s;
		Encpairs2_d[icNB * (idx + N2) + 1].x = NencpairsJ2_s;

	}
}

// **************************************
//This kernel performs the seccond kick of the time step, in the case NB = 256. NB is the next bigger number of N
//which is a power of two.
//It calculates the acceleration between all bodies with respect to the changeover function K.
//It also calculates all accelerations from bodies not beeing in a close encounter. This values will then be used 
//it the next time step.
//It performs also a precheck for close encouter candidates. This pairs are stored in the array Encpairs_d.
//The number of close encounter candidates is stored in Nencpairs_d.
//
//The Kernel is launched with N/4 blocks a 128 theads.
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
// ****************************************
template <int Bl, int NB>
__global__ void acc256_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double *rcrit_d, double *rcritv_d, int *groupIndex_d, const double dtksq, int N4, int *Nencpairs_d, int2 *Encpairs_d, int2 *Encpairs2_d, double *test_d, int N, int icNB, double t){
	int idy = threadIdx.x;
	int idx = blockIdx.x;

	__shared__ double3 a1_s[Bl];
	__shared__ double3 a2_s[Bl];
	__shared__ double3 a3_s[Bl];
	__shared__ double3 a4_s[Bl];

	__shared__ int NencpairsI_s;
	__shared__ int NencpairsI2_s;
	__shared__ int NencpairsI3_s;
	__shared__ int NencpairsI4_s;

	__shared__ int NencpairsJ_s;
	__shared__ int NencpairsJ2_s;
	__shared__ int NencpairsJ3_s;
	__shared__ int NencpairsJ4_s;


	double4 x4i = x4_d[idx];
	double4 x4i2 = x4_d[idx + N4];
	double4 x4i3 = x4_d[idx + 2*N4];
	double4 x4i4 = x4_d[idx + 3*N4];

	double rcritvi = rcritv_d[idx];
	double rcritvi2 = rcritv_d[idx + N4];
	double rcritvi3 = rcritv_d[idx + 2*N4];
	double rcritvi4 = rcritv_d[idx + 3*N4];
#if G3 == 1
	int groupIndexi = groupIndex_d[idx];
	int groupIndexi2 = groupIndex_d[idx + N4];
	int groupIndexi3 = groupIndex_d[idx + 2*N4];
	int groupIndexi4 = groupIndex_d[idx + 3*N4];
#endif


	if(idy == 0){
		NencpairsI_s = 0;
		NencpairsJ_s = 0;
	}
	if(idy == 32){
		NencpairsI2_s = 0;
		NencpairsJ2_s = 0;
	}
	if(idy == 64){
		NencpairsI3_s = 0;
		NencpairsJ3_s = 0;
	}
	if(idy == 96){
		NencpairsI4_s = 0;
		NencpairsJ4_s = 0;
	}
	
	__syncthreads();

	a1_s[idy].x = 0.0;
	a1_s[idy].y = 0.0;
	a1_s[idy].z = 0.0;
	
	a2_s[idy].x = 0.0;
	a2_s[idy].y = 0.0;
	a2_s[idy].z = 0.0;

	a3_s[idy].x = 0.0;
	a3_s[idy].y = 0.0;
	a3_s[idy].z = 0.0;
	
	a4_s[idy].x = 0.0;
	a4_s[idy].y = 0.0;
	a4_s[idy].z = 0.0;

	__syncthreads();


	for(int i = 0; i < NB; i += Bl){ 
		if(idy + i < N){
			double4 x4j = x4_d[idy + i];
			double rcritvj = rcritv_d[idy + i];
			accb(a1_s[idy], x4i, x4j, rcritvi, rcritvj, &NencpairsI_s, &NencpairsJ_s, Encpairs2_d, idy + i, idx, icNB);
			accb(a2_s[idy], x4i2, x4j, rcritvi2, rcritvj, &NencpairsI2_s, &NencpairsJ2_s, Encpairs2_d, idy + i, idx + N4, icNB);
			accb(a3_s[idy], x4i3, x4j, rcritvi3, rcritvj, &NencpairsI3_s, &NencpairsJ3_s, Encpairs2_d, idy + i, idx + 2*N4, icNB);
			accb(a4_s[idy], x4i4, x4j, rcritvi4, rcritvj, &NencpairsI4_s, &NencpairsJ4_s, Encpairs2_d, idy + i, idx + 3*N4, icNB);
		}
	}
	__syncthreads();
	volatile double3 *a1 = a1_s;
	volatile double3 *a2 = a2_s;
	volatile double3 *a3 = a3_s;
	volatile double3 *a4 = a4_s;

	if(Bl >= 256){
		if(idy < 128){
			a1[idy].x += a1[idy + 128].x;
			a1[idy].y += a1[idy + 128].y;
			a1[idy].z += a1[idy + 128].z;

			a2[idy].x += a2[idy + 128].x;
			a2[idy].y += a2[idy + 128].y;
			a2[idy].z += a2[idy + 128].z;

			a3[idy].x += a3[idy + 128].x;
			a3[idy].y += a3[idy + 128].y;
			a3[idy].z += a3[idy + 128].z;

			a4[idy].x += a4[idy + 128].x;
			a4[idy].y += a4[idy + 128].y;
			a4[idy].z += a4[idy + 128].z;

		}
	}
	__syncthreads();

	if(idy < 64){
		a1[idy].x += a1[idy + 64].x;
		a1[idy].y += a1[idy + 64].y;
		a1[idy].z += a1[idy + 64].z;

		a2[idy].x += a2[idy + 64].x;
		a2[idy].y += a2[idy + 64].y;
		a2[idy].z += a2[idy + 64].z;

		a3[idy].x += a3[idy + 64].x;
		a3[idy].y += a3[idy + 64].y;
		a3[idy].z += a3[idy + 64].z;

		a4[idy].x += a4[idy + 64].x;
		a4[idy].y += a4[idy + 64].y;
		a4[idy].z += a4[idy + 64].z;
	}
	__syncthreads();

	if(idy < 32){
		a1[idy].x += a1[idy + 32].x;
		a1[idy].x += a1[idy + 16].x;
		a1[idy].x += a1[idy + 8].x;
		a1[idy].x += a1[idy + 4].x;
		a1[idy].x += a1[idy + 2].x;
		a1[idy].x += a1[idy + 1].x;

		a1[idy].y += a1[idy + 32].y;
		a1[idy].y += a1[idy + 16].y;
		a1[idy].y += a1[idy + 8].y;
		a1[idy].y += a1[idy + 4].y;
		a1[idy].y += a1[idy + 2].y;
		a1[idy].y += a1[idy + 1].y;

		a1[idy].z += a1[idy + 32].z;
		a1[idy].z += a1[idy + 16].z;
		a1[idy].z += a1[idy + 8].z;
		a1[idy].z += a1[idy + 4].z;
		a1[idy].z += a1[idy + 2].z;
		a1[idy].z += a1[idy + 1].z;
	}

	else{
		if(idy < 64){
			a2[idy-32].x += a2[idy + 32-32].x;
			a2[idy-32].x += a2[idy + 16-32].x;
			a2[idy-32].x += a2[idy + 8-32].x;
			a2[idy-32].x += a2[idy + 4-32].x;
			a2[idy-32].x += a2[idy + 2-32].x;
			a2[idy-32].x += a2[idy + 1-32].x;

			a2[idy-32].y += a2[idy + 32-32].y;
			a2[idy-32].y += a2[idy + 16-32].y;
			a2[idy-32].y += a2[idy + 8-32].y;
			a2[idy-32].y += a2[idy + 4-32].y;
			a2[idy-32].y += a2[idy + 2-32].y;
			a2[idy-32].y += a2[idy + 1-32].y;
	
			a2[idy-32].z += a2[idy + 32-32].z;
			a2[idy-32].z += a2[idy + 16-32].z;
			a2[idy-32].z += a2[idy + 8-32].z;
			a2[idy-32].z += a2[idy + 4-32].z;
			a2[idy-32].z += a2[idy + 2-32].z;
			a2[idy-32].z += a2[idy + 1-32].z;	
		}
		else{
			if(idy < 96){
				a3[idy-64].x += a3[idy + 32-64].x;
				a3[idy-64].x += a3[idy + 16-64].x;
				a3[idy-64].x += a3[idy + 8-64].x;
				a3[idy-64].x += a3[idy + 4-64].x;
				a3[idy-64].x += a3[idy + 2-64].x;
				a3[idy-64].x += a3[idy + 1-64].x;
				
				a3[idy-64].y += a3[idy + 32-64].y;
				a3[idy-64].y += a3[idy + 16-64].y;
				a3[idy-64].y += a3[idy + 8-64].y;
				a3[idy-64].y += a3[idy + 4-64].y;
				a3[idy-64].y += a3[idy + 2-64].y;
				a3[idy-64].y += a3[idy + 1-64].y;
				
				a3[idy-64].z += a3[idy + 32-64].z;
				a3[idy-64].z += a3[idy + 16-64].z;
				a3[idy-64].z += a3[idy + 8-64].z;
				a3[idy-64].z += a3[idy + 4-64].z;
				a3[idy-64].z += a3[idy + 2-64].z;
				a3[idy-64].z += a3[idy + 1-64].z;
			}
			else{
				if(idy < 128){
					a4[idy-96].x += a4[idy + 32-96].x;
					a4[idy-96].x += a4[idy + 16-96].x;
					a4[idy-96].x += a4[idy + 8-96].x;
					a4[idy-96].x += a4[idy + 4-96].x;
					a4[idy-96].x += a4[idy + 2-96].x;
					a4[idy-96].x += a4[idy + 1-96].x;

					a4[idy-96].y += a4[idy + 32-96].y;
					a4[idy-96].y += a4[idy + 16-96].y;
					a4[idy-96].y += a4[idy + 8-96].y;
					a4[idy-96].y += a4[idy + 4-96].y;
					a4[idy-96].y += a4[idy + 2-96].y;
					a4[idy-96].y += a4[idy + 1-96].y;

					a4[idy-96].z += a4[idy + 32-96].z;
					a4[idy-96].z += a4[idy + 16-96].z;
					a4[idy-96].z += a4[idy + 8-96].z;
					a4[idy-96].z += a4[idy + 4-96].z;
					a4[idy-96].z += a4[idy + 2-96].z;
					a4[idy-96].z += a4[idy + 1-96].z;
				}
			}
		}
	}
	__shared__ int Ne1;
	__shared__ int Ne2;
	__shared__ int Ne3;
	__shared__ int Ne4;
	__syncthreads();

	if(idy == 0){
		acck_d[idx].x = a1[0].x * dtksq;
		acck_d[idx].y = a1[0].y * dtksq;
		acck_d[idx].z = a1[0].z * dtksq;
		Ne1 = atomicAdd(Nencpairs_d, NencpairsI_s);
	}

	if(idy == 32){
		acck_d[idx + N4].x = a2[0].x * dtksq;
		acck_d[idx + N4].y = a2[0].y * dtksq;
		acck_d[idx + N4].z = a2[0].z * dtksq;
		Ne2 = atomicAdd(Nencpairs_d, NencpairsI2_s);
	}

	if(idy == 64){
		acck_d[idx + 2*N4].x = a3[0].x * dtksq;
		acck_d[idx + 2*N4].y = a3[0].y * dtksq;
		acck_d[idx + 2*N4].z = a3[0].z * dtksq;
		Ne3 = atomicAdd(Nencpairs_d, NencpairsI3_s);
	}
	
	if(idy == 96){
		acck_d[idx + 3*N4].x = a4[0].x * dtksq;
		acck_d[idx + 3*N4].y = a4[0].y * dtksq;
		acck_d[idx + 3*N4].z = a4[0].z * dtksq;
		Ne4 = atomicAdd(Nencpairs_d, NencpairsI4_s);
	}

	__syncthreads();

	for(int i = 0; i < NencpairsI_s; i += Bl){
		if(idy + i < NencpairsI_s){
			Encpairs_d[idy + i + Ne1] = Encpairs2_d[icNB * idx + idy + i];
		}
	}
	__syncthreads();
	for(int i = 0; i < NencpairsI2_s; i += Bl){
		if(idy + i < NencpairsI2_s){
			Encpairs_d[idy + i + Ne2] = Encpairs2_d[icNB * (idx + N4) + idy + i];
		}
	}
	__syncthreads();
	for(int i = 0; i < NencpairsI3_s; i += Bl){
		if(idy + i < NencpairsI3_s){
			Encpairs_d[idy + i + Ne3] = Encpairs2_d[icNB * (idx + 2*N4) + idy + i];
		}
	}
	__syncthreads();
	for(int i = 0; i < NencpairsI4_s; i += Bl){
		if(idy + i < NencpairsI4_s){
		Encpairs_d[idy + i + Ne4] = Encpairs2_d[icNB * (idx + 3*N4) + idy + i];
		}
	}
	__syncthreads();

	if(idy == 0){
		Encpairs2_d[icNB * idx].x = NencpairsI_s;
		Encpairs2_d[icNB * idx + 1].x = NencpairsJ_s;
	}
	if(idy == 32){
		Encpairs2_d[icNB * (idx + N4)].x = NencpairsI2_s;
		Encpairs2_d[icNB * (idx + N4) + 1].x = NencpairsJ2_s;
	}
	if(idy == 64){
		Encpairs2_d[icNB * (idx + 2*N4)].x = NencpairsI3_s;
		Encpairs2_d[icNB * (idx + 2*N4) + 1].x = NencpairsJ3_s;
	}
	if(idy == 96){
		Encpairs2_d[icNB * (idx + 3*N4)].x = NencpairsI4_s;
		Encpairs2_d[icNB * (idx + 3*N4) + 1].x = NencpairsJ4_s;
	}
}

// **************************************
//This kernel performs the seccond kick of the time step, in the case NB > 256. NB is the next bigger number of N
//which is a power of two.
//It calculates the acceleration between all bodies with respect to the changeover function K.
//It also calculates all accelerations from bodies not beeing in a close encounter. This values will then be used 
//it the next time step.
//It performs also a precheck for close encouter candidates. This pairs are stored in the array Encpairs_d.
//The number of close encounter candidates is stored in Nencpairs_d.
//
//The Kernel is launched with N/4 blocks a 128 theads.
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
// ****************************************
template <int Bl>
__global__ void acc4_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double *rcrit_d, double *rcritv_d, int *groupIndex_d, const double dtksq, int N4, int *Nencpairs_d, int2 *Encpairs_d, int2 *Encpairs2_d, double *test_d, int N, int icNB, int NB, double t){
	int idy = threadIdx.x;
	int idx = blockIdx.x;

	int Bl_2 = Bl/2;

	__shared__ double3 a1_s[Bl/2];
	__shared__ double3 a2_s[Bl/2];
	__shared__ double3 a3_s[Bl/2];
	__shared__ double3 a4_s[Bl/2];

	__shared__ int NencpairsI_s;
	__shared__ int NencpairsI2_s;
	__shared__ int NencpairsI3_s;
	__shared__ int NencpairsI4_s;

	__shared__ int NencpairsJ_s;
	__shared__ int NencpairsJ2_s;
	__shared__ int NencpairsJ3_s;
	__shared__ int NencpairsJ4_s;

	double4 x4i = x4_d[idx];
	double4 x4i2 = x4_d[idx+N4];
	double4 x4i3 = x4_d[idx+2*N4];
	double4 x4i4 = x4_d[idx+3*N4];

	double rcritvi = rcritv_d[idx];
	double rcritvi2 = rcritv_d[idx+N4];
	double rcritvi3 = rcritv_d[idx+2*N4];
	double rcritvi4 = rcritv_d[idx+3*N4];
#if G3 == 1
	int groupIndexi = groupIndex_d[idx];
	int groupIndexi2 = groupIndex_d[idx + N4];
	int groupIndexi3 = groupIndex_d[idx + 2*N4];
	int groupIndexi4 = groupIndex_d[idx + 3*N4];
#endif

	if(idy == 0){
		NencpairsI_s = 0;
		NencpairsJ_s = 0;
	}
	if(idy == 32){
		NencpairsI2_s = 0;
		NencpairsJ2_s = 0;
	}
	if(idy == 64){
		NencpairsI3_s = 0;
		NencpairsJ3_s = 0;
	}
	if(idy == 96){
		NencpairsI4_s = 0;
		NencpairsJ4_s = 0;
	}
	

	__syncthreads();

	if(idy < Bl_2) {
		a1_s[idy].x = 0.0;
		a1_s[idy].y = 0.0;
		a1_s[idy].z = 0.0;

		a3_s[idy].x = 0.0;
		a3_s[idy].y = 0.0;
		a3_s[idy].z = 0.0;

		__syncthreads();
		for(int i = 0; i < NB; i += Bl_2){
			if(idy + i < N){
				double4 x4j = x4_d[idy + i];
				double rcritvj = rcritv_d[idy + i];
				accb(a1_s[idy], x4i, x4j, rcritvi, rcritvj, &NencpairsI_s, &NencpairsJ_s, Encpairs2_d, idy + i, idx, icNB);
				accb(a3_s[idy], x4i3, x4j, rcritvi3, rcritvj, &NencpairsI3_s, &NencpairsJ3_s, Encpairs2_d, idy + i, idx +2*N4, icNB);
			}	
		}
	}
	else{
		a2_s[idy-Bl_2].x = 0.0;
		a2_s[idy-Bl_2].y = 0.0;
		a2_s[idy-Bl_2].z = 0.0;

		a4_s[idy-Bl_2].x = 0.0;
		a4_s[idy-Bl_2].y = 0.0;
		a4_s[idy-Bl_2].z = 0.0;

		__syncthreads();

		for(int i = 0; i < NB; i += Bl_2){
			if(idy-Bl_2 + i < N){
				double4 x4j = x4_d[idy-Bl_2 + i];
				double rcritvj = rcritv_d[idy-Bl_2 + i];
				accb(a2_s[idy-Bl_2], x4i2, x4j, rcritvi2, rcritvj, &NencpairsI2_s, &NencpairsJ2_s, Encpairs2_d, idy-Bl_2 + i, idx +N4, icNB);
				accb(a4_s[idy-Bl_2], x4i4, x4j, rcritvi4, rcritvj, &NencpairsI4_s, &NencpairsJ4_s, Encpairs2_d, idy-Bl_2 + i, idx +3*N4, icNB);
			}
		}
	}
	__syncthreads();

	volatile double3 *a1 = a1_s;
	volatile double3 *a2 = a2_s;
	volatile double3 *a3 = a3_s;
	volatile double3 *a4 = a4_s;

	int s = Bl/4;

	for(int i = 6; i < log2f(Bl/2); ++i){
		if( idy < s ) {
			a1[idy].x += a1[idy + s].x;
			a1[idy].y += a1[idy + s].y;
			a1[idy].z += a1[idy + s].z;

			a2[idy].x += a2[idy + s].x;
			a2[idy].y += a2[idy + s].y;
			a2[idy].z += a2[idy + s].z;

			a3[idy].x += a3[idy + s].x;
			a3[idy].y += a3[idy + s].y;
			a3[idy].z += a3[idy + s].z;

			a4[idy].x += a4[idy + s].x;
			a4[idy].y += a4[idy + s].y;
			a4[idy].z += a4[idy + s].z;
		}
		__syncthreads();
		s /= 2;
	}

	if(idy < 32){
		a1[idy].x += a1[idy + 32].x;
		a1[idy].x += a1[idy + 16].x;
		a1[idy].x += a1[idy + 8].x;
		a1[idy].x += a1[idy + 4].x;
		a1[idy].x += a1[idy + 2].x;
		a1[idy].x += a1[idy + 1].x;

		a1[idy].y += a1[idy + 32].y;
		a1[idy].y += a1[idy + 16].y;
		a1[idy].y += a1[idy + 8].y;
		a1[idy].y += a1[idy + 4].y;
		a1[idy].y += a1[idy + 2].y;
		a1[idy].y += a1[idy + 1].y;

		a1[idy].z += a1[idy + 32].z;
		a1[idy].z += a1[idy + 16].z;
		a1[idy].z += a1[idy + 8].z;
		a1[idy].z += a1[idy + 4].z;
		a1[idy].z += a1[idy + 2].z;
		a1[idy].z += a1[idy + 1].z;
	}
	else{
		if(idy < 64){
			a2[idy-32].x += a2[idy + 32-32].x;
			a2[idy-32].x += a2[idy + 16-32].x;
			a2[idy-32].x += a2[idy + 8-32].x;
			a2[idy-32].x += a2[idy + 4-32].x;
			a2[idy-32].x += a2[idy + 2-32].x;
			a2[idy-32].x += a2[idy + 1-32].x;

			a2[idy-32].y += a2[idy + 32-32].y;
			a2[idy-32].y += a2[idy + 16-32].y;
			a2[idy-32].y += a2[idy + 8-32].y;
			a2[idy-32].y += a2[idy + 4-32].y;
			a2[idy-32].y += a2[idy + 2-32].y;
			a2[idy-32].y += a2[idy + 1-32].y;

			a2[idy-32].z += a2[idy + 32-32].z;
			a2[idy-32].z += a2[idy + 16-32].z;
			a2[idy-32].z += a2[idy + 8-32].z;
			a2[idy-32].z += a2[idy + 4-32].z;
			a2[idy-32].z += a2[idy + 2-32].z;
			a2[idy-32].z += a2[idy + 1-32].z;
		}
		else{
			if(idy < 96){
				a3[idy-64].x += a3[idy + 32-64].x;
				a3[idy-64].x += a3[idy + 16-64].x;
				a3[idy-64].x += a3[idy + 8-64].x;
				a3[idy-64].x += a3[idy + 4-64].x;
				a3[idy-64].x += a3[idy + 2-64].x;
				a3[idy-64].x += a3[idy + 1-64].x;

				a3[idy-64].y += a3[idy + 32-64].y;
				a3[idy-64].y += a3[idy + 16-64].y;
				a3[idy-64].y += a3[idy + 8-64].y;
				a3[idy-64].y += a3[idy + 4-64].y;
				a3[idy-64].y += a3[idy + 2-64].y;
				a3[idy-64].y += a3[idy + 1-64].y;

				a3[idy-64].z += a3[idy + 32-64].z;
				a3[idy-64].z += a3[idy + 16-64].z;
				a3[idy-64].z += a3[idy + 8-64].z;
				a3[idy-64].z += a3[idy + 4-64].z;
				a3[idy-64].z += a3[idy + 2-64].z;
				a3[idy-64].z += a3[idy + 1-64].z;
			}
			else{
				if(idy < 128){
					a4[idy-96].x += a4[idy + 32-96].x;
					a4[idy-96].x += a4[idy + 16-96].x;
					a4[idy-96].x += a4[idy + 8-96].x;
					a4[idy-96].x += a4[idy + 4-96].x;
					a4[idy-96].x += a4[idy + 2-96].x;
					a4[idy-96].x += a4[idy + 1-96].x;

					a4[idy-96].y += a4[idy + 32-96].y;
					a4[idy-96].y += a4[idy + 16-96].y;
					a4[idy-96].y += a4[idy + 8-96].y;
					a4[idy-96].y += a4[idy + 4-96].y;
					a4[idy-96].y += a4[idy + 2-96].y;
					a4[idy-96].y += a4[idy + 1-96].y;

					a4[idy-96].z += a4[idy + 32-96].z;
					a4[idy-96].z += a4[idy + 16-96].z;
					a4[idy-96].z += a4[idy + 8-96].z;
					a4[idy-96].z += a4[idy + 4-96].z;
					a4[idy-96].z += a4[idy + 2-96].z;
					a4[idy-96].z += a4[idy + 1-96].z;
				}
			}
		}
	}

	__shared__ int Ne1;
	__shared__ int Ne2;
	__shared__ int Ne3;
	__shared__ int Ne4;

	__syncthreads();

	if(idy == 0){
		acck_d[idx].x = a1[0].x * dtksq;
		acck_d[idx].y = a1[0].y * dtksq;
		acck_d[idx].z = a1[0].z * dtksq;
		Ne1 = atomicAdd(Nencpairs_d, NencpairsI_s);
	}
	if(idy == 32){
		acck_d[idx + N4].x = a2[0].x * dtksq;
		acck_d[idx + N4].y = a2[0].y * dtksq;
		acck_d[idx + N4].z = a2[0].z * dtksq;
		Ne2 = atomicAdd(Nencpairs_d, NencpairsI2_s);
	}
	if(idy == 64){
		acck_d[idx + 2*N4].x = a3[0].x * dtksq;
		acck_d[idx + 2*N4].y = a3[0].y * dtksq;
		acck_d[idx + 2*N4].z = a3[0].z * dtksq;
		Ne3 = atomicAdd(Nencpairs_d, NencpairsI3_s);
	}
	if(idy == 96){
		acck_d[idx + 3*N4].x = a4[0].x * dtksq;
		acck_d[idx + 3*N4].y = a4[0].y * dtksq;
		acck_d[idx + 3*N4].z = a4[0].z * dtksq;
		Ne4 = atomicAdd(Nencpairs_d, NencpairsI4_s);
	}
	__syncthreads();
	for(int i = 0; i < NencpairsI_s; i += Bl){
		if(idy + i < NencpairsI_s){
			Encpairs_d[idy + i + Ne1] = Encpairs2_d[icNB * idx + idy + i];
		}
	}
	__syncthreads();
	for(int i = 0; i < NencpairsI2_s; i += Bl){
		if(idy + i < NencpairsI2_s){
			Encpairs_d[idy + i + Ne2] = Encpairs2_d[icNB * (idx + N4) + idy + i];
		}
	}
	__syncthreads();
	for(int i = 0; i < NencpairsI3_s; i += Bl){
		if(idy + i < NencpairsI3_s){
			Encpairs_d[idy + i + Ne3] = Encpairs2_d[icNB * (idx + 2*N4) + idy + i];
		}
	}
	__syncthreads();
	for(int i = 0; i < NencpairsI4_s; i += Bl){
		if(idy + i < NencpairsI4_s){
			Encpairs_d[idy + i + Ne4] = Encpairs2_d[icNB * (idx + 3*N4) + idy + i];
		}
	}
	__syncthreads();
	if(idy == 0){
		Encpairs2_d[icNB * idx].x = NencpairsI_s;
		Encpairs2_d[icNB * idx + 1].x = NencpairsJ_s;
	}
	if(idy == 32){
		Encpairs2_d[icNB * (idx + N4)].x = NencpairsI2_s;
		Encpairs2_d[icNB * (idx + N4) + 1].x = NencpairsJ2_s;
	}
	if(idy == 64){
		Encpairs2_d[icNB * (idx + 2*N4)].x = NencpairsI3_s;
		Encpairs2_d[icNB * (idx + 2*N4) + 1].x = NencpairsJ3_s;
	}
	if(idy == 96){
		Encpairs2_d[icNB * (idx + 3*N4)].x = NencpairsI4_s;
		Encpairs2_d[icNB * (idx + 3*N4) + 1].x = NencpairsJ4_s;
	}

}


//******************************************************
// This function calculates the force between body i and j
// it must be called n^2/2 times

//Author: Simon Grimm, Joachim Stadel
// January 2015
//********************************************************
__device__  void forceij(double4 x4i, double4 x4j, double4 &fi, double4 &fj, int *NencpairsI, int *NencpairsJ, int2 *Encpairs2_d, int j, int i, int icNB){

	double3 r3ij;
	int Ni, Nj;

	r3ij.x = x4j.x - x4i.x;
	r3ij.y = x4j.y - x4i.y;
	r3ij.z = x4j.z - x4i.z;

	double rsq = r3ij.x*r3ij.x + r3ij.y*r3ij.y + r3ij.z*r3ij.z;

	double rcritv = fmax(fi.w, fj.w);
	int cl = (rsq < pc * rcritv * rcritv) ? 1 : 0;
  
	if(cl && (x4i.w > 0.0 || x4j.w > 0.0)){  //prechecker
		Ni = atomicAdd(NencpairsI, 1);
		Nj = atomicAdd(NencpairsJ, 1);
		Encpairs2_d[icNB * i + Ni].y = j;
		Encpairs2_d[icNB * j + icNB - 1 - Nj].y = i;
	}
	
	double ir = 1.0 / sqrt(rsq);

	double ir3 = ir * ir * ir;
	double s;

	if(cl){
		s = 0.0;
	}
	else{
		s = x4j.w * x4i.w * ir3;
	}

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
__device__ void  accc(double3 &ac, double4 &x4i, double4 &x4j, double rcritvi, double rcritvj, int *NencpairsI, int *NencpairsJ, int2 *Encpairs2_d, int j, int i, int icNB){
	if( i != j && x4i.w >= 0.0 && x4j.w >= 0.0){
		double rsq, ir, ir3, s;
		double3 r3ij;
		double rcritv;
		int Ni, Nj;

		r3ij.x = x4j.x - x4i.x;
		r3ij.y = x4j.y - x4i.y;
		r3ij.z = x4j.z - x4i.z;

		rsq = r3ij.x*r3ij.x + r3ij.y*r3ij.y + r3ij.z*r3ij.z;
		rcritv = fmax(rcritvi, rcritvj);

		int cl = (rsq < pc * rcritv * rcritv) ? 1 : 0;

		if(cl && (x4i.w > 0.0 || x4j.w > 0.0)){  //prechecker
			if( i < j){
//printf("Precheck %d %d\n", i, j);
				Ni = atomicAdd(NencpairsI, 1);
				Nj = atomicAdd(NencpairsJ, 1);
				Encpairs2_d[icNB * i + Ni].y = j;
				Encpairs2_d[icNB * j + icNB - 1 - Nj].y = i;
			}
		}

		ir = 1.0/sqrt(rsq);
		ir3 = ir*ir*ir;

		if(cl){
			s = 0.0;
		}
		else{
			s = x4j.w * ir3;
		}

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
__global__ void ForceTri_kernel(double4 *x4_d, double3 *f_d, double *rcritv_d, int2 *Encpairs_d, int2 *Encpairs2_d, int icNB, int I, int II, int nb){

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
			forceij(x4i, x4_s[j], fi, fj_s[j], &Encpairs2_d[icNB * iii].x, &Encpairs2_d[icNB * jjj + 1].x, Encpairs2_d, jjj, iii, icNB);
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
__global__ void ForceDiag_kernel(double4 *x4_d, double3 *f_d, double *rcritv_d, int2 *Encpairs_d, int2 *Encpairs2_d, int icNB){

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
		accc(ai, x4i, x4_s[j], rcritvi, rcritv_s[j], &Encpairs2_d[icNB * iii].x, &Encpairs2_d[icNB * jjj + 1].x, Encpairs2_d, jjj, iii, icNB);
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
__global__ void ForceSq_kernel(double4 *x4_d, double3 *f_d, double *rcritv_d, int2 *Encpairs_d, int2 *Encpairs2_d, int icNB, int I, int nb){

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
			forceij(x4i, x4_s[j], fi, fj_s[j], &Encpairs2_d[icNB * iii].x, &Encpairs2_d[icNB * jjj + 1].x, Encpairs2_d, jjj, iii, icNB);
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

__global__ void EncpairsZero(int2 *Encpairs2_d, double3 *a_d, int icNB){

	int id = threadIdx.x + blockIdx.x * blockDim.x;


	Encpairs2_d[icNB * id].x = 0;
	Encpairs2_d[icNB * id + 1].x = 0;

	a_d[id].x = 0.0;
	a_d[id].y = 0.0;
	a_d[id].z = 0.0;

}


__global__ void acclargeN_kernel(double4 *x4_d, double3 *f_d, int *Nencpairs_d, int2 *Encpairs_d, int2 *Encpairs2_d, double dtksq, int icNB, int N){

	int id = threadIdx.x + blockIdx.x * blockDim.x;

	if(id < N){

		double im = 1.0 / x4_d[id].w;

		f_d[id].x *= im * dtksq;
		f_d[id].y *= im * dtksq;
		f_d[id].z *= im * dtksq;

		int NI = Encpairs2_d[id * icNB].x;
		int Ne = atomicAdd(Nencpairs_d, NI);

		for(int i = 0; i < NI; ++i){
			Encpairs_d[Ne + i].x = id;
			Encpairs_d[Ne + i].y = Encpairs2_d[icNB * id + i].y;
		}

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
__host__ void ForceDriver(double4 *x4_d, double *rcritv_d, double3 *f_d, int *Nencpairs_d, int2 *Encpairs_d, int2 *Encpairs2_d, double dtksq, int icNB, int NB, int N){

	const int p = 256;
	const int nb = NB / (2 * p);

	//set NencpairsI and NencpairsJ to zero
	EncpairsZero <<< (NB + p - 1) / p, p >>> (Encpairs2_d, f_d, icNB);
	//Blocks on the Diagonal
	ForceDiag_kernel < p > <<< NB / p, p>>> (x4_d, f_d, rcritv_d, Encpairs_d, Encpairs2_d, icNB);

	//Combine upper left quarter triangle with lower right quarter triangle
	for(int ii = 1; ii < nb; ii *= 2){
                for(int k = 0; k < ii; ++k){
			int i = ii + k;
			ForceTri_kernel < p > <<< nb, p>>> (x4_d, f_d, rcritv_d, Encpairs_d, Encpairs2_d, icNB, i, ii, nb);
		}
	}

	//Lower left quarter
	for(int i = 0; i < nb; ++i){
		ForceSq_kernel < p > <<< nb, p >>> (x4_d, f_d, rcritv_d, Encpairs_d, Encpairs2_d, icNB, i, nb);
	}

	acclargeN_kernel <<< (N + p - 1) / p, p >>> (x4_d, f_d, Nencpairs_d, Encpairs_d, Encpairs2_d, dtksq, icNB, N);

}




#endif

#ifndef KICK4_H
#define KICK4_H
#include "define.h"

//**************************************
//This function computes the terms a = mi/rij^3 * Kij.
//This function also finds the pairs of bodies which are separated less than pc * rcritv^2. The index of those 
//pairs are stored in the boolean matrix Encpairsb_d. This indexes are then used
//in the EncMatrix_kernel.

//Authors: Simon Grimm
//November 2016
//****************************************
__device__ void  acc_c(double3 &ac, double4 &x4i, double4 &x4j, double rcritvi, double rcritvj, bool *Encpairsb_d, int j, int i, int NconstT, int nn){
	volatile double rsq, ir, ir3, s;
	double3 r3ij;
	double rcritv;

	//ignore ghost particles
	bool bm = (x4i.w >= 0.0 && x4j.w >= 0.0) ? true : false;

	r3ij.x = x4j.x - x4i.x;
	r3ij.y = x4j.y - x4i.y;
	r3ij.z = x4j.z - x4i.z;

	rsq = (r3ij.x*r3ij.x) + (r3ij.y*r3ij.y) + (r3ij.z*r3ij.z);
	rcritv = fmax(rcritvi, rcritvj);
	bool cl = (rsq < def_pc * rcritv * rcritv && (x4i.w > 0.0 || x4j.w > 0.0)) ? true : false;
	long long int clij = (long long int)(NconstT) * (long long int)(i - nn) + j;
//if (cl && i == 319) printf("cl %d %d %d %d %d %lld %g %g %g %g\n", nn, i, i - nn, j, NconstT, clij, x4i.x, x4j.x, x4i.w, x4j.w);
//if (cl && i != j) printf("cl %d %d %d %d %d %lld %g %g %g %g\n", nn, i, i - nn, j, NconstT, clij, x4i.x, x4j.x, x4i.w, x4j.w);
	Encpairsb_d[clij] = cl;

	ir = 1.0/sqrt(rsq);
	ir3 = ir*ir*ir;

	s = (x4j.w * ir3) * (!cl) * bm * (i != j);

	ac.x += __dmul_rn(r3ij.x, s);
	ac.y += __dmul_rn(r3ij.y, s);
	ac.z += __dmul_rn(r3ij.z, s);
//printf("%d %d %.20g %.20g %.20g\n", i, j, __dmul_rn(r3ij.x, s), __dmul_rn(r3ij.y, s), __dmul_rn(r3ij.z, s));
}

// **************************************
//This kernel performs the second kick of the time step, in the case NB = 128. NB is the next bigger number of N
//which is a power of two.
//It performs a precheck for close encouter candidates and mark them in a boolean matrix. 
//It calculates the acceleration between all bodies with are not in a close encounter.
//
//The Kernel is launched with N/4 blocks a 128 theads.
//
//Authors: Simon Grimm
//November 2015
//
// ****************************************
template <int Bl>
__global__ void acc128b_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double *rcrit_d, double *rcritv_d, int *groupIndex_d, bool *Encpairsb_d, int2 *Encpairs2_d, double *test_d, int N, int N2, int NconstT, int NencMax, double t){

	int idy = threadIdx.x;
	int idx = blockIdx.x;

	__shared__ double3 a1_s[Bl];
	__shared__ double3 a2_s[Bl];

	double4 x4i, x4i2;
	double rcritvi, rcritvi2;

	if(idx < N){
		x4i = x4_d[idx];
		rcritvi = rcritv_d[idx];
	}
	if(idx + N2 < N){
		x4i2 = x4_d[idx + N2];
		rcritvi2 = rcritv_d[idx + N2];
	}
	double4 x4j = x4_d[idy];

	double rcritvj = rcritv_d[idy];
#if G3 > 0
	int groupIndexj = groupIndex_d[idy];
#endif

#if G3 > 0
	int groupIndexi, groupIndexi2;
	if(idx < N) groupIndexi = groupIndex_d[idx];
	if(idx  + N2 < N) groupIndexi2 = groupIndex_d[idx + N2];
#endif

	a1_s[idy].x = 0.0;
	a1_s[idy].y = 0.0;
	a1_s[idy].z = 0.0;
	
	a2_s[idy].x = 0.0;
	a2_s[idy].y = 0.0;
	a2_s[idy].z = 0.0;

	__syncthreads();
	if(idy < N){
		if(idx < N)      acc_c(a1_s[idy], x4i, x4j, rcritvi, rcritvj, Encpairsb_d, idy, idx, NconstT, 0);
		if(idx + N2 < N) acc_c(a2_s[idy], x4i2, x4j, rcritvi2, rcritvj, Encpairsb_d, idy, idx + N2, NconstT, 0);
	}
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

	__syncthreads();

	if(idy == 0 && idx < N){
		acck_d[idx].x = a1[0].x;
		acck_d[idx].y = a1[0].y;
		acck_d[idx].z = a1[0].z;
		Encpairs2_d[idx * NencMax].x = 0; //NI
	}

	if(idy == 32 && idx + N2 < N){
		acck_d[idx + N2].x = a2[0].x;
		acck_d[idx + N2].y = a2[0].y;
		acck_d[idx + N2].z = a2[0].z;
		Encpairs2_d[(idx + N2) * NencMax].x = 0; //NI
	}
}

// **************************************
//This kernel performs the second kick of the time step, in the case NB = 256. NB is the next bigger number of N
//which is a power of two.
//It performs a precheck for close encouter candidates and mark them in a boolean matrix. 
//It calculates the acceleration between all bodies with are not in a close encounter.
//
//The Kernel is launched with N/4 blocks a 128 theads.
//
//Authors: Simon Grimm
//November 2015
// ****************************************
template <int Bl>
__global__ void acc256b_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double *rcrit_d, double *rcritv_d, int *groupIndex_d, int N4, bool *Encpairsb_d, int2 *Encpairs2_d, double *test_d, int N, int NconstT, int NencMax, double t){
	int idy = threadIdx.x;
	int idx = blockIdx.x;

	__shared__ double3 a1_s[Bl];
	__shared__ double3 a2_s[Bl];
	__shared__ double3 a3_s[Bl];
	__shared__ double3 a4_s[Bl];

	double4 x4i, x4i2, x4i3, x4i4;
	double rcritvi, rcritvi2, rcritvi3, rcritvi4;

	if(idx < N){
		x4i = x4_d[idx];
		rcritvi = rcritv_d[idx];
	}
	if(idx + N4 < N){
		x4i2 = x4_d[idx+N4];
		rcritvi2 = rcritv_d[idx+N4];
	}
	if(idx + 2*N4 < N){
		x4i3 = x4_d[idx+2*N4];
		rcritvi3 = rcritv_d[idx+2*N4];
	}
	if(idx + 3*N4 < N){
		x4i4 = x4_d[idx+3*N4];
		rcritvi4 = rcritv_d[idx+3*N4];
	}
#if G3 > 0
	int groupIndexi, groupIndexi2, groupIndexi3, groupIndexi4;

	if(idx < N) groupIndexi = groupIndex_d[idx];
	if(idx + N4< N) groupIndexi2 = groupIndex_d[idx + N4];
	if(idx + 2*N4< N) groupIndexi3 = groupIndex_d[idx + 2*N4];
	if(idx + 3*N4< N) groupIndexi4 = groupIndex_d[idx + 3*N4];
#endif

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

	for(int i = 0; i < N; i += Bl){ 
		if(idy + i < N){
			double4 x4j = x4_d[idy + i];
			double rcritvj = rcritv_d[idy + i];
			if(idx < N)        acc_c(a1_s[idy], x4i, x4j, rcritvi, rcritvj, Encpairsb_d, idy + i, idx, NconstT, 0);
			if(idx + N4 < N)   acc_c(a2_s[idy], x4i2, x4j, rcritvi2, rcritvj, Encpairsb_d, idy + i, idx + N4, NconstT, 0);
			if(idx + 2*N4 < N) acc_c(a3_s[idy], x4i3, x4j, rcritvi3, rcritvj, Encpairsb_d, idy + i, idx + 2*N4, NconstT, 0);
			if(idx + 3*N4 < N) acc_c(a4_s[idy], x4i4, x4j, rcritvi4, rcritvj, Encpairsb_d, idy + i, idx + 3*N4, NconstT, 0);
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
	__syncthreads();

	if(idy == 0 && idx < N){
		acck_d[idx].x = a1[0].x;
		acck_d[idx].y = a1[0].y;
		acck_d[idx].z = a1[0].z;
		Encpairs2_d[idx * NencMax].x = 0; //NI
	}

	if(idy == 32 && idx + N4 < N){
		acck_d[idx + N4].x = a2[0].x;
		acck_d[idx + N4].y = a2[0].y;
		acck_d[idx + N4].z = a2[0].z;
		Encpairs2_d[(idx + N4) * NencMax].x = 0; //NI
	}

	if(idy == 64 && idx + 2*N4 < N){
		acck_d[idx + 2*N4].x = a3[0].x;
		acck_d[idx + 2*N4].y = a3[0].y;
		acck_d[idx + 2*N4].z = a3[0].z;
		Encpairs2_d[(idx + 2*N4) * NencMax].x = 0; //NI
	}
	
	if(idy == 96 && idx + 3*N4 < N){
		acck_d[idx + 3*N4].x = a4[0].x;
		acck_d[idx + 3*N4].y = a4[0].y;
		acck_d[idx + 3*N4].z = a4[0].z;
		Encpairs2_d[(idx + 3 *N4) * NencMax].x = 0; //NI
	}
}

// **************************************
//This kernel performs the second kick of the time step, in the case NB > 256. NB is the next bigger number of N
//which is a power of two.
//It performs a precheck for close encouter candidates and mark them in a boolean matrix. 
//It calculates the acceleration between all bodies with are not in a close encounter.
//
//The Kernel is launched with N/4 blocks a 128 theads.
//
//Authors: Simon Grimm
//November 2015
// ****************************************
template <int Bl>
__global__ void acc4b_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double *rcrit_d, double *rcritv_d, int *groupIndex_d, int N4, bool *Encpairsb_d, int2 *Encpairs2_d, double *test_d, int N, int NconstT, int NencMax, double t){
	int idy = threadIdx.x;
	int idx = blockIdx.x;

	int Bl_2 = Bl/2;

	__shared__ double3 a1_s[Bl/2];
	__shared__ double3 a2_s[Bl/2];
	__shared__ double3 a3_s[Bl/2];
	__shared__ double3 a4_s[Bl/2];

	double4 x4i, x4i2, x4i3, x4i4;
	double rcritvi, rcritvi2, rcritvi3, rcritvi4;

	if(idx < N){
		x4i = x4_d[idx];
		rcritvi = rcritv_d[idx];
	}
	if(idx + N4 < N){
		x4i2 = x4_d[idx+N4];
		rcritvi2 = rcritv_d[idx+N4];
	}
	if(idx + 2*N4 < N){
		x4i3 = x4_d[idx+2*N4];
		rcritvi3 = rcritv_d[idx+2*N4];
	}
	if(idx + 3*N4 < N){
		x4i4 = x4_d[idx+3*N4];
		rcritvi4 = rcritv_d[idx+3*N4];
	}
#if G3 > 0
	int groupIndexi, groupIndexi2, groupIndexi3, groupIndexi4;

	if(idx < N) groupIndexi = groupIndex_d[idx];
	if(idx + N4< N) groupIndexi2 = groupIndex_d[idx + N4];
	if(idx + 2*N4< N) groupIndexi3 = groupIndex_d[idx + 2*N4];
	if(idx + 3*N4< N) groupIndexi4 = groupIndex_d[idx + 3*N4];
#endif

	if(idy < Bl_2) {
		a1_s[idy].x = 0.0;
		a1_s[idy].y = 0.0;
		a1_s[idy].z = 0.0;

		a3_s[idy].x = 0.0;
		a3_s[idy].y = 0.0;
		a3_s[idy].z = 0.0;

		__syncthreads();
		for(int i = 0; i < N; i += Bl_2){
			if(idy + i < N){
				double4 x4j = x4_d[idy + i];
				double rcritvj = rcritv_d[idy + i];
				if(idx < N)        acc_c(a1_s[idy], x4i, x4j, rcritvi, rcritvj, Encpairsb_d, idy + i, idx, NconstT, 0);
				if(idx + 2*N4 < N) acc_c(a3_s[idy], x4i3, x4j, rcritvi3, rcritvj, Encpairsb_d, idy + i, idx +2*N4, NconstT, 0);
//if(idx + N4 == 1045 && idy + i == 0) printf("acc1 %d %d %.20g %.20g %.20g\n", idx + N4, idy + i, a1_s[idy].x, x4i4.x, x4j.x);
//if(idx + 2*N4 == 1045 && idy + i == 0) printf("acc2 %d %d %.20g %.20g %.20g\n", idx + 2*N4, idy + i, a2_s[idy].x, x4i4.x, x4j.x);
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
		for(int i = 0; i < N; i += Bl_2){
			if(idy-Bl_2 + i < N){
				double4 x4j = x4_d[idy-Bl_2 + i];
				double rcritvj = rcritv_d[idy-Bl_2 + i];
				if(idx + N4 < N)   acc_c(a2_s[idy-Bl_2], x4i2, x4j, rcritvi2, rcritvj, Encpairsb_d, idy-Bl_2 + i, idx +N4, NconstT, 0);
				if(idx + 3*N4 < N) acc_c(a4_s[idy-Bl_2], x4i4, x4j, rcritvi4, rcritvj, Encpairsb_d, idy-Bl_2 + i, idx +3*N4, NconstT, 0);
//if(idx + N4 == 1045 && idy-Bl_2 + i == 0) printf("acc2 %d %d %.20g %.20g %.20g\n", idx + N4, idy-Bl_2 + i, a2_s[idy-Bl_2].x, x4i2.x, x4j.x);
//if(idx + 3*N4 == 1045 && idy-Bl_2 + i == 0) printf("acc4 %d %d %.20g %.20g %.20g\n", idx + 3*N4, idy-Bl_2 + i, a4_s[idy-Bl_2].x, x4i4.x, x4j.x);

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

	__syncthreads();

	if(idy == 0 && idx < N){
		acck_d[idx].x = a1[0].x;
		acck_d[idx].y = a1[0].y;
		acck_d[idx].z = a1[0].z;
		Encpairs2_d[idx * NencMax].x = 0; //NI
	}
	if(idy == 32 && idx + N4 < N){
		acck_d[idx + N4].x = a2[0].x;
		acck_d[idx + N4].y = a2[0].y;
		acck_d[idx + N4].z = a2[0].z;
		Encpairs2_d[(idx + N4) * NencMax].x = 0; //NI
	}
	if(idy == 64 && idx + 2*N4 < N){
		acck_d[idx + 2*N4].x = a3[0].x;
		acck_d[idx + 2*N4].y = a3[0].y;
		acck_d[idx + 2*N4].z = a3[0].z;
		Encpairs2_d[(idx + 2*N4) * NencMax].x = 0; //NI
	}
	if(idy == 96 && idx + 3*N4 < N){
		acck_d[idx + 3*N4].x = a4[0].x;
		acck_d[idx + 3*N4].y = a4[0].y;
		acck_d[idx + 3*N4].z = a4[0].z;
		Encpairs2_d[(idx + 3*N4) * NencMax].x = 0; //NI
	}

}

//******************************************************
//This kernel reads the boolean encounter matrix and creates lists
//of encounter pairs for the Kick32Ab kernel and the encounter kernel
//Author: Simon Grimm
//November 2016
//********************************************************
__global__ void EncMatrix_kernel(bool *Encpairsb_d, int2 *Encpairs_d, int2 *Encpairs2_d, int *Nencpairs_d, int NconstT, int NencMax, int Nx, int Ny, int nn, int *EncFlag_d){

	int j = threadIdx.y;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int ii = i + nn;

	for(int jj = 0; jj < Nx; jj += blockDim.y){ 
		if(i < Ny && j + jj < Nx){
			long long int clij = (long long int)(NconstT) * (long long int)(i) + (long long int)(jj + j);
			bool cl = Encpairsb_d[clij];
			if(cl){
				if(ii != jj + j){
					int Ni = atomicAdd(&Encpairs2_d[ii * NencMax].x, 1);
//printf("enc1 %d %d %d\n", ii, jj + j, Ni);
					if(Ni >= NencMax) atomicMax(&EncFlag_d[0], Ni);

					Encpairs2_d[NencMax * ii + Ni].y = jj + j;
				}
				if(ii < jj + j){
					int Ne = atomicAdd(Nencpairs_d, 1);
					Encpairs_d[Ne].x = ii;
					Encpairs_d[Ne].y = jj + j;
				}
			}
		}
	}
}
template < int E >
__global__ void EncMatrixsmall_kernel(bool *Encpairsb_d, int2 *Encpairs_d, int2 *Encpairs2_d, int *Nencpairs_d, int NconstT, int NencMax, int Nx, int Ny, int nn, int *EncFlag_d){

	int j = threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int ii = i + nn;

	for(int jj = 0; jj < Nx; jj += blockDim.x){ 
		if(i < Ny && j + jj < Nx){
			long long int clij = (long long int)(NconstT) * (long long int)(i) + (long long int)(jj + j);
			bool cl = Encpairsb_d[clij];
			if(cl){
				if(ii != jj + j){
					if(E == 1){
						int Ni = atomicAdd(&Encpairs2_d[ii * NencMax].x, 1);
//if(ii == 319) printf("enc1 %d %d %d\n", ii, jj + j, Ni);
						if(Ni >= NencMax) atomicMax(&EncFlag_d[0], Ni);

						Encpairs2_d[NencMax * ii + Ni].y = jj + j;
					}
					if(E == 2/* && ii >= Nx*/){
						int Ni = atomicAdd(&Encpairs2_d[(jj + j) * NencMax].x, 1);
//if(ii == 319 && jj + j == 23133) printf("enc2 %d %d %d\n", jj + j, ii, Ni);
						if(Ni >= NencMax) atomicMax(&EncFlag_d[0], Ni);

						Encpairs2_d[NencMax * (jj + j) + Ni].y = ii;
					}
				}
				if(E == 1 && ii > jj + j){
//if(jj + j == 319) printf("encp %d %d\n", jj + j, ii);
					int Ne = atomicAdd(Nencpairs_d, 1);
					Encpairs_d[Ne].x = ii;
					Encpairs_d[Ne].y = jj + j;
				}
			}
		}
	}
}
// **************************************
//This kernel performs the second kick of the time step, in the test particle mode. NB is the next bigger number of N
//which is a power of two.
//It performs a precheck for close encouter candidates and mark them in a boolean matrix. 
//It calculates the acceleration between all bodies with are not in a close encounter.
//
//E = 1: used for normal test particles
//E = 2: used for semi test particles
//
//Author: Simon Grimm
//August 2016
// ****************************************
template <int Bl, int E>
__global__ void accsmall_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double *rcrit_d, double *rcritv_d, int *groupIndex_d, bool *Encpairsb_d, int2 *Encpairs2_d, int Nx, int Ny, int nn, int NconstT, int NencMax, double t){

	int idy = threadIdx.x;
	int itx = threadIdx.y;
	int idx = blockIdx.y * blockDim.y + itx;
	if(E == 1){
		idx += nn;
		Nx += nn;
	}
	__shared__ double3 a1_s[Bl];


	int N2a, N2b;
	if(E == 1){
		N2a = 0;
		N2b = Ny;
	}
	if(E == 2){
		N2a = nn;
		N2b = nn + Ny;
	}

	if(idx < Nx){
		double4 x4i = x4_d[idx];

		double rcritvi = rcritv_d[idx];
#if G3 > 0
		int groupIndexi = groupIndex_d[idx];
#endif

		a1_s[idy + blockDim.x * itx].x = 0.0;
		a1_s[idy + blockDim.x * itx].y = 0.0;
		a1_s[idy + blockDim.x * itx].z = 0.0;

		__syncthreads();
		for(int i = N2a; i < N2b; i += blockDim.x){
			if(idy + i < N2b){
				if(E == 1) acc_c(a1_s[idy + blockDim.x * itx], x4i, x4_d[idy + i], rcritvi, rcritv_d[idy + i], Encpairsb_d, idy + i, idx, NconstT, nn);
				if(E == 2) acc_c(a1_s[idy + blockDim.x * itx], x4i, x4_d[idy + i], rcritvi, rcritv_d[idy + i], Encpairsb_d, idx, idy + i, NconstT, nn);

			}	
		}
		__syncthreads();

		volatile double3 *a1 = a1_s;
		int s = blockDim.x/2;

		for(int i = 6; i < log2f(blockDim.x); ++i){
			if( idy < s ) {
				a1[idy + blockDim.x * itx].x += a1[(idy + s) + blockDim.x * itx].x;
				a1[idy + blockDim.x * itx].y += a1[(idy + s) + blockDim.x * itx].y;
				a1[idy + blockDim.x * itx].z += a1[(idy + s) + blockDim.x * itx].z;
			}
			__syncthreads();
			s /= 2;
		}

		if(idy < 32){
			if(blockDim.x >= 64 && idy + 32 < blockDim.x) a1[idy + blockDim.x * itx].x += a1[(idy + 32) + blockDim.x * itx].x;
			if(blockDim.x >= 32 && idy + 16 < blockDim.x) a1[idy + blockDim.x * itx].x += a1[(idy + 16) + blockDim.x * itx].x;
			if(blockDim.x >= 16 && idy + 8 < blockDim.x) a1[idy + blockDim.x * itx].x += a1[(idy + 8) + blockDim.x * itx].x;
			if(blockDim.x >= 8 && idy + 4 < blockDim.x) a1[idy + blockDim.x * itx].x += a1[(idy + 4) + blockDim.x * itx].x;
			if(blockDim.x >= 4 && idy + 2 < blockDim.x) a1[idy + blockDim.x * itx].x += a1[(idy + 2) + blockDim.x * itx].x;
			if(blockDim.x >= 2 && idy + 1 < blockDim.x) a1[idy + blockDim.x * itx].x += a1[(idy + 1) + blockDim.x * itx].x;

			if(blockDim.x >= 64 && idy + 32 < blockDim.x) a1[idy + blockDim.x * itx].y += a1[(idy + 32) + blockDim.x * itx].y;
			if(blockDim.x >= 32 && idy + 16 < blockDim.x) a1[idy + blockDim.x * itx].y += a1[(idy + 16) + blockDim.x * itx].y;
			if(blockDim.x >= 16 && idy + 8 < blockDim.x) a1[idy + blockDim.x * itx].y += a1[(idy + 8) + blockDim.x * itx].y;
			if(blockDim.x >= 8 && idy + 4 < blockDim.x) a1[idy + blockDim.x * itx].y += a1[(idy + 4) + blockDim.x * itx].y;
			if(blockDim.x >= 4 && idy + 2 < blockDim.x) a1[idy + blockDim.x * itx].y += a1[(idy + 2) + blockDim.x * itx].y;
			if(blockDim.x >= 2 && idy + 1 < blockDim.x) a1[idy + blockDim.x * itx].y += a1[(idy + 1) + blockDim.x * itx].y;

			if(blockDim.x >= 64 && idy + 32 < blockDim.x) a1[idy + blockDim.x * itx].z += a1[(idy + 32) + blockDim.x * itx].z;
			if(blockDim.x >= 32 && idy + 16 < blockDim.x) a1[idy + blockDim.x * itx].z += a1[(idy + 16) + blockDim.x * itx].z;
			if(blockDim.x >= 16 && idy + 8 < blockDim.x) a1[idy + blockDim.x * itx].z += a1[(idy + 8) + blockDim.x * itx].z;
			if(blockDim.x >= 8 && idy + 4 < blockDim.x) a1[idy + blockDim.x * itx].z += a1[(idy + 4) + blockDim.x * itx].z;
			if(blockDim.x >= 4 && idy + 2 < blockDim.x) a1[idy + blockDim.x * itx].z += a1[(idy + 2) + blockDim.x * itx].z;
			if(blockDim.x >= 2 && idy + 1 < blockDim.x) a1[idy + blockDim.x * itx].z += a1[(idy + 1) + blockDim.x * itx].z;
		}

		__syncthreads();

		if(idy == 0){
			if(E == 1){
				acck_d[idx].x = a1[0 + blockDim.x * itx].x;
				acck_d[idx].y = a1[0 + blockDim.x * itx].y;
				acck_d[idx].z = a1[0 + blockDim.x * itx].z;
//printf("a %d %d %.20g %.20g %.20g\n", E, idx, a1[0 + blockDim.x * itx].x, a1[0 + blockDim.x * itx].y, a1[0 + blockDim.x * itx].z);
			}
			if(E == 2){
				acck_d[idx].x += a1[0 + blockDim.x * itx].x;
				acck_d[idx].y += a1[0 + blockDim.x * itx].y;
				acck_d[idx].z += a1[0 + blockDim.x * itx].z;
//printf("a %d %d %.20g %.20g %.20g\n", E, idx, a1[0 + blockDim.x * itx].x, a1[0 + blockDim.x * itx].y, a1[0 + blockDim.x * itx].z);
			}
//printf("aa %d %d %.20g %.20g %.20g\n", E, idx, acck_d[idx].x, acck_d[idx].y, acck_d[idx].z);
			if(E == 1) Encpairs2_d[idx * NencMax].x = 0; //NI
		}
	}
}

//******************************************************
// This function calculates the force between body i and j
// it must be called n^2/2 times

//Author: Simon Grimm, Joachim Stadel
// January 2015
//********************************************************
__device__  void forceij(double4 x4i, double4 x4j, double4 &fi, double4 &fj, bool *Encpairsb_d, int j, int i, int NconstT){

	double3 r3ij;

	r3ij.x = x4j.x - x4i.x;
	r3ij.y = x4j.y - x4i.y;
	r3ij.z = x4j.z - x4i.z;

	double rsq = r3ij.x*r3ij.x + r3ij.y*r3ij.y + r3ij.z*r3ij.z;

	double rcritv = fmax(fi.w, fj.w);
	bool cl = (rsq < def_pc * rcritv * rcritv && (x4i.w > 0.0 || x4j.w > 0.0)) ? true : false;
	Encpairsb_d[NconstT * i + j] = cl; 
	
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


__device__ void  accc(double3 &ac, double4 &x4i, double4 &x4j, double rcritvi, double rcritvj, bool *Encpairsb_d, int j, int i, int NconstT){
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
		Encpairsb_d[NconstT * i + j] = cl;


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
__global__ void ForceTri_kernel(double4 *x4_d, double3 *f_d, double *rcritv_d, bool *Encpairsb_d, int NconstT, int I, int II, int nb){

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
			forceij(x4i, x4_s[j], fi, fj_s[j], Encpairsb_d, jjj, iii, NconstT);
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
__global__ void ForceDiag_kernel(double4 *x4_d, double3 *f_d, double *rcritv_d, bool *Encpairsb_d, int NconstT){

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
		accc(ai, x4i, x4_s[j], rcritvi, rcritv_s[j], Encpairsb_d, jjj, iii, NconstT);
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
__global__ void ForceSq_kernel(double4 *x4_d, double3 *f_d, double *rcritv_d, bool *Encpairsb_d, int NconstT, int I, int nb){

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
			forceij(x4i, x4_s[j], fi, fj_s[j], Encpairsb_d, jjj, iii, NconstT);
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

__global__ void EncpairsZero(int2 *Encpairs2_d, double3 *a_d, int NencMax){

	int id = threadIdx.x + blockIdx.x * blockDim.x;


	Encpairs2_d[NencMax * id].x = 0;
	Encpairs2_d[NencMax * id + 1].x = 0;

	a_d[id].x = 0.0;
	a_d[id].y = 0.0;
	a_d[id].z = 0.0;

}


__global__ void acclargeN_kernel(double4 *x4_d, double3 *f_d, double dtksq, int N){

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
__host__ void ForceDriver(double4 *x4_d, double *rcritv_d, double3 *f_d, bool *Encpairsb_d, int2 *Encpairs2_d, double dtksq, int NconstT, int NencMax, int NB, int N){

	const int p = 256;
	const int nb = NB / (2 * p);

	//set NencpairsI and NencpairsJ to zero
	EncpairsZero <<< (NB + p - 1) / p, p >>> (Encpairs2_d, f_d, NencMax);
	//Blocks on the Diagonal
	ForceDiag_kernel < p > <<< NB / p, p>>> (x4_d, f_d, rcritv_d, Encpairsb_d, NconstT);

	//Combine upper left quarter triangle with lower right quarter triangle
	for(int ii = 1; ii < nb; ii *= 2){
		for(int k = 0; k < ii; ++k){
			int i = ii + k;
			ForceTri_kernel < p > <<< nb, p>>> (x4_d, f_d, rcritv_d, Encpairsb_d, NconstT, i, ii, nb);
		}
	}

	//Lower left quarter
	for(int i = 0; i < nb; ++i){
		ForceSq_kernel < p > <<< nb, p >>> (x4_d, f_d, rcritv_d, Encpairsb_d, NconstT, i, nb);
	}

	acclargeN_kernel <<< (N + p - 1) / p, p >>> (x4_d, f_d, dtksq, N);

}




#endif

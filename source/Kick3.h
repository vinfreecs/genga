#ifndef KICK_H
#define KICK_H
#include "define.h"

// **************************************
// This function computes the term a = mi/rij^3 * Kij
// ****************************************
__device__ void  accA(double3 &ac, double4 &x4i, double4 &x4j, double rcritvi, double rcritvj, int j, int i){
	if( i != j && x4i.w >= 0.0 && x4j.w > 0.0){
		double rsq, ir, ir3, s;
		double3 r3ij;
		double rcritv, rcritv2;
		double y, yy;

		r3ij.x = x4j.x - x4i.x;
		r3ij.y = x4j.y - x4i.y;
		r3ij.z = x4j.z - x4i.z;

		rsq = r3ij.x*r3ij.x + r3ij.y*r3ij.y + r3ij.z*r3ij.z;
		rcritv = fmax(rcritvi, rcritvj);
		rcritv2 = rcritv * rcritv;

		ir = 1.0/sqrt(rsq);
		ir3 = ir*ir*ir;

		if(rsq >= 1.0 * rcritv2){
			s = x4j.w * ir3;
		}
		else{
			if(rsq <= 0.01 * rcritv2){
				s = 0.0;
			}
			else{
				y = (rsq * ir - 0.1 * rcritv)/(0.9*rcritv);
				yy = y * y;
				s = ir3 * yy / (2.0*yy - 2.0*y + 1.0) * x4j.w;
			}
		}
		ac.x += __dmul_rn(r3ij.x, s);
		ac.y += __dmul_rn(r3ij.y, s);
		ac.z += __dmul_rn(r3ij.z, s);
	}
}

//**************************************
//This function computes the terms a = mi/rij^3 * Kij and b = mi/rij.
//This function also finds the pairs of bodies which are separated less than pc * rcritv^2. The index of those 
//pairs are stored in the array Encpairs_d in two different ways. This indexes are then used
//in the KickA32 kernel and in the Encounter kernel.
//
//E = 0: a + b + precheck (initial step)
//E = 1: a + b + precheck
//E = 2: a + precheck
//E = 20: a + b + precheck. (initial step) used for Test particles in Test Particle Mode
//E = 21: a + b + precheck. used for Test particles in Test Particle Mode
//E = 22: a + precheck. used for Test particles in Test Particle Mode
//E = 10: a + b + precheck. (initial step) used for massiv bodies in Test Particle Mode
//E = 11: a + b + precheck. used for massiv bodies in Test Particle Mode
//E = 12: a + precheck. used for massiv bodies in Test Particle Mode

//Authors: Simon Grimm, Joachim Stadel
//March 2014
//****************************************
template < int E >
__device__ void  acc(double3 &ac, double3 &b, double4 &x4i, double4 &x4j, double rcriti, double rcritvi, double rcritj, double rcritvj, int *NencpairsI, int *NencpairsJ, int2 *Encpairs_d, int j, int i, int NencMax, double &test){
	if( i != j && x4i.w >= 0.0 && x4j.w >= 0.0){
		double rsq, ir, ir3, s, sb;
		double3 r3ij;
		double rcritv, rcritv2;
//		double rcrit, rcrit2;
		double y, yy;
		int Ni, Nj;

		r3ij.x = x4j.x - x4i.x;
		r3ij.y = x4j.y - x4i.y;
		r3ij.z = x4j.z - x4i.z;

		rsq = r3ij.x*r3ij.x + r3ij.y*r3ij.y + r3ij.z*r3ij.z;
//		rcrit = fmax(rcriti, rcritj);
		rcritv = fmax(rcritvi, rcritvj);

//		rcrit2 = rcrit * rcrit;
		rcritv2 = rcritv * rcritv;
		if(E <= 2){	
			if(rsq < def_pc * rcritv2 && (x4i.w > 0.0 || x4j.w > 0.0)){  //prechecker
//printf("Precheck %d %d\n", i, j);
				if( i < j){
					Ni = atomicAdd(NencpairsI, 1);
					Encpairs_d[NencMax * i + Ni].x = i;
					Encpairs_d[NencMax * i + Ni].y = j;
				}
				else{
					Nj = atomicAdd(NencpairsJ, 1);
					Encpairs_d[NencMax * i + NencMax - 1 - Nj].y = j;

				}
			}
		}
		if(E <= 22 && E >= 20){ // prechecker used for Test Particle Mode
			if(rsq < def_pc * rcritv2){
//printf("Precheck 20 %d %d %d %d\n", i, j, NencMax, *NencpairsJ);
				Encpairs_d[NencMax * i + *NencpairsJ].x = j;
				Encpairs_d[NencMax * i + *NencpairsJ].y = i;
				Encpairs_d[NencMax * i + NencMax - 1 - *NencpairsJ].y = j;
				*NencpairsJ += 1;
			}
		}
		if(E <= 12 && E >=10){ //prechecker used for Test Particle Mode
			if(rsq < def_pc * rcritv2){
				if(i < j){
//printf("Precheck 10 %d %d %d %d\n", i, j, NencMax, *NencpairsI);
					Encpairs_d[NencMax * i + *NencpairsI].x = i;
					Encpairs_d[NencMax * i + *NencpairsI].y = j;
					*NencpairsI += 1;
				}
				else{
					Encpairs_d[NencMax * i + NencMax - 1 - *NencpairsJ].y = j;
					*NencpairsJ += 1;
				}
			}
		}
		ir = 1.0/sqrt(rsq);
		ir3 = ir*ir*ir;
		sb = 0.0;

		if(rsq >= 1.0 * rcritv2){
			s = x4j.w * ir3;
			if( rsq >= def_pc * rcritv2) sb = s;
//printf("%d %d %g %g %g Kick\n", i, j, 1.0, 1.0 / ir, s);
		}
		else{
			if(rsq <= 0.01 * rcritv2){
				s = 0.0;
//printf("%d %d %g %g %g Kick\n", i, j, 0, 1.0 / ir, s);

			}
			else{
				y = (rsq * ir - 0.1 * rcritv)/(0.9*rcritv);
				yy = y * y;
				s = ir3 * yy / (2.0*yy - 2.0*y + 1.0) * x4j.w;
//printf("%d %d %g %g %g Kick\n", i, j, yy / (2.0*yy - 2.0*y + 1.0), 1.0/ir, s);

			}
		}
		ac.x += __dmul_rn(r3ij.x, s);
		ac.y += __dmul_rn(r3ij.y, s);
		ac.z += __dmul_rn(r3ij.z, s);

		if(E % 10 != 2){
			b.x += __dmul_rn(r3ij.x, sb);
			b.y += __dmul_rn(r3ij.y, sb);
			b.z += __dmul_rn(r3ij.z, sb);
		}
//printf("%d %d %g %g Kick\n", i, j, s, ac.x);
	}
}

// *********************************
//Only here for testing
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
// ********************************
template < int E >
__device__ void  accG3(double3 &ac, double3 &b, double4 &x4i, double4 &x4j, double rcriti, double rcritvi, double rcritj, double rcritvj, int groupIndexi, int groupIndexj, int *NencpairsI, int *NencpairsJ, int2 *Encpairs_d, int j, int i, int icNB, int NencMax, double &test, double t){
	if( i != j && x4i.w >= 0.0 && x4j.w >= 0.0){
		double rsq, ir, ir3, s;
		double3 r3ij;
		double rcritv, rcritv2;
//		double rcrit, rcrit2;
		int Ni, Nj;

		r3ij.x = x4j.x - x4i.x;
		r3ij.y = x4j.y - x4i.y;
		r3ij.z = x4j.z - x4i.z;

		rsq = r3ij.x*r3ij.x + r3ij.y*r3ij.y + r3ij.z*r3ij.z;
		ir = 1.0/sqrt(rsq);
		ir3 = ir*ir*ir;

		double B = x4i.w * x4j.w * ir;

		rcritv = fmax(rcritvi, rcritvj);

		rcritv2 = rcritv * rcritv;
		if(E <= 2){	
#if G3 == 1
			if(((groupIndexi == groupIndexj && groupIndexi >= 0 && groupIndexi < icNB) || rsq < def_pc * rcritv) && x4i.w > 0.0 && x4j.w > 0.0){	
#else
			if(((groupIndexi == groupIndexj && groupIndexi >= 0 && groupIndexi < icNB) || B > G3Limit2) && x4i.w > 0.0 && x4j.w > 0.0){	

#endif
//printf("Precheck %d %d\n", i, j);
				if( i < j){
					Ni = atomicAdd(NencpairsI, 1);
					Encpairs_d[NencMax * i + Ni].x = i;
					Encpairs_d[NencMax * i + Ni].y = j;
				}
				else{
					Nj = atomicAdd(NencpairsJ, 1);
					Encpairs_d[NencMax * i + NencMax - 1 - Nj].y = j;

				}
			}
		}
		if(E <= 22 && E >= 20){ // prechecker used for Test Particle Mode
			if(rsq < def_pc * rcritv2){
				Encpairs_d[NencMax * i + *NencpairsI].x = i;
				Encpairs_d[NencMax * i + *NencpairsI].y = j;
				*NencpairsI += 1;
			}
		}
		if(E <= 12 && E >=10){ //prechecker used for Test Particle Mode
			if(rsq < def_pc * rcritv2){
				if(i < j){
					Encpairs_d[NencMax * i + *NencpairsI].x = i;
					Encpairs_d[NencMax * i + *NencpairsI].y = j;
					*NencpairsI += 1;
				}
				else{
					Encpairs_d[NencMax * i + NencMax - 1 - *NencpairsJ].y = j;
					*NencpairsJ += 1;
				}
			}
		}

		s = x4j.w * ir3;

		if(groupIndexi == groupIndexj && groupIndexi >= 0 && groupIndexi < icNB) s = 0.0;

		ac.x += __dmul_rn(r3ij.x, s);
		ac.y += __dmul_rn(r3ij.y, s);
		ac.z += __dmul_rn(r3ij.z, s);
		if(E % 10 != 2){
			b.x += __dmul_rn(r3ij.x, s);
			b.y += __dmul_rn(r3ij.y, s);
			b.z += __dmul_rn(r3ij.z, s);
		}
// /*if(s != 0.0)*/ printf("%.20g %d %d %.20g %d %d %.20g Kick\n", t, i, j, s, groupIndexi, groupIndexj, ac.x);

	}
}

// **************************************
//This kernel performs the first kick of the time step, in the case of no close encounters.
//It reuses the values from the seccond kick in the previous time step.

//Authors: Simon Grimm
//April 2016
//
// ****************************************
__global__ void kick32B_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, int N){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	if(id < N && x4_d[id].w >= 0.0){
		v4_d[id].x += acck_d[id].x;
		v4_d[id].y += acck_d[id].y;
		v4_d[id].z += acck_d[id].z;
//printf("KickB %d %g %g %g %g\n", id, acck_d[id].x, acck_d[id].y, acck_d[id].z, v4_d[id].x);
	}
}
// *******************************************
//This kernel is used to sort the close encoutner list, to be able to reproduce simulations exactly
//It shoud be used only for debugging or special cases.

//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
// *********************************************
__global__ void Sort_kernel(int2 *Encpairs2_d, int N, int NencMax){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;	

	if(id < N){
		int NI = Encpairs2_d[id * NencMax].x;
		int NJ = Encpairs2_d[id * NencMax + 1].x;

		int stop = 0;
		while(stop == 0){
			stop = 1;
			for(int i = 0; i < NI - 1; ++i){
				int jj = Encpairs2_d[id * NencMax + i].y;
				int jjnext = Encpairs2_d[id * NencMax + i + 1].y;

				if(jjnext < jj){
					//swap
					Encpairs2_d[id * NencMax + i].y = jjnext;
					Encpairs2_d[id * NencMax + i + 1].y = jj;
					stop = 0;

				}
			}
		}
		stop = 0;
		while(stop == 0){
			stop = 1;
			for(int i = 0; i < NJ - 1; ++i){
				int jj = Encpairs2_d[id * NencMax + NencMax - 1 - i].y;
				int jjnext = Encpairs2_d[id * NencMax + NencMax - 1 - i - 1].y;

				if(jjnext < jj){
					//swap
					Encpairs2_d[id * NencMax + NencMax - 1 - i].y = jjnext;
					Encpairs2_d[id * NencMax + NencMax - 1 - i - 1].y = jj;
					stop = 0;
				}			
			}
		}

	}

}
__global__ void Sortb_kernel(int2 *Encpairs2_d, int N, int NencMax){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;	

	if(id < N){
		int NI = Encpairs2_d[id * NencMax].x;

		int stop = 0;
		while(stop == 0){
			stop = 1;
			for(int i = 0; i < NI - 1; ++i){
				int jj = Encpairs2_d[id * NencMax + i].y;
				int jjnext = Encpairs2_d[id * NencMax + i + 1].y;

				if(jjnext < jj){
					//swap
					Encpairs2_d[id * NencMax + i].y = jjnext;
					Encpairs2_d[id * NencMax + i + 1].y = jj;
					stop = 0;

				}
			}
		}
		stop = 0;
	}

}



// **************************************
//This kernel performs the first kick of the time step, in the case of close interactions.
//It reuses the values from the seccond kick in the previous time step, and adds the terms aij*dt*Kij for all
//the bodies involved in a close encounter.
//NI is the number of bodies involved in a close encounter with body i which have a bigger index than i.
//NJ is the number of bodies involved in a close encounter with body i which have a smaller index than i.

//Authors: Simon Grimm, Joachim Stadel
////March 2014
//

// ****************************************
template <int Bl>
__global__ void kick32A_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double *rcritv_d, const double dtksq, int2 *Encpairs2_d, double *test_d, int N, int NencMax, double t){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;	

	__shared__ double3 a_s[Bl];

	a_s[idy].x = 0.0;
	a_s[idy].y = 0.0;
	a_s[idy].z = 0.0;
	__syncthreads();
	if(id < N){
		int NI = Encpairs2_d[id * NencMax].x;
		int NJ = Encpairs2_d[id * NencMax + 1].x;
//if(NI > 0 || NJ > 0) printf("NI NJ %d %d %d\n", id, NI, NJ);
		double4 x4i = x4_d[id];
		double rcritvi = rcritv_d[id];
		__syncthreads();
		for(int i = 0; i < NI; ++i){
			int jj = Encpairs2_d[id * NencMax + i].y;
			double4 x4j = x4_d[jj];
//printf("AI %d %d %d %.40g %.40g %.40g %.40g\n", id, jj, NI, x4i.x, x4j.x, v4_d[id].z, v4_d[jj].z);
			double rcritvj = rcritv_d[jj];
			accA(a_s[idy], x4i, x4j, rcritvi, rcritvj, jj, id);
		}
		__syncthreads();
		for(int i = 0; i < NJ; ++i){
			int jj = Encpairs2_d[id * NencMax + NencMax - 1 - i].y;
			double4 x4j = x4_d[jj];
//printf("AJ %d %d %d %.40g %.40g %.40g %.40g\n", id, jj, NJ, x4i.x, x4j.x, v4_d[id].z, v4_d[jj].z);
			double rcritvj = rcritv_d[jj];
			accA(a_s[idy], x4i, x4j, rcritvi, rcritvj, jj, id);
		}
		__syncthreads();
		v4_d[id].x +=  __dmul_rn(a_s[idy].x, dtksq) + acck_d[id].x;
		v4_d[id].y +=  __dmul_rn(a_s[idy].y, dtksq) + acck_d[id].y;
		v4_d[id].z +=  __dmul_rn(a_s[idy].z, dtksq) + acck_d[id].z;
//printf("K %d %.40g %.40g %.40g\n", id, v4_d[id].x, v4_d[id].y, v4_d[id].z);
	}
}
template <int Bl>
__global__ void kick32Ab_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double *rcritv_d, const double dtksq, int2 *Encpairs2_d, double *test_d, int N, int NencMax, double t){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;	

	__shared__ double3 a_s[Bl];

	a_s[idy].x = 0.0;
	a_s[idy].y = 0.0;
	a_s[idy].z = 0.0;
	__syncthreads();
	if(id < N){
		int NI = Encpairs2_d[id * NencMax].x;
//if(NI > 0) printf("NI %d %d\n", id, NI);
		double4 x4i = x4_d[id];
		double rcritvi = rcritv_d[id];
		__syncthreads();
		for(int i = 0; i < NI; ++i){
			int jj = Encpairs2_d[id * NencMax + i].y;
			double4 x4j = x4_d[jj];
//printf("AI %d %d %d %.40g %.40g %.40g %.40g\n", id, jj, NI, x4i.x, x4j.x, v4_d[id].z, v4_d[jj].z);
			double rcritvj = rcritv_d[jj];
			accA(a_s[idy], x4i, x4j, rcritvi, rcritvj, jj, id);
		}
		__syncthreads();
		v4_d[id].x +=  __dmul_rn(a_s[idy].x, dtksq) + acck_d[id].x;
		v4_d[id].y +=  __dmul_rn(a_s[idy].y, dtksq) + acck_d[id].y;
		v4_d[id].z +=  __dmul_rn(a_s[idy].z, dtksq) + acck_d[id].z;
//printf("K %d %.40g %.40g %.40g\n", id, v4_d[id].x, v4_d[id].y, v4_d[id].z);
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
//E = 0: Precheck + acck. used in initial step
//E = 1: Kick + Precheck + acck. used in main steps
//E = 2: Kick + Precheck. used in mid term steps of higher order integration
//
//The Kernel is launched with N blocks a NB theads.
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
//****************************************
template <const int Bl, int Bl2, int E>
__global__ void kick16_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double *rcrit_d, double *rcritv_d, int *groupIndex_d, const double dtksq, int *Nencpairs_d, int2 *Encpairs_d, int2 *Encpairs2_d, double *test_d, int icNB, int NencMax, double t){
	int idy = threadIdx.x;
	int idx = blockIdx.x;

	__shared__ double3 ab1_s[Bl2]; 		//the b1_s array is here stored in ab1_s[idy + 16]
	__shared__ int NencpairsI_s;
	__shared__ int NencpairsJ_s;

	double4 x4i = x4_d[idx];
	double rcriti = rcrit_d[idx];
	double rcritvi = rcritv_d[idx];
#if G3 > 0
	int groupIndexi = groupIndex_d[idx];
#endif

	double4 x4j;
	double rcritj, rcritvj;

	if(idy < Bl){
		x4j = x4_d[idy];
		rcritj = rcrit_d[idy];
		rcritvj = rcritv_d[idy];
	}
	else{
		x4j.x = 0.0;
		x4j.y = 0.0;
		x4j.z = 0.0;
		x4j.w = 0.0;
		rcritj = 0.0;
		rcritvj = 0.0;
	}
#if G3 > 0
	int groupIndexj;
	if(idy < Bl) groupIndexj = groupIndex_d[idy];
	else groupIndexj = 0;
#endif


	double test;

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
#if G3 == 0
		acc<E>(ab1_s[idy], ab1_s[idy + 16], x4i, x4j, rcriti, rcritvi, rcritj, rcritvj, &NencpairsI_s, &NencpairsJ_s, Encpairs2_d, idy, idx, NencMax, test); 
#else
		accG3<E>(ab1_s[idy], ab1_s[idy + 16], x4i, x4j, rcriti, rcritvi, rcritj, rcritvj, groupIndexi, groupIndexj, &NencpairsI_s, &NencpairsJ_s, Encpairs2_d, idy, idx, icNB, NencMax, test, t);
#endif
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

		if(E >= 1){
			v4_d[idx].x += ab1[0].x * dtksq;
			v4_d[idx].y += ab1[0].y * dtksq;
			v4_d[idx].z += ab1[0].z * dtksq;
//printf("Kick %d %g %g %g\n", idx, ab1[0].x, ab1[0].x * dtksq, v4_d[idx].x);
		}
		if(E <= 1){
			acck_d[idx].x += ab1[16].x * dtksq;
			acck_d[idx].y += ab1[16].y * dtksq;
			acck_d[idx].z += ab1[16].z * dtksq;
		}
		if(E <= 2){
			Ne = atomicAdd(Nencpairs_d, NencpairsI_s);
		}
	}
	if(E <= 2){
		__syncthreads();
		if(idy < Bl){
			for(int i = 0; i < NencpairsI_s; i += Bl){
				if(idy + i < NencpairsI_s){
					Encpairs_d[idy + i + Ne] = Encpairs2_d[NencMax * idx + idy + i];
				}
			}
		}
		__syncthreads();
		if(idy == 0){
			Encpairs2_d[NencMax * idx].x = NencpairsI_s;
			Encpairs2_d[NencMax * idx + 1].x = NencpairsJ_s;

		}
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
//E = 0: Precheck + acck. used in initial step
//E = 1: Kick + Precheck + acck. used in main steps
//E = 2: Kick + Precheck. used in mid term steps of higher order integration*
//
//The Kernel is launched with N blocks a NB theads.

//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
//****************************************
template <const int Bl, int Bl2, int E>
__global__ void kick32_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double *rcrit_d, double *rcritv_d, int *groupIndex_d, const double dtksq, int *Nencpairs_d, int2 *Encpairs_d, int2 *Encpairs2_d, double *test_d, int icNB, int NencMax, double t){
	int idy = threadIdx.x;
	int idx = blockIdx.x;

	__shared__ double3 a1_s[Bl2];
	__shared__ double3 b1_s[Bl2];
	__shared__ int NencpairsI_s;
	__shared__ int NencpairsJ_s;

	double4 x4i = x4_d[idx];
	double rcriti = rcrit_d[idx];
	double rcritvi = rcritv_d[idx];
#if G3 > 0
	int groupIndexi = groupIndex_d[idx];
#endif

	double4 x4j = x4_d[idy];
	double rcritj = rcrit_d[idy];
	double rcritvj = rcritv_d[idy];
#if G3 > 0
	int groupIndexj = groupIndex_d[idy];
#endif

	double test;

	if(idy == 0){
		NencpairsI_s = 0;
		NencpairsJ_s = 0;
	}
	__syncthreads();
	a1_s[idy].x = 0.0;
	a1_s[idy].y = 0.0;
	a1_s[idy].z = 0.0;

	b1_s[idy].x = 0.0;
	b1_s[idy].y = 0.0;
	b1_s[idy].z = 0.0;

	__syncthreads();
	if(idy < Bl){
#if G3 == 0
		acc<E>(a1_s[idy], b1_s[idy], x4i, x4j, rcriti, rcritvi, rcritj, rcritvj, &NencpairsI_s, &NencpairsJ_s, Encpairs2_d, idy, idx, NencMax, test); 
#else
		accG3<E>(a1_s[idy], b1_s[idy], x4i, x4j, rcriti, rcritvi, rcritj, rcritvj, groupIndexi, groupIndexj, &NencpairsI_s, &NencpairsJ_s, Encpairs2_d, idy, idx, icNB, NencMax, test, t); 

#endif
	}

	__syncthreads();
	volatile double3 *a1 = a1_s;
	volatile double3 *b1 = b1_s;

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
	else{
		if(E <= 1){
			if(Bl >= 64) b1[idy-32].x += b1[idy + 32-32].x;
			if(Bl >= 32) b1[idy-32].x += b1[idy + 16-32].x;
			b1[idy-32].x += b1[idy + 8-32].x;
			b1[idy-32].x += b1[idy + 4-32].x;
			b1[idy-32].x += b1[idy + 2-32].x;
			b1[idy-32].x += b1[idy + 1-32].x;

			if(Bl >= 64) b1[idy-32].y += b1[idy + 32-32].y;
			if(Bl >= 32) b1[idy-32].y += b1[idy + 16-32].y;
			b1[idy-32].y += b1[idy + 8-32].y;
			b1[idy-32].y += b1[idy + 4-32].y;
			b1[idy-32].y += b1[idy + 2-32].y;
			b1[idy-32].y += b1[idy + 1-32].y;

			if(Bl >= 64) b1[idy-32].z += b1[idy + 32-32].z;
			if(Bl >= 32) b1[idy-32].z += b1[idy + 16-32].z;
			b1[idy-32].z += b1[idy + 8-32].z;
			b1[idy-32].z += b1[idy + 4-32].z;
			b1[idy-32].z += b1[idy + 2-32].z;
			b1[idy-32].z += b1[idy + 1-32].z;
		}
	}


	__shared__ int Ne;
	__syncthreads();

	if(idy == 0){

		if(E >= 1){
			v4_d[idx].x += a1[0].x * dtksq;
			v4_d[idx].y += a1[0].y * dtksq;
			v4_d[idx].z += a1[0].z * dtksq;
		}
		if(E <= 1){
			acck_d[idx].x += b1[0].x * dtksq;
			acck_d[idx].y += b1[0].y * dtksq;
			acck_d[idx].z += b1[0].z * dtksq;
		}
		if(E <= 2){
			Ne = atomicAdd(Nencpairs_d, NencpairsI_s);
		}
	}
	if(E <= 2){
		__syncthreads();
		if(idy < Bl){
			for(int i = 0; i < NencpairsI_s; i += Bl){
				if(idy + i < NencpairsI_s){
					Encpairs_d[idy + i + Ne] = Encpairs2_d[NencMax * idx + idy + i];
				}
			}
		}
		__syncthreads();
		if(idy == 0){
			Encpairs2_d[NencMax * idx].x = NencpairsI_s;
			Encpairs2_d[NencMax * idx + 1].x = NencpairsJ_s;

		}
	}
}

// **************************************
//This kernel performs the seccond kick of the time step for test particles
//It calculates the acceleration between all bodies with respect to the changeover function K.
//It also calculates all accelerations from bodies not beeing in a close encounter. This values will then be used 
//it the next time step.
//It performs also a precheck for close encouter candidates. This pairs are stored in the array Encpairs_d.
//The number of close encounter candidates is stored in Nencpairs_d.
//
//E = 0: Precheck + acck. used in initial step
//E = 1: Kick + Precheck + acck. used in main steps
//E = 2: Kick + Precheck. used in mid term steps of higher order integration
//
//
//Authors: Simon Grimm, Joachim Stadel
//March 2014
//
// ****************************************
template <int Bl, int E>
__global__ void kicksmallb_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double *rcrit_d, double *rcritv_d, int *groupIndex_d, const double dtksq, int N, int *Nencpairs_d, int2 *Encpairs_d, int2 *Encpairs2_d, int Nsmall, int NencMax){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	__shared__ double3 a_s[Bl];
	__shared__ double3 b_s[Bl];
	int NencpairsI;
	int NencpairsJ;

	a_s[idy].x = 0.0;
	a_s[idy].y = 0.0;
	a_s[idy].z = 0.0;

	b_s[idy].x = 0.0;
	b_s[idy].y = 0.0;
	b_s[idy].z = 0.0;

	NencpairsI = 0;
	NencpairsJ = 0;


	__syncthreads();

	if(id < Nsmall){
		double rcritvi = rcritv_d[id + N];
		double4 x4i = x4_d[id + N];
		double test;
		for(int j = 0; j < N; ++j){
			acc<E + 20>(a_s[idy], b_s[idy], x4i, x4_d[j], 0.0 , rcritvi, rcrit_d[j], rcritv_d[j], &NencpairsI, &NencpairsJ, Encpairs2_d, j, id + N, NencMax, test);
		}
		if(E >= 1){
			v4_d[id + N].x += a_s[idy].x * dtksq;
			v4_d[id + N].y += a_s[idy].y * dtksq;
			v4_d[id + N].z += a_s[idy].z * dtksq;
		}
		if(E <= 1){
			acck_d[id + N].x += b_s[idy].x * dtksq;
			acck_d[id + N].y += b_s[idy].y * dtksq;
			acck_d[id + N].z += b_s[idy].z * dtksq;
		}
		if(E <= 2){
			int Ne = atomicAdd(Nencpairs_d, NencpairsJ);
			for(int ii = 0; ii < NencpairsJ; ++ii){
				Encpairs_d[ii + Ne] = Encpairs2_d[NencMax * (id + N) + ii];
			}
			Encpairs2_d[NencMax * (id + N)].x = NencpairsI;
			Encpairs2_d[NencMax * (id + N) + 1].x = NencpairsJ;
		}
	}
}

// **************************************
//This kernel performs the seccond kick of the time step for test particles
//It calculates the acceleration between all bodies with respect to the changeover function K.
//It also calculates all accelerations from bodies not beeing in a close encounter. This values will then be used 
//it the next time step.
//It performs also a precheck for close encouter candidates. This pairs are stored in the array Encpairs_d.
//The number of close encounter candidates is stored in Nencpairs_d.
//
//E = 0: Precheck + acck. used in initial step
//E = 1: Kick + Precheck + acck. used in main steps
//E = 2: Kick + Precheck. used in mid term steps of higher order integration
//
//
//Authors: Simon Grimm, Joachim Stadel
//March 2014
//
// ****************************************
template <int Bl, int E>
__global__ void kicksmall_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double *rcrit_d, double *rcritv_d, int *groupIndex_d, const double dtksq, int N, int *Nencpairs_d, int2 *Encpairs_d, int2 *Encpairs2_d, int Nsmall, int NencMax, double MinMass){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	__shared__ double3 a_s[Bl];
	__shared__ double3 b_s[Bl];
	int NencpairsI;
	int NencpairsJ;

	a_s[idy].x = 0.0;
	a_s[idy].y = 0.0;
	a_s[idy].z = 0.0;

	b_s[idy].x = 0.0;
	b_s[idy].y = 0.0;
	b_s[idy].z = 0.0;

	NencpairsI = 0;
	NencpairsJ = 0;


	__syncthreads();

	if(id < N){
		double4 x4i = x4_d[id];
		double rcriti = rcrit_d[id];
		double rcritvi = rcritv_d[id];
		double test;
		int N1 = N;
		if(MinMass > 0.0) N1 = N + Nsmall;
		for(int j = 0; j < N1; ++j){
			acc<E + 10>(a_s[idy], b_s[idy], x4i, x4_d[j], rcriti, rcritvi, rcrit_d[j], rcritv_d[j], &NencpairsI, &NencpairsJ, Encpairs2_d, j, id, NencMax, test);
		}
		if(E >= 1){
			v4_d[id].x += a_s[idy].x * dtksq;
			v4_d[id].y += a_s[idy].y * dtksq;
			v4_d[id].z += a_s[idy].z * dtksq;
		}
		if(E <= 1){
			acck_d[id].x += b_s[idy].x * dtksq;
			acck_d[id].y += b_s[idy].y * dtksq;
			acck_d[id].z += b_s[idy].z * dtksq;
			int Ne = atomicAdd(Nencpairs_d, NencpairsI);
			for(int ii = 0; ii < NencpairsI; ++ii){
				Encpairs_d[ii + Ne] = Encpairs2_d[NencMax * id + ii];
			}
			Encpairs2_d[NencMax * id].x = NencpairsI;
			Encpairs2_d[NencMax * id + 1].x = NencpairsJ;
		}
	}
	else if(id < N + Nsmall){
		double rcritvi = rcritv_d[id];
		double4 x4i = x4_d[id];
		double test;
		for(int j = 0; j < N; ++j){
			acc<E + 20>(a_s[idy], b_s[idy], x4i, x4_d[j], 0.0 , rcritvi, rcrit_d[j], rcritv_d[j], &NencpairsI, &NencpairsJ, Encpairs2_d, j, id, NencMax, test);
		}
		if(E >= 1){
			v4_d[id].x += a_s[idy].x * dtksq;
			v4_d[id].y += a_s[idy].y * dtksq;
			v4_d[id].z += a_s[idy].z * dtksq;
		}
		if(E <= 1){
			acck_d[id].x += b_s[idy].x * dtksq;
			acck_d[id].y += b_s[idy].y * dtksq;
			acck_d[id].z += b_s[idy].z * dtksq;
		}
		if(E <= 2){
			int Ne = atomicAdd(Nencpairs_d, NencpairsJ);
			for(int ii = 0; ii < NencpairsJ; ++ii){
				Encpairs_d[ii + Ne] = Encpairs2_d[NencMax * id + ii];
			}
			Encpairs2_d[NencMax * id].x = NencpairsI;
			Encpairs2_d[NencMax * id + 1].x = NencpairsJ;
		}
	}
}

// **************************************
//This kernel performs the seccond kick of the time step.
//It calculates the acceleration between all bodies with respect to the changeover function K.
//It also calculates all accelerations from bodies not beeing in a close encounter and store it in accK_d. This values will then be used 
//it the next time step.
//It performs also a precheck for close encouter candidates. This pairs are stored in the array Encpairs_d.
//The number of close encounter candidates is stored in Nencpairs_d.
//
//E = 0: Precheck + acck. used in initial step
//E = 1: Kick + Precheck + acck. used in main steps
//E = 2: Kick + Precheck. used in mid term steps of higher order integration
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
// ****************************************
template <int Bl, int Bl2, int Nmax, int E>
__global__ void KickM2_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double *rcrit_d, double *rcritv_d, int *Nencpairs_d, int2 *Encpairs_d, double *dtksq_d, double Kt, int *index_d, int NT, double *test_d){

	int idy = threadIdx.x;
	int id = blockIdx.x * Bl2 + idy - Nmax;

	__shared__ volatile double3 a_s[Bl + Nmax];
	__shared__ volatile double3 b_s[Bl + Nmax];
	__shared__ double4 x4_s[Bl + Nmax];
	//__shared__ double rcrit_s[Bl + Nmax];
	__shared__ double rcritv_s[Bl + Nmax];
	__shared__ int st_s[Bl + Nmax];

	a_s[idy].x = 0.0;
	a_s[idy].y = 0.0;
	a_s[idy].z = 0.0;
	b_s[idy].x = 0.0;
	b_s[idy].y = 0.0;
	b_s[idy].z = 0.0;

	double dtksqKt = 0.0;

	if(id < NT && id >= 0){
		st_s[idy] = index_d[id] / 100;
		x4_s[idy] = x4_d[id];
		//rcrit_s[idy] = rcrit_d[id];
		rcritv_s[idy] = rcritv_d[id];
		dtksqKt = dtksq_d[st_s[idy]] * Kt;
	}
	else{
		st_s[idy] = -idy-1;
		x4_s[idy].x = 0.0; 
		x4_s[idy].y = 0.0;
		x4_s[idy].z = 0.0;
		x4_s[idy].w = 0.0;
		//rcrit_s[idy] = 0.0;
		rcritv_s[idy] = 0.0;
	}
	//halo
	if(idy < Nmax){
		a_s[idy + Bl].x = 0.0;
		a_s[idy + Bl].y = 0.0;
		a_s[idy + Bl].z = 0.0;
		b_s[idy + Bl].x = 0.0;
		b_s[idy + Bl].y = 0.0;
		b_s[idy + Bl].z = 0.0;	
		//right
		if(id + Bl < NT){
			st_s[idy + Bl] = index_d[id + Bl] / 100;
			x4_s[idy + Bl] = x4_d[id + Bl];
			//rcrit_s[idy + Bl] = rcrit_d[id + Bl];
			rcritv_s[idy + Bl] = rcritv_d[id + Bl];
		}
		else{
			st_s[idy + Bl] = -idy-Bl-1;
			x4_s[idy + Bl].x = 0.0;
			x4_s[idy + Bl].y = 0.0;
			x4_s[idy + Bl].z = 0.0;
			x4_s[idy + Bl].w = 0.0;
			//rcrit_s[idy + Bl] = 0.0;
			rcritv_s[idy + Bl] = 0.0;
		}
	}

	volatile double a;
	volatile double b;
	volatile double rx, ry, rz;
	volatile double rsq, ir, ir3;
	volatile double rcritv, rcritv2;
	volatile double y, yy, K;
	volatile double si, sj;

	int2 ij;
	
	for(volatile int j = Nmax - 1; j > 0; --j){
		__syncthreads();
		if((st_s[idy] - st_s[idy + j]) == 0 && x4_s[idy].w >= 0.0 && x4_s[idy + j].w >= 0.0){
			rcritv = fmax(rcritv_s[idy], rcritv_s[idy + j]);
			rcritv2 = rcritv * rcritv;
			rx = x4_s[idy + j].x - x4_s[idy].x;
			ry = x4_s[idy + j].y - x4_s[idy].y;
			rz = x4_s[idy + j].z - x4_s[idy].z;
			rsq = rx * rx + ry * ry + rz * rz;
			ir = 1.0 / sqrt(rsq);
			ir3 = ir * ir * ir;
			if(E <= 2){
				if(rsq < def_pc * rcritv2 && (x4_s[idy].w > 0.0 || x4_s[idy + j].w > 0.0)){  //prechecker
					if(idy >= Nmax){
						int Ne = atomicAdd(Nencpairs_d, 1);
						atomicAdd(Nencpairs_d + st_s[idy] + 1, 1);
						ij.x = id;
						ij.y = id + j;	
						Encpairs_d[Ne] = ij;
					}
				}
			}

			if(rsq >= rcritv2){
				si = x4_s[idy + j].w * ir3;
				sj = x4_s[idy].w * ir3;
			}
			else{
				if(rsq <= 0.01 * rcritv2){
					si = 0.0;
					sj = 0.0;
				}
				else{
					y = (rsq * ir - 0.1 * rcritv) / (0.9 * rcritv);
					yy = y * y;
					K = ir3 * yy / (2.0 * yy - 2.0 * y + 1.0);
					si = K * x4_s[idy + j].w;
					sj = K * x4_s[idy].w;
				}
			}
			a_s[idy].x += __dmul_rn(rx, si);
			a_s[idy].y += __dmul_rn(ry, si);
			a_s[idy].z += __dmul_rn(rz, si);

			b_s[idy + j].x += __dmul_rn(-rx, sj);
			b_s[idy + j].y += __dmul_rn(-ry, sj);
			b_s[idy + j].z += __dmul_rn(-rz, sj);

		}
	}
	__syncthreads();
	if(id < NT && id >= 0 && idy >= Nmax){
		if(E >= 1){
			v4_d[id].x += __dmul_rn((a_s[idy].x + b_s[idy].x), dtksqKt);
			v4_d[id].y += __dmul_rn((a_s[idy].y + b_s[idy].y), dtksqKt);
			v4_d[id].z += __dmul_rn((a_s[idy].z + b_s[idy].z), dtksqKt);
		}
		if(E <= 1){
			acck_d[id].x = __dmul_rn((a_s[idy].x + b_s[idy].x), dtksqKt);
			acck_d[id].y = __dmul_rn((a_s[idy].y + b_s[idy].y), dtksqKt);
			acck_d[id].z = __dmul_rn((a_s[idy].z + b_s[idy].z), dtksqKt);
		}
	}
}
#endif

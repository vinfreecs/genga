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
__device__ void  acc(double3 &ac, double3 &b, double4 &x4i, double4 &x4j, double rcriti, double rcritvi, double rcritj, double rcritvj, int *NencpairsI, int *NencpairsJ, int2 *Encpairs_d, int j, int i, int icNB, double &test, const int Nconst){
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
			if(rsq < pc * rcritv2 && (x4i.w > 0.0 || x4j.w > 0.0)){  //prechecker
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
		}
		if(E <= 22 && E >= 20){ // prechecker used for Test Particle Mode
			if(rsq < pc * rcritv2){
//printf("Precheck 20 %d %d %d %d %d\n", i - icNB, j, icNB, Nconst, *NencpairsI);
				Encpairs_d[Nconst * (i - icNB) + *NencpairsI].x = i - icNB;
				Encpairs_d[Nconst * (i - icNB) + *NencpairsI].y = j;
				*NencpairsI += 1;
			}
		}
		if(E <= 12 && E >=10){ //prechecker used for Test Particle Mode
			if(rsq < pc * rcritv2){
				if(i < j){
//printf("Precheck 10 %d %d\n", i, j);
					Encpairs_d[icNB * i + *NencpairsI].x = i;
					Encpairs_d[icNB * i + *NencpairsI].y = j;
					*NencpairsI += 1;
				}
				else{
					Encpairs_d[icNB * i + icNB - 1 - *NencpairsJ].y = j;
					*NencpairsJ += 1;
				}
			}
		}
		ir = 1.0/sqrt(rsq);
		ir3 = ir*ir*ir;
		sb = 0.0;

		if(rsq >= 1.0 * rcritv2){
			s = x4j.w * ir3;
			if( rsq >= pc * rcritv2) sb = s;
//printf("%d %d %g %g\n", i, j, 1.0, 1.0 / ir);
		}
		else{
			if(rsq <= 0.01 * rcritv2){
				s = 0.0;
//printf("%d %d %g %g\n", i, j, 0, 1.0 / ir);

			}
			else{
				y = (rsq * ir - 0.1 * rcritv)/(0.9*rcritv);
				yy = y * y;
				s = ir3 * yy / (2.0*yy - 2.0*y + 1.0) * x4j.w;
//printf("%d %d %g %g\n", i, j, yy / (2.0*yy - 2.0*y + 1.0), 1.0/ir);

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
__device__ void  accG3(double3 &ac, double3 &b, double4 &x4i, double4 &x4j, double rcriti, double rcritvi, double rcritj, double rcritvj, int groupIndexi, int groupIndexj, int *NencpairsI, int *NencpairsJ, int2 *Encpairs_d, int j, int i, int icNB, double &test, const int Nconst, double t){
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
			if(((groupIndexi == groupIndexj && groupIndexi >= 0 && groupIndexi < icNB) || B > G3Limit2) && x4i.w > 0.0 && x4j.w > 0.0){	
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
		}
		if(E <= 22 && E >= 20){ // prechecker used for Test Particle Mode
			if(rsq < pc * rcritv2){
				Encpairs_d[Nconst * (i - icNB) + *NencpairsI].x = i - icNB;
				Encpairs_d[Nconst * (i - icNB) + *NencpairsI].y = j;
				*NencpairsI += 1;
			}
		}
		if(E <= 12 && E >=10){ //prechecker used for Test Particle Mode
			if(rsq < pc * rcritv2){
				if(i < j){
					Encpairs_d[icNB * i + *NencpairsI].x = i;
					Encpairs_d[icNB * i + *NencpairsI].y = j;
					*NencpairsI += 1;
				}
				else{
					Encpairs_d[icNB * i + icNB - 1 - *NencpairsJ].y = j;
					*NencpairsJ += 1;
				}
			}
		}

		s = x4j.w * ir3;

		if(groupIndexi == groupIndexj && groupIndexi >= 0 && groupIndexi < icNB) s = 0.0;
//if(s != 0.0) printf("%.10g %d %d %g %g %g %g %g %g %g %g %g %g %g %g %g %d %d\n", t, i, j, 0.0, s, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, groupIndexi, groupIndexj);
		ac.x += __dmul_rn(r3ij.x, s);
		ac.y += __dmul_rn(r3ij.y, s);
		ac.z += __dmul_rn(r3ij.z, s);
		if(E % 10 != 2){
			b.x += __dmul_rn(r3ij.x, s);
			b.y += __dmul_rn(r3ij.y, s);
			b.z += __dmul_rn(r3ij.z, s);
		}

	}
}

// **************************************
//This kernel performs the first kick of the time step, in the case of no close encounters.
//It reuses the values from the seccond kick in the previous time step.

//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
// ****************************************
__global__ void kick32B_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	if(x4_d[id].w >= 0.0){
		v4_d[id].x += acck_d[id].x;
		v4_d[id].y += acck_d[id].y;
		v4_d[id].z += acck_d[id].z;
	}
}
// **************************************
//This kernel performs the first kick of the time step for test particles, in the case of no close encounters.
//It reuses the values from the seccond kick in the previous time step.

//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
// ****************************************
__global__ void kickBsmall_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double4 *x4small_d, double4 *v4small_d, double3 *accksmall_d, int N, int Nsmall){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;
	if(id < N){
		if(x4_d[id].w >= 0.0){
			v4_d[id].x += acck_d[id].x;
			v4_d[id].y += acck_d[id].y;
			v4_d[id].z += acck_d[id].z;
		}
	}
	else if(id < Nsmall + N){
		if(x4small_d[id - N].w >= 0.0){
			v4small_d[id - N].x += accksmall_d[id - N].x;
			v4small_d[id - N].y += accksmall_d[id - N].y;
			v4small_d[id - N].z += accksmall_d[id - N].z;
		}
	}
}
// **************************************
//This kernel performs the first kick of the time step for the multi simulation mode, in the case of no close encounters.
//It reuses the values from the seccond kick in the previous time step.

//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
// ****************************************
__global__ void kickMB_kernel(double4 *v4_d, double3 *acck_d, double *test_d, int NT){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;	
	if(id < NT){
		v4_d[id].x += acck_d[id].x;
		v4_d[id].y += acck_d[id].y;
		v4_d[id].z += acck_d[id].z;
	}
}

// *******************************************
//This kernel is used to sort the close encoutner list, to be able to reproduce simulations exactly
//It shoud be used only for debugging or special cases.

//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
// *********************************************
__global__ void Sort_kernel(int2 *Encpairs2_d, int N, int icNB){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;	

	if(id < N){
		int NI = Encpairs2_d[id * icNB].x;
		int NJ = Encpairs2_d[id * icNB + 1].x;

		int stop = 0;
		while(stop == 0){
			stop = 1;
			for(int i = 0; i < NI - 1; ++i){
				int jj = Encpairs2_d[id * icNB + i].y;
				int jjnext = Encpairs2_d[id * icNB + i + 1].y;

				if(jjnext < jj){
					//swap
					Encpairs2_d[id * icNB + i].y = jjnext;
					Encpairs2_d[id * icNB + i + 1].y = jj;
					stop = 0;

				}
			}
		}
		stop = 0;
		while(stop == 0){
			stop = 1;
			for(int i = 0; i < NJ - 1; ++i){
				int jj = Encpairs2_d[id * icNB + icNB - 1 - i].y;
				int jjnext = Encpairs2_d[id * icNB + icNB - 1 - i - 1].y;

				if(jjnext < jj){
					//swap
					Encpairs2_d[id * icNB + icNB - 1 - i].y = jjnext;
					Encpairs2_d[id * icNB + icNB - 1 - i - 1].y = jj;
					stop = 0;
				}			
			}
		}

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
__global__ void kick32A_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double *rcritv_d, int *groupIndex_d, const double dtksq, int2 *Encpairs2_d, double *test_d, int N, int icNB, double t){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;	

	__shared__ double3 a_s[Bl];

	a_s[idy].x = 0.0;
	a_s[idy].y = 0.0;
	a_s[idy].z = 0.0;
	__syncthreads();
	if(id < N){
		int NI = Encpairs2_d[id * icNB].x;
		int NJ = Encpairs2_d[id * icNB + 1].x;
		double4 x4i = x4_d[id];
		double rcritvi = rcritv_d[id];
#if G3 == 1		
		int groupIndexi = groupIndex_d[id];
#endif
		__syncthreads();
		for(int i = 0; i < NI; ++i){
			int jj = Encpairs2_d[id * icNB + i].y;
			double4 x4j = x4_d[jj];
//printf("AI %d %d %.40g %.40g %.40g %.40g\n", id, jj, x4i.x, x4j.x, v4_d[id].z, v4_d[jj].z);
			double rcritvj = rcritv_d[jj];
#if G3 == 1
			int groupIndexj = groupIndex_d[jj];
			accAG3(a_s[idy], x4i, x4j, rcritvi, rcritvj, groupIndexi, groupIndexj, jj, id, icNB, t);
#else
			accA(a_s[idy], x4i, x4j, rcritvi, rcritvj, jj, id);

#endif
		}
		__syncthreads();
		for(int i = 0; i < NJ; ++i){
			int jj = Encpairs2_d[id * icNB + icNB - 1 - i].y;
			double4 x4j = x4_d[jj];
//printf("AJ %d %d %.40g %.40g %.40g %.40g\n", id, jj, x4i.x, x4j.x, v4_d[id].z, v4_d[jj].z);
			double rcritvj = rcritv_d[jj];
#if G3 == 1
			int groupIndexj = groupIndex_d[jj];
			accAG3(a_s[idy], x4i, x4j, rcritvi, rcritvj, groupIndexi, groupIndexj, jj, id, icNB, t);
#else
			accA(a_s[idy], x4i, x4j, rcritvi, rcritvj, jj, id);
#endif
		}
		__syncthreads();
		v4_d[id].x +=  __dmul_rn(a_s[idy].x, dtksq) + acck_d[id].x;
		v4_d[id].y +=  __dmul_rn(a_s[idy].y, dtksq) + acck_d[id].y;
		v4_d[id].z +=  __dmul_rn(a_s[idy].z, dtksq) + acck_d[id].z;
	}
}
// **************************************
//This kernel performs the first kick of the time step for test particles, in the case of close interactions.
//It reuses the values from the seccond kick in the previous time step, and adds the terms aij*dt*Kij for all
//the bodies involved in a close encounter.
//NI is the number of bodies involved in a close encounter with body i which have a bigger index than i.
//NJ is the number of bodies involved in a close encounter with body i which have a smaller index than i.

//Authors: Simon Grimm, Joachim Stadel
////March 2014
//

// ****************************************
template <int Bl>
__global__ void kickAsmall_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double *rcritv_d, const double dtksq, int2 *Encpairs2_d, double *test_d, int N, double4 *x4small_d, double4 *v4small_d, int2 *Encpairssmall2_d, int Nsmall, double3 *accksmall_d, int NB, const int Nconst){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;	

	__shared__ double3 a_s[Bl];

	a_s[idy].x = 0.0;
	a_s[idy].y = 0.0;
	a_s[idy].z = 0.0;
	__syncthreads();
	
	if(id < N){
		int NI = Encpairs2_d[id * NB].x;
		int NJ = Encpairs2_d[id * NB + 1].x;
		double4 x4i = x4_d[id];
		double rcritvi = rcritv_d[id];

		for(int i = 0; i < NI; ++i){
			int jj = Encpairs2_d[id * NB + i].y;
			double4 x4j = x4_d[jj];
//printf("AI %d %d %.40g %.40g\n", id, jj, x4i.x, x4j.x);
			double rcritvj = rcritv_d[jj];
			accA(a_s[idy], x4i, x4j, rcritvi, rcritvj, jj, id);
		}
		for(int i = 0; i < NJ; ++i){
			int jj = Encpairs2_d[id * NB + NB - 1 - i].y;
			double4 x4j = x4_d[jj];
//printf("AJ %d %d %.40g %.40g\n", id, jj, x4i.x, x4j.x);
			double rcritvj = rcritv_d[jj];
			accA(a_s[idy], x4i, x4j, rcritvi, rcritvj, jj, id);
		}
		v4_d[id].x +=  a_s[idy].x * dtksq  + acck_d[id].x;
		v4_d[id].y +=  a_s[idy].y * dtksq  + acck_d[id].y;
		v4_d[id].z +=  a_s[idy].z * dtksq  + acck_d[id].z;
	}
	else if(id < Nsmall + N){
		int NI =  Encpairssmall2_d[(id - N) * Nconst].x;
		double4 x4i = x4small_d[id - N];
		for(int i = 0; i < NI; ++i){
			int jj = Encpairssmall2_d[(id-N) * Nconst + i].y;
			double4 x4j = x4_d[jj];
//printf("AIsmall %d %d %.40g %.40g\n", id - N, jj, x4i.x, x4j.x);
			double rcritvj = rcritv_d[jj];
			accA(a_s[idy], x4i, x4j, 0.0, rcritvj, jj, id);
		}
		v4small_d[id - N].x +=  __dadd_rn(a_s[idy].x * dtksq, accksmall_d[id - N].x);
		v4small_d[id - N].y +=  __dadd_rn(a_s[idy].y * dtksq, accksmall_d[id - N].y);
		v4small_d[id - N].z +=  __dadd_rn(a_s[idy].z * dtksq, accksmall_d[id - N].z);
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
__global__ void kick16_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double *rcrit_d, double *rcritv_d, int *groupIndex_d, const double dtksq, int *Nencpairs_d, int2 *Encpairs_d, int2 *Encpairs2_d, double *test_d, int icNB, double t){
	int idy = threadIdx.x;
	int idx = blockIdx.x;

	__shared__ double3 ab1_s[Bl2]; 		//the b1_s array is here stored in ab1_s[idy + 16]
	__shared__ int NencpairsI_s;
	__shared__ int NencpairsJ_s;

	double4 x4i = x4_d[idx];
	double rcriti = rcrit_d[idx];
	double rcritvi = rcritv_d[idx];
#if G3 == 1
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
#if G3 == 1
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
		acc<E>(ab1_s[idy], ab1_s[idy + 16], x4i, x4j, rcriti, rcritvi, rcritj, rcritvj, &NencpairsI_s, &NencpairsJ_s, Encpairs2_d, idy, idx, icNB, test, 0); 
#else
		accG3<E>(ab1_s[idy], ab1_s[idy + 16], x4i, x4j, rcriti, rcritvi, rcritj, rcritvj, groupIndexi, groupIndexj, &NencpairsI_s, &NencpairsJ_s, Encpairs2_d, idy, idx, icNB, test, 0, t);
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
__global__ void kick32_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double *rcrit_d, double *rcritv_d, int *groupIndex_d, const double dtksq, int *Nencpairs_d, int2 *Encpairs_d, int2 *Encpairs2_d, double *test_d, int icNB, double t){
	int idy = threadIdx.x;
	int idx = blockIdx.x;

	__shared__ double3 a1_s[Bl2];
	__shared__ double3 b1_s[Bl2];
	__shared__ int NencpairsI_s;
	__shared__ int NencpairsJ_s;

	double4 x4i = x4_d[idx];
	double rcriti = rcrit_d[idx];
	double rcritvi = rcritv_d[idx];
#if G3 == 1
	int groupIndexi = groupIndex_d[idx];
#endif

	double4 x4j = x4_d[idy];
	double rcritj = rcrit_d[idy];
	double rcritvj = rcritv_d[idy];
#if G3 == 1
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
		acc<E>(a1_s[idy], b1_s[idy], x4i, x4j, rcriti, rcritvi, rcritj, rcritvj, &NencpairsI_s, &NencpairsJ_s, Encpairs2_d, idy, idx, icNB, test, 0); 
#else
		accG3<E>(a1_s[idy], b1_s[idy], x4i, x4j, rcriti, rcritvi, rcritj, rcritvj, groupIndexi, groupIndexj, &NencpairsI_s, &NencpairsJ_s, Encpairs2_d, idy, idx, icNB, test, 0, t); 

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
//E = 0: Precheck + acck. used in initial step
//E = 1: Kick + Precheck + acck. used in main steps
//E = 2: Kick + Precheck. used in mid term steps of higher order integration
//
//The Kernel is launched with N/2 blocks a 128 theads.
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
// ****************************************
template <int Bl, int E>
__global__ void kick128_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double *rcrit_d, double *rcritv_d, int *groupIndex_d, const double dtksq, int *Nencpairs_d, int2 *Encpairs_d, int2 *Encpairs2_d, double *test_d, int N2, int icNB, double t){

	int idy = threadIdx.x;
	int idx = blockIdx.x;

	__shared__ double3 a1_s[Bl];
	__shared__ double3 a2_s[Bl];
	__shared__ double3 b1_s[Bl];
	__shared__ double3 b2_s[Bl];

	__shared__ int NencpairsI_s;
	__shared__ int NencpairsI2_s;
	__shared__ int NencpairsJ_s;
	__shared__ int NencpairsJ2_s;


	double4 x4i = x4_d[idx];
	double4 x4i2 = x4_d[idx + N2];
	double4 x4j = x4_d[idy];

	double rcritj = rcrit_d[idy];
	double rcritvj = rcritv_d[idy];
#if G3 == 1
	int groupIndexj = groupIndex_d[idy];
#endif

	double rcriti = rcrit_d[idx];
	double rcriti2 = rcrit_d[idx + N2];
	double rcritvi = rcritv_d[idx];
	double rcritvi2 = rcritv_d[idx + N2];
#if G3 == 1
	int groupIndexi = groupIndex_d[idx];
	int groupIndexi2 = groupIndex_d[idx + N2];
#endif
	double test;

	a1_s[idy].x = 0.0;
	a1_s[idy].y = 0.0;
	a1_s[idy].z = 0.0;
	
	a2_s[idy].x = 0.0;
	a2_s[idy].y = 0.0;
	a2_s[idy].z = 0.0;

	b1_s[idy].x = 0.0;
	b1_s[idy].y = 0.0;
	b1_s[idy].z = 0.0;

	b2_s[idy].x = 0.0;
	b2_s[idy].y = 0.0;
	b2_s[idy].z = 0.0;

	if(idy == 0){
		NencpairsI_s = 0;
		NencpairsJ_s = 0;
	}
	if(idy == 32){
		NencpairsI2_s = 0;
		NencpairsJ2_s = 0;
	}

	__syncthreads();
#if G3 == 0
	acc<E>(a1_s[idy], b1_s[idy], x4i, x4j, rcriti, rcritvi, rcritj, rcritvj, &NencpairsI_s, &NencpairsJ_s, Encpairs2_d, idy, idx, icNB, test, 0);
	acc<E>(a2_s[idy], b2_s[idy], x4i2, x4j, rcriti2, rcritvi2, rcritj, rcritvj, &NencpairsI2_s, &NencpairsJ2_s, Encpairs2_d, idy, idx + N2, icNB, test, 0);
#else
	accG3<E>(a1_s[idy], b1_s[idy], x4i, x4j, rcriti, rcritvi, rcritj, rcritvj, groupIndexi, groupIndexj, &NencpairsI_s, &NencpairsJ_s, Encpairs2_d, idy, idx, icNB, test, 0, t);
	accG3<E>(a2_s[idy], b2_s[idy], x4i2, x4j, rcriti2, rcritvi2, rcritj, rcritvj, groupIndexi2, groupIndexj, &NencpairsI2_s, &NencpairsJ2_s, Encpairs2_d, idy, idx + N2, icNB, test, 0, t);

#endif

	__syncthreads();
	volatile double3 *a1 = a1_s;
	volatile double3 *a2 = a2_s;
	volatile double3 *b1 = b1_s;
	volatile double3 *b2 = b2_s;

	if(idy < 64){
		a1[idy].x += a1[idy + 64].x;
		a1[idy].y += a1[idy + 64].y;
		a1[idy].z += a1[idy + 64].z;

		a2[idy].x += a2[idy + 64].x;
		a2[idy].y += a2[idy + 64].y;
		a2[idy].z += a2[idy + 64].z;

		if(E <= 1){
			b1[idy].x += b1[idy + 64].x;
			b1[idy].y += b1[idy + 64].y;
			b1[idy].z += b1[idy + 64].z;

			b2[idy].x += b2[idy + 64].x;
			b2[idy].y += b2[idy + 64].y;
			b2[idy].z += b2[idy + 64].z;
		}
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

		if(E <= 1){
			b1[idy].x += b1[idy + 32].x;
			b1[idy].x += b1[idy + 16].x;
			b1[idy].x += b1[idy + 8].x;
			b1[idy].x += b1[idy + 4].x;
			b1[idy].x += b1[idy + 2].x;
			b1[idy].x += b1[idy + 1].x;

			b1[idy].y += b1[idy + 32].y;
			b1[idy].y += b1[idy + 16].y;
			b1[idy].y += b1[idy + 8].y;
			b1[idy].y += b1[idy + 4].y;
			b1[idy].y += b1[idy + 2].y;
			b1[idy].y += b1[idy + 1].y;

			b1[idy].z += b1[idy + 32].z;
			b1[idy].z += b1[idy + 16].z;
			b1[idy].z += b1[idy + 8].z;
			b1[idy].z += b1[idy + 4].z;
			b1[idy].z += b1[idy + 2].z;
			b1[idy].z += b1[idy + 1].z;
		}
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

			if(E <= 1){
				b2[idy-32].x += b2[idy + 32-32].x;
				b2[idy-32].x += b2[idy + 16-32].x;
				b2[idy-32].x += b2[idy + 8-32].x;
				b2[idy-32].x += b2[idy + 4-32].x;
				b2[idy-32].x += b2[idy + 2-32].x;
				b2[idy-32].x += b2[idy + 1-32].x;

				b2[idy-32].y += b2[idy + 32-32].y;
				b2[idy-32].y += b2[idy + 16-32].y;
				b2[idy-32].y += b2[idy + 8-32].y;
				b2[idy-32].y += b2[idy + 4-32].y;
				b2[idy-32].y += b2[idy + 2-32].y;
				b2[idy-32].y += b2[idy + 1-32].y;

				b2[idy-32].z += b2[idy + 32-32].z;
				b2[idy-32].z += b2[idy + 16-32].z;
				b2[idy-32].z += b2[idy + 8-32].z;
				b2[idy-32].z += b2[idy + 4-32].z;
				b2[idy-32].z += b2[idy + 2-32].z;
				b2[idy-32].z += b2[idy + 1-32].z;
			}
		}
	}
	__shared__ int Ne1;
	__shared__ int Ne2;

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
			Ne1 = atomicAdd(Nencpairs_d, NencpairsI_s);
		}
	}

	if(idy == 32){
		if(E >= 1){
			v4_d[idx + N2].x += a2[0].x * dtksq;
			v4_d[idx + N2].y += a2[0].y * dtksq;
			v4_d[idx + N2].z += a2[0].z * dtksq;
		}
		if(E <= 1){
			acck_d[idx + N2].x += b2[0].x * dtksq;
			acck_d[idx + N2].y += b2[0].y * dtksq;
			acck_d[idx + N2].z += b2[0].z * dtksq;
		}
		if(E <= 2){
			Ne2 = atomicAdd(Nencpairs_d, NencpairsI2_s);
		}
	}

	__syncthreads();

	if(E <= 2){

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
//E = 0: Precheck + acck. used in initial step
//E = 1: Kick + Precheck + acck. used in main steps
//E = 2: Kick + Precheck. used in mid term steps of higher order integration
//
//The Kernel is launched with N/4 blocks a 128 theads.
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
// ****************************************
template <int Bl, int NB, int E>
__global__ void kick256_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double *rcrit_d, double *rcritv_d, int *groupIndex_d, const double dtksq, int N4, int *Nencpairs_d, int2 *Encpairs_d, int2 *Encpairs2_d, double *test_d, int N, int icNB, double t){
	int idy = threadIdx.x;
	int idx = blockIdx.x;

	__shared__ double3 a1_s[Bl];
	__shared__ double3 a2_s[Bl];
	__shared__ double3 a3_s[Bl];
	__shared__ double3 a4_s[Bl];

	__shared__ double3 b1_s[Bl];
	__shared__ double3 b2_s[Bl];
	__shared__ double3 b3_s[Bl];
	__shared__ double3 b4_s[Bl];

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

	double rcriti = rcrit_d[idx];
	double rcriti2 = rcrit_d[idx + N4];
	double rcriti3 = rcrit_d[idx + 2*N4];
	double rcriti4 = rcrit_d[idx + 3*N4];
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


	double test;

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

	b1_s[idy].x = 0.0;
	b1_s[idy].y = 0.0;
	b1_s[idy].z = 0.0;

	b2_s[idy].x = 0.0;
	b2_s[idy].y = 0.0;
	b2_s[idy].z = 0.0;

	b3_s[idy].x = 0.0;
	b3_s[idy].y = 0.0;
	b3_s[idy].z = 0.0;

	b4_s[idy].x = 0.0;
	b4_s[idy].y = 0.0;
	b4_s[idy].z = 0.0;

	__syncthreads();


	for(int i = 0; i < NB; i += Bl){ 
		if(idy + i < N){
			double4 x4j = x4_d[idy + i];
			double rcritj = rcrit_d[idy + i];
			double rcritvj = rcritv_d[idy + i];
#if G3 == 0
			acc<E>(a1_s[idy], b1_s[idy], x4i, x4j, rcriti, rcritvi, rcritj, rcritvj, &NencpairsI_s, &NencpairsJ_s, Encpairs2_d, idy + i, idx, icNB, test, 0);
			acc<E>(a2_s[idy], b2_s[idy], x4i2, x4j, rcriti2, rcritvi2, rcritj, rcritvj, &NencpairsI2_s, &NencpairsJ2_s, Encpairs2_d, idy + i, idx + N4, icNB, test, 0);
			acc<E>(a3_s[idy], b3_s[idy], x4i3, x4j, rcriti3, rcritvi3, rcritj, rcritvj, &NencpairsI3_s, &NencpairsJ3_s, Encpairs2_d, idy + i, idx + 2*N4, icNB, test, 0);
			acc<E>(a4_s[idy], b4_s[idy], x4i4, x4j, rcriti4, rcritvi4, rcritj, rcritvj, &NencpairsI4_s, &NencpairsJ4_s, Encpairs2_d, idy + i, idx + 3*N4, icNB, test, 0);
#else
			int groupIndexj = groupIndex_d[idy + i];
			accG3<E>(a1_s[idy], b1_s[idy], x4i, x4j, rcriti, rcritvi, rcritj, rcritvj, groupIndexi, groupIndexj, &NencpairsI_s, &NencpairsJ_s, Encpairs2_d, idy + i, idx, icNB, test, 0, t);
			accG3<E>(a2_s[idy], b2_s[idy], x4i2, x4j, rcriti2, rcritvi2, rcritj, rcritvj, groupIndexi2, groupIndexj, &NencpairsI2_s, &NencpairsJ2_s, Encpairs2_d, idy + i, idx + N4, icNB, test, 0, t);
			accG3<E>(a3_s[idy], b3_s[idy], x4i3, x4j, rcriti3, rcritvi3, rcritj, rcritvj, groupIndexi3, groupIndexj, &NencpairsI3_s, &NencpairsJ3_s, Encpairs2_d, idy + i, idx + 2*N4, icNB, test, 0, t);
			accG3<E>(a4_s[idy], b4_s[idy], x4i4, x4j, rcriti4, rcritvi4, rcritj, rcritvj, groupIndexi4, groupIndexj, &NencpairsI4_s, &NencpairsJ4_s, Encpairs2_d, idy + i, idx + 3*N4, icNB, test, 0, t);


#endif
		}
	}
	__syncthreads();
	volatile double3 *a1 = a1_s;
	volatile double3 *a2 = a2_s;
	volatile double3 *a3 = a3_s;
	volatile double3 *a4 = a4_s;
	volatile double3 *b1 = b1_s;
	volatile double3 *b2 = b2_s;
	volatile double3 *b3 = b3_s;
	volatile double3 *b4 = b4_s;

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

			if(E <= 1){
				b1[idy].x += b1[idy + 128].x;
				b1[idy].y += b1[idy + 128].y;
				b1[idy].z += b1[idy + 128].z;

				b2[idy].x += b2[idy + 128].x;
				b2[idy].y += b2[idy + 128].y;
				b2[idy].z += b2[idy + 128].z;

				b3[idy].x += b3[idy + 128].x;
				b3[idy].y += b3[idy + 128].y;
				b3[idy].z += b3[idy + 128].z;

				b4[idy].x += b4[idy + 128].x;
				b4[idy].y += b4[idy + 128].y;
				b4[idy].z += b4[idy + 128].z;
			}
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

		if(E <= 1){
			b1[idy].x += b1[idy + 64].x;
			b1[idy].y += b1[idy + 64].y;
			b1[idy].z += b1[idy + 64].z;

			b2[idy].x += b2[idy + 64].x;
			b2[idy].y += b2[idy + 64].y;
			b2[idy].z += b2[idy + 64].z;

			b3[idy].x += b3[idy + 64].x;
			b3[idy].y += b3[idy + 64].y;
			b3[idy].z += b3[idy + 64].z;

			b4[idy].x += b4[idy + 64].x;
			b4[idy].y += b4[idy + 64].y;
			b4[idy].z += b4[idy + 64].z;
		}
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

		if(E <= 1){
			b1[idy].x += b1[idy + 32].x;
			b1[idy].x += b1[idy + 16].x;
			b1[idy].x += b1[idy + 8].x;
			b1[idy].x += b1[idy + 4].x;
			b1[idy].x += b1[idy + 2].x;
			b1[idy].x += b1[idy + 1].x;

			b1[idy].y += b1[idy + 32].y;
			b1[idy].y += b1[idy + 16].y;
			b1[idy].y += b1[idy + 8].y;
			b1[idy].y += b1[idy + 4].y;
			b1[idy].y += b1[idy + 2].y;
			b1[idy].y += b1[idy + 1].y;

			b1[idy].z += b1[idy + 32].z;
			b1[idy].z += b1[idy + 16].z;
			b1[idy].z += b1[idy + 8].z;
			b1[idy].z += b1[idy + 4].z;
			b1[idy].z += b1[idy + 2].z;
			b1[idy].z += b1[idy + 1].z;
		}
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

			if(E <= 1){
				b2[idy-32].x += b2[idy + 32-32].x;
				b2[idy-32].x += b2[idy + 16-32].x;
				b2[idy-32].x += b2[idy + 8-32].x;
				b2[idy-32].x += b2[idy + 4-32].x;
				b2[idy-32].x += b2[idy + 2-32].x;
				b2[idy-32].x += b2[idy + 1-32].x;

				b2[idy-32].y += b2[idy + 32-32].y;
				b2[idy-32].y += b2[idy + 16-32].y;
				b2[idy-32].y += b2[idy + 8-32].y;
				b2[idy-32].y += b2[idy + 4-32].y;
				b2[idy-32].y += b2[idy + 2-32].y;
				b2[idy-32].y += b2[idy + 1-32].y;

				b2[idy-32].z += b2[idy + 32-32].z;
				b2[idy-32].z += b2[idy + 16-32].z;
				b2[idy-32].z += b2[idy + 8-32].z;
				b2[idy-32].z += b2[idy + 4-32].z;
				b2[idy-32].z += b2[idy + 2-32].z;
				b2[idy-32].z += b2[idy + 1-32].z;
			}
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

				if(E <= 1){
					b3[idy-64].x += b3[idy + 32-64].x;
					b3[idy-64].x += b3[idy + 16-64].x;
					b3[idy-64].x += b3[idy + 8-64].x;
					b3[idy-64].x += b3[idy + 4-64].x;
					b3[idy-64].x += b3[idy + 2-64].x;
					b3[idy-64].x += b3[idy + 1-64].x;

					b3[idy-64].y += b3[idy + 32-64].y;
					b3[idy-64].y += b3[idy + 16-64].y;
					b3[idy-64].y += b3[idy + 8-64].y;
					b3[idy-64].y += b3[idy + 4-64].y;
					b3[idy-64].y += b3[idy + 2-64].y;
					b3[idy-64].y += b3[idy + 1-64].y;

					b3[idy-64].z += b3[idy + 32-64].z;
					b3[idy-64].z += b3[idy + 16-64].z;
					b3[idy-64].z += b3[idy + 8-64].z;
					b3[idy-64].z += b3[idy + 4-64].z;
					b3[idy-64].z += b3[idy + 2-64].z;
					b3[idy-64].z += b3[idy + 1-64].z;
				}
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

					if(E <= 1){
						b4[idy-96].x += b4[idy + 32-96].x;
						b4[idy-96].x += b4[idy + 16-96].x;
						b4[idy-96].x += b4[idy + 8-96].x;
						b4[idy-96].x += b4[idy + 4-96].x;
						b4[idy-96].x += b4[idy + 2-96].x;
						b4[idy-96].x += b4[idy + 1-96].x;

						b4[idy-96].y += b4[idy + 32-96].y;
						b4[idy-96].y += b4[idy + 16-96].y;
						b4[idy-96].y += b4[idy + 8-96].y;
						b4[idy-96].y += b4[idy + 4-96].y;
						b4[idy-96].y += b4[idy + 2-96].y;
						b4[idy-96].y += b4[idy + 1-96].y;

						b4[idy-96].z += b4[idy + 32-96].z;
						b4[idy-96].z += b4[idy + 16-96].z;
						b4[idy-96].z += b4[idy + 8-96].z;
						b4[idy-96].z += b4[idy + 4-96].z;
						b4[idy-96].z += b4[idy + 2-96].z;
						b4[idy-96].z += b4[idy + 1-96].z;
					}
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
			Ne1 = atomicAdd(Nencpairs_d, NencpairsI_s);
		}
	}

	if(idy == 32){
		if(E >= 1){
			v4_d[idx + N4].x += a2[0].x * dtksq;
			v4_d[idx + N4].y += a2[0].y * dtksq;
			v4_d[idx + N4].z += a2[0].z * dtksq;
		}
		if(E <= 1){
			acck_d[idx + N4].x += b2[0].x * dtksq;
			acck_d[idx + N4].y += b2[0].y * dtksq;
			acck_d[idx + N4].z += b2[0].z * dtksq;
		}
		if(E <= 2){
			Ne2 = atomicAdd(Nencpairs_d, NencpairsI2_s);
		}
	}

	if(idy == 64){
		if(E >= 1){
			v4_d[idx + 2*N4].x += a3[0].x * dtksq;
			v4_d[idx + 2*N4].y += a3[0].y * dtksq;
			v4_d[idx + 2*N4].z += a3[0].z * dtksq;
		} 
		if(E <= 1){
			acck_d[idx + 2*N4].x += b3[0].x * dtksq;
			acck_d[idx + 2*N4].y += b3[0].y * dtksq;
			acck_d[idx + 2*N4].z += b3[0].z * dtksq;
		}
		if(E <= 2){
			Ne3 = atomicAdd(Nencpairs_d, NencpairsI3_s);
		}
	}
	
	if(idy == 96){
		if(E >= 1){
			v4_d[idx + 3*N4].x += a4[0].x * dtksq;
			v4_d[idx + 3*N4].y += a4[0].y * dtksq;
			v4_d[idx + 3*N4].z += a4[0].z * dtksq;
		} 
		if(E <= 1){
			acck_d[idx + 3*N4].x += b4[0].x * dtksq;
			acck_d[idx + 3*N4].y += b4[0].y * dtksq;
			acck_d[idx + 3*N4].z += b4[0].z * dtksq;
		}
		if(E <= 2){
			Ne4 = atomicAdd(Nencpairs_d, NencpairsI4_s);
		}
	}

	if(E <= 2){
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
//E = 0: Precheck + acck. used in initial step
//E = 1: Kick + Precheck + acck. used in main steps
//E = 2: Kick + Precheck. used in mid term steps of higher order integration
//
//The Kernel is launched with N/4 blocks a 128 theads.
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
// ****************************************
template <int Bl, int NB, int E>
__global__ void kick4_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double *rcrit_d, double *rcritv_d, int *groupIndex_d, const double dtksq, int N4, int *Nencpairs_d, int2 *Encpairs_d, int2 *Encpairs2_d, double *test_d, int N, int icNB, double t){
	int idy = threadIdx.x;
	int idx = blockIdx.x;

	int Bl_2 = Bl/2;

	__shared__ double3 a1_s[Bl/2];
	__shared__ double3 a2_s[Bl/2];
	__shared__ double3 a3_s[Bl/2];
	__shared__ double3 a4_s[Bl/2];

	__shared__ double3 b1_s[Bl/2];
	__shared__ double3 b2_s[Bl/2];
	__shared__ double3 b3_s[Bl/2];
	__shared__ double3 b4_s[Bl/2];

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

	double rcriti = rcrit_d[idx];
	double rcriti2 = rcrit_d[idx+N4];
	double rcriti3 = rcrit_d[idx+2*N4];
	double rcriti4 = rcrit_d[idx+3*N4];
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
	double test;

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

		b1_s[idy].x = 0.0;
		b1_s[idy].y = 0.0;
		b1_s[idy].z = 0.0;

		b3_s[idy].x = 0.0;
		b3_s[idy].y = 0.0;
		b3_s[idy].z = 0.0;

		__syncthreads();
		for(int i = 0; i < NB; i += Bl_2){
			if(idy + i < N){
				double4 x4j = x4_d[idy + i];
				double rcritj = rcrit_d[idy + i];
				double rcritvj = rcritv_d[idy + i];
#if G3 == 0
				acc<E>(a1_s[idy], b1_s[idy], x4i, x4j, rcriti, rcritvi, rcritj, rcritvj, &NencpairsI_s, &NencpairsJ_s, Encpairs2_d, idy + i, idx, icNB, test, 0);
				acc<E>(a3_s[idy], b3_s[idy], x4i3, x4j, rcriti3, rcritvi3, rcritj, rcritvj, &NencpairsI3_s, &NencpairsJ3_s, Encpairs2_d, idy + i, idx +2*N4, icNB, test, 0);
#else
				int groupIndexj = groupIndex_d[idy + i];
				accG3<E>(a1_s[idy], b1_s[idy], x4i, x4j, rcriti, rcritvi, rcritj, rcritvj, groupIndexi, groupIndexj, &NencpairsI_s, &NencpairsJ_s, Encpairs2_d, idy + i, idx, icNB, test, 0, t);
				accG3<E>(a3_s[idy], b3_s[idy], x4i3, x4j, rcriti3, rcritvi3, rcritj, rcritvj, groupIndexi3, groupIndexj, &NencpairsI3_s, &NencpairsJ3_s, Encpairs2_d, idy + i, idx +2*N4, icNB, test, 0, t);


#endif
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

		b2_s[idy-Bl_2].x = 0.0;
		b2_s[idy-Bl_2].y = 0.0;
		b2_s[idy-Bl_2].z = 0.0;

		b4_s[idy-Bl_2].x = 0.0;
		b4_s[idy-Bl_2].y = 0.0;
		b4_s[idy-Bl_2].z = 0.0;

		__syncthreads();

		for(int i = 0; i < NB; i += Bl_2){
			if(idy-Bl_2 + i < N){
				double4 x4j = x4_d[idy-Bl_2 + i];
				double rcritj = rcrit_d[idy-Bl_2 + i];
				double rcritvj = rcritv_d[idy-Bl_2 + i];
#if G3 == 0
				acc<E>(a2_s[idy-Bl_2], b2_s[idy-Bl_2], x4i2, x4j, rcriti2, rcritvi2, rcritj, rcritvj, &NencpairsI2_s, &NencpairsJ2_s, Encpairs2_d, idy-Bl_2 + i, idx +N4, icNB, test, 0);
				acc<E>(a4_s[idy-Bl_2], b4_s[idy-Bl_2], x4i4, x4j, rcriti4, rcritvi4, rcritj, rcritvj, &NencpairsI4_s, &NencpairsJ4_s, Encpairs2_d, idy-Bl_2 + i, idx +3*N4, icNB, test, 0);
#else
				int groupIndexj = groupIndex_d[idy-Bl_2 + i];
				accG3<E>(a2_s[idy-Bl_2], b2_s[idy-Bl_2], x4i2, x4j, rcriti2, rcritvi2, rcritj, rcritvj, groupIndexi2, groupIndexj, &NencpairsI2_s, &NencpairsJ2_s, Encpairs2_d, idy-Bl_2 + i, idx +N4, icNB, test, 0, t);
				accG3<E>(a4_s[idy-Bl_2], b4_s[idy-Bl_2], x4i4, x4j, rcriti4, rcritvi4, rcritj, rcritvj, groupIndexi4, groupIndexj, &NencpairsI4_s, &NencpairsJ4_s, Encpairs2_d, idy-Bl_2 + i, idx +3*N4, icNB, test, 0, t);

#endif
			}
		}
	}
	__syncthreads();

	volatile double3 *a1 = a1_s;
	volatile double3 *a2 = a2_s;
	volatile double3 *a3 = a3_s;
	volatile double3 *a4 = a4_s;
	volatile double3 *b1 = b1_s;
	volatile double3 *b2 = b2_s;
	volatile double3 *b3 = b3_s;
	volatile double3 *b4 = b4_s;

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

			if(E <= 1){
				b1[idy].x += b1[idy + s].x;
				b1[idy].y += b1[idy + s].y;
				b1[idy].z += b1[idy + s].z;

				b2[idy].x += b2[idy + s].x;
				b2[idy].y += b2[idy + s].y;
				b2[idy].z += b2[idy + s].z;

				b3[idy].x += b3[idy + s].x;
				b3[idy].y += b3[idy + s].y;
				b3[idy].z += b3[idy + s].z;
				
				b4[idy].x += b4[idy + s].x;
				b4[idy].y += b4[idy + s].y;
				b4[idy].z += b4[idy + s].z;
			}
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

		if(E <= 1){
			b1[idy].x += b1[idy + 32].x;
			b1[idy].x += b1[idy + 16].x;
			b1[idy].x += b1[idy + 8].x;
			b1[idy].x += b1[idy + 4].x;
			b1[idy].x += b1[idy + 2].x;
			b1[idy].x += b1[idy + 1].x;

			b1[idy].y += b1[idy + 32].y;
			b1[idy].y += b1[idy + 16].y;
			b1[idy].y += b1[idy + 8].y;
			b1[idy].y += b1[idy + 4].y;
			b1[idy].y += b1[idy + 2].y;
			b1[idy].y += b1[idy + 1].y;

			b1[idy].z += b1[idy + 32].z;
			b1[idy].z += b1[idy + 16].z;
			b1[idy].z += b1[idy + 8].z;
			b1[idy].z += b1[idy + 4].z;
			b1[idy].z += b1[idy + 2].z;
			b1[idy].z += b1[idy + 1].z;
		}
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

			if(E <= 1){
				b2[idy-32].x += b2[idy + 32-32].x;
				b2[idy-32].x += b2[idy + 16-32].x;
				b2[idy-32].x += b2[idy + 8-32].x;
				b2[idy-32].x += b2[idy + 4-32].x;
				b2[idy-32].x += b2[idy + 2-32].x;
				b2[idy-32].x += b2[idy + 1-32].x;

				b2[idy-32].y += b2[idy + 32-32].y;
				b2[idy-32].y += b2[idy + 16-32].y;
				b2[idy-32].y += b2[idy + 8-32].y;
				b2[idy-32].y += b2[idy + 4-32].y;
				b2[idy-32].y += b2[idy + 2-32].y;
				b2[idy-32].y += b2[idy + 1-32].y;

				b2[idy-32].z += b2[idy + 32-32].z;
				b2[idy-32].z += b2[idy + 16-32].z;
				b2[idy-32].z += b2[idy + 8-32].z;
				b2[idy-32].z += b2[idy + 4-32].z;
				b2[idy-32].z += b2[idy + 2-32].z;
				b2[idy-32].z += b2[idy + 1-32].z;
			}
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

				if(E <= 1){
					b3[idy-64].x += b3[idy + 32-64].x;
					b3[idy-64].x += b3[idy + 16-64].x;
					b3[idy-64].x += b3[idy + 8-64].x;
					b3[idy-64].x += b3[idy + 4-64].x;
					b3[idy-64].x += b3[idy + 2-64].x;
					b3[idy-64].x += b3[idy + 1-64].x;

					b3[idy-64].y += b3[idy + 32-64].y;
					b3[idy-64].y += b3[idy + 16-64].y;
					b3[idy-64].y += b3[idy + 8-64].y;
					b3[idy-64].y += b3[idy + 4-64].y;
					b3[idy-64].y += b3[idy + 2-64].y;
					b3[idy-64].y += b3[idy + 1-64].y;

					b3[idy-64].z += b3[idy + 32-64].z;
					b3[idy-64].z += b3[idy + 16-64].z;
					b3[idy-64].z += b3[idy + 8-64].z;
					b3[idy-64].z += b3[idy + 4-64].z;
					b3[idy-64].z += b3[idy + 2-64].z;
					b3[idy-64].z += b3[idy + 1-64].z;
				}
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

					if(E <= 1){
						b4[idy-96].x += b4[idy + 32-96].x;
						b4[idy-96].x += b4[idy + 16-96].x;
						b4[idy-96].x += b4[idy + 8-96].x;
						b4[idy-96].x += b4[idy + 4-96].x;
						b4[idy-96].x += b4[idy + 2-96].x;
						b4[idy-96].x += b4[idy + 1-96].x;

						b4[idy-96].y += b4[idy + 32-96].y;
						b4[idy-96].y += b4[idy + 16-96].y;
						b4[idy-96].y += b4[idy + 8-96].y;
						b4[idy-96].y += b4[idy + 4-96].y;
						b4[idy-96].y += b4[idy + 2-96].y;
						b4[idy-96].y += b4[idy + 1-96].y;

						b4[idy-96].z += b4[idy + 32-96].z;
						b4[idy-96].z += b4[idy + 16-96].z;
						b4[idy-96].z += b4[idy + 8-96].z;
						b4[idy-96].z += b4[idy + 4-96].z;
						b4[idy-96].z += b4[idy + 2-96].z;
						b4[idy-96].z += b4[idy + 1-96].z;
					}
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
			Ne1 = atomicAdd(Nencpairs_d, NencpairsI_s);
		}
	}
	if(idy == 32){
		if(E >= 1){
			v4_d[idx + N4].x += a2[0].x * dtksq;
			v4_d[idx + N4].y += a2[0].y * dtksq;
			v4_d[idx + N4].z += a2[0].z * dtksq;
		}
		if(E <= 1){
			acck_d[idx + N4].x += b2[0].x * dtksq;
			acck_d[idx + N4].y += b2[0].y * dtksq;
			acck_d[idx + N4].z += b2[0].z * dtksq;
		}
		if(E <= 2){
			Ne2 = atomicAdd(Nencpairs_d, NencpairsI2_s);
		}
	}
	if(idy == 64){
		if(E >= 1){
			v4_d[idx + 2*N4].x += a3[0].x * dtksq;
			v4_d[idx + 2*N4].y += a3[0].y * dtksq;
			v4_d[idx + 2*N4].z += a3[0].z * dtksq;
		}
		if(E <= 1){
			acck_d[idx + 2*N4].x += b3[0].x * dtksq;
			acck_d[idx + 2*N4].y += b3[0].y * dtksq;
			acck_d[idx + 2*N4].z += b3[0].z * dtksq;
		}
		if(E <= 2){
			Ne3 = atomicAdd(Nencpairs_d, NencpairsI3_s);
		}
	}
	if(idy == 96){
		if(E >= 1){
			v4_d[idx + 3*N4].x += a4[0].x * dtksq;
			v4_d[idx + 3*N4].y += a4[0].y * dtksq;
			v4_d[idx + 3*N4].z += a4[0].z * dtksq;
		}
		if(E <= 1){
			acck_d[idx + 3*N4].x += b4[0].x * dtksq;
			acck_d[idx + 3*N4].y += b4[0].y * dtksq;
			acck_d[idx + 3*N4].z += b4[0].z * dtksq;
		}
		if(E <= 2){
			Ne4 = atomicAdd(Nencpairs_d, NencpairsI4_s);
		}
	}
	if(E <= 2){

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
////March 2014
//
// ****************************************
template <int Bl, int E>
__global__ void kicksmall_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double *rcrit_d, double *rcritv_d, int *groupIndex_d, const double dtksq, int N, int *Nencpairs_d, int2 *Encpairs_d, int2 *Encpairs2_d, double4 *x4small_d, double4 *v4small_d, double3 *accksmall_d, int Nsmall, double *rcritvsmall_d, int *groupIndexsmall_d, int *Nencpairssmall_d, int2 *Encpairssmall_d, int2 *Encpairssmall2_d, int NB, const int Nconst){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	__shared__ double3 a_s[Bl];
	__shared__ double3 b_s[Bl];
	int NencpairsI;
	int NencpairsJ;
	__shared__ int Nesmall;
	__shared__ int Ne;

	a_s[idy].x = 0.0;
	a_s[idy].y = 0.0;
	a_s[idy].z = 0.0;

	b_s[idy].x = 0.0;
	b_s[idy].y = 0.0;
	b_s[idy].z = 0.0;

	NencpairsI = 0;
	NencpairsJ = 0;

	if(idy == 0){
		Nesmall = 0;
		Ne = 0;
	}

	__syncthreads();

	if(id < N){
		
		double4 x4i = x4_d[id];
		double rcriti = rcrit_d[id];
		double rcritvi = rcritv_d[id];
		double test;
		for(int j = 0; j < N; ++j){
			acc<E + 10>(a_s[idy], b_s[idy], x4i, x4_d[j], rcriti, rcritvi, rcrit_d[j], rcritv_d[j], &NencpairsI, &NencpairsJ, Encpairs2_d, j, id, NB, test, 0);
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
			Ne = atomicAdd(Nencpairs_d, NencpairsI);
			for(int ii = 0; ii < NencpairsI; ++ii){
				Encpairs_d[ii + Ne] = Encpairs2_d[NB * id + ii];
			}
			Encpairs2_d[NB * id].x = NencpairsI;
			Encpairs2_d[NB * id + 1].x = NencpairsJ;
		}
	}
	else if(id < N + Nsmall){
		double rcritvi = rcritvsmall_d[id - N];
		double4 x4i = x4small_d[id - N];
		double test;
		for(int j = 0; j < N; ++j){
			acc<E + 20>(a_s[idy], b_s[idy], x4i, x4_d[j], 0.0 , rcritvi, rcrit_d[j], rcritv_d[j], &NencpairsI, &NencpairsJ, Encpairssmall2_d, j, id, N, test, Nconst);
		}
		if(E >= 1){
			v4small_d[id - N].x += a_s[idy].x * dtksq;
			v4small_d[id - N].y += a_s[idy].y * dtksq;
			v4small_d[id - N].z += a_s[idy].z * dtksq;
		}
		if(E <= 1){
			accksmall_d[id - N].x += b_s[idy].x * dtksq;
			accksmall_d[id - N].y += b_s[idy].y * dtksq;
			accksmall_d[id - N].z += b_s[idy].z * dtksq;
		}
		if(E <= 2){
			Nesmall = atomicAdd(Nencpairssmall_d, NencpairsI);
			for(int ii = 0; ii < NencpairsI; ++ii){
				Encpairssmall_d[ii + Nesmall] = Encpairssmall2_d[Nconst * (id - N) + ii];
			}
			Encpairssmall2_d[Nconst * (id - N)].x = NencpairsI;
		}
	}
}

// **************************************
//This kernel performs the seccond kick of the time step, in the case NB = 16 in the Multi Simulation Mode. NB is the next bigger number of N
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
//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
// ****************************************
template <int Bl, int Bl2, int Nmax, int E, int NB>
__global__ void KickM2_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double *rcrit_d, double *rcritv_d, int *Nencpairs_d, int2 *Encpairs_d, double dtksqKt, int *index_d, int NT, double *test_d){

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

	if(id < NT && id >= 0){
		st_s[idy] = index_d[id] / 100;
		x4_s[idy] = x4_d[id];
		//rcrit_s[idy] = rcrit_d[id];
		rcritv_s[idy] = rcritv_d[id];
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
		if((st_s[idy] - st_s[idy + j]) == 0 && x4_s[idy].w > 0.0 && x4_s[idy + j].w > 0.0){
			rcritv = fmax(rcritv_s[idy], rcritv_s[idy + j]);
			rcritv2 = rcritv * rcritv;
			rx = x4_s[idy + j].x - x4_s[idy].x;
			ry = x4_s[idy + j].y - x4_s[idy].y;
			rz = x4_s[idy + j].z - x4_s[idy].z;
			rsq = rx * rx + ry * ry + rz * rz;
			ir = 1.0 / sqrt(rsq);
			ir3 = ir * ir * ir;
			if(E <= 2){
				if(rsq < pc * rcritv2 && (x4_s[idy].w > 0.0 || x4_s[idy + j].w > 0.0)){  //prechecker
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

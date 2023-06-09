#ifndef KICK_H
#define KICK_H
#include "define.h"

// **************************************
// This function computes the term a = mi/rij^3 * Kij
// ****************************************
__device__ void accA(double3 &ac, double4 &x4i, double4 &x4j, double rcritvi, double rcritvj, int j, int i){
	if( i != j && x4i.w >= 0.0 && x4j.w > 0.0){
		volatile double rsq, ir, ir3, s;
		double3 r3ij;
		double rcritv, rcritv2;
		volatile double y, yy;

		r3ij.x = x4j.x - x4i.x;
		r3ij.y = x4j.y - x4i.y;
		r3ij.z = x4j.z - x4i.z;

		rsq = (r3ij.x*r3ij.x) + (r3ij.y*r3ij.y) + (r3ij.z*r3ij.z);
		rcritv = fmax(rcritvi, rcritvj);
		rcritv2 = rcritv * rcritv;

		ir = 1.0/sqrt(rsq);
		ir3 = ir*ir*ir;

		if(rsq >= 1.0 * rcritv2){
			s = x4j.w * ir3;
//if(i == 13723) printf("acc %d %d %.40g %.40g %.40g %.40g %.40g\n", i, j, rsq, ac.z, 0.0, s, x4j.w);
		}
		else{
			if(rsq <= 0.01 * rcritv2){
				s = 0.0;
//if(i == 13723) printf("acc %d %d %.40g %.40g %.40g %.40g %.40g\n", i, j, rsq, ac.z, 0.0, s, x4j.w);
			}
			else{
				y = (rsq * ir - 0.1 * rcritv)/(0.9*rcritv);
				yy = y * y;
				s = (ir3 * yy) / (2.0*yy - 2.0*y + 1.0) * x4j.w;
//if(i == 13723) printf("acc %d %d %.40g %.40g %.40g %.40g %.40g\n", i, j, rsq, ac.z, y, s, x4j.w);
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
//E = 10: a + b + precheck. (initial step) used for Test Particle Mode
//E = 11: a + b + precheck. used for Test Particle Mode
//E = 12: a + precheck. used for Test Particle Mode

//Authors: Simon Grimm
//August 2016
//****************************************
__device__ void acc_d(double3 &ac, double3 &b, double4 &x4i, double4 &x4j, double rcritvi, double rcritvj, int *NencpairsI, int2 *Encpairs2_d, int *EncFlag_d, int j, int i, const int NencMax, const int E){
	if( i != j && x4i.w >= 0.0 && x4j.w >= 0.0){
		volatile double rsq, ir, ir3, s, sb;
		double3 r3ij;
		double rcritv, rcritv2;
//		double rcrit, rcrit2;
		volatile double y, yy;

		r3ij.x = x4j.x - x4i.x;
		r3ij.y = x4j.y - x4i.y;
		r3ij.z = x4j.z - x4i.z;
		rsq = (r3ij.x*r3ij.x) + (r3ij.y*r3ij.y) + (r3ij.z*r3ij.z);
		rcritv = fmax(rcritvi, rcritvj);

//		rcrit2 = rcrit * rcrit;
		rcritv2 = rcritv * rcritv;
		if(E <= 2){	
			if(rsq < def_pc * rcritv2){	//prechecker
				int Ni = atomicAdd(NencpairsI, 1);
				if(Ni >= NencMax){
					atomicMax(&EncFlag_d[0], Ni);
				}
				else{
					Encpairs2_d[NencMax * i + Ni].y = j;
				}
//printf("Precheck acc_d %d %d %d\n", i, j, Ni);
			}
		}
		if(E <= 12 && E >= 10){ //prechecker used for Test Particle Mode
			if(rsq < def_pc * rcritv2 && *NencpairsI < NencMax){	//prechecker
//printf("Precheck %d %d\n", i, j);
				Encpairs2_d[NencMax * i + *NencpairsI].y = j;
				*NencpairsI += 1;
			}
		}
		ir = 1.0/sqrt(rsq);
		ir3 = ir*ir*ir;
		sb = 0.0;

		if(rsq >= 1.0 * rcritv2){
			s = x4j.w * ir3;
			if( rsq >= def_pc * rcritv2) sb = s;
		}
		else{
			if(rsq <= 0.01 * rcritv2){
				s = 0.0;
			}
			else{
				y = (rsq * ir - 0.1 * rcritv)/(0.9*rcritv);
				yy = y * y;
				s = (ir3 * yy) / (2.0*yy - 2.0*y + 1.0) * x4j.w;
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
//float version
__device__ void acc_df(float3 &ac, float3 &b, float4 &x4i, float4 &x4j, float rcritvi, float rcritvj, int *NencpairsI, int2 *Encpairs2_d, int *EncFlag_d, int j, int i, const int NencMax, const int E){
	if( i != j && x4i.w >= 0.0f && x4j.w >= 0.0f){
		volatile float rsq, ir, ir3, s, sb;
		float3 r3ij;
		float rcritv, rcritv2;
//		float rcrit, rcrit2;
		volatile float y, yy;

		r3ij.x = x4j.x - x4i.x;
		r3ij.y = x4j.y - x4i.y;
		r3ij.z = x4j.z - x4i.z;

		rsq = (r3ij.x*r3ij.x) + (r3ij.y*r3ij.y) + (r3ij.z*r3ij.z);
		rcritv = fmax(rcritvi, rcritvj);

//		rcrit2 = rcrit * rcrit;
		rcritv2 = rcritv * rcritv;
		if(E <= 2){
			if(rsq < def_pcf * rcritv2){	//prechecker
				int Ni = atomicAdd(NencpairsI, 1);
				if(Ni >= NencMax){
					atomicMax(&EncFlag_d[0], Ni);
				}
				else{
					Encpairs2_d[NencMax * i + Ni].y = j;
				}
//printf("Precheck %d %d %d %d %g %g %g\n", i, j, Ni, NencMax, rsq, rcritvi, rcritvj);
			}
		}
		if(E <= 12 && E >= 10){ //prechecker used for Test Particle Mode
			if(rsq < def_pcf * rcritv2 && *NencpairsI < NencMax){	//prechecker
//printf("Precheck %d %d\n", i, j);
				Encpairs2_d[NencMax * i + *NencpairsI].y = j;
				*NencpairsI += 1;
			}
		}
		ir = 1.0f/sqrtf(rsq);
		ir3 = ir*ir*ir;
		sb = 0.0f;

		if(rsq >= 1.0f * rcritv2){
			s = x4j.w * ir3;
			if( rsq >= def_pcf * rcritv2) sb = s;
		}
		else{
			if(rsq <= 0.01f * rcritv2){
				s = 0.0f;
			}
			else{
				y = (rsq * ir - 0.1f * rcritv)/(0.9f*rcritv);
				yy = y * y;
				s = (ir3 * yy) / (2.0f*yy - 2.0f*y + 1.0f) * x4j.w;
			}
		}
		ac.x += __fmul_rn(r3ij.x, s);
		ac.y += __fmul_rn(r3ij.y, s);
		ac.z += __fmul_rn(r3ij.z, s);

		if(E % 10 != 2){
			b.x += __fmul_rn(r3ij.x, sb);
			b.y += __fmul_rn(r3ij.y, sb);
			b.z += __fmul_rn(r3ij.z, sb);
		}
//printf("%d %d %g %g Kick\n", i, j, s, ac.x);
	}
}

// ******************************************************
// Version of acc which is called from the recursive symplectic sub step method
// Author: Simon Grimm
// Janury 2019
// ******************************************************
__device__ void accS(double4 x4i, double4 x4j, double3 &ac, double *rcritv_d, int &NencpairsI, int2 *Encpairs2_d, const int i, const int j, const int NconstT, const int NencMax, const int SLevel, const int SLevels, const int E){

	if(i != j){

		double3 r3;
		double rsq;
		double ir, ir3;
		double y, yy;
		double rcritv, rcritv2;
		double rcritvi, rcritvj;

		r3.x = x4j.x - x4i.x;
		r3.y = x4j.y - x4i.y;
		r3.z = x4j.z - x4i.z;

		rsq = r3.x*r3.x + r3.y*r3.y + r3.z*r3.z + 1.0e-30;
		ir = 1.0/sqrt(rsq);
		ir3 = ir * ir * ir;

		double s = x4j.w * ir3 * def_ksq;

		for(int l = 0; l < SLevel; ++l){
		// (1 - K) factors of the previous levels 
//if(i == 0) printf(" (1-K%d)  ",l);
			rcritvi = rcritv_d[i + NconstT * l];
			rcritvj = rcritv_d[j + NconstT * l];

			rcritv = fmax(rcritvi, rcritvj);
			rcritv2 = rcritv * rcritv;

			if(rsq < 1.0 * rcritv2){
				if(rsq <= 0.01 * rcritv2){
					s *= 1.0;
				}
				else{
					y = (rsq * ir - 0.1 * rcritv)/(0.9*rcritv);
					yy = y * y;
					s *= (1.0 - yy / (2.0*yy - 2.0*y + 1.0));
				}
			}
			else s = 0.0;
		}

		
		if(SLevel < SLevels){
		//if(SLevel < SLevels - 1){ //<- use that for a complete last level Kick without BS
		// K factor of the current level
//if(i == 0) printf(" K%d  ",SLevel);
			rcritvi = rcritv_d[i + NconstT * SLevel];
			rcritvj = rcritv_d[j + NconstT * SLevel];

			rcritv = fmax(rcritvi, rcritvj);

			rcritv2 = rcritv * rcritv;


			if(rsq >= 1.0 * rcritv2){
				s *= 1.0;
			}
			else{
				if(rsq <= 0.01 * rcritv2){
					s = 0.0;

				}
				else{
					y = (rsq * ir - 0.1 * rcritv)/(0.9*rcritv);
					yy = y * y;
					s *= yy / (2.0*yy - 2.0*y + 1.0);

				}
			}
			//prechecker
			if(E == 0 || E == 2){	
				if(rsq < def_pc * rcritv2){	//prechecker
//printf("Precheck %d %d\n", i, j);
					Encpairs2_d[NencMax * i + NencpairsI].y = j;
					++NencpairsI;
				}
			}
		}

//if(i == 0 && j == 1) printf("\n");
		ac.x += __dmul_rn(r3.x, s);
		ac.y += __dmul_rn(r3.y, s);
		ac.z += __dmul_rn(r3.z, s);
//printf("%.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g\n", x4i.w, x4i.x, x4i.y, x4i.z, x4j.w, x4j.x, x4j.y, x4j.z);
	}
}

// *********************************
//Only here for testing
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
// ********************************
__device__ void accG3(double3 &ac, double3 &b, double4 &x4i, double4 &x4j, double rcritvi, double rcritvj, int groupIndexi, int groupIndexj, int *NencpairsI, int *NencpairsJ, int2 *Encpairs_d, int j, int i, const int NconstT, const int NencMax, double t, const int E){
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
#if def_G3 == 1
			if(((groupIndexi == groupIndexj && groupIndexi >= 0 && groupIndexi < NconstT) || rsq < def_pc * rcritv) && x4i.w > 0.0 && x4j.w > 0.0){	
#else
			if(((groupIndexi == groupIndexj && groupIndexi >= 0 && groupIndexi < NconstT) || B > def_G3Limit2) && x4i.w > 0.0 && x4j.w > 0.0){	

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

		if(groupIndexi == groupIndexj && groupIndexi >= 0 && groupIndexi < NconstT) s = 0.0;

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
//It reuses the values from the second kick in the previous time step.

//Authors: Simon Grimm
//November 2016
// ****************************************
__global__ void kick32BM_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double3 *ab_d, int *index_d, const int N, double *dt_d, double Kt, const int Nstart){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy + Nstart;
	if(id < N + Nstart){
		int st = index_d[id] / def_MaxIndex;
		double dtksqKt = dt_d[st] * Kt * def_ksq;

		double3 a = acck_d[id];
		if(x4_d[id].w >= 0.0){
			v4_d[id].x += __dmul_rn(a.x, dtksqKt);
			v4_d[id].y += __dmul_rn(a.y, dtksqKt);
			v4_d[id].z += __dmul_rn(a.z, dtksqKt);
//printf("Kick32BM %d %g %g %g %g\n", id, acck_d[id].x, acck_d[id].y, acck_d[id].z, v4_d[id].x);
		}
		ab_d[id].x = a.x;
		ab_d[id].y = a.y;
		ab_d[id].z = a.z;
	}
}
__global__ void kick32BMSimple_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double3 *ab_d, int *index_d, int N, double *dt_d, double Kt, double *time_d, double *idt_d, double *ict_d, long long timeStep, const int Nst, const int Nstart){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy + Nstart;
	if(id < N + Nstart){
		int st = index_d[id] / def_MaxIndex;
		double dtksqKt = dt_d[st] * Kt * def_ksq;

		double3 a = acck_d[id];
		if(x4_d[id].w >= 0.0){
			v4_d[id].x += __dmul_rn(a.x, dtksqKt);
			v4_d[id].y += __dmul_rn(a.y, dtksqKt);
			v4_d[id].z += __dmul_rn(a.z, dtksqKt);
//printf("KickB %d %g %g %g %g\n", id, acck_d[id].x, acck_d[id].y, acck_d[id].z, v4_d[id].x);
		}
		if(id < Nst) time_d[id] = timeStep * idt_d[id] + ict_d[id] * 365.25;
		ab_d[id].x = a.x;
		ab_d[id].y = a.y;
		ab_d[id].z = a.z;
	}
}

// **************************************
//This kernel performs the kick for a backup step.
//It reuses the values from the kick in the original time step.

//Authors: Simon Grimm
//November 2016
// ****************************************
__global__ void kick32C_kernel(double4 *x4_d, double4 *v4_d, double3 *ab_d, const int N, double dtksq){

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < N){
		double3 a = ab_d[id];
		if(x4_d[id].w >= 0.0){
			v4_d[id].x += __dmul_rn(a.x, dtksq);
			v4_d[id].y += __dmul_rn(a.y, dtksq);
			v4_d[id].z += __dmul_rn(a.z, dtksq);
//printf("KickB %d %g %g %g %g\n", id, acck_d[id].x, acck_d[id].y, acck_d[id].z, v4_d[id].x);
		}
	}
}

__global__ void kick32BMTTV_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double3 *ab_d, int *index_d, const int N, double *dt_d, double Kt, double2 *Msun_d, int *Ntransit_d, int *Transit_d, const int Nstart){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy + Nstart;
	if(id < N + Nstart){
		int st = index_d[id] / def_MaxIndex;
		double dtksqKt = dt_d[st] * Kt * def_ksq;
		double dt = dt_d[st];
		double Msun = Msun_d[st].x;
		double Rsun = Msun_d[st].y;
		double3 a = acck_d[id];
		double4 x4i = x4_d[id];
		double4 v4i = v4_d[id];
		if(x4i.w >= 0.0){
			v4_d[id].x += __dmul_rn(a.x, dtksqKt);
			v4_d[id].y += __dmul_rn(a.y, dtksqKt);
			v4_d[id].z += __dmul_rn(a.z, dtksqKt);
//printf("KickB %d %.20e %.20e %.20e %.20e %.20e %.20e\n", id, a.x, a.y, a.z, v4_d[id].x * dayUnit, v4_d[id].y * dayUnit, v4_d[id].z * dayUnit);
//printf("KickBMTTV %d %.20e %.20e %.20e %.20e %.20e %.20e %.20e\n", id, a.x, a.y, a.z, v4_d[id].x, v4_d[id].y, v4_d[id].z, dtksqKt);

			//calculate acceleration from the central star
			double rsq = x4i.x*x4i.x + x4i.y*x4i.y + x4i.z*x4i.z + 1.0e-30;
			double ir = 1.0 / sqrt(rsq);
			double ir3 = ir * ir * ir;
			double s = - def_ksq * Msun * ir3;

			a.x += s * x4i.x;
			a.y += s * x4i.y;
			a.z += s * x4i.z;

			double g = x4i.x * v4i.x + x4i.y * v4i.y;
			double gd = v4i.x * v4i.x + v4i.y * v4i.y + x4i.x * a.x + x4i.y * a.y;
			double rsky = sqrt(x4i.x * x4i.x + x4i.y * x4i.y);
			double v = sqrt(v4i.x * v4i.x + v4i.y * v4i.y);
			double R = Rsun + v4i.w;

//if(id % 8 == 6) printf("TTV %d g %g gd %g g/gd %.20g x %.10g y %.10g z %.10g dt %.20g rsky %g R %g R+ %g\n", id, g, gd, -g / gd, x4i.x, x4i.y, x4i.z, dt, rsky, R, R + v * dt);
			if(dt > 0){
				if(x4i.z > 0.0 && gd > 0.0 && fabs(g / gd) < 3.5 * fabs(dt) && rsky < R + v * fabs(dt)){

					if(g <= 0.0){
						int Nt = atomicAdd(Ntransit_d, 1);
						Nt = min(Nt, def_NtransitMax - 1);
						Transit_d[Nt] = id;
//if(id % 8 == 6) printf("check Transit %d %d\n", id, Nt);
					}
				}
			}
			else{
				if(x4i.z > 0.0 && gd > 0.0 && fabs(g / gd) < 3.5 * fabs(dt) && rsky < R + v * fabs(dt)){
					if(g >= 0.0){
						int Nt = atomicAdd(Ntransit_d, 1);
						Nt = min(Nt, def_NtransitMax - 1);
						Transit_d[Nt] = id;
					}
				}
			}

		}
		ab_d[id].x = a.x;
		ab_d[id].y = a.y;
		ab_d[id].z = a.z;
	}
}
__global__ void kick32BMTTVSimple_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double3 *ab_d, int *index_d, const int N, double *dt_d, double Kt, double2 *Msun_d, int *Ntransit_d, int *Transit_d, double *time_d, double *idt_d, double *ict_d, long long timeStep, int Nst, const int Nstart){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy + Nstart;
	if(id < N + Nstart){
		int st = index_d[id] / def_MaxIndex;
		double dtksqKt = dt_d[st] * Kt * def_ksq;
		double dt = dt_d[st];
		double Msun = Msun_d[st].x;
		double Rsun = Msun_d[st].y;
		double3 a = acck_d[id];
		double4 x4i = x4_d[id];
		double4 v4i = v4_d[id];
		if(x4i.w >= 0.0){
			v4_d[id].x += __dmul_rn(a.x, dtksqKt);
			v4_d[id].y += __dmul_rn(a.y, dtksqKt);
			v4_d[id].z += __dmul_rn(a.z, dtksqKt);
//printf("KickB %d %.20e %.20e %.20e %.20e %.20e %.20e\n", id, a.x, a.y, a.z, v4_d[id].x * dayUnit, v4_d[id].y * dayUnit, v4_d[id].z * dayUnit);

			//calculate acceleration from the central star
			double rsq = x4i.x*x4i.x + x4i.y*x4i.y + x4i.z*x4i.z + 1.0e-30;
			double ir = 1.0 / sqrt(rsq);
			double ir3 = ir * ir * ir;
			double s = - def_ksq * Msun * ir3;

			a.x += s * x4i.x;
			a.y += s * x4i.y;
			a.z += s * x4i.z;

			double g = x4i.x * v4i.x + x4i.y * v4i.y;
			double gd = v4i.x * v4i.x + v4i.y * v4i.y + x4i.x * a.x + x4i.y * a.y;
			double rsky = sqrt(x4i.x * x4i.x + x4i.y * x4i.y);
			double v = sqrt(v4i.x * v4i.x + v4i.y * v4i.y);
			double R = Rsun + v4i.w;

//printf("TTV %d g %g gd %g g/gd %.20g x %.10g y %.10g z %.10g dt %.20g rsky %g R %g R+ %g\n", id, g, gd, -g / gd, x4i.x, x4i.y, x4i.z, dt, rsky, R, R + v * dt);
			if(dt > 0){
				if(x4i.z > 0.0 && gd > 0.0 && fabs(g / gd) < 3.5 * fabs(dt) && rsky < R + v * fabs(dt)){

					if(g <= 0.0){
						int Nt = atomicAdd(Ntransit_d, 1);
						Nt = min(Nt, def_NtransitMax - 1);
						Transit_d[Nt] = id;
					}
				}
			}
			else{
				if(x4i.z > 0.0 && gd > 0.0 && fabs(g / gd) < 3.5 * fabs(dt) && rsky < R + v * fabs(dt)){
					if(g >= 0.0){
						int Nt = atomicAdd(Ntransit_d, 1);
						Nt = min(Nt, def_NtransitMax - 1);
						Transit_d[Nt] = id;
					}
				}
			}

		}
		ab_d[id].x = a.x;
		ab_d[id].y = a.y;
		ab_d[id].z = a.z;
		if(id < Nst) time_d[id] = timeStep * idt_d[id] + ict_d[id] * 365.25;

	}
}

// *******************************************
//This kernel is used to sort the close encounter list, to be able to reproduce simulations exactly
//It shoud be used only for debugging or special cases.

//Authors: Simon Grimm
//August 2016
// *********************************************
__global__ void Sortb_kernel(int2 *Encpairs2_d, const int Nstart, const int N, const int NencMax){

	int id = blockIdx.x * blockDim.x + threadIdx.x + Nstart;	

	if(id < N){
		int NI = Encpairs2_d[id * NencMax].x;
		NI = min(NI, NencMax);

		int stop = 0;
		while(stop == 0){
			stop = 1;
			for(int i = 0; i < NI - 1; ++i){
				int jj = Encpairs2_d[id * NencMax + i].y;
				int jjnext = Encpairs2_d[id * NencMax + i + 1].y;
//if(id == 13723) printf("sort %d %d %d %d\n", id, NI, jj, jjnext);
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

__global__ void SortSb_kernel(int *Encpairs3_d, int *Nencpairs3_d, const int N, const int NencMax){

	int idd = blockIdx.x * blockDim.x + threadIdx.x;

	if(idd < Nencpairs3_d[0]){
		int id = Encpairs3_d[idd * NencMax + 1];
		if(id >= 0 && id < N){
			int NI = Encpairs3_d[id * NencMax + 2];
			NI = min(NI, NencMax);

			int stop = 0;
			while(stop == 0){
				stop = 1;
				for(int i = 0; i < NI - 1; ++i){
					int jj = Encpairs3_d[id * NencMax + i + 4];
					int jjnext = Encpairs3_d[id * NencMax + i + 1 + 4];
//printf("sortSb %d %d %d %d\n", id, NI, jj, jjnext);
					if(jjnext < jj){
						//swap
						Encpairs3_d[id * NencMax + i + 4] = jjnext;
						Encpairs3_d[id * NencMax + i + 1 + 4] = jj;
						stop = 0;

					}
				}
			}
		}
	}
}



// **************************************
//This kernel performs the first kick of the time step, in the case of close interactions.
//It reuses the values from the second kick in the previous time step, and adds the terms aij*dt*Kij for all
//the bodies involved in a close encounter.
//NI is the number of bodies involved in a close encounter with body i 

//Authors: Simon Grimm
//December 2016

// EE = 0  Do not update velocities
// EE >= 1, Do update velocities
// ****************************************
__global__ void kick32Ab_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double3 *ab_d, double *rcritv_d, const double dtksq, int *Nencpairs_d, int2 *Encpairs2_d, const int Nstart, const int N, const int NencMax, const int EE){

	int id = blockIdx.x * blockDim.x + threadIdx.x + Nstart;	

	if(id < N){
		double3 a = {0.0, 0.0, 0.0};
		double4 x4i = x4_d[id];
		if(x4i.w >= 0.0){
			if(Nencpairs_d[0] > 0){
				int NI = Encpairs2_d[id * NencMax].x;
				NI = min(NI, NencMax);
//if(NI > 0) printf("NI %d %d\n", id, NI);
				double rcritvi = rcritv_d[id];
				for(int i = 0; i < NI; ++i){
					int jj = Encpairs2_d[id * NencMax + i].y;
					double4 x4j = x4_d[jj];
//printf("AI %d %d %d %.40g %.40g %.40g %.20g %.20g %.40g\n", id, jj, NI, x4i.x, x4j.x, v4_d[id].z, x4j.x, x4j.w, a.z);
					double rcritvj = rcritv_d[jj];
					accA(a, x4i, x4j, rcritvi, rcritvj, jj, id);
				}
				__syncthreads();
			
				double3 aa;
				aa.x = a.x + acck_d[id].x;
				aa.y = a.y + acck_d[id].y;
				aa.z = a.z + acck_d[id].z;

				if(EE >= 1){
					v4_d[id].x += __dmul_rn(aa.x, dtksq);
					v4_d[id].y += __dmul_rn(aa.y, dtksq);
					v4_d[id].z += __dmul_rn(aa.z, dtksq);
				}
				ab_d[id] = aa;
			}
			else{
			
				double3 a = acck_d[id];
				if(EE >= 1){
					v4_d[id].x += __dmul_rn(a.x, dtksq);
					v4_d[id].y += __dmul_rn(a.y, dtksq);
					v4_d[id].z += __dmul_rn(a.z, dtksq);
				}
//printf("KickB %d %.16e %.16e %.16e %.16e %.16e %.16e\n", id, acck_d[id].x, acck_d[id].y, acck_d[id].z, v4_d[id].x * dayUnit, v4_d[id].y * dayUnit, v4_d[id].z * dayUnit);
				ab_d[id] = a;
			}

		}
//if(id == 50) printf("K %d %.40g %.40g %.40g %.20g %.20g %.20g %.20g\n", id, v4_d[id].x, v4_d[id].y, v4_d[id].z, a.x, a.y, acck_d[id].x, acck_d[id].y);
	}
}

// *****************************************************
// This kernel collects the Encpairs2 information from multiple GPUs into the main array.
//
// Author: Simon Grimm
// November 2022
// ********************************************************
__global__ void CollectGPUsAb_kernel(double4 *x4_dj, int2 *Encpairs2_dj, int2 *Encpairs2_d, const int Nstart, const int N, const int NencMax){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy + Nstart;	

	if(id < N){
		double4 x4i = x4_dj[id];
		if(x4i.w >= 0.0){
			int NI = Encpairs2_dj[id * NencMax].x;
			Encpairs2_d[id * NencMax].x = NI;
			NI = min(NI, NencMax);
			for(int i = 0; i < NI; ++i){
				int jj = Encpairs2_dj[id * NencMax + i].y;
				Encpairs2_d[id * NencMax + i].y = jj;
			}
		}
	}
}

// *****************************************************
// Version of the Kick kernel which is called from the recursive symplectic sub step method
//
// Author: Simon Grimm
// January 2019
// ********************************************************
__global__ void kickS_kernel(double4 *x4_d, double4 *v4_d, double4 *xold_d, double4 *vold_d, double *rcritv_d, const double dtksq, int *Nencpairs_d, int2 *Encpairs_d, int2 *Encpairs2_d, int *Nencpairs3_d, int *Encpairs3_d, const int N, const int NconstT, const int NencMax, const int SLevel, const int SLevels, const int E){

	int idd = blockIdx.x * blockDim.x + threadIdx.x;	

	if(idd < Nencpairs3_d[0]){
		double3 a = {0.0, 0.0, 0.0};
		int id = Encpairs3_d[idd * NencMax + 1];

		if(id >= 0 && id < N){
			double4 x4i = xold_d[id];
			double4 v4i = vold_d[id];
			int NencpairsI = 0;
			if(x4i.w >= 0.0){
				__syncthreads();
				int NI = Encpairs3_d[id * NencMax + 2];
				NI = min(NI, NencMax);
//if(NI > 0) printf("NI %d %d %d\n", idd, id, NI);
				for(int i = 0; i < NI; ++i){
					int jj = Encpairs3_d[id * NencMax + i + 4];
					double4 x4j = xold_d[jj];
					if(x4j.w >= 0.0){
//if(E == 0) printf("AI %d %d %d %d %.40g %.40g %.40g %.40g\n", idd, id, jj, NI, x4i.x, x4j.x, v4_d[id].z, a.z);
						accS(x4i, x4j, a, rcritv_d, NencpairsI, Encpairs2_d, id, jj, NconstT, NencMax, SLevel, SLevels, E);
					}
				}
				__syncthreads();
				double3 aa;
				aa.x = __dmul_rn(a.x, dtksq);
				aa.y = __dmul_rn(a.y, dtksq);
				aa.z = __dmul_rn(a.z, dtksq);

				v4i.x += aa.x;
				v4i.y += aa.y;
				v4i.z += aa.z;

				if(E == 0){
					x4_d[id] = x4i;
				}
				v4_d[id] = v4i;
//printf("KickS %d %d %.20g %.20g %.20g\n", idd, id, v4_d[id].x, v4_d[id].y, v4_d[id].z);
				if(E == 0 || E == 2){
					for(int i = 0; i < NencpairsI; ++i){
						int jj = Encpairs2_d[id * NencMax + i].y;
						if(id > jj){
#if def_CPU == 0
							int Ne = atomicAdd(Nencpairs_d, 1);
#else
							int Ne;
							#pragma omp atomic capture
							Ne = Nencpairs_d[0]++;
#endif
//printf("KickS %d %d %d\n", Ne, id, jj);
							Encpairs_d[Ne].x = id;
							Encpairs_d[Ne].y = jj;
						}
					}
				}
			}
		}
	}
}


// **************************************
//This kernel performs the first kick of the time step, in the case of close interactions.
//It reuses the values from the second kick in the previous time step, and adds the terms aij*dt*Kij for all
//the bodies involved in a close encounter.
//NI is the number of bodies involved in a close encounter with body i 
//It checks if a transit occurs in the next time step

//Authors: Simon Grimm
//December 2016
// ****************************************
__global__ void kick32ATTV_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double3 *ab_d, double *rcritv_d, const double dtksq, int *Nencpairs_d, int2 *Encpairs2_d, const int N, const int NencMax, double t, double dt, double Msun, double Rsun, int *Ntransit_d, int *Transit_d){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;	

	double3 a;

	a.x = 0.0;
	a.y = 0.0;
	a.z = 0.0;

	if(id < N){
		double4 x4i = x4_d[id];
//if(NI > 0) printf("NI %d %d\n", id, NI);
		if(x4i.w >= 0.0){
			double3 aa;
			double4 v4i = v4_d[id];
			if(Nencpairs_d[0] > 0){
				int NI = Encpairs2_d[id * NencMax].x;
				NI = min(NI, NencMax);
				double rcritvi = rcritv_d[id];
				for(int i = 0; i < NI; ++i){
					int jj = Encpairs2_d[id * NencMax + i].y;
					double4 x4j = x4_d[jj];
//printf("AI %d %d %d %.40g %.40g %.40g %.40g\n", id, jj, NI, x4i.x, x4j.x, v4_d[id].z, v4_d[jj].z);
					double rcritvj = rcritv_d[jj];
					accA(a, x4i, x4j, rcritvi, rcritvj, jj, id);
				}
				__syncthreads();
		
				aa.x = a.x + acck_d[id].x;
				aa.y = a.y + acck_d[id].y;
				aa.z = a.z + acck_d[id].z;

				v4_d[id].x += __dmul_rn(aa.x, dtksq);
				v4_d[id].y += __dmul_rn(aa.y, dtksq);
				v4_d[id].z += __dmul_rn(aa.z, dtksq);
				ab_d[id] = aa;
			}
			else{
				aa = acck_d[id];
				v4_d[id].x += __dmul_rn(aa.x, dtksq);
				v4_d[id].y += __dmul_rn(aa.y, dtksq);
				v4_d[id].z += __dmul_rn(aa.z, dtksq);
				ab_d[id] = aa;
			}

			//calculate acceleration from the central star
			double rsq = x4i.x*x4i.x + x4i.y*x4i.y + x4i.z*x4i.z + 1.0e-30;
			double ir = 1.0 / sqrt(rsq);
			double ir3 = ir * ir * ir;
			double s = - def_ksq * Msun * ir3;

			aa.x += s * x4i.x;
			aa.y += s * x4i.y;
			aa.z += s * x4i.z;

			double g = x4i.x * v4i.x + x4i.y * v4i.y;
			double gd = v4i.x * v4i.x + v4i.y * v4i.y + x4i.x * aa.x + x4i.y * aa.y;
			double rsky = sqrt(x4i.x * x4i.x + x4i.y * x4i.y);
			double v = sqrt(v4i.x * v4i.x + v4i.y * v4i.y);
			double R = Rsun + v4i.w;

//printf("TTV %d g %g gd %g g/gd %.20g x %.10g y %.10g z %.10g dt %.20g rsky %g R %g R+ %g\n", id, g, gd, -g / gd, x4i.x, x4i.y, x4i.z, dt, rsky, R, R + v * dt);
			if(dt > 0){
				if(x4i.z > 0.0 && gd > 0.0 && fabs(g / gd) < 3.5 * fabs(dt) && rsky < R + v * fabs(dt)){
					if(g <= 0.0){
						int Nt = atomicAdd(Ntransit_d, 1);
						Nt = min(Nt, def_NtransitMax - 1);
						Transit_d[Nt] = id;
					}
				}
			}
			else{
				if(x4i.z > 0.0 && gd > 0.0 && fabs(g / gd) < 3.5 * fabs(dt) && rsky < R + v * fabs(dt)){
					if(g >= 0.0){
						int Nt = atomicAdd(Ntransit_d, 1);
						Nt = min(Nt, def_NtransitMax - 1);
						Transit_d[Nt] = id;
					}
				}
			}
		}
//if(id == 25) printf("K %d %.40g %.40g %.40g %.40g %.40g\n", id, v4_d[id].x, v4_d[id].y, v4_d[id].z, a.z, acck_d[id].z);
	}
}

// **************************************
//This kernel performs the second kick of the time step, in the case NB = 16. NB is the next bigger number of N
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
//Authors: Simon Grimm
//April 2019
//****************************************
__global__ void kick16c_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double *rcritv_d, const double dtksq, int *Nencpairs_d, int2 *Encpairs_d, int2 *Encpairs2_d, int *EncFlag_d, const int NencMax, const int N, const int E){
	int idy = threadIdx.x;
	int idx = blockIdx.x;

	if(idx < N){

		__shared__ int NencpairsI_s;

		double4 x4i = x4_d[idx];
		double rcritvi = rcritv_d[idx];

		double3 a = {0.0, 0.0, 0.0};
		double3 b = {0.0, 0.0, 0.0};

		if(idy == 0){
			NencpairsI_s = 0;
		}

		__syncthreads();


		if(idy < N){
			double4 x4j = x4_d[idy];
			double rcritvj = rcritv_d[idy];
			acc_d(a, b, x4i, x4j, rcritvi, rcritvj, &NencpairsI_s, Encpairs2_d, EncFlag_d, idy, idx, NencMax, E); 
//printf("Kick1 %d %d %g %g %.20g %.20g %.20g\n", idx, idy, x4i.w, x4j.w, a.x, a.y, a.z);
		}
//if(idx == 0) printf("Kick %d %d %.20g %.20g %.20g\n", idx, idy, a.x, a.y, a.z);

		__syncthreads();

		for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
			a.x += __shfl_xor_sync(0xffffffff, a.x, i, warpSize);
			a.y += __shfl_xor_sync(0xffffffff, a.y, i, warpSize);
			a.z += __shfl_xor_sync(0xffffffff, a.z, i, warpSize);

			b.x += __shfl_xor_sync(0xffffffff, b.x, i, warpSize);
			b.y += __shfl_xor_sync(0xffffffff, b.y, i, warpSize);
			b.z += __shfl_xor_sync(0xffffffff, b.z, i, warpSize);
#else
			a.x += __shfld_xor(a.x, i);
			a.y += __shfld_xor(a.y, i);
			a.z += __shfld_xor(a.z, i);

			b.x += __shfld_xor(b.x, i);
			b.y += __shfld_xor(b.y, i);
			b.z += __shfld_xor(b.z, i);
#endif
//if(idx == 0 && i >= 16) printf("KickA %d %d %.20g %.20g\n", idy, i, a.x, b.x);
		}

		if(idy == 0){
//printf("Kick %d %g %.20g %.20g %.20g | %.20g %.20g %.20g\n", idx, x4_d[idx].w, a.x, a.y, a.z, b.x, b.y, b.z);
			if(E >= 1){
				v4_d[idx].x += __dmul_rn(a.x, dtksq);
				v4_d[idx].y += __dmul_rn(a.y, dtksq);
				v4_d[idx].z += __dmul_rn(a.z, dtksq);
//printf("Kick %d %g %g %g %g\n", idx, x4_d[idx].w, __dmul_rn(a.x, dtksq), __dmul_rn(a.y, dtksq), __dmul_rn(a.z, dtksq));
			}
			if(E <= 1){
				acck_d[idx].x = b.x;
				acck_d[idx].y = b.y;
				acck_d[idx].z = b.z;
			}
		}
		if(E <= 2){
			if(idy == 0){
				Encpairs2_d[NencMax * idx].x = NencpairsI_s;
			}
			if(idy < NencpairsI_s && idy < NencMax){
				int jj = Encpairs2_d[idx * NencMax + idy].y;
				if(idx < jj){
					int Ne = atomicAdd(Nencpairs_d, 1);
					Encpairs_d[Ne].x = idx;
					Encpairs_d[Ne].y = jj;
				}
			}
		}
	}
}
//float Version
__global__ void kick16cf_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double *rcritv_d, const double dtksq, int *Nencpairs_d, int2 *Encpairs_d, int2 *Encpairs2_d, int *EncFlag_d, const int NencMax, const int N, const int E){
	int idy = threadIdx.x;
	int idx = blockIdx.x;

	if(idx < N){

		__shared__ int NencpairsI_s;

		float4 x4i;
		x4i.x = x4_d[idx].x;
		x4i.y = x4_d[idx].y;
		x4i.z = x4_d[idx].z;
		x4i.w = x4_d[idx].w;
		float rcritvi = rcritv_d[idx];

		float3 a = {0.0f, 0.0f, 0.0f};
		float3 b = {0.0f, 0.0f, 0.0f};

		if(idy == 0){
			NencpairsI_s = 0;
		}

		__syncthreads();


		if(idy < N){
			float4 x4j;
			x4j.x = x4_d[idy].x;
			x4j.y = x4_d[idy].y;
			x4j.z = x4_d[idy].z;
			x4j.w = x4_d[idy].w;
			float rcritvj = rcritv_d[idy];
			acc_df(a, b, x4i, x4j, rcritvi, rcritvj, &NencpairsI_s, Encpairs2_d, EncFlag_d, idy, idx, NencMax, E); 
//printf("Kick1 %d %d %g %g %g %g %g\n", idx, idy, x4i.w, x4j.w, a.x, a.y, a.z);
		}

		__syncthreads();

		for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
			a.x += __shfl_xor_sync(0xffffffff, a.x, i, warpSize);
			a.y += __shfl_xor_sync(0xffffffff, a.y, i, warpSize);
			a.z += __shfl_xor_sync(0xffffffff, a.z, i, warpSize);

			b.x += __shfl_xor_sync(0xffffffff, b.x, i, warpSize);
			b.y += __shfl_xor_sync(0xffffffff, b.y, i, warpSize);
			b.z += __shfl_xor_sync(0xffffffff, b.z, i, warpSize);
#else
			a.x += __shfld_xor(a.x, i);
			a.y += __shfld_xor(a.y, i);
			a.z += __shfld_xor(a.z, i);

			b.x += __shfld_xor(b.x, i);
			b.y += __shfld_xor(b.y, i);
			b.z += __shfld_xor(b.z, i);
#endif
		}

		if(idy == 0){
			if(E >= 1){
				v4_d[idx].x += __fmul_rn(a.x, dtksq);
				v4_d[idx].y += __fmul_rn(a.y, dtksq);
				v4_d[idx].z += __fmul_rn(a.z, dtksq);
//printf("Kick %d %g %g %g %g\n", idx, x4_d[idx].w, __dmul_rn(a.x, dtksq), __dmul_rn(a.y, dtksq), __dmul_rn(a.z, dtksq));
			}
			if(E <= 1){
				acck_d[idx].x = b.x;
				acck_d[idx].y = b.y;
				acck_d[idx].z = b.z;
			}
		}
		if(E <= 2){
			if(idy == 0){
				Encpairs2_d[NencMax * idx].x = NencpairsI_s;
			}
			if(idy < NencpairsI_s && idy < NencMax){
				int jj = Encpairs2_d[idx * NencMax + idy].y;
				if(idx < jj){
					int Ne = atomicAdd(Nencpairs_d, 1);
					Encpairs_d[Ne].x = idx;
					Encpairs_d[Ne].y = jj;
				}
			}
		}
	}
}

// **************************************
//This kernel performs the second kick of the time step, in the case 32 <= NB < 128. NB is the next bigger number of N
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

//Authors: Simon Grimm
//April 2019
//
//****************************************
__global__ void kick32c_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double *rcritv_d, const double dtksq, int *Nencpairs_d, int2 *Encpairs_d, int2 *Encpairs2_d, int *EncFlag_d, const int NencMax, const int N, const int E){
	int idy = threadIdx.x;
	int idx = blockIdx.x;

	if(idx < N){

		__shared__ int NencpairsI_s;

		double4 x4i = x4_d[idx];
		double rcritvi = rcritv_d[idx];

		double3 a = {0.0, 0.0, 0.0};
		double3 b = {0.0, 0.0, 0.0};

		if(idy == 0){
			NencpairsI_s = 0;
		}

		__syncthreads();


		for(int i = 0; i < N; i += blockDim.x){
			if(i + idy < N){
				double4 x4j = x4_d[i + idy];
				double rcritvj = rcritv_d[i + idy];
				acc_d(a, b, x4i, x4j, rcritvi, rcritvj, &NencpairsI_s, Encpairs2_d, EncFlag_d, i + idy, idx, NencMax, E); 
			}
		}

		__syncthreads();

		for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
			a.x += __shfl_xor_sync(0xffffffff, a.x, i, warpSize);
			a.y += __shfl_xor_sync(0xffffffff, a.y, i, warpSize);
			a.z += __shfl_xor_sync(0xffffffff, a.z, i, warpSize);

			b.x += __shfl_xor_sync(0xffffffff, b.x, i, warpSize);
			b.y += __shfl_xor_sync(0xffffffff, b.y, i, warpSize);
			b.z += __shfl_xor_sync(0xffffffff, b.z, i, warpSize);
#else
			a.x += __shfld_xor(a.x, i);
			a.y += __shfld_xor(a.y, i);
			a.z += __shfld_xor(a.z, i);

			b.x += __shfld_xor(b.x, i);
			b.y += __shfld_xor(b.y, i);
			b.z += __shfld_xor(b.z, i);
#endif
//if(idx == 0 && i >= 16) printf("KickA %d %d %.20g %.20g\n", idy, i, a.x, b.x);
		}

		if(blockDim.x > warpSize){
			//reduce across warps
			extern __shared__ double3 Kick32c_s[];
			double3 *a_s = Kick32c_s;			//size: warpSize
			double3 *b_s = (double3*)&a_s[warpSize];	//size: warpSize

			int lane = threadIdx.x % warpSize;
			int warp = threadIdx.x / warpSize;
			if(warp == 0){
				a_s[threadIdx.x].x = 0.0;
				a_s[threadIdx.x].y = 0.0;
				a_s[threadIdx.x].z = 0.0;
				b_s[threadIdx.x].x = 0.0;
				b_s[threadIdx.x].y = 0.0;
				b_s[threadIdx.x].z = 0.0;
			}
			__syncthreads();

			if(lane == 0){
				a_s[warp] = a;
				b_s[warp] = b;
			}

			__syncthreads();
			//reduce previous warp results in the first warp
			if(warp == 0){
				a = a_s[threadIdx.x];
				b = b_s[threadIdx.x];
				for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
					a.x += __shfl_xor_sync(0xffffffff, a.x, i, warpSize);
					a.y += __shfl_xor_sync(0xffffffff, a.y, i, warpSize);
					a.z += __shfl_xor_sync(0xffffffff, a.z, i, warpSize);
		
					b.x += __shfl_xor_sync(0xffffffff, b.x, i, warpSize);
					b.y += __shfl_xor_sync(0xffffffff, b.y, i, warpSize);
					b.z += __shfl_xor_sync(0xffffffff, b.z, i, warpSize);
#else
					a.x += __shfld_xor(a.x, i);
					a.y += __shfld_xor(a.y, i);
					a.z += __shfld_xor(a.z, i);

					b.x += __shfld_xor(b.x, i);
					b.y += __shfld_xor(b.y, i);
					b.z += __shfld_xor(b.z, i);
#endif
//if(idx == 0 && i >= 16) printf("KickA2 %d %d %.20g %.20g\n", idy, i, a.x, b.x);

				}
				if(lane == 0){
					a_s[0] = a;
					b_s[0] = b;
				}
			}
			__syncthreads();

			a = a_s[0];
			b = b_s[0];
		}


		if(idy == 0){
//printf("Kick %d %g %.20g %.20g %.20g | %.20g %.20g %.20g\n", idx, x4_d[idx].w, a.x, a.y, a.z, b.x, b.y, b.z);
			if(E >= 1){
				v4_d[idx].x += __dmul_rn(a.x, dtksq);
				v4_d[idx].y += __dmul_rn(a.y, dtksq);
				v4_d[idx].z += __dmul_rn(a.z, dtksq);
			}
			if(E <= 1){
				acck_d[idx].x = b.x;
				acck_d[idx].y = b.y;
				acck_d[idx].z = b.z;
			}
		}
		if(E <= 2){
			if(idy == 0){
				Encpairs2_d[NencMax * idx].x = NencpairsI_s;
			}
			if(idy < NencpairsI_s && idy < NencMax){
				int jj = Encpairs2_d[idx * NencMax + idy].y;
				if(idx < jj){
					int Ne = atomicAdd(Nencpairs_d, 1);
					Encpairs_d[Ne].x = idx;
					Encpairs_d[Ne].y = jj;
				}
			}
		}
	}
}
//float Version
__global__ void kick32cf_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double *rcritv_d, const double dtksq, int *Nencpairs_d, int2 *Encpairs_d, int2 *Encpairs2_d, int *EncFlag_d, const int NencMax, const int N, const int E){
	int idy = threadIdx.x;
	int idx = blockIdx.x;

	if(idx < N){

		__shared__ int NencpairsI_s;

		float4 x4i;
		x4i.x = x4_d[idx].x;
		x4i.y = x4_d[idx].y;
		x4i.z = x4_d[idx].z;
		x4i.w = x4_d[idx].w;
		float rcritvi = rcritv_d[idx];

		float3 a = {0.0f, 0.0f, 0.0f};
		float3 b = {0.0f, 0.0f, 0.0f};

		if(idy == 0){
			NencpairsI_s = 0;
		}

		__syncthreads();


		for(int i = 0; i < N; i += blockDim.x){
			if(i + idy < N){
				float4 x4j;
				x4j.x = x4_d[i + idy].x;
				x4j.y = x4_d[i + idy].y;
				x4j.z = x4_d[i + idy].z;
				x4j.w = x4_d[i + idy].w;
				float rcritvj = rcritv_d[i + idy];
				acc_df(a, b, x4i, x4j, rcritvi, rcritvj, &NencpairsI_s, Encpairs2_d, EncFlag_d, i + idy, idx, NencMax, E); 
			}
		}

		__syncthreads();

		for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
			a.x += __shfl_xor_sync(0xffffffff, a.x, i, warpSize);
			a.y += __shfl_xor_sync(0xffffffff, a.y, i, warpSize);
			a.z += __shfl_xor_sync(0xffffffff, a.z, i, warpSize);

			b.x += __shfl_xor_sync(0xffffffff, b.x, i, warpSize);
			b.y += __shfl_xor_sync(0xffffffff, b.y, i, warpSize);
			b.z += __shfl_xor_sync(0xffffffff, b.z, i, warpSize);
#else
			a.x += __shfld_xor(a.x, i);
			a.y += __shfld_xor(a.y, i);
			a.z += __shfld_xor(a.z, i);

			b.x += __shfld_xor(b.x, i);
			b.y += __shfld_xor(b.y, i);
			b.z += __shfld_xor(b.z, i);
#endif
		}

		if(blockDim.x > warpSize){
			//reduce across warps
			extern __shared__ float3 Kick32cf_s[];
			float3 *a_s = Kick32cf_s;			//size: warpSize
			float3 *b_s = (float3*)&a_s[warpSize];	//size: warpSize

			int lane = threadIdx.x % warpSize;
			int warp = threadIdx.x / warpSize;
			if(warp == 0){
				a_s[threadIdx.x].x = 0.0f;
				a_s[threadIdx.x].y = 0.0f;
				a_s[threadIdx.x].z = 0.0f;
				b_s[threadIdx.x].x = 0.0f;
				b_s[threadIdx.x].y = 0.0f;
				b_s[threadIdx.x].z = 0.0f;
			}
			__syncthreads();

			if(lane == 0){
				a_s[warp] = a;
				b_s[warp] = b;
			}

			__syncthreads();
			//reduce previous warp results in the first warp
			if(warp == 0){
				a = a_s[threadIdx.x];
				b = b_s[threadIdx.x];
				for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
					a.x += __shfl_xor_sync(0xffffffff, a.x, i, warpSize);
					a.y += __shfl_xor_sync(0xffffffff, a.y, i, warpSize);
					a.z += __shfl_xor_sync(0xffffffff, a.z, i, warpSize);

					b.x += __shfl_xor_sync(0xffffffff, b.x, i, warpSize);
					b.y += __shfl_xor_sync(0xffffffff, b.y, i, warpSize);
					b.z += __shfl_xor_sync(0xffffffff, b.z, i, warpSize);
#else
					a.x += __shfld_xor(a.x, i);
					a.y += __shfld_xor(a.y, i);
					a.z += __shfld_xor(a.z, i);

					b.x += __shfld_xor(b.x, i);
					b.y += __shfld_xor(b.y, i);
					b.z += __shfld_xor(b.z, i);
#endif

				}
				if(lane == 0){
					a_s[0] = a;
					b_s[0] = b;
				}
			}
			__syncthreads();

			a = a_s[0];
			b = b_s[0];
		}


		if(idy == 0){
			if(E >= 1){
				v4_d[idx].x += __fmul_rn(a.x, dtksq);
				v4_d[idx].y += __fmul_rn(a.y, dtksq);
				v4_d[idx].z += __fmul_rn(a.z, dtksq);
			}
			if(E <= 1){
				acck_d[idx].x = b.x;
				acck_d[idx].y = b.y;
				acck_d[idx].z = b.z;
			}
		}
		if(E <= 2){
			if(idy == 0){
				Encpairs2_d[NencMax * idx].x = NencpairsI_s;
			}
			if(idy < NencpairsI_s && idy < NencMax){
				int jj = Encpairs2_d[idx * NencMax + idy].y;
				if(idx < jj){
					int Ne = atomicAdd(Nencpairs_d, 1);
					Encpairs_d[Ne].x = idx;
					Encpairs_d[Ne].y = jj;
				}
			}
		}
	}
}

// **************************************
//This kernel performs the second kick of the time step.
//It calculates the acceleration between all bodies with respect to the changeover function K.
//It also calculates all accelerations from bodies not beeing in a close encounter and store it in accK_d. This values will then be used 
//it the next time step.
//It performs also a precheck for close encouter candidates. This pairs are stored in the array Encpairs_d.
//The number of close encounter candidates is stored in Nencpairs_d.
//NT is the total number of bodies, Nstart is the starting index
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
__global__ void KickM2_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double *rcritv_d, int *Nencpairs_d, int2 *Encpairs_d, double *dt_d, double Kt, int *index_d, int NT, int Nstart){

	int idy = threadIdx.x;
	int id = blockIdx.x * Bl2 + idy - Nmax + Nstart;

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

	if(id < NT + Nstart && id >= Nstart){
		st_s[idy] = index_d[id] / def_MaxIndex;
		x4_s[idy] = x4_d[id];
		rcritv_s[idy] = rcritv_d[id];
		dtksqKt = dt_d[st_s[idy]] * Kt * def_ksq;
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
		if(id + Bl < NT + Nstart){
			st_s[idy + Bl] = index_d[id + Bl] / def_MaxIndex;
			x4_s[idy + Bl] = x4_d[id + Bl];
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
				if(rsq < def_pc * rcritv2 && (x4_s[idy].w > 0.0 || x4_s[idy + j].w > 0.0)){	//prechecker
					if(idy >= Nmax){
						int Ne = atomicAdd(Nencpairs_d, 1);
						atomicAdd(Nencpairs_d + st_s[idy] + 1, 1);
						ij.x = id;
						ij.y = id + j;
//printf("precheck E %d %d %d %d %d %d\n", E, st_s[idy], id, id + j, index_d[id], index_d[id + j]);	
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
//printf("Kick1 %d %d %g %g %g %g %g\n", id, id + j, x4_s[idy].w, x4_s[idy + j].w, __dmul_rn(rx, si), __dmul_rn(ry, si), __dmul_rn(rz, si));
		}
	}
	__syncthreads();
	if(id < NT + Nstart && id >= Nstart && idy >= Nmax){
		if(E >= 1){
			v4_d[id].x += __dmul_rn((a_s[idy].x + b_s[idy].x), dtksqKt);
			v4_d[id].y += __dmul_rn((a_s[idy].y + b_s[idy].y), dtksqKt);
			v4_d[id].z += __dmul_rn((a_s[idy].z + b_s[idy].z), dtksqKt);
//printf("Kick %d %g %g %g %g\n", id, x4_d[id].w, __dmul_rn((a_s[idy].x + b_s[idy].x), dtksqKt), __dmul_rn((a_s[idy].y + b_s[idy].y), dtksqKt), __dmul_rn((a_s[idy].z + b_s[idy].z), dtksqKt));
		}
		if(E <= 1){
			acck_d[id].x = (a_s[idy].x + b_s[idy].x);
			acck_d[id].y = (a_s[idy].y + b_s[idy].y);
			acck_d[id].z = (a_s[idy].z + b_s[idy].z);
//printf("acc %d %.20g %.20g %.20g %.20g %.20g %.20g %g\n", id, v4_d[id].x, v4_d[id].y, v4_d[id].z, acck_d[id].x, acck_d[id].y, acck_d[id].z, x4_d[id].w);
		}
	}
}

template <int Bl, int Bl2, int Nmax, int E>
__global__ void KickM2Simple_kernel(double4 *x4_d, double4 *v4_d, double4 *v4b_d, double3 *acck_d, double *dt_d, double Kt, int *index_d, int NT, int Nst, int Nstart){

	int idy = threadIdx.x;
	int id = blockIdx.x * Bl2 + idy - Nmax + Nstart;

	__shared__ volatile double3 a_s[Bl + Nmax];
	__shared__ volatile double3 b_s[Bl + Nmax];
	__shared__ double4 x4_s[Bl + Nmax];
	__shared__ int st_s[Bl + Nmax];

	a_s[idy].x = 0.0;
	a_s[idy].y = 0.0;
	a_s[idy].z = 0.0;
	b_s[idy].x = 0.0;
	b_s[idy].y = 0.0;
	b_s[idy].z = 0.0;

	double dtksqKt = 0.0;

	if(id < NT + Nstart && id >= Nstart){
		st_s[idy] = index_d[id] / def_MaxIndex;
		x4_s[idy] = x4_d[id];
		dtksqKt = dt_d[st_s[idy]] * Kt * def_ksq;
	}
	else{
		st_s[idy] = -idy-1;
		x4_s[idy].x = 0.0; 
		x4_s[idy].y = 0.0;
		x4_s[idy].z = 0.0;
		x4_s[idy].w = 0.0;
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
		if(id + Bl < NT + Nstart){
			st_s[idy + Bl] = index_d[id + Bl] / def_MaxIndex;
			x4_s[idy + Bl] = x4_d[id + Bl];
		}
		else{
			st_s[idy + Bl] = -idy-Bl-1;
			x4_s[idy + Bl].x = 0.0;
			x4_s[idy + Bl].y = 0.0;
			x4_s[idy + Bl].z = 0.0;
			x4_s[idy + Bl].w = 0.0;
		}
	}

	volatile double rx, ry, rz;
	volatile double rsq, ir, ir3;
	volatile double si, sj;
	
	for(volatile int j = Nmax - 1; j > 0; --j){
		__syncthreads();
		if((st_s[idy] - st_s[idy + j]) == 0 && x4_s[idy].w >= 0.0 && x4_s[idy + j].w >= 0.0){
			rx = x4_s[idy + j].x - x4_s[idy].x;
			ry = x4_s[idy + j].y - x4_s[idy].y;
			rz = x4_s[idy + j].z - x4_s[idy].z;
			rsq = rx * rx + ry * ry + rz * rz;
			ir = 1.0 / sqrt(rsq);
			ir3 = ir * ir * ir;

			si = x4_s[idy + j].w * ir3;
			sj = x4_s[idy].w * ir3;

			a_s[idy].x += __dmul_rn(rx, si);
			a_s[idy].y += __dmul_rn(ry, si);
			a_s[idy].z += __dmul_rn(rz, si);

			b_s[idy + j].x += __dmul_rn(-rx, sj);
			b_s[idy + j].y += __dmul_rn(-ry, sj);
			b_s[idy + j].z += __dmul_rn(-rz, sj);

		}
	}
	__syncthreads();
	if(id < NT + Nstart && id >= Nstart && idy >= Nmax){
		if(E >= 1){
			double vx = __dmul_rn((a_s[idy].x + b_s[idy].x), dtksqKt);
			double vy = __dmul_rn((a_s[idy].y + b_s[idy].y), dtksqKt);
			double vz = __dmul_rn((a_s[idy].z + b_s[idy].z), dtksqKt);
			v4_d[id].x += vx;
			v4_d[id].y += vy;
			v4_d[id].z += vz;
//printf("acc %d %.20g %.20g %.20g\n", id, v4_d[id].x, v4_d[id].y, v4_d[id].z);
			if(E == 1){
				v4b_d[id] = v4_d[id];
			}

		}
		if(E <= 1){
			acck_d[id].x = (a_s[idy].x + b_s[idy].x);
			acck_d[id].y = (a_s[idy].y + b_s[idy].y);
			acck_d[id].z = (a_s[idy].z + b_s[idy].z);
//printf("acc %d %.20g %.20g %.20g %.20g %.20g %.20g\n", id, v4_d[id].x, v4_d[id].y, v4_d[id].z, acck_d[id].x, acck_d[id].y, acck_d[id].z);
		}
	}
}
template <int Bl, int Bl2, int Nmax, int E>
__global__ void KickM2TTV_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double *rcritv_d, int *Nencpairs_d, int2 *Encpairs_d, double *dt_d, double Kt, int *index_d, int NT, double2 *Msun_d, int *Ntransit_d, int *Transit_d, int Nstart){

	int idy = threadIdx.x;
	int id = blockIdx.x * Bl2 + idy - Nmax + Nstart;

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
	double dt = 0.0;
	double4 v4i;
	double Msun = 0.0;
	double Rsun = 0.0;

	if(id < NT + Nstart && id >= Nstart){
		st_s[idy] = index_d[id] / def_MaxIndex;
		x4_s[idy] = x4_d[id];
		v4i = v4_d[id];
		rcritv_s[idy] = rcritv_d[id];
		dtksqKt = dt_d[st_s[idy]] * Kt * def_ksq;
		dt = dt_d[st_s[idy]];
		Msun = Msun_d[st_s[idy]].x;
		Rsun = Msun_d[st_s[idy]].y;
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
		if(id + Bl < NT + Nstart){
			st_s[idy + Bl] = index_d[id + Bl] / def_MaxIndex;
			x4_s[idy + Bl] = x4_d[id + Bl];
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
				if(rsq < def_pc * rcritv2 && (x4_s[idy].w > 0.0 || x4_s[idy + j].w > 0.0)){	//prechecker
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
	if(id < NT + Nstart && id >= Nstart && idy >= Nmax){
		if(E >= 1){
			v4_d[id].x += __dmul_rn((a_s[idy].x + b_s[idy].x), dtksqKt);
			v4_d[id].y += __dmul_rn((a_s[idy].y + b_s[idy].y), dtksqKt);
			v4_d[id].z += __dmul_rn((a_s[idy].z + b_s[idy].z), dtksqKt);
		}
		if(E <= 1){
			acck_d[id].x = (a_s[idy].x + b_s[idy].x);
			acck_d[id].y = (a_s[idy].y + b_s[idy].y);
			acck_d[id].z = (a_s[idy].z + b_s[idy].z);
		}
//printf("KickM2TTV %d %.20e %.20e %.20e %.20e %.20e %.20e\n", id, a_s[idy].x, a_s[idy].y, a_s[idy].z, v4_d[id].x, v4_d[id].y, v4_d[id].z);

		//calculate acceleration from the central star
		double rsq = x4_s[idy].x*x4_s[idy].x + x4_s[idy].y*x4_s[idy].y + x4_s[idy].z*x4_s[idy].z + 1.0e-30;
		double ir = 1.0 / sqrt(rsq);
		double ir3 = ir * ir * ir;
		double s = - def_ksq * Msun * ir3;
		double3 a;
		a.x = a_s[idy].x + b_s[idy].x;
		a.y = a_s[idy].y + b_s[idy].y;
		a.z = a_s[idy].z + b_s[idy].z;

		a.x += s * x4_s[idy].x;
		a.y += s * x4_s[idy].y;
		a.z += s * x4_s[idy].z;

		double g = x4_s[idy].x * v4i.x + x4_s[idy].y * v4i.y;
		double gd = v4i.x * v4i.x + v4i.y * v4i.y + x4_s[idy].x * a.x + x4_s[idy].y * a.y;
		double rsky = sqrt(x4_s[idy].x * x4_s[idy].x + x4_s[idy].y * x4_s[idy].y);
		double v = sqrt(v4i.x * v4i.x + v4i.y * v4i.y);
		double R = Rsun + v4i.w;

//printf("TTVA %d g %g gd %g g/gd %.20g x %.20g y %.20g z %.10g dt %.20g rsky %g R %g R+ %g\n", id, g, gd, -g / gd, x4_s[idy].x, x4_s[idy].y, x4_s[idy].z, dt, rsky, R, R + v * dt);
		if(dt > 0){
			if(x4_s[idy].z > 0.0 && gd > 0.0 && fabs(g / gd) < 3.5 * fabs(dt) && rsky < R + v * fabs(dt)){
				if(g <= 0.0){
					int Nt = atomicAdd(Ntransit_d, 1);
					Nt = min(Nt, def_NtransitMax - 1);
					Transit_d[Nt] = id;
				}
			}
		}
		else{
			if(x4_s[idy].z > 0.0 && gd > 0.0 && fabs(g / gd) < 3.5 * fabs(dt) && rsky < R + v * fabs(dt)){
				if(g >= 0.0){
					int Nt = atomicAdd(Ntransit_d, 1);
					Nt = min(Nt, def_NtransitMax - 1);
					Transit_d[Nt] = id;
				}
			}
		}

	}
}

// CC = 1, do pre-check
// CC = 0, don't do pre-check
__global__ void KickM3_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double *rcritv_d, int *Nencpairs_d, int2 *Encpairs_d, double *dt_d, double Kt, int *index_d, int NT, int *N_d, int *NBS_d, const int EE, const int CC){

	int idy = threadIdx.x;	//must be in x dimension in order to be in the same warp
	int id = blockIdx.x * blockDim.y + threadIdx.y;

//if(idy == 0) printf("id %d %d %d %d\n", id, blockIdx.x, blockDim.x, threadIdx.x);

	if(id < NT){

		int st = index_d[id] / def_MaxIndex;
		int NBS = NBS_d[st];
		double4 x4i = x4_d[id];
		double rcritvi = rcritv_d[id];
		int Ni = N_d[st];
		
		double3 a = {0.0, 0.0, 0.0};

		__syncthreads();

		for(int i = 0; i < Ni; i += blockDim.x){
			if(idy + i < Ni){
				double4 x4j = x4_d[NBS + idy + i];

				if(NBS + idy + i != id && x4i.w >= 0.0 && x4j.w >= 0.0){
					double rcritvj = rcritv_d[NBS + idy + i];

					double rcritv = fmax(rcritvi, rcritvj);
					double rcritv2 = rcritv * rcritv;
					double rx = x4j.x - x4i.x;
					double ry = x4j.y - x4i.y;
					double rz = x4j.z - x4i.z;

					double rsq = rx * rx + ry * ry + rz * rz;
					double ir = 1.0 / sqrt(rsq);
					double ir3 = ir * ir * ir;
//if( EE == 0 && idy + i == 0) printf("Kick %d %d %d %d\n", id, NBS + idy + i, index_d[id], index_d[NBS + idy + i]);
					if(CC == 1){
						if(rsq < def_pc * rcritv2 && (x4i.w > 0.0 || x4j.w > 0.0)){	//prechecker
							if(id < NBS + idy + i){
								int Ne = atomicAdd(Nencpairs_d, 1);
								atomicAdd(Nencpairs_d + st + 1, 1);
								int2 ij;
								ij.x = id;
								ij.y = NBS + idy + i;
//printf("precheck E %d %d %d %d %d %d\n", EE, st, id, NBS + idy + i, index_d[id], index_d[NBS + idy + i]);
								Encpairs_d[Ne] = ij;
							}
						}
					}

					double s;
					if(rsq >= rcritv2){
						s = x4j.w * ir3;
					}
					else{
						if(rsq <= 0.01 * rcritv2){
							s = 0.0;
						}
						else{
							double y = (rsq * ir - 0.1 * rcritv) / (0.9 * rcritv);
							double yy = y * y;
							double K = ir3 * yy / (2.0 * yy - 2.0 * y + 1.0);
							s = K * x4j.w;
						}
					}

					a.x += __dmul_rn(rx, s);
					a.y += __dmul_rn(ry, s);
					a.z += __dmul_rn(rz, s);
//if(id == 0) printf("Kick %d %d %.20g %.20g %.20g\n", id, idy + i, a.x, a.y, a.z);
				}
			}
		}

		__syncthreads();

		for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
			a.x += __shfl_xor_sync(0xffffffff, a.x, i, warpSize);
			a.y += __shfl_xor_sync(0xffffffff, a.y, i, warpSize);
			a.z += __shfl_xor_sync(0xffffffff, a.z, i, warpSize);
#else
			a.x += __shfld_xor(a.x, i);
			a.y += __shfld_xor(a.y, i);
			a.z += __shfld_xor(a.z, i);
#endif
		//if(idx == 0 && i >= 16) printf("KickA %d %d %.20g\n", idy, i, a.x);
		}

		__syncthreads();

		if(blockDim.x > warpSize){
			//reduce across warps
			extern __shared__ double3 KickM3_s[];
			double3 *a_s = KickM3_s;			//size: warpSize

			int lane = threadIdx.x % warpSize;
			int warp = threadIdx.x / warpSize;
			if(warp == 0){
				a_s[threadIdx.x].x = 0.0;
				a_s[threadIdx.x].y = 0.0;
				a_s[threadIdx.x].z = 0.0;
			}
			__syncthreads();

			if(lane == 0){
				a_s[warp] = a;
			}

			__syncthreads();
			//reduce previous warp results in the first warp
			if(warp == 0){
				a = a_s[threadIdx.x];
				for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
					a.x += __shfl_xor_sync(0xffffffff, a.x, i, warpSize);
					a.y += __shfl_xor_sync(0xffffffff, a.y, i, warpSize);
					a.z += __shfl_xor_sync(0xffffffff, a.z, i, warpSize);
#else
					a.x += __shfld_xor(a.x, i);
					a.y += __shfld_xor(a.y, i);
					a.z += __shfld_xor(a.z, i);
#endif
				//if(idx == 0 && i >= 16) printf("KickA2 %d %d %.20g\n", idy, i, a.x);

				}
				if(lane == 0){
					a_s[0] = a;
				}
			}
			__syncthreads();

			a = a_s[0];
		}

		__syncthreads();

//if(idy == 0 && idx < 10)	printf("Kick %d %d %d %.20g %.20g %.20g\n", id, threadIdx.y, idy, a_s[idy].x, a_s[idy].y, a_s[idy].z); 


		if(EE <= 1){
			if(idy == 0){
				acck_d[id].x = a.x;
				acck_d[id].y = a.y;
				acck_d[id].z = a.z;
			}
		}
		if(EE >= 1){
			if(idy == 0){
				double dtksqKt = dt_d[st] * Kt * def_ksq;
				v4_d[id].x += __dmul_rn(a.x, dtksqKt);
				v4_d[id].y += __dmul_rn(a.y, dtksqKt);
				v4_d[id].z += __dmul_rn(a.z, dtksqKt);
			}
		}
	}
}
#endif

#ifndef KICK_H
#define KICK_H
#include "define.h"

// **************************************
// This function computes the term a = mi/rij^3 * Kij
// ****************************************
__device__ void  accA(double3 &ac, double4 &x4i, double4 &x4j, double rcritvi, double rcritvj, int j, int i){
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
template < int E >
__device__ void  acc_d(double3 &ac, double3 &b, double4 &x4i, double4 &x4j, double rcritvi, double rcritvj, int *NencpairsI, int2 *Encpairs2_d, int j, int i, int NencMax, double &test){
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
			if(rsq < def_pc * rcritv2){  //prechecker
//printf("Precheck %d %d\n", i, j);
				int Ni = atomicAdd(NencpairsI, 1);
				Encpairs2_d[NencMax * i + Ni].y = j;
			}
		}
		if(E <= 12 && E >=10){ //prechecker used for Test Particle Mode
			if(rsq < def_pc * rcritv2){  //prechecker
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
//if(i == 0) printf("%d %d %.40g %.40g %.40g Kick\n", i, j, 1.0, 1.0 / ir, s);
		}
		else{
			if(rsq <= 0.01 * rcritv2){
				s = 0.0;
//if(i == 0) printf("%d %d %.40g %.40g %.40g Kick\n", i, j, 0, 1.0 / ir, s);

			}
			else{
				y = (rsq * ir - 0.1 * rcritv)/(0.9*rcritv);
				yy = y * y;
				s = (ir3 * yy) / (2.0*yy - 2.0*y + 1.0) * x4j.w;
//if(i == 0) printf("%d %d %.40g %.40g %.40g Kick\n", i, j, yy / (2.0*yy - 2.0*y + 1.0), 1.0/ir, s);

			}
		}
//printf("acc %d %d %.20e %.20e %20e\n", i, j, rsq, ir3, x4j.w);
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

// ******************************************************
// Version of acc which is called from the recursive symplectic sub step method
// Author: Simon Grimm
// Janury 2019
// ******************************************************
__device__ void accS(double4 x4i, double4 x4j, double3 &ac, double *rcritv_d, double &test, int &NencpairsI, int2 *Encpairs2_d, const int i, const int j, const int NconstT, const int NencMax, const int SLevel, const int SLevels, const int E){

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

			if(rsq <  1.0 * rcritv2){
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
				if(rsq < def_pc * rcritv2){  //prechecker
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
template < int E >
__device__ void  accG3(double3 &ac, double3 &b, double4 &x4i, double4 &x4j, double rcritvi, double rcritvj, int groupIndexi, int groupIndexj, int *NencpairsI, int *NencpairsJ, int2 *Encpairs_d, int j, int i, int NconstT, int NencMax, double &test, double t){
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
			if(((groupIndexi == groupIndexj && groupIndexi >= 0 && groupIndexi < NconstT) || rsq < def_pc * rcritv) && x4i.w > 0.0 && x4j.w > 0.0){	
#else
			if(((groupIndexi == groupIndexj && groupIndexi >= 0 && groupIndexi < NconstT) || B > G3Limit2) && x4i.w > 0.0 && x4j.w > 0.0){	

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
__global__ void kick32B_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double3 *ab_d, int N, double dtksq){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;
	if(id < N){
		double3 a = acck_d[id];
		if(x4_d[id].w >= 0.0){
			v4_d[id].x += __dmul_rn(a.x, dtksq);
			v4_d[id].y += __dmul_rn(a.y, dtksq);
			v4_d[id].z += __dmul_rn(a.z, dtksq);
//printf("KickB %d %.16e %.16e %.16e %.16e %.16e %.16e\n", id, acck_d[id].x, acck_d[id].y, acck_d[id].z, v4_d[id].x * dayUnit, v4_d[id].y * dayUnit, v4_d[id].z * dayUnit);
		}
		ab_d[id].x = a.x;
		ab_d[id].y = a.y;
		ab_d[id].z = a.z;
	}
}
// **************************************
//This kernel performs the first kick of the time step, in the case of no close encounters.
//It reuses the values from the second kick in the previous time step.

//Authors: Simon Grimm
//November 2016
// ****************************************
__global__ void kick32BM_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double3 *ab_d, int *index_d, int N, double *dt_d, double Kt, int Nstart){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy + Nstart;
	if(id < N + Nstart){
		int st = index_d[id] / 100;
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
__global__ void kick32BMSimple_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double3 *ab_d, int *index_d, int N, double *dt_d, double Kt, double *time_d, double *idt_d, double *ict_d, long long timeStep, int Nst, int Nstart){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy + Nstart;
	if(id < N + Nstart){
		int st = index_d[id] / 100;
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
__global__ void kick32C_kernel(double4 *x4_d, double4 *v4_d, double3 *ab_d, int N, double dtksq){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;
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

// **************************************
//This kernel performs the first kick of the time step, in the case of no close encounters.
//It reuses the values from the second kick in the previous time step.
//It checks if a transit occurs in the next time step

//Authors: Simon Grimm
//November 2016
// ****************************************
__global__ void kick32BTTV_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double3 *ab_d, int N, double dtksq, double dt, double Msun, double Rsun, int *Ntransit_d, int *Transit_d){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;
	if(id < N){
		double3 a = acck_d[id];
		double4 x4i = x4_d[id];
		double4 v4i = v4_d[id];
		if(x4i.w >= 0.0){
			v4_d[id].x += __dmul_rn(a.x, dtksq);
			v4_d[id].y += __dmul_rn(a.y, dtksq);
			v4_d[id].z += __dmul_rn(a.z, dtksq);
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

//if(id == 0) printf("TTV %d g %g gd %g g/gd %.20g x %.10g y %.10g z %.10g dt %.20g rsky %g R %g R+ %g\n", id, g, gd, -g / gd, x4i.x, x4i.y, x4i.z, dt, rsky, R, R + v * dt);

			if(dt > 0){
				if(x4i.z > 0.0 && gd > 0.0 && fabs(g / gd) < 3.5 * fabs(dt) && rsky < R + v * fabs(dt)){
					if(g <= 0.0){
//if(id == 0) printf("----TTV %d g %g gd %g g/gd %.20g x %.10g y %.10g z %.10g dt %.20g rsky %g R %g R+ %g\n", id, g, gd, -g / gd, x4i.x, x4i.y, x4i.z, dt, rsky, R, R + v * dt);
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
	}
}
__global__ void kick32BMTTV_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double3 *ab_d, int *index_d, int N, double *dt_d, double Kt, double4 *Msun_d, int *Ntransit_d, int *Transit_d, int Nstart){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy + Nstart;
	if(id < N + Nstart){
		int st = index_d[id] / 100;
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
	}
}
__global__ void kick32BMTTVSimple_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double3 *ab_d, int *index_d, int N, double *dt_d, double Kt, double4 *Msun_d, int *Ntransit_d, int *Transit_d, double *time_d, double *idt_d, double *ict_d, long long timeStep, int Nst,int Nstart){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy + Nstart;
	if(id < N + Nstart){
		int st = index_d[id] / 100;
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
//This kernel is used to sort the close encoutner list, to be able to reproduce simulations exactly
//It shoud be used only for debugging or special cases.

//Authors: Simon Grimm
//August 2016
// *********************************************
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



// **************************************
//This kernel performs the first kick of the time step, in the case of close interactions.
//It reuses the values from the second kick in the previous time step, and adds the terms aij*dt*Kij for all
//the bodies involved in a close encounter.
//NI is the number of bodies involved in a close encounter with body i 

//Authors: Simon Grimm
//December 2016
// ****************************************
__global__ void kick32Ab_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double3 *ab_d, double *rcritv_d, const double dtksq, int2 *Encpairs2_d, double *test_d, int N, int NencMax, double t){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;	

	double3 a;

	a.x = 0.0;
	a.y = 0.0;
	a.z = 0.0;

	if(id < N){
		int NI = Encpairs2_d[id * NencMax].x;
//if(NI > 0) printf("NI %d %d\n", id, NI);
		double4 x4i = x4_d[id];
		if(x4i.w >= 0.0){
			double rcritvi = rcritv_d[id];
			__syncthreads();
			for(int i = 0; i < NI; ++i){
				int jj = Encpairs2_d[id * NencMax + i].y;
				double4 x4j = x4_d[jj];
//printf("AI %d %d %d %.40g %.40g %.40g %.40g\n", id, jj, NI, x4i.x, x4j.x, v4_d[id].z, a.z);
				double rcritvj = rcritv_d[jj];
				accA(a, x4i, x4j, rcritvi, rcritvj, jj, id);
			}
			__syncthreads();
			double3 aa;
			aa.x = __dmul_rn(a.x + acck_d[id].x, dtksq);
			aa.y = __dmul_rn(a.y + acck_d[id].y, dtksq);
			aa.z = __dmul_rn(a.z + acck_d[id].z, dtksq);

			ab_d[id] = aa;

			v4_d[id].x += aa.x;
			v4_d[id].y += aa.y;
			v4_d[id].z += aa.z;
		}
//if(id == 13723) printf("K %d %.40g %.40g %.40g %.40g %.40g\n", id, v4_d[id].x, v4_d[id].y, v4_d[id].z, a.z, acck_d[id].z);
	}
}

// *****************************************************
// Version of the Kick kernel which is called from the recursive symplectic sub step method
//
// Author: Simon Grimm
// January 2019
// ********************************************************
__global__ void kickS_kernel(double4 *x4_d, double4 *v4_d, double4 *xold_d, double4 *vold_d, double *rcritv_d, const double dtksq, int *Nencpairs_d, int2 *Encpairs_d, int2 *Encpairs2_d, int *Nencpairs3_d, int *Encpairs3_d, const int N, const int NconstT, const int NencMax, const int SLevel, const int SLevels, const int E){

	int idy = threadIdx.x;
	int idd = blockIdx.x * blockDim.x + idy;	

	double3 a;

	a.x = 0.0;
	a.y = 0.0;
	a.z = 0.0;

	double test;
	if(idd < Nencpairs3_d[0]){
		int id = Encpairs3_d[idd * NencMax + 1];
		if(id >= 0 && id < N){
//if(NI > 0) printf("NI %d %d\n", id, NI);
			double4 x4i = xold_d[id];
			double4 v4i = vold_d[id];
			int NencpairsI = 0;
			if(x4i.w >= 0.0){
				__syncthreads();
				int NI = Encpairs3_d[id * NencMax + 2];
				for(int i = 0; i < NI; ++i){
					int jj = Encpairs3_d[id * NencMax + i + 3];
					double4 x4j = xold_d[jj];
					if(x4j.w >= 0.0){
//if(E == 0) printf("AI %d %d %d %.40g %.40g %.40g %.40g\n", id, jj, NI, x4i.x, x4j.x, v4_d[id].z, a.z);
						accS(x4i, x4j, a, rcritv_d, test, NencpairsI, Encpairs2_d, id, jj, NconstT, NencMax, SLevel, SLevels, E);
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

				if(E == 0 || E == 2){
					for(int i = 0; i < NencpairsI; ++i){
						int jj = Encpairs2_d[id * NencMax + i].y;
						if(id > jj){
							int Ne = atomicAdd(Nencpairs_d, 1);
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
__global__ void kick32ATTV_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double3 *ab_d, double *rcritv_d, const double dtksq, int2 *Encpairs2_d, double *test_d, const int N, const int NencMax, double t, double dt, double Msun, double Rsun, int *Ntransit_d, int *Transit_d){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;	

	double3 a;

	a.x = 0.0;
	a.y = 0.0;
	a.z = 0.0;

	if(id < N){
		int NI = Encpairs2_d[id * NencMax].x;
//if(NI > 0) printf("NI %d %d\n", id, NI);
		double4 x4i = x4_d[id];
		if(x4i.w >= 0.0){
			double4 v4i = v4_d[id];
			double rcritvi = rcritv_d[id];
			__syncthreads();
			for(int i = 0; i < NI; ++i){
				int jj = Encpairs2_d[id * NencMax + i].y;
				double4 x4j = x4_d[jj];
	//printf("AI %d %d %d %.40g %.40g %.40g %.40g\n", id, jj, NI, x4i.x, x4j.x, v4_d[id].z, v4_d[jj].z);
				double rcritvj = rcritv_d[jj];
				accA(a, x4i, x4j, rcritvi, rcritvj, jj, id);
			}
			__syncthreads();
			double3 aa;
			aa.x = a.x + acck_d[id].x;
			aa.y = a.y + acck_d[id].y;
			aa.z = a.z + acck_d[id].z;

			ab_d[id] = aa;

			v4_d[id].x += __dmul_rn(aa.x, dtksq);
			v4_d[id].y += __dmul_rn(aa.y, dtksq);
			v4_d[id].z += __dmul_rn(aa.z, dtksq);

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
//August 2016
//****************************************
template <int Bl2, int E>
__global__ void kick16b_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double *rcritv_d, int *groupIndex_d, const double dtksq, int *Nencpairs_d, int2 *Encpairs_d, int2 *Encpairs2_d, double *test_d, int NencMax, double t, int N){
	int idy = threadIdx.x;
	int idx = blockIdx.x;

	__shared__ double3 ab1_s[Bl2]; 		//the b1_s array is here stored in ab1_s[idy + 16]
	__shared__ int NencpairsI_s;

	double4 x4i = x4_d[idx];
	double rcritvi = rcritv_d[idx];
#if G3 > 0
	int groupIndexi = groupIndex_d[idx];
#endif

	double4 x4j;
	double rcritvj;

	if(idy < N){
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
#if G3 > 0
	int groupIndexj;
	if(idy < N) groupIndexj = groupIndex_d[idy];
	else groupIndexj = 0;
#endif


	double test;

	if(idy == 0){
		NencpairsI_s = 0;
	}

	__syncthreads();

	ab1_s[idy].x = 0.0;
	ab1_s[idy].y = 0.0;
	ab1_s[idy].z = 0.0;

	ab1_s[idy + 8].x = 0.0;
	ab1_s[idy + 8].y = 0.0;
	ab1_s[idy + 8].z = 0.0;

	__syncthreads();

	if(idy < N){
#if G3 == 0
		acc_d<E>(ab1_s[idy], ab1_s[idy + 16], x4i, x4j, rcritvi, rcritvj, &NencpairsI_s, Encpairs2_d, idy, idx, NencMax, test); 
#else
		//accG3<E>(ab1_s[idy], ab1_s[idy + 16], x4i, x4j, rcritvi, rcritvj, groupIndexi, groupIndexj, &NencpairsI_s, &NencpairsJ_s, Encpairs2_d, idy, idx, NconstT, NencMax, test, t);
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

	__syncthreads();

	if(idy == 0){
		if(E >= 1){
			v4_d[idx].x += __dmul_rn(ab1[0].x, dtksq);
			v4_d[idx].y += __dmul_rn(ab1[0].y, dtksq);
			v4_d[idx].z += __dmul_rn(ab1[0].z, dtksq);
		}
		if(E <= 1){
			acck_d[idx].x += ab1[16].x;
			acck_d[idx].y += ab1[16].y;
			acck_d[idx].z += ab1[16].z;
		}
	}
	if(E <= 2){
		if(idy == 0){
			Encpairs2_d[NencMax * idx].x = NencpairsI_s;
		}
		if(idy < NencpairsI_s){
			int jj = Encpairs2_d[idx * NencMax + idy].y;
			if(idx < jj){
				int Ne = atomicAdd(Nencpairs_d, 1);
				Encpairs_d[Ne].x = idx;
				Encpairs_d[Ne].y = jj;
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
//August 2016
//
//****************************************
template < int Bl, int Bl2, int E>
__global__ void kick32b_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double *rcritv_d, int *groupIndex_d, const double dtksq, int *Nencpairs_d, int2 *Encpairs_d, int2 *Encpairs2_d, double *test_d, int NconstT, int NencMax, double t, int N){
	int idy = threadIdx.x;
	int idx = blockIdx.x;

	__shared__ double3 a1_s[Bl2];
	__shared__ double3 b1_s[Bl2];
	__shared__ int NencpairsI_s;

	double4 x4i = x4_d[idx];
	double rcritvi = rcritv_d[idx];
#if G3 > 0
	int groupIndexi = groupIndex_d[idx];
#endif


	double4 x4j;
	double rcritvj;
	if(idy < N){
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
#if G3 > 0
	int groupIndexj;
	if(idy < N) groupIndexj = groupIndex_d[idy];
	else groupIndexj = 0;
#endif

	double test;

	if(idy == 0){
		NencpairsI_s = 0;
	}
	__syncthreads();
	a1_s[idy].x = 0.0;
	a1_s[idy].y = 0.0;
	a1_s[idy].z = 0.0;

	b1_s[idy].x = 0.0;
	b1_s[idy].y = 0.0;
	b1_s[idy].z = 0.0;

	__syncthreads();
	if(idy < N){
#if G3 == 0
		acc_d<E>(a1_s[idy], b1_s[idy], x4i, x4j, rcritvi, rcritvj, &NencpairsI_s, Encpairs2_d, idy, idx, NencMax, test); 
#else
		//accG3<E>(a1_s[idy], b1_s[idy], x4i, x4j, rcritvi, rcritvj, groupIndexi, groupIndexj, &NencpairsI_s, &NencpairsJ_s, Encpairs2_d, idy, idx, NconstT, NencMax, test, t); 

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

	__syncthreads();

	if(idy == 0){

		if(E >= 1){
			v4_d[idx].x += __dmul_rn(a1[0].x, dtksq);
			v4_d[idx].y += __dmul_rn(a1[0].y, dtksq);
			v4_d[idx].z += __dmul_rn(a1[0].z, dtksq);
		}
		if(E <= 1){
			acck_d[idx].x += b1[0].x;
			acck_d[idx].y += b1[0].y;
			acck_d[idx].z += b1[0].z;
		}
	}

	if(E <= 2){
		if(idy == 0){
			Encpairs2_d[NencMax * idx].x = NencpairsI_s;
		}
		if(idy < NencpairsI_s){
			int jj = Encpairs2_d[idx * NencMax + idy].y;
			if(idx < jj){
				int Ne = atomicAdd(Nencpairs_d, 1);
				Encpairs_d[Ne].x = idx;
				Encpairs_d[Ne].y = jj;
			}
		}
	}
}
// **************************************
//This kernel performs the second kick of the time step for test particles
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
//Authors: Simon Grimm
//August 2016
// ****************************************
template < int E >
__global__ void kicksmall_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double *rcritv_d, int *groupIndex_d, const double dtksq, int N, int *Nencpairs_d, int2 *Encpairs_d, int2 *Encpairs2_d, int Nsmall, int NencMax, int UseTestParticles){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	double3 a;
	double3 b;

	a.x = 0.0;
	a.y = 0.0;
	a.z = 0.0;

	b.x = 0.0;
	b.y = 0.0;
	b.z = 0.0;

	int NencpairsI = 0;

	double4 x4i;
	double rcritvi;
	double test;

	if(id < N + Nsmall){
		x4i = x4_d[id];
		rcritvi = rcritv_d[id];
	}

	if(UseTestParticles == 2){
		if(id < N){
			for(int j = N; j < N + Nsmall; ++j){
				acc_d <E + 10> (a, b, x4i, x4_d[j], rcritvi, rcritv_d[j], &NencpairsI, Encpairs2_d, j, id, NencMax, test);
			}
		}
	}
	if(id < N + Nsmall){
		for(int j = 0; j < N; ++j){
			acc_d <E + 10> (a, b, x4i, x4_d[j], rcritvi, rcritv_d[j], &NencpairsI, Encpairs2_d, j, id, NencMax, test);
		}
		if(E >= 1){
			v4_d[id].x += __dmul_rn(a.x, dtksq);
			v4_d[id].y += __dmul_rn(a.y, dtksq);
			v4_d[id].z += __dmul_rn(a.z, dtksq);
//if(id == 25) printf("K %d %.40g %.40g %.40g %.40g\n", id, v4_d[id].x, v4_d[id].y, v4_d[id].z, a.z);
		}
		if(E <= 1){
			acck_d[id].x += b.x;
			acck_d[id].y += b.y;
			acck_d[id].z += b.z;
//printf("aa %d %d %.20g %.20g %.20g\n", E, id, acck_d[id].x, acck_d[id].y, acck_d[id].z);
		}
		if(E <= 2){
			Encpairs2_d[NencMax * id].x = NencpairsI;
			for(int ii = 0; ii < NencpairsI; ++ii){
				int jj = Encpairs2_d[id * NencMax + ii].y;
				if(id > jj){
					int Ne = atomicAdd(Nencpairs_d, 1);
					Encpairs_d[Ne].x = id;
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
__global__ void KickM2_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double *rcritv_d, int *Nencpairs_d, int2 *Encpairs_d, double *dt_d, double Kt, int *index_d, int NT, double *test_d, int Nstart){

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
		st_s[idy] = index_d[id] / 100;
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
			st_s[idy + Bl] = index_d[id + Bl] / 100;
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
//printf("precheck %d %d %d %d %d\n", st_s[idy], id, id + j, index_d[id], index_d[id + j]);	
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
//printf("acc %d %.20g %.20g %.20g %.20g %.20g %.20g %g\n", id, v4_d[id].x, v4_d[id].y, v4_d[id].z, acck_d[id].x, acck_d[id].y, acck_d[id].z, x4_d[id].w);
		}
	}
}

template <int Bl, int Bl2, int Nmax, int E>
__global__ void KickM2Simple_kernel(double4 *x4_d, double4 *v4_d, double4 *v4b_d, double3 *acck_d, double *dt_d, double Kt, int *index_d, int NT, double *test_d, int Nst, int Nstart){

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
		st_s[idy] = index_d[id] / 100;
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
			st_s[idy + Bl] = index_d[id + Bl] / 100;
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

	volatile double a;
	volatile double b;
	volatile double rx, ry, rz;
	volatile double rsq, ir, ir3;
	volatile double y, yy, K;
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
__global__ void KickM2TTV_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double *rcritv_d, int *Nencpairs_d, int2 *Encpairs_d, double *dt_d, double Kt, int *index_d, int NT, double *test_d, double4 *Msun_d, int *Ntransit_d, int *Transit_d, int Nstart){

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
		st_s[idy] = index_d[id] / 100;
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
			st_s[idy + Bl] = index_d[id + Bl] / 100;
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
#endif

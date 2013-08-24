#ifndef ENCOUNTER_H
#define ENCOUNTER_H

#include "Host2.h"
#include "Orbit2.h"
#endif




// **************************************
//This function estimates the minimal separation of two bodies
//during a time step, using a third order interpolation. 
//
//The interpolation scheme is based on the mercury code from Chambers.
//
//If the minimal separation is less than the critical radius, a
//close encounter is reported.
//
//E = 0: Used for Critical Radius 
//E = 1: Used for Physical Radius 
//E = 2: Udes for Critical Radius with Test Particles
//Code is adapted from Mercury
// ****************************************
#ifndef ENCOUNTER
#define ENCOUNTER
template<int E>
__device__ int encouter(double4 x4i, double4 v4i, double4 x4oldi, double4 v4oldi, double4 x4j, double4 v4j, double4 x4oldj, double4 v4oldj, double rcriti, double rcritj, double rcritvi, double rcritvj, double dt, int i, int j, double *test_d, int2 *encpairs, int &Nenc, int N){

//if(E == 0 || E >= 2)printf("E %d %d\n", i,j);
//if(E == 1) printf("E1 %d %d\n", i, j);

	int Enc = 0;
	if(i < j && (x4i.w > 0.0 || x4j.w > 0.0) && x4i.w >= 0.0 && x4j.w >= 0.0){
		double d0, d1, dd0, dd1;
		double4 r1, r0;
		double4 rd0, rd1;
		double a,b,c,cc;
		double w,q;
		double t1,t2,t12,t22,tt1,tt2,tt12,tt22;
		double delta1, delta2;
		double delta;
		double sgnb;
		double rcrit;
		double rcritv;
		int Ni;
		double f;
	
		if(E == 0 || E == 3){
			rcrit = fmax(rcriti, rcritj);
			rcritv = fmax(rcritvi, rcritvj);
			f = cef;
		}
		if(E == 1){
			rcrit = 0.0;
			rcritv = rcriti + rcritj;
			f = 1.0;
		}
		if(E == 2){
			rcrit = rcritj;
			rcritv = rcritvj;
			f = cef;
		}

		r1.x = x4j.x - x4i.x;
		r1.y = x4j.y - x4i.y;
		r1.z = x4j.z - x4i.z;
		d1 = r1.x*r1.x + r1.y*r1.y+ r1.z*r1.z;

		r0.x = x4oldj.x - x4oldi.x;
		r0.y = x4oldj.y - x4oldi.y;
		r0.z = x4oldj.z - x4oldi.z;
		d0 = r0.x*r0.x + r0.y*r0.y+ r0.z*r0.z;
			
		rd0.x = v4oldj.x - v4oldi.x;
		rd0.y = v4oldj.y - v4oldi.y;
		rd0.z = v4oldj.z - v4oldi.z;

		rd1.x = v4j.x - v4i.x;
		rd1.y = v4j.y - v4i.y;
		rd1.z = v4j.z - v4i.z;

		dd0 = (r0.x*rd0.x + r0.y*rd0.y+ r0.z*rd0.z) * 2.0;
		dd1 = (r1.x*rd1.x + r1.y*rd1.y+ r1.z*rd1.z) * 2.0;
		t1 = 6.0 *(d0-d1); 
		a = t1 + 3.0*dt*(dd0+dd1);
		b = -t1 - 2.0*dt*(2.0*dd0+ dd1);
		c = dt*dd0;
		cc = dt*dd1;

		if(b < 0){
			sgnb = -1.0;
		}
		else sgnb = 1.0;
		t1 = 0.0;
		t2 = 0.0;

		w = b*b - 4.0*a*c;
		if(w < 0.0) w = 0.0;
		if( b != 0){
			q = -0.5 * (b + sgnb * sqrt(w));
			if(q != 0){
				if( a != 0){
					t1 = q/a;
					t2 = c/q;
				}
				else{
					t1 = -c/b;
					t2 = t1;
				}
			}	
		}
		else{
			if( a != 0){
				t1 = sqrt(-c/a);
				t2 = -t1;
			}
		}

		if(0 <= t1 && t1 <= 1){
			t12 = t1*t1;
			tt1 = 1.0-t1;
			tt12 = tt1*tt1;
			delta1 = tt12*(1.0 + 2.0*t1)*d0 + t12*(3.0 - 2.0*t1)*d1 + t1*tt12*c - t12*tt1*cc;
		}
		else delta1 = Rcut;
		if(0 <= t2 && t2 <= 1){
			t22 = t2*t2;
			tt2 = 1.0-t2;
			tt22 = tt2*tt2;
			delta2 = tt22*(1.0 + 2.0*t2)*d0 + t22*(3.0 - 2.0*t2)*d1 + t2*tt22*c - t22*tt2*cc;
		}
		else delta2 = Rcut;

		delta = min(delta1,delta2);
		if(delta < 0) delta = 0.0;
		
		delta = fmin(delta, d1);
		delta = fmin(delta, d0);

		if(delta < f * rcritv*rcritv){
			Enc = 2;
//if (E == 0 || E >= 2)printf("EE %d %d %.40g %.40g %.40g %.40g\n", i, j, x4i.x, x4j.x, v4i.x, v4j.x);
//if (E == 1)printf("EE1 %d %d %.40g %.40g %.40g %.40g\n", i, j, x4i.x, x4j.x, v4i.x, v4j.x);
			if(E < 2){ 
				Ni = atomicAdd(&Nenc, 1);
				if(x4i.w >= x4j.w){
					encpairs[Ni].x = i;
					encpairs[Ni].y = j;
				}
				else{
                                	encpairs[Ni].x = j;
                                	encpairs[Ni].y = i;
                        	}
			}
			if(E == 2){ //used for collision detetion
				
				Ni = atomicAdd(&Nenc, 1);	
				encpairs[Ni].x = i;
                                encpairs[Ni].y = j - N;	
			}
		}
		else Enc = 0;
		if(delta < rcrit*rcrit){
			Enc = 1;
		}
		return Enc;
	}
	else return 0;
}
extern template __device__ int encouter < 0 > (double4, double4, double4, double4, double4, double4, double4, double4, double, double, double, double, double, int, int, double *, int2 *, int &, int);
extern template __device__ int encouter < 1 > (double4, double4, double4, double4, double4, double4, double4, double4, double, double, double, double, double, int, int, double *, int2 *, int &, int);
extern template __device__ int encouter < 2 > (double4, double4, double4, double4, double4, double4, double4, double4, double, double, double, double, double, int, int, double *, int2 *, int &, int);
#endif


extern __global__ void encounter_kernel(double4 *, double4 *, double4 *, double4 *, double *, double *, double, int *, int2 *, int *, int2 *, double *, int *, int);
extern __global__ void encountersmall_kernel(double4 *, double4 *, double4 *, double4 *, double *, double *, double, int *, int2 *, int *, int2 *, double *, double4 *, double4 *, double4 *, double4 *, int *, int2 *, int *, int2 *, int, int *, int *, int);
extern __global__ void encounterM_kernel(double4 *, double4 *, double4 *, double4 *, double *, double *, double, int *, int2 *, int *, int2 *, double *, int *, int *, int *, int, double, int);

extern __global__ void groupsmall1_kernel(int *, int2 *, const int, int);
extern __global__ void groupsmall2_kernel(int *, int2 *, int *, int2 *, const int);
extern __global__ void groupsmall3_kernel(int *, int2 *, int2 *, const int);
// **************************************
//This Kernel sorts all close encounter pairs into independent groups, using a 
//parallel sorting algorithm. 
//This kernel works only in the case of less than 512 close encounter pairs, and 
//less than 1025 Bodies.
//It classifies the groups into sets of equal sizes.
//The size of group i is stored in Encpairs2_d[i].y, the elements j of the 
//group i are stored in Encpairs2_d[i * BN + j].x
//In Nenc_d[0] is stored the total number of groups.
//in Nenc_d[i] is stored the number of groups with: 2^(2-1) < size of group < 2^(2+1)
//
//This Kernel must be launched only with one block!.
// ****************************************
#ifndef GROUP16
#define GROUP16
template <int BN, int Bl>
__global__ void group_kernel16(int *Nenc_d, double *test_d, int *Nencpairs2_d, int2 *Encpairs2_d){

	int idy = threadIdx.x;

	__shared__ int2 encpairs_s[Bl];
	__shared__ int A_s[Bl];
	__shared__ int AOld_s[Bl];
	__shared__ int B_s[BN];
	__shared__ int B2_s[BN];
	__shared__ volatile int T_s;
	__shared__ int Nenc_s[12];

	int Ne = *Nencpairs2_d;
	int BN2 = BN * BN -1;
	__syncthreads();

	if(idy == 0){
		T_s = 1;
	}
	if(idy < 12) Nenc_s[idy] = 0;
	__syncthreads();

	if(idy < Ne){ 
		encpairs_s[idy] = Encpairs2_d[idy];
		A_s[idy] = encpairs_s[idy].x;
	}
	/*encpairs_s[idy] contains the two close encounter pairs*/
	else{
		encpairs_s[idy].x = -1;
		encpairs_s[idy].y = -1;
		A_s[idy] = -1;
	}
	if(idy < BN){
		B_s[idy] = BN2;
		B2_s[idy] = BN2;
	}
	__syncthreads();

	AOld_s[idy] = A_s[idy];

	__syncthreads();

	for(int tt = 0; tt < 100; ++ tt){ 
		T_s = 0;
		if(idy < Ne){
			if (A_s[idy] < B_s[encpairs_s[idy].x]) atomicMin(&B_s[encpairs_s[idy].x], A_s[idy]);
		}
		__syncthreads();
		if(idy < Ne){
			if (A_s[idy] < B_s[encpairs_s[idy].y]) atomicMin(&B_s[encpairs_s[idy].y], A_s[idy]);
		}
		__syncthreads();

		if(idy < BN ){
			if(B_s[idy] < BN2) B2_s[idy] = B_s[B_s[idy]];
		}
		__syncthreads();
		if(A_s[idy] > -1) A_s[idy] = min(B2_s[encpairs_s[idy].x], B2_s[encpairs_s[idy].y]);
		__syncthreads();
		if(AOld_s[idy] != A_s[idy]) T_s = 1;
		__syncthreads();
		if(idy < BN){
			B_s[idy] = B2_s[idy];
		}
		AOld_s[idy] = A_s[idy];
		__syncthreads();
		if(T_s == 0) break;
		__syncthreads();

	}
	// *At this point B_s[idy] contains the smallest index of the group* /
	__syncthreads();

	if(idy < BN) B2_s[idy] = -1;
	__syncthreads();
	// *Check now for new groups and increase the total number of groups* /
	if(idy < BN){
		if(B_s[idy] == idy){
			B2_s[idy] =  atomicAdd(&Nenc_s[0],1);
		}		
	}
	__syncthreads();
	// *Transform now the smallest index of the group into a consecutive group index* /
	if(idy < BN){
		if(B_s[idy] < BN2) B_s[idy] = B2_s[B_s[idy]];
		encpairs_s[idy].y = 0;
	}
	// *At this point B_s[idy] contains a consecutive group index* /
	__syncthreads();
	if(idy < BN){
		if(B_s[idy] < BN2){
			Encpairs2_d[B_s[idy] * BN + atomicAdd(&encpairs_s[B_s[idy]].y,1)].x = idy;
		}
                // *At this point Encpairs2_d.x contains now line by line the members of the groups, encpairs_s.y contains the sizes of the groups* /
	}
//********* //This pice of code is usefull for a serial version of the grouping	
//	if(idy == 0){
//		for(int i = BN - 1; i >=0; --i){
//			if(B_s[i] < BN2){
//				Encpairs2_d[B_s[i] * BN + atomicAdd(&encpairs_s[B_s[i]].y,1)].x = i;
//			}
//		}
//	}
//********
	__syncthreads();

	if(idy < BN){
		int nn = encpairs_s[idy].y;
		int ne2 = 2;
		if(nn > 0){
			for(int ii = 0; ii < 11; ++ii){
				if(nn <= ne2){
					Encpairs2_d[ (ii+1) * BN + atomicAdd(&Nenc_s[ii + 1],1)].y = idy;
					break;
				} 
				else{
					ne2 *= 2;
				}
			}
		}
	}
	__syncthreads();

	if(idy < BN){
		Encpairs2_d[idy].y = encpairs_s[idy].y;
	}

	if(idy < 12){
		Nenc_d[idy] = Nenc_s[idy];
	}

}
extern template __global__ void group_kernel16 < 16, 512 > (int *, double *, int *, int2 *);
extern template __global__ void group_kernel16 < 32, 512 > (int *, double *, int *, int2 *);
extern template __global__ void group_kernel16 < 64, 512 > (int *, double *, int *, int2 *);
extern template __global__ void group_kernel16 < 128, 512 > (int *, double *, int *, int2 *);
extern template __global__ void group_kernel16 < 256, 512 > (int *, double *, int *, int2 *);
extern template __global__ void group_kernel16 < 512, 512 > (int *, double *, int *, int2 *);
#endif


// **************************************
//This Kernel sorts all close encounter pairs into independent groups, using a 
//parallel sorting algorithm. 
//This kernel works in the case of more than 512 bodies. 
//It has to functionalities:
//
//For less than 513 close encounters:
//It classifies the groups into sets of equal sizes.
//The size of group i is stored in Encpairs2_d[i].y, the elements j of the 
//group i are stored in Encpairs2_d[i * BN + j].x
//In Nenc_d[0] is stored the total number of groups.
//in Nenc_d[i] is stored the number of groups with: 2^(2-1) < size of group < 2^(2+1)
//
//For more than 512 cose encounters:
//it sorts a sub-set of close encounter pairs into independent groups.
//
//This Kernel is launched with Ne/512 blocks a 512 threads, with Ne the number of close encounters.
//For more than 512 close encounter, a Fusion Kernel has to be called to merger the sub sets. 
// ****************************************
#ifndef GROUP1024
#define GROUP1024
template <int BN, int Bl>
__global__ void group1024_kernel(int *Nenc_d, double *test_d, int *Nencpairs2_d, int2 *Encpairs2_d, int2 *Encpairs_d, double4 *x4_d, double *rcrit_d){

	int idy = threadIdx.x;
	int idx = blockIdx.x;
	int id = idx * blockDim.x + idy;
	int iy = idx * BN + idy;

	__shared__ int2 encpairs_s[Bl];
	__shared__ int A_s[Bl];
	__shared__ int AOld_s[Bl];
	__shared__ volatile int T_s;
	__shared__ int Nenc_s[12];
	
	int BN2 = BN * BN -1;

	int Ne = *Nencpairs2_d;
	__syncthreads();

	if(idy == 0){
		T_s = 1;
	}
	if(idy < 12) Nenc_s[idy] = 0;
	__syncthreads();
	

	if(id < Ne){ 
		encpairs_s[idy] = Encpairs2_d[id];
		A_s[idy] = encpairs_s[idy].x;
	}
	/*encpairs_s[idy] contains the two close encounter pairs*/
	else{
		encpairs_s[idy].x = -1;
		encpairs_s[idy].y = -1;
		A_s[idy] = -1;
	}
	for(int i = 0; i < BN; i += Bl){
		Encpairs_d[iy + i].x = BN2; //B
		Encpairs_d[iy + i].y = BN2; //B2
	}
	__syncthreads();

	AOld_s[idy] = A_s[idy];

	__syncthreads();

	for(int tt = 0; tt < 100; ++ tt){ 
		T_s = 0;	
		if(id < Ne){
			if (A_s[idy] < Encpairs_d[idx * BN + encpairs_s[idy].x].x){
				atomicMin(&Encpairs_d[idx * BN + encpairs_s[idy].x].x, A_s[idy]);
			}
		}
		__syncthreads();

		if(id < Ne){	
			if (A_s[idy] < Encpairs_d[idx * BN + encpairs_s[idy].y].x){
				atomicMin(&Encpairs_d[idx * BN + encpairs_s[idy].y].x, A_s[idy]);
			}
		}

		__syncthreads();
		for(int i = 0; i < BN; i += Bl){
			if(Encpairs_d[iy + i].x < BN2){
				Encpairs_d[iy + i].y = Encpairs_d[idx * BN + Encpairs_d[iy + i].x].x;
			}
		}

		__syncthreads();
		if(id < Ne){
			if(A_s[idy] > -1) A_s[idy] = min(Encpairs_d[idx * BN + encpairs_s[idy].x].y, Encpairs_d[idx * BN + encpairs_s[idy].y].y);
		}
		__syncthreads();

		if(AOld_s[idy] - A_s[idy] != 0){
			T_s = 1;
		}
		__syncthreads();
		for(int i = 0; i < BN; i += Bl){
			Encpairs_d[iy + i].x = Encpairs_d[iy + i].y;
		}
		AOld_s[idy] = A_s[idy];
		__syncthreads();
		if(T_s == 0) break;
		__syncthreads();
	
	}
	/*At this point Encpairs_d[iy + i].x contains the smallest index of the group*/
	__syncthreads();
	for(int i = 0; i < BN; i += Bl){
		Encpairs_d[iy + i].y = -1;
		if(gridDim.x == 1) Encpairs2_d[idy + i].y = 0;
	}
	__syncthreads();
	/*Check now for new groups and increase the total number of groups*/
	for(int i = 0; i < BN; i += Bl){
		if(Encpairs_d[iy + i].x == (idy + i)){
			Encpairs_d[iy + i].y = atomicAdd(&Nenc_s[0],1);
		}		
	}
	__syncthreads();
	/*Transform now the smallest index of the group into a consecutive group index*/
	for(int i = 0; i < BN; i += Bl){
	      if(Encpairs_d[iy + i].x < BN2){
			Encpairs_d[iy + i].x = Encpairs_d[idx * BN + Encpairs_d[iy + i].x].y + Bl * idx;
		}
	}
	/*At this point Encpairs_d[iy + i].x contains a consecutive group index*/
	__syncthreads();
	if(gridDim.x == 1){
		for(int i = 0; i < BN; i += Bl){
			if(Encpairs_d[idy + i].x < BN2){
				Encpairs2_d[Encpairs_d[idy + i].x * BN + atomicAdd(&Encpairs2_d[Encpairs_d[idy + i].x].y,1)].x = idy + i;
			}
			/*At this point Encpairs2_d.x contains now line by line the members of the groups, Encpsirs2_d.y contains the sizes of the groups*/

			__syncthreads();
		}
		__syncthreads();

__syncthreads();
		for(int i = 0; i < BN; i += Bl){
			int nn = Encpairs2_d[idy + i].y;
//			if(Encpairs_d[idy + i].x < BN2) nn = Encpairs2_d[Encpairs_d[idy + i].x].y;
//			else nn = 0;
			int ne2 = 2;
			if(nn > 0){
				for(int ii = 0; ii < 11; ++ii){
					if(nn <= ne2){
						Encpairs2_d[ (ii+1) * BN + atomicAdd(&Nenc_s[ii + 1], 1)].y = idy + i;
						break;
					}
					else{
						ne2 *= 2;
					}
				}
			} 
		}
		__syncthreads();
		if(idy < 12){
			Nenc_d[idy] = Nenc_s[idy];
		}
	}
	else{
		for(int i = 0; i < BN; i += Bl){
			Encpairs_d[iy + i].y = BN2;
		}
	}
}
extern template __global__ void group1024_kernel < 1024, 512 > (int *, double *, int *, int2 *, int2 *, double4 *, double *);
extern template __global__ void group1024_kernel < 2048, 512 > (int *, double *, int *, int2 *, int2 *, double4 *, double *);
#endif

// **************************************
//This kernel merges differrent sub-sets of close encounter groups, usig
//a parallel sorting algorithm.
//It classifies the groups into sets of equal sizes.
//The size of group i is stored in Encpairs2_d[i].y, the elements j of the 
//group i are stored in Encpairs2_d[i * BN + j].x
//In Nenc_d[0] is stored the total number of groups.
//in Nenc_d[i] is stored the number of groups with: 2^(2-1) < size of group < 2^(2+1)
//
//This Kernel must be launched  with only one block!
// ****************************************
#ifndef FUSIONB
#define FUSIONB
template <int BN, int Bl> 
__global__ void fusionB_kernel(int *Nenc_d, int2 *Encpairs_d, int2 *Encpairs2_d, int NBlock, double *test_d, double4 *x4_d, double *rcrit_d){

	int idy = threadIdx.x;
	__shared__ int encpairs_s[BN];
	__shared__ int encpairs2_s[BN];
	__shared__ volatile int T_s;
	__shared__ int Nenc_s[12];
	
	 int BN2 = BN * BN -1;
	
	for(int i = 0; i < BN; i += Bl){
		encpairs_s[idy + i] = Encpairs_d[idy + i].x;
	}
	if(idy < 12) Nenc_s[idy] = 0;

	__syncthreads();
	for(int k = 1; k < NBlock; ++k){

		for(int i = 0; i < BN; i += Bl){
			encpairs2_s[idy + i] = Encpairs_d[k * BN + idy + i].x;
		}
		T_s = 1;
		for(int i = 0; i < BN; i += Bl){
			__syncthreads();
			if(encpairs_s[idy + i] >= BN2 && encpairs2_s[idy + i] < BN2){
				encpairs_s[idy + i] = encpairs2_s[idy + i];
			}
			if(encpairs2_s[idy + i] >= BN2 && encpairs_s[idy + i] < BN2){ 
				encpairs2_s[idy + i] = encpairs_s[idy + i];
			}
		}
		__syncthreads();

		for(int tt = 0; tt < 100; ++ tt){
			T_s = 0;

			for(int i = 0; i < BN; i += Bl){
				if(encpairs_s[idy + i] < BN2){
					atomicMin(&Encpairs_d[encpairs2_s[idy + i]].y, encpairs_s[idy + i]);
				}
			}
			__syncthreads();
			for(int i = 0; i < BN; i += Bl){
				if(encpairs_s[idy + i] < BN2){
					encpairs2_s[idy + i] = Encpairs_d[encpairs2_s[idy + i]].y;
				}
			}
			__syncthreads();
			for(int i = 0; i < BN; i += Bl){
				if(encpairs_s[idy + i] < BN2){
					atomicMin(&Encpairs_d[encpairs_s[idy + i]].y, encpairs2_s[idy + i]);
				}
			}
			__syncthreads();
			for(int i = 0; i < BN; i += Bl){
				if(encpairs_s[idy + i] < BN2){
					encpairs_s[idy + i] = Encpairs_d[encpairs_s[idy + i]].y;
				}
			}
			__syncthreads();
			for(int i = 0; i < BN; i += Bl){
				if(encpairs_s[idy + i] != encpairs2_s[idy + i]){
					T_s = 1;
				}
			}
			__syncthreads();
			if(T_s == 0) break;
			__syncthreads();

		}
	}

	__syncthreads();

	for(int k = 0; k < NBlock; ++k){
		for(int i = 0; i < BN; i += Bl){
			Encpairs_d[k * BN + idy + i].y = BN2;
			Encpairs2_d[k * BN + idy + i].x = BN2;
		}
	}
	
	__syncthreads();
	for(int i = 0; i < BN; i += Bl){
		encpairs2_s[idy + i] = 0;
		if(encpairs_s[idy + i] < BN2){
			Encpairs2_d[encpairs_s[idy + i]].x = -5;
		}
	}
	__syncthreads();

	for(int k = 0; k < NBlock; ++k){
		for(int i = 0; i < BN; i += Bl){
			if(Encpairs2_d[k * BN + idy + i].x == -5){
				Encpairs_d[k * BN + idy + i].y = atomicAdd(&Nenc_s[0], 1);
			}
		}
	}
	__syncthreads();


	for(int i = 0; i < BN; i += Bl){
		if(encpairs_s[idy + i] < BN2){	
			encpairs_s[idy + i] = Encpairs_d[encpairs_s[idy + i]].y;
		}	
	}
	__syncthreads();

	for(int i = 0; i < BN; i += Bl){
		if(encpairs_s[idy + i] < BN2){
			Encpairs2_d[encpairs_s[idy + i] * BN + atomicAdd(&encpairs2_s[encpairs_s[idy + i]],1)].x = idy + i; 
		}
	}
	__syncthreads();
	
	for(int i = 0; i < BN; i += Bl){
		int nn = encpairs2_s[idy + i];
		//if(encpairs_s[idy + i] < BN2) nn = encpairs2_s[encpairs_s[idy + i]];
		//else nn = 0;
		Encpairs2_d[idy + i].y = encpairs2_s[idy + i];
		int ne2 = 2;
		if(nn > 0){
			for(int ii = 0; ii < 11; ++ii){
				if(nn <= ne2){
					Encpairs2_d[ (ii+1) * BN + atomicAdd(&Nenc_s[ii + 1], 1)].y = idy + i;
					break;
				}
				else{
					ne2 *= 2;
				}
			}
		}
 
	}
		__syncthreads();
	
	if(idy < 12){
		Nenc_d[idy] = Nenc_s[idy];
	}
}
extern template __global__ void fusionB_kernel < 2048, 512 > (int *, int2 *, int2 *, int, double *, double4 *, double *);
#endif

#ifndef GROUPM1
#define GROUPM1
template <int BN, int Bl>
__global__ void groupM1_kernel(int *Nencpairs2_d, int2 *Encpairs_d, int2 *Encpairs2_d, int *NBS_d, int *N_d, int Nst){

	int idy = threadIdx.x;
	int idx = blockIdx.x;

	int st = Encpairs_d[idx].y;

	__shared__ int2 encpairs_s[Bl];
	__shared__ int A_s[Bl];
	__shared__ int AOld_s[Bl];
	__shared__ int B_s[BN];
	__shared__ int B2_s[BN];
	__shared__ volatile int T_s;
	__shared__ int Nenc_s;

	int NBS = NBS_d[st];
	int N = N_d[st];

	int Ne = Nencpairs2_d[st + 1];
	int BN2 = BN * BN - 1;
	__syncthreads();
//if(idy == 0) printf("G %d %d %d\n", idx, st, Ne);

	if(idy == 0){
		T_s = 1;
		Nenc_s = 0;
		
	}
	__syncthreads();

	if(idy < Ne){ 
		encpairs_s[idy].x = Encpairs2_d[idy + NBS * 16].x - NBS;
		encpairs_s[idy].y = Encpairs2_d[idy + NBS * 16].y - NBS;
		A_s[idy] = encpairs_s[idy].x;
	}
	//encpairs_s[idy] contains the two close encounter pairs//
	else{
		encpairs_s[idy].x = -1;
		encpairs_s[idy].y = -1;
		A_s[idy] = -1;
	}
	if(idy < BN){
		B_s[idy] = BN2;
		B2_s[idy] = BN2;
	}
	__syncthreads();

	AOld_s[idy] = A_s[idy];

	__syncthreads();

	for(int tt = 0; tt < 100; ++ tt){
		T_s = 0;
		if(idy < Ne){
			if (A_s[idy] < B_s[encpairs_s[idy].x]) atomicMin(&B_s[encpairs_s[idy].x], A_s[idy]);
		}
		__syncthreads();
		if(idy < Ne){
			if (A_s[idy] < B_s[encpairs_s[idy].y]) atomicMin(&B_s[encpairs_s[idy].y], A_s[idy]);
		}
		__syncthreads();

		if(idy < BN ){
			if(B_s[idy] < BN2) B2_s[idy] = B_s[B_s[idy]];
		}
		__syncthreads();
		if(A_s[idy] > -1) A_s[idy] = min(B2_s[encpairs_s[idy].x], B2_s[encpairs_s[idy].y]);
		__syncthreads();
		if(AOld_s[idy] != A_s[idy]) T_s = 1;
		__syncthreads();
		if(idy < BN){
			B_s[idy] = B2_s[idy];
		}
		AOld_s[idy] = A_s[idy];
		__syncthreads();
		if(T_s == 0) break;
		__syncthreads();

	}
	//At this point B_s[idy] contains the smallest index of the group//
	__syncthreads();
	if(idy < BN) B2_s[idy] = -1;
	__syncthreads();
	//Check now for new groups and increase the total number of groups//
	if(idy < BN){
		if(B_s[idy] == idy){
			B2_s[idy] =  atomicAdd(&Nenc_s,1);
		}		
	}
	__syncthreads();
	//Transform now the smallest index of the group into a consecutive group index//
	if(idy < BN){
		if(B_s[idy] < BN2) B_s[idy] = B2_s[B_s[idy]];
		encpairs_s[idy].y = 0;
	}
	//At this point B_s[idy] contains a consecutive group index//
	__syncthreads();
	if(idy < BN){
		if(B_s[idy] < BN2){
			int ne = atomicAdd(&encpairs_s[B_s[idy]].y,1);
			Encpairs_d[(B_s[idy] + NBS) * BN + ne].x = idy + NBS;
		}

		//At this point Encpairs_d.x contains now line by line the members of the groups, encpairs_s.y contains the sizes of the groups//
	}
	__syncthreads();

	if(idy < N){
		Encpairs_d[idy + NBS + Nst].y = encpairs_s[idy].y;
//printf("S %d\n", encpairs_s[idy].y);
	}
}
extern template __global__ void groupM1_kernel < 16, 256 > (int *, int2 *, int2 *, int *, int *, int);
#endif

extern __global__ void groupM2_kernel(int2 *, int2 *, int *, int *, int *, int);


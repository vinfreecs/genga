#include "Encounter3.h"

template __device__ int encouter < 0 > (double4, double4, double4, double4, double4, double4, double4, double4, double, double, double, double, double, int, int, double *, int2 *, int &, int);
template __device__ int encouter < 1 > (double4, double4, double4, double4, double4, double4, double4, double4, double, double, double, double, double, int, int, double *, int2 *, int &, int);
template __device__ int encouter < 2 > (double4, double4, double4, double4, double4, double4, double4, double4, double, double, double, double, double, int, int, double *, int2 *, int &, int);

__global__ void encountersmall_kernel(double4 *x4_d, double4 *v4_d, double4 *xold_d, double4 *vold_d, double *rcrit_d, double *rcritv_d, double dt, int *Nencpairs_d, int2 *Encpairs_d, int *Nencpairs2_d, int2 *Encpairs2_d, double *test_d, double4 *x4small_d, double4 *v4small_d, double4 *xoldsmall_d, double4 *voldsmall_d, int *Nencpairssmall_d, int2 *Encpairssmall_d, int *Nencpairssmall2_d, int2 *Encpairssmall2_d, int N, int *enccount_d, int *enccountsmall_d, int si){
	int idy = threadIdx.x;
	int idx = blockIdx.x;
	int id = idx * blockDim.x + idy;

	int ii = 0;
	int jj = 0;

	int enccount = 0;

	if(id < *Nencpairs_d){
		ii = Encpairs_d[id].x;
		jj = Encpairs_d[id].y;
	}
	else if(id < *Nencpairs_d + *Nencpairssmall_d){
		ii = Encpairssmall_d[id - *Nencpairs_d].x;
		jj = Encpairssmall_d[id - *Nencpairs_d].y;
	}
	__syncthreads();
	
	if(id < *Nencpairs_d){
		enccount = encouter<0>(x4_d[ii], v4_d[ii], xold_d[ii], vold_d[ii], x4_d[jj], v4_d[jj], xold_d[jj], vold_d[jj], rcrit_d[ii], rcrit_d[jj], rcritv_d[ii], rcritv_d[jj], dt, ii, jj , test_d, Encpairs2_d, *Nencpairs2_d, 0);
		if(si == 0 && enccount > 0){
			atomicAdd(&enccount_d[ii], 1);
			atomicAdd(&enccount_d[jj], 1);
		}
	}
        else if(id < *Nencpairs_d + *Nencpairssmall_d){
                enccount = encouter<2>(x4small_d[ii], v4small_d[ii], xoldsmall_d[ii], voldsmall_d[ii], x4_d[jj], v4_d[jj], xold_d[jj], vold_d[jj], 0.0, rcrit_d[jj], 0.0, rcritv_d[jj], dt, jj, ii + N , test_d, Encpairssmall2_d, *Nencpairssmall2_d, N);
		if(si == 0 && enccount > 0){
			atomicAdd(&enccountsmall_d[ii], 1);
			atomicAdd(&enccountsmall_d[jj], 1);
		}
        }

}

__global__ void encounterM_kernel(double4 *x4_d, double4 *v4_d, double4 *xold_d, double4 *vold_d, double *rcrit_d, double *rcritv_d, double dt, int *Nencpairs_d, int2 *Encpairs_d, int *Nencpairs2_d, int2 *Encpairs2_d, double *test_d, int *index_d, int *NBS_d, int *enccount_d, int si, double FGt, int Nst){
	int idy = threadIdx.x;
	int idx = blockIdx.x;
	int id = idx * blockDim.x + idy;

	int ii = 0;
	int jj = 0;

	int st = 0;
	int NBS = 0;

	if(id < Nencpairs_d[0]){
		ii = Encpairs_d[id].x;
		jj = Encpairs_d[id].y;
		if(ii >= 0 && jj >= 0){
			st = index_d[ii] / 100;
			NBS = NBS_d[st];
		}
	}
	__syncthreads();

	if(id < Nencpairs_d[0] && ii >= 0 && jj >= 0 && st < Nst){
		int enccount = encouter<3>(x4_d[ii], v4_d[ii], xold_d[ii], vold_d[ii], x4_d[jj], v4_d[jj], xold_d[jj], vold_d[jj], rcrit_d[ii], rcrit_d[jj], rcritv_d[ii], rcritv_d[jj], dt * FGt, ii, jj , test_d, Encpairs2_d, Nencpairs2_d[st], 0);
//printf("enc %d %d %d %d %d\n", ii, jj, enccount, st, Nencpairs2_d[st + 1]);
		if(enccount > 0){
			int Ne = atomicAdd(&Nencpairs2_d[st + 1], 1);
			if(Ne == 0){
				//write a list with simulations containing close encounters
				int NT = atomicAdd(Nencpairs2_d, 1);
				Encpairs_d[NT].y = st;
			}
			if(x4_d[ii].w >= x4_d[jj].w){
				Encpairs2_d[Ne + NBS * 16].x = ii;
				Encpairs2_d[Ne + NBS * 16].y = jj;
			}
			else{
				Encpairs2_d[Ne + NBS * 16].x = jj;
				Encpairs2_d[Ne + NBS * 16].y = ii;
			}
		}
		if(si == 0 && enccount > 0){
			atomicAdd(&enccount_d[ii], 1);
			atomicAdd(&enccount_d[jj], 1);
		}
	}
}

// **************************************
//This reads all encounter pairs from the prechecker, and calls the encounter function
//to detect close encounter pairs.
//All close encounter pairs are stored in the array Encpairs2_d. 
//The number of close encounter pairs is stored in Nencpairs2_d.
// ****************************************
__global__ void encounter_kernel(double4 *x4_d, double4 *v4_d, double4 *xold_d, double4 *vold_d, double *rcrit_d, double *rcritv_d, double dt, int *Nencpairs_d, int2 *Encpairs_d, int *Nencpairs2_d, int2 *Encpairs2_d, double *test_d, int *enccount_d, int si){
	int idy = threadIdx.x;
	int idx = blockIdx.x;
	int id = idx * blockDim.x + idy;

	int ii = 0;
	int jj = 0;
	if(id < *Nencpairs_d){
		ii = Encpairs_d[id].x;
		jj = Encpairs_d[id].y;
	}
	__syncthreads();
	
	if(id < *Nencpairs_d){
		int enccount = encouter<0>(x4_d[ii], v4_d[ii], xold_d[ii], vold_d[ii], x4_d[jj], v4_d[jj], xold_d[jj], vold_d[jj], rcrit_d[ii], rcrit_d[jj], rcritv_d[ii], rcritv_d[jj], dt, ii, jj , test_d, Encpairs2_d, *Nencpairs2_d, 0);
		if(si == 0 && enccount > 0){
			atomicAdd(&enccount_d[ii], 1);
			atomicAdd(&enccount_d[jj], 1);
		}
	}
}

__global__ void groupsmall1_kernel(int *Nencpairssmall_d, int2 *Encpairssmall_d, const int Nconst, int Nsmall){
        int idx = blockIdx.x;
        int id = idx * blockDim.x + threadIdx.x;

	if(id == 0) *Nencpairssmall_d = 0;

	if(id < Nsmall){
		Encpairssmall_d[id * Nconst].y = 1;
		Encpairssmall_d[id * Nconst].x = id;
	}
}

__global__ void groupsmall2_kernel(int *Nencpairssmall2_d, int2 *Encpairssmall2_d, int *Nencsmall_d, int2 *Encpairssmall_d, const int Nconst){
	int idx = blockIdx.x;
	int id = idx * blockDim.x + threadIdx.x;

	if(id < *Nencpairssmall2_d){
		int i = Encpairssmall2_d[id].x;
		int j = Encpairssmall2_d[id].y;
		int Nj = atomicAdd(&Encpairssmall_d[j * Nconst].y, 1);

		Encpairssmall_d[j * Nconst + Nj].x = i;
		if(Nj == 1){
			int Ne = atomicAdd(&Nencsmall_d[0], 1);
			Encpairssmall_d[Ne * Nconst + 1].y = j;
		}
	}
}

__global__ void groupsmall3_kernel(int *Nencsmall_d, int2 *Encpairssmall_d, int2 *Encpairssmall2_d, const int Nconst){
        int idx = blockIdx.x;
        int id = idx * blockDim.x + threadIdx.x;

	if(id < Nencsmall_d[0]){
		int i = Encpairssmall_d[id * Nconst + 1].y;
		int nn = Encpairssmall_d[i * Nconst].y;
                int ne2 = 2;
                if(nn > 0){
                        for(int ii = 0; ii < 11; ++ii){
                                if(nn <= ne2){
					int Ne = atomicAdd(&Nencsmall_d[ii + 1], 1);
                                        Encpairssmall2_d[Ne * Nconst + ii].y = i;
                                        break;
                                }
                                else{
                                        ne2 *= 2;
                                }
                        }
                }
	}
}




template __global__ void group_kernel16 < 16, 512 > (int *, double *, int *, int2 *);
template __global__ void group_kernel16 < 32, 512 > (int *, double *, int *, int2 *);
template __global__ void group_kernel16 < 64, 512 > (int *, double *, int *, int2 *);
template __global__ void group_kernel16 < 128, 512 > (int *, double *, int *, int2 *);
template __global__ void group_kernel16 < 256, 512 > (int *, double *, int *, int2 *);
template __global__ void group_kernel16 < 512, 512 > (int *, double *, int *, int2 *);


//**************************************
//This Kernel sorts a sub-set of close encounter pairs into independent groups, using a 
//parallel sorting algorithm. 
//This kernel works in the case of more than 512 close encounter pairs, and 
//less than 1025 Bodies.
//
//This Kernel is launched with Ne/512 blocks a 512 threads, with Ne the number of close encounters.
//A Fusion Kernel has to be called to merger the sub sets.
//
//****************************************
template <int BN, int Bl>
__global__ void group512_kernel(double *test_d, int *Nencpairs2_d, int2 *Encpairs2_d, int2 *Encpairs_d){

	int idy = threadIdx.x;
	int idx = blockIdx.x;
	int id = idx * blockDim.x + idy;

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
	

	if(id  < Ne){ 
		encpairs_s[idy] = Encpairs2_d[id];
		A_s[idy] = encpairs_s[idy].x;
	}
	else{
		encpairs_s[idy].x = -1;
		encpairs_s[idy].y = -1;
		A_s[idy] = -1;
	}
	for(int i = 0; i < BN; i += Bl){
		B_s[idy + i] = BN2;
		B2_s[idy + i] = BN2;
	}
	__syncthreads();
	
	AOld_s[idy] = A_s[idy];

	for(int tt = 0; tt < 100; ++tt){
		T_s = 0;
		if(id < Ne){
			if (A_s[idy] < B_s[encpairs_s[idy].x]) atomicMin(&B_s[encpairs_s[idy].x], A_s[idy]);
		}
		__syncthreads();
		if(id < Ne){
			if (A_s[idy] < B_s[encpairs_s[idy].y]) atomicMin(&B_s[encpairs_s[idy].y], A_s[idy]);
		}
		__syncthreads();

		for(int i = 0; i < BN; i += Bl){
			if(B_s[idy + i] < BN2) B2_s[idy + i] = B_s[B_s[idy + i]];
		}
		__syncthreads();
		if(A_s[idy] > -1) {
			A_s[idy] = min(B2_s[encpairs_s[idy].x], B2_s[encpairs_s[idy].y]);
		}
		__syncthreads();

		if(AOld_s[idy] != A_s[idy]) T_s = 1;
		__syncthreads();
		for(int i = 0; i < BN; i += Bl){
			B_s[idy + i] = B2_s[idy + i];
		}
		AOld_s[idy] = A_s[idy];
		__syncthreads();
		if(T_s == 0) break;
		__syncthreads();
	}

	__syncthreads();

	for(int i = 0; i < BN; i += Bl){
		B2_s[idy + i] = -1;
	}
	__syncthreads();

	for(int i = 0; i < BN; i += Bl){
		if(B_s[idy + i] == idy + i){
			B2_s[idy + i] =  atomicAdd(&Nenc_s[0],1);
		}		
	}
	__syncthreads();

	for(int i = 0; i < BN; i += Bl){
		if(B_s[idy + i] < BN2){
			B_s[idy + i] = B2_s[B_s[idy + i]] + Bl * idx;
		}
		Encpairs_d[idx * BN + idy + i].x = B_s[idy + i];
		Encpairs_d[idx * BN + idy + i].y = BN2;
	}
}

template __global__ void group1024_kernel < 1024, 512 > (int *, double *, int *, int2 *, int2 *, double4 *, double *);
template __global__ void group1024_kernel < 2048, 512 > (int *, double *, int *, int2 *, int2 *, double4 *, double *);


template __global__ void groupM1_kernel < 16, 256 > (int *, int2 *, int2 *, int *, int *, int);
// **************************************
//This kernel merges two differrenit sub-sets from a tree of close encounter groups, usig
//a parallel sorting algorithm.
//This kernel works in the case of more than 1025 close encounter pairs, and 
//less than 1025 bodies.
//
//This Kernel is launched with Ne/512 blocks, with Ne the number of close encounters.
// ****************************************
template <int BN, int Bl>	//fusion tree
__global__ void fusionA_kernel(int2 *Encpairs_d, int2 *Encpairs2_d, int NB2, double *test_d){

	int idx = blockIdx.x;
	int idy = threadIdx.x;
	__shared__ int encpairs_s[BN];
	__shared__ int encpairs2_s[BN];
	__shared__ int T_s;
	
	int BN2 = BN * BN -1;
	
	for(int i = 0; i < BN; i += Bl){
		encpairs_s[idy + i] = Encpairs_d[idx * BN + idy + i].x;
		encpairs2_s[idy + i] = Encpairs_d[(idx + NB2) * BN + idy + i].x;
	}

	int m = 0;

	__syncthreads();
	T_s = 1;
	m = 0;
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

	while(T_s ==1 && m < 100){
		__syncthreads();
		++m;
		T_s = 0;
		__syncthreads();

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

	}
	for(int i = 0; i < BN; i += Bl){
		Encpairs_d[idx * BN + idy + i].x = encpairs_s[idy + i];
	}

}

// **************************************
//This kernel merges two differrenit sub-sets of close encounter groups, usig
//a parallel sorting algorithm.
//This kernel works in the case of less than 1025 close encounter pairs, and 
//less than 1025 bodies.
//It classifies the groups into sets of equal sizes.
//The size of group i is stored in Encpairs2_d[i].y, the elements j of the 
//group i are stored in Encpairs2_d[i * BN + j].x
//In Nenc_d[0] is stored the total number of groups.
//in Nenc_d[i] is stored the number of groups with: 2^(2-1) < size of group < 2^(2+1)
//
//This Kernel must be launched  with two blocks!
// ****************************************
template <int BN, int Bl>
__global__ void fusion_kernel(int *Nenc_d, int2 *Encpairs_d, int2 *Encpairs2_d, int NBlock, double *test_d){

	int idy = threadIdx.x;
	__shared__ int encpairs_s[BN];
	__shared__ int encpairs2_s[BN];
	__shared__ int T_s;
	__shared__ int Nenc_s[12];
	
	for(int i = 0; i < BN; i += Bl){
		encpairs_s[idy + i] = Encpairs_d[idy + i].x;
		encpairs2_s[idy + i] = Encpairs_d[BN + idy + i].x;
	}

	int m = 0;
	if(idy < 12) Nenc_s[idy] = 0;
	
	int BN2 = BN * BN -1;

	__syncthreads();
	T_s = 1;
	m = 0;
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

	while(T_s ==1 && m < 100){
		++m;
		T_s = 0;
		__syncthreads();

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
	if(idy < 12) Nenc_d[idy] = Nenc_s[idy];

}

template __global__ void fusionB_kernel < 2048, 512 > (int *, int2 *, int2 *, int, double *, double4 *, double *);


// **************************************
//This kernel merges two differrenit sub-sets from a tree of close encounter groups, usig
//a parallel sorting algorithm.
//This kernel works in the case of more than 1025 close encounter pairs, and 
//more than 1024 bodies.
//
//This Kernel is launched with Ne/512 blocks, with Ne the number of close encounters.
// ****************************************
template <int BN, int Bl>  //fusion tree for 2048 bodies
__global__ void fusionA2_kernel(int2 *Encpairs_d, int2 *Encpairs2_d, int NB2, double *test_d){

	int idx = blockIdx.x;
	int idy = threadIdx.x;

	int iy1 = idx * BN + idy;
	int iy2 = (idx + NB2) * BN + idy;
	__shared__ int T_s;
	
	int BN2 = BN * BN -1;
	
	if(idy == 0){
		T_s = 1;
	}
	__syncthreads();

	Encpairs_d[idx * Bl + idy].y = BN2;	
	Encpairs_d[(idx + NB2) * Bl + idy].y = BN2;

	for(int i = 0; i < BN; i += Bl){
		__syncthreads();

		if(Encpairs_d[iy1 + i].x >= BN2 && Encpairs_d[iy2 + i].x < BN2){
			Encpairs_d[iy1 + i].x = Encpairs_d[iy2 + i].x;
		}
		__syncthreads();
		if(Encpairs_d[iy2 + i].x >= BN2 && Encpairs_d[iy1 + i].x < BN2){
			Encpairs_d[iy2 + i].x = Encpairs_d[iy1 + i].x;
		}
	}
	__syncthreads();

	for(int tt = 0; tt < 100; ++ tt){
	if(T_s == 1){
		__syncthreads();
		T_s = 0;
		__syncthreads();
		for(int i = 0; i < BN; i += Bl){
			if(Encpairs_d[iy2 + i].x < BN2){
				atomicMin(&Encpairs_d[Encpairs_d[iy2 + i].x].y, Encpairs_d[iy1 + i].x);
			}
			__syncthreads();
		}
		__syncthreads();
		for(int i = 0; i < BN; i += Bl){
			if(Encpairs_d[iy2 + i].x < BN2){
				Encpairs_d[iy2 + i].x = Encpairs_d[Encpairs_d[iy2 + i].x].y;
			}
			__syncthreads();
		}
		__syncthreads();
		for(int i = 0; i < BN; i += Bl){
			if(Encpairs_d[iy1 + i].x < BN2){
				atomicMin(&Encpairs_d[Encpairs_d[iy1 + i].x].y, Encpairs_d[iy2 + i].x);
			}
			__syncthreads();
		}
		__syncthreads();
		for(int i = 0; i < BN; i += Bl){
			if(Encpairs_d[iy1 + i].x < BN2){
				Encpairs_d[iy1 + i].x = Encpairs_d[Encpairs_d[iy1 + i].x].y;
			}
			__syncthreads();
		}
		__syncthreads();
		for(int i = 0; i < BN; i += Bl){
			if(Encpairs_d[iy1 + i].x != Encpairs_d[iy2 + i].x){
				T_s = 1;
			}
			__syncthreads();
		}
		__syncthreads();
	}
	}
}

// **************************************
//This kernel merges two differrenit sub-sets of close encounter groups, usig
//a parallel sorting algorithm.
//This kernel works in the case of less than 1025 close encounter pairs, and 
//more than 1024 bodies.
//It classifies the groups into sets of equal sizes.
//The size of group i is stored in Encpairs2_d[i].y, the elements j of the 
//group i are stored in Encpairs2_d[i * BN + j].x
//In Nenc_d[0] is stored the total number of groups.
//in Nenc_d[i] is stored the number of groups with: 2^(2-1) < size of group < 2^(2+1)
//
//This Kernel must be launched  with two blocks!
// ****************************************
template <int BN, int Bl>
__global__ void fusion2_kernel(int *Nenc_d, int2 *Encpairs_d, int2 *Encpairs2_d, int NBlock, double *test_d){

	int idy = threadIdx.x;

	__shared__ int T_s;
	__shared__ int Nenc_s[12];

	if(idy < 12 ) Nenc_s[idy] = 0;
	
	int BN2 = BN * BN -1;
	
	__syncthreads();
	T_s = 1;
	for(int i = 0; i < BN; i += Bl){
		__syncthreads();
		if(Encpairs_d[idy + i].x >= BN2 && Encpairs_d[BN + idy + i].x < BN2){
			Encpairs_d[idy + i].x = Encpairs_d[BN + idy + i].x;
		}
		if(Encpairs_d[BN + idy + i].x >= BN2 && Encpairs_d[idy + i].x < BN2){
			Encpairs_d[BN + idy + i].x = Encpairs_d[idy + i].x;
		}
	}
	__syncthreads();

	for(int tt = 0; tt < 100; ++ tt){
	if(T_s == 1){
		__syncthreads();
		T_s = 0;
		__syncthreads();
		for(int i = 0; i < BN; i += Bl){
			if(Encpairs_d[idy + i].x < BN2){
				atomicMin(&Encpairs_d[Encpairs_d[BN + idy + i].x].y, Encpairs_d[idy + i].x);
			}
		}
		__syncthreads();
		for(int i = 0; i < BN; i += Bl){
			if(Encpairs_d[idy + i].x < BN2){
				Encpairs_d[BN + idy + i].x = Encpairs_d[Encpairs_d[BN + idy + i].x].y;
			}
		}
		__syncthreads();
		for(int i = 0; i < BN; i += Bl){
			if(Encpairs_d[idy + i].x < BN2){
				atomicMin(&Encpairs_d[Encpairs_d[idy + i].x].y, Encpairs_d[BN + idy + i].x);
			}
		}
		__syncthreads();
		for(int i = 0; i < BN; i += Bl){
			if(Encpairs_d[idy + i].x < BN2){
				Encpairs_d[idy + i].x = Encpairs_d[Encpairs_d[idy + i].x].y;
			}
		}
		__syncthreads();
		for(int i = 0; i < BN; i += Bl){
			if(Encpairs_d[idy + i].x != Encpairs_d[BN + idy + i].x){
				T_s = 1;
			}
		}
		__syncthreads();
	}
	}

	
	__syncthreads();

	for(int k = 0; k < NBlock; ++k){
		for(int i = 0; i < BN; i += Bl){
			Encpairs_d[k * BN + idy + i].y = BN2;
			Encpairs2_d[k * BN + idy + i].x = BN2;
			Encpairs2_d[k * BN + idy + i].y = 0;
		}
	}

	__syncthreads();
	for(int i = 0; i < BN; i += Bl){
		if(Encpairs_d[idy + i].x < BN2){
			Encpairs2_d[Encpairs_d[idy + i].x].x = -5;
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
		if(Encpairs_d[idy + i].x < BN2){	
			Encpairs_d[idy + i].x = Encpairs_d[Encpairs_d[idy + i].x].y;
		}	
	}
	__syncthreads();



	for(int i = 0; i < BN; i += Bl){
		if(Encpairs_d[idy + i].x < BN2){
			Encpairs2_d[Encpairs_d[idy + i].x * BN + atomicAdd(&Encpairs2_d[Encpairs_d[idy + i].x].y,1)].x = idy + i; 
		}
	}
	
	__syncthreads();

	for(int i = 0; i < BN; i += Bl){
		int nn = Encpairs2_d[idy + i].y;
		//if(encpairs_s[idy + i] < BN2) nn = encpairs2_s[encpairs_s[idy + i]];
		//else nn = 0;
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
	if(idy < 12) Nenc_d[idy] = Nenc_s[idy];


}



__global__ void groupM2_kernel(int2 *Encpairs_d, int2 *Encpairs2_d, int *Nenc_d, int *NBS_d, int *N_d, int Nst){

	int idy = threadIdx.x;
	int idx = blockIdx.x;

	int st = Encpairs_d[idx].y;

	int NBS = NBS_d[st];
	int N = N_d[st];

	if(idy < N){

		int nn = Encpairs_d[idy + NBS + Nst].y;
//printf("n %d %d %d %d\n", st, idy, nn, NBS);

		int ne2 = 2;
		if(nn > 0){
//printf("nn %d %d %d %d %d %d\n", st, idy, nn, Encpairs_d[(idy + NBS)* 16].x, Encpairs_d[((idy  + NBS)* 16)+ 1].x, Encpairs_d[((idy + NBS) * 16) + 2].x);
			for(int ii = 0; ii < 11; ++ii){
				if(nn <= ne2){
					Encpairs2_d[ (ii+1) + 16 * atomicAdd(&Nenc_d[ii + 1],1)].y = idy + NBS;
					break;
				} 
				else{
					ne2 *= 2;
				}
			}
		}
	}
}




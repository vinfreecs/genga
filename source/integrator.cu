#include "Orbit2.h"
#include "Kick3.h"
#include "HC.h"
#include "FG2.h"
#include "Encounter3.h"
#include "BSB.h"
#include "BSB64.h"
#include "BSBsmall.h"
#include "BSB64small.h" 
#include "BSBM.h"
#include "ComEnergy.h"

#if G3 == 1
//	#include "BSBKG3.h"
	#include "BSBG3.h"
#endif

#include "GENGAKick.h"
#include "Kick4.h"

int SIn;		//Number of direction steps
int SIM;		//half of steps
double *Ct;		//time factor for HC Kick steps
double *FGt;		//time factor for Drift steps
double *Kt;		//time factor for Kick steps

int EjectionFlag2 = 0;

// *****************************************************
//This function set the time factors fot the symplectic integrator for a given order
//Authors: Simon Grimm, Joachim Stadel
//March 2014
// ****************************************************
__host__ void  Data::SymplecticP(){
	SIn = 1;
	SIM = 1;
	double SIw[4]; //for maximal SI6

	//second order
	if(P.SIO == 2){
		//SI2
		SIn = 1;
		SIM = (SIn + 1) / 2;

		SIw[0] = 1.0;
	}
	//4th order
	//From Yoshida
	if(P.SIO == 4){
		//SI4
		SIn = 3;
		SIM = (SIn + 1) / 2;

		double two3r = cbrt(2.0);
		SIw[0] = - two3r / (2.0 - two3r);
		SIw[1] = 1.0 / (2.0 - two3r);
	
	}
	//6th order
	if(P.SIO == 6){
		//SI6
		SIn = 7;
		SIM = (SIn + 1) / 2;

		//Solution A from Yoshida
		SIw[1] = -0.117767998417887e1;
		SIw[2] = 0.235573213359357e0;
		SIw[3] = 0.784513610477560e0;
		SIw[0] = 1.0 - 2.0 * (SIw[1] + SIw[2] + SIw[3]);
	}
	Ct = (double*)malloc(SIn*sizeof(double));
	FGt = (double*)malloc(SIn*sizeof(double));
	Kt = (double*)malloc(SIn*sizeof(double));

	for(int sim = 0; sim < SIM; ++sim){
		FGt[sim] = SIw[SIM - sim - 1];
	}
	for(int sim = SIM; sim < SIn; ++sim){
		FGt[sim] = SIw[sim - SIM + 1];
	}

	for(int si = 0; si < SIn; ++si){
		Ct[si] = 0.5 * FGt[si];
	}
	for(int si = 0; si < SIn - 1; ++si){
		Kt[si] = 0.5 * (FGt[si] + FGt[si + 1]);
	}       
	Kt[SIn - 1] = 0.5 * FGt[SIn - 1];
}

// **************************************
// This kernel sets initial values for the Encouter pair arrays
//Authors: Simon Grimm, Joachim Stadel
//March 2014
// **************************************3
template <int Bl>
__global__ void initial_kernel(int2 *Encpairs_d, int2 *Encpairs2_d, double3 *acck_d, double *K_d, double *Kold_d, int *BddSign_d, double4 *StopTime_d, int *groupIndex_d, int NB){
	int idy = threadIdx.x;
	int idx = blockIdx.x;

	for(int i = 0; i < NB; i += Bl){
#if G3 == 1
		K_d[(idy +i)* NB + idx] = 1.0;
		Kold_d[(idy +i)* NB + idx] = 1.0;
		BddSign_d[(idy +i)* NB + idx] = 0;
		StopTime_d[(idy +i)* NB + idx].x = -1.0;
		StopTime_d[(idy +i)* NB + idx].y = -1.0;
		StopTime_d[(idy +i)* NB + idx].z = -1.0;
		StopTime_d[(idy +i)* NB + idx].w = -1.0;
#endif
		Encpairs_d[(idy +i)* NB + idx].x = -1;
		Encpairs_d[(idy +i)* NB + idx].y = -1;

		Encpairs2_d[(idy +i)* NB + idx].x = -1;
		Encpairs2_d[(idy +i)* NB + idx].y = -1;
		
		if(idx == 0){
			acck_d[idy + i].x = 0.0;
			acck_d[idy + i].y = 0.0;
			acck_d[idy + i].z = 0.0;
#if G3 == 1
			groupIndex_d[idy + i] = -1;
#endif
		}
	}
}

// **************************************
//This kernel sets initial values for the test particle mode
__global__ void initialsmall_kernel(int2 *Encpairs_d, int2 *Encpairs2_d, int NB, int2 *Encpairssmall_d, int2 *Encpairssmall2_d, int Nsmall, int *groupIndexsmall_d){

	int idx = blockIdx.x;
	int id = blockIdx.x * blockDim.x + idx;

	if(id < NB * NB){
		Encpairs_d[id].x = -1;
		Encpairs_d[id].y = -1;

		Encpairs2_d[id].x = -1;
		Encpairs2_d[id].y = -1;
	}

	if(id < Nsmall){
		for(int i = 0; i < 2 * NmaxTestParticles; ++i){
			Encpairssmall_d[id * 2 * NmaxTestParticles + i].x = -1;
			Encpairssmall_d[id * 2 * NmaxTestParticles + i].y = -1;
			Encpairssmall2_d[id * 2 * NmaxTestParticles + i].x = -1;
			Encpairssmall2_d[id * 2 * NmaxTestParticles + i].y = -1;
		}
	}
#if G3 == 1
	if(id < Nsmall){
		groupIndexsmall_d[id] = -1;
	}
#endif

}

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

/*
__global__ void test_kernel(double4 *x4_d, double3 *a_d, int *index_d, int N){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	if(id < N && fabs(a_d[id].x) > 10) printf("test %d %.g\n", id, a_d[id].x);

}
*/


// **************************************
//This kernel calculates the critical radius rcrit = max(n1 * Rh, n2 * dt * v), with the 
//Hill radius Rh = r * (m/(3Msun))^1/3, the velocity v and two constants n1 and  n2.
//rcritv is used for the the prechecker.
//In Rh we use the radius instead of the semi major axis.
//It searches also for ejections.
//This Kernel is launched wich NB/Bl blocks with Bl threads. NB is the next bigger number of N
//which is a power of two.
//
//Authors: Simon Grimm, Joachim Stadel
//March 2014
//****************************************/
template <int Bl>
__global__ void Rcrit_kernel(double4 *__restrict__ x4_d, double4 *__restrict__ v4_d, double4 * __restrict__ x4G3_d, double4 *__restrict__ v4G3_d, double Msun, double *__restrict__ rcrit_d, double *__restrict__ rcritv_d, double dt, double *__restrict__ test_d, double n1, double n2, int *EjectionFlag_d, int N){

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
#if G3 == 1
		x4G3_d[id] = x4_s[idy];
		v4G3_d[id] = v4_s[idy];


                double iai = 2.0 / sqrt(rsq) - vsq / (Msun);
                double a = 1.0 / iai;
                double T = 2.0 * M_PI * sqrt(a * a * a / Msun);

                rcritv_d[id] = T / dt; //nT number of time steps for 1 orbit
#endif

	}
}
// **************************************
// For test particles
//This kernel calculates the critical radius rcrit = max(n1 * Rh, n2 * dt * v), with the 
//Hill radius Rh = r * (m/(3Msun))^1/3, the velocity v and two constants n1 and  n2.
//rcritv is used for the the prechecker.
//In Rh we use the radius instead of the semi major axis.
//It searches also for ejections.
//This Kernel is launched wich NB/Bl blocks with Bl threads. NB is the next bigger number of N
//which is a power of two.
//
//Authors: Simon Grimm, Joachim Stadel
//March 2014
//****************************************/

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

		rcrit_d[id] = fmax(rcrit, rcrit_d[id]);
		rcritv_d[id] = fmax(rcritv, rcritv_d[id]);
//		rcrit_d[id] = rcrit;
//		rcritv_d[id] = rcritv;
	}
	if(id < N + Nsmall){
        	//Check for Ejections or to small distances to the Sun
        	if((rsq > Rcut * Rcut || rsq < RcutSun * RcutSun) && x4_s[idy].w >= 0.0){
		 	EjectionFlag_d[0] = 1;
		}
	}

}

// **************************************
//For the multi simulation mode
//This kernel calculates the critical radius rcrit = max(n1 * Rh, n2 * dt * v), with the 
//Hill radius Rh = r * (m/(3Msun))^1/3, the velocity v and two constants n1 and  n2.
//critv is used for the the prechecker.
//In Rh we use the radius instead of the semi major axis.
//It searches also for ejections.
//This Kernel is launched wich NB/Bl blocks with Bl threads. NB is the next bigger number of N
//which is a power of two.
//
//Authors: Simon Grimm, Joachim Stadel
//March 2014
//
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




__host__ void Data::firstKick_16(){
	initial_kernel < 16 > <<< NB[0] , 16 >>>(Encpairs_d, Encpairs2_d, a_d, K_d, Kold_d, BddSign_d, StopTime_d, groupIndex_d, NB[0]);
	Rcrit_kernel < 16 > <<< 1, 16 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0], rcrit_d, rcritv_d, dt, test_d, n1_h[0], n2_h[0], EjectionFlag_d, N_h[0]);
	kick16_kernel < 16, 40, 0 > <<< N_h[0] , 32 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, icNB[0], 0.0);
}
__host__ void Data::firstKick_32(){
	initial_kernel < 32 > <<< NB[0] , 32 >>>(Encpairs_d, Encpairs2_d, a_d, K_d, Kold_d, BddSign_d, StopTime_d, groupIndex_d, NB[0]);
	Rcrit_kernel < 32 > <<< 1, 32 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0], rcrit_d, rcritv_d, dt, test_d, n1_h[0], n2_h[0], EjectionFlag_d, N_h[0]);
	kick32_kernel < 32, 64, 0 > <<< N_h[0] , 64 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, icNB[0], 0.0);
}
__host__ void Data::firstKick_64(){
	initial_kernel < 64 > <<< NB[0] , 64 >>>(Encpairs_d, Encpairs2_d, a_d, K_d, Kold_d, BddSign_d, StopTime_d, groupIndex_d, NB[0]);
	Rcrit_kernel < 32 > <<< 2, 32 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0], rcrit_d, rcritv_d, dt, test_d, n1_h[0], n2_h[0], EjectionFlag_d, N_h[0]);
	kick32_kernel < 64, 64, 0 > <<< N_h[0] , 64 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, icNB[0], 0.0);
}
__host__ void Data::firstKick_128(){
	initial_kernel < 128 > <<< NB[0] , 128 >>>(Encpairs_d, Encpairs2_d, a_d, K_d, Kold_d, BddSign_d, StopTime_d, groupIndex_d, NB[0]);
	Rcrit_kernel < 32 > <<< 4, 32 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0], rcrit_d, rcritv_d, dt, test_d, n1_h[0], n2_h[0], EjectionFlag_d, N_h[0]);
	kick128_kernel < 128, 0 > <<< N2[0] , 128 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, N2[0], icNB[0], 0.0);
}
__host__ void Data::firstKick_256(){
	initial_kernel < 256 > <<< NB[0] , 256 >>>(Encpairs_d, Encpairs2_d, a_d, K_d, Kold_d, BddSign_d, StopTime_d, groupIndex_d, NB[0]);
	Rcrit_kernel < 32 > <<< 8, 32 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0], rcrit_d, rcritv_d, dt, test_d, n1_h[0], n2_h[0], EjectionFlag_d, N_h[0]);
	kick256_kernel <128, 256, 0> <<< N4[0] , 128 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], N4[0], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, N_h[0], icNB[0], 0.0);
}
__host__ void Data::firstKick_512(){
	initial_kernel < 512 > <<< NB[0] , 512 >>>(Encpairs_d, Encpairs2_d, a_d, K_d, Kold_d, BddSign_d, StopTime_d, groupIndex_d, NB[0]);
	Rcrit_kernel < 32 > <<< 16, 32 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0], rcrit_d, rcritv_d, dt, test_d, n1_h[0], n2_h[0], EjectionFlag_d, N_h[0]);
	kick4_kernel < 256, 512, 0 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], N4[0], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, N_h[0], icNB[0], 0.0);
}
__host__ void Data::firstKick_1024(){
	initial_kernel < 512 > <<< NB[0] , 512 >>>(Encpairs_d, Encpairs2_d, a_d, K_d, Kold_d, BddSign_d, StopTime_d, groupIndex_d, NB[0]);
	Rcrit_kernel < 32 > <<< 32, 32 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0], rcrit_d, rcritv_d, dt, test_d, n1_h[0], n2_h[0], EjectionFlag_d, N_h[0]);
	kick4_kernel < 256, 1024, 0 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], N4[0], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, N_h[0], icNB[0], 0.0);
}
__host__ void Data::firstKick_2048(){
	initial_kernel< 512 > <<< NB[0] , 512 >>>(Encpairs_d, Encpairs2_d, a_d, K_d, Kold_d, BddSign_d, StopTime_d, groupIndex_d, NB[0]);
	Rcrit_kernel < 64 > <<< 32, 64 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0], rcrit_d, rcritv_d, dt, test_d, n1_h[0], n2_h[0], EjectionFlag_d, N_h[0]);
	kick4_kernel < 256, 2048, 0 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], N4[0], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, N_h[0], icNB[0], 0.0);
}
__host__ void Data::firstKick_largeN(){
	initial_kernel< 512 > <<< NB[0] , 512 >>>(Encpairs_d, Encpairs2_d, a_d, K_d, Kold_d, BddSign_d, StopTime_d, groupIndex_d, NB[0]);
	Rcrit_kernel < 64 > <<< (NB[0] + 63) / 64, 64 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0], rcrit_d, rcritv_d, dt, test_d, n1_h[0], n2_h[0], EjectionFlag_d, N_h[0]);
	acc4_kernel < 256 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], N4[0], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, N_h[0], icNB[0], NB[0], 0.0);
}
__host__ void Data::firstKick_small(){
	cudaMemset(a_d, 0, NT*sizeof(double3));
	cudaMemset(asmall_d, 0, NsmallT*sizeof(double3));

	int nbInitialsmall = (max(Nsmall_h[0], NB[0] * NB[0]) + 255) / 256;
	initialsmall_kernel <<< nbInitialsmall, 256 >>> (Encpairs_d, Encpairs2_d, NB[0], Encpairssmall_d, Encpairssmall2_d, Nsmall_h[0], groupIndexsmall_d);
	Rcritsmall_kernel <128> <<< (N_h[0] + Nsmall_h[0] + 127)/128, 128 >>> (x4_d, v4_d, Msun_h[0], rcrit_d, rcritv_d, dt, test_d, n1_h[0], n2_h[0], EjectionFlag_d, x4small_d, v4small_d, Nsmall_h[0], N_h[0]);
	kicksmall_kernel < 128, 0 > <<< (N_h[0] + Nsmall_h[0] + 127)/128, 128 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], N_h[0], Nencpairs_d, Encpairs_d, Encpairs2_d, x4small_d, v4small_d, asmall_d, Nsmall_h[0], rcritvsmall_d, groupIndexsmall_d, Nencpairssmall_d, Encpairssmall_d, Encpairssmall2_d, NB[0], Nconst[0]);
}
__host__ void Data::firstKick_M(){
	initialM_kernel <16> <<< (NT + 31) / 32, 32 >>>(Encpairs_d, Encpairs2_d, a_d, NT);
	RcritM_kernel <32> <<< (NT + 31) / 32, 32>>> (x4_d, v4_d, Msun_d, rcrit_d, rcritv_d, dt, test_d, n1_d, n2_d, EjectionFlag_d, index_d, Nst, NT);
	KickM2_kernel < KM_Bl, KM_Bl2, NmaxM, 0, 16 > <<< (NT + KM_Bl2 - 1) / KM_Bl2, KM_Bl>>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, Nencpairs_d, Encpairs_d, dtksq * Kt[SIn - 1], index_d, NT, test_d);
}


//BSstep_kernel <16,16> <<< 1, 16 >>> (x4_d, v4_d, xold_d, vold_d, Encpairs2_d, Msun, dt, test_d, N, Nenc_d, U_d, rcrit_d, dtBS_d, k_d);
cudaStream_t stream[12];
__host__ void Data::BSCall(int NB, int si, double t){
	for(int st = 0; st < 6; ++st)   cudaStreamCreate(&stream[st]);
#if G3 == 0
		if(Nenc_m[1] > 0) BSBStep_kernel <2, 2> <<< Nenc_m[1], 4, 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt * FGt[si], Msun_h[0], U_d, 0, index_d, Ncoll_d, Coll_d, t, spin_d, Nst, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NB, K_d, Kold_d, groupIndex_d, groupIndexOld_d);
		if(Nenc_m[2] > 0) BSBStep_kernel <4, 4> <<< Nenc_m[2], 16, 0, stream[1] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt * FGt[si], Msun_h[0], U_d, 1, index_d, Ncoll_d, Coll_d, t, spin_d, Nst, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NB, K_d, Kold_d, groupIndex_d, groupIndexOld_d);
		if(Nenc_m[3] > 0) BSBStep_kernel <8, 8> <<< Nenc_m[3], 64, 0, stream[2] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt * FGt[si], Msun_h[0], U_d, 2, index_d, Ncoll_d, Coll_d, t, spin_d, Nst, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NB, K_d, Kold_d, groupIndex_d, groupIndexOld_d);
		if(Nenc_m[4] > 0) BSBStep_kernel <16, 16> <<< Nenc_m[4], 256, 0, stream[3] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt * FGt[si], Msun_h[0], U_d, 3, index_d, Ncoll_d, Coll_d, t, spin_d, Nst, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NB, K_d, Kold_d, groupIndex_d, groupIndexOld_d);
		if(Nenc_m[5] > 0) BSBStep_kernel <32, 8> <<< Nenc_m[5], 256, 0, stream[4] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt * FGt[si], Msun_h[0], U_d, 4, index_d, Ncoll_d, Coll_d, t, spin_d, Nst, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NB, K_d, Kold_d, groupIndex_d, groupIndexOld_d);
		if(Nenc_m[6] > 0) BSBStep64_kernel <64, 4> <<< Nenc_m[6], 256, 0, stream[5] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt * FGt[si], Msun_h[0], U_d, 5, index_d, Ncoll_d, Coll_d, t, spin_d, Nst, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NB, K_d, Kold_d, groupIndex_d, groupIndexOld_d);
#else
		if(Nenc_m[1] > 0) BSBKStep_kernel <2, 2> <<< Nenc_m[1], 4, 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt * FGt[si], Msun_h[0], U_d, 0, index_d, Ncoll_d, Coll_d, t, spin_d, Nst, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NB, K_d, Kold_d, BddSign_d, groupIndex_d, groupIndexOld_d);
		if(Nenc_m[2] > 0) BSBKStep_kernel <4, 4> <<< Nenc_m[2], 16, 0, stream[1] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt * FGt[si], Msun_h[0], U_d, 1, index_d, Ncoll_d, Coll_d, t, spin_d, Nst, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NB, K_d, Kold_d, BddSign_d, groupIndex_d, groupIndexOld_d);
		if(Nenc_m[3] > 0) BSBKStep_kernel <8, 8> <<< Nenc_m[3], 64, 0, stream[2] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt * FGt[si], Msun_h[0], U_d, 2, index_d, Ncoll_d, Coll_d, t, spin_d, Nst, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NB, K_d, Kold_d, BddSign_d, groupIndex_d, groupIndexOld_d);
//16 only possible in BSBG3.h
		if(Nenc_m[4] > 0) BSBKStep_kernel <16, 16> <<< Nenc_m[4], 256, 0, stream[3] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt * FGt[si], Msun_h[0], U_d, 2, index_d, Ncoll_d, Coll_d, t, spin_d, Nst, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NB, K_d, Kold_d, BddSign_d, groupIndex_d, groupIndexOld_d);

//for more than 16 bodies the l loop is needed again

#endif
	for(int st = 0; st < 6; ++st) cudaStreamDestroy(stream[st]);
	cudaDeviceSynchronize();
}

__host__ void Data::BSsmallCall(int si, double t){
	for(int st = 0; st < 12; ++st)   cudaStreamCreate(&stream[st]);
		if(Nencsmall_m[1] > 0) BSBStepsmall_kernel <2, 2> <<< Nencsmall_m[1], 4, 0, stream[6] >>> (x4small_d, v4small_d, xold_d, vold_d, xoldsmall_d, voldsmall_d, rcrit_d, rcritv_d, Encpairssmall_d, Encpairssmall2_d, dt * FGt[si], Msun_h[0], U_d, 0, index_d, indexsmall_d, Ncoll_d, Coll_d, t, spin_d, spinsmall_d, Nconst[0]);
		if(Nencsmall_m[2] > 0) BSBStepsmall_kernel <4, 4> <<< Nencsmall_m[2], 16, 0, stream[7] >>> (x4small_d, v4small_d, xold_d, vold_d, xoldsmall_d, voldsmall_d, rcrit_d, rcritv_d, Encpairssmall_d, Encpairssmall2_d, dt * FGt[si], Msun_h[0], U_d, 1, index_d, indexsmall_d, Ncoll_d, Coll_d, t, spin_d, spinsmall_d, Nconst[0]);
		if(Nencsmall_m[3] > 0) BSBStepsmall_kernel <8, 8> <<< Nencsmall_m[3], 64, 0, stream[8] >>> (x4small_d, v4small_d, xold_d, vold_d, xoldsmall_d, voldsmall_d, rcrit_d, rcritv_d, Encpairssmall_d, Encpairssmall2_d, dt * FGt[si], Msun_h[0], U_d, 2, index_d, indexsmall_d, Ncoll_d, Coll_d, t, spin_d, spinsmall_d, Nconst[0]);
		if(Nencsmall_m[4] > 0) BSBStepsmall_kernel <16, 16> <<< Nencsmall_m[4], 256, 0, stream[9] >>> (x4small_d, v4small_d, xold_d, vold_d, xoldsmall_d, voldsmall_d, rcrit_d, rcritv_d, Encpairssmall_d, Encpairssmall2_d, dt * FGt[si], Msun_h[0], U_d, 3, index_d, indexsmall_d, Ncoll_d, Coll_d, t, spin_d, spinsmall_d, Nconst[0]);
		if(Nencsmall_m[5] > 0) BSBStepsmall_kernel <32, 8> <<< Nencsmall_m[5], 256, 0, stream[10] >>> (x4small_d, v4small_d, xold_d, vold_d, xoldsmall_d, voldsmall_d, rcrit_d, rcritv_d, Encpairssmall_d, Encpairssmall2_d, dt * FGt[si], Msun_h[0], U_d, 4, index_d, indexsmall_d, Ncoll_d, Coll_d, t, spin_d, spinsmall_d, Nconst[0]);
		if(Nencsmall_m[6] > 0) BSBStep64small_kernel <64, 4> <<< Nencsmall_m[6], 256, 0, stream[11] >>> (x4small_d, v4small_d, xold_d, vold_d, xoldsmall_d, voldsmall_d, rcrit_d, rcritv_d, Encpairssmall_d, Encpairssmall2_d, dt * FGt[si], Msun_h[0], U_d, 5, index_d, indexsmall_d, Ncoll_d, Coll_d, t, spin_d, spinsmall_d, Nconst[0]);

		if(Nenc_m[1] > 0) BSBStep_kernel < 2, 2 > <<< Nenc_m[1], 4,   0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt * FGt[si], Msun_h[0], U_d, 0, index_d, Ncoll_d, Coll_d, t, spin_d, Nst, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NmaxTestParticles, K_d, Kold_d, groupIndex_d, groupIndexOld_d);
		if(Nenc_m[2] > 0) BSBStep_kernel < 4, 4 > <<< Nenc_m[2], 16,  0, stream[1] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt * FGt[si], Msun_h[0], U_d, 1, index_d, Ncoll_d, Coll_d, t, spin_d, Nst, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NmaxTestParticles, K_d, Kold_d, groupIndex_d, groupIndexOld_d);
		if(Nenc_m[3] > 0) BSBStep_kernel < 8, 8 > <<< Nenc_m[3], 64,  0, stream[2] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt * FGt[si], Msun_h[0], U_d, 2, index_d, Ncoll_d, Coll_d, t, spin_d, Nst, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NmaxTestParticles, K_d, Kold_d, groupIndex_d, groupIndexOld_d);
		if(Nenc_m[4] > 0) BSBStep_kernel <16, 16> <<< Nenc_m[4], 256, 0, stream[3] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt * FGt[si], Msun_h[0], U_d, 3, index_d, Ncoll_d, Coll_d, t, spin_d, Nst, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NmaxTestParticles, K_d, Kold_d, groupIndex_d, groupIndexOld_d);
		if(Nenc_m[5] > 0) BSBStep_kernel <32, 8 > <<< Nenc_m[5], 256, 0, stream[4] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt * FGt[si], Msun_h[0], U_d, 4, index_d, Ncoll_d, Coll_d, t, spin_d, Nst, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NmaxTestParticles, K_d, Kold_d, groupIndex_d, groupIndexOld_d);
		if(Nenc_m[6] > 0) BSBStep64_kernel <64, 4> <<< Nenc_m[6], 256, 0, stream[5] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt * FGt[si], Msun_h[0], U_d, 5, index_d, Ncoll_d, Coll_d, t, spin_d, Nst, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NmaxTestParticles, K_d, Kold_d, groupIndex_d, groupIndexOld_d);


	for(int st = 0; st < 12; ++st) cudaStreamDestroy(stream[st]);
	cudaDeviceSynchronize();
}

__host__ void Data::BSBMCall(int si, double t){
	for(int st = 0; st < 4; ++st)  cudaStreamCreate(&stream[st]);
		if(Nenc_m[1] > 0) BSBMStep_kernel <2, 2> <<< Nenc_m[1], 4, 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs_d, Encpairs2_d, dt * FGt[si], Msun_d, U_d, 0, index_d, Ncoll_d, Coll_d, t, spin_d, Nst, NBS_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, 16);
		if(Nenc_m[2] > 0) BSBMStep_kernel <4, 4> <<< Nenc_m[2], 16, 0, stream[1] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs_d, Encpairs2_d, dt * FGt[si], Msun_d, U_d, 1, index_d, Ncoll_d, Coll_d, t, spin_d, Nst, NBS_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, 16);
		if(Nenc_m[3] > 0) BSBMStep_kernel <8, 8> <<< Nenc_m[3], 64, 0, stream[2] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs_d, Encpairs2_d, dt * FGt[si], Msun_d, U_d, 2, index_d, Ncoll_d, Coll_d, t, spin_d, Nst, NBS_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, 16);
		if(Nenc_m[4] > 0) BSBMStep_kernel <16, 16> <<< Nenc_m[4], 256, 0, stream[3] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs_d, Encpairs2_d, dt * FGt[si], Msun_d, U_d, 3, index_d, Ncoll_d, Coll_d, t, spin_d, Nst, NBS_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, 16);
	for(int st = 0; st < 4; ++st)  cudaStreamDestroy(stream[st]);
	cudaDeviceSynchronize();
}

__host__ int Data::CollisionCall(){
	printCollisions();
	int NminFlag = remove();
	if(NminFlag == 1){
		fprintf(masterfile, "Number of bodies smaller than Nmin, simulation stopped\n");
		printf("Number of bodies smaller than Nmin, simulation stopped\n");
		return 0;
	}
	Ncoll_m[0] = 0;
	return 1;
}
__host__ void Data::CollisionMCall(){
	printCollisions();
	int NminFlag = remove();

	if(NminFlag == 1){
		stopSimulations();
		NminFlag = 0;
	}
	Ncoll_m[0] = 0;
}


__host__ int Data::EjectionCall(double t){
	Ejection(t);
	Ejectionsmall(t);
	int NminFlag = remove();
	if(NminFlag == 1){
		fprintf(masterfile, "Number of bodies smaller than Nmin, simulation stopped\n");
		printf("Number of bodies smaller than Nmin, simulation stopped\n");
		return 0;
	}
	EjectionFlag_m[0] = 0;
	EjectionFlag_m[1] = 0;
	EjectionFlag2 = 1;
	return 1;
}
__host__ void Data::EjectionMCall(double t){
	Ejection(t);
	Ejectionsmall(t);
	int NminFlag = remove();
	if(NminFlag == 1){
		stopSimulations();
		NminFlag = 0;
	}
	EjectionFlag_m[0] = 0;
	EjectionFlag2 = 1;
}

// ******************************************
// This fucntions calls the PoincareSection kernel
// It prints the section of surface: time, particle ID, x, v, to the file Poincare_X.dat
//Authors: Simon Grimm, Joachim Stadel
//March 2014
// *******************************************
#if poincareFlag == 1
__host__ int Data::PoincareSectionCall(int NB, double t){
	if(SIn > 1){
		printf("Compute Poincare Sections only with the second Order integrator!\n");
		fprintf(masterfile, "Compute Poincare Sections only with the second Order integrator!\n");
		return 0;
	}
	switch(NB){
		case 16: PoincareSection <<< 1, 16 >>> (x4_d, v4_d, xold_d, vold_d, Msun_h[0], N_h[0], 0, PFlag_d);
		break;
		case 32: PoincareSection <<< 1, 32 >>> (x4_d, v4_d, xold_d, vold_d, Msun_h[0], N_h[0], 0, PFlag_d);
		break;
		case 64: PoincareSection <<< 2, 32 >>> (x4_d, v4_d, xold_d, vold_d, Msun_h[0], N_h[0], 0, PFlag_d);
		break;
		case 128: PoincareSection <<< 4, 32 >>> (x4_d, v4_d, xold_d, vold_d, Msun_h[0], N_h[0], 0, PFlag_d);
		break;
		case 256: PoincareSection <<< 8, 32 >>> (x4_d, v4_d, xold_d, vold_d, Msun_h[0], N_h[0], 0, PFlag_d);
		break;
		case 512: PoincareSection <<< 16, 32 >>> (x4_d, v4_d, xold_d, vold_d, Msun_h[0], N_h[0], 0, PFlag_d);
		break;
		case 1024: PoincareSection <<< 32, 32 >>> (x4_d, v4_d, xold_d, vold_d, Msun_h[0], N_h[0], 0, PFlag_d);
		break;
		case 2048: PoincareSection <<< 64, 32 >>> (x4_d, v4_d, xold_d, vold_d, Msun_h[0], N_h[0], 0, PFlag_d);
		break;
	}
	if(NB > 2048){
		PoincareSection <<< (NB + 31) / 32, 32 >>> (x4_d, v4_d, xold_d, vold_d, Msun_h[0], N_h[0], 0, PFlag_d);
	}

	cudaMemcpy(PFlag_h, PFlag_d, sizeof(int), cudaMemcpyDeviceToHost);
	if(PFlag_h[0] == 1){
		cudaMemcpy(xold_h, xold_d, N_h[0] * sizeof(double4), cudaMemcpyDeviceToHost);
		cudaMemcpy(vold_h, vold_d, N_h[0] * sizeof(double4), cudaMemcpyDeviceToHost);
		cudaMemcpy(index_h, index_d, N_h[0] * sizeof(int), cudaMemcpyDeviceToHost);
		for(int i = 0; i < N_h[0]; ++i){
			if(vold_h[i].w < 0.0 && xold_h[i].w >= 0.0){
				fprintf(poincarefile, "%.16g %d %g %g\n", t/365.25, index_h[i], xold_h[i].x, vold_h[i].x);

			}
		}
		PFlag_h[0] = 0; 
		cudaMemcpy(PFlag_d, PFlag_h, sizeof(int), cudaMemcpyHostToDevice);
	}
	return 1;
}
#endif
__global__ void testA_kernel(double4 *x4_d, double4 *v4_d, int A){
        int idy = threadIdx.x;
        int id = blockIdx.x * blockDim.x + idy;

if(id == 203) printf("%d %.20g %.20g %d\n", id, x4_d[id].z, v4_d[id].z, A);

}

__host__ int Data::step_16(double t){
#if useGas > 0
com32_kernel < 16, 32 > <<<1, 16 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], 1);
GasAccCall_16(t, dt * Ct[0]);
com32_kernel < 16, 32 > <<<1, 16 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], -1);
#endif
	Rcrit_kernel <16> <<< 1, 16 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0], rcrit_d, rcritv_d, dt, test_d, n1_h[0], n2_h[0], EjectionFlag_d, N_h[0]);
#if SERIAL_GROUPING == 1
	Sort_kernel<<<1, 16 >>>(Encpairs2_d, N_h[0], icNB[0]);
#endif
#if G3 == 0
	if(Nencpairs_h[0] > 0 || EjectionFlag2 > 0) kick32A_kernel <16> <<< 1, 16 >>> (x4_d, v4_d, a_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], Encpairs2_d, test_d, N_h[0], icNB[0], t);
	else kick32B_kernel <<< 1, 16 >>> (x4_d, v4_d, a_d);
#else
	kick32B_kernel <<< 1, 16 >>> (x4_d, v4_d, a_d);
#endif
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		HC32_kernel <16, 32, 1> <<< 3, 16 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0]);
		fg_kernel < 16 > <<< 1, 16 >>> (x4_d, v4_d, xold_d, vold_d, a_d, groupIndex_d, groupIndexOld_d, dt * FGt[si], Msun_h[0], test_d, N_h[0], aelimits_d, aecount_d, Gridaecount_d, si);
		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 31)/ 32, 32 >>> (x4_d, v4_d, xold_d, vold_d, x4G3_d, v4G3_d, rcrit_d, rcritv_d, dt * FGt[si], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, enccount_d, si, K_d, Kold_d, StopTime_d, NB[0], t, groupIndexOld_d);
			group_kernel16<16, 512> <<<1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, groupIndex_d);
			cudaDeviceSynchronize();
			BSCall(16, si, t);
		}
	EjectionFlag2 = 0;
		if(Ncoll_m[0] > 0){
			int col = CollisionCall();
			if(col == 0) return 0;
		}
		if(si < SIn - 1){
			HC32_kernel <16, 32, 2> <<< 3, 16 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0]);
			kick16_kernel<16, 40, 2> <<< N_h[0], 32 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[si], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, icNB[0], t);
			//acc16_kernel<16, 40> <<< N_h[0], 32 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[si], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, icNB[0], t);
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
			//if(Nencpairs_h[0] > 0) kick32A_kernel < 32 > <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, rcritv_d, groupIndex_d, dtksq * Kt[si], Encpairs2_d, test_d, N_h[0], icNB[0], t);
			//else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d);
#if useGas > 0
com32_kernel < 16, 32 > <<<1, 16 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], 1);
GasAccCall_16(t, dt * Ct[si]);
com32_kernel < 16, 32 > <<<1, 16 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], -1);
#endif
		}
	}
	HC32_kernel <16, 32, 2> <<< 3, 16 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[SIn - 1], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0]);
	EjectionFlag2 = 0;
	kick16_kernel<16, 40, 1> <<< N_h[0], 32 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, icNB[0], t);
	//acc16_kernel<16, 40> <<< N_h[0], 32 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, icNB[0], t);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
	//if(Nencpairs_h[0] > 0) kick32A_kernel < 32 > <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], Encpairs2_d, test_d, N_h[0], icNB[0], t);
	//else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d);
#if useGas > 0
com32_kernel < 16, 32 > <<<1, 16 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], 1);
GasAccCall_16(t, dt * Ct[SIn - 1]);
com32_kernel < 16, 32 > <<<1, 16 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], -1);
#endif
	if(EjectionFlag_m[0] > 0){
		int Ej = EjectionCall(t);
		if(Ej == 0) return 0;
	}
#if poincareFlag == 1
	int per = PoincareSectionCall(16, t);
	if(per == 0) return 0;
#endif
	return 1;
}


__host__ int Data::step_32(double t){
#if useGas > 0
com32_kernel < 32, 64 > <<<1, 32 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], 1);
GasAccCall_32(t, dt * Ct[0]);
com32_kernel < 32, 64 > <<<1, 32 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], -1);
#endif
	Rcrit_kernel <32> <<< 1, 32 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0], rcrit_d, rcritv_d, dt, test_d, n1_h[0], n2_h[0], EjectionFlag_d, N_h[0]);
#if SERIAL_GROUPING == 1
	Sort_kernel<<<1, 32 >>>(Encpairs2_d, N_h[0], icNB[0]);
#endif
#if G3 == 0
	if(Nencpairs_h[0] > 0 || EjectionFlag2 > 0) kick32A_kernel <32> <<<1, 32 >>> (x4_d, v4_d, a_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], Encpairs2_d, test_d, N_h[0], icNB[0], t);
	else kick32B_kernel <<<1, 32 >>> (x4_d, v4_d, a_d);
#else
	kick32B_kernel <<<1, 32 >>> (x4_d, v4_d, a_d);
#endif
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		HC32_kernel <32, 64, 1> <<< 3, 32 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0]);
		fg_kernel <32> <<< 1, 32 >>> (x4_d, v4_d, xold_d, vold_d, a_d, groupIndex_d, groupIndexOld_d, dt * FGt[si], Msun_h[0], test_d, N_h[0], aelimits_d, aecount_d, Gridaecount_d, si);
		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 31)/ 32, 32 >>> (x4_d, v4_d, xold_d, vold_d, x4G3_d, v4G3_d, rcrit_d, rcritv_d, dt * FGt[si], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, enccount_d, si, K_d, Kold_d, StopTime_d, NB[0], t, groupIndexOld_d);
			group_kernel16<32, 512> <<<1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, groupIndex_d);
			cudaDeviceSynchronize();

			BSCall(32, si, t);
		}
		if(Ncoll_m[0] > 0){
			int col = CollisionCall();
			if(col == 0) return 0;
		}
		if(si < SIn - 1){
			HC32_kernel<32, 64, 2> <<< 3, 32 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0]);
			kick32_kernel<32, 64, 2> <<< N_h[0] , 64 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[si], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, icNB[0], t);
			//acc32_kernel<32, 64> <<< N_h[0] , 64 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[si], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, icNB[0], t);
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
			//if(Nencpairs_h[0] > 0) kick32A_kernel < 32 > <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, rcritv_d, groupIndex_d, dtksq * Kt[si], Encpairs2_d, test_d, N_h[0], icNB[0], t);
			//else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d);
#if useGas > 0
com32_kernel < 32, 64 > <<<1, 32 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], 1);
GasAccCall_32(t, dt * Ct[si]);
com32_kernel < 32, 64 > <<<1, 32 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], -1);
#endif
		}
	}
	HC32_kernel<32, 64, 2> <<< 3, 32 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[SIn - 1], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0]);
	kick32_kernel<32, 64, 1> <<< N_h[0] , 64 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, icNB[0], t);
	//acc32_kernel<32, 64> <<< N_h[0] , 64 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, icNB[0], t);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
	//if(Nencpairs_h[0] > 0) kick32A_kernel < 32 > <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], Encpairs2_d, test_d, N_h[0], icNB[0], t);
	//else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d);
#if useGas > 0
com32_kernel < 32, 64 > <<<1, 32 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], 1);
GasAccCall_32(t, dt * Ct[SIn - 1]);
com32_kernel < 32, 64 > <<<1, 32 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], -1);
#endif
	if(EjectionFlag_m[0] > 0){
		int Ej = EjectionCall(t);
		if(Ej == 0) return 0;
	}
#if poincareFlag == 1
	int per = PoincareSectionCall(32, t);
	if(per == 0) return 0;
#endif
	return 1;
}
__host__ int Data::step_64(double t){
#if useGas > 0
com32_kernel < 64, 64 > <<<1, 64 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], 1);
GasAccCall_64(t, dt * Ct[0]);
com32_kernel < 64, 64 > <<<1, 64 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], -1);
#endif
	Rcrit_kernel <32> <<< 2, 32 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0], rcrit_d, rcritv_d, dt, test_d, n1_h[0], n2_h[0], EjectionFlag_d, N_h[0]);
#if SERIAL_GROUPING == 1
	Sort_kernel<<<2, 32 >>>(Encpairs2_d, N_h[0], icNB[0]);
#endif
#if G3 == 0
	if(*Nencpairs_h > 0 || EjectionFlag2 > 0) kick32A_kernel <32> <<<2, 32 >>> (x4_d, v4_d, a_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], Encpairs2_d, test_d, N_h[0], icNB[0], t);
	else kick32B_kernel <<<2, 32 >>> (x4_d, v4_d, a_d);
#else
	kick32B_kernel <<<2, 32 >>> (x4_d, v4_d, a_d);
#endif
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		HC32_kernel<64, 64, 1> <<< 3, 64 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0]);
		fg_kernel <32> <<< 2, 32 >>>(x4_d, v4_d, xold_d, vold_d, a_d, groupIndex_d, groupIndexOld_d, dt * FGt[si], Msun_h[0], test_d, N_h[0], aelimits_d, aecount_d, Gridaecount_d, si);
		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 31)/ 32, 32 >>> (x4_d, v4_d, xold_d, vold_d, x4G3_d, v4G3_d, rcrit_d, rcritv_d, dt * FGt[si], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, enccount_d, si, K_d, Kold_d, StopTime_d, NB[0], t, groupIndexOld_d);
			group_kernel16<64, 512> <<<1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, groupIndex_d);
			cudaDeviceSynchronize();
			BSCall(64, si, t);
		}
		if(Ncoll_m[0] > 0){
			int col = CollisionCall();
			if(col == 0) return 0;
		}
		if(si < SIn - 1){
			HC32_kernel<64, 64, 2> <<< 3, 64 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0]);
			kick32_kernel<64, 64, 2 > <<< N_h[0] , 64 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[si], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, icNB[0], t);
			//acc32_kernel<64, 64> <<< N_h[0] , 64 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[si], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, icNB[0], t);
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
			//if(Nencpairs_h[0] > 0) kick32A_kernel < 32 > <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, rcritv_d, groupIndex_d, dtksq * Kt[si], Encpairs2_d, test_d, N_h[0], icNB[0], t);
			//else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d);
#if useGas > 0
com32_kernel < 64, 64 > <<<1, 64 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], 1);
GasAccCall_64(t, dt * Ct[si]);
com32_kernel < 64, 64 > <<<1, 64 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], -1);
#endif
		}
	}
	HC32_kernel<64, 64, 2> <<< 3, 64 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[SIn - 1], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0]);
	kick32_kernel<64, 64, 1 > <<< N_h[0] , 64 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, icNB[0], t);
	//acc32_kernel<64, 64> <<< N_h[0] , 64 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, icNB[0], t);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
	//if(Nencpairs_h[0] > 0) kick32A_kernel < 32 > <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], Encpairs2_d, test_d, N_h[0], icNB[0], t);
	//else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d);
#if useGas > 0
com32_kernel < 64, 64 > <<<1, 64 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], 1);
GasAccCall_64(t, dt * Ct[SIn - 1]);
com32_kernel < 64, 64 > <<<1, 64 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], -1);
#endif
	if(EjectionFlag_m[0] > 0){
		int Ej = EjectionCall(t);
		if(Ej == 0) return 0;
	}
#if poincareFlag == 1
	int per = PoincareSectionCall(64, t);
	if(per == 0) return 0;
#endif
	return 1;
}
__host__ int Data::step_128(double t){
#if useGas > 0
com128_kernel < 128 > <<<1, 128 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], 1);
GasAccCall_128(t, dt * Ct[0]);
com128_kernel < 128 > <<<1, 128 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], -1);
#endif
	Rcrit_kernel < 32 > <<< 4, 32 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0], rcrit_d, rcritv_d, dt, test_d, n1_h[0], n2_h[0], EjectionFlag_d, N_h[0]);
#if SERIAL_GROUPING == 1
	Sort_kernel<<<4, 32 >>>(Encpairs2_d, N_h[0], icNB[0]);
#endif
#if G3 == 0
	if(*Nencpairs_h > 0 || EjectionFlag2 > 0) kick32A_kernel <32> <<<4, 32 >>> (x4_d, v4_d, a_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], Encpairs2_d, test_d, N_h[0], icNB[0], t);
	else kick32B_kernel <<<4, 32 >>> (x4_d, v4_d, a_d);
#else
	kick32B_kernel <<<4, 32 >>> (x4_d, v4_d, a_d);
#endif
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		HC128_kernel < 128, 128, 1 > <<< 3, 128 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], t);
		fg_kernel <32> <<< 4, 32 >>>(x4_d, v4_d, xold_d, vold_d, a_d, groupIndex_d, groupIndexOld_d, dt * FGt[si], Msun_h[0], test_d, N_h[0], aelimits_d, aecount_d, Gridaecount_d, si);
		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 31)/ 32, 32 >>> (x4_d, v4_d, xold_d, vold_d, x4G3_d, v4G3_d, rcrit_d, rcritv_d, dt * FGt[si], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, enccount_d, si, K_d, Kold_d, StopTime_d, NB[0], t, groupIndexOld_d);
			group_kernel16<128, 512> <<<1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, groupIndex_d);
			cudaDeviceSynchronize();
			BSCall(128, si, t);
		}
		if(Ncoll_m[0] > 0){
			int col = CollisionCall();
			if(col == 0) return 0;
		}
		if(si < SIn - 1){
			HC128_kernel < 128, 128, 2 > <<< 3, 128 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], t);
//			kick128_kernel<128, 2 > <<< N2[0] , 128 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[si], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, N2[0], icNB[0], t);
			acc128_kernel<128> <<< N2[0] , 128 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[si], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, N2[0], icNB[0], t);
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
			if(Nencpairs_h[0] > 0) kick32A_kernel < 32 > <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, rcritv_d, groupIndex_d, dtksq * Kt[si], Encpairs2_d, test_d, N_h[0], icNB[0], t);
			else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d);
#if useGas > 0
com128_kernel < 128 > <<<1, 128 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], 1);
GasAccCall_128(t, dt * Ct[si]);
com128_kernel < 128 > <<<1, 128 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], -1);
#endif
		}
	}
	HC128_kernel < 128, 128, 2 > <<< 3, 128 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[SIn - 1], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], t);
//	kick128_kernel<128, 1 > <<< N2[0] , 128 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, N2[0], icNB[0], t);
	acc128_kernel<128> <<< N2[0] , 128 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, N2[0], icNB[0], t);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
	if(Nencpairs_h[0] > 0) kick32A_kernel < 32 > <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], Encpairs2_d, test_d, N_h[0], icNB[0], t);
	else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d);
#if useGas > 0
com128_kernel < 128 > <<<1, 128 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], 1);
GasAccCall_128(t, dt * Ct[SIn - 1]);
com128_kernel < 128 > <<<1, 128 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], -1);
#endif
	if(EjectionFlag_m[0] > 0){
		int Ej = EjectionCall(t);
		if(Ej == 0) return 0;
	}
#if poincareFlag == 1
	int per = PoincareSectionCall(128, t);
	if(per == 0) return 0;
#endif
	return 1;
}
__host__ int Data::step_256(double t){
#if useGas > 0
com128_kernel < 256 > <<<1, 256 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], 1);
GasAccCall_256(t, dt * Ct[0]);
com128_kernel < 256 > <<<1, 256 >>>(x4_d, v4_d, U_d, Msun_h[0],  test_d, N_h[0], -1);
#endif
	Rcrit_kernel < 32 > <<< 8, 32 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0], rcrit_d, rcritv_d, dt, test_d, n1_h[0], n2_h[0], EjectionFlag_d, N_h[0]);
#if SERIAL_GROUPING == 1
	Sort_kernel<<<8, 32 >>>(Encpairs2_d, N_h[0], icNB[0]);
#endif
#if G3 == 0
	if(Nencpairs_h[0] > 0 || EjectionFlag2 > 0) kick32A_kernel <32> <<<8, 32 >>> (x4_d, v4_d, a_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], Encpairs2_d, test_d, N_h[0], icNB[0], t);
	else kick32B_kernel <<<8, 32 >>> (x4_d, v4_d, a_d);
#else
	kick32B_kernel <<<8, 32 >>> (x4_d, v4_d, a_d);
#endif
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		HC128_kernel < 256, 256, 1 > <<< 3, 256 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], t);
		fg_kernel <32> <<< 8, 32 >>>(x4_d, v4_d, xold_d, vold_d, a_d, groupIndex_d, groupIndexOld_d, dt * FGt[si], Msun_h[0], test_d, N_h[0], aelimits_d, aecount_d, Gridaecount_d, si);
		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 31)/ 32, 32 >>> (x4_d, v4_d, xold_d, vold_d, x4G3_d, v4G3_d, rcrit_d, rcritv_d, dt * FGt[si], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, enccount_d, si, K_d, Kold_d, StopTime_d, NB[0], t, groupIndexOld_d);
			group_kernel16<256, 512> <<<1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, groupIndex_d);
			cudaDeviceSynchronize();
			BSCall(256, si, t);		
		}
		if(Ncoll_m[0] > 0){
			int col = CollisionCall();
			if(col == 0) return 0;
		}
		if(si < SIn - 1){
			HC128_kernel < 256, 256, 2 > <<< 3, 256 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], t);
//			kick256_kernel <128, 256, 2> <<< N4[0] , 128 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[si], N4[0], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, N_h[0], icNB[0], t);
			acc256_kernel <128, 256> <<< N4[0] , 128 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[si], N4[0], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, N_h[0], icNB[0], t);
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
			if(Nencpairs_h[0] > 0) kick32A_kernel < 32 > <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, rcritv_d, groupIndex_d, dtksq * Kt[si], Encpairs2_d, test_d, N_h[0], icNB[0], t);
			else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d);
#if useGas > 0
com128_kernel < 256 > <<<1, 256 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], 1);
GasAccCall_256(t, dt * Ct[si]);
com128_kernel < 256 > <<<1, 256 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], -1);
#endif
		}
	}
	HC128_kernel < 256, 256, 2 > <<< 3, 256 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[SIn - 1], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], t);
//	kick256_kernel <128, 256, 1> <<< N4[0] , 128 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], N4[0], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, N_h[0], icNB[0], t);
	acc256_kernel <128, 256> <<< N4[0] , 128 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], N4[0], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, N_h[0], icNB[0], t);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
	if(Nencpairs_h[0] > 0) kick32A_kernel < 32 > <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], Encpairs2_d, test_d, N_h[0], icNB[0], t);
	else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d);
#if useGas > 0
com128_kernel < 256 > <<<1, 256 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], 1);
GasAccCall_256(t, dt * Ct[SIn - 1]);
com128_kernel < 256 > <<<1, 256 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], -1);
#endif
	if(EjectionFlag_m[0] > 0){
		int Ej = EjectionCall(t);
		if(Ej == 0) return 0;
	}
#if poincareFlag == 1
	int per = PoincareSectionCall(256, t);
	if(per == 0) return 0;
#endif
	return 1;
}

__host__ int Data::step_512(double t){
#if useGas > 0
com128_kernel < 256 > <<<1, 256 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], 1);
GasAccCall_512(t, dt * Ct[0]);
com128_kernel < 256 > <<<1, 256 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], -1);
#endif
	Rcrit_kernel < 32 > <<< 16, 32 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0], rcrit_d, rcritv_d, dt, test_d, n1_h[0], n2_h[0], EjectionFlag_d, N_h[0]);
#if SERIAL_GROUPING == 1
	Sort_kernel<<<16, 32 >>>(Encpairs2_d, N_h[0], icNB[0]);
#endif
#if G3 == 0
	if(Nencpairs_h[0] > 0 || EjectionFlag2 > 0) kick32A_kernel <32> <<<16, 32 >>> (x4_d, v4_d, a_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], Encpairs2_d, test_d, N_h[0], icNB[0], t);
	else kick32B_kernel <<<16, 32 >>> (x4_d, v4_d, a_d);
#else
	kick32B_kernel <<<16, 32 >>> (x4_d, v4_d, a_d);
#endif
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		HC128_kernel < 512, 512, 1 > <<< 3, 512 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], t);
		fg_kernel <32> <<< 16, 32 >>>(x4_d, v4_d, xold_d, vold_d, a_d, groupIndex_d, groupIndexOld_d, dt * FGt[si], Msun_h[0], test_d, N_h[0], aelimits_d, aecount_d, Gridaecount_d, si);
		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 31)/ 32, 32 >>> (x4_d, v4_d, xold_d, vold_d, x4G3_d, v4G3_d, rcrit_d, rcritv_d, dt * FGt[si], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, enccount_d, si, K_d, Kold_d, StopTime_d, NB[0], t, groupIndexOld_d);
			group_kernel16<512, 512> <<<1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, groupIndex_d);
			cudaDeviceSynchronize();
			BSCall(512, si, t);
		}
		if(Ncoll_m[0] > 0){
			int col = CollisionCall();
			if(col == 0) return 0;
		}
		if(si < SIn - 1){
			HC128_kernel < 512, 512, 2 > <<< 3, 512 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], t);
//			kick4_kernel < 256, 512, 2 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[si], N4[0], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, N_h[0], icNB[0], t);
			acc4_kernel < 256 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[si], N4[0], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, N_h[0], icNB[0], NB[0], t);
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
			if(Nencpairs_h[0] > 0) kick32A_kernel < 32 > <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, rcritv_d, groupIndex_d, dtksq * Kt[si], Encpairs2_d, test_d, N_h[0], icNB[0], t);
			else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d);

#if useGas > 0
com128_kernel < 256 > <<<1, 256 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], 1);
GasAccCall_512(t, dt * Ct[si]);
com128_kernel < 256 > <<<1, 256 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], -1);
#endif
		}
	}

	HC128_kernel < 512, 512, 2 > <<< 3, 512 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[SIn - 1], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], t);
//	kick4_kernel < 256, 512, 1 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], N4[0], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, N_h[0], icNB[0], t);
	acc4_kernel < 256 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], N4[0], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, N_h[0], icNB[0], NB[0], t);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
	if(Nencpairs_h[0] > 0) kick32A_kernel < 32 > <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], Encpairs2_d, test_d, N_h[0], icNB[0], t);
	else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d);
#if useGas > 0
com128_kernel < 256 > <<<1, 256 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], 1);
GasAccCall_512(t, dt * Ct[SIn - 1]);
com128_kernel < 256 > <<<1, 256 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], -1);
#endif

	if(EjectionFlag_m[0] > 0){
		int Ej = EjectionCall(t);
		if(Ej == 0) return 0;
	}
#if poincareFlag == 1
	int per = PoincareSectionCall(512, t);
	if(per == 0) return 0;
#endif
	return 1;
}
__host__ int Data::step_1024(double t){
#if useGas > 0
com128_kernel < 256 > <<<1, 256 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], 1);
GasAccCall_1024(t, dt * Ct[0]);
com128_kernel < 256 > <<<1, 256 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], -1);
#endif
	Rcrit_kernel < 32 > <<< 32, 32 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0], rcrit_d, rcritv_d, dt, test_d, n1_h[0], n2_h[0], EjectionFlag_d, N_h[0]);
#if SERIAL_GROUPING == 1
	Sort_kernel<<<32, 32 >>>(Encpairs2_d, N_h[0], icNB[0]);
#endif
#if G3 == 0
	if(Nencpairs_h[0] > 0 || EjectionFlag2 > 0) kick32A_kernel <32> <<<32, 32 >>> (x4_d, v4_d, a_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], Encpairs2_d, test_d, N_h[0], icNB[0], t);
	else kick32B_kernel <<<32, 32 >>> (x4_d, v4_d, a_d);
#else
	kick32B_kernel <<<32, 32 >>> (x4_d, v4_d, a_d);
#endif
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		HC128_kernel < 512, 1024, 1 > <<< 3, 512 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], t);
		fg_kernel <32> <<< 32, 32 >>>(x4_d, v4_d, xold_d, vold_d, a_d, groupIndex_d, groupIndexOld_d, dt * FGt[si], Msun_h[0], test_d, N_h[0], aelimits_d, aecount_d, Gridaecount_d, si);
		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 31)/ 32, 32 >>> (x4_d, v4_d, xold_d, vold_d, x4G3_d, v4G3_d, rcrit_d, rcritv_d, dt * FGt[si], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, enccount_d, si, K_d, Kold_d, StopTime_d, NB[0], t, groupIndexOld_d);
			group1024_kernel <1024, 512> <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, x4_d, rcrit_d, groupIndex_d);
			cudaDeviceSynchronize();
			BSCall(1024, si, t);
		}
		if(Ncoll_m[0] > 0){
			int col = CollisionCall();
			if(col == 0) return 0;
		}
		if(si < SIn - 1){
			HC128_kernel < 512, 1024, 2 > <<< 3, 512 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], t);
//			kick4_kernel < 256, 1024, 2 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[si], N4[0], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, N_h[0], icNB[0], t);
			acc4_kernel < 256 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[si], N4[0], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, N_h[0], icNB[0], NB[0], t);
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
			if(Nencpairs_h[0] > 0) kick32A_kernel < 32 > <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, rcritv_d, groupIndex_d, dtksq * Kt[si], Encpairs2_d, test_d, N_h[0], icNB[0], t);
			else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d);
#if useGas > 0
com128_kernel < 256 > <<<1, 256 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], 1);
GasAccCall_1024(t, dt * Ct[si]);
com128_kernel < 256 > <<<1, 256 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], -1);
#endif
		}
	}
	HC128_kernel < 512, 1024, 2 > <<< 3, 512 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[SIn - 1], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], t);
//	kick4_kernel < 256, 1024, 1 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], N4[0], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, N_h[0], icNB[0], t); 
	acc4_kernel < 256 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], N4[0], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, N_h[0], icNB[0], NB[0], t);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
	if(Nencpairs_h[0] > 0) kick32A_kernel < 32 > <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], Encpairs2_d, test_d, N_h[0], icNB[0], t);
	else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d);
#if useGas > 0
com128_kernel < 256 > <<<1, 256 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], 1);
GasAccCall_1024(t, dt * Ct[SIn - 1]);
com128_kernel < 256 > <<<1, 256 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], -1);
#endif
	if(EjectionFlag_m[0] > 0){
		int Ej = EjectionCall(t);
		if(Ej == 0) return 0;
	}
#if poincareFlag == 1
	int per = PoincareSectionCall(1024, t);
	if(per == 0) return 0;
#endif
	return 1;
}

__host__ int Data::step_2048(double t){
#if useGas > 0
com128_kernel < 256 > <<<1, 256 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], 1);
GasAccCall_2048(t, dt * Ct[0]);
com128_kernel < 256 > <<<1, 256 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], -1);
#endif
	Rcrit_kernel < 32 > <<< 64, 32 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0], rcrit_d, rcritv_d, dt, test_d, n1_h[0], n2_h[0], EjectionFlag_d, N_h[0]);
#if SERIAL_GROUPING == 1
	Sort_kernel<<<64, 32 >>>(Encpairs2_d, N_h[0], icNB[0]);
#endif
#if G3 == 0
	if(Nencpairs_h[0] > 0 || EjectionFlag2 > 0) kick32A_kernel<32> <<<64, 32 >>> (x4_d, v4_d, a_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], Encpairs2_d, test_d, N_h[0], icNB[0], t);
	else kick32B_kernel <<<64, 32 >>> (x4_d, v4_d, a_d);
#else
	kick32B_kernel <<<64, 32 >>> (x4_d, v4_d, a_d);
#endif
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		HC128_kernel < 512, 2048, 1 > <<< 3, 512 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], t);
		fg_kernel <32> <<< 64, 32 >>>(x4_d, v4_d, xold_d, vold_d, a_d, groupIndex_d, groupIndexOld_d, dt * FGt[si], Msun_h[0], test_d, N_h[0], aelimits_d, aecount_d, Gridaecount_d, si);
		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 31)/ 32, 32 >>> (x4_d, v4_d, xold_d, vold_d, x4G3_d, v4G3_d, rcrit_d, rcritv_d, dt * FGt[si], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, enccount_d, si, K_d, Kold_d, StopTime_d, NB[0], t, groupIndexOld_d);
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();

			int NF = (Nencpairs2_h[0] + 511)/(512);
			if(NF > 0) group1024_kernel <2048, 512> <<< NF, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, x4_d, rcrit_d, groupIndex_d);

			if(NF > 1) fusionB_kernel <2048, 512> <<<1, 512>>>(Nenc_d, Encpairs_d, Encpairs2_d, NF, test_d, x4_d, rcrit_d, groupIndex_d);
			cudaDeviceSynchronize();
			BSCall(2048, si, t);
		}
		if(Ncoll_m[0] > 0){
			int col = CollisionCall();
			if(col == 0) return 0;
		}
		if(si < SIn - 1){
			HC128_kernel < 512, 2048, 2 > <<< 3, 512 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], t);
//			kick4_kernel < 256, 2048, 2 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[si], N4[0], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, N_h[0], icNB[0], t);
			acc4_kernel < 256 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[si], N4[0], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, N_h[0], icNB[0], NB[0], t);
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
			if(Nencpairs_h[0] > 0) kick32A_kernel < 32 > <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, rcritv_d, groupIndex_d, dtksq * Kt[si], Encpairs2_d, test_d, N_h[0], icNB[0], t);
			else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d);
#if useGas > 0
com128_kernel < 256 > <<<1, 256 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], 1);
GasAccCall_2048(t, dt * Ct[si]);
com128_kernel < 256 > <<<1, 256 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], -1);
#endif
		}
	}
	HC128_kernel < 512, 2048, 2 > <<< 3, 512 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[SIn - 1], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], t);
//	kick4_kernel < 256, 2048, 1 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], N4[0], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, N_h[0], icNB[0], t);
	acc4_kernel < 256 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], N4[0], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, N_h[0], icNB[0], NB[0], t);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
	if(Nencpairs_h[0] > 0) kick32A_kernel < 32 > <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], Encpairs2_d, test_d, N_h[0], icNB[0], t);
	else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d);
#if useGas > 0
com128_kernel < 256 > <<<1, 256 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], 1);
GasAccCall_2048(t, dt * Ct[SIn - 1]);
com128_kernel < 256 > <<<1, 256 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], -1);
#endif
	if(EjectionFlag_m[0] > 0){
		int Ej = EjectionCall(t);
		if(Ej == 0) return 0;
	}
#if poincareFlag == 1
	int per = PoincareSectionCall(2048, t);
	if(per == 0) return 0;
#endif
	return 1;
}

__host__ int Data::step_largeN(double t){
#if useGas > 0
com128_kernel < 256 > <<<1, 256 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], 1);
GasAccCall_largeN(t, dt * Ct[0]);
com128_kernel < 256 > <<<1, 256 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], -1);
#endif
	Rcrit_kernel < 64 > <<< (N_h[0] + 63) / 64, 64 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0], rcrit_d, rcritv_d, dt, test_d, n1_h[0], n2_h[0], EjectionFlag_d, N_h[0]);
#if SERIAL_GROUPING == 1
	Sort_kernel<<< (NB[0] + 31) / 32, 32 >>>(Encpairs2_d, N_h[0], icNB[0]);
#endif
#if G3 == 0
	if(Nencpairs_h[0] > 0 || EjectionFlag2 > 0) kick32A_kernel < 32 > <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], Encpairs2_d, test_d, N_h[0], icNB[0], t);
	else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d);
#else
	kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d);
#endif
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		HC128b_kernel < 512, 1 > <<< 3, 512 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], t);
		fg_kernel <32> <<< (N_h[0] + 31) / 32, 32 >>>(x4_d, v4_d, xold_d, vold_d, a_d, groupIndex_d, groupIndexOld_d, dt * FGt[si], Msun_h[0], test_d, N_h[0], aelimits_d, aecount_d, Gridaecount_d, si);
		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 31)/ 32, 32 >>> (x4_d, v4_d, xold_d, vold_d, x4G3_d, v4G3_d, rcrit_d, rcritv_d, dt * FGt[si], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, enccount_d, si, K_d, Kold_d, StopTime_d, NB[0], t, groupIndexOld_d);
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();

			int NF = (Nencpairs2_h[0] + 511)/(512);
			if(NF > 0) group1024b_kernel <512> <<< NF, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, x4_d, rcrit_d, groupIndex_d, NB[0]);

			if(NF > 1){
		//		if(NB[0] == 2048) fusionB_kernel <2048, 512> <<<1, 512>>>(Nenc_d, Encpairs_d, Encpairs2_d, NF, test_d, x4_d, rcrit_d, groupIndex_d);
		//		if(NB[0] == 4096) fusionB_kernel <4096, 512> <<<1, 512>>>(Nenc_d, Encpairs_d, Encpairs2_d, NF, test_d, x4_d, rcrit_d, groupIndex_d);
				int NF2 = (NF + 1)/2;
				cudaThreadSynchronize();
				fusionA2_kernel < 512 > <<< NF/2, 512>>>(Encpairs_d, Encpairs2_d, NB[0], NF2, test_d);
				for(int f = 0; f < log2f(NF2); ++f){
					fusionA2_kernel < 512 > <<< NF2/2 , 512>>>(Encpairs_d, Encpairs2_d, NB[0], (NF2 + 1)/2, test_d);
					NF2 = (NF2 + 1)/2;
				}
				fusion2_kernel < 512 > <<<1, 512>>>(Nenc_d, Encpairs_d, Encpairs2_d, NB[0], NF, test_d, groupIndex_d);
			}

			cudaDeviceSynchronize();
			BSCall(NB[0], si, t);
		}
		if(Ncoll_m[0] > 0){
			int col = CollisionCall();
			if(col == 0) return 0;
		}
		if(si < SIn - 1){
			HC128b_kernel < 512, 2 > <<< 3, 512 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], t);
			//kick4_kernel < 256, 2048, 2 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[si], N4[0], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, N_h[0], icNB[0], t);
			acc4_kernel < 256 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[si], N4[0], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, N_h[0], icNB[0], NB[0], t);
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
			if(Nencpairs_h[0] > 0) kick32A_kernel < 32 > <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, rcritv_d, groupIndex_d, dtksq * Kt[si], Encpairs2_d, test_d, N_h[0], icNB[0], t);
			else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d);


#if useGas > 0
com128_kernel < 256 > <<<1, 256 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], 1);
GasAccCall_largeN(t, dt * Ct[si]);
com128_kernel < 256 > <<<1, 256 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], -1);
#endif
		}
	}
	HC128b_kernel < 512, 2 > <<< 3, 512 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[SIn - 1], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], t);
//	kick4_kernel < 256, 2048, 1 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], N4[0], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, N_h[0], icNB[0], t);
	acc4_kernel < 256 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], N4[0], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, N_h[0], icNB[0], NB[0], t);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
	if(Nencpairs_h[0] > 0) kick32A_kernel < 32 > <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], Encpairs2_d, test_d, N_h[0], icNB[0], t);
	else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d);


#if useGas > 0
com128_kernel < 256 > <<<1, 256 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], 1);
GasAccCall_largeN(t, dt * Ct[SIn - 1]);
com128_kernel < 256 > <<<1, 256 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], -1);
#endif
	if(EjectionFlag_m[0] > 0){
		int Ej = EjectionCall(t);
		if(Ej == 0) return 0;
	}
#if poincareFlag == 1
	int per = PoincareSectionCall(NB[0], t);
	if(per == 0) return 0;
#endif
	return 1;
}

__host__ int Data::step_small(double t){
#if useGas > 0
com32_kernel < 16, 32 > <<<1, 16 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], 1);
GasAccCall_16(t, dt * Ct[0]);
GasAccCall_small(t, dt * Ct[0]);
com32_kernel < 16, 32 > <<<1, 16 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], -1);
#endif
	Rcritsmall_kernel <128> <<< (N_h[0] + Nsmall_h[0] + 127)/128, 128 >>> (x4_d, v4_d, Msun_h[0], rcrit_d, rcritv_d, dt, test_d, n1_h[0], n2_h[0], EjectionFlag_d, x4small_d, v4small_d, Nsmall_h[0], N_h[0]);
	if(Nencpairs_h[0] > 0 || Nencpairssmall_h[0] > 0 || EjectionFlag2 > 0) kickAsmall_kernel <128> <<< (N_h[0] + Nsmall_h[0] + 127)/128, 128 >>> (x4_d, v4_d, a_d, rcritv_d, dtksq * Kt[SIn - 1], Encpairs2_d, test_d, N_h[0], x4small_d, v4small_d, Encpairssmall2_d, Nsmall_h[0], asmall_d, NB[0], Nconst[0]);
	else kickBsmall_kernel <<< (N_h[0] + Nsmall_h[0] + 127)/128, 128 >>> (x4_d, v4_d, a_d, x4small_d, v4small_d, asmall_d, N_h[0], Nsmall_h[0]);
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		HCsmall_kernel <256, 1> <<< 3, 256 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairssmall_d, Nencpairs2_d, Nencpairssmall2_d, Nenc_d, Nencsmall_d, x4small_d, v4small_d, Nsmall_h[0], N_h[0]);
		fgsmall_kernel < 128 > <<<(N_h[0] + Nsmall_h[0] + 127)/128, 128 >>> (x4_d, v4_d, xold_d, vold_d, a_d, dt * FGt[si], Msun_h[0], test_d, x4small_d, v4small_d, xoldsmall_d, voldsmall_d, asmall_d, Nsmall_h[0], N_h[0], aelimits_d, aelimitssmall_d, aecount_d, aecountsmall_d, Gridaecount_d, si);
		if(Nencpairs_h[0] > 0 || Nencpairssmall_h[0] > 0){
			encountersmall_kernel <<< (Nencpairs_h[0] + Nencpairssmall_h[0] + 31)/ 32, 32 >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, dt * FGt[si], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, x4small_d, v4small_d, xoldsmall_d, v4small_d, Nencpairssmall_d, Encpairssmall_d, Nencpairssmall2_d, Encpairssmall2_d, N_h[0], enccount_d, enccountsmall_d, si);
			cudaMemcpy(Nencpairssmall2_h, Nencpairssmall2_d, sizeof(int), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();

			if(Nsmall_h[0] > 0) groupsmall1_kernel <<< (Nsmall_h[0] + 127)/128, 128 >>>(Nencpairssmall_d, Encpairssmall_d, Nconst[0], Nsmall_h[0]);
			if(Nencpairssmall2_h[0] > 0) groupsmall2_kernel <<< (Nencpairssmall2_h[0] + 127)/128, 128 >>> (Nencpairssmall2_d, Encpairssmall2_d, Nencsmall_d, Encpairssmall_d, Nconst[0]);
			if(NmaxTestParticles <= 512)	group_kernel16 <NmaxTestParticles, 512> <<<1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, groupIndex_d);
			else if(NmaxTestParticles <= 1024) group1024_kernel <NmaxTestParticles, 512> <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, x4_d, rcrit_d, groupIndex_d);
			else if(NmaxTestParticles <= 2048){
				cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
				cudaDeviceSynchronize();
				int NF = (Nencpairs2_h[0] + 511)/(512);
				if(NF > 0) group1024_kernel <NmaxTestParticles, 512> <<< NF, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, x4_d, rcrit_d, groupIndex_d);
				if(NF > 1) fusionB_kernel <NmaxTestParticles, 512> <<<1, 512>>>(Nenc_d, Encpairs_d, Encpairs2_d, NF, test_d, x4_d, rcrit_d, groupIndex_d);
			} 
			cudaDeviceSynchronize();
			if(Nencsmall_m[0] > 0) groupsmall3_kernel<<< (Nencsmall_m[0] + 127)/128, 128 >>> (Nencsmall_d, Encpairssmall_d, Encpairssmall2_d, Nconst[0]);
			cudaDeviceSynchronize();
			BSsmallCall(si, t);
		}
		if(Ncoll_m[0] > 0){
			int col = CollisionCall();
			if(col == 0) return 0;
		}
		if(si < SIn - 1){
			HCsmall_kernel<256, 3> <<< 3, 256 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairssmall_d, Nencpairs2_d, Nencpairssmall2_d, Nenc_d, Nencsmall_d, x4small_d, v4small_d, Nsmall_h[0], N_h[0]);
			kicksmall_kernel < 128, 2 > <<< (N_h[0] + Nsmall_h[0] + 127)/128, 128 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[si], N_h[0], Nencpairs_d, Encpairs_d, Encpairs2_d, x4small_d, v4small_d, asmall_d, Nsmall_h[0], rcritvsmall_d, groupIndexsmall_d, Nencpairssmall_d, Encpairssmall_d,Encpairssmall2_d, NB[0], Nconst[0]);
#if useGas > 0
com32_kernel < 16, 32 > <<<1, 16 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], 1);
GasAccCall_16(t, dt * Ct[si]);
GasAccCall_small(t, dt * Ct[si]);
com32_kernel < 16, 32 > <<<1, 16 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], -1);
#endif
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(Nencpairssmall_h, Nencpairssmall_d, sizeof(int), cudaMemcpyDeviceToHost);
		}
	}
	HCsmall_kernel<256, 2> <<< 3, 256 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[SIn - 1], Nencpairs_d, Nencpairssmall_d, Nencpairs2_d, Nencpairssmall2_d, Nenc_d, Nencsmall_d, x4small_d, v4small_d, Nsmall_h[0], N_h[0]);
	kicksmall_kernel < 128, 1 > <<< (N_h[0] + Nsmall_h[0] + 127)/128, 128 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq * Kt[SIn - 1], N_h[0], Nencpairs_d, Encpairs_d, Encpairs2_d, x4small_d, v4small_d, asmall_d, Nsmall_h[0], rcritvsmall_d, groupIndexsmall_d, Nencpairssmall_d, Encpairssmall_d, Encpairssmall2_d, NB[0], Nconst[0]);

#if useGas > 0
com32_kernel < 16, 32 > <<<1, 16 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], 1);
GasAccCall_16(t, dt * Ct[SIn - 1]);
GasAccCall_small(t, dt * Ct[SIn - 1]);
com32_kernel < 16, 32 > <<<1, 16 >>>(x4_d, v4_d, U_d, Msun_h[0], test_d, N_h[0], -1);
#endif
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(Nencpairssmall_h, Nencpairssmall_d, sizeof(int), cudaMemcpyDeviceToHost);
	if(EjectionFlag_m[0] > 0){
		int Ej = EjectionCall(t);
		if(Ej == 0) return 0;
	}
	return 1;
}
__host__ int Data::step_M(double t){
#if useGas > 0
comM_kernel < HCM_Bl, HCM_Bl2, NmaxM > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, Msun_d, U_d, index_d, NT, test_d, 1);
GasAccCall_M(t, dt * Ct[0]);
comM_kernel < HCM_Bl, HCM_Bl2, NmaxM > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, Msun_d, U_d, index_d, NT, test_d, -1);
#endif
	RcritM_kernel <128> <<< (NT + 127) / 128, 128>>> (x4_d, v4_d, Msun_d, rcrit_d, rcritv_d, dt, test_d, n1_d, n2_d, EjectionFlag_d, index_d, Nst, NT);
	if(Nencpairs_h[0] > 0 || EjectionFlag2 > 0) KickM2_kernel< KM_Bl, KM_Bl2, NmaxM, 3, 16 > <<< (NT + KM_Bl2 - 1) / KM_Bl2, KM_Bl>>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, Nencpairs_d, Encpairs_d, dtksq * Kt[SIn - 1], index_d, NT, test_d);
	else kickMB_kernel <<< (NT + 127) / 128, 128>>> (v4_d, a_d, test_d, NT);
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		HCM2_kernel < HCM_Bl, HCM_Bl2, NmaxM, 1 > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl>>> (x4_d, v4_d, dtiMsun_d, index_d, NT, Ct[si], test_d, Nencpairs_d, Nencpairs2_d, Nenc_d, Nst);
		fgM_kernel <128> <<< (NT + 127) / 128, 128>>> (x4_d, v4_d, xold_d, vold_d, dt, Msun_d, test_d, index_d, NT, aelimits_d, aecount_d, Gridaecount_d, FGt[si], si);
		if(Nencpairs_h[0] > 0){
			encounterM_kernel <<< (Nencpairs_h[0] + 31) / 32 , 32 >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, dt, Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, index_d, NBS_d, enccount_d, si, FGt[si], Nst);
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			if(Nencpairs2_h[0] > 0){
				groupM1_kernel <16, 256> <<< Nencpairs2_h[0], 256 >>> (Nencpairs2_d, Encpairs_d, Encpairs2_d, NBS_d, N_d, Nst);
				groupM2_kernel <<< Nencpairs2_h[0], 16 >>> (Encpairs_d, Encpairs2_d, Nenc_d, NBS_d, N_d, Nst);
				cudaDeviceSynchronize();
				BSBMCall(si, t);
			}
		}
		if(Ncoll_m[0] > 0){
			CollisionMCall();
		}
		if(si < SIn - 1){
			HCM2_kernel < HCM_Bl, HCM_Bl2, NmaxM, 2 > <<< (NT + 103) / 104, 128>>> (x4_d, v4_d, dtiMsun_d, index_d, NT, Ct[si], test_d, Nencpairs_d, Nencpairs2_d, Nenc_d, Nst);
			KickM2_kernel< KM_Bl, KM_Bl2, NmaxM, 2, 16 > <<< (NT + KM_Bl2 - 1) / KM_Bl2, KM_Bl>>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, Nencpairs_d, Encpairs_d, dtksq * Kt[si], index_d, NT, test_d);
#if useGas > 0
comM_kernel < HCM_Bl, HCM_Bl2, NmaxM > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, Msun_d, U_d, index_d, NT, test_d, 1);
GasAccCall_M(t, dt * Ct[si]);
comM_kernel < HCM_Bl, HCM_Bl2, NmaxM > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, Msun_d, U_d, index_d, NT, test_d, -1);
#endif
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
		}
	}
	HCM2_kernel < HCM_Bl, HCM_Bl2, NmaxM, 2 > <<< (NT + 103) / 104, 128>>> (x4_d, v4_d, dtiMsun_d, index_d, NT, Ct[SIn - 1], test_d, Nencpairs_d, Nencpairs2_d, Nenc_d, Nst);
	KickM2_kernel< KM_Bl, KM_Bl2, NmaxM, 1, 16 > <<< (NT + KM_Bl2 - 1) / KM_Bl2, KM_Bl>>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, Nencpairs_d, Encpairs_d, dtksq * Kt[SIn - 1], index_d, NT, test_d);
#if useGas > 0
comM_kernel < HCM_Bl, HCM_Bl2, NmaxM > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, Msun_d, U_d, index_d, NT, test_d, 1);
GasAccCall_M(t, dt * Ct[SIn - 1]);
comM_kernel < HCM_Bl, HCM_Bl2, NmaxM > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, Msun_d, U_d, index_d, NT, test_d, -1);
#endif
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
	if(EjectionFlag_m[0] > 0){
		EjectionMCall(t);
	}
	return 1;
}


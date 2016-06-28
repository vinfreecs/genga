#include "Orbit2.h"
#include "Kick3.h"
#include "HC.h"
#include "FG2.h"
#include "Encounter3.h"
#include "BSB.h"
#include "BSBM.h"
#include "BSB64M.h"
#include "ComEnergy.h"
#include "force.h"
#include "Kick4.h"
#include "BSA.h"


#if G3 > 0
	#include "BSBG3.h"
#endif

int SIn;		//Number of direction steps
int SIM;		//half of steps
double *Ct;		//time factor for HC Kick steps
double *FGt;		//time factor for Drift steps
double *Kt;		//time factor for Kick steps

int EjectionFlag2 = 0;

// *****************************************************
// This function set the time factors fot the symplectic integrator for a given order
// The first time it must be called with E = 0, afterwards with E = 1
// Authors: Simon Grimm
// June 2015
// ****************************************************
__host__ void  Data::SymplecticP(int E){
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
	if(E == 0){
		Ct = (double*)malloc(SIn*sizeof(double));
		FGt = (double*)malloc(SIn*sizeof(double));
		Kt = (double*)malloc(SIn*sizeof(double));
	}

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

// *****************************************************
// This function set the time factors for an irregular output step
// dTau is the modified time step
// Authors: Simon Grimm
// June 2015
// ****************************************************
__host__ void  Data::IrregularStep(double dTau){
	SIn = 1;
	SIM = 1;

	FGt[0] = dTau;
	Ct[0] = dTau * 0.5;
	Kt[0] = dTau * 0.5;

}

// **************************************
// This kernel sets initial values for the Encouter pair arrays
//Authors: Simon Grimm, Joachim Stadel
//March 2016
// **************************************3
template <int Bl>
__global__ void initial_kernel(int2 *Encpairs_d, int2 *Encpairs2_d, double *K_d, double *Kold_d, double4 *StopTime_d, int *groupIndex_d, int NB, int NencMax){
	int idy = threadIdx.x;
	int idx = blockIdx.x;

	for(int i = 0; i < NB; i += Bl){
#if G3 > 0
		K_d[(idy +i)* NB + idx] = 1.0;
		Kold_d[(idy +i)* NB + idx] = 1.0;
		StopTime_d[(idy +i)* NB + idx].x = -1.0;
		StopTime_d[(idy +i)* NB + idx].y = -1.0;
		StopTime_d[(idy +i)* NB + idx].z = -1.0;
		StopTime_d[(idy +i)* NB + idx].w = -1.0;
#endif
		if(idx < NencMax){
			Encpairs_d[(idy + i) * NencMax + idx].x = -1;
			Encpairs_d[(idy + i) * NencMax + idx].y = -1;

			Encpairs2_d[(idy + i) * NencMax + idx].x = -1;
			Encpairs2_d[(idy + i) * NencMax + idx].y = -1;	
		}
		if(idx == 0){
#if G3 > 0
			groupIndex_d[idy + i] = -1;
#endif
		}
	}
}

// **************************************
//This kernel sets initial values for the test particle mode
__global__ void initialsmall_kernel(int2 *Encpairs_d, int2 *Encpairs2_d, int NconstT, int NencMax){

	int idx = blockIdx.x;
	int id = blockIdx.x * blockDim.x + idx;

	if(id < NconstT * NencMax){
		Encpairs_d[id].x = -1;
		Encpairs_d[id].y = -1;

		Encpairs2_d[id].x = -1;
		Encpairs2_d[id].y = -1;
	}
}

template <int Bl>
__global__ void initialM_kernel(int2 *Encpairs_d, int2 *Encpairs2_d, int NT){
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
//
//Authors: Simon Grimm, Joachim Stadel
//March 2016
//****************************************/
__global__ void Rcrit_kernel(double4 *__restrict__ x4_d, double4 *__restrict__ v4_d, double4 * __restrict__ x4G3_d, double4 *__restrict__ v4G3_d, double Msun, double *__restrict__ rcrit_d, double *__restrict__ rcritv_d, double dt, double *__restrict__ test_d, double n1, double n2, double Rcut, double RcutSun, int *EjectionFlag_d, int N){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	double4 x4i;
	double4 v4i;

	double rcrit, rcritv ;
	double rsq, vsq, r, v;
	if(id < N){
       		x4i = x4_d[id];
		v4i = v4_d[id];
		rsq = x4i.x*x4i.x + x4i.y*x4i.y + x4i.z*x4i.z + 1.0e-30;
		vsq = v4i.x*v4i.x + v4i.y*v4i.y + v4i.z*v4i.z + 1.0e-30;

		r = sqrt(rsq);
		v = sqrt(vsq);

		rcrit = n1 * r * cbrt(x4i.w  / ( Msun * 3.0));
		rcritv = fmax(rcrit, n2 * dt * v);

		rcrit_d[id] = fmax(rcrit, rcrit_d[id]);
		rcritv_d[id] = fmax(rcritv, rcritv_d[id]);

/*	
if(x4i.w == 0){
rcrit_d[id] = 0.0;
rcritv_d[id] = 0.0;
}	
*/
		//Check for Ejections or to small distances to the Sun
		if((rsq > Rcut * Rcut || rsq < RcutSun * RcutSun) && x4_d[id].w >= 0){
			 EjectionFlag_d[0] = 1;
		}
#if G3 > 0
		x4G3_d[id] = x4i;
		v4G3_d[id] = v4i;


                double iai = 2.0 / sqrt(rsq) - vsq / (Msun);
                double a = 1.0 / iai;
                double T = 2.0 * M_PI * sqrt(a * a * a / Msun);

                rcrit_d[id] = T / dt; //nT number of time steps for 1 orbit
#endif
	}
}

// **************************************
//For the multi simulation mode
//This kernel calculates the critical radius rcrit = max(n1 * Rh, n2 * dt * v), with the 
//Hill radius Rh = r * (m/(3Msun))^1/3, the velocity v and two constants n1 and  n2.
//critv is used for the the prechecker.
//In Rh we use the radius instead of the semi major axis.
//It searches also for ejections.
//
//Authors: Simon Grimm, Joachim Stadel
//March 2016
//
// ****************************************
__global__ void RcritM_kernel(double4 *x4_d, double4 *v4_d, double4 *Msun_d, double *rcrit_d, double *rcritv_d, double *dt_d, double *test_d, double *n1_d, double *n2_d, double *Rcut_d, double *RcutSun_d, int *EjectionFlag_d, int *index_d, int Nst, int NT, double *time_d, double *idt_d, double *ict_d, long long timeStep){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;
	int st = 0;

	if(id < NT) st = index_d[id] / 100;
	if(id < Nst) time_d[id] = timeStep * idt_d[id] + ict_d[id] * 365.25;

	double4 x4i;
	double4 v4i;

	double rcrit, rcritv ;
	double rsq, vsq, r, v;

	if(id < NT){
		double Msun = Msun_d[st].x;
		double n1 = n1_d[st];
		double n2 = n2_d[st];
		double Rcut = Rcut_d[st];
		double RcutSun = RcutSun_d[st];
		double dt = dt_d[st];
		x4i = x4_d[id];
		v4i = v4_d[id];

		__syncthreads();

		rsq = x4i.x*x4i.x + x4i.y*x4i.y + x4i.z*x4i.z + 1.0e-30;
		vsq = v4i.x*v4i.x + v4i.y*v4i.y + v4i.z*v4i.z + 1.0e-30;
		r = sqrt(rsq);
		v = sqrt(vsq);

		rcrit = n1 * r * cbrt(x4i.w  / ( Msun * 3.0));
		rcritv = fmax(rcrit, n2 * dt * v);

		rcrit_d[id] = fmax(rcrit, rcrit_d[id]);
		rcritv_d[id] = fmax(rcritv, rcritv_d[id]);
		

		//Check for Ejections or to small distances to the Sun
		if((rsq > Rcut * Rcut || rsq < RcutSun * RcutSun) && x4_d[id].w >= 0){
			EjectionFlag_d[st + 1] = 1;
			EjectionFlag_d[0] = 1;
		}
	}
}

__host__ int Data::step(){
	int er;
	//Multi simulation mode
	if(MultiSim == 1){
		er = step_M();
		if(er == 0) return 0;
	}
	else{
		//Test particles
		if(P.UseTestParticles == 1){
			er = step_small();
			if(er == 0) return 0;
		}
		//check the number of massive particles
		else{
			switch(NB[0]){
				case 16: er = step_16();
				break;
				case 32: er = step_32();
				break;
				case 64: er = step_64();
				break;
				case 128: er = step_128();
				break;
				case 256: er = step_256();
				break;
				case 512: er = step_512();
				break;
				case 1024: er = step_1024();
				break;
				case 2048: er = step_2048();
				//case 2048: er = step_largeN();
				break;
			}
			if(NB[0] > 2048) er = step_largeN();
			if(er == 0) return 0;
		}
	}
	return 1;
}


__host__ void Data::firstKick_16(){
	cudaMemset(a_d, 0, NconstT * sizeof(double3));
	initial_kernel < 16 > <<< NB[0] , 16 >>>(Encpairs_d, Encpairs2_d, K_d, Kold_d, StopTime_d, groupIndex_d, NB[0], P.NencMax);
	Rcrit_kernel <<< 1, 16 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0]);
	kick16_kernel < 16, 40, 0 > <<< N_h[0] , 32 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq_h[0] * Kt[SIn - 1], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, icNB, P.NencMax, 0.0);
}
__host__ void Data::firstKick_32(){
	cudaMemset(a_d, 0, NconstT * sizeof(double3));
	initial_kernel < 32 > <<< NB[0] , 32 >>>(Encpairs_d, Encpairs2_d, K_d, Kold_d, StopTime_d, groupIndex_d, NB[0], P.NencMax);
	Rcrit_kernel <<< 1, 32 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0]);
	kick32_kernel < 32, 64, 0 > <<< N_h[0] , 64 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq_h[0] * Kt[SIn - 1], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, icNB, P.NencMax, 0.0);
}
__host__ void Data::firstKick_64(){
	cudaMemset(a_d, 0, NconstT * sizeof(double3));
	initial_kernel < 64 > <<< NB[0] , 64 >>>(Encpairs_d, Encpairs2_d, K_d, Kold_d, StopTime_d, groupIndex_d, NB[0], P.NencMax);
	Rcrit_kernel <<< 2, 32 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0]);
	kick32_kernel < 64, 64, 0 > <<< N_h[0] , 64 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq_h[0] * Kt[SIn - 1], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, icNB, P.NencMax, 0.0);
}
__host__ void Data::firstKick_128(){
	cudaMemset(a_d, 0, NconstT * sizeof(double3));
	initial_kernel < 128 > <<< NB[0] , 128 >>>(Encpairs_d, Encpairs2_d, K_d, Kold_d, StopTime_d, groupIndex_d, NB[0], P.NencMax);
	Rcrit_kernel <<< 4, 32 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0]);
	acc128b_kernel<128> <<< N2[0] , 128 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq_h[0] * Kt[SIn - 1], Encpairsb_d, Encpairs2_d, test_d, N_h[0], N2[0], icNB, P.NencMax, time_h[0]);
	EncMatrix_kernel < 128 ><<< dim3(1, N_h[0], 1), dim3(128, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, icNB, P.NencMax, N_h[0], EncFlag_d);
}
__host__ void Data::firstKick_256(){
	cudaMemset(a_d, 0, NconstT * sizeof(double3));
	initial_kernel < 256 > <<< NB[0] , 256 >>>(Encpairs_d, Encpairs2_d, K_d, Kold_d, StopTime_d, groupIndex_d, NB[0], P.NencMax);
	Rcrit_kernel <<< 8, 32 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0]);
	acc256b_kernel < 128 > <<< N4[0] , 128 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq_h[0] * Kt[SIn - 1], N4[0], Encpairsb_d, Encpairs2_d, test_d, N_h[0], icNB, P.NencMax, time_h[0]);
	EncMatrix_kernel < 256 ><<< dim3(1, N_h[0], 1), dim3(256, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, icNB, P.NencMax, N_h[0], EncFlag_d);
}
__host__ void Data::firstKick_512(){
	cudaMemset(a_d, 0, NconstT * sizeof(double3));
	initial_kernel < 512 > <<< NB[0] , 512 >>>(Encpairs_d, Encpairs2_d, K_d, Kold_d, StopTime_d, groupIndex_d, NB[0], P.NencMax);
	Rcrit_kernel <<< 16, 32 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0]);
	acc4b_kernel < 256 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq_h[0] * Kt[SIn - 1], N4[0], Encpairsb_d, Encpairs2_d, test_d, N_h[0], icNB, P.NencMax, time_h[0]);
	EncMatrix_kernel < 512 ><<< dim3(1, N_h[0], 1), dim3(512, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, icNB, P.NencMax, N_h[0], EncFlag_d);
}
__host__ void Data::firstKick_1024(){
	cudaMemset(a_d, 0, NconstT * sizeof(double3));
	initial_kernel < 512 > <<< NB[0] , 512 >>>(Encpairs_d, Encpairs2_d, K_d, Kold_d, StopTime_d, groupIndex_d, NB[0], P.NencMax);
	Rcrit_kernel <<< 32, 32 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0]);
	acc4b_kernel < 256 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq_h[0] * Kt[SIn - 1], N4[0], Encpairsb_d, Encpairs2_d, test_d, N_h[0], icNB, P.NencMax, time_h[0]);
	EncMatrix_kernel < 512 ><<< dim3(1, N_h[0], 1), dim3(512, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, icNB, P.NencMax, N_h[0], EncFlag_d);
}
__host__ void Data::firstKick_2048(){
	cudaMemset(a_d, 0, NconstT * sizeof(double3));
	initial_kernel< 512 > <<< NB[0] , 512 >>>(Encpairs_d, Encpairs2_d, K_d, Kold_d, StopTime_d, groupIndex_d, NB[0], P.NencMax);
	Rcrit_kernel <<< 32, 64 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0]);
	acc4b_kernel < 256 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq_h[0] * Kt[SIn - 1], N4[0], Encpairsb_d, Encpairs2_d, test_d, N_h[0], icNB, P.NencMax, time_h[0]);
	EncMatrix_kernel < 512 ><<< dim3(1, N_h[0], 1), dim3(512, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, icNB, P.NencMax, N_h[0], EncFlag_d);
}
__host__ void Data::firstKick_largeN(){
	cudaMemset(a_d, 0, NconstT * sizeof(double3));
	initial_kernel< 512 > <<< NB[0] , 512 >>>(Encpairs_d, Encpairs2_d, K_d, Kold_d, StopTime_d, groupIndex_d, NB[0], P.NencMax);
	Rcrit_kernel <<< (NB[0] + 63) / 64, 64 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0]);
	acc4b_kernel < 256 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq_h[0] * Kt[SIn - 1], N4[0], Encpairsb_d, Encpairs2_d, test_d, N_h[0], icNB, P.NencMax, time_h[0]);
	EncMatrix_kernel < 512 ><<< dim3(1, N_h[0], 1), dim3(512, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, icNB, P.NencMax, N_h[0], EncFlag_d);
}
__host__ void Data::firstKick_small(){
	cudaMemset(a_d, 0, NconstT*sizeof(double3));

	int nbInitialsmall = ((icNB + Nsmall_h[0]) * P.NencMax + 255) / 256;
	initialsmall_kernel <<< nbInitialsmall, 256 >>> (Encpairs_d, Encpairs2_d, NconstT, P.NencMax);
	Rcrit_kernel <<< (N_h[0] + Nsmall_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0] + Nsmall_h[0]);
	kicksmall_kernel < 128, 0 > <<< (N_h[0] + Nsmall_h[0] + 127)/128, 128 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq_h[0] * Kt[SIn - 1], N_h[0], Nencpairs_d, Encpairs_d, Encpairs2_d, Nsmall_h[0], P.NencMax);
}
__host__ void Data::firstKick_M(long long ts){
	cudaMemset(a_d, 0, NT*sizeof(double3));
	initialM_kernel < NmaxM > <<< (NT + 31) / 32, 32 >>>(Encpairs_d, Encpairs2_d, NT);
	RcritM_kernel <<< (NT + 31) / 32, 32>>> (x4_d, v4_d, Msun_d, rcrit_d, rcritv_d, dt_d, test_d, n1_d, n2_d, Rcut_d, RcutSun_d, EjectionFlag_d, index_d, Nst, NT, time_d, idt_d, ict_d, ts);
	KickM2_kernel < KM_Bl, KM_Bl2, NmaxM, 0> <<< (NT + KM_Bl2 - 1) / KM_Bl2, KM_Bl>>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, Nencpairs_d, Encpairs_d, dtksq_d, Kt[SIn - 1], index_d, NT, test_d);
}


cudaStream_t stream[12];
__host__ void Data::BSCall(int NB, int si, double t){
	for(int st = 0; st < 9; ++st)   cudaStreamCreate(&stream[st]);
#if G3 < 2

		if(Nenc_m[1] > 0) BSBStep_kernel <2, 2> <<< Nenc_m[1], 4, 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 0, index_d, Ncoll_d, Coll_d, t, spin_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NB, K_d, Kold_d, groupIndex_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d);
		if(Nenc_m[2] > 0) BSBStep_kernel <4, 4> <<< Nenc_m[2], 16, 0, stream[1] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 1, index_d, Ncoll_d, Coll_d, t, spin_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NB, K_d, Kold_d, groupIndex_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d);
		if(Nenc_m[3] > 0) BSBStep_kernel <8, 8> <<< Nenc_m[3], 64, 0, stream[2] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 2, index_d, Ncoll_d, Coll_d, t, spin_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NB, K_d, Kold_d, groupIndex_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d);
		if(Nenc_m[4] > 0) BSBStep_kernel <16, 16> <<< Nenc_m[4], 256, 0, stream[3] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 3, index_d, Ncoll_d, Coll_d, t, spin_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NB, K_d, Kold_d, groupIndex_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d);
		if(Nenc_m[5] > 0) BSBStep_kernel <32, 8> <<< Nenc_m[5], 256, 0, stream[4] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 4, index_d, Ncoll_d, Coll_d, t, spin_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NB, K_d, Kold_d, groupIndex_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d);


		if(Nenc_m[6] > 0) BSA_kernel < 64 > <<< Nenc_m[6], 64 , 0, stream[5] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 5, NB, P.NencMax, Ncoll_d, Coll_d, t, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d);
		if(Nenc_m[7] > 0) BSA_kernel < 128 > <<< Nenc_m[7], 128 , 0, stream[6] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 6, NB, P.NencMax, Ncoll_d, Coll_d, t, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d);
		if(Nenc_m[8] > 0) BSA_kernel < 256 > <<< Nenc_m[8], 256 , 0, stream[7] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 7, NB, P.NencMax, Ncoll_d, Coll_d, t, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d);


		if(Nenc_m[9] > 0) BSA512_kernel < 512, 512 > <<< Nenc_m[9], 512 , 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dx_d, dv_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 8, NB, P.NencMax, Ncoll_d, Coll_d, t, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d);
		if(Nenc_m[10] > 0) BSA512_kernel < 1024, 512 > <<< Nenc_m[10], 512 , 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dx_d, dv_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 9, NB, P.NencMax, Ncoll_d, Coll_d, t, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d);
		if(Nenc_m[11] > 0) BSA512_kernel < 2048, 512 > <<< Nenc_m[11], 512 , 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dx_d, dv_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 10, NB, P.NencMax, Ncoll_d, Coll_d, t, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d);

		int nn = 4096;
		for(int st = 11; st < def_GMax - 1; ++st){
			if(Nenc_m[st + 1] > 0) BSACall(st, nn, Nenc_m[st + 1], si, t, FGt[si]);
			nn *= 2;
		}

#else
		if(Nenc_m[1] > 0) BSBKStep_kernel <2, 2> <<< Nenc_m[1], 4, 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 0, index_d, Ncoll_d, Coll_d, t, spin_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NB, K_d, Kold_d, groupIndex_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d);
		if(Nenc_m[2] > 0) BSBKStep_kernel <4, 4> <<< Nenc_m[2], 16, 0, stream[1] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 1, index_d, Ncoll_d, Coll_d, t, spin_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NB, K_d, Kold_d, groupIndex_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d);
		if(Nenc_m[3] > 0) BSBKStep_kernel <8, 8> <<< Nenc_m[3], 64, 0, stream[2] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 2, index_d, Ncoll_d, Coll_d, t, spin_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NB, K_d, Kold_d, groupIndex_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d);
		if(Nenc_m[4] > 0) BSBKStep_kernel <16, 16> <<< Nenc_m[4], 256, 0, stream[3] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 2, index_d, Ncoll_d, Coll_d, t, spin_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NB, K_d, Kold_d, groupIndex_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d);

//for more than 16 bodies the l loop is needed again

#endif
	for(int st = 0; st < 9; ++st) cudaStreamDestroy(stream[st]);
	cudaDeviceSynchronize();
}

__host__ void Data::BSsmallCall(int si, double t){
	for(int st = 0; st < 9; ++st)   cudaStreamCreate(&stream[st]);
		if(Nenc_m[1] > 0) BSBStep_kernel <2, 2> <<< Nenc_m[1], 4, 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 0, index_d, Ncoll_d, Coll_d, t, spin_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, N_h[0] + Nsmall_h[0], K_d, Kold_d, groupIndex_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d);
		if(Nenc_m[2] > 0) BSBStep_kernel <4, 4> <<< Nenc_m[2], 16, 0, stream[1] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 1, index_d, Ncoll_d, Coll_d, t, spin_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, N_h[0] + Nsmall_h[0], K_d, Kold_d, groupIndex_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d);
		if(Nenc_m[3] > 0) BSBStep_kernel <8, 8> <<< Nenc_m[3], 64, 0, stream[2] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 2, index_d, Ncoll_d, Coll_d, t, spin_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, N_h[0] + Nsmall_h[0], K_d, Kold_d, groupIndex_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d);
		if(Nenc_m[4] > 0) BSBStep_kernel <16, 16> <<< Nenc_m[4], 256, 0, stream[3] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 3, index_d, Ncoll_d, Coll_d, t, spin_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, N_h[0] + Nsmall_h[0], K_d, Kold_d, groupIndex_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d);
		if(Nenc_m[5] > 0) BSBStep_kernel <32, 8> <<< Nenc_m[5], 256, 0, stream[4] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 4, index_d, Ncoll_d, Coll_d, t, spin_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, N_h[0] + Nsmall_h[0], K_d, Kold_d, groupIndex_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d);

		if(Nenc_m[6] > 0) BSA_kernel < 64 > <<< Nenc_m[6], 64 , 0, stream[5] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 5, N_h[0] + Nsmall_h[0], P.NencMax, Ncoll_d, Coll_d, t, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d);
		if(Nenc_m[7] > 0) BSA_kernel < 128 > <<< Nenc_m[7], 128 , 0, stream[6] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 6, N_h[0] + Nsmall_h[0], P.NencMax, Ncoll_d, Coll_d, t, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d);
		if(Nenc_m[8] > 0) BSA_kernel < 256 > <<< Nenc_m[8], 256 , 0, stream[7] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 7, N_h[0] + Nsmall_h[0], P.NencMax, Ncoll_d, Coll_d, t, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d);


		if(Nenc_m[9] > 0) BSA512_kernel < 512, 512 > <<< Nenc_m[9], 512 , 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dx_d, dv_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 8, N_h[0] + Nsmall_h[0], P.NencMax, Ncoll_d, Coll_d, t, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d);
		if(Nenc_m[10] > 0) BSA512_kernel < 1024, 512 > <<< Nenc_m[10], 512 , 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dx_d, dv_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 9, N_h[0] + Nsmall_h[0], P.NencMax, Ncoll_d, Coll_d, t, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d);
		if(Nenc_m[11] > 0) BSA512_kernel < 2048, 512 > <<< Nenc_m[11], 512 , 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dx_d, dv_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 10, N_h[0] + Nsmall_h[0], P.NencMax, Ncoll_d, Coll_d, t, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d);

		int nn = 4096;
		for(int st = 11; st < def_GMax - 1; ++st){
			if(Nenc_m[st + 1] > 0) BSAsmallCall(st, nn, Nenc_m[st + 1], si, t, FGt[si]);
			nn *= 2;
		}

	for(int st = 0; st < 9; ++st) cudaStreamDestroy(stream[st]);
	cudaDeviceSynchronize();
}

__host__ void Data::BSBMCall(int si){
	for(int st = 0; st < 6; ++st)  cudaStreamCreate(&stream[st]);
		if(Nenc_m[1] > 0) BSBMStep_kernel <2, 2> <<< Nenc_m[1], 4, 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs_d, Encpairs2_d, dt_d, FGt[si], Msun_d, U_d, 0, index_d, Ncoll_d, Coll_d, time_d, spin_d, Nst, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NmaxM, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d);
		if(Nenc_m[2] > 0) BSBMStep_kernel <4, 4> <<< Nenc_m[2], 16, 0, stream[1] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs_d, Encpairs2_d, dt_d, FGt[si], Msun_d, U_d, 1, index_d, Ncoll_d, Coll_d, time_d, spin_d, Nst, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NmaxM, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d);
		if(Nenc_m[3] > 0) BSBMStep_kernel <8, 8> <<< Nenc_m[3], 64, 0, stream[2] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs_d, Encpairs2_d, dt_d, FGt[si], Msun_d, U_d, 2, index_d, Ncoll_d, Coll_d, time_d, spin_d, Nst, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NmaxM, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d);
		if(Nenc_m[4] > 0) BSBMStep_kernel <16, 16> <<< Nenc_m[4], 256, 0, stream[3] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs_d, Encpairs2_d, dt_d, FGt[si], Msun_d, U_d, 3, index_d, Ncoll_d, Coll_d, time_d, spin_d, Nst, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NmaxM, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d);
		if(Nenc_m[5] > 0) BSBMStep_kernel <32, 8> <<< Nenc_m[5], 256, 0, stream[4] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs_d, Encpairs2_d, dt_d, FGt[si], Msun_d, U_d, 4, index_d, Ncoll_d, Coll_d, time_d, spin_d, Nst, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NmaxM, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d);
		//if(Nenc_m[6] > 0) BSBMStep64_kernel <64, 4> <<< Nenc_m[6], 256, 0, stream[5] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs_d, Encpairs2_d, dt_d, FGt[si], Msun_d, U_d, 5, index_d, Ncoll_d, Coll_d, time_d, spin_d, Nst, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, NmaxM, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d);
	for(int st = 0; st < 6; ++st)  cudaStreamDestroy(stream[st]);
	cudaDeviceSynchronize();
}


__host__ int Data::RemoveCall(){
	int NminFlag = remove();
	if(NminFlag == 1){
		fprintf(masterfile, "Number of bodies smaller than Nmin, simulation stopped\n");
		printf("Number of bodies smaller than Nmin, simulation stopped\n");
		return 0;
	}
	CollisionFlag = 0;
	return 1;
}

__host__ int Data::CollisionCall(){
	if(Ncoll_m[0] >= MaxColl - 1){
		fprintf(masterfile, "Error: More Collisions than MaxColl, simulation stopped\n");
		printf("Error: More Collisions than MaxColl, simulation stopped\n");
		return 0;
	}
	printCollisions();
	CollisionFlag = 1;
	Ncoll_m[0] = 0;
	return 1;
}
__host__ int Data::CollisionMCall(){
	if(Ncoll_m[0] >= MaxColl - 1){
		fprintf(masterfile, "Error: More Collisions than MaxColl, simulation stopped\n");
		printf("Error: More Collisions than MaxColl, simulation stopped\n");
		return 0;
	}
	printCollisions();
	CollisionFlag = 1;
	int NminFlag = remove();

	if(NminFlag == 1){
		stopSimulations();
		NminFlag = 0;
	}
	Ncoll_m[0] = 0;
	return 1;
}

__host__ int Data::writeEncCall(){
	int er = printEncounters();
	if(er == 0){
		return 0;
	}
	NWriteEnc_m[0] = 0;
	return 1;
}

__host__ int Data::EjectionCall(){
	Ejection();
	EjectionFlag_m[0] = 0;
	EjectionFlag_m[1] = 0;
	EjectionFlag2 = 1;
	int NminFlag = remove();
	if(NminFlag == 1){
		return 0;
	}
	return 1;
}
__host__ void Data::EjectionMCall(){
	Ejection();
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
		case 16: PoincareSection <<< 1, 16 >>> (x4_d, v4_d, xold_d, vold_d, index_d, Msun_h[0].x, N_h[0], 0, PFlag_d);
		break;
		case 32: PoincareSection <<< 1, 32 >>> (x4_d, v4_d, xold_d, vold_d, index_d, Msun_h[0].x, N_h[0], 0, PFlag_d);
		break;
		case 64: PoincareSection <<< 2, 32 >>> (x4_d, v4_d, xold_d, vold_d, index_d, Msun_h[0].x, N_h[0], 0, PFlag_d);
		break;
		case 128: PoincareSection <<< 4, 32 >>> (x4_d, v4_d, xold_d, vold_d, index_d, Msun_h[0].x, N_h[0], 0, PFlag_d);
		break;
		case 256: PoincareSection <<< 8, 32 >>> (x4_d, v4_d, xold_d, vold_d, index_d, Msun_h[0].x, N_h[0], 0, PFlag_d);
		break;
		case 512: PoincareSection <<< 16, 32 >>> (x4_d, v4_d, xold_d, vold_d, index_d, Msun_h[0].x, N_h[0], 0, PFlag_d);
		break;
		case 1024: PoincareSection <<< 32, 32 >>> (x4_d, v4_d, xold_d, vold_d, index_d, Msun_h[0].x, N_h[0], 0, PFlag_d);
		break;
		case 2048: PoincareSection <<< 64, 32 >>> (x4_d, v4_d, xold_d, vold_d, index_d, Msun_h[0].x, N_h[0], 0, PFlag_d);
		break;
	}
	if(NB > 2048){
		PoincareSection <<< (NB + 31) / 32, 32 >>> (x4_d, v4_d, xold_d, vold_d, Msun_h[0].x, N_h[0], 0, PFlag_d);
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
/*
__host__ int Data::step_16Simple(){
	Rcrit_kernel <<< 1, 16 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0]);
	if(Nencpairs_h[0] > 0 || EjectionFlag2 > 0) kick32A_kernel <16> <<< 1, 16 >>> (x4_d, v4_d, a_d, rcritv_d, dtksq_h[0] * Kt[SIn - 1], Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
	else kick32B_kernel <<< 1, 16 >>> (x4_d, v4_d, a_d, N_h[0]);
	EjectionFlag2 = 0;
		HC32_kernel <16, 32, 1> <<< 3, 16 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0]);
		fg_kernel <<< 1, 16 >>> (x4_d, v4_d, xold_d, vold_d, a_d, index_d, groupIndex_d, dt_h[0] * FGt[si], Msun_h[0].x, test_d, N_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, si);
		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 31)/ 32, 32 >>> (x4_d, v4_d, xold_d, vold_d, x4G3_d, v4G3_d, rcrit_d, rcritv_d, dt_h[0] * FGt[si], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, enccount_d, si, K_d, Kold_d, StopTime_d, NB[0], time_h[0], P.WriteEncounters, P.WriteEncountersRadius);
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			int NF = (Nencpairs2_h[0] + 511)/(512);
			if(NF == 1) group_kernel < 16, 512, 1 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, NB[0], N_h[0]);
			if(NF > 1) group_kernel < 16, 512, 2 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, NB[0], N_h[0]);
			cudaDeviceSynchronize();
			BSCall(16, si, time_h[0]);
		}
		EjectionFlag2 = 0;
		if(Ncoll_m[0] > 0){
			int col = CollisionCall();
			if(col == 0) return 0;
		}
		if(CollisionFlag == 1 && P.ei > 0 && timeStep % P.ei == 0){
			int rem = RemoveCall();
			if( rem == 0) return 0;
		}
		if(NWriteEnc_m[0] > 0){
			int enc = writeEncCall();
			if(enc == 0) return 0;
		}
	HC32_kernel <16, 32, 2> <<< 3, 16 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[SIn - 1], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0]);
	EjectionFlag2 = 0;
	kick16_kernel<16, 40, 1> <<< N_h[0], 32 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq_h[0] * Kt[SIn - 1], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, icNB, P.NencMax, time_h[0]);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
	if(EjectionFlag_m[0] > 0){
		int Ej = EjectionCall();
		if(Ej == 0) return 0;
	}
	return 1;
}
*/
__host__ int Data::step_16(){
//if(P.Usegas == 1 || P.UseForce > 0 || P.setElements > 0){
//	com32_kernel < 16, 32 > <<< 1, 16 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
//if(P.setElements > 0) setElements <<< 1, 16 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst);
//	if(P.Usegas == 1) GasAccCall_16(time_d, dt_d, Kt[0]);
//	if(P.UseForce > 0) force <<< 1, 16 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[0], time_d, N_h[0], Nst, P.UseForce);
//	//if(P.UseForce == 2) CallYarkovsky2 <<< 1, 16 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[0], N_h[0], Nst);
//	com32_kernel < 16, 32 > <<< 1, 16 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
//}
	Rcrit_kernel <<< 1, 16 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0]);
#if SERIAL_GROUPING == 1
	Sort_kernel<<<1, 16 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
#if G3 == 0
	if(Nencpairs_h[0] > 0 || EjectionFlag2 > 0) kick32A_kernel <16> <<< 1, 16 >>> (x4_d, v4_d, a_d, rcritv_d, dtksq_h[0] * Kt[SIn - 1], Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
	else kick32B_kernel <<< 1, 16 >>> (x4_d, v4_d, a_d, N_h[0]);
#else
	kick32B_kernel <<< 1, 16 >>> (x4_d, v4_d, a_d, N_h[0]);
#endif
if(P.Usegas == 1 || P.UseForce > 0 || P.setElements > 0){
	com32_kernel < 16, 32 > <<< 1, 16 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
if(P.setElements > 0) setElements <<< 1, 16 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst);
	if(P.Usegas == 1) GasAccCall_16(time_d, dt_d, Kt[0]);
	if(P.UseForce > 0) force <<< 1, 16 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[0], time_d, N_h[0], Nst, P.UseForce);
	//if(P.UseForce == 2) CallYarkovsky2 <<< 1, 16 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[0], N_h[0], Nst);
	com32_kernel < 16, 32 > <<< 1, 16 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
}
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		HC32_kernel <16, 32, 1> <<< 3, 16 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0]);
		fg_kernel <<< 1, 16 >>> (x4_d, v4_d, xold_d, vold_d, a_d, index_d, groupIndex_d, dt_h[0] * FGt[si], Msun_h[0].x, test_d, N_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, si);
		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 31)/ 32, 32 >>> (x4_d, v4_d, xold_d, vold_d, x4G3_d, v4G3_d, rcrit_d, rcritv_d, dt_h[0] * FGt[si], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, enccount_d, si, K_d, Kold_d, StopTime_d, NB[0], time_h[0], P.WriteEncounters, P.WriteEncountersRadius);
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			int NF = (Nencpairs2_h[0] + 511)/(512);
			if(NF == 1) group_kernel < 16, 512, 1 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, NB[0], N_h[0]);
			if(NF > 1) group_kernel < 16, 512, 2 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, NB[0], N_h[0]);
			cudaDeviceSynchronize();
			BSCall(16, si, time_h[0]);
		}
		EjectionFlag2 = 0;
		if(Ncoll_m[0] > 0){
			int col = CollisionCall();
			if(col == 0) return 0;
		}
		if(CollisionFlag == 1 && P.ei > 0 && timeStep % P.ei == 0){
			int rem = RemoveCall();
			if( rem == 0) return 0;
		}
		if(NWriteEnc_m[0] > 0){
			int enc = writeEncCall();
			if(enc == 0) return 0;
		}
		if(si < SIn - 1){
			HC32_kernel <16, 32, 2> <<< 3, 16 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0]);
			kick16_kernel<16, 40, 2> <<< N_h[0], 32 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq_h[0] * Kt[si], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, icNB, P.NencMax, time_h[0]);
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
if(P.Usegas == 1 || P.UseForce > 0){
	com32_kernel < 16, 32 > <<< 1, 16 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
	if(P.Usegas == 1) GasAccCall_16(time_d, dt_d, Kt[si]);
	if(P.UseForce > 0) force <<< 1, 16 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[si], time_d, N_h[0], Nst, P.UseForce);
	//if(P.UseForce == 2) CallYarkovsky2 <<< 1, 16 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], N_h[0], Nst);
	com32_kernel < 16, 32 > <<< 1, 16 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
}
		}
	}
	HC32_kernel <16, 32, 2> <<< 3, 16 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[SIn - 1], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0]);
	EjectionFlag2 = 0;
	kick16_kernel<16, 40, 1> <<< N_h[0], 32 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq_h[0] * Kt[SIn - 1], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, icNB, P.NencMax, time_h[0]);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
if(P.Usegas == 1 || P.UseForce > 0){
	com32_kernel < 16, 32 > <<< 1, 16 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
	if(P.Usegas == 1) GasAccCall_16(time_d, dt_d, Kt[SIn - 1]);
	if(P.UseForce > 0) force <<< 1, 16 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, N_h[0], Nst, P.UseForce);
	//if(P.UseForce == 2) CallYarkovsky2 <<< 1, 16 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst);
	com32_kernel < 16, 32 > <<< 1, 16 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
}
	if(EjectionFlag_m[0] > 0){
		int Ej = EjectionCall();
		if(Ej == 0) return 0;
	}
#if poincareFlag == 1
	int per = PoincareSectionCall(16, time_h[0]);
	if(per == 0) return 0;
#endif
	return 1;
}


__host__ int Data::step_32(){
if(P.Usegas == 1 || P.UseForce > 0 || P.setElements > 0){
	com32_kernel < 32, 64 > <<< 1, 32 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
if(P.setElements > 0) setElements <<< 1, 32 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst);
	if(P.Usegas == 1) GasAccCall_32(time_d, dt_d, Kt[0]);
	if(P.UseForce > 0) force <<< 1, 32 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[0], time_d, N_h[0], Nst, P.UseForce);
	com32_kernel < 32, 64 > <<< 1, 32 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
}
	Rcrit_kernel <<< 1, 32 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0]);
#if SERIAL_GROUPING == 1
	Sort_kernel<<<1, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
#if G3 == 0
	if(Nencpairs_h[0] > 0 || EjectionFlag2 > 0) kick32A_kernel <32> <<<1, 32 >>> (x4_d, v4_d, a_d, rcritv_d, dtksq_h[0] * Kt[SIn - 1], Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
	else kick32B_kernel <<<1, 32 >>> (x4_d, v4_d, a_d, N_h[0]);
#else
	kick32B_kernel <<<1, 32 >>> (x4_d, v4_d, a_d, N_h[0]);
#endif
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		HC32_kernel <32, 64, 1> <<< 3, 32 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0]);
		fg_kernel <<< 1, 32 >>> (x4_d, v4_d, xold_d, vold_d, a_d, index_d, groupIndex_d, dt_h[0] * FGt[si], Msun_h[0].x, test_d, N_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, si);
		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 31)/ 32, 32 >>> (x4_d, v4_d, xold_d, vold_d, x4G3_d, v4G3_d, rcrit_d, rcritv_d, dt_h[0] * FGt[si], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, enccount_d, si, K_d, Kold_d, StopTime_d, NB[0], time_h[0], P.WriteEncounters, P.WriteEncountersRadius);
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			int NF = (Nencpairs2_h[0] + 511)/(512);
			if(NF == 1) group_kernel < 32, 512, 1 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, NB[0], N_h[0]);
			if(NF > 1) group_kernel < 32, 512, 2 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, NB[0], N_h[0]);
			cudaDeviceSynchronize();

			BSCall(32, si, time_h[0]);
		}
		if(Ncoll_m[0] > 0){
			int col = CollisionCall();
			if(col == 0) return 0;
		}
		if(CollisionFlag == 1 && P.ei > 0 && timeStep % P.ei == 0){
			int rem = RemoveCall();
			if( rem == 0) return 0;
		}
		if(NWriteEnc_m[0] > 0){
			int enc = writeEncCall();
			if(enc == 0) return 0;
		}
		if(si < SIn - 1){
			HC32_kernel<32, 64, 2> <<< 3, 32 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0]);
			kick32_kernel<32, 64, 2> <<< N_h[0] , 64 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq_h[0] * Kt[si], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, icNB, P.NencMax, time_h[0]);
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
if(P.Usegas == 1 || P.UseForce > 0){
	com32_kernel < 32, 64 > <<< 1, 32 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
	if(P.Usegas == 1) GasAccCall_32(time_d, dt_d, Kt[si]);
	if(P.UseForce > 0) force <<< 1, 32 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[si], time_d, N_h[0], Nst, P.UseForce);
	com32_kernel < 32, 64 > <<< 1, 32 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
}
		}
	}
	HC32_kernel<32, 64, 2> <<< 3, 32 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[SIn - 1], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0]);
	kick32_kernel<32, 64, 1> <<< N_h[0] , 64 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq_h[0] * Kt[SIn - 1], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, icNB, P.NencMax, time_h[0]);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
if(P.Usegas == 1 || P.UseForce > 0){
	com32_kernel < 32, 64 > <<< 1, 32 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
	if(P.Usegas == 1) GasAccCall_32(time_d, dt_d, Kt[SIn - 1]);
	if(P.UseForce > 0) force <<< 1, 32 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, N_h[0], Nst, P.UseForce);
	com32_kernel < 32, 64 > <<< 1, 32 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
}
	if(EjectionFlag_m[0] > 0){
		int Ej = EjectionCall();
		if(Ej == 0) return 0;
	}
#if poincareFlag == 1
	int per = PoincareSectionCall(32, time_h[0]);
	if(per == 0) return 0;
#endif
	return 1;
}
__host__ int Data::step_64(){
if(P.Usegas == 1 || P.UseForce > 0 || P.setElements > 0){
	com32_kernel < 64, 64 > <<< 1, 64 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
if(P.setElements > 0) setElements <<< 1, 64 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst);
	if(P.Usegas == 1) GasAccCall_64(time_d, dt_d, Kt[0]);
	if(P.UseForce > 0) force <<< 1, 64 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[0], time_d, N_h[0], Nst, P.UseForce);
	com32_kernel < 64, 64 > <<< 1, 64 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
}
	Rcrit_kernel <<< 2, 32 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0]);
#if SERIAL_GROUPING == 1
	Sort_kernel<<<2, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
#if G3 == 0
	if(*Nencpairs_h > 0 || EjectionFlag2 > 0) kick32A_kernel <32> <<<2, 32 >>> (x4_d, v4_d, a_d, rcritv_d, dtksq_h[0] * Kt[SIn - 1], Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
	else kick32B_kernel <<<2, 32 >>> (x4_d, v4_d, a_d, N_h[0]);
#else
	kick32B_kernel <<<2, 32 >>> (x4_d, v4_d, a_d, N_h[0]);
#endif
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		HC32_kernel<64, 64, 1> <<< 3, 64 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0]);
		fg_kernel <<< 2, 32 >>>(x4_d, v4_d, xold_d, vold_d, a_d, index_d, groupIndex_d, dt_h[0] * FGt[si], Msun_h[0].x, test_d, N_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, si);
		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 31)/ 32, 32 >>> (x4_d, v4_d, xold_d, vold_d, x4G3_d, v4G3_d, rcrit_d, rcritv_d, dt_h[0] * FGt[si], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, enccount_d, si, K_d, Kold_d, StopTime_d, NB[0], time_h[0], P.WriteEncounters, P.WriteEncountersRadius);
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			int NF = (Nencpairs2_h[0] + 511)/(512);
			if(NF == 1) group_kernel < 64, 512, 1 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, NB[0], N_h[0]);
			if(NF > 1) group_kernel < 64, 512, 2 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, NB[0], N_h[0]);
			cudaDeviceSynchronize();
			BSCall(64, si, time_h[0]);
		}
		if(Ncoll_m[0] > 0){
			int col = CollisionCall();
			if(col == 0) return 0;
		}
		if(CollisionFlag == 1 && P.ei > 0 && timeStep % P.ei == 0){
			int rem = RemoveCall();
			if( rem == 0) return 0;
		}
		if(NWriteEnc_m[0] > 0){
			int enc = writeEncCall();
			if(enc == 0) return 0;
		}
		if(si < SIn - 1){
			HC32_kernel<64, 64, 2> <<< 3, 64 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0]);
			kick32_kernel<64, 64, 2 > <<< N_h[0] , 64 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq_h[0] * Kt[si], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, icNB, P.NencMax, time_h[0]);
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
if(P.Usegas == 1 || P.UseForce > 0){
	com32_kernel < 64, 64 > <<< 1, 64 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
	if(P.Usegas == 1) GasAccCall_64(time_d, dt_d, Kt[si]);
	if(P.UseForce > 0) force <<< 1, 64 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[si], time_d, N_h[0], Nst, P.UseForce);
	com32_kernel < 64, 64 > <<< 1, 64 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
}
		}
	}
	HC32_kernel<64, 64, 2> <<< 3, 64 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[SIn - 1], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0]);
	kick32_kernel<64, 64, 1 > <<< N_h[0] , 64 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq_h[0] * Kt[SIn - 1], Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, icNB, P.NencMax, time_h[0]);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
if(P.Usegas == 1 || P.UseForce > 0){
	com32_kernel < 64, 64 > <<< 1, 64 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
	if(P.Usegas == 1) GasAccCall_64(time_d, dt_d, Kt[SIn - 1]);
	if(P.UseForce > 0) force <<< 1, 64 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, N_h[0], Nst, P.UseForce);
	com32_kernel < 64, 64 > <<< 1, 64 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
}
	if(EjectionFlag_m[0] > 0){
		int Ej = EjectionCall();
		if(Ej == 0) return 0;
	}
#if poincareFlag == 1
	int per = PoincareSectionCall(64, time_h[0]);
	if(per == 0) return 0;
#endif
	return 1;
}
__host__ int Data::step_128(){
if(P.Usegas == 1 || P.UseForce > 0 || P.setElements > 0){
	com128_kernel < 128 > <<< 1, 128 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
if(P.setElements > 0) setElements <<< 1, 128 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst);
	if(P.UseForce > 0) force <<< 1, 128 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[0], time_d, N_h[0], Nst, P.UseForce);
	if(P.Usegas == 1) GasAccCall_128(time_d, dt_d, Kt[0]);
	com128_kernel < 128 > <<< 1, 128 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
}
	Rcrit_kernel <<< 4, 32 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0]);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<<4, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
#if G3 == 0
	if(*Nencpairs_h > 0 || EjectionFlag2 > 0) kick32Ab_kernel <32> <<<4, 32 >>> (x4_d, v4_d, a_d, rcritv_d, dtksq_h[0] * Kt[SIn - 1], Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
	else kick32B_kernel <<<4, 32 >>> (x4_d, v4_d, a_d, N_h[0]);
#else
	kick32B_kernel <<<4, 32 >>> (x4_d, v4_d, a_d, N_h[0]);
#endif
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		HC128b_kernel < 128, 1 > <<< 3, 128 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], time_h[0]);
		fg_kernel <<< 4, 32 >>>(x4_d, v4_d, xold_d, vold_d, a_d, index_d, groupIndex_d, dt_h[0] * FGt[si], Msun_h[0].x, test_d, N_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, si);
		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 31)/ 32, 32 >>> (x4_d, v4_d, xold_d, vold_d, x4G3_d, v4G3_d, rcrit_d, rcritv_d, dt_h[0] * FGt[si], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, enccount_d, si, K_d, Kold_d, StopTime_d, NB[0], time_h[0], P.WriteEncounters, P.WriteEncountersRadius);
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			int NF = (Nencpairs2_h[0] + 511)/(512);
			if(NF == 1) group_kernel < 128, 512, 1 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, NB[0], N_h[0]);
			if(NF > 1) group_kernel < 128, 512, 2 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, NB[0], N_h[0]);
			cudaDeviceSynchronize();
			BSCall(128, si, time_h[0]);
		}
		if(Ncoll_m[0] > 0){
			int col = CollisionCall();
			if(col == 0) return 0;
		}
		if(CollisionFlag == 1 && P.ei > 0 && timeStep % P.ei == 0){
			int rem = RemoveCall();
			if( rem == 0) return 0;
		}
		if(NWriteEnc_m[0] > 0){
			int enc = writeEncCall();
			if(enc == 0) return 0;
		}
		if(si < SIn - 1){
			HC128b_kernel < 128, 2 > <<< 3, 128 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], time_h[0]);
			acc128b_kernel<128> <<< N2[0] , 128 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq_h[0] * Kt[si], Encpairsb_d, Encpairs2_d, test_d, N_h[0], N2[0], icNB, P.NencMax, time_h[0]);
			EncMatrix_kernel < 128 ><<< dim3(1, N_h[0], 1), dim3(128, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, icNB, P.NencMax, N_h[0], EncFlag_d);
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<<4, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
			if(Nencpairs_h[0] > 0) kick32Ab_kernel < 32 > <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, rcritv_d, dtksq_h[0] * Kt[si], Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
			else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, N_h[0]);
if(P.Usegas == 1 || P.UseForce > 0){
	com128_kernel < 128 > <<< 1, 128 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
	if(P.Usegas == 1) GasAccCall_128(time_d, dt_d, Kt[si]);
	if(P.UseForce > 0) force <<< 1, 128 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[si], time_d, N_h[0], Nst, P.UseForce);
	com128_kernel < 128 > <<< 1, 128 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
}
		}
	}
	HC128b_kernel < 128, 2 > <<< 3, 128 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[SIn - 1], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], time_h[0]);
	acc128b_kernel<128> <<< N2[0] , 128 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq_h[0] * Kt[SIn - 1], Encpairsb_d, Encpairs2_d, test_d, N_h[0], N2[0], icNB, P.NencMax, time_h[0]);
	EncMatrix_kernel < 128 ><<< dim3(1, N_h[0], 1), dim3(128, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, icNB, P.NencMax, N_h[0], EncFlag_d);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<<4, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
	if(Nencpairs_h[0] > 0) kick32Ab_kernel < 32 > <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, rcritv_d, dtksq_h[0] * Kt[SIn - 1], Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
	else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, N_h[0]);
if(P.Usegas == 1 || P.UseForce > 0){
	com128_kernel < 128 > <<< 1, 128 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
	if(P.Usegas == 1) GasAccCall_128(time_d, dt_d, Kt[SIn - 1]);
	if(P.UseForce > 0) force <<< 1, 128 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, N_h[0], Nst, P.UseForce);
	com128_kernel < 128 > <<< 1, 128 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
}
	if(EjectionFlag_m[0] > 0){
		int Ej = EjectionCall();
		if(Ej == 0) return 0;
	}
#if poincareFlag == 1
	int per = PoincareSectionCall(128, time_h[0]);
	if(per == 0) return 0;
#endif
	return 1;
}
__host__ int Data::step_256(){
if(P.Usegas == 1 || P.UseForce > 0 || P.setElements > 0){
	com128_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
if(P.setElements > 0) setElements <<< 1, 256 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst);
	if(P.Usegas == 1) GasAccCall_256(time_d, dt_d, Kt[0]);
	if(P.UseForce > 0) force <<< 1, 256 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[0], time_d, N_h[0], Nst, P.UseForce);
	com128_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x,  test_d, N_h[0], -1);
}
	Rcrit_kernel <<< 8, 32 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0]);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<<8, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
#if G3 == 0
	if(Nencpairs_h[0] > 0 || EjectionFlag2 > 0) kick32Ab_kernel <32> <<<8, 32 >>> (x4_d, v4_d, a_d, rcritv_d, dtksq_h[0] * Kt[SIn - 1], Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
	else kick32B_kernel <<<8, 32 >>> (x4_d, v4_d, a_d, N_h[0]);
#else
	kick32B_kernel <<<8, 32 >>> (x4_d, v4_d, a_d, N_h[0]);
#endif
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		HC128b_kernel < 256, 1 > <<< 3, 256 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], time_h[0]);
		fg_kernel <<< 8, 32 >>>(x4_d, v4_d, xold_d, vold_d, a_d, index_d, groupIndex_d, dt_h[0] * FGt[si], Msun_h[0].x, test_d, N_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, si);
		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 31)/ 32, 32 >>> (x4_d, v4_d, xold_d, vold_d, x4G3_d, v4G3_d, rcrit_d, rcritv_d, dt_h[0] * FGt[si], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, enccount_d, si, K_d, Kold_d, StopTime_d, NB[0], time_h[0], P.WriteEncounters, P.WriteEncountersRadius);
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			int NF = (Nencpairs2_h[0] + 511)/(512);
			if(NF == 1) group_kernel < 256, 512, 1 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, NB[0], N_h[0]);
			if(NF > 1) group_kernel < 256, 512, 2 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, NB[0], N_h[0]);

			cudaDeviceSynchronize();
			BSCall(256, si, time_h[0]);		
		}
		if(Ncoll_m[0] > 0){
			int col = CollisionCall();
			if(col == 0) return 0;
		}
		if(CollisionFlag == 1 && P.ei > 0 && timeStep % P.ei == 0){
			int rem = RemoveCall();
			if( rem == 0) return 0;
		}
		if(NWriteEnc_m[0] > 0){
			int enc = writeEncCall();
			if(enc == 0) return 0;
		}
		if(si < SIn - 1){
			HC128b_kernel < 256, 2 > <<< 3, 256 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], time_h[0]);
			acc256b_kernel < 128 > <<< N4[0] , 128 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq_h[0] * Kt[si], N4[0], Encpairsb_d, Encpairs2_d, test_d, N_h[0], icNB, P.NencMax, time_h[0]);
			EncMatrix_kernel < 256 ><<< dim3(1, N_h[0], 1), dim3(256, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, icNB, P.NencMax, N_h[0], EncFlag_d);
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<<8, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
			if(Nencpairs_h[0] > 0) kick32Ab_kernel < 32 > <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, rcritv_d, dtksq_h[0] * Kt[si], Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
			else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, N_h[0]);
if(P.Usegas == 1 || P.UseForce > 0){
	com128_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
	if(P.Usegas == 1) GasAccCall_256(time_d, dt_d, Kt[si]);
	if(P.UseForce > 0) force <<< 1, 256 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[si], time_d, N_h[0], Nst, P.UseForce);
	com128_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
}
		}
	}
	HC128b_kernel < 256, 2 > <<< 3, 256 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[SIn - 1], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], time_h[0]);
	acc256b_kernel < 128 > <<< N4[0] , 128 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq_h[0] * Kt[SIn - 1], N4[0], Encpairsb_d, Encpairs2_d, test_d, N_h[0], icNB, P.NencMax, time_h[0]);
	EncMatrix_kernel < 256 ><<< dim3(1, N_h[0], 1), dim3(256, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, icNB, P.NencMax, N_h[0], EncFlag_d);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<<8, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
	if(Nencpairs_h[0] > 0) kick32Ab_kernel < 32 > <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, rcritv_d, dtksq_h[0] * Kt[SIn - 1], Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
	else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, N_h[0]);
if(P.Usegas == 1 || P.UseForce > 0){
	com128_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
	if(P.Usegas == 1) GasAccCall_256(time_d, dt_d, Kt[SIn - 1]);
	if(P.UseForce > 0) force <<< 1, 256 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, N_h[0], Nst, P.UseForce);
	com128_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
}
	if(EjectionFlag_m[0] > 0){
		int Ej = EjectionCall();
		if(Ej == 0) return 0;
	}
#if poincareFlag == 1
	int per = PoincareSectionCall(256, time_h[0]);
	if(per == 0) return 0;
#endif
	return 1;
}

__host__ int Data::step_512(){
if(P.Usegas == 1 || P.UseForce > 0 || P.setElements > 0){
	com128_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
if(P.setElements > 0) setElements <<< 2, 256 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst);
	if(P.Usegas == 1) GasAccCall_512(time_d, dt_d, Kt[0]);
	if(P.UseForce > 0) force <<< 2, 256 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[0], time_d, N_h[0], Nst, P.UseForce);
	com128_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
}
	Rcrit_kernel <<< 16, 32 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0]);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<<16, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
#if G3 == 0
	if(Nencpairs_h[0] > 0 || EjectionFlag2 > 0) kick32Ab_kernel <32> <<<16, 32 >>> (x4_d, v4_d, a_d, rcritv_d, dtksq_h[0] * Kt[SIn - 1], Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
	else kick32B_kernel <<<16, 32 >>> (x4_d, v4_d, a_d, N_h[0]);
#else
	kick32B_kernel <<<16, 32 >>> (x4_d, v4_d, a_d, N_h[0]);
#endif
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		HC128b_kernel < 512, 1 > <<< 3, 512 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], time_h[0]);
		fg_kernel <<< 16, 32 >>>(x4_d, v4_d, xold_d, vold_d, a_d, index_d, groupIndex_d, dt_h[0] * FGt[si], Msun_h[0].x, test_d, N_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, si);
		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 31)/ 32, 32 >>> (x4_d, v4_d, xold_d, vold_d, x4G3_d, v4G3_d, rcrit_d, rcritv_d, dt_h[0] * FGt[si], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, enccount_d, si, K_d, Kold_d, StopTime_d, NB[0], time_h[0], P.WriteEncounters, P.WriteEncountersRadius);
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			int NF = (Nencpairs2_h[0] + 511)/(512);
			if(NF == 1) group_kernel < 512, 512, 1 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, NB[0], N_h[0]);
			if(NF > 1) group_kernel < 512, 512, 2 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, NB[0], N_h[0]);
			cudaDeviceSynchronize();
			BSCall(512, si, time_h[0]);
		}
		if(Ncoll_m[0] > 0){
			int col = CollisionCall();
			if(col == 0) return 0;
		}
		if(CollisionFlag == 1 && P.ei > 0 && timeStep % P.ei == 0){
			int rem = RemoveCall();
			if( rem == 0) return 0;
		}
		if(NWriteEnc_m[0] > 0){
			int enc = writeEncCall();
			if(enc == 0) return 0;
		}
		if(si < SIn - 1){
			HC128b_kernel < 512, 2 > <<< 3, 512 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], time_h[0]);
			acc4b_kernel < 256 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq_h[0] * Kt[si], N4[0], Encpairsb_d, Encpairs2_d, test_d, N_h[0], icNB, P.NencMax, time_h[0]);
			EncMatrix_kernel < 512 ><<< dim3(1, N_h[0], 1), dim3(512, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, icNB, P.NencMax, N_h[0], EncFlag_d);
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<<16, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
			if(Nencpairs_h[0] > 0) kick32Ab_kernel < 32 > <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, rcritv_d, dtksq_h[0] * Kt[si], Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
			else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, N_h[0]);

if(P.Usegas == 1 || P.UseForce > 0){
	com128_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
	if(P.Usegas == 1) GasAccCall_512(time_d, dt_d, Kt[si]);
	if(P.UseForce > 0) force <<< 2, 256 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[si], time_d, N_h[0], Nst, P.UseForce);
	com128_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
}
		}
	}

	HC128b_kernel < 512, 2 > <<< 3, 512 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[SIn - 1], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], time_h[0]);
	acc4b_kernel < 256 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq_h[0] * Kt[SIn - 1], N4[0], Encpairsb_d, Encpairs2_d, test_d, N_h[0], icNB, P.NencMax, time_h[0]);
	EncMatrix_kernel < 512 ><<< dim3(1, N_h[0], 1), dim3(512, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, icNB, P.NencMax, N_h[0], EncFlag_d);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<<16, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
	if(Nencpairs_h[0] > 0) kick32Ab_kernel < 32 > <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, rcritv_d, dtksq_h[0] * Kt[SIn - 1], Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
	else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, N_h[0]);
if(P.Usegas == 1 || P.UseForce > 0){
	com128_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
	if(P.Usegas == 1) GasAccCall_512(time_d, dt_d, Kt[SIn - 1]);
	if(P.UseForce > 0) force <<< 2, 256 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, N_h[0], Nst, P.UseForce);
	com128_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
}

	if(EjectionFlag_m[0] > 0){
		int Ej = EjectionCall();
		if(Ej == 0) return 0;
	}
#if poincareFlag == 1
	int per = PoincareSectionCall(512, time_h[0]);
	if(per == 0) return 0;
#endif
	return 1;
}
__host__ int Data::step_1024(){
if(P.Usegas == 1 || P.UseForce > 0 || P.setElements > 0){
	com128_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
if(P.setElements > 0) setElements <<< 4, 256 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst);
	if(P.Usegas == 1) GasAccCall_1024(time_d, dt_d, Kt[0]);
	if(P.UseForce > 0) force <<< 4, 256 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[0], time_d, N_h[0], Nst, P.UseForce);
	com128_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
}
	Rcrit_kernel <<< 32, 32 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0]);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<<32, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
#if G3 == 0
	if(Nencpairs_h[0] > 0 || EjectionFlag2 > 0) kick32Ab_kernel <32> <<<32, 32 >>> (x4_d, v4_d, a_d, rcritv_d, dtksq_h[0] * Kt[SIn - 1], Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
	else kick32B_kernel <<<32, 32 >>> (x4_d, v4_d, a_d, N_h[0]);
#else
	kick32B_kernel <<<32, 32 >>> (x4_d, v4_d, a_d, N_h[0]);
#endif
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		HC128b_kernel < 512, 1 > <<< 3, 512 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], time_h[0]);
		fg_kernel <<< 32, 32 >>>(x4_d, v4_d, xold_d, vold_d, a_d, index_d, groupIndex_d, dt_h[0] * FGt[si], Msun_h[0].x, test_d, N_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, si);
		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 31)/ 32, 32 >>> (x4_d, v4_d, xold_d, vold_d, x4G3_d, v4G3_d, rcrit_d, rcritv_d, dt_h[0] * FGt[si], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, enccount_d, si, K_d, Kold_d, StopTime_d, NB[0], time_h[0], P.WriteEncounters, P.WriteEncountersRadius);
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			int NF = (Nencpairs2_h[0] + 511)/(512);
			if(NF == 1) group_kernel < 1, 512, 3> <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, NB[0], N_h[0]);
			if(NF > 1) group_kernel < 1, 512, 4> <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, NB[0], N_h[0]);
			cudaDeviceSynchronize();
			BSCall(1024, si, time_h[0]);
		}
		if(Ncoll_m[0] > 0){
			int col = CollisionCall();
			if(col == 0) return 0;
		}
		if(CollisionFlag == 1 && P.ei > 0 && timeStep % P.ei == 0){
			int rem = RemoveCall();
			if( rem == 0) return 0;
		}
		if(NWriteEnc_m[0] > 0){
			int enc = writeEncCall();
			if(enc == 0) return 0;
		}
		if(si < SIn - 1){
			HC128b_kernel < 512, 2 > <<< 3, 512 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], time_h[0]);
			acc4b_kernel < 256 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq_h[0] * Kt[si], N4[0], Encpairsb_d, Encpairs2_d, test_d, N_h[0], icNB, P.NencMax, time_h[0]);
			EncMatrix_kernel < 512 ><<< dim3(1, N_h[0], 1), dim3(512, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, icNB, P.NencMax, N_h[0], EncFlag_d);
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<<32, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
			if(Nencpairs_h[0] > 0) kick32Ab_kernel < 32 > <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, rcritv_d, dtksq_h[0] * Kt[si], Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
			else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, N_h[0]);
if(P.Usegas == 1 || P.UseForce > 0){
	com128_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
	if(P.Usegas == 1) GasAccCall_1024(time_d, dt_d, Kt[si]);
	if(P.UseForce > 0) force <<< 4, 256 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[si], time_d, N_h[0], Nst, P.UseForce);
	com128_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
}
		}
	}
	HC128b_kernel < 512, 2 > <<< 3, 512 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[SIn - 1], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], time_h[0]);
	acc4b_kernel < 256 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq_h[0] * Kt[SIn - 1], N4[0], Encpairsb_d, Encpairs2_d, test_d, N_h[0], icNB, P.NencMax, time_h[0]);
	EncMatrix_kernel < 512 ><<< dim3(1, N_h[0], 1), dim3(512, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, icNB, P.NencMax, N_h[0], EncFlag_d);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<<32, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
	if(Nencpairs_h[0] > 0) kick32Ab_kernel < 32 > <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, rcritv_d, dtksq_h[0] * Kt[SIn - 1], Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
	else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, N_h[0]);
if(P.Usegas == 1 || P.UseForce > 0){
	com128_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
	if(P.Usegas == 1) GasAccCall_1024(time_d, dt_d, Kt[SIn - 1]);
	if(P.UseForce > 0) force <<< 4, 256 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, N_h[0], Nst, P.UseForce);
	com128_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
}
	if(EjectionFlag_m[0] > 0){
		int Ej = EjectionCall();
		if(Ej == 0) return 0;
	}
#if poincareFlag == 1
	int per = PoincareSectionCall(1024, time_h[0]);
	if(per == 0) return 0;
#endif
	return 1;
}

__host__ int Data::step_2048(){
if(P.Usegas == 1 || P.UseForce > 0 || P.setElements > 0){
	com128_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
if(P.setElements > 0) setElements <<< 8, 256 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst);
	if(P.Usegas == 1) GasAccCall_2048(time_d, dt_d, Kt[0]);
	if(P.UseForce > 0) force <<< 8, 256 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[0], time_d, N_h[0], Nst, P.UseForce);
	com128_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
}
	Rcrit_kernel <<< 64, 32 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0]);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<<64, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
#if G3 == 0
	if(Nencpairs_h[0] > 0 || EjectionFlag2 > 0) kick32Ab_kernel<32> <<<64, 32 >>> (x4_d, v4_d, a_d, rcritv_d, dtksq_h[0] * Kt[SIn - 1], Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
	else kick32B_kernel <<<64, 32 >>> (x4_d, v4_d, a_d, N_h[0]);
#else
	kick32B_kernel <<<64, 32 >>> (x4_d, v4_d, a_d, N_h[0]);
#endif
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		HC128b_kernel < 512, 1 > <<< 3, 512 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], time_h[0]);
		fg_kernel <<< 64, 32 >>>(x4_d, v4_d, xold_d, vold_d, a_d, index_d, groupIndex_d, dt_h[0] * FGt[si], Msun_h[0].x, test_d, N_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, si);
		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 31)/ 32, 32 >>> (x4_d, v4_d, xold_d, vold_d, x4G3_d, v4G3_d, rcrit_d, rcritv_d, dt_h[0] * FGt[si], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, enccount_d, si, K_d, Kold_d, StopTime_d, NB[0], time_h[0], P.WriteEncounters, P.WriteEncountersRadius);
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			int NF = (Nencpairs2_h[0] + 511)/(512);
			if(NF == 1) group_kernel < 1, 512, 3 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, NB[0], N_h[0]);
			if(NF > 1) group_kernel < 1, 512, 4 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, NB[0], N_h[0]);
			cudaDeviceSynchronize();
			BSCall(2048, si, time_h[0]);
		}
		if(Ncoll_m[0] > 0){
			int col = CollisionCall();
			if(col == 0) return 0;
		}
		if(CollisionFlag == 1 && P.ei > 0 && timeStep % P.ei == 0){
			int rem = RemoveCall();
			if( rem == 0) return 0;
		}
		if(NWriteEnc_m[0] > 0){
			int enc = writeEncCall();
			if(enc == 0) return 0;
		}
		if(si < SIn - 1){
			HC128b_kernel < 512, 2 > <<< 3, 512 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], time_h[0]);
			acc4b_kernel < 256 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq_h[0] * Kt[si], N4[0], Encpairsb_d, Encpairs2_d, test_d, N_h[0], icNB, P.NencMax, time_h[0]);
			EncMatrix_kernel < 512 ><<< dim3(1, N_h[0], 1), dim3(512, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, icNB, P.NencMax, N_h[0], EncFlag_d);
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<<64, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
			if(Nencpairs_h[0] > 0) kick32Ab_kernel < 32 > <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, rcritv_d, dtksq_h[0] * Kt[si], Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
			else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, N_h[0]);
if(P.Usegas == 1 || P.UseForce > 0){
	com128_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
	if(P.Usegas == 1) GasAccCall_2048(time_d, dt_d, Kt[si]);
	if(P.UseForce > 0) force <<< 8, 256 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[si], time_d, N_h[0], Nst, P.UseForce);
	com128_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
}
		}
	}
	HC128b_kernel < 512, 2 > <<< 3, 512 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[SIn - 1], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], time_h[0]);
	acc4b_kernel < 256 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq_h[0] * Kt[SIn - 1], N4[0], Encpairsb_d, Encpairs2_d, test_d, N_h[0], icNB, P.NencMax, time_h[0]);
	EncMatrix_kernel < 512 ><<< dim3(1, N_h[0], 1), dim3(512, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, icNB, P.NencMax, N_h[0], EncFlag_d);
	//ForceDriver(x4_d, rcritv_d, a_d, Nencpairs_d, Encpairs_d, Encpairs2_d, dtksq_h[0] * Kt[SIn - 1], icNB[0], NB[0], N_h[0]);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<<64, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
	if(Nencpairs_h[0] > 0) kick32Ab_kernel < 32 > <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, rcritv_d, dtksq_h[0] * Kt[SIn - 1], Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
	else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, N_h[0]);
if(P.Usegas == 1 || P.UseForce > 0){
	com128_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
	if(P.Usegas == 1) GasAccCall_2048(time_d, dt_d, Kt[SIn - 1]);
	if(P.UseForce > 0) force <<< 8, 256 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, N_h[0], Nst, P.UseForce);
	com128_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
}
	if(EjectionFlag_m[0] > 0){
		int Ej = EjectionCall();
		if(Ej == 0) return 0;
	}
#if poincareFlag == 1
	int per = PoincareSectionCall(2048, time_h[0]);
	if(per == 0) return 0;
#endif
	return 1;
}

__host__ int Data::step_largeN(){

if(P.Usegas == 1 || P.UseForce > 0 || P.setElements > 0){
	com128_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
if(P.setElements > 0) setElements <<< (N_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst);
	if(P.Usegas == 1) GasAccCall_largeN(time_d, dt_d, Kt[0]);
	if(P.UseForce > 0) force <<< (N_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[0], time_d, N_h[0], Nst, P.UseForce);
	com128_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
}
	Rcrit_kernel <<< (N_h[0] + 63) / 64, 64 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0]);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<< (NB[0] + 31) / 32, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
#if G3 == 0
	if(Nencpairs_h[0] > 0 || EjectionFlag2 > 0) kick32Ab_kernel < 32 > <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, rcritv_d, dtksq_h[0] * Kt[SIn - 1], Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
	else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, N_h[0]);
#else
	kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, N_h[0]);
#endif
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		HC128b_kernel < 512, 1 > <<< 3, 512 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], time_h[0]);
		fg_kernel <<< (N_h[0] + 31) / 32, 32 >>>(x4_d, v4_d, xold_d, vold_d, a_d, index_d, groupIndex_d, dt_h[0] * FGt[si], Msun_h[0].x, test_d, N_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, si);
		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 31)/ 32, 32 >>> (x4_d, v4_d, xold_d, vold_d, x4G3_d, v4G3_d, rcrit_d, rcritv_d, dt_h[0] * FGt[si], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, enccount_d, si, K_d, Kold_d, StopTime_d, NB[0], time_h[0], P.WriteEncounters, P.WriteEncountersRadius);
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			int NF = (Nencpairs2_h[0] + 511)/(512);
			if(NF == 1) group_kernel < 1, 512, 3 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, NB[0], N_h[0]);
			if(NF > 1) group_kernel < 1, 512, 4 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, NB[0], N_h[0]);
			cudaDeviceSynchronize();
			BSCall(NB[0], si, time_h[0]);
		}
		if(Ncoll_m[0] > 0){
			int col = CollisionCall();
			if(col == 0) return 0;
		}
		if(CollisionFlag == 1 && P.ei > 0 && timeStep % P.ei == 0){
			int rem = RemoveCall();
			if( rem == 0) return 0;
		}
		if(NWriteEnc_m[0] > 0){
			int enc = writeEncCall();
			if(enc == 0) return 0;
		}
		if(si < SIn - 1){
			HC128b_kernel < 512, 2 > <<< 3, 512 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], time_h[0]);
			acc4b_kernel < 256 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq_h[0] * Kt[si], N4[0], Encpairsb_d, Encpairs2_d, test_d, N_h[0], icNB, P.NencMax, time_h[0]);
			EncMatrix_kernel < 512 ><<< dim3(1, N_h[0], 1), dim3(512, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, icNB, P.NencMax, N_h[0], EncFlag_d);
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<< (NB[0] + 31) / 32, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
			if(Nencpairs_h[0] > 0) kick32Ab_kernel < 32 > <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, rcritv_d, dtksq_h[0] * Kt[si], Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
			else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, N_h[0]);


if(P.Usegas == 1 || P.UseForce > 0){
	com128_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
	if(P.Usegas == 1) GasAccCall_largeN(time_d, dt_d, Kt[si]);
	if(P.UseForce > 0) force <<< (N_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[si], time_d, N_h[0], Nst, P.UseForce);
	com128_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
}
		}
	}

	HC128b_kernel < 512, 2 > <<< 3, 512 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[SIn - 1], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], time_h[0]);
	acc4b_kernel < 256 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq_h[0] * Kt[SIn - 1], N4[0], Encpairsb_d, Encpairs2_d, test_d, N_h[0], icNB, P.NencMax, time_h[0]);
	EncMatrix_kernel < 512 ><<< dim3(1, N_h[0], 1), dim3(512, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, icNB, P.NencMax, N_h[0], EncFlag_d);

	//ForceDriver(x4_d, rcritv_d, a_d, Nencpairs_d, Encpairs_d, Encpairs2_d, dtksq_h[0] * Kt[SIn - 1], icNB[0], NB[0], N_h[0]);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<< (NB[0] + 31) / 32, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
	if(Nencpairs_h[0] > 0) kick32Ab_kernel < 32 > <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, rcritv_d, dtksq_h[0] * Kt[SIn - 1], Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
	else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, N_h[0]);


if(P.Usegas == 1 || P.UseForce > 0){
	com128_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
	if(P.Usegas == 1) GasAccCall_largeN(time_d, dt_d, Kt[SIn - 1]);
	if(P.UseForce > 0) force <<< (N_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, N_h[0], Nst, P.UseForce);
	com128_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
}
	if(EjectionFlag_m[0] > 0){
		int Ej = EjectionCall();
		if(Ej == 0) return 0;
	}
#if poincareFlag == 1
	int per = PoincareSectionCall(NB[0], time_h[0]);
	if(per == 0) return 0;
#endif
	return 1;
}

__host__ int Data::step_small(){
if(P.Usegas == 1 || P.UseForce > 0 || P.setElements > 0){
	com128_kernel < 512 > <<< 1, 512 >>> (x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0] + Nsmall_h[0], 1);
if(P.setElements > 0) setElements <<< 1, 16 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst);
	if(P.Usegas == 1){
		GasAccCall_16(time_d, dt_d, Kt[0]);
		GasAccCall_small(time_d, dt_d, Kt[0]);
	}
	if(P.UseForce > 0){
		force <<< (N_h[0] + Nsmall_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[0], time_d, N_h[0] + Nsmall_h[0], Nst, P.UseForce);
		if(P.UseForce == 32) CallYarkovsky2 <<< (N_h[0] + Nsmall_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[0], N_h[0] + Nsmall_h[0], Nst);
		if(P.UseForce == 32) PoyntingRobertsonDrag <<< (N_h[0] + Nsmall_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[0], N_h[0] + Nsmall_h[0], Nst);
	}
	com128_kernel < 512 > <<< 1, 512 >>> (x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0] + Nsmall_h[0], -1);
}
	Rcrit_kernel <<< (N_h[0] + Nsmall_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0] + Nsmall_h[0]);
	if(Nencpairs_h[0] > 0 || EjectionFlag2 > 0) kick32A_kernel <128> <<< (N_h[0] + Nsmall_h[0] + 127)/128, 128 >>> (x4_d, v4_d, a_d, rcritv_d, dtksq_h[0] * Kt[SIn - 1], Encpairs2_d, test_d, N_h[0] + Nsmall_h[0], P.NencMax, time_h[0]);
	else kick32B_kernel <<< (N_h[0] + Nsmall_h[0] + 127)/128, 128 >>> (x4_d, v4_d, a_d, N_h[0] + Nsmall_h[0]);
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		HC128b_kernel <512, 1> <<< 3, 512 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0] + Nsmall_h[0], time_h[0]);
		fg_kernel <<<(N_h[0] + Nsmall_h[0] + 127)/128, 128 >>> (x4_d, v4_d, xold_d, vold_d, a_d, index_d, groupIndex_d, dt_h[0] * FGt[si], Msun_h[0].x, test_d, Nsmall_h[0] + N_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, si);
		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 31)/ 32, 32 >>> (x4_d, v4_d, xold_d, vold_d, x4G3_d, v4G3_d, rcrit_d, rcritv_d, dt_h[0] * FGt[si], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, enccount_d, si, K_d, Kold_d, StopTime_d, N_h[0] + Nsmall_h[0], time_h[0], P.WriteEncounters, P.WriteEncountersRadius);
			cudaDeviceSynchronize();

			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			int NF = (Nencpairs2_h[0] + 511)/(512);
			if(NF == 1) group_kernel < 1, 512, 3 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, N_h[0] + Nsmall_h[0], N_h[0]);
			if(NF > 1) group_kernel < 1, 512, 4 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, N_h[0] + Nsmall_h[0], N_h[0]);

			cudaDeviceSynchronize();
			BSsmallCall(si, time_h[0]);
		}
		if(Ncoll_m[0] > 0){
			int col = CollisionCall();
			if(col == 0) return 0;
		}
if(P.UseForce == 32){
	fragmentCall(random_d, x4_d, v4_d, spin_d, index_d, N_h, N_d, Nsmall_h, Nsmall_d, dt_d, Nst, NconstT, Fragments_d, time_h[0], nFragments_m, nFragments_d, MaxIndex, x4_h, v4_h, spin_h, index_h);
	if(nFragments_m[0] > 0){
		int er = printFragments(nFragments_m[0]);
		if(er == 0) return 0;
		er = RemoveCall();
		if(er == 0) return 0;
	}

	rotationCall(random_d, x4_d, v4_d, spin_d, index_d, N_h, N_d, Nsmall_h, Nsmall_d, dt_d, Nst, Fragments_d, time_h[0], nFragments_m, nFragments_d);
	if(nFragments_m[0] > 0){
		int er = printRotation();
		if(er == 0) return 0;
	}
}
		if(CollisionFlag == 1 && P.ei > 0 && timeStep % P.ei == 0){
			int rem = RemoveCall();
			if( rem == 0) return 0;
		}
		if(NWriteEnc_m[0] > 0){
			int enc = writeEncCall();
			if(enc == 0) return 0;
		}
		if(si < SIn - 1){
			HC128b_kernel<512, 2> <<< 3, 512 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0] + Nsmall_h[0], time_h[0]);
			kicksmall_kernel < 128, 2 > <<< (N_h[0] + Nsmall_h[0] + 127)/128, 128 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq_h[0] * Kt[si], N_h[0], Nencpairs_d, Encpairs_d, Encpairs2_d, Nsmall_h[0], P.NencMax);

if(P.Usegas == 1 || P.UseForce > 0){
	com128_kernel < 512 > <<< 1, 512 >>> (x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0] + Nsmall_h[0], 1);
	if(P.Usegas == 1){
		GasAccCall_16(time_d, dt_d, Kt[si]);
		GasAccCall_small(time_d, dt_d, Kt[si]);
	}
	if(P.UseForce > 0){
		force <<< (N_h[0] + Nsmall_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[si], time_d, N_h[0] + Nsmall_h[0], Nst, P.UseForce);
		if(P.UseForce == 32) CallYarkovsky2 <<< (N_h[0] + Nsmall_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], N_h[0] + Nsmall_h[0], Nst);
		if(P.UseForce == 32) PoyntingRobertsonDrag <<< (N_h[0] + Nsmall_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[si], N_h[0] + Nsmall_h[0], Nst);
	}
	com128_kernel < 512 > <<< 1, 512 >>> (x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0] + Nsmall_h[0], -1);
}
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
		}
	}
	HC128b_kernel<512, 2> <<< 3, 512 >>> (x4_d, v4_d, dtiMsun_h[0] * Ct[SIn - 1], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0] + Nsmall_h[0], time_h[0]);
	kicksmall_kernel < 128, 1 > <<< (N_h[0] + Nsmall_h[0] + 127)/128, 128 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq_h[0] * Kt[SIn - 1], N_h[0], Nencpairs_d, Encpairs_d, Encpairs2_d, Nsmall_h[0], P.NencMax);

/*
accsmall_kernel < 128 > <<< Nsmall_h[0], 128 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq_h[0] * Kt[SIn - 1], Nencpairs_d, Encpairs_d, Encpairs2_d, N_h[0], Nsmall_h[0], P.NencMax, time_h[0], EncFlag_d);
acc4b_kernel < 256 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dtksq_h[0] * Kt[SIn - 1], N4[0], Encpairsb_d, Encpairs2_d, test_d, N_h[0], icNB, P.NencMax, time_h[0]);
EncMatrix_kernel < 512 ><<< dim3(1, N_h[0], 1), dim3(512, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, icNB, P.NencMax, N_h[0], EncFlag_d);
cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
if(Nencpairs_h[0] > 0) kick32A_kernel <128> <<< (N_h[0] + Nsmall_h[0] + 127)/128, 128 >>> (x4_d, v4_d, a_d, rcritv_d, dtksq_h[0] * Kt[SIn - 1], Encpairs2_d, test_d, N_h[0] + Nsmall_h[0], P.NencMax, time_h[0]);
else kick32B_kernel <<< (N_h[0] + Nsmall_h[0] + 127)/128, 128 >>> (x4_d, v4_d, a_d, N_h[0] + Nsmall_h[0]);
*/
// ******

if(P.Usegas == 1 || P.UseForce > 0){
	com128_kernel < 512 > <<< 1, 512 >>> (x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0] + Nsmall_h[0], 1);
	if(P.Usegas == 1){
		GasAccCall_16(time_d, dt_d, Kt[SIn - 1]);
		GasAccCall_small(time_d, dt_d, Kt[SIn - 1]);
	}
	if(P.UseForce > 0){
		force <<< (N_h[0] + Nsmall_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, N_h[0] + Nsmall_h[0], Nst, P.UseForce);
		if(P.UseForce == 32) CallYarkovsky2 <<< (N_h[0] + Nsmall_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst);
		if(P.UseForce == 32) PoyntingRobertsonDrag <<< (N_h[0] + Nsmall_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst);
	}
	com128_kernel < 512 > <<< 1, 512 >>> (x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0] + Nsmall_h[0], -1);
}
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
	if(EjectionFlag_m[0] > 0){
		int Ej = EjectionCall();
		if(Ej == 0) return 0;
	}
	return 1;
}
__host__ int Data::step_M(){
if(P.Usegas == 1 || P.UseForce > 0){
	comM_kernel < HCM_Bl, HCM_Bl2, NmaxM > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, vcom_d, Msun_d, index_d, NBS_d, NT, test_d, 1);
	if(P.Usegas == 1) GasAccCall_M(time_d, dt_d, Kt[0]);
	if(P.UseForce > 0) force <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[0], time_d, NT, Nst, P.UseForce);
	comM_kernel < HCM_Bl, HCM_Bl2, NmaxM > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, vcom_d, Msun_d, index_d, NBS_d, NT, test_d, -1);
}
	RcritM_kernel <<< (NT + 127) / 128, 128>>> (x4_d, v4_d, Msun_d, rcrit_d, rcritv_d, dt_d, test_d, n1_d, n2_d, Rcut_d, RcutSun_d, EjectionFlag_d, index_d, Nst, NT, time_d, idt_d, ict_d, timeStep);
	if(Nencpairs_h[0] > 0 || EjectionFlag2 > 0) KickM2_kernel < KM_Bl, KM_Bl2, NmaxM, 3 > <<< (NT + KM_Bl2 - 1) / KM_Bl2, KM_Bl>>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, Nencpairs_d, Encpairs_d, dtksq_d, Kt[SIn - 1], index_d, NT, test_d);
	else kick32B_kernel <<< (NT + 127) / 128, 128>>> (x4_d, v4_d, a_d, NT);
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		HCM2_kernel < HCM_Bl, HCM_Bl2, NmaxM, 1 > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, dtiMsun_d, index_d, NT, Ct[si], test_d, Nencpairs_d, Nencpairs2_d, Nenc_d, Nst);
		fgM_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, xold_d, vold_d, dt_d, Msun_d, test_d, index_d, NT, aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, FGt[si], si);
		if(Nencpairs_h[0] > 0){
			encounterM_kernel < NmaxM > <<< (Nencpairs_h[0] + 31) / 32 , 32 >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, dt_d, Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, index_d, NBS_d, enccount_d, si, FGt[si], Nst, time_d, P.WriteEncounters, P.WriteEncountersRadius);
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			if(Nencpairs2_h[0] > 0){
				groupM1_kernel < NmaxM, 256> <<< Nencpairs2_h[0], 256 >>> (Nencpairs2_d, Encpairs_d, Encpairs2_d, NBS_d, N_d, Nst);
				groupM2_kernel <<< Nencpairs2_h[0], 16 >>> (Encpairs_d, Encpairs2_d, Nenc_d, NBS_d, N_d, Nst);
				cudaDeviceSynchronize();
				BSBMCall(si);
			}
		}
		if(Ncoll_m[0] > 0){
			int col = CollisionMCall();
			if(col == 0) return 0;
		}
		if(CollisionFlag == 1 && P.ei > 0 && timeStep % P.ei == 0){
			int rem = RemoveCall();
			if( rem == 0) return 0;
		}
		if(NWriteEnc_m[0] > 0){
			int enc = writeEncCall();
			if(enc == 0) return 0;
		}
		if(si < SIn - 1){
			HCM2_kernel < HCM_Bl, HCM_Bl2, NmaxM, 2 > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, dtiMsun_d, index_d, NT, Ct[si], test_d, Nencpairs_d, Nencpairs2_d, Nenc_d, Nst);
			KickM2_kernel < KM_Bl, KM_Bl2, NmaxM, 2 > <<< (NT + KM_Bl2 - 1) / KM_Bl2, KM_Bl>>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, Nencpairs_d, Encpairs_d, dtksq_d, Kt[si], index_d, NT, test_d);
if(P.Usegas == 1 || P.UseForce > 0){
	comM_kernel < HCM_Bl, HCM_Bl2, NmaxM > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, vcom_d, Msun_d, index_d, NBS_d, NT, test_d, 1);
	if(P.Usegas == 1) GasAccCall_M(time_d, dt_d, Kt[si]);
	if(P.UseForce > 0) force <<< (NT + 127) / 128, 128  >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[si], time_d, NT, Nst, P.UseForce);
	comM_kernel < HCM_Bl, HCM_Bl2, NmaxM > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, vcom_d, Msun_d, NBS_d, index_d, NT, test_d, -1);
}
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
		}
	}
	HCM2_kernel < HCM_Bl, HCM_Bl2, NmaxM, 2 > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, dtiMsun_d, index_d, NT, Ct[SIn - 1], test_d, Nencpairs_d, Nencpairs2_d, Nenc_d, Nst);
	KickM2_kernel < KM_Bl, KM_Bl2, NmaxM, 1 > <<< (NT + KM_Bl2 - 1) / KM_Bl2, KM_Bl>>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, Nencpairs_d, Encpairs_d, dtksq_d, Kt[SIn - 1], index_d, NT, test_d);
if(P.Usegas == 1 || P.UseForce > 0){
	comM_kernel < HCM_Bl, HCM_Bl2, NmaxM > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, vcom_d, Msun_d, index_d, NBS_d, NT, test_d, 1);
	if(P.Usegas == 1) GasAccCall_M(time_d, dt_d, Kt[SIn - 1]);
	if(P.UseForce > 0) force <<< (NT + 127) / 128, 128  >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, NT, Nst, P.UseForce);
	comM_kernel < HCM_Bl, HCM_Bl2, NmaxM > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, vcom_d, Msun_d, index_d, NBS_d, NT, test_d, -1);
}
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
	if(EjectionFlag_m[0] > 0){
		EjectionMCall();
	}
	return 1;
}


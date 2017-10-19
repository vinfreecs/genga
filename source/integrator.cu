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
#include "forceYarkovskyOld.h"
#include "Kick4.h"
#include "BSA.h"
#if def_TTV > 0
 #include "BSTTV.h"
 #include "BSTTV2.h"
#endif


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
// This function calls all necessary sub steps for comuting 
// one time step.
// Authors: Simon Grimm
// March 2017
// ****************************************************
__host__ int Data::timeStepLoop(int interrupted){
	time_h[0] = timeStep * idt_h[0] + ict_h[0] * 365.25;
	cudaMemcpy(time_d, time_h, sizeof(double), cudaMemcpyHostToDevice);


	int er;	
	er = step();
	if(er == 0){
		 return 0;
	}


	if(interrupted == 1){
		printf("GENGA is interrupted by SIGINT signal at time step %lld\n", timeStep);
		fprintf(masterfile, "GENGA is interrupted by SIGINT signal at time step %lld\n", timeStep);
		interrupt = 1;
	}
	cudaDeviceSynchronize();
	//Check for too many encounters
	if(EncFlag_m[0] > 0){
		printf("Error: more encounters than allowed. %d %d\n", EncFlag_m[0], P.NencMax);
		fprintf(masterfile, "Error: more encounters than allowed. %d %d\n", EncFlag_m[0], P.NencMax);
		return 0;
	}

	//Check for too big groups//
	if(Nst == 1){
		er = MaxGroups();
		if(er == 0) return 0;
	}

	error = cudaGetLastError();
	if(error != 0){
		printf("Step error = %d = %s %lld\n",error, cudaGetErrorString(error), timeStep);
		fprintf(masterfile, "Step error = %d = %s %lld\n",error, cudaGetErrorString(error), timeStep);
		CoordinateOutput(4);
		return 0;
	}


	//Print Energy and log information//
	if((P.ei > 0 && timeStep % P.ei == 0) || interrupt == 1){
		if(bufferCount + 1 >= P.Buffer || interrupt == 1){
			er = EnergyOutput();
			if(er == 0) return 0;
		}
	}

	if(P.UseaeGrid == 1){ 
		if(timeStep % 10000 == 0){
			copyGridae();
		}
	}
	//update Gas Disk
	if(P.Usegas == 2 && time_h[0] / 365.25 > GasDatatime.y){
		er = readGasFile2(time_h[0] / 365.25);
		if(er == 0){
			return 0;
		}
	}

//test_kernel <<< 1, 16 >>> (x4_d, v4_d, index_d);

	//Print Output//
	if((P.ci > 0 && ((timeStep - 1) % P.ci >= P.ci - P.nci)) || interrupt == 1){
		if(P.Buffer == 1){
			CoordinateOutput(0);
		}
		else if(bufferCount + 1 >= P.Buffer || interrupt == 1){
			//write out buffer
			timestepBuffer[bufferCount] = timeStep;
			for(int st = 0; st < Nst; ++st){
				NBuffer[Nst * (bufferCount) + st].x = N_h[st];
				NBuffer[Nst * (bufferCount) + st].y = Nsmall_h[st];
			}
			CoordinateToBuffer(bufferCount, 0, 0.0);
			++bufferCount;	
			CoordinateOutputBuffer(0);
		}
		else{
			//store in buffer
			timestepBuffer[bufferCount] = timeStep;
			for(int st = 0; st < Nst; ++st){
				NBuffer[Nst * (bufferCount) + st].x = N_h[st];
				NBuffer[Nst * (bufferCount) + st].y = Nsmall_h[st];
			}
			CoordinateToBuffer(bufferCount, 0, 0.0);
			++bufferCount;	
		}
		if(P.UseaeGrid == 1){
			GridaeOutput();
		}

#if poincareFlag == 1
		if((timeStep - 1) % P.ci == P.ci - P.nci){
			fclose(poincarefile);
			sprintf(poincarefilename, "%sPoincare%s_%.12ld.dat", GSF[0].path, GSF[0].X, timeStep);
			//Erase old Poincare files
			poincarefile = fopen(poincarefilename, "w");
		}
#endif
	}


	//print irregular outputs
	if(interrupt == 1 && P.Buffer > 1){
		//write out buffer
		CoordinateOutputBuffer(1);
	}
 	if(P.IrregularOutputs == 1 && irrTimeStep < NIrrOutputs && time_h[0] >= IrrOutputs[irrTimeStep]){

		int ni = 1; //multiple outputs per time step
		for(int i = 0; i < ni; ++i){
			double dTau = -(time_h[0] - IrrOutputs[irrTimeStep]) / idt_h[0];
			IrregularStep(dTau);
			for(int st = 0; st < Nst; ++st){
				time_h[st] += dTau * idt_h[st];
			}
			if(Nst > 1){
				cudaMemcpy(time_d, time_h, Nst * sizeof(double), cudaMemcpyHostToDevice);
			}

			step();

			if(P.Buffer == 1){
				CoordinateOutput(1);
			}
			else if(bufferCountIrr + 1 >= P.Buffer){
				//write out buffer
				timestepBufferIrr[bufferCountIrr] = timeStep;
				for(int st = 0; st < Nst; ++st){
					NBufferIrr[Nst * (bufferCountIrr) + st].x = N_h[st];
					NBufferIrr[Nst * (bufferCountIrr) + st].y = Nsmall_h[st];
				}
				CoordinateToBuffer(bufferCountIrr, 1, dTau);
				++bufferCountIrr;
				CoordinateOutputBuffer(1);
				bufferCountIrr = 0;
				irrTimeStepOut += P.Buffer;
			}
			else{
				//store in buffer
				timestepBufferIrr[bufferCountIrr] = timeStep;
				for(int st = 0; st < Nst; ++st){
					NBufferIrr[Nst * (bufferCountIrr) + st].x = N_h[st];
					NBufferIrr[Nst * (bufferCountIrr) + st].y = Nsmall_h[st];
				}
				CoordinateToBuffer(bufferCountIrr, 1, dTau);
				++bufferCountIrr;
			}

			IrregularStep(-dTau);
			for(int st = 0; st < Nst; ++st){
				time_h[st] -= dTau * idt_h[st];
			}
			if(Nst > 1){
				cudaMemcpy(time_d, time_h, Nst * sizeof(double), cudaMemcpyHostToDevice);
			}

			step();
			SymplecticP(1);

			++irrTimeStep;
		
			dTau = -(time_h[0] - IrrOutputs[irrTimeStep]) / idt_h[0];
			if(dTau <= 0) ++ni;

			if(ni + irrTimeStep - 1 > NIrrOutputs) break;
		}
	}

#if USE_NAF == 1
	//compute the x and y arrays for the naf algorithm
	if(timeStep % P.NAFinterval == 0){
		naf.getnafvarsCall(x4_d, v4_d, index_d, NBS_d, vcom_d, test_d, P.NAFvars, naf.x_d, naf.y_d, Msun_d, Msun_h[0].x, NT, Nst, naf.n, NAFstep, NB[0], N_h[0], Nsmall_h[0], P.UseTestParticles);
		++NAFstep;
		if(NAFstep % P.NAFn0 == 0){
			er = naf.nafCall(NT, N_h, N_d, Nsmall_h, Nsmall_d, Nst, GSF, time_h, time_d, idt_h, P.NAFformat, P.NAFinterval, index_h, index_d, NBS_h);
			if(er == 0) return 0;
			NAFstep = 0;
		}
	}
#endif
	// print time information //
	// this should be the last thing to print, because it is used to restart at the last possible timestep
	if((P.ci > 0 && timeStep % P.ci == 0) || interrupt == 1){
		if(bufferCount >= P.Buffer || P.Buffer == 1 || interrupt == 1){
			er = printTime();
			if(er == 0) return 0;
			fflush(masterfile);
			bufferCount = 0;
		}
	}
	if(interrupt == 1){
		printf("GENGA is terminated by SIGINT signal at time step %lld\n", timeStep);
		fprintf(masterfile, "GENGA is terminated by SIGINT signal at time step %lld\n", timeStep);
		cudaDeviceSynchronize();
		return 0;
	}

	return 1;

}


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
__global__ void initial_kernel(double *K_d, double *Kold_d, double4 *StopTime_d, int *groupIndex_d, int NB){
	int idy = threadIdx.x;
	int idx = blockIdx.x;

	for(int i = 0; i < NB; i += Bl){
		K_d[(idy +i)* NB + idx] = 1.0;
		Kold_d[(idy +i)* NB + idx] = 1.0;
		StopTime_d[(idy +i)* NB + idx].x = -1.0;
		StopTime_d[(idy +i)* NB + idx].y = -1.0;
		StopTime_d[(idy +i)* NB + idx].z = -1.0;
		StopTime_d[(idy +i)* NB + idx].w = -1.0;
		if(idx == 0){
			groupIndex_d[idy + i] = -1;
		}
	}
}

// **************************************
__global__ void initialb_kernel(int2 *Encpairs_d, int2 *Encpairs2_d, int NBNencT){

	int idx = blockIdx.x;
	int id = blockIdx.x * blockDim.x + idx;

	if(id < NBNencT){
		Encpairs_d[id].x = -1;
		Encpairs_d[id].y = -1;

		Encpairs2_d[id].x = -1;
		Encpairs2_d[id].y = -1;
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
//Authors: Simon Grimm
//November 2016
//****************************************/
__global__ void Rcrit_kernel(double4 *__restrict__ x4_d, double4 *__restrict__ v4_d, double4 * __restrict__ x4b_d, double4 *__restrict__ v4b_d, double4 * __restrict__ x4G3_d, double4 *__restrict__ v4G3_d, double Msun, double *__restrict__ rcrit_d, double *__restrict__ rcritv_d, int * __restrict__ index_d, int * __restrict__  indexb_d, double dt, double *__restrict__ test_d, double n1, double n2, double Rcut, double RcutSun, int *EjectionFlag_d, const int N, const int f){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	double4 x4i;
	double4 v4i;

	double rcrit, rcritv ;
	double rsq, vsq, r, v;
	if(id < N){

#if def_StopAtCollision != 0
		if(f == 0){
			x4i = x4_d[id];
			v4i = v4_d[id];

			x4b_d[id] = x4i;
			v4b_d[id] = v4i;
			indexb_d[id] = index_d[id];
		}
		if(f == 1 || f == -1){
			//restore old coordinates
			x4i = x4b_d[id];
			v4i = v4b_d[id];

			v4i.w *= def_CollTshift;

			x4_d[id] = x4i;
			v4_d[id] = v4i;
			index_d[id] = indexb_d[id];
		}
		if(f == 2){
			//restore old coordinates
			x4i = x4b_d[id];
			v4i = v4b_d[id];

			x4_d[id] = x4i;
			v4_d[id] = v4i;
			index_d[id] = indexb_d[id];
		}
#else
		x4i = x4_d[id];
		v4i = v4_d[id];
 #if def_TTV == 1
		v4b_d[id] = v4i;
 #endif
 #if def_TTV == 2
		x4b_d[id] = x4i;
		v4b_d[id] = v4i;
 #endif
#endif
		rsq = x4i.x*x4i.x + x4i.y*x4i.y + x4i.z*x4i.z + 1.0e-30;
		vsq = v4i.x*v4i.x + v4i.y*v4i.y + v4i.z*v4i.z + 1.0e-30;

		r = sqrt(rsq);
		v = sqrt(vsq);

		rcrit = n1 * r * cbrt(x4i.w  / ( Msun * 3.0));
		rcritv = fmax(rcrit, n2 * dt * v);

#if def_StopAtEncounter > 0
		//rescale non n2 rcrit 
		rcrit = def_StopAtEncounterRadius * rcrit / n1;

#endif

		rcrit_d[id] = fmax(rcrit, rcrit_d[id]);
		rcritv_d[id] = fmax(rcritv, rcritv_d[id]);
		//Check for Ejections or to small distances to the Sun
		if((rsq > Rcut * Rcut || rsq < RcutSun * RcutSun) && x4_d[id].w >= 0.0){
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
//Author: Simon Grimm
//November 2016
// ****************************************
__global__ void RcritM_kernel(double4 * __restrict__ x4_d, double4 * __restrict__ v4_d, double4 * __restrict__ x4b_d, double4 * __restrict__ v4b_d, double4 *Msun_d, double *rcrit_d, double *rcritv_d, double *dt_d, double *test_d, double *n1_d, double *n2_d, double *Rcut_d, double *RcutSun_d, int *EjectionFlag_d, int * __restrict__ index_d, int * __restrict__ indexb_d, const int Nst, const int NT, double *time_d, double *idt_d, double *ict_d, long long *delta_d, long long timeStep, int *StopFlag_d, const int f, const int Nstart){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy + Nstart;
	int st = 0;

	if(id < NT + Nstart) st = index_d[id] / 100;
	if(id < Nst + Nstart) time_d[id - Nstart] = timeStep * idt_d[id - Nstart] + ict_d[id - Nstart] * 365.25;

	double4 x4i;
	double4 v4i;

	double rcrit, rcritv;
	double rsq, vsq, r, v;

	if(id < NT + Nstart){
//printf("Rcrit %d %.20g %.20g %.20g %.20g %.20g %.20g\n", id, x4_d[id].x, x4_d[id].y, x4_d[id].z, v4_d[id].x, v4_d[id].y, v4_d[id].z);
		double Msun = Msun_d[st].x;
		double n1 = n1_d[st];
		double n2 = n2_d[st];
		double Rcut = Rcut_d[st];
		double RcutSun = RcutSun_d[st];
		double dt = dt_d[st];

#if def_StopAtCollision != 0
		if(f == 0){
			x4i = x4_d[id];
			v4i = v4_d[id];

			x4b_d[id] = x4i;
			v4b_d[id] = v4i;
			indexb_d[id] = index_d[id];
		}
		if(f == 1 || f == -1){
			x4i = x4b_d[id];
			v4i = v4b_d[id];

			v4i.w *= def_CollTshift;

			x4_d[id] = x4i;
			v4_d[id] = v4i;
			index_d[id] = indexb_d[id];
		}
		if(f == 2){
			x4i = x4b_d[id];
			v4i = v4b_d[id];

			x4_d[id] = x4i;
			v4_d[id] = v4i;
			index_d[id] = indexb_d[id];
		}
#else
		x4i = x4_d[id];
		v4i = v4_d[id];
 #if def_TTV == 1
		v4b_d[id] = v4i;
 #endif
 #if def_TTV == 2
		x4b_d[id] = x4i;
		v4b_d[id] = v4i;
 #endif
#endif

		__syncthreads();

		rsq = x4i.x*x4i.x + x4i.y*x4i.y + x4i.z*x4i.z + 1.0e-30;
		vsq = v4i.x*v4i.x + v4i.y*v4i.y + v4i.z*v4i.z + 1.0e-30;
		r = sqrt(rsq);
		v = sqrt(vsq);

		rcrit = n1 * r * cbrt(x4i.w  / ( Msun * 3.0));
		rcritv = fmax(rcrit, n2 * dt * v);

#if def_StopAtEncounter > 0
		//rescale non n2 rcrit 
		rcrit = def_StopAtEncounterRadius * rcrit / n1;

#endif

		rcrit_d[id] = fmax(rcrit, rcrit_d[id]);
		rcritv_d[id] = fmax(rcritv, rcritv_d[id]);
		//Check for Ejections or to small distances to the Sun
		if((rsq > Rcut * Rcut || rsq < RcutSun * RcutSun) && x4_d[id].w >= 0.0){
			EjectionFlag_d[st + 1] = 1;
			EjectionFlag_d[0] = 1;
		}
		if(timeStep >= delta_d[st]){
			StopFlag_d[0] = 1;
		}
	}
}

__host__ int Data::step(){
	int er;
	//Multi simulation mode
	if(MultiSim == 1){
#if NoEncounters == 0
		er = step_M();
#else
		er = step_MSimple();
#endif
		if(er == 0) return 0;
	}
	else{
		//Test particles
		if(P.UseTestParticles > 0){
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
	cudaMemset(ab_d, 0, NconstT * sizeof(double3));
	initialb_kernel <<< (NBNencT + 255) / 256, 256 >>> (Encpairs_d, Encpairs2_d, NBNencT);
	Rcrit_kernel <<< 1, 16 >>> (x4_d, v4_d, x4b_d, v4b_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, index_d, indexb_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0], 0);
	kick16b_kernel < 40, 0> <<< N_h[0], 32 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, P.NencMax, time_h[0], N_h[0]);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
}
__host__ void Data::firstKick_32(){
	cudaMemset(a_d, 0, NconstT * sizeof(double3));
	cudaMemset(ab_d, 0, NconstT * sizeof(double3));
	initialb_kernel <<< (NBNencT + 255) / 256, 256 >>> (Encpairs_d, Encpairs2_d, NBNencT);
	Rcrit_kernel <<< 1, 32 >>> (x4_d, v4_d, x4b_d, v4b_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, index_d, indexb_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0], 0);
	kick32b_kernel < 32, 64, 0 > <<< N_h[0] , 64 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, NconstT, P.NencMax, time_h[0], N_h[0]);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
}
__host__ void Data::firstKick_64(){
	cudaMemset(a_d, 0, NconstT * sizeof(double3));
	cudaMemset(ab_d, 0, NconstT * sizeof(double3));
	initialb_kernel <<< (NBNencT + 255) / 256, 256 >>> (Encpairs_d, Encpairs2_d, NBNencT);
	Rcrit_kernel <<< 2, 32 >>> (x4_d, v4_d, x4b_d, v4b_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, index_d, indexb_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0], 0);
	kick32b_kernel < 64, 64, 0 > <<< N_h[0] , 64 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, NconstT, P.NencMax, time_h[0], N_h[0]);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
}
__host__ void Data::firstKick_128(){
	cudaMemset(a_d, 0, NconstT * sizeof(double3));
	cudaMemset(ab_d, 0, NconstT * sizeof(double3));
	initialb_kernel <<< (NBNencT + 255) / 256, 256 >>> (Encpairs_d, Encpairs2_d, NBNencT);
	Rcrit_kernel <<< 4, 32 >>> (x4_d, v4_d, x4b_d, v4b_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, index_d, indexb_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0], 0);
	acc128b_kernel < 128 > <<< N2[0] , 128 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, Encpairsb_d, Encpairs2_d, test_d, N_h[0], N2[0], NconstT, P.NencMax, time_h[0]);
	EncMatrix_kernel <<< dim3(1, N_h[0], 1), dim3(128, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, NconstT, P.NencMax, N_h[0], N_h[0], EncFlag_d);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
}
__host__ void Data::firstKick_256(){
	cudaMemset(a_d, 0, NconstT * sizeof(double3));
	cudaMemset(ab_d, 0, NconstT * sizeof(double3));
	initialb_kernel <<< (NBNencT + 255) / 256, 256 >>> (Encpairs_d, Encpairs2_d, NBNencT);
	Rcrit_kernel <<< 8, 32 >>> (x4_d, v4_d, x4b_d, v4b_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, index_d, indexb_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0], 0);
	acc256b_kernel < 128 > <<< N4[0] , 128 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, N4[0], Encpairsb_d, Encpairs2_d, test_d, N_h[0], NconstT, P.NencMax, time_h[0]);
	EncMatrix_kernel <<< dim3(1, N_h[0], 1), dim3(256, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, NconstT, P.NencMax, N_h[0], N_h[0], EncFlag_d);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
}
__host__ void Data::firstKick_512(){
	cudaMemset(a_d, 0, NconstT * sizeof(double3));
	cudaMemset(ab_d, 0, NconstT * sizeof(double3));
	initialb_kernel <<< (NBNencT + 255) / 256, 256 >>> (Encpairs_d, Encpairs2_d, NBNencT);
	Rcrit_kernel <<< 16, 32 >>> (x4_d, v4_d, x4b_d, v4b_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, index_d, indexb_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0], 0);
	acc4b_kernel < 256 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, N4[0], Encpairsb_d, Encpairs2_d, test_d, N_h[0], NconstT, P.NencMax, time_h[0]);
	EncMatrix_kernel <<< dim3(1, N_h[0], 1), dim3(512, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, NconstT, P.NencMax, N_h[0], N_h[0], EncFlag_d);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
}
__host__ void Data::firstKick_1024(){
	cudaMemset(a_d, 0, NconstT * sizeof(double3));
	cudaMemset(ab_d, 0, NconstT * sizeof(double3));
	initialb_kernel <<< (NBNencT + 255) / 256, 256 >>> (Encpairs_d, Encpairs2_d, NBNencT);
	Rcrit_kernel <<< 32, 32 >>> (x4_d, v4_d, x4b_d, v4b_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, index_d, indexb_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0], 0);
	acc4b_kernel < 256 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, N4[0], Encpairsb_d, Encpairs2_d, test_d, N_h[0], NconstT, P.NencMax, time_h[0]);
	EncMatrix_kernel <<< dim3(1, N_h[0], 1), dim3(512, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, NconstT, P.NencMax, N_h[0], N_h[0], EncFlag_d);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
}
__host__ void Data::firstKick_2048(){
	cudaMemset(a_d, 0, NconstT * sizeof(double3));
	cudaMemset(ab_d, 0, NconstT * sizeof(double3));
	initialb_kernel <<< (NBNencT + 255) / 256, 256 >>> (Encpairs_d, Encpairs2_d, NBNencT);
	Rcrit_kernel <<< 32, 64 >>> (x4_d, v4_d, x4b_d, v4b_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, index_d, indexb_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0], 0);
	acc4b_kernel < 256 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, N4[0], Encpairsb_d, Encpairs2_d, test_d, N_h[0], NconstT, P.NencMax, time_h[0]);
	EncMatrix_kernel <<< dim3(1, N_h[0], 1), dim3(512, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, NconstT, P.NencMax, N_h[0], N_h[0], EncFlag_d);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
}
__host__ void Data::firstKick_largeN(){
	cudaMemset(a_d, 0, NconstT * sizeof(double3));
	cudaMemset(ab_d, 0, NconstT * sizeof(double3));
	initialb_kernel <<< (NBNencT + 255) / 256, 256 >>> (Encpairs_d, Encpairs2_d, NBNencT);
	Rcrit_kernel <<< (NB[0] + 63) / 64, 64 >>> (x4_d, v4_d, x4b_d, v4b_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, index_d, indexb_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0], 0);
	acc4b_kernel < 256 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, N4[0], Encpairsb_d, Encpairs2_d, test_d, N_h[0], NconstT, P.NencMax, time_h[0]);
	EncMatrix_kernel <<< dim3(1, N_h[0], 1), dim3(512, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, NconstT, P.NencMax, N_h[0], N_h[0], EncFlag_d);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
}
__host__ void Data::firstKick_small(){
	cudaMemset(a_d, 0, NconstT*sizeof(double3));
	cudaMemset(ab_d, 0, NconstT * sizeof(double3));
	initialb_kernel <<< (NBNencT + 255) / 256, 256 >>> (Encpairs_d, Encpairs2_d, NBNencT);
	Rcrit_kernel <<< (N_h[0] + Nsmall_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, x4b_d, v4b_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, index_d, indexb_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0] + Nsmall_h[0], 0);
	if(NB[0] <= 32 && (P.UseTestParticles == 2 && Nsmall_h[0] < 32)){
		kicksmall_kernel < 0 > <<< (N_h[0] + Nsmall_h[0] + 127)/128, 128 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dt_h[0] * Kt[SIn - 1] * def_ksq, N_h[0], Nencpairs_d, Encpairs_d, Encpairs2_d, Nsmall_h[0], P.NencMax, P.UseTestParticles);
	}
	else{
		int ntx = min(256, ((N_h[0] + 31) / 32) * 32); 
		int nty = 512 / ntx;

		for(int nn = 0; nn < N_h[0] + Nsmall_h[0]; nn += def_MatrixMaxSize){
			int N1 = min(N_h[0] + Nsmall_h[0] - nn, def_MatrixMaxSize);
			int nby = (N1 + nty - 1)/ nty;
			accsmall_kernel < 512, 1 > <<< dim3(1, nby , 1), dim3(ntx, nty, 1) >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, Encpairsb_d, Encpairs2_d, N1, N_h[0], nn, NconstT, P.NencMax, time_h[0]);
			EncMatrixsmall_kernel < 1 > <<< dim3(1, N1, 1), dim3(512, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, NconstT, P.NencMax, N_h[0], N1, nn, EncFlag_d);
		}
		if(P.UseTestParticles == 2){
			for(int nn = N_h[0]; nn < N_h[0] + Nsmall_h[0]; nn += def_MatrixMaxSize){
				int N1 = min(N_h[0] + Nsmall_h[0] - nn, def_MatrixMaxSize);
				ntx = min(256, ((N1+ 31) / 32) * 32); 
				nty = 512 / ntx;
				int nby = (N_h[0] + nty - 1)/ nty;
				accsmall_kernel < 512, 2 > <<< dim3(1, nby , 1), dim3(ntx, nty, 1) >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, Encpairsb_d, Encpairs2_d, N_h[0], N1, nn, NconstT, P.NencMax, time_h[0]);
				
				ntx = min(256, ((N_h[0]+ 31) / 32) * 32); 
				nty = 512 / ntx;
				nby = (N1 + nty - 1)/ nty;
				EncMatrixsmall_kernel < 2 > <<< dim3(1, N1, 1), dim3(512, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, NconstT, P.NencMax, N_h[0], N1, nn, EncFlag_d);
			}
		}

		cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
		if(Nencpairs_h[0] > 0) kick32Ab_kernel <<< (N_h[0] + Nsmall_h[0] + 127)/128, 128 >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Encpairs2_d, test_d, N_h[0] + Nsmall_h[0], P.NencMax, time_h[0]);
		else kick32B_kernel <<< (N_h[0] + Nsmall_h[0] + 127)/128, 128 >>> (x4_d, v4_d, a_d, ab_d, N_h[0] + Nsmall_h[0], dt_h[0] * Kt[SIn - 1] * def_ksq);
	}

	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
}
__host__ void Data::firstKick_M(long long ts){

	cudaMemset(a_d, 0, NconstT*sizeof(double3));
	cudaMemset(ab_d, 0, NconstT * sizeof(double3));
	initialb_kernel <<< (NBNencT + 255) / 256, 256 >>> (Encpairs_d, Encpairs2_d, NBNencT);
	RcritM_kernel <<< (NT + 31) / 32, 32>>> (x4_d, v4_d, x4b_d, v4b_d, Msun_d, rcrit_d, rcritv_d, dt_d, test_d, n1_d, n2_d, Rcut_d, RcutSun_d, EjectionFlag_d, index_d, indexb_d, Nst, NT, time_d, idt_d, ict_d, delta_d, ts, StopFlag_d, 0, Nstart);
	KickM2_kernel < KM_Bl, KM_Bl2, NmaxM, 0 > <<< (NT + KM_Bl2 - 1) / KM_Bl2, KM_Bl >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, Nencpairs_d, Encpairs_d, dt_d, Kt[SIn - 1] * def_ksq, index_d, NT, test_d, Nstart);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
}


cudaStream_t stream[12];
__host__ void Data::BSCall(int si, double time, int noColl){

	time -= dt_h[0] / dayUnit;

//printf(" %d | %d %d %d | %d %d %d | %d %d %d\n", Nenc_m[0], Nenc_m[1], Nenc_m[2], Nenc_m[3], Nenc_m[6], Nenc_m[7], Nenc_m[8], Nenc_m[9], Nenc_m[10], Nenc_m[11]);
	for(int st = 0; st < 9; ++st)   cudaStreamCreate(&stream[st]);
#if G3 < 2

		if(Nenc_m[1] > 0) BSBStep_kernel <2, 2> <<< Nenc_m[1], 4, 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 0, index_d, Ncoll_d, Coll_d, time, spin_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, N_h[0], K_d, Kold_d, groupIndex_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, noColl);
		if(Nenc_m[2] > 0) BSBStep_kernel <4, 4> <<< Nenc_m[2], 16, 0, stream[1] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 1, index_d, Ncoll_d, Coll_d, time, spin_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, N_h[0], K_d, Kold_d, groupIndex_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, noColl);
		if(Nenc_m[3] > 0) BSBStep_kernel <8, 8> <<< Nenc_m[3], 64, 0, stream[2] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 2, index_d, Ncoll_d, Coll_d, time, spin_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, N_h[0], K_d, Kold_d, groupIndex_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, noColl);
		if(Nenc_m[4] > 0) BSBStep_kernel <16, 16> <<< Nenc_m[4], 256, 0, stream[3] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 3, index_d, Ncoll_d, Coll_d, time, spin_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, N_h[0], K_d, Kold_d, groupIndex_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, noColl);
		if(Nenc_m[5] > 0) BSBStep_kernel <32, 8> <<< Nenc_m[5], 256, 0, stream[4] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 4, index_d, Ncoll_d, Coll_d, time, spin_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, N_h[0], K_d, Kold_d, groupIndex_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, noColl);


		if(Nenc_m[6] > 0) BSA_kernel < 64 > <<< Nenc_m[6], 64 , 0, stream[5] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 5, N_h[0], P.NencMax, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, noColl);
		if(Nenc_m[7] > 0) BSA_kernel < 128 > <<< Nenc_m[7], 128 , 0, stream[6] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 6, N_h[0], P.NencMax, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, noColl);
		if(Nenc_m[8] > 0) BSA_kernel < 256 > <<< Nenc_m[8], 256 , 0, stream[7] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 7, N_h[0], P.NencMax, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, noColl);

//		if(Nenc_m[8] > 0) BSA512_kernel < 256, 256 > <<< Nenc_m[8], 256 , 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dx_d, dv_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 7, N_h[0], P.NencMax, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d, dtgr_d, P.UseForce, P.MinMass, P.UseTestParticles, noColl);

		if(Nenc_m[9] > 0) BSA512_kernel < 512, 512 > <<< Nenc_m[9], 512 , 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dx_d, dv_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 8, N_h[0], P.NencMax, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d, dtgr_d, P.UseForce, P.MinMass, P.UseTestParticles, noColl);
//		if(Nenc_m[10] > 0) BSA512_kernel < 1024, 512 > <<< Nenc_m[10], 512 , 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dx_d, dv_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 9, N_h[0], P.NencMax, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d, dtgr_d, P.UseForce, P.MinMass, P.UseTestParticles, noColl);
//		if(Nenc_m[11] > 0) BSA512_kernel < 2048, 512 > <<< Nenc_m[11], 512 , 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dx_d, dv_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 10, N_h[0], P.NencMax, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d, dtgr_d, P.UseForce, P.MinMass, P.UseTestParticles, noColl);

		//if(Nenc_m[8] > 0) BSACall(7, 256, Nenc_m[8], si, time, FGt[si], noColl);
		//if(Nenc_m[9] > 0) BSACall(8, 512, Nenc_m[9], si, time, FGt[si], noColl);
		if(Nenc_m[10] > 0) BSACall(9, 1024, Nenc_m[10], si, time, FGt[si], noColl);
		if(Nenc_m[11] > 0) BSACall(10, 2048, Nenc_m[11], si, time, FGt[si], noColl);

		int nn = 4096;
		for(int st = 11; st < def_GMax - 1; ++st){
			if(Nenc_m[st + 1] > 0) BSACall(st, nn, Nenc_m[st + 1], si, time, FGt[si], noColl);
			nn *= 2;
		}

#else
		if(Nenc_m[1] > 0) BSBKStep_kernel <2, 2> <<< Nenc_m[1], 4, 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 0, index_d, Ncoll_d, Coll_d, time, spin_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, N_h[0], K_d, Kold_d, groupIndex_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d, P.MinMass, P.UseTestParticles, noColl);
		if(Nenc_m[2] > 0) BSBKStep_kernel <4, 4> <<< Nenc_m[2], 16, 0, stream[1] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 1, index_d, Ncoll_d, Coll_d, time, spin_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, N_h[0], K_d, Kold_d, groupIndex_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d, P.MinMass, P.UseTestParticles, noColl);
		if(Nenc_m[3] > 0) BSBKStep_kernel <8, 8> <<< Nenc_m[3], 64, 0, stream[2] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 2, index_d, Ncoll_d, Coll_d, time, spin_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, N_h[0], K_d, Kold_d, groupIndex_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d, P.MinMass, P.UseTestParticles, noColl);
		if(Nenc_m[4] > 0) BSBKStep_kernel <16, 16> <<< Nenc_m[4], 256, 0, stream[3] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 2, index_d, Ncoll_d, Coll_d, time, spin_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, N_h[0], K_d, Kold_d, groupIndex_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d, P.MinMass, P.UseTestParticles, noColl);

//for more than 16 bodies the l loop is needed again

#endif
	for(int st = 0; st < 9; ++st) cudaStreamDestroy(stream[st]);
	cudaDeviceSynchronize();
}

__host__ void Data::BSsmallCall(int si, double time, int noColl){
	time -= dt_h[0] / dayUnit;

	for(int st = 0; st < 9; ++st)   cudaStreamCreate(&stream[st]);
		if(Nenc_m[1] > 0) BSBStep_kernel <2, 2> <<< Nenc_m[1], 4, 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 0, index_d, Ncoll_d, Coll_d, time, spin_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, N_h[0] + Nsmall_h[0], K_d, Kold_d, groupIndex_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, noColl);
		if(Nenc_m[2] > 0) BSBStep_kernel <4, 4> <<< Nenc_m[2], 16, 0, stream[1] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 1, index_d, Ncoll_d, Coll_d, time, spin_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, N_h[0] + Nsmall_h[0], K_d, Kold_d, groupIndex_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, noColl);
		if(Nenc_m[3] > 0) BSBStep_kernel <8, 8> <<< Nenc_m[3], 64, 0, stream[2] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 2, index_d, Ncoll_d, Coll_d, time, spin_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, N_h[0] + Nsmall_h[0], K_d, Kold_d, groupIndex_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, noColl);
		if(Nenc_m[4] > 0) BSBStep_kernel <16, 16> <<< Nenc_m[4], 256, 0, stream[3] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 3, index_d, Ncoll_d, Coll_d, time, spin_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, N_h[0] + Nsmall_h[0], K_d, Kold_d, groupIndex_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, noColl);
		if(Nenc_m[5] > 0) BSBStep_kernel <32, 8> <<< Nenc_m[5], 256, 0, stream[4] >>> (x4_d, v4_d, xold_d, vold_d, a_d, rcrit_d, rcritv_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 4, index_d, Ncoll_d, Coll_d, time, spin_d, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, N_h[0] + Nsmall_h[0], K_d, Kold_d, groupIndex_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, noColl);


		if(Nenc_m[6] > 0) BSA_kernel < 64 > <<< Nenc_m[6], 64 , 0, stream[5] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 5, N_h[0] + Nsmall_h[0], P.NencMax, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, noColl);
		if(Nenc_m[7] > 0) BSA_kernel < 128 > <<< Nenc_m[7], 128 , 0, stream[6] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 6, N_h[0] + Nsmall_h[0], P.NencMax, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, noColl);
		if(Nenc_m[8] > 0) BSA_kernel < 256 > <<< Nenc_m[8], 256 , 0, stream[7] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 7, N_h[0] + Nsmall_h[0], P.NencMax, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, noColl);



		if(Nenc_m[9] > 0) BSA512_kernel < 512, 512 > <<< Nenc_m[9], 512 , 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dx_d, dv_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 8, N_h[0] + Nsmall_h[0], P.NencMax, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d, dtgr_d, P.UseForce, P.MinMass, P.UseTestParticles, noColl);
		//if(Nenc_m[10] > 0) BSA512_kernel < 1024, 512 > <<< Nenc_m[10], 512 , 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dx_d, dv_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 9, N_h[0] + Nsmall_h[0], P.NencMax, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d, dtgr_d, P.UseForce, P.MinMass, P.UseTestParticles, noColl);
//		if(Nenc_m[11] > 0) BSA512_kernel < 2048, 512 > <<< Nenc_m[11], 512 , 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, xp_d, vp_d, xt_d, vt_d, rcrit_d, rcritv_d, index_d, spin_d, Encpairs_d, Encpairs2_d, dx_d, dv_d, dt_h[0] * FGt[si], Msun_h[0].x, U_d, 10, N_h[0] + Nsmall_h[0], P.NencMax, Ncoll_d, Coll_d, time, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d, dtgr_d, P.UseForce, P.MinMass, P.UseTestParticles, noColl);

		if(Nenc_m[10] > 0) BSAsmallCall(9, 1024, Nenc_m[10], si, time, FGt[si], noColl);
		if(Nenc_m[11] > 0) BSAsmallCall(10, 2048, Nenc_m[11], si, time, FGt[si], noColl);

		int nn = 4096;
		for(int st = 11; st < def_GMax - 1; ++st){
			if(Nenc_m[st + 1] > 0) BSAsmallCall(st, nn, Nenc_m[st + 1], si, time, FGt[si], noColl);
			nn *= 2;
		}

	for(int st = 0; st < 9; ++st) cudaStreamDestroy(stream[st]);
	cudaDeviceSynchronize();
}

__host__ void Data::BSBMCall(int si, int noColl){
	for(int st = 0; st < 6; ++st)  cudaStreamCreate(&stream[st]);
		if(Nenc_m[1] > 0) BSBMStep_kernel <2, 2> <<< Nenc_m[1], 4, 0, stream[0] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs_d, Encpairs2_d, dt_d, FGt[si], Msun_d, U_d, 0, index_d, Ncoll_d, Coll_d, time_d, spin_d, Nst, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, noColl);
		if(Nenc_m[2] > 0) BSBMStep_kernel <4, 4> <<< Nenc_m[2], 16, 0, stream[1] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs_d, Encpairs2_d, dt_d, FGt[si], Msun_d, U_d, 1, index_d, Ncoll_d, Coll_d, time_d, spin_d, Nst, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, noColl);
		if(Nenc_m[3] > 0) BSBMStep_kernel <8, 8> <<< Nenc_m[3], 64, 0, stream[2] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs_d, Encpairs2_d, dt_d, FGt[si], Msun_d, U_d, 2, index_d, Ncoll_d, Coll_d, time_d, spin_d, Nst, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, noColl);
		if(Nenc_m[4] > 0) BSBMStep_kernel <16, 16> <<< Nenc_m[4], 256, 0, stream[3] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs_d, Encpairs2_d, dt_d, FGt[si], Msun_d, U_d, 3, index_d, Ncoll_d, Coll_d, time_d, spin_d, Nst, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, noColl);
		if(Nenc_m[5] > 0) BSBMStep_kernel <32, 8> <<< Nenc_m[5], 256, 0, stream[4] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs_d, Encpairs2_d, dt_d, FGt[si], Msun_d, U_d, 4, index_d, Ncoll_d, Coll_d, time_d, spin_d, Nst, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d, P.UseForce, P.MinMass, P.UseTestParticles, noColl);
		//if(Nenc_m[6] > 0) BSBMStep64_kernel <64, 4> <<< Nenc_m[6], 256, 0, stream[5] >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, Encpairs_d, Encpairs2_d, dt_d, FGt[si], Msun_d, U_d, 5, index_d, Ncoll_d, Coll_d, time_d, spin_d, Nst, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, P.WriteEncounters, P.WriteEncountersRadius, NWriteEnc_d, writeEnc_d. P.UseForce, P.MinMass, P.UseTestParticles, noColl);
	for(int st = 0; st < 6; ++st)  cudaStreamDestroy(stream[st]);
	cudaDeviceSynchronize();
}

#if def_TTV == 2
__host__ void Data::BSTTVCall(int n){
	BSTTVStep_kernel < 8, 8 > <<< n * Nst, 64 >>> (x4b_d, v4b_d, Transit_d, N_d, dt_d, Msun_d, index_d, time_d, NBS_d, P.UseForce, P.MinMass, P.UseTestParticles, Nst, TransitTime_d, NtransitsT_d, n);
}
#endif

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
	if(Ncoll_m[0] > def_MaxColl){
		fprintf(masterfile, "Error: More Collisions than def_MaxColl, simulation stopped\n");
		printf("Error: More Collisions than def_MaxColl, simulation stopped\n");
		return 0;
	}
	double Coltime = 1.0e100;
	int stopAtCollision = printCollisions(Coltime);
	CollisionFlag = 1;
#if def_StopAtCollision == 0 
	Ncoll_m[0] = 0;
 	return 1;
#else
	if(stopAtCollision == 1){

printf("Backup step %.20g %.20g %.20g\n", Coltime * 365.25, time_h[0] - idt_h[0], (Coltime * 365.25 - time_h[0] + idt_h[0]) / idt_h[0]);

		IrregularStep(1.0 * ((Coltime * 365.25 - time_h[0] + idt_h[0]) / idt_h[0]));
		double ColtimeOld = Coltime;

		if(def_CollTshift > 1.0){
			bStep(1);
			cudaDeviceSynchronize();
			cudaMemcpy(Coll_h, Coll_d, sizeof(double) * 25 * Ncoll_m[0], cudaMemcpyDeviceToHost);	
			Coltime = Coll_h[0];
			if(Coltime >= ColtimeOld){
printf("Revert time step\n");
				Coltime = -idt_h[0] / 365.25;
				IrregularStep(-1.0);
printf("Backup step3 %.20g %.20g %.20g\n", Coltime * 365.25, time_h[0] - idt_h[0], (Coltime * 365.25 - time_h[0] + idt_h[0]) / idt_h[0]);
				bStep(-1);
				cudaDeviceSynchronize();
				cudaMemcpy(Coll_h, Coll_d, sizeof(double) * 25 * Ncoll_m[0], cudaMemcpyDeviceToHost);
				Coltime = Coll_h[0];
				if(Coltime >= ColtimeOld){
					printf("Error: Collision time could not be reconstructed. Maybe def_CollTshift is too large.\n");
				}

			}
			IrregularStep(1.0 * ((Coltime * 365.25 - time_h[0] + idt_h[0]) / idt_h[0]));
printf("Backup step2 %.20g %.20g %.20g\n", Coltime * 365.25, time_h[0] - idt_h[0], (Coltime * 365.25 - time_h[0] + idt_h[0]) / idt_h[0]);
			bStep(2);
		}
		else{
			bStep(2);
		}
		cudaDeviceSynchronize();
		time_h[0] = Coltime * 365.25;
		CoordinateOutput(2);
		Ncoll_m[0] = 0;

		P.ci = -1;

		return 0;
	}
	else{
		Ncoll_m[0] = 0;
		return 1;
	}
#endif
}
__host__ int Data::CollisionMCall(){
	if(Ncoll_m[0] > def_MaxColl){
		fprintf(masterfile, "Error: More Collisions than def_MaxColl, simulation stopped\n");
		printf("Error: More Collisions than def_MaxColl, simulation stopped\n");
		return 0;
	}
	double Coltime = 1.0e100;
	printCollisions(Coltime);
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
#if def_TTV == 0
	Ejection();
	EjectionFlag_m[0] = 0;
	EjectionFlag_m[1] = 0;
	EjectionFlag2 = 1;
	int NminFlag = remove();
	if(NminFlag == 1){
		return 0;
	}
#endif
	return 1;
}
__host__ void Data::EjectionMCall(){
#if def_TTV == 0
	Ejection();
	int NminFlag = remove();
	if(NminFlag == 1){
		stopSimulations();
		NminFlag = 0;
	}
	EjectionFlag_m[0] = 0;
	EjectionFlag2 = 1;
#endif
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
		cudaMemcpy(x4_h, xold_d, N_h[0] * sizeof(double4), cudaMemcpyDeviceToHost);
		cudaMemcpy(v4_h, vold_d, N_h[0] * sizeof(double4), cudaMemcpyDeviceToHost);
		cudaMemcpy(index_h, index_d, N_h[0] * sizeof(int), cudaMemcpyDeviceToHost);
		for(int i = 0; i < N_h[0]; ++i){
			if(v4_h[i].w < 0.0 && x4_h[i].w >= 0.0){
				fprintf(poincarefile, "%.16g %d %g %g\n", t/365.25, index_h[i], x4_h[i].x, v4_h[i].x);

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

__host__ int Data::bStep(int noColl){
	Rcrit_kernel <<< (N_h[0] + 31) / 32, 32 >>> (x4_d, v4_d, x4b_d, v4b_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, index_d, indexb_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0], noColl);
	kick32C_kernel <<< (N_h[0] + 31) / 32, 32 >>> (x4_d, v4_d, ab_d, N_h[0], dt_h[0] * Kt[0]);

	HC128b_kernel < 512, 0 > <<< 3, 512 >>> (x4_d, v4_d, dt_h[0] * Ct[0], dt_h[0] / Msun_h[0].x * Ct[0], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], time_h[0], P.UseForce);

	fg_kernel <<< (N_h[0] + 31) / 32, 32 >>> (x4_d, v4_d, xold_d, vold_d, a_d, index_d, groupIndex_d, dt_h[0] * FGt[0], Msun_h[0].x, test_d, N_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, 0, P.UseForce);
	cudaDeviceSynchronize();
	BSCall(0, time_h[0], noColl);

	HC128b_kernel < 512, 0 > <<< 3, 512 >>> (x4_d, v4_d, dt_h[0] * Ct[0], dt_h[0] / Msun_h[0].x * Ct[0], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], time_h[0], P.UseForce);
	kick32C_kernel <<< (N_h[0] + 31) / 32, 32 >>> (x4_d, v4_d, ab_d, N_h[0], dt_h[0] * Kt[0]);

	return 0;
}

__host__ int Data::step_16(){
	Rcrit_kernel <<< 1, 16 >>> (x4_d, v4_d, x4b_d, v4b_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, index_d, indexb_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0], 0);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<<1, 16 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
#if G3 == 0
 #if def_TTV != 1
	if(Nencpairs_h[0] > 0 || EjectionFlag2 > 0){
		kick32Ab_kernel <<<1, 16 >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
	}
	else kick32B_kernel <<< 1, 16 >>> (x4_d, v4_d, a_d, ab_d, N_h[0], dt_h[0] * Kt[SIn - 1] * def_ksq);
 #else
	if(Nencpairs_h[0] > 0 || EjectionFlag2 > 0){
		kick32ATTV_kernel <<<1, 16 >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0], dt_h[0], Msun_h[0].x, Msun_h[0].y, Ntransit_d, Transit_d);
	}
	else kick32BTTV_kernel <<<1, 16 >>> (x4_d, v4_d, a_d, ab_d, N_h[0], dt_h[0] * Kt[SIn - 1] * def_ksq, dt_h[0], Msun_h[0].x, Msun_h[0].y, Ntransit_d, Transit_d);
	cudaDeviceSynchronize();
	if(Ntransit_m[0] > 0){
		if(Ntransit_m[0] >= def_NtransitMax - 1){
printf("more Transits than allowed in def_NtransitMax: %d\n", def_NtransitMax);
			return 0;
		}
		BSTTVStep_kernel < 8, 8 > <<< Ntransit_m[0], 64 >>> (x4_d, v4b_d, Transit_d, N_d, dt_d, Msun_d, index_d, time_d, NBS_d, P.UseForce, P.MinMass, P.UseTestParticles, Nst, TransitTime_d, NtransitsT_d, Ntransit_m[0]);
	//	BSTTV2Step_kernel < 8, 4 > <<< dim3((Ntransit_m[0] + 3) / 4, 1, 1), dim3(4, 8, 1) >>> (x4_d, v4b_d, Transit_d, N_d, dt_d, Msun_d, index_d, time_d, NBS_d, P.UseForce, P.MinMass, P.UseTestParticles, Nst, TransitTime_d, NtransitsT_d, Ntransit_m[0]);
		Ntransit_m[0] = 0;
	}
 #endif
#else
	kick32B_kernel <<< 1, 16 >>> (x4_d, v4_d, a_d, ab_d, N_h[0], dt_h[0] * Kt[SIn - 1] * def_ksq);
#endif
	if(ForceFlag > 0 || P.setElements > 0){
		com_kernel < 32 > <<< 1, 16 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
		if(P.setElements > 0) setElements <<< 1, 16 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst);
		if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[SIn - 1]);
		if(P.UseForce > 0) force <<< 1, 16 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, N_h[0], Nst, P.UseForce, 0);
		if(P.UseYarkovsky == 1) CallYarkovsky2 <<< 1, 16 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky <<< 1, 16 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< 1, 16 >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonDrag <<< 1, 16 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		com_kernel < 32 > <<< 1, 16 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
	}
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		HC32_kernel < 32, 1 > <<< 3, 16 >>> (x4_d, v4_d, dt_h[0] * Ct[si], dt_h[0] / Msun_h[0].x * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], P.UseForce);
		fg_kernel <<< 1, 16 >>> (x4_d, v4_d, xold_d, vold_d, a_d, index_d, groupIndex_d, dt_h[0] * FGt[si], Msun_h[0].x, test_d, N_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, si, P.UseForce);
		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 31)/ 32, 32 >>> (x4_d, v4_d, xold_d, vold_d, x4G3_d, v4G3_d, rcrit_d, rcritv_d, dt_h[0] * FGt[si], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, enccount_d, si, K_d, Kold_d, StopTime_d, NB[0], time_h[0], P.WriteEncounters, P.WriteEncountersRadius, P.MinMass);
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			int NF = (Nencpairs2_h[0] + 511)/(512);
			if(NF == 1) group_kernel < 16, 512, 1 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, N_h[0], N_h[0]);
			if(NF > 1) group_kernel < 16, 512, 2 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, N_h[0], N_h[0]);
			cudaDeviceSynchronize();
			BSCall(si, time_h[0], 0);
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
		HC32_kernel < 32, 2> <<< 3, 16 >>> (x4_d, v4_d, dt_h[0] * Ct[si], dt_h[0] / Msun_h[0].x * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], P.UseForce);
		if(si < SIn - 1){
			kick16b_kernel < 40, 2> <<< N_h[0], 32 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dt_h[0] * Kt[si] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, P.NencMax, time_h[0], N_h[0]);
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
			if(ForceFlag > 0){
				com_kernel < 32 > <<< 1, 16 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
				if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[si]);
				if(P.UseForce > 0) force <<< 1, 16 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[si], time_d, N_h[0], Nst, P.UseForce, 0);
				if(P.UseYarkovsky == 1) CallYarkovsky2 <<< 1, 16 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], N_h[0], Nst, 0);
				if(P.UseYarkovsky == 2) CallYarkovsky <<< 1, 16 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], N_h[0], Nst, 0);
				if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< 1, 16 >>> (x4_d, v4_d, index_d, dt_d, Kt[si], N_h[0], Nst, 0);
				if(P.UsePR == 2) PoyntingRobertsonDrag <<< 1, 16 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[si], N_h[0], Nst, 0);
				com_kernel < 32 > <<< 1, 16 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
			}
		}
	}
	EjectionFlag2 = 0;
	kick16b_kernel < 40, 1> <<< N_h[0], 32 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, P.NencMax, time_h[0], N_h[0]);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
	if(ForceFlag > 0){
		com_kernel < 32 > <<< 1, 16 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
		if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[SIn - 1]);
		if(P.UseForce > 0) force <<< 1, 16 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, N_h[0], Nst, P.UseForce, 0);
		if(P.UseYarkovsky == 1) CallYarkovsky2 <<< 1, 16 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky <<< 1, 16 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< 1, 16 >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonDrag <<< 1, 16 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		com_kernel < 32 > <<< 1, 16 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
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
	Rcrit_kernel <<< 1, 32 >>> (x4_d, v4_d, x4b_d, v4b_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, index_d, indexb_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0], 0);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<<1, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
#if G3 == 0
	if(Nencpairs_h[0] > 0 || EjectionFlag2 > 0){
		kick32Ab_kernel <<<1, 32 >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
	}
	else kick32B_kernel <<<1, 32 >>> (x4_d, v4_d, a_d, ab_d, N_h[0], dt_h[0] * Kt[SIn - 1] * def_ksq);
#else
	kick32B_kernel <<<1, 32 >>> (x4_d, v4_d, a_d, ab_d, N_h[0], dt_h[0] * Kt[SIn - 1] * def_ksq);
#endif
	if(ForceFlag > 0 || P.setElements > 0){
		com_kernel < 64 > <<< 1, 32 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
	if(P.setElements > 0) setElements <<< 1, 32 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst);
		if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[SIn - 1]);
		if(P.UseForce > 0) force <<< 1, 32 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, N_h[0], Nst, P.UseForce, 0);
		if(P.UseYarkovsky == 1) CallYarkovsky2 <<< 1, 32 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky <<< 1, 32 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< 1, 32 >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonDrag <<< 1, 32 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		com_kernel < 64 > <<< 1, 32 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
	}
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		HC32_kernel < 64, 1> <<< 3, 32 >>> (x4_d, v4_d, dt_h[0] * Ct[si], dt_h[0] / Msun_h[0].x * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], P.UseForce);
		fg_kernel <<< 1, 32 >>> (x4_d, v4_d, xold_d, vold_d, a_d, index_d, groupIndex_d, dt_h[0] * FGt[si], Msun_h[0].x, test_d, N_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, si, P.UseForce);
		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 31)/ 32, 32 >>> (x4_d, v4_d, xold_d, vold_d, x4G3_d, v4G3_d, rcrit_d, rcritv_d, dt_h[0] * FGt[si], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, enccount_d, si, K_d, Kold_d, StopTime_d, NB[0], time_h[0], P.WriteEncounters, P.WriteEncountersRadius, P.MinMass);
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			int NF = (Nencpairs2_h[0] + 511)/(512);
			if(NF == 1) group_kernel < 32, 512, 1 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, N_h[0], N_h[0]);
			if(NF > 1) group_kernel < 32, 512, 2 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, N_h[0], N_h[0]);
			cudaDeviceSynchronize();

			BSCall(si, time_h[0], 0);
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
		HC32_kernel< 64, 2> <<< 3, 32 >>> (x4_d, v4_d, dt_h[0] * Ct[si], dt_h[0] / Msun_h[0].x * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], P.UseForce);
		if(si < SIn - 1){
			kick32b_kernel<32, 64, 2> <<< N_h[0] , 64 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dt_h[0] * Kt[si] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, NconstT, P.NencMax, time_h[0], N_h[0]);
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
			if(ForceFlag > 0){
				com_kernel < 64 > <<< 1, 32 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
				if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[si]);
				if(P.UseForce > 0) force <<< 1, 32 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[si], time_d, N_h[0], Nst, P.UseForce, 0);
				if(P.UseYarkovsky == 1) CallYarkovsky2 <<< 1, 32 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], N_h[0], Nst, 0);
				if(P.UseYarkovsky == 2) CallYarkovsky <<< 1, 32 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], N_h[0], Nst, 0);
				if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< 1, 32 >>> (x4_d, v4_d, index_d, dt_d, Kt[si], N_h[0], Nst, 0);
				if(P.UsePR == 2) PoyntingRobertsonDrag <<< 1, 32 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[si], N_h[0], Nst, 0);
				com_kernel < 64 > <<< 1, 32 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
			}
		}
	}
	kick32b_kernel<32, 64, 1> <<< N_h[0] , 64 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, NconstT, P.NencMax, time_h[0], N_h[0]);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
	if(ForceFlag > 0){
		com_kernel < 64 > <<< 1, 32 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
		if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[SIn - 1]);
		if(P.UseForce > 0) force <<< 1, 32 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, N_h[0], Nst, P.UseForce, 0);
		if(P.UseYarkovsky == 1) CallYarkovsky2 <<< 1, 32 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky <<< 1, 32 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< 1, 32 >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonDrag <<< 1, 32 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		com_kernel < 64 > <<< 1, 32 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
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
	Rcrit_kernel <<< 2, 32 >>> (x4_d, v4_d, x4b_d, v4b_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, index_d, indexb_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0], 0);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<<2, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
#if G3 == 0
	if(Nencpairs_h[0] > 0 || EjectionFlag2 > 0){
		kick32Ab_kernel <<<2, 32 >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
	}
	else kick32B_kernel <<<2, 32 >>> (x4_d, v4_d, a_d, ab_d, N_h[0], dt_h[0] * Kt[SIn - 1] * def_ksq);
#else
	kick32B_kernel <<<2, 32 >>> (x4_d, v4_d, a_d, ab_d, N_h[0], dt_h[0] * Kt[SIn - 1] * def_ksq);
#endif
	if(ForceFlag > 0 || P.setElements > 0){
		com_kernel < 64 > <<< 1, 64 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
	if(P.setElements > 0) setElements <<< 1, 64 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst);
		if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[SIn - 1]);
		if(P.UseForce > 0) force <<< 1, 64 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, N_h[0], Nst, P.UseForce, 0);
		if(P.UseYarkovsky == 1) CallYarkovsky2 <<< 1, 64 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky <<< 1, 64 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< 1, 64 >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonDrag <<< 1, 64 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		com_kernel < 64 > <<< 1, 64 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
	}
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		HC32_kernel < 64, 1 > <<< 3, 64 >>> (x4_d, v4_d, dt_h[0] * Ct[si], dt_h[0] / Msun_h[0].x * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], P.UseForce);
		fg_kernel <<< 2, 32 >>>(x4_d, v4_d, xold_d, vold_d, a_d, index_d, groupIndex_d, dt_h[0] * FGt[si], Msun_h[0].x, test_d, N_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, si, P.UseForce);
		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 31)/ 32, 32 >>> (x4_d, v4_d, xold_d, vold_d, x4G3_d, v4G3_d, rcrit_d, rcritv_d, dt_h[0] * FGt[si], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, enccount_d, si, K_d, Kold_d, StopTime_d, NB[0], time_h[0], P.WriteEncounters, P.WriteEncountersRadius, P.MinMass);
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			int NF = (Nencpairs2_h[0] + 511)/(512);
			if(NF == 1) group_kernel < 64, 512, 1 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, N_h[0], N_h[0]);
			if(NF > 1) group_kernel < 64, 512, 2 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, N_h[0], N_h[0]);
			cudaDeviceSynchronize();
			BSCall(si, time_h[0], 0);
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
		HC32_kernel < 64, 2 > <<< 3, 64 >>> (x4_d, v4_d, dt_h[0] * Ct[si], dt_h[0] / Msun_h[0].x * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], P.UseForce);
		if(si < SIn - 1){
			kick32b_kernel<64, 64, 2 > <<< N_h[0] , 64 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dt_h[0] * Kt[si] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, NconstT, P.NencMax, time_h[0], N_h[0]);
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
			if(ForceFlag > 0){
				com_kernel < 64 > <<< 1, 64 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
				if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[si]);
				if(P.UseForce > 0) force <<< 1, 64 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[si], time_d, N_h[0], Nst, P.UseForce, 0);
				if(P.UseYarkovsky == 1) CallYarkovsky2 <<< 1, 64 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], N_h[0], Nst, 0);
				if(P.UseYarkovsky == 2) CallYarkovsky <<< 1, 64 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], N_h[0], Nst, 0);
				if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< 1, 64 >>> (x4_d, v4_d, index_d, dt_d, Kt[si], N_h[0], Nst, 0);
				if(P.UsePR == 2) PoyntingRobertsonDrag <<< 1, 64 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[si], N_h[0], Nst, 0);
				com_kernel < 64 > <<< 1, 64 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
			}
		}
	}
	kick32b_kernel<64, 64, 1 > <<< N_h[0] , 64 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Nencpairs_d, Encpairs_d, Encpairs2_d, test_d, NconstT, P.NencMax, time_h[0], N_h[0]);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
	if(ForceFlag > 0){
		com_kernel < 64 > <<< 1, 64 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
		if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[SIn - 1]);
		if(P.UseForce > 0) force <<< 1, 64 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, N_h[0], Nst, P.UseForce, 0);
		if(P.UseYarkovsky == 1) CallYarkovsky2 <<< 1, 64 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky <<< 1, 64 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< 1, 64 >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonDrag <<< 1, 64 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		com_kernel < 64 > <<< 1, 64 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
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
	Rcrit_kernel <<< 4, 32 >>> (x4_d, v4_d, x4b_d, v4b_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, index_d, indexb_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0], 0);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<<4, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
#if G3 == 0
	if(Nencpairs_h[0] > 0 || EjectionFlag2 > 0){
		kick32Ab_kernel <<<4, 32 >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
	}
	else kick32B_kernel <<<4, 32 >>> (x4_d, v4_d, a_d, ab_d, N_h[0], dt_h[0] * Kt[SIn - 1] * def_ksq);
#else
	kick32B_kernel <<<4, 32 >>> (x4_d, v4_d, a_d, ab_d, N_h[0], dt_h[0] * Kt[SIn - 1] * def_ksq);
#endif
	if(ForceFlag > 0 || P.setElements > 0){
		com_kernel < 128 > <<< 1, 128 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
	if(P.setElements > 0) setElements <<< 1, 128 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst);
		if(P.UseForce > 0) force <<< 1, 128 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, N_h[0], Nst, P.UseForce, 0);
		if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[SIn - 1]);
		if(P.UseYarkovsky == 1) CallYarkovsky2 <<< 1, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky <<< 1, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< 1, 128 >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonDrag <<< 1, 128 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		com_kernel < 128 > <<< 1, 128 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
	}
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		HC128b_kernel < 128, 1 > <<< 3, 128 >>> (x4_d, v4_d, dt_h[0] * Ct[si], dt_h[0] / Msun_h[0].x * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], time_h[0], P.UseForce);
		fg_kernel <<< 4, 32 >>>(x4_d, v4_d, xold_d, vold_d, a_d, index_d, groupIndex_d, dt_h[0] * FGt[si], Msun_h[0].x, test_d, N_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, si, P.UseForce);
		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 31)/ 32, 32 >>> (x4_d, v4_d, xold_d, vold_d, x4G3_d, v4G3_d, rcrit_d, rcritv_d, dt_h[0] * FGt[si], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, enccount_d, si, K_d, Kold_d, StopTime_d, NB[0], time_h[0], P.WriteEncounters, P.WriteEncountersRadius, P.MinMass);
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			int NF = (Nencpairs2_h[0] + 511)/(512);
			if(NF == 1) group_kernel < 128, 512, 1 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, N_h[0], N_h[0]);
			if(NF > 1) group_kernel < 128, 512, 2 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, N_h[0], N_h[0]);
			cudaDeviceSynchronize();
			BSCall(si, time_h[0], 0);
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
		HC128b_kernel < 128, 2 > <<< 3, 128 >>> (x4_d, v4_d, dt_h[0] * Ct[si], dt_h[0] / Msun_h[0].x * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], time_h[0], P.UseForce);
		if(si < SIn - 1){
			acc128b_kernel<128> <<< N2[0] , 128 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, Encpairsb_d, Encpairs2_d, test_d, N_h[0], N2[0], NconstT, P.NencMax, time_h[0]);
			EncMatrix_kernel <<< dim3(1, N_h[0], 1), dim3(128, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, NconstT, P.NencMax, N_h[0], N_h[0], EncFlag_d);
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<<4, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
			if(Nencpairs_h[0] > 0){
				kick32Ab_kernel <<< 4, 32 >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[si] * def_ksq, Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
			}
			else kick32B_kernel <<< 4, 32 >>> (x4_d, v4_d, a_d, ab_d, N_h[0], dt_h[0] * Kt[si] * def_ksq);
			if(ForceFlag > 0){
				com_kernel < 128 > <<< 1, 128 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
				if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[si]);
				if(P.UseForce > 0) force <<< 1, 128 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[si], time_d, N_h[0], Nst, P.UseForce, 0);
				if(P.UseYarkovsky == 1) CallYarkovsky2 <<< 1, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], N_h[0], Nst, 0);
				if(P.UseYarkovsky == 2) CallYarkovsky <<< 1, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], N_h[0], Nst, 0);
				if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< 1, 128 >>> (x4_d, v4_d, index_d, dt_d, Kt[si], N_h[0], Nst, 0);
				if(P.UsePR == 2) PoyntingRobertsonDrag <<< 1, 128 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[si], N_h[0], Nst, 0);
				com_kernel < 128 > <<< 1, 128 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
			}
		}
	}
	acc128b_kernel<128> <<< N2[0] , 128 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, Encpairsb_d, Encpairs2_d, test_d, N_h[0], N2[0], NconstT, P.NencMax, time_h[0]);
	EncMatrix_kernel <<< dim3(1, N_h[0], 1), dim3(128, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, NconstT, P.NencMax, N_h[0], N_h[0], EncFlag_d);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<<4, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
	if(Nencpairs_h[0] > 0){
			 kick32Ab_kernel <<< 4, 32 >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
	}
	else kick32B_kernel <<< 4, 32 >>> (x4_d, v4_d, a_d, ab_d, N_h[0], dt_h[0] * Kt[SIn - 1] * def_ksq);
	if(ForceFlag > 0){
		com_kernel < 128 > <<< 1, 128 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
		if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[SIn - 1]);
		if(P.UseForce > 0) force <<< 1, 128 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, N_h[0], Nst, P.UseForce, 0);
		if(P.UseYarkovsky == 1) CallYarkovsky2 <<< 1, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky <<< 1, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< 1, 128 >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonDrag <<< 1, 128 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		com_kernel < 128 > <<< 1, 128 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
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
	Rcrit_kernel <<< 8, 32 >>> (x4_d, v4_d, x4b_d, v4b_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, index_d, indexb_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0], 0);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<<8, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
#if G3 == 0
	if(Nencpairs_h[0] > 0 || EjectionFlag2 > 0){
		kick32Ab_kernel <<<8, 32 >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
	}
	else kick32B_kernel <<<8, 32 >>> (x4_d, v4_d, a_d, ab_d, N_h[0], dt_h[0] * Kt[SIn - 1] * def_ksq);
#else
	kick32B_kernel <<<8, 32 >>> (x4_d, v4_d, a_d, ab_d, N_h[0], dt_h[0] * Kt[SIn - 1]) * def_ksq;
#endif
	if(ForceFlag > 0 || P.setElements > 0){
		com_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
	if(P.setElements > 0) setElements <<< 1, 256 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst);
		if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[SIn - 1]);
		if(P.UseForce > 0) force <<< 1, 256 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, N_h[0], Nst, P.UseForce, 0);
		if(P.UseYarkovsky == 1) CallYarkovsky2 <<< 1, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky <<< 1, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< 1, 256 >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonDrag <<< 1, 256 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		com_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x,  test_d, N_h[0], -1);
	}
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		HC128b_kernel < 256, 1 > <<< 3, 256 >>> (x4_d, v4_d, dt_h[0] * Ct[si], dt_h[0] / Msun_h[0].x * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], time_h[0], P.UseForce);
		fg_kernel <<< 8, 32 >>>(x4_d, v4_d, xold_d, vold_d, a_d, index_d, groupIndex_d, dt_h[0] * FGt[si], Msun_h[0].x, test_d, N_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, si, P.UseForce);
		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 31)/ 32, 32 >>> (x4_d, v4_d, xold_d, vold_d, x4G3_d, v4G3_d, rcrit_d, rcritv_d, dt_h[0] * FGt[si], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, enccount_d, si, K_d, Kold_d, StopTime_d, NB[0], time_h[0], P.WriteEncounters, P.WriteEncountersRadius, P.MinMass);
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			int NF = (Nencpairs2_h[0] + 511)/(512);
			if(NF == 1) group_kernel < 256, 512, 1 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, N_h[0], N_h[0]);
			if(NF > 1) group_kernel < 256, 512, 2 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, N_h[0], N_h[0]);

			cudaDeviceSynchronize();
			BSCall(si, time_h[0], 0);		
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
		HC128b_kernel < 256, 2 > <<< 3, 256 >>> (x4_d, v4_d, dt_h[0] * Ct[si], dt_h[0] / Msun_h[0].x * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], time_h[0], P.UseForce);
		if(si < SIn - 1){
			acc256b_kernel < 128 > <<< N4[0] , 128 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, N4[0], Encpairsb_d, Encpairs2_d, test_d, N_h[0], NconstT, P.NencMax, time_h[0]);
			EncMatrix_kernel <<< dim3(1, N_h[0], 1), dim3(256, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, NconstT, P.NencMax, N_h[0], N_h[0], EncFlag_d);
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<<8, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
			if(Nencpairs_h[0] > 0){
				kick32Ab_kernel <<< 8, 32 >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[si] * def_ksq, Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
			}
			else kick32B_kernel <<< 8, 32 >>> (x4_d, v4_d, a_d, ab_d, N_h[0], dt_h[0] * Kt[si] * def_ksq);
			if(ForceFlag > 0){
				com_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
				if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[si]);
				if(P.UseForce > 0) force <<< 1, 256 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[si], time_d, N_h[0], Nst, P.UseForce, 0);
				if(P.UseYarkovsky == 1) CallYarkovsky2 <<< 1, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], N_h[0], Nst, 0);
				if(P.UseYarkovsky == 2) CallYarkovsky <<< 1, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], N_h[0], Nst, 0);
				if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< 1, 256 >>> (x4_d, v4_d, index_d, dt_d, Kt[si], N_h[0], Nst, 0);
				if(P.UsePR == 2) PoyntingRobertsonDrag <<< 1, 256 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[si], N_h[0], Nst, 0);
				com_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
			}
		}
	}
	acc256b_kernel < 128 > <<< N4[0] , 128 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, N4[0], Encpairsb_d, Encpairs2_d, test_d, N_h[0], NconstT, P.NencMax, time_h[0]);
	EncMatrix_kernel <<< dim3(1, N_h[0], 1), dim3(256, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, NconstT, P.NencMax, N_h[0], N_h[0], EncFlag_d);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<<8, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
	if(Nencpairs_h[0] > 0){
		kick32Ab_kernel <<< 8, 32 >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
	}
	else kick32B_kernel <<< 8, 32 >>> (x4_d, v4_d, a_d, ab_d, N_h[0], dt_h[0] * Kt[SIn - 1] * def_ksq);
	if(ForceFlag > 0){
		com_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
		if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[SIn - 1]);
		if(P.UseForce > 0) force <<< 1, 256 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, N_h[0], Nst, P.UseForce, 0);
		if(P.UseYarkovsky == 1) CallYarkovsky2 <<< 1, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky <<< 1, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< 1, 256 >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonDrag <<< 1, 256 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		com_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
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
	Rcrit_kernel <<< 16, 32 >>> (x4_d, v4_d, x4b_d, v4b_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, index_d, indexb_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0], 0);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<<16, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
#if G3 == 0
	if(Nencpairs_h[0] > 0 || EjectionFlag2 > 0) kick32Ab_kernel <<<16, 32 >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
	else kick32B_kernel <<<16, 32 >>> (x4_d, v4_d, a_d, ab_d, N_h[0], dt_h[0] * Kt[SIn - 1] * def_ksq);
#else
	kick32B_kernel <<<16, 32 >>> (x4_d, v4_d, a_d, ab_d, N_h[0], dt_h[0] * Kt[SIn - 1] * def_ksq);
#endif
	if(ForceFlag > 0 || P.setElements > 0){
		com_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
	if(P.setElements > 0) setElements <<< 2, 256 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst);
		if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[SIn - 1]);
		if(P.UseForce > 0) force <<< 2, 256 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, N_h[0], Nst, P.UseForce, 0);
		if(P.UseYarkovsky == 1) CallYarkovsky2 <<< 2, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky <<< 2, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< 2, 256 >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonDrag <<< 2, 256 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		com_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
	}
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		HC128b_kernel < 512, 1 > <<< 3, 512 >>> (x4_d, v4_d, dt_h[0] * Ct[si], dt_h[0] / Msun_h[0].x * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], time_h[0], P.UseForce);
		fg_kernel <<< 16, 32 >>>(x4_d, v4_d, xold_d, vold_d, a_d, index_d, groupIndex_d, dt_h[0] * FGt[si], Msun_h[0].x, test_d, N_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, si, P.UseForce);
		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 31)/ 32, 32 >>> (x4_d, v4_d, xold_d, vold_d, x4G3_d, v4G3_d, rcrit_d, rcritv_d, dt_h[0] * FGt[si], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, enccount_d, si, K_d, Kold_d, StopTime_d, NB[0], time_h[0], P.WriteEncounters, P.WriteEncountersRadius, P.MinMass);
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			int NF = (Nencpairs2_h[0] + 511)/(512);
			if(NF == 1) group_kernel < 512, 512, 1 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, N_h[0], N_h[0]);
			if(NF > 1) group_kernel < 512, 512, 2 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, N_h[0], N_h[0]);
			cudaDeviceSynchronize();
			BSCall(si, time_h[0], 0);
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
		HC128b_kernel < 512, 2 > <<< 3, 512 >>> (x4_d, v4_d, dt_h[0] * Ct[si], dt_h[0] / Msun_h[0].x * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], time_h[0], P.UseForce);
		if(si < SIn - 1){
			acc4b_kernel < 256 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, N4[0], Encpairsb_d, Encpairs2_d, test_d, N_h[0], NconstT, P.NencMax, time_h[0]);
			EncMatrix_kernel <<< dim3(1, N_h[0], 1), dim3(512, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, NconstT, P.NencMax, N_h[0], N_h[0], EncFlag_d);
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<<16, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
			if(Nencpairs_h[0] > 0) kick32Ab_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[si] * def_ksq, Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
			else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, ab_d, N_h[0], dt_h[0] * Kt[si] * def_ksq);

			if(ForceFlag > 0){
				com_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
				if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[si]);
				if(P.UseForce > 0) force <<< 2, 256 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[si], time_d, N_h[0], Nst, P.UseForce, 0);
				if(P.UseYarkovsky == 1) CallYarkovsky2 <<< 2, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], N_h[0], Nst, 0);
				if(P.UseYarkovsky == 2) CallYarkovsky <<< 2, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], N_h[0], Nst, 0);
				if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< 2, 256 >>> (x4_d, v4_d, index_d, dt_d, Kt[si], N_h[0], Nst, 0);
				if(P.UsePR == 2) PoyntingRobertsonDrag <<< 2, 256 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[si], N_h[0], Nst, 0);
				com_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
			}
		}
	}

	acc4b_kernel < 256 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, N4[0], Encpairsb_d, Encpairs2_d, test_d, N_h[0], NconstT, P.NencMax, time_h[0]);
	EncMatrix_kernel <<< dim3(1, N_h[0], 1), dim3(512, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, NconstT, P.NencMax, N_h[0], N_h[0], EncFlag_d);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<<16, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
	if(Nencpairs_h[0] > 0) kick32Ab_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
	else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, ab_d, N_h[0], dt_h[0] * Kt[SIn - 1] * def_ksq);
	if(ForceFlag > 0){
		com_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
		if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[SIn - 1]);
		if(P.UseForce > 0) force <<< 2, 256 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, N_h[0], Nst, P.UseForce, 0);
		if(P.UseYarkovsky == 1) CallYarkovsky2 <<< 2, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky <<< 2, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< 2, 256 >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonDrag <<< 2, 256 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		com_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
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
	Rcrit_kernel <<< 32, 32 >>> (x4_d, v4_d, x4b_d, v4b_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, index_d, indexb_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0], 0);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<<32, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
#if G3 == 0
	if(Nencpairs_h[0] > 0 || EjectionFlag2 > 0) kick32Ab_kernel <<<32, 32 >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
	else kick32B_kernel <<<32, 32 >>> (x4_d, v4_d, a_d, ab_d, N_h[0], dt_h[0] * Kt[SIn - 1] * def_ksq);
#else
	kick32B_kernel <<<32, 32 >>> (x4_d, v4_d, a_d, ab_d, N_h[0], dt_h[0] * Kt[SIn - 1] * def_ksq);
#endif
	if(ForceFlag > 0 || P.setElements > 0){
		com_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
	if(P.setElements > 0) setElements <<< 4, 256 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst);
		if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[SIn - 1]);
		if(P.UseForce > 0) force <<< 4, 256 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, N_h[0], Nst, P.UseForce, 0);
		if(P.UseYarkovsky == 1) CallYarkovsky2 <<< 4, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky <<< 4, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< 4, 256 >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonDrag <<< 4, 256 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		com_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
	}
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		HC128b_kernel < 512, 1 > <<< 3, 512 >>> (x4_d, v4_d, dt_h[0] * Ct[si], dt_h[0] / Msun_h[0].x * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], time_h[0], P.UseForce);
		fg_kernel <<< 32, 32 >>>(x4_d, v4_d, xold_d, vold_d, a_d, index_d, groupIndex_d, dt_h[0] * FGt[si], Msun_h[0].x, test_d, N_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, si, P.UseForce);
		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 31)/ 32, 32 >>> (x4_d, v4_d, xold_d, vold_d, x4G3_d, v4G3_d, rcrit_d, rcritv_d, dt_h[0] * FGt[si], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, enccount_d, si, K_d, Kold_d, StopTime_d, NB[0], time_h[0], P.WriteEncounters, P.WriteEncountersRadius, P.MinMass);
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			int NF = (Nencpairs2_h[0] + 511)/(512);
			if(NF == 1) group_kernel < 1, 512, 3> <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, N_h[0], N_h[0]);
			if(NF > 1) group_kernel < 1, 512, 4> <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, N_h[0], N_h[0]);
			cudaDeviceSynchronize();
			BSCall(si, time_h[0], 0);
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
		HC128b_kernel < 512, 2 > <<< 3, 512 >>> (x4_d, v4_d, dt_h[0] * Ct[si], dt_h[0] / Msun_h[0].x * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], time_h[0], P.UseForce);
		if(si < SIn - 1){
			acc4b_kernel < 256 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, N4[0], Encpairsb_d, Encpairs2_d, test_d, N_h[0], NconstT, P.NencMax, time_h[0]);
			EncMatrix_kernel <<< dim3(1, N_h[0], 1), dim3(512, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, NconstT, P.NencMax, N_h[0], N_h[0], EncFlag_d);
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<<32, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
			if(Nencpairs_h[0] > 0) kick32Ab_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[si] * def_ksq, Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
			else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, ab_d, N_h[0], dt_h[0] * Kt[si] * def_ksq);
			if(ForceFlag > 0){
				com_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
				if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[si]);
				if(P.UseForce > 0) force <<< 4, 256 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[si], time_d, N_h[0], Nst, P.UseForce, 0);
				if(P.UseYarkovsky == 1) CallYarkovsky2 <<< 4, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], N_h[0], Nst, 0);
				if(P.UseYarkovsky == 2) CallYarkovsky <<< 4, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], N_h[0], Nst, 0);
				if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< 4, 256 >>> (x4_d, v4_d, index_d, dt_d, Kt[si], N_h[0], Nst, 0);
				if(P.UsePR == 2) PoyntingRobertsonDrag <<< 4, 256 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[si], N_h[0], Nst, 0);
				com_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
			}
		}
	}
	acc4b_kernel < 256 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, N4[0], Encpairsb_d, Encpairs2_d, test_d, N_h[0], NconstT, P.NencMax, time_h[0]);
	EncMatrix_kernel <<< dim3(1, N_h[0], 1), dim3(512, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, NconstT, P.NencMax, N_h[0], N_h[0], EncFlag_d);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<<32, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
	if(Nencpairs_h[0] > 0) kick32Ab_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
	else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, ab_d, N_h[0], dt_h[0] * Kt[SIn - 1] * def_ksq);
	if(ForceFlag > 0){
		com_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
		if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[SIn - 1]);
		if(P.UseForce > 0) force <<< 4, 256 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, N_h[0], Nst, P.UseForce, 0);
		if(P.UseYarkovsky == 1) CallYarkovsky2 <<< 4, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky <<< 4, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< 4, 256 >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonDrag <<< 4, 256 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		com_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
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
	Rcrit_kernel <<< 64, 32 >>> (x4_d, v4_d, x4b_d, v4b_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, index_d, indexb_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0], 0);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<<64, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
#if G3 == 0
	if(Nencpairs_h[0] > 0 || EjectionFlag2 > 0) kick32Ab_kernel <<<64, 32 >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
	else kick32B_kernel <<<64, 32 >>> (x4_d, v4_d, a_d, ab_d, N_h[0], dt_h[0] * Kt[SIn - 1] * def_ksq);
#else
	kick32B_kernel <<<64, 32 >>> (x4_d, v4_d, a_d, ab_d, N_h[0], dt_h[0] * Kt[SIn - 1] * def_ksq);
#endif
	if(ForceFlag > 0 || P.setElements > 0){
		com_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
	if(P.setElements > 0) setElements <<< 8, 256 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst);
		if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[SIn - 1]);
		if(P.UseForce > 0) force <<< 8, 256 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, N_h[0], Nst, P.UseForce, 0);
		if(P.UseYarkovsky == 1) CallYarkovsky2 <<< 8, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky <<< 8, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< 8, 256 >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonDrag <<< 8, 256 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		com_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
	}
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		HC128b_kernel < 512, 1 > <<< 3, 512 >>> (x4_d, v4_d, dt_h[0] * Ct[si], dt_h[0] / Msun_h[0].x * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], time_h[0], P.UseForce);
		fg_kernel <<< 64, 32 >>>(x4_d, v4_d, xold_d, vold_d, a_d, index_d, groupIndex_d, dt_h[0] * FGt[si], Msun_h[0].x, test_d, N_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, si, P.UseForce);
		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 31)/ 32, 32 >>> (x4_d, v4_d, xold_d, vold_d, x4G3_d, v4G3_d, rcrit_d, rcritv_d, dt_h[0] * FGt[si], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, enccount_d, si, K_d, Kold_d, StopTime_d, NB[0], time_h[0], P.WriteEncounters, P.WriteEncountersRadius, P.MinMass);
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			int NF = (Nencpairs2_h[0] + 511)/(512);
			if(NF == 1) group_kernel < 1, 512, 3 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, N_h[0], N_h[0]);
			if(NF > 1) group_kernel < 1, 512, 4 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, N_h[0], N_h[0]);
			cudaDeviceSynchronize();
			BSCall(si, time_h[0], 0);
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
		HC128b_kernel < 512, 2 > <<< 3, 512 >>> (x4_d, v4_d, dt_h[0] * Ct[si], dt_h[0] / Msun_h[0].x * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], time_h[0], P.UseForce);
		if(si < SIn - 1){
			acc4b_kernel < 256 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, N4[0], Encpairsb_d, Encpairs2_d, test_d, N_h[0], NconstT, P.NencMax, time_h[0]);
			EncMatrix_kernel <<< dim3(1, N_h[0], 1), dim3(512, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, NconstT, P.NencMax, N_h[0], N_h[0], EncFlag_d);
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<<64, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
			if(Nencpairs_h[0] > 0) kick32Ab_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[si] * def_ksq, Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
			else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, ab_d, N_h[0], dt_h[0] * Kt[si] * def_ksq);
			if(ForceFlag > 0){
				com_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
				if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[si]);
				if(P.UseForce > 0) force <<< 8, 256 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[si], time_d, N_h[0], Nst, P.UseForce, 0);
				if(P.UseYarkovsky == 1) CallYarkovsky2 <<< 8, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], N_h[0], Nst, 0);
				if(P.UseYarkovsky == 2) CallYarkovsky <<< 8, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], N_h[0], Nst, 0);
				if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< 8, 256 >>> (x4_d, v4_d, index_d, dt_d, Kt[si], N_h[0], Nst, 0);
				if(P.UsePR == 2) PoyntingRobertsonDrag <<< 8, 256 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[si], N_h[0], Nst, 0);
				com_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
			}
		}
	}
	acc4b_kernel < 256 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, N4[0], Encpairsb_d, Encpairs2_d, test_d, N_h[0], NconstT, P.NencMax, time_h[0]);
	EncMatrix_kernel <<< dim3(1, N_h[0], 1), dim3(512, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, NconstT, P.NencMax, N_h[0], N_h[0], EncFlag_d);
	//ForceDriver(x4_d, rcritv_d, a_d, Nencpairs_d, Encpairs_d, Encpairs2_d, dt_h[0] * Kt[SIn - 1] * def_ksq, NconstT[0], NB[0], N_h[0]);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<<64, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
	if(Nencpairs_h[0] > 0) kick32Ab_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
	else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, ab_d, N_h[0], dt_h[0] * Kt[SIn - 1] * def_ksq);
	if(ForceFlag > 0){
		com_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
		if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[SIn - 1]);
		if(P.UseForce > 0) force <<< 8, 256 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, N_h[0], Nst, P.UseForce, 0);
		if(P.UseYarkovsky == 1) CallYarkovsky2 <<< 8, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky <<< 8, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< 8, 256 >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonDrag <<< 8, 256 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		com_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
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

	Rcrit_kernel <<< (N_h[0] + 63) / 64, 64 >>> (x4_d, v4_d, x4b_d, v4b_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, index_d, indexb_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0], 0);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<< (NB[0] + 31) / 32, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
#if G3 == 0
	if(Nencpairs_h[0] > 0 || EjectionFlag2 > 0) kick32Ab_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
	else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, ab_d, N_h[0], dt_h[0] * Kt[SIn - 1] * def_ksq);
#else
	kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, ab_d, N_h[0], dt_h[0] * Kt[SIn - 1] * def_ksq);
#endif
	if(ForceFlag > 0 || P.setElements > 0){
		com_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
	if(P.setElements > 0) setElements <<< (N_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst);
		if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[SIn - 1]);
		if(P.UseForce > 0) force <<< (N_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, N_h[0], Nst, P.UseForce, 0);
		if(P.UseYarkovsky == 1) CallYarkovsky2 <<< (N_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky <<< (N_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< (N_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonDrag <<< (N_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		com_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
	}
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		HC128b_kernel < 512, 1 > <<< 3, 512 >>> (x4_d, v4_d, dt_h[0] * Ct[si], dt_h[0] / Msun_h[0].x * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], time_h[0], P.UseForce);
		fg_kernel <<< (N_h[0] + 31) / 32, 32 >>>(x4_d, v4_d, xold_d, vold_d, a_d, index_d, groupIndex_d, dt_h[0] * FGt[si], Msun_h[0].x, test_d, N_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, si, P.UseForce);
		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 31)/ 32, 32 >>> (x4_d, v4_d, xold_d, vold_d, x4G3_d, v4G3_d, rcrit_d, rcritv_d, dt_h[0] * FGt[si], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, enccount_d, si, K_d, Kold_d, StopTime_d, NB[0], time_h[0], P.WriteEncounters, P.WriteEncountersRadius, P.MinMass);
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			int NF = (Nencpairs2_h[0] + 511)/(512);
			if(NF == 1) group_kernel < 1, 512, 3 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, N_h[0], N_h[0]);
			if(NF > 1) group_kernel < 1, 512, 4 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, N_h[0], N_h[0]);
			cudaDeviceSynchronize();
			BSCall(si, time_h[0], 0);
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

		HC128b_kernel < 512, 2 > <<< 3, 512 >>> (x4_d, v4_d, dt_h[0] * Ct[si], dt_h[0] / Msun_h[0].x * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0], time_h[0], P.UseForce);
		if(si < SIn - 1){
			acc4b_kernel < 256 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, N4[0], Encpairsb_d, Encpairs2_d, test_d, N_h[0], NconstT, P.NencMax, time_h[0]);
			EncMatrix_kernel <<< dim3(1, N_h[0], 1), dim3(512, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, NconstT, P.NencMax, N_h[0], N_h[0], EncFlag_d);
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<< (NB[0] + 31) / 32, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
			if(Nencpairs_h[0] > 0) kick32Ab_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[si] * def_ksq, Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
			else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, ab_d, N_h[0], dt_h[0] * Kt[si] * def_ksq);


			if(ForceFlag > 0){
				com_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
				if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[si]);
				if(P.UseForce > 0) force <<< (N_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[si], time_d, N_h[0], Nst, P.UseForce, 0);
				if(P.UseYarkovsky == 1) CallYarkovsky2 <<< (N_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], N_h[0], Nst, 0);
				if(P.UseYarkovsky == 2) CallYarkovsky <<< (N_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], N_h[0], Nst, 0);
				if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< (N_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, index_d, dt_d, Kt[si], N_h[0], Nst, 0);
				if(P.UsePR == 2) PoyntingRobertsonDrag <<< (N_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[si], N_h[0], Nst, 0);
				com_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
			}
		}
	}

	acc4b_kernel < 256 > <<< N4[0] , 256 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, N4[0], Encpairsb_d, Encpairs2_d, test_d, N_h[0], NconstT, P.NencMax, time_h[0]);
	EncMatrix_kernel <<< dim3(1, N_h[0], 1), dim3(512, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, NconstT, P.NencMax, N_h[0], N_h[0], EncFlag_d);

	//ForceDriver(x4_d, rcritv_d, a_d, Nencpairs_d, Encpairs_d, Encpairs2_d, dt_h[0] * Kt[SIn - 1] * def_ksq, NconstT[0], NB[0], N_h[0]);
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
#if SERIAL_GROUPING == 1
	Sortb_kernel<<< (NB[0] + 31) / 32, 32 >>>(Encpairs2_d, N_h[0], P.NencMax);
#endif
	if(Nencpairs_h[0] > 0) kick32Ab_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Encpairs2_d, test_d, N_h[0], P.NencMax, time_h[0]);
	else kick32B_kernel <<< (NB[0] + 31) / 32, 32 >>> (x4_d, v4_d, a_d, ab_d, N_h[0], dt_h[0] * Kt[SIn - 1] * def_ksq);

	if(ForceFlag > 0){
		com_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], 1);
		if(P.Usegas == 1) GasAccCall(time_d, dt_d, Kt[SIn - 1]);
		if(P.UseForce > 0) force <<< (N_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, N_h[0], Nst, P.UseForce, 0);
		if(P.UseYarkovsky == 1) CallYarkovsky2 <<< (N_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky <<< (N_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< (N_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonDrag <<< (N_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0], Nst, 0);
		com_kernel < 256 > <<< 1, 256 >>>(x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0], -1);
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
	Rcrit_kernel <<< (N_h[0] + Nsmall_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, x4b_d, v4b_d, x4G3_d, v4G3_d, Msun_h[0].x, rcrit_d, rcritv_d, index_d, indexb_d, dt_h[0], test_d, n1_h[0], n2_h[0], Rcut_h[0], RcutSun_h[0], EjectionFlag_d, N_h[0] + Nsmall_h[0], 0);
	if(Nencpairs_h[0] > 0 || EjectionFlag2 > 0) kick32Ab_kernel <<< (N_h[0] + Nsmall_h[0] + 127)/128, 128 >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Encpairs2_d, test_d, N_h[0] + Nsmall_h[0], P.NencMax, time_h[0]);
	else kick32B_kernel <<< (N_h[0] + Nsmall_h[0] + 127)/128, 128 >>> (x4_d, v4_d, a_d, ab_d, N_h[0] + Nsmall_h[0], dt_h[0] * Kt[SIn - 1] * def_ksq);
	if(ForceFlag > 0 || P.setElements > 0){
		com_kernel < 512 > <<< 1, 512 >>> (x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0] + Nsmall_h[0], 1);
	if(P.setElements > 0) setElements <<< 1, 16 >>> (x4_d, v4_d, index_d, setElementsData_d, setElementsLine_d, Msun_d, dt_d, time_d, N_h[0], Nst);
		if(P.Usegas == 1){
			GasAccCall(time_d, dt_d, Kt[SIn - 1]);
			GasAccCall_small(time_d, dt_d, Kt[SIn - 1]);
		}
		if(P.Usegas == 2){
			//GasAccCall(time_d, dt_d, Kt[SIn - 1]);
			GasAccCall2_small(time_d, dt_d, Kt[SIn - 1]);
		}
		if(P.UseForce > 0) force <<< (N_h[0] + Nsmall_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, N_h[0] + Nsmall_h[0], Nst, P.UseForce, 0);
		if(P.UseYarkovsky == 1) CallYarkovsky2 <<< (N_h[0] + Nsmall_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky <<< (N_h[0] + Nsmall_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< (N_h[0] + Nsmall_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonDrag <<< (N_h[0] + Nsmall_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0);
		com_kernel < 512 > <<< 1, 512 >>> (x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0] + Nsmall_h[0], -1);
	}
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		HC128b_kernel < 512, 1 > <<< 3, 512 >>> (x4_d, v4_d, dt_h[0] * Ct[si], dt_h[0] / Msun_h[0].x * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0] + Nsmall_h[0], time_h[0], P.UseForce);
		fg_kernel <<<(N_h[0] + Nsmall_h[0] + 127)/128, 128 >>> (x4_d, v4_d, xold_d, vold_d, a_d, index_d, groupIndex_d, dt_h[0] * FGt[si], Msun_h[0].x, test_d, N_h[0] + Nsmall_h[0], aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, si, P.UseForce);
		if(Nencpairs_h[0] > 0){
			encounter_kernel <<< (Nencpairs_h[0] + 31)/ 32, 32 >>> (x4_d, v4_d, xold_d, vold_d, x4G3_d, v4G3_d, rcrit_d, rcritv_d, dt_h[0] * FGt[si], Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, enccount_d, si, K_d, Kold_d, StopTime_d, N_h[0] + Nsmall_h[0], time_h[0], P.WriteEncounters, P.WriteEncountersRadius, P.MinMass);
			cudaDeviceSynchronize();

			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			int NF = (Nencpairs2_h[0] + 511)/(512);
			if(P.UseTestParticles < 2){
				if(NF == 1) group_kernel < 1, 512, 3 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, N_h[0] + Nsmall_h[0], N_h[0]);
				if(NF > 1) group_kernel < 1, 512, 4 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, N_h[0] + Nsmall_h[0], N_h[0]);
			}
			else{
				if(NF == 1) group_kernel < 1, 512, 3 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, N_h[0] + Nsmall_h[0], N_h[0] + Nsmall_h[0]);
				if(NF > 1) group_kernel < 1, 512, 4 > <<< 1, 512 >>> (Nenc_d, test_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, groupIndex_d, N_h[0] + Nsmall_h[0], N_h[0] + Nsmall_h[0]);
			}

			cudaDeviceSynchronize();
			BSsmallCall(si, time_h[0], 0);
		}
		if(Ncoll_m[0] > 0){
			int col = CollisionCall();
			if(col == 0) return 0;
		}
		if(P.UseSmallCollisions == 1){
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
		HC128b_kernel < 512, 2 > <<< 3, 512 >>> (x4_d, v4_d, dt_h[0] * Ct[si], dt_h[0] / Msun_h[0].x * Ct[si], Nencpairs_d, Nencpairs2_d, Nenc_d, N_h[0] + Nsmall_h[0], time_h[0], P.UseForce);
		if(si < SIn - 1){
			if(NB[0] <= 32 && (P.UseTestParticles == 2 && Nsmall_h[0] < 32)){
				kicksmall_kernel < 2 > <<< (N_h[0] + Nsmall_h[0] + 127)/128, 128 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dt_h[0] * Kt[si] * def_ksq, N_h[0], Nencpairs_d, Encpairs_d, Encpairs2_d, Nsmall_h[0], P.NencMax, P.UseTestParticles);
			}
			else{
				int ntx = min(256, ((N_h[0] + 31) / 32) * 32); 
				int nty = 512 / ntx;
				for(int nn = 0; nn < N_h[0] + Nsmall_h[0]; nn += def_MatrixMaxSize){
					int N1 = min(N_h[0] + Nsmall_h[0] - nn, def_MatrixMaxSize);
					int nby = (N1 + nty - 1)/ nty;
					accsmall_kernel < 512, 1 > <<< dim3(1, nby, 1), dim3(ntx, nty, 1) >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, Encpairsb_d, Encpairs2_d, N1, N_h[0], nn, NconstT, P.NencMax, time_h[0]);
					EncMatrixsmall_kernel < 1 > <<< dim3(1, N1, 1), dim3(512, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, NconstT, P.NencMax, N_h[0], N1, nn, EncFlag_d);
				}
				if(P.UseTestParticles == 2){
					ntx = min(256, ((Nsmall_h[0]+ 31) / 32) * 32); 
					nty = 512 / ntx;
					for(int nn = N_h[0]; nn < N_h[0] + Nsmall_h[0]; nn += def_MatrixMaxSize){
						int N1 = min(N_h[0] + Nsmall_h[0]- nn, def_MatrixMaxSize);
						int nby = (N1 + nty - 1)/ nty;
						accsmall_kernel < 512, 2 > <<< dim3(1, nby , 1), dim3(ntx, nty, 1) >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, Encpairsb_d, Encpairs2_d, N_h[0], N1, nn, NconstT, P.NencMax, time_h[0]);
						ntx = min(256, ((N_h[0]+ 31) / 32) * 32); 
						nty = 512 / ntx;
						nby = (N1 + nty - 1)/ nty;
						EncMatrixsmall_kernel < 2 > <<< dim3(1, N1, 1), dim3(512, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, NconstT, P.NencMax, N_h[0], N1, nn, EncFlag_d);
					}
				}

				cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
				if(Nencpairs_h[0] > 0) kick32Ab_kernel <<< (N_h[0] + Nsmall_h[0] + 127)/128, 128 >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[si] * def_ksq, Encpairs2_d, test_d, N_h[0] + Nsmall_h[0], P.NencMax, time_h[0]);
				else kick32B_kernel <<< (N_h[0] + Nsmall_h[0] + 127)/128, 128 >>> (x4_d, v4_d, a_d, ab_d, N_h[0] + Nsmall_h[0], dt_h[0] * Kt[si] * def_ksq);
			}
			if(ForceFlag > 0){
				com_kernel < 512 > <<< 1, 512 >>> (x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0] + Nsmall_h[0], 1);
				if(P.Usegas == 1){
					GasAccCall(time_d, dt_d, Kt[si]);
					GasAccCall_small(time_d, dt_d, Kt[si]);
				}
				if(P.Usegas == 2){
					//GasAccCall(time_d, dt_d, Kt[si]);
					GasAccCall2_small(time_d, dt_d, Kt[si]);
				}
				if(P.UseForce > 0) force <<< (N_h[0] + Nsmall_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[si], time_d, N_h[0] + Nsmall_h[0], Nst, P.UseForce, 0);
				if(P.UseYarkovsky == 1) CallYarkovsky2 <<< (N_h[0] + Nsmall_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], N_h[0] + Nsmall_h[0], Nst, 0);
				if(P.UseYarkovsky == 2) CallYarkovsky <<< (N_h[0] + Nsmall_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], N_h[0] + Nsmall_h[0], Nst, 0);
				if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< (N_h[0] + Nsmall_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, index_d, dt_d, Kt[si], N_h[0] + Nsmall_h[0], Nst, 0);
				if(P.UsePR == 2) PoyntingRobertsonDrag <<< (N_h[0] + Nsmall_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[si], N_h[0] + Nsmall_h[0], Nst, 0);
				
				com_kernel < 512 > <<< 1, 512 >>> (x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0] + Nsmall_h[0], -1);
			}
			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
		}
	}
	if(NB[0] <= 32 && (P.UseTestParticles == 2 && Nsmall_h[0] < 32)){
		kicksmall_kernel < 1 > <<< (N_h[0] + Nsmall_h[0] + 127)/128, 128 >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, dt_h[0] * Kt[SIn - 1] * def_ksq, N_h[0], Nencpairs_d, Encpairs_d, Encpairs2_d, Nsmall_h[0], P.NencMax, P.UseTestParticles);
	}
	else{
		int ntx = min(256, ((N_h[0] + 31) / 32) * 32); 
		int nty = 512 / ntx;
		for(int nn = 0; nn < N_h[0] + Nsmall_h[0]; nn += def_MatrixMaxSize){
			int N1 = min(N_h[0] + Nsmall_h[0] - nn, def_MatrixMaxSize);
			int nby = (N1 + nty - 1)/ nty;
			accsmall_kernel < 512, 1 > <<< dim3(1, nby, 1), dim3(ntx, nty, 1) >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, Encpairsb_d, Encpairs2_d, N1, N_h[0], nn, NconstT, P.NencMax, time_h[0]);
			EncMatrixsmall_kernel < 1 > <<< dim3(1, N1, 1), dim3(512, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, NconstT, P.NencMax, N_h[0], N1, nn, EncFlag_d);
		}
		if(P.UseTestParticles == 2){
			ntx = min(256, ((Nsmall_h[0] + 31) / 32) * 32);
			nty = 512 / ntx;
			for(int nn = N_h[0]; nn < N_h[0] + Nsmall_h[0]; nn += def_MatrixMaxSize){
				int N1 = min(N_h[0] + Nsmall_h[0]- nn, def_MatrixMaxSize);
				int nby = (N1 + nty - 1)/ nty;
				accsmall_kernel < 512, 2 > <<< dim3(1, nby , 1), dim3(ntx, nty, 1) >>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, groupIndex_d, Encpairsb_d, Encpairs2_d, N_h[0], N1, nn, NconstT, P.NencMax, time_h[0]);
				ntx = min(256, ((N_h[0]+ 31) / 32) * 32); 
				nty = 512 / ntx;
				nby = (N1 + nty - 1)/ nty;
				EncMatrixsmall_kernel < 2 > <<< dim3(1, N1, 1), dim3(512, 1, 1) >>> (Encpairsb_d, Encpairs_d, Encpairs2_d, Nencpairs_d, NconstT, P.NencMax, N_h[0], N1, nn, EncFlag_d);
			}
		}

		cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
		if(Nencpairs_h[0] > 0) kick32Ab_kernel <<< (N_h[0] + Nsmall_h[0] + 127)/128, 128 >>> (x4_d, v4_d, a_d, ab_d, rcritv_d, dt_h[0] * Kt[SIn - 1] * def_ksq, Encpairs2_d, test_d, N_h[0] + Nsmall_h[0], P.NencMax, time_h[0]);
		else kick32B_kernel <<< (N_h[0] + Nsmall_h[0] + 127)/128, 128 >>> (x4_d, v4_d, a_d, ab_d, N_h[0] + Nsmall_h[0], dt_h[0] * Kt[SIn - 1] * def_ksq);
	}

	if(ForceFlag > 0){
		com_kernel < 512 > <<< 1, 512 >>> (x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0] + Nsmall_h[0], 1);
		if(P.Usegas == 1){
			GasAccCall(time_d, dt_d, Kt[SIn - 1]);
			GasAccCall_small(time_d, dt_d, Kt[SIn - 1]);
		}
		if(P.Usegas == 2){
			//GasAccCall(time_d, dt_d, Kt[SIn - 1]);
			GasAccCall2_small(time_d, dt_d, Kt[SIn - 1]);
		}
		if(P.UseForce > 0) force <<< (N_h[0] + Nsmall_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, N_h[0] + Nsmall_h[0], Nst, P.UseForce, 0);
		if(P.UseYarkovsky == 1) CallYarkovsky2 <<< (N_h[0] + Nsmall_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0);
		if(P.UseYarkovsky == 2) CallYarkovsky <<< (N_h[0] + Nsmall_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0);
		if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< (N_h[0] + Nsmall_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0);
		if(P.UsePR == 2) PoyntingRobertsonDrag <<< (N_h[0] + Nsmall_h[0] + 255) / 256, 256 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], N_h[0] + Nsmall_h[0], Nst, 0);
		
		com_kernel < 512 > <<< 1, 512 >>> (x4_d, v4_d, vcom_d, Msun_h[0].x, test_d, N_h[0] + Nsmall_h[0], -1);
	}
	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
	if(EjectionFlag_m[0] > 0){
		int Ej = EjectionCall();
		if(Ej == 0) return 0;
	}
	return 1;
}
__host__ int Data::step_M(){
	RcritM_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, x4b_d, v4b_d, Msun_d, rcrit_d, rcritv_d, dt_d, test_d, n1_d, n2_d, Rcut_d, RcutSun_d, EjectionFlag_d, index_d, indexb_d, Nst, NT, time_d, idt_d, ict_d, delta_d, timeStep, StopFlag_d, 0, Nstart);
#if def_TTV != 1
	if(Nencpairs_h[0] > 0 || EjectionFlag2 > 0) KickM2_kernel < KM_Bl, KM_Bl2, NmaxM, 3 > <<< (NT + KM_Bl2 - 1) / KM_Bl2, KM_Bl>>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, Nencpairs_d, Encpairs_d, dt_d, Kt[SIn - 1], index_d, NT, test_d, Nstart);
	else kick32BM_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, a_d, ab_d, index_d, NT, dt_d, Kt[SIn - 1], Nstart);
#else
	if(Nencpairs_h[0] > 0 || EjectionFlag2 > 0) KickM2TTV_kernel < KM_Bl, KM_Bl2, NmaxM, 3 > <<< (NT + KM_Bl2 - 1) / KM_Bl2, KM_Bl>>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, Nencpairs_d, Encpairs_d, dt_d, Kt[SIn - 1], index_d, NT, test_d, Msun_d, Ntransit_d, Transit_d, Nstart);
	else kick32BMTTV_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, a_d, ab_d, index_d, NT, dt_d, Kt[SIn - 1], Msun_d, Ntransit_d, Transit_d, Nstart);
	cudaDeviceSynchronize();
	if(Ntransit_m[0] > 0){
		if(Ntransit_m[0] >= def_NtransitMax - 1){
printf("more Transits than allowed in def_NtransitMax: %d\n", def_NtransitMax);
			return 0;
		}
		BSTTVStep_kernel < 8, 8 > <<< Ntransit_m[0], 64 >>> (x4_d, v4b_d, Transit_d, N_d, dt_d, Msun_d, index_d, time_d, NBS_d, P.UseForce, P.MinMass, , P.UseTestParticlesNst, TransitTime_d, NtransitsT_d, Ntransit_m[0]);
	//	BSTTV2Step_kernel < 8, 4 > <<< dim3((Ntransit_m[0] + 3) / 4, 1, 1), dim3(4, 8, 1) >>> (x4_d, v4b_d, Transit_d, N_d, dt_d, Msun_d, index_d, time_d, NBS_d, P.UseForce, P.MinMass, P.UseTestParticles, Nst, TransitTime_d, NtransitsT_d, Ntransit_m[0]);
		Ntransit_m[0] = 0;
	}
#endif
	if(ForceFlag > 0){
		comM_kernel < HCM_Bl, HCM_Bl2, NmaxM > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, vcom_d, Msun_d, index_d, NBS_d, NT, test_d, 1, Nstart);
		if(P.Usegas == 1) GasAccCall_M(time_d, dt_d, Kt[SIn - 1]);
		if(P.UseForce > 0) force <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, NT, Nst, P.UseForce, Nstart);
		if(P.UseYarkovsky == 1) CallYarkovsky2 <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], NT, Nst, Nstart);
		if(P.UseYarkovsky == 2) CallYarkovsky <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], NT, Nst, Nstart);
		if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], NT, Nst, Nstart);
		if(P.UsePR == 2) PoyntingRobertsonDrag <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], NT, Nst, Nstart);
		comM_kernel < HCM_Bl, HCM_Bl2, NmaxM > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, vcom_d, Msun_d, index_d, NBS_d, NT, test_d, -1, Nstart);
	}
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		HCM2_kernel < HCM_Bl, HCM_Bl2, NmaxM, 1 > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, dt_d, Msun_d, index_d, NT, Ct[si], test_d, Nencpairs_d, Nencpairs2_d, Nenc_d, Nst, P.UseForce, Nstart);
		fgM_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, xold_d, vold_d, dt_d, Msun_d, test_d, index_d, NT, aelimits_d, aecount_d, Gridaecount_d, Gridaicount_d, FGt[si], si, P.UseForce, Nstart);
		if(Nencpairs_h[0] > 0){
			encounterM_kernel < NmaxM > <<< (Nencpairs_h[0] + 31) / 32 , 32 >>> (x4_d, v4_d, xold_d, vold_d, rcrit_d, rcritv_d, dt_d, Nencpairs_d, Encpairs_d, Nencpairs2_d, Encpairs2_d, test_d, index_d, NBS_d, N_d, enccount_d, si, FGt[si], Nst, time_d, P.WriteEncounters, P.WriteEncountersRadius, StopFlag_d, P.MinMass);
			cudaMemcpy(Nencpairs2_h, Nencpairs2_d, sizeof(int), cudaMemcpyDeviceToHost);
			if(Nencpairs2_h[0] > 0){
				groupM1_kernel < 256> <<< Nencpairs2_h[0], 256 >>> (Nencpairs2_d, Encpairs_d, Encpairs2_d, NBS_d, N_d, Nst);
				groupM2_kernel <<< Nencpairs2_h[0], 16 >>> (Encpairs_d, Encpairs2_d, Nenc_d, NBS_d, N_d, Nst);
				cudaDeviceSynchronize();
				BSBMCall(si, 0);
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

		HCM2_kernel < HCM_Bl, HCM_Bl2, NmaxM, 2 > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, dt_d, Msun_d, index_d, NT, Ct[si], test_d, Nencpairs_d, Nencpairs2_d, Nenc_d, Nst, P.UseForce, Nstart);
		if(si < SIn - 1){
			KickM2_kernel < KM_Bl, KM_Bl2, NmaxM, 2 > <<< (NT + KM_Bl2 - 1) / KM_Bl2, KM_Bl>>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, Nencpairs_d, Encpairs_d, dt_d, Kt[si], index_d, NT, test_d, Nstart);
			if(ForceFlag > 0){
				comM_kernel < HCM_Bl, HCM_Bl2, NmaxM > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, vcom_d, Msun_d, index_d, NBS_d, NT, test_d, 1, Nstart);
				if(P.Usegas == 1) GasAccCall_M(time_d, dt_d, Kt[si]);
				if(P.UseForce > 0) force <<< (NT + 127) / 128, 128  >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[si], time_d, NT, Nst, P.UseForce, Nstart);
				if(P.UseYarkovsky == 1) CallYarkovsky2 <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], NT, Nst, Nstart);
				if(P.UseYarkovsky == 2) CallYarkovsky <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[si], NT, Nst, Nstart);
				if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, dt_d, Kt[si], NT, Nst, Nstart);
				if(P.UsePR == 2) PoyntingRobertsonDrag <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[si], NT, Nst, Nstart);
				comM_kernel < HCM_Bl, HCM_Bl2, NmaxM > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, vcom_d, Msun_d, index_d, NBS_d, NT, test_d, -1, Nstart);
			}
 			cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
		}
	}
	KickM2_kernel < KM_Bl, KM_Bl2, NmaxM, 1 > <<< (NT + KM_Bl2 - 1) / KM_Bl2, KM_Bl>>> (x4_d, v4_d, a_d, rcrit_d, rcritv_d, Nencpairs_d, Encpairs_d, dt_d, Kt[SIn - 1], index_d, NT, test_d, Nstart);
	if(ForceFlag > 0){
		comM_kernel < HCM_Bl, HCM_Bl2, NmaxM > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, vcom_d, Msun_d, index_d, NBS_d, NT, test_d, 1, Nstart);
		if(P.Usegas == 1) GasAccCall_M(time_d, dt_d, Kt[SIn - 1]);
		if(P.UseForce > 0) force <<< (NT + 127) / 128, 128  >>> (x4_d, v4_d, index_d, spin_d, love_d, Msun_d, Spinsun_d, dt_d, Kt[SIn - 1], time_d, NT, Nst, P.UseForce, Nstart);
		if(P.UseYarkovsky == 1) CallYarkovsky2 <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], NT, Nst, Nstart);
		if(P.UseYarkovsky == 2) CallYarkovsky <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, spin_d, index_d, Msun_d, dt_d, Kt[SIn - 1], NT, Nst, Nstart);
		if(P.UsePR == 1) PoyntingRobertsonDrag2 <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, dt_d, Kt[SIn - 1], NT, Nst, Nstart);
		if(P.UsePR == 2) PoyntingRobertsonDrag <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, Msun_d, dt_d, Kt[SIn - 1], NT, Nst, Nstart);
		comM_kernel < HCM_Bl, HCM_Bl2, NmaxM > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, vcom_d, Msun_d, index_d, NBS_d, NT, test_d, -1, Nstart);
	}

	cudaMemcpy(Nencpairs_h, Nencpairs_d, sizeof(int), cudaMemcpyDeviceToHost);
	if(EjectionFlag_m[0] > 0){
		EjectionMCall();
	}
	if(P.ci < 0){
		cudaMemcpy(StopFlag_h, StopFlag_d, sizeof(int), cudaMemcpyDeviceToHost);
		if(StopFlag_h[0] == 1){
			CoordinateOutput(3);

			cudaMemcpy(N_h, N_d, Nst * sizeof(int), cudaMemcpyDeviceToHost);
			stopSimulations();
			StopFlag_h[0] = 0;
			cudaMemcpy(StopFlag_d, StopFlag_h, sizeof(int), cudaMemcpyHostToDevice);
		}
	}

	return 1;
}
__host__ int Data::step_MSimple(){
#if def_TTV != 1
	kick32BMSimple_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, a_d, ab_d, index_d, NT, dt_d, Kt[SIn - 1], time_d, idt_d, ict_d, timeStep, Nst, Nstart);
#else
	kick32BMTTVSimple_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, a_d, ab_d, index_d, NT, dt_d, Kt[SIn - 1], Msun_d, Ntransit_d, Transit_d, time_d, idt_d, ict_d, timeStep, Nst, Nstart);
	cudaDeviceSynchronize();
	if(Ntransit_m[0] > 0){
		if(Ntransit_m[0] >= def_NtransitMax - 1){
printf("more Transits than allowed in def_NtransitMax: %d\n", def_NtransitMax);
			return 0;
		}
		BSTTVStep_kernel < 8, 8 > <<< Ntransit_m[0], 64 >>> (x4_d, v4b_d, Transit_d, N_d, dt_d, Msun_d, index_d, time_d, NBS_d, P.UseForce, P.MinMass, P.UseTestParticles, Nst, TransitTime_d, NtransitsT_d, Ntransit_m[0]);
		Ntransit_m[0] = 0;
	}
#endif
	EjectionFlag2 = 0;
	for(int si = 0; si < SIn; ++si){
		HCM2_kernel < HCM_Bl, HCM_Bl2, NmaxM, 1 > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, dt_d, Msun_d, index_d, NT, Ct[si], test_d, Nencpairs_d, Nencpairs2_d, Nenc_d, Nst, P.UseForce, Nstart);
		fgMSimple_kernel <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, xold_d, vold_d, dt_d, Msun_d, test_d, index_d, NT, FGt[si], si, P.UseForce, Nstart);
		HCM2_kernel < HCM_Bl, HCM_Bl2, NmaxM, 2 > <<< (NT + HCM_Bl2 - 1) / HCM_Bl2, HCM_Bl >>> (x4_d, v4_d, dt_d, Msun_d, index_d, NT, Ct[si], test_d, Nencpairs_d, Nencpairs2_d, Nenc_d, Nst, P.UseForce, Nstart);

		if(si < SIn - 1){
			KickM2Simple_kernel < KM_Bl, KM_Bl2, NmaxM, 2 > <<< (NT + KM_Bl2 - 1) / KM_Bl2, KM_Bl>>> (x4_d, v4_d, v4b_d, a_d, dt_d, Kt[si], index_d, NT, test_d, Nst, Nstart);
		}
	}
	KickM2Simple_kernel < KM_Bl, KM_Bl2, NmaxM, 1 > <<< (NT + KM_Bl2 - 1) / KM_Bl2, KM_Bl>>> (x4_d, v4_d, v4b_d, a_d, dt_d, Kt[SIn - 1], index_d, NT, test_d, Nst, Nstart);

	return 1;
}


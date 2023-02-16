#include "Host2.h"

// **************************************
//This function converts heliocentric coordinates to democratic coordinates.
__global__ void HelioToDemo_kernel(double4 *x4_d, double4 *v4_d, int *NBS_d, double Msun, int Nst, int N){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	if(id < Nst){
		double mtot = 0.0;
		double3 vcom;
		vcom.x = 0.0;
		vcom.y = 0.0;
		vcom.z = 0.0;

		int NBS = NBS_d[id];
		
		for(int i = 0; i < N; ++i){
			double m = x4_d[i + NBS].w;
			if(m > 0.0){
				mtot += m;
				vcom.x += m * v4_d[i + NBS].x;
				vcom.y += m * v4_d[i + NBS].y;
				vcom.z += m * v4_d[i + NBS].z;
			}
		}
		mtot += Msun;
		vcom.x /= mtot;
		vcom.y /= mtot;
		vcom.z /= mtot;

		for(int i = 0; i < N; ++i){
			v4_d[i + NBS].x -= vcom.x;
			v4_d[i + NBS].y -= vcom.y;
			v4_d[i + NBS].z -= vcom.z;
//printf("%d %.20g %.20g\n", i + NBS, x4_d[i + NBS].x, v4_d[i + NBS].x);
		}
	}
}
//This function converts heliocentric coordinates to barycentric coordinates.
//the zeroth body must bes the central star
__global__ void HelioToBary_kernel(double4 *x4_d, double4 *v4_d, int *NBS_d, double Msun, int Nst, int N){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	if(id < Nst){
		double mtot = 0.0;
		double3 xcom;
		xcom.x = 0.0;
		xcom.y = 0.0;
		xcom.z = 0.0;
		double3 vcom;
		vcom.x = 0.0;
		vcom.y = 0.0;
		vcom.z = 0.0;

		int NBS = NBS_d[id];
		
		for(int i = 0; i < N; ++i){
			double m = x4_d[i + NBS].w;
			if(m > 0.0){
				mtot += m;
				xcom.x += m * x4_d[i + NBS].x;
				xcom.y += m * x4_d[i + NBS].y;
				xcom.z += m * x4_d[i + NBS].z;
				vcom.x += m * v4_d[i + NBS].x;
				vcom.y += m * v4_d[i + NBS].y;
				vcom.z += m * v4_d[i + NBS].z;
			}
		}
		xcom.x /= mtot;
		xcom.y /= mtot;
		xcom.z /= mtot;
		vcom.x /= mtot;
		vcom.y /= mtot;
		vcom.z /= mtot;

		for(int i = 0; i < N; ++i){
			x4_d[i + NBS].x -= xcom.x;
			x4_d[i + NBS].y -= xcom.y;
			x4_d[i + NBS].z -= xcom.z;
			v4_d[i + NBS].x -= vcom.x;
			v4_d[i + NBS].y -= vcom.y;
			v4_d[i + NBS].z -= vcom.z;
		}
	}
}

// **********************************************************
// This kernel initializes the TTV probability in a way, 
// that the initial mcmc step is alway accepted.
//
// Author: Simon Grimm
// April 2017
// **********************************************************
__global__ void SetTTVP(double4 *elementsP_d, double *elementsSA_d, int Nst){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	if(id < Nst){
		//elementsP_d[id].x = -1.0e-3;		//initial value for product (TTVstep2)
		elementsP_d[id].x = 1.0e300;		//initial value for sum
		elementsP_d[id].y = 0.0;		//contains later a random number
		elementsP_d[id].z = 1.0e300;		//new p
	}
}

// **********************************************************
// This kernel resets arrays that were changed due to 
// close encounter stops
//
// Author: Simon Grimm
// February 2020
// **********************************************************
__global__ void SetTTVP1(double *n1_d, double *rcrit_d, double *rcritv_d, int *index_d, double n1, int NT, int N0, int Nst){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	if(id < NT){
		rcrit_d[id] = 0.0;
		rcritv_d[id] = 0.0;
		index_d[id] = id / N0 * def_MaxIndex + id % N0;
	}
	if(id < Nst){
		n1_d[id] = n1;
	}
}

// ********************************************************************************************
// This kernel computes the value p = (tObs - tCalc)/sigma for each transit, and sums up p * p for each planet.
// The sum of p * p for planet id is stored in TransitTime_d[id];
//
// Author: Simon Grimm
// April 2017
// *******************************************************************************************
__global__ void TTVstep(double *TransitTime_d, double2 *TransitTimeObs_d, int2 *NtransitsT_d, int *NtransitsTObs_d, int *N_d, double4 *elementsT_d, int NT, int ittv){
	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	if(id < NT){
		int Nt = 0;
		int NtOld = NtransitsT_d[id].y;
		int NtObs = NtransitsTObs_d[id % N_d[0]];
//if(id <= 8) printf("NtObs %d\n", NtObs);
		double pp = 0.0;
		int Epoch = 0;
		int setEpoch = 0;

		for(int EpochObs = 0; EpochObs <= NtObs; ++EpochObs){
			double T = TransitTime_d[id * def_NtransitTimeMax + Epoch + 1];
			double T1 = TransitTime_d[id * def_NtransitTimeMax + Epoch + 2];	//next transit
			double2 Tobs = TransitTimeObs_d[(id % N_d[0]) * def_NtransitTimeMax + EpochObs + 1];
//if(id % N_d[0] == 2) printf("--------- %d %.20g %.20g %.20g | %g %g %g | %d %d %d\n", id, T, T1, Tobs.x, Tobs.x - T, Tobs.x - T1, T - Tobs.x, Epoch, EpochObs, id * def_NtransitTimeMax + Epoch + 1);

			if(setEpoch == 0 && fabs(Tobs.x - T) < fabs(Tobs.x - T1) && T != 0.0 && Tobs.x != 0.0){
				 setEpoch = 1;
//if(id % N_d[0] == 2) printf("set Epoch %d %.20g %.20g %d %d\n", id, T, Tobs.x, Epoch, EpochObs);
			}

			if(setEpoch == 0 && T != 0 && Tobs.x != 0 && fabs(Tobs.x - T) < fabs(Tobs.x - T1)){
//if(id % N_d[0] == 2) printf("********* %d %.20g %.20g %d %d\n", id, T, Tobs.x, Epoch, EpochObs);
				++EpochObs;
				if(EpochObs >= NtObs) break;
				Tobs = TransitTimeObs_d[(id % N_d[0]) * def_NtransitTimeMax + EpochObs + 1];
//if(id % N_d[0] == 2) printf("********+ %d %.20g %.20g %d %d\n", id, T, Tobs.x, Epoch, EpochObs);
			}

			//recheck setEpoch 
			if(setEpoch == 0 && fabs(Tobs.x - T) < fabs(Tobs.x - T1) && T != 0.0 && Tobs.x != 0.0){
				 setEpoch = 1;
//if(id % N_d[0] == 2) printf("set Epoch %d %.20g %.20g %d %d\n", id, T, Tobs.x, Epoch, EpochObs);
			}



			if(setEpoch == 0 && T != 0 && Tobs.x != 0 && fabs(Tobs.x - T) >= fabs(Tobs.x - T1)){
//if(id % N_d[0] == 2) printf("#########  %d %.20g %.20g %d %d\n", id, T, Tobs.x, Epoch, EpochObs);
				++Epoch;
//if(id % N_d[0] == 2) printf("########+  %d %.20g %.20g %d %d\n", id, T, Tobs.x, Epoch, EpochObs);
				--EpochObs;
				continue;
			}

			double p = (T - Tobs.x) / Tobs.y;
			if(T == 0.0 || Tobs.x == 0.0) p = 0.0;
			p = p * p * 0.5;
			pp += p;

			// --------------------------
			//log Student-t distribution
			//double nu = 3.9;
			//double V1 = 0.85;
			//double p = (T - Tobs.x) / Tobs.y;
			//p = 0.5 * (nu + 1.0) * log(1.0 + p * p / (nu * V1));
			//pp += p;
			// --------------------------

			if(T > 0 && Tobs.x > 0) ++ Nt;
			++Epoch;
//if(id % N_d[0] == 2) printf(" p %d NtOld Nt NtObs %d %d %d Epoch EpochObs %d %d %.20g %.20g %g %g\n", id, NtOld, Nt, NtObs, Epoch, EpochObs, T, Tobs.x, Tobs.y, p);
		}
		if(Nt < NtOld){
			pp = 1.0e300; //penalty for missing transits
printf("missing transit %d %d %d\n", id, Nt, NtOld);
		}
		double Tr0 = TransitTime_d[id * def_NtransitTimeMax + 0 + 1];
		double Tr1 = TransitTime_d[id * def_NtransitTimeMax + 0 + Nt];
		elementsT_d[id].y = Tr0; 
		elementsT_d[id].w = (Tr1 - Tr0) / ((double)(Nt - 1)); 
		NtransitsT_d[id].x = Nt;
		TransitTime_d[id * def_NtransitTimeMax] = pp;
if(id < N_d[0]) printf("pp %d %14.8e %d %d | %.20g %.20g\n", id, pp, Nt, NtOld, Tr0, (Tr1 - Tr0) / ((double)(Nt - 1)));
	}
}

//This algorithm refines the period based on linear perturbations
__global__ void TTVstepRefine(double *TransitTime_d, double2 *TransitTimeObs_d, int2 *NtransitsT_d, int *NtransitsTObs_d, int *N_d, double4 *elementsT_d, int NT, int ittv){
	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	if(id < NT){
		double a, b, c, d;		//parameters for golden section search
		double phi = (sqrt(5.0) + 1.0) * 0.5;
		a = -1.0e-3;
		b =  1.0e-3;
		c = b - (b - a) / phi;
		d = a + (b - a) / phi;
		double pc, pd;
		int Nt;
		double pp;
		for(int k = 0; k < 100; ++ k){
			for(int j = 0; j < 2; ++ j){

				Nt = 0;
				int NtObs = NtransitsTObs_d[id % N_d[0]];
				pp = 0.0;
				int Epoch = 0;
				int setEpoch = 0;
				//double P = TransitTimeObs_d[(id % N_d[0]) * def_NtransitTimeMax].y; //period
				for(int EpochObs = 0; EpochObs <= NtObs; ++EpochObs){

					double T = TransitTime_d[id * def_NtransitTimeMax + Epoch + 1];
					double T1 = TransitTime_d[id * def_NtransitTimeMax + Epoch + 2];	//next transit
					if(j == 0){
						T += c * Epoch;
						T1 += c * Epoch;
					}
					if(j == 1){
						T += d * Epoch;
						T1 += d * Epoch;
					}
					double2 Tobs = TransitTimeObs_d[(id % N_d[0]) * def_NtransitTimeMax + EpochObs + 1];

					if(setEpoch == 0 && fabs(Tobs.x - T) < fabs(Tobs.x - T1) && T != 0.0 && Tobs.x != 0.0){
						 setEpoch = 1;
					}

					if(setEpoch == 0 && T != 0 && Tobs.x != 0 && fabs(Tobs.x - T) < fabs(Tobs.x - T1)){
						++EpochObs;
						if(EpochObs >= NtObs) break;
						Tobs = TransitTimeObs_d[(id % N_d[0]) * def_NtransitTimeMax + EpochObs + 1];
					}

					//recheck setEpoch 
					if(setEpoch == 0 && fabs(Tobs.x - T) < fabs(Tobs.x - T1) && T != 0.0 && Tobs.x != 0.0){
						 setEpoch = 1;
					}

					if(setEpoch == 0 && T != 0 && Tobs.x != 0 && fabs(Tobs.x - T) >= fabs(Tobs.x - T1)){
						++Epoch;
						--EpochObs;
						continue;
					}

					double p = (T - Tobs.x) / Tobs.y;
					if(T == 0.0 || Tobs.x == 0.0) p = 0.0;
					p = p * p * 0.5;
					pp += p;
					if(T > 0 && Tobs.x > 0) ++ Nt;
					++Epoch;

				}
				if(j == 0) pc = pp;
				if(j == 1) pd = pp;
			}
//if(id < N_d[0] && k % 10 == 0) printf(" ppR j %d id %d %.20g %.20g %.20g %.20g\n", k, id, c, d, pc, pd);
			if(pc < pd){
				b = d;
			}
			else{
				a = c;
			}	
			c = b - (b - a) / phi;
			d = a + (b - a) / phi;
		}
		double dP = (b + a) * 0.5;

//if(id % N_d[0] == 6){
		elementsT_d[id].z += dP;
printf("ppPP %d %14.8e %.20g %d | %.20g\n", id, pp, elementsT_d[id].z, Nt, dP);
//}
//if(id == 4) elementsT_d[id].z += 1.0e-5;

//printf("PP %d %.20g %g %g %g %g %.20g %.20g %.20g %.20g %.20g\n", id * N0 + ii, T, M, dM, da, da2, Tobs, Pobs, a, a + da, a + da + da2);
	}
}



// ********************************************************************************************
//This kernel computes the value p = (tObs - tCalc)/sigma for each RV data, and sums up p * p.
//The sum of p * p for each chain id is stored in RVP_d[id];
//
// Author: Simon Grimm
// December 2019
// *******************************************************************************************
__global__ void RVstep(double2 *RV_d, double3 *RVObs_d, int2 *NRVT_d, double *RVP_d, int Nst){

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < Nst){
		double pp = 0.0;
		for(int i = 0; i < NRVT_d[idx].x; ++i){
			double T = RV_d[idx * def_NRVMax + i].y;
			double TObs = RVObs_d[idx * def_NRVMax + i].y;
			double sigma = RVObs_d[idx * def_NRVMax + i].z;

			double p = (TObs - T) / sigma;
			if(T == 0.0 || TObs == 0.0) p = 0.0;
			pp += p * p * 0.5;
		}
		RVP_d[idx] = pp;
if(idx < 3) printf("RVP %d %g\n", idx, pp);
	}
}

// ********************************************************************************************
// This kernel performs a parallel summation of the values p*p / 2, it is also parallel for multi simulations.
// elementsP_d[id] contains the current p
//
// Author: Simon Grimm
// February 2020
// *******************************************************************************************
template <int Bl, int Bl2, int Nmax>
__global__ void TTVstep1(int *index_d, double *TransitTime_d, double *RVP_d, double4 *elementsP_d, int2 *NtransitsT_d, double* n1_d, int NT, int N0, int Nst){

	int idy = threadIdx.x;
	int id = blockIdx.x * Bl2 + idy - Nmax;

	__shared__ volatile double p_s[Bl + Nmax / 2];
	__shared__ int st_s[Bl + Nmax / 2];

	if(id < NT && id >= 0){
		if(Nst > 1){
			st_s[idy] = index_d[id] / def_MaxIndex;
		}
		else st_s[idy] = 0;
		p_s[idy] = TransitTime_d[id * def_NtransitTimeMax];
	}
	else{
		st_s[idy] = -idy-1;
		p_s[idy] = 0.0;
	}
	//halo
	if(idy < Nmax / 2){
		//right
		if(id + Bl < NT){
			if(Nst > 1){
				st_s[idy + Bl] = index_d[id + Bl] / def_MaxIndex;
			}
			else st_s[idy + Bl] = 0;
			p_s[idy + Bl] = TransitTime_d[(id + Bl) * def_NtransitTimeMax];
		}
		else{
			st_s[idy + Bl] = -idy-Bl-1;
			p_s[idy + Bl] = 0.0;
		}
	}
//__syncthreads();
//printf("p0 %d %d %.20g %d\n", idy, id, p_s[idy], st_s[idy]);

	volatile int f;
	volatile double p;
	if(Nmax >= 64){
		__syncthreads();
		f = ((st_s[idy] - st_s[idy + 32]) == 0);	//one if sti == stj, zero else
		p = p_s[idy + 32] * f;	

		__syncthreads();
	
		p_s[idy] += p;
	}

	if(Nmax >= 32){
		__syncthreads();
		f = ((st_s[idy] - st_s[idy + 16]) == 0);	//one if sti == stj, zero else
		p = p_s[idy + 16] * f;	

		__syncthreads();
	
		p_s[idy] += p;
	}

	if(Nmax >= 16){
		__syncthreads();
		f = ((st_s[idy] - st_s[idy + 8]) == 0);		//one if sti == stj, zero else
		p = p_s[idy + 8] * f;	

		__syncthreads();
	
		p_s[idy] += p;
	}

	if(Nmax >= 8){
		__syncthreads();
		f = ((st_s[idy] - st_s[idy + 4]) == 0);		//one if sti == stj, zero else
		p = p_s[idy + 4] * f;

		__syncthreads();

		p_s[idy] += p;
	}

	__syncthreads();

	f = ((st_s[idy] - st_s[idy + 2]) == 0);			//one if sti == stj, zero else
	p = p_s[idy + 2] * f;

	__syncthreads();

	p_s[idy] += p;

	__syncthreads();

	f = ((st_s[idy] - st_s[idy + 1]) == 0);			//one if sti == stj, zero else
	p = p_s[idy + 1] * f;

	__syncthreads();

	p_s[idy] += p;

	__syncthreads();


#if def_RV == 1
	p_s[idy] += RVP_d[st_s[idy]];
	__syncthreads();
#endif

	//penalty for close encounters stops
	if(id < NT && id >= 0){
		if(n1_d[st_s[idy]] < 0.0){
			p_s[idy] = 1.0e200 / N0 - 1.0; 
		}
	}
//__syncthreads();
//printf("p3 %d %.20g %d f: %d\n", idy, p_s[idy], st_s[idy], f);

	//sum is complete, now distribute solution
	f = ((st_s[idy] - st_s[idy + 1]) == 0);
	p = (p_s[idy]) * f + (1 - f) * p_s[idy + 1];

	__syncthreads();
	p_s[idy + 1] = p;
	__syncthreads();

	f = ((st_s[idy] - st_s[idy + 2]) == 0);
	p = (p_s[idy]) * f + (1 - f) * p_s[idy + 2];

	__syncthreads();
	p_s[idy + 2] = p;
	__syncthreads();

	if(Nmax >= 8){
		f = ((st_s[idy] - st_s[idy + 4]) == 0);
		p = (p_s[idy]) * f + (1 - f) * p_s[idy + 4];

		__syncthreads();
		p_s[idy + 4] = p;
		__syncthreads();
	}

	if(Nmax >= 16){
		f = ((st_s[idy] - st_s[idy + 8]) == 0);
		p = (p_s[idy]) * f + (1 - f) * p_s[idy + 8];

		__syncthreads();
		p_s[idy + 8] = p;
		__syncthreads();
	}

	if(Nmax >= 32){
		f = ((st_s[idy] - st_s[idy + 16]) == 0);
		p = (p_s[idy]) * f + (1 - f) * p_s[idy + 16];

		__syncthreads();
		p_s[idy + 16] = p;
		__syncthreads();
	}

	if(Nmax >= 64){
		f = ((st_s[idy] - st_s[idy + 32]) == 0);
		p = (p_s[idy]) * f + (1 - f) * p_s[idy + 32];

		__syncthreads();
		p_s[idy + 32] = p;
		__syncthreads();
	}

	if(id < NT && id >= 0 && idy >= Nmax && idy < Bl - Nmax / 2){
		elementsP_d[st_s[idy]].z = p_s[idy];

	}
}


// used for TTV = 2, instead of TTVStep  and TTVStep1
__global__ void TTVstepb(double4 *elementsP_d, double *TTV_d, int2 *NtransitsT_d, int2 *EpochCount_d, int Nst, int N0){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	if(id < Nst){

		elementsP_d[id].z = 0.0;

		for(int i = 0; i < N0; ++i){
			int Nt = EpochCount_d[id * N0 + i].y;
			NtransitsT_d[id * N0 + i].x = Nt;
			int NtOld = NtransitsT_d[id * N0 + i].y;

			double pp = TTV_d[id * N0 + i];
			if(Nt < NtOld){
				pp = 1.0e300; //penalty for missing transits
printf("missing transit %d %d %d\n", id * N0 + i, Nt, NtOld);
			}
if(id == 1) printf("pp %d %14.8e %d %d\n", id * N0 + i, pp, Nt, NtOld);


			elementsP_d[id].z += pp;
		}
	}
}


// ********************************************************************************************
// This kernel performs a parallel summation of the values p*p / 2, it is also parallel for multi simulations.
//
// At the end, this kernel cheks, if the mcmc step is accepted or not.
// elementsP_d.y contains a random number
//
// Author: Simon Grimm
// February 2020
// *******************************************************************************************
__global__ void TTVstep3(int *index_d, double4 *elementsA_d, double4 *elementsB_d, double4 *elementsT_d, double4 *elementsSpin_d, double4 *elementsAOld_d, double4 *elementsBOld_d, double4 *elementsTOld_d, double4 *elementsSpinOld_d, double4 *elementsP_d, double *elementsSA_d, int2 *elementsC_d, int2 *NtransitsT_d, double2 *Msun_d, double *elementsM_d, int NT, int N0, int Nst, int mcmcNE){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	if(id < NT){
		int st = index_d[id] / def_MaxIndex;

		int iT = id / (N0 * Nst / MCMC_NT);			//index of temperature in parallel tempering
		double p, pOld;
		p = elementsP_d[st].z;
		pOld = elementsP_d[st].x;
		double lnrandom = log(elementsP_d[st].y);
		if(lnrandom > 1.0e300) pOld = elementsP_d[st].z; //used in quadratic estimation substep, not up to date
		
		__syncthreads();
// ***************************************************************************************** //
// Here is the acceptance step
		int accept = 0;
		__syncthreads();
		double lnz = 0.0;		//z^N -1 for affine invariant mcmc
		double temp = elementsSA_d[st];

		double lnq = lnz + (-p + pOld) / temp; 
if(id % (N0 * Nst / 3) < N0 /*|| elementsP_d[st].w < 0.0001*/)  printf("p %5d %5d %20.15g %20.15g lnq %12.8g lnrandom %12.8g pp %12.8g lnz %12.8g w %12.8g T %g Nt %d NtOld %d\n", id, st, 2.0 * p, 2.0 * pOld, lnq, lnrandom, (-p + pOld) / temp, lnz, elementsP_d[st].w, temp, NtransitsT_d[id].x, NtransitsT_d[id].y);
		if(lnq > lnrandom){
			if(id % N0 == 0){
				elementsC_d[st + MCMC_NT].x = atomicAdd(&elementsC_d[iT].x, 1);
			}
			accept = 1;
		}
		__syncthreads();
		if(accept == 1){
			elementsAOld_d[id] = elementsA_d[id];
			elementsBOld_d[id] = elementsB_d[id];
			elementsTOld_d[id].x = elementsT_d[id].x;
			elementsTOld_d[id].z = elementsT_d[id].z;
			elementsSpinOld_d[id] = elementsSpin_d[id];
			NtransitsT_d[id].y = NtransitsT_d[id].x; //NtOld = Nt
//			elementsM_d[st] = Msun_d[st].x;

			elementsP_d[st].z = p;	//current p
			elementsP_d[st].x = p;    //accepted p
			//accept
			if(id % N0 == 0) elementsC_d[st].y += N0;
if(id /*% (N0 * Nst / MCMC_NT)*/ < N0) printf("%d %15.10g %15.10g %15.10g %15.10g %15.10g %15.10g %15.10g %15.10g %15.10g %15.10g %15.10g %15.10g %15.10g\n", id, elementsA_d[id].w, elementsB_d[id].w, elementsA_d[id].x, elementsA_d[id].y, elementsA_d[id].z, elementsB_d[id].x, elementsB_d[id].y, elementsB_d[id].z, elementsT_d[id].x, elementsT_d[id].y, 2.0 * p, elementsP_d[st].w, temp);
if(id /*% N0*/ == 0) printf("accept     %d %d %d | %d %g\n", id, iT, st, elementsC_d[st].y / N0, 2.0 * p);
		}
		else{
			elementsP_d[st].z = pOld;	//current p
			elementsP_d[st].x = p;	//not accepted p
//			Msun_d[st].x = elementsM_d[st];
if(id < N0) printf("%d %15.10g %15.10g %15.10g %15.10g %15.10g %15.10g %15.10g %15.10g %15.10g %15.10g %15.10g\n", id, elementsA_d[id].w, elementsB_d[id].w, elementsA_d[id].x, elementsA_d[id].y, elementsA_d[id].z, elementsB_d[id].x, elementsB_d[id].y, elementsB_d[id].z, 2.0 * p, elementsP_d[st].w, temp);
if(id /*% N0*/ == 0) printf("not accept %d %d %d | %d %g\n", id, iT, st, elementsC_d[st].y / N0, 2.0 * p);
		}
// ****************************************************************************************** //

	}
}



__global__ void setNtransits(int2 *NtransitsT_d, int NT){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;
	if(id < NT){

		NtransitsT_d[id].x = 0;

	}
}

//this kernel reduces the sampling temperature
__global__ void SetSA_kernel(double* elementsSA_d, int Nst){

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < Nst){
		//elementsSA_d[id] *= 0.996;
		//elementsSA_d[id] *= 0.998;

	}
}



// *******************************************************
// The following kernes generates random numbers X for the mcmc move
// and computes Z = L X, where L is the Choleski decomposition part 
// of the covariance matrix C

// The kernels overwrites tuning lengths elementsL 
//Date: February 2020
//Author: Simon Grimm
// *******************************************************
__global__ void setCovarianceRandom1(curandState *random_d, elements10 *elementsL_d, int Nst, int N0){
#if USE_RANDOM == 1

	int idy = threadIdx.x;
	int idx = blockIdx.x;
	curandState random;	

	if(idx < Nst){

		if(idy < N0){
			random = random_d[idx * N0 + idy];
			//generate random number vector X
			//P
			double rd = curand_normal(&random);
			elementsL_d[idx * N0 + idy].P = rd;
 #if MCMC_NCOV > 1
			//T
			rd = curand_normal(&random);
			elementsL_d[idx * N0 + idy].T = rd;
 #endif
 #if MCMC_NCOV > 2
			//m
			rd = curand_normal(&random);
			elementsL_d[idx * N0 + idy].m = rd;
 #endif
 #if MCMC_NCOV > 3
			//e
			rd = curand_normal(&random);
			elementsL_d[idx * N0 + idy].e = rd;
 #endif
 #if MCMC_NCOV > 4
			//w
			rd = curand_normal(&random);
			elementsL_d[idx * N0 + idy].w = rd;
 #endif
			random_d[idx * N0 + idy] = random;

		}
	}
#endif
}

__global__ void setCovarianceRandom(double *elementsCOV_d, elements10 *elementsL_d, int Nst, int N0){
	int idy = threadIdx.x;
	int idx = blockIdx.x;

	if(idx < Nst){

		if(idy < N0){
			double zP = 0.0;
			double zT = 0.0;
			double zm = 0.0;
			double ze = 0.0;
			double zw = 0.0;
			__syncthreads();

			//Z = LX
			for(int q = 0; q < MCMC_NCOV; ++q){
				for(int j = 0; j < N0; ++j){
					int ii = idx * N0 * MCMC_NCOV + idy * MCMC_NCOV + q;
					int jj = j * MCMC_NCOV;

					for(int k = 0; k < MCMC_NCOV; ++k){
						double rd;
						if(k == 0) rd = elementsL_d[idx * N0 + j].P; //P
						if(k == 1) rd = elementsL_d[idx * N0 + j].T; //T
						if(k == 3) rd = elementsL_d[idx * N0 + j].m; //m
						if(k == 4) rd = elementsL_d[idx * N0 + j].e; //e
						if(k == 5) rd = elementsL_d[idx * N0 + j].w; //w


						if(q == 0) zP += rd * elementsCOV_d[ii * N0 * MCMC_NCOV + jj + k];
						if(q == 1) zT += rd * elementsCOV_d[ii * N0 * MCMC_NCOV + jj + k];
						if(q == 2) zm += rd * elementsCOV_d[ii * N0 * MCMC_NCOV + jj + k];
						if(q == 2) ze += rd * elementsCOV_d[ii * N0 * MCMC_NCOV + jj + k];
						if(q == 2) zw += rd * elementsCOV_d[ii * N0 * MCMC_NCOV + jj + k];

//if(idx == 0) printf("P %d %d %d %d %g %g %g\n", idx, idy, ii, jj + k, zP, rd, elementsCOV_d[ii * N0 * MCMC_NCOV + jj + k]);
					}
				}
			}
			__syncthreads();
				
			if(MCMC_NCOV > 0) elementsL_d[idx * N0 + idy].P = zP;
			if(MCMC_NCOV > 1) elementsL_d[idx * N0 + idy].T = zT;
			if(MCMC_NCOV > 2) elementsL_d[idx * N0 + idy].m = zm;
			if(MCMC_NCOV > 3) elementsL_d[idx * N0 + idy].e = ze;
			if(MCMC_NCOV > 4) elementsL_d[idx * N0 + idy].w = zw;
//printf("COV %d %d %g %g %g\n", idx * N0 + idy, idy, zP, elementsCOV_d[idy * N0 + 0], elementsCOV_d[(idx * N0 + idy) * N0 * MCMC_NCOV + 1]);
		}
	}
}


__global__ void setHyperParameters(elements8 *elementsGh_d, const int NT, const int N0, const int Nst){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	//these elements scale the step size eta in the optimizers
	if(id < NT){

		//P
		elementsGh_d[id].P = 1.0e-5;
		//T
		elementsGh_d[id].T = 1.0e-4;
		//m
		elementsGh_d[id].m = 1.0e-5;
		//e
		elementsGh_d[id].e = 1.0e-3;
		//w
		elementsGh_d[id].w = 1.0e-1;
		
	}
}

__global__ void Normalize(double4 *elementsAOld_d, double4 *elementsBOld_d, double4 *elementsTOld_d, elements10 *elementsL_d, elements8 *elementsMean_d, elements8 *elementsVar_d, const int NT, int ittv){
	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;


	ittv += 1;
	if(id < NT){

		double dP = (elementsTOld_d[35].z - elementsMean_d[35].P);
		double dT = (elementsTOld_d[35].x - elementsMean_d[35].T);
		double dm = (elementsAOld_d[35].w - elementsMean_d[35].m);
		double de = (elementsAOld_d[35].y - elementsMean_d[35].e);
		double dw = (elementsBOld_d[35].y - elementsMean_d[35].w);

		elementsMean_d[id].P += 1.0 / (ittv + 1.0) * dP; 
		elementsMean_d[id].T += 1.0 / (ittv + 1.0) * dT; 
		elementsMean_d[id].m += 1.0 / (ittv + 1.0) * dm; 
		elementsMean_d[id].e += 1.0 / (ittv + 1.0) * de; 
		elementsMean_d[id].w += 1.0 / (ittv + 1.0) * dw; 

		elementsVar_d[id].P = (ittv - 1.0) / (ittv) * elementsVar_d[id].P + 1.0 / ittv * dP * dP + 1.0e-4; 
		elementsVar_d[id].T = (ittv - 1.0) / (ittv) * elementsVar_d[id].T + 1.0 / ittv * dT * dT + 1.0e-4; 
		elementsVar_d[id].m = (ittv - 1.0) / (ittv) * elementsVar_d[id].m + 1.0 / ittv * dm * dm + 1.0e-4; 
		elementsVar_d[id].e = (ittv - 1.0) / (ittv) * elementsVar_d[id].e + 1.0 / ittv * de * de + 1.0e-4;  
		elementsVar_d[id].w = (ittv - 1.0) / (ittv) * elementsVar_d[id].w + 1.0 / ittv * dw * dw + 1.0e-4; 

//printf("mean var P: %d %g %g\n", id, elementsMean_d[id].P, elementsVar_d[id].P);
//printf("mean var T: %d %g %g\n", id, elementsMean_d[id].T, elementsVar_d[id].T);
//printf("mean var m: %d %g %g\n", id, elementsMean_d[id].m, elementsVar_d[id].m);
//printf("mean var e: %d %g %g\n", id, elementsMean_d[id].e, elementsVar_d[id].e);
//printf("mean var w: %d %g %g\n", id, elementsMean_d[id].w, elementsVar_d[id].w);

		//xs = (x - mu) / sigma
		elementsTOld_d[id].z = (elementsTOld_d[id].z - elementsMean_d[id].P) / (elementsVar_d[id].P); 
		elementsTOld_d[id].x = (elementsTOld_d[id].x - elementsMean_d[id].T) / (elementsVar_d[id].T); 
		elementsAOld_d[id].w = (elementsAOld_d[id].w - elementsMean_d[id].m) / (elementsVar_d[id].m); 
		elementsAOld_d[id].y = (elementsAOld_d[id].y - elementsMean_d[id].e) / (elementsVar_d[id].e); 
		elementsBOld_d[id].y = (elementsBOld_d[id].y - elementsMean_d[id].w) / (elementsVar_d[id].w); 

		elementsL_d[id].P = (elementsL_d[id].P - elementsMean_d[id].P) / (elementsVar_d[id].P); 
		elementsL_d[id].T = (elementsL_d[id].T - elementsMean_d[id].T) / (elementsVar_d[id].T); 
		elementsL_d[id].m = (elementsL_d[id].m - elementsMean_d[id].m) / (elementsVar_d[id].m); 
		elementsL_d[id].e = (elementsL_d[id].e - elementsMean_d[id].e) / (elementsVar_d[id].e); 
		elementsL_d[id].w = (elementsL_d[id].w - elementsMean_d[id].w) / (elementsVar_d[id].w); 

	}
}

__global__ void deNormalize(double4 *elementsAOld_d, double4 *elementsBOld_d, double4 *elementsTOld_d, elements10 *elementsL_d, elements8 *elementsMean_d, elements8 *elementsVar_d, const int NT, int ittv){
	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;


	ittv += 1;
	if(id < NT){
		//x = xs * sigma + mu)

		elementsTOld_d[id].z = (elementsTOld_d[id].z * elementsVar_d[id].P) + elementsMean_d[id].P; 
		elementsTOld_d[id].x = (elementsTOld_d[id].x * elementsVar_d[id].T) + elementsMean_d[id].T; 
		elementsAOld_d[id].w = (elementsAOld_d[id].w * elementsVar_d[id].m) + elementsMean_d[id].m; 
		elementsAOld_d[id].y = (elementsAOld_d[id].y * elementsVar_d[id].e) + elementsMean_d[id].e; 
		elementsBOld_d[id].y = (elementsBOld_d[id].y * elementsVar_d[id].w) + elementsMean_d[id].w; 

		elementsL_d[id].P = (elementsL_d[id].P * elementsVar_d[id].P) + elementsMean_d[id].P; 
		elementsL_d[id].T = (elementsL_d[id].T * elementsVar_d[id].T) + elementsMean_d[id].T; 
		elementsL_d[id].m = (elementsL_d[id].m * elementsVar_d[id].m) + elementsMean_d[id].m; 
		elementsL_d[id].e = (elementsL_d[id].e * elementsVar_d[id].e) + elementsMean_d[id].e; 
		elementsL_d[id].w = (elementsL_d[id].w * elementsVar_d[id].w) + elementsMean_d[id].w; 
	}

}

//For SVGD
__global__ void Variance(double4 *elementsAOld_d, double4 *elementsBOld_d, double4 *elementsTOld_d, elements8 *elementsMean_d, elements8 *elementsVar_d, const int N0, const int Ne, const int Nst){
	int idy = threadIdx.x;
	int ii = blockIdx.x * blockDim.x + idy;


	//calculate mean and variation over all Stein points
	if(ii < N0){
		double P = 0.0;	
		double T = 0.0;	
		double m = 0.0;	
		double e = 0.0;	
		double w = 0.0;	

		double sP = 0.0;	
		double sT = 0.0;	
		double sm = 0.0;	
		double se = 0.0;	
		double sw = 0.0;	
		for(int jjd = Ne * N0; jjd < Nst; jjd += Ne * N0 + 1){
			P += elementsTOld_d[jjd * N0 + ii].z;
			T += elementsTOld_d[jjd * N0 + ii].x;
			m += elementsAOld_d[jjd * N0 + ii].w;
			e += elementsAOld_d[jjd * N0 + ii].y;
			w += elementsBOld_d[jjd * N0 + ii].y;
		}
		double n = Nst / (Ne * N0 + 1);
		P /= n;
		T /= n;
		m /= n;
		e /= n;
		w /= n;

		for(int jjd = Ne * N0; jjd < Nst; jjd += Ne * N0 + 1){
			double dP = elementsTOld_d[jjd * N0 + ii].z - P;
			double dT = elementsTOld_d[jjd * N0 + ii].x - T;
			double dm = elementsAOld_d[jjd * N0 + ii].w - m;
			double de = elementsAOld_d[jjd * N0 + ii].y - e;
			double dw = elementsBOld_d[jjd * N0 + ii].y - w;

			sP += dP * dP;
			sT += dT * dT;
			sm += dm * dm;
			se += de * de;
			sw += dw * dw;
		}

		sP = sqrt(sP);
		sT = sqrt(sT);
		sm = sqrt(sm);
		se = sqrt(se);
		sw = sqrt(sw);

printf("id P %d %.20g %g\n", ii, P, sP);
printf("id T %d %.20g %g\n", ii, T, sT);
printf("id m %d %.20g %g\n", ii, m, sm);
printf("id e %d %.20g %g\n", ii, e, se);
printf("id w %d %.20g %g\n", ii, w, sw);



		elementsMean_d[ii].P = P;
		elementsMean_d[ii].T = T;
		elementsMean_d[ii].m = m;
		elementsMean_d[ii].e = e;
		elementsMean_d[ii].w = w;

		elementsVar_d[ii].P = sP;
		elementsVar_d[ii].T = sT;
		elementsVar_d[ii].m = sm;
		elementsVar_d[ii].e = se;
		elementsVar_d[ii].w = sw;


	}

}

__global__ void alpha(double4 *elementsAOld_d, double4 *elementsBOld_d, double4 *elementsTOld_d, elements10 *elementsL_d, elementsS *elementsStep_d, elementsH *elementsHist_d, double4 *elementsP_d, const int N0, const int Ne, const int Nst, const int ittv){
	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	if(id < 1){

		double alpha = elementsStep_d[35 * N0].alpha;

		double c1 = 1.0e-4;
		double c2 = 0.9;	//Wolfe line search parameters


		double dx, gx;

		double t1 = 0.0;
		double t2 = 0.0;

		int accept = 0;

		if(ittv == 1){
			alpha = 2.0e-18;
			for(int j = 0; j < Nst; ++j){
				for(int ii = 0; ii < N0; ++ii){
					elementsStep_d[j * N0 + ii].alpha = alpha;
				}
			}
		}
		else{
			for(int j = 0; j < Nst - 1; ++j){
				int jjj = j % Ne;
				int iii = j / Ne;
				if(jjj == 0){
					t1 += elementsStep_d[35 * N0 + iii].pP * elementsStep_d[35 * N0 + iii].gP;
					dx = elementsL_d[j * N0 + iii].P;
					gx = -2.0 * (elementsP_d[35].z - elementsP_d[j].z) / dx;
					t2 += elementsStep_d[35 * N0 + iii].pP * gx;
printf("tP %d %d %g %g %g %g | %g %g\n", j, iii, dx, gx, elementsStep_d[35 * N0 + iii].gP, elementsStep_d[35 * N0 + iii].pP, t1, t2);
				}
				if(jjj == 1){
					t1 += elementsStep_d[35 * N0 + iii].pT * elementsStep_d[35 * N0 + iii].gT;
					dx = elementsL_d[j * N0 + iii].T;
					gx = -2.0 * (elementsP_d[35].z - elementsP_d[j].z) / dx;
					t2 += elementsStep_d[35 * N0 + iii].pT * gx;
printf("tT %d %d %g %g %g %g | %g %g\n", j, iii, dx, gx, elementsStep_d[35 * N0 + iii].gT, elementsStep_d[35 * N0 + iii].pT, t1, t2);
				}
				if(jjj == 2){
					t1 += elementsStep_d[35 * N0 + iii].pm * elementsStep_d[35 * N0 + iii].gm;
					dx = elementsL_d[j * N0 + iii].m;
					gx = -2.0 * (elementsP_d[35].z - elementsP_d[j].z) / dx;
					t2 += elementsStep_d[35 * N0 + iii].pm * gx;
printf("tm %d %d %g %g %g %g | %g %g\n", j, iii, dx, gx, elementsStep_d[35 * N0 + iii].gm, elementsStep_d[35 * N0 + iii].pm, t1, t2);
				}
				if(jjj == 3){
					t1 += elementsStep_d[35 * N0 + iii].pe * elementsStep_d[35 * N0 + iii].ge;
					dx = elementsL_d[j * N0 + iii].e;
					gx = -2.0 * (elementsP_d[35].z - elementsP_d[j].z) / dx;
					t2 += elementsStep_d[35 * N0 + iii].pe * gx;
printf("te %d %d %g %g %g %g | %g %g\n", j, iii, dx, gx, elementsStep_d[35 * N0 + iii].ge, elementsStep_d[35 * N0 + iii].pe, t1, t2);
				}
				if(jjj == 4){
					t1 += elementsStep_d[35 * N0 + iii].pw * elementsStep_d[35 * N0 + iii].gw;
					dx = elementsL_d[j * N0 + iii].w;
					gx = -2.0 * (elementsP_d[35].z - elementsP_d[j].z) / dx;
					t2 += elementsStep_d[35 * N0 + iii].pw * gx;
printf("tw %d %d %g %g %g %g | %g %g\n", j, iii, dx, gx, elementsStep_d[35 * N0 + iii].gw, elementsStep_d[35 * N0 + iii].pw, t1, t2);
				}
			}
			double f1 = 2.0 * elementsP_d[35].z;
			double f0 = elementsStep_d[35 * N0].f0;

			if(f1 > f0 + c1 * alpha * t1){
				for(int j = 0; j < Nst; ++j){
					elementsStep_d[j * N0].alpha = alpha * 0.5;
				}
printf("reduce alpha %d %g %.20g %.20g\n", id, elementsStep_d[0 * N0].alpha, f1, f0 + c1 * alpha * t1);
			}	
			else{
				// --------------------------------------
//				if(t2 < c2 * t1){
//					for(int j = 0; j < Nst; ++j){
//						elementsStep_d[j * N0].alpha = alpha * 2.1;
//					}
//printf("increase alpha %d %g %.20g %.20g %.20g %.20g\n", id, elementsStep_d[0 * N0].alpha, f1, f0 + c1 * alpha * t1, t2, c2 * t1);
//				}
//				else{
printf("accept alpha %d %g %.20g %.20g %.20g %.20g | %d\n", id, elementsStep_d[0 * N0].alpha, f1, f0 + c1 * alpha * t1, t2, c2 * t1, elementsStep_d[0 * N0].count + 1);
					for(int j = 0; j < Nst; ++j){
						elementsStep_d[j * N0].alpha = 1.0;
						accept = 1;
						++elementsStep_d[j * N0].count; 
					}
//				}

			}

			if(elementsStep_d[0 * N0].alpha < 1.0e-6 && elementsStep_d[0 * N0].count > 0){
printf("accept alpha B %d %g %.20g %.20g %.20g %.20g | %d\n", id, elementsStep_d[0 * N0].alpha, f1, f0 + c1 * alpha * t1, t2, c2 * t1, elementsStep_d[0 * N0].count + 1);
				for(int j = 0; j < Nst; ++j){
					elementsStep_d[j * N0].alpha = 1.0;
					accept = 1;
					++elementsStep_d[j * N0].count; 
				}
			}

		}

		int t = elementsStep_d[0 * N0].count - 1;
		int tt = t % MCMC_NH;
		int m = MCMC_NH;

		if(ittv == 1 || accept == 1){


			for(int j = 0; j < Nst - 1; ++j){
				int jj = j % Ne;
				int ii = j / Ne;
				if(jj == 0){
printf("store old step %d %d %d %d\n", j, ii, t, tt);
					if(accept == 1){
						elementsHist_d[tt * N0 + ii].sP = elementsTOld_d[35 * N0 + ii].z - elementsStep_d[35 * N0 + ii].P0;
						elementsHist_d[tt * N0 + ii].sT = elementsTOld_d[35 * N0 + ii].x - elementsStep_d[35 * N0 + ii].T0;
						elementsHist_d[tt * N0 + ii].sm = elementsAOld_d[35 * N0 + ii].w - elementsStep_d[35 * N0 + ii].m0;
						elementsHist_d[tt * N0 + ii].se = elementsAOld_d[35 * N0 + ii].y - elementsStep_d[35 * N0 + ii].e0;
						elementsHist_d[tt * N0 + ii].sw = elementsBOld_d[35 * N0 + ii].y - elementsStep_d[35 * N0 + ii].w0;
					}
	
					for(int jjj = 0; jjj < Nst; ++jjj){
						elementsStep_d[jjj * N0 + ii].f0 = 2.0 * elementsP_d[35].z;
						elementsStep_d[jjj * N0 + ii].P0 = elementsTOld_d[35 * N0 + ii].z;
						elementsStep_d[jjj * N0 + ii].T0 = elementsTOld_d[35 * N0 + ii].x;
						elementsStep_d[jjj * N0 + ii].m0 = elementsAOld_d[35 * N0 + ii].w;
						elementsStep_d[jjj * N0 + ii].e0 = elementsAOld_d[35 * N0 + ii].y;
						elementsStep_d[jjj * N0 + ii].w0 = elementsBOld_d[35 * N0 + ii].y;
					}
				}

				if(jj == 0){
					//P
					dx = elementsL_d[j * N0 + ii].P;
					gx = -2.0 * (elementsP_d[35].z - elementsP_d[j].z) / dx;
printf("gP %d %d %g %.20g\n", j, ii, dx, gx);
					if(accept == 1) elementsHist_d[tt * N0 + ii].yP = gx - elementsStep_d[35 * N0 + ii].gP;
					for(int jjj = 0; jjj < Nst; ++jjj){
						elementsStep_d[jjj * N0 + ii].gP = gx;
						if(ittv == 1) elementsStep_d[jjj * N0 + ii].pP = -gx;
					}
				}
				if(jj == 1){
					//T
					dx = elementsL_d[j * N0 + ii].T;
					gx = -2.0 * (elementsP_d[35].z - elementsP_d[j].z) / dx;
printf("gT %d %d %g %.20g\n", j, ii, dx, gx);
					if(accept == 1) elementsHist_d[tt * N0 + ii].yT = gx - elementsStep_d[35 * N0 + ii].gT;
					for(int jjj = 0; jjj < Nst; ++jjj){
						elementsStep_d[jjj * N0 + ii].gT = gx;
						if(ittv == 1) elementsStep_d[jjj * N0 + ii].pT = -gx;
					}
				}
				if(jj == 2){
					//m
					dx = elementsL_d[j * N0 + ii].m;
					gx = -2.0 * (elementsP_d[35].z - elementsP_d[j].z) / dx;
printf("gm %d %d %g %.20g\n", j, ii, dx, gx);
					if(accept == 1) elementsHist_d[tt * N0 + ii].ym = gx - elementsStep_d[35 * N0 + ii].gm;
					for(int jjj = 0; jjj < Nst; ++jjj){
						elementsStep_d[jjj * N0 + ii].gm = gx;
						if(ittv == 1) elementsStep_d[jjj * N0 + ii].pm = -gx;
					}
				}
				if(jj == 3){
					//e
					dx = elementsL_d[j * N0 + ii].e;
					gx = -2.0 * (elementsP_d[35].z - elementsP_d[j].z) / dx;
printf("ge %d %d %g %.20g\n", j, ii, dx, gx);
					if(accept == 1) elementsHist_d[tt * N0 + ii].ye = gx - elementsStep_d[35 * N0 + ii].ge;
					for(int jjj = 0; jjj < Nst; ++jjj){
						elementsStep_d[jjj * N0 + ii].ge = gx;
						if(ittv == 1) elementsStep_d[jjj * N0 + ii].pe = -gx;
					}
				}
				if(jj == 4){
					//w
					dx = elementsL_d[j * N0 + ii].w;
					gx = -2.0 * (elementsP_d[35].z - elementsP_d[j].z) / dx;
printf("gw %d %d %g %.20g\n", j, ii, dx, gx);
					if(accept == 1) elementsHist_d[tt * N0 + ii].yw = gx - elementsStep_d[35 * N0 + ii].gw;
					for(int jjj = 0; jjj < Nst; ++jjj){
						elementsStep_d[jjj * N0 + ii].gw = gx;
						if(ittv == 1) elementsStep_d[jjj * N0 + ii].pw = -gx;

					}
				}
			}
		}

		if(accept == 1){
			double alpha[MCMC_NH];
			double irho[MCMC_NH];

			for(int i = 0; i < N0; ++i){
				elementsStep_d[35 * N0 + i].pP = -elementsStep_d[35 * N0 + i].gP;
				elementsStep_d[35 * N0 + i].pT = -elementsStep_d[35 * N0 + i].gT;
				elementsStep_d[35 * N0 + i].pm = -elementsStep_d[35 * N0 + i].gm;
				elementsStep_d[35 * N0 + i].pe = -elementsStep_d[35 * N0 + i].ge;
				elementsStep_d[35 * N0 + i].pw = -elementsStep_d[35 * N0 + i].gw;
	
			}

			for(int k = t; k >= max(t - m + 1, 0); --k){
				int kk = k % m;
				irho[kk] = 0.0;
				alpha[kk] = 0.0;
				for(int i = 0; i < N0; ++i){
//printf("SYP %d %d %g %g\n", k, i, elementsHist_d[kk * N0 + i].yP, elementsHist_d[kk * N0 + i].sP);
//printf("SYT %d %d %g %g\n", k, i, elementsHist_d[kk * N0 + i].yT, elementsHist_d[kk * N0 + i].sT);
//printf("SYm %d %d %g %g\n", k, i, elementsHist_d[kk * N0 + i].ym, elementsHist_d[kk * N0 + i].sm);
//printf("SYe %d %d %g %g\n", k, i, elementsHist_d[kk * N0 + i].ye, elementsHist_d[kk * N0 + i].se);
//printf("SYw %d %d %g %g\n", k, i, elementsHist_d[kk * N0 + i].yw, elementsHist_d[kk * N0 + i].sw);
					irho[kk] += elementsHist_d[kk * N0 + i].yP * elementsHist_d[kk * N0 + i].sP; 
					irho[kk] += elementsHist_d[kk * N0 + i].yT * elementsHist_d[kk * N0 + i].sT; 
					irho[kk] += elementsHist_d[kk * N0 + i].ym * elementsHist_d[kk * N0 + i].sm; 
					irho[kk] += elementsHist_d[kk * N0 + i].ye * elementsHist_d[kk * N0 + i].se; 
					irho[kk] += elementsHist_d[kk * N0 + i].yw * elementsHist_d[kk * N0 + i].sw; 
				}
				for(int i = 0; i < N0; ++i){
					alpha[kk] += elementsHist_d[kk * N0 + i].sP * elementsStep_d[35 * N0 + i].pP;	
					alpha[kk] += elementsHist_d[kk * N0 + i].sT * elementsStep_d[35 * N0 + i].pT;	
					alpha[kk] += elementsHist_d[kk * N0 + i].sm * elementsStep_d[35 * N0 + i].pm;	
					alpha[kk] += elementsHist_d[kk * N0 + i].se * elementsStep_d[35 * N0 + i].pe;	
					alpha[kk] += elementsHist_d[kk * N0 + i].sw * elementsStep_d[35 * N0 + i].pw;	
				}
				alpha[kk] /= irho[kk];
//printf("alpha rho %d %d %d %d %g %g\n", id, t, k, kk, alpha[kk], irho[kk]);
				for(int i = 0; i < N0; ++i){
					elementsStep_d[35 * N0 + i].pP -= alpha[kk] * elementsHist_d[kk * N0 + i].yP;
					elementsStep_d[35 * N0 + i].pT -= alpha[kk] * elementsHist_d[kk * N0 + i].yT;
					elementsStep_d[35 * N0 + i].pm -= alpha[kk] * elementsHist_d[kk * N0 + i].ym;
					elementsStep_d[35 * N0 + i].pe -= alpha[kk] * elementsHist_d[kk * N0 + i].ye;
					elementsStep_d[35 * N0 + i].pw -= alpha[kk] * elementsHist_d[kk * N0 + i].yw;
				}
			}
			double gammaSY = 0.0;
			double gammaYY = 0.0;
			for(int i = 0; i < N0; ++i){
				gammaSY += elementsHist_d[tt * N0 + i].sP * elementsHist_d[tt * N0 + i].yP;
				gammaYY += elementsHist_d[tt * N0 + i].yP * elementsHist_d[tt * N0 + i].yP;
				gammaSY += elementsHist_d[tt * N0 + i].sT * elementsHist_d[tt * N0 + i].yT;
				gammaYY += elementsHist_d[tt * N0 + i].yT * elementsHist_d[tt * N0 + i].yT;
				gammaSY += elementsHist_d[tt * N0 + i].sm * elementsHist_d[tt * N0 + i].ym;
				gammaYY += elementsHist_d[tt * N0 + i].ym * elementsHist_d[tt * N0 + i].ym;
				gammaSY += elementsHist_d[tt * N0 + i].se * elementsHist_d[tt * N0 + i].ye;
				gammaYY += elementsHist_d[tt * N0 + i].ye * elementsHist_d[tt * N0 + i].ye;
				gammaSY += elementsHist_d[tt * N0 + i].sw * elementsHist_d[tt * N0 + i].yw;
				gammaYY += elementsHist_d[tt * N0 + i].yw * elementsHist_d[tt * N0 + i].yw;
			}
			double gamma = gammaSY / gammaYY;
printf("gamma %g\n", gamma);
			for(int i = 0; i < N0; ++i){
				elementsStep_d[35 * N0 + i].pP *= gamma;
				elementsStep_d[35 * N0 + i].pT *= gamma;
				elementsStep_d[35 * N0 + i].pm *= gamma;
				elementsStep_d[35 * N0 + i].pe *= gamma;
				elementsStep_d[35 * N0 + i].pw *= gamma;
			}

			for(int k = max(t - m + 1, 0); k <= t; ++k){
				int kk = k % m;
				double beta = 0.0;
				for(int i = 0; i < N0; ++i){
					beta += elementsHist_d[kk * N0 + i].yP * elementsStep_d[35 * N0 + i].pP;	
					beta += elementsHist_d[kk * N0 + i].yT * elementsStep_d[35 * N0 + i].pT;	
					beta += elementsHist_d[kk * N0 + i].ym * elementsStep_d[35 * N0 + i].pm;	
					beta += elementsHist_d[kk * N0 + i].ye * elementsStep_d[35 * N0 + i].pe;	
					beta += elementsHist_d[kk * N0 + i].yw * elementsStep_d[35 * N0 + i].pw;
				}
				beta /= irho[kk];
				for(int i = 0; i < N0; ++i){
					elementsStep_d[35 * N0 + i].pP += elementsHist_d[kk * N0 + i].sP * (alpha[kk] - beta);
					elementsStep_d[35 * N0 + i].pT += elementsHist_d[kk * N0 + i].sT * (alpha[kk] - beta);
					elementsStep_d[35 * N0 + i].pm += elementsHist_d[kk * N0 + i].sm * (alpha[kk] - beta);
					elementsStep_d[35 * N0 + i].pe += elementsHist_d[kk * N0 + i].se * (alpha[kk] - beta);
					elementsStep_d[35 * N0 + i].pw += elementsHist_d[kk * N0 + i].sw * (alpha[kk] - beta);
				}
			}
for(int i = 0; i < N0; ++i){
	printf("pP %d %d %.20g\n", id, i, elementsStep_d[35 * N0 + i].pP);
	printf("pT %d %d %.20g\n", id, i, elementsStep_d[35 * N0 + i].pT);
	printf("pm %d %d %.20g\n", id, i, elementsStep_d[35 * N0 + i].pm);
	printf("pe %d %d %.20g\n", id, i, elementsStep_d[35 * N0 + i].pe);
	printf("pw %d %d %.20g\n", id, i, elementsStep_d[35 * N0 + i].pw);
}

			for(int j = 0; j < Nst; ++j){
				for(int i = 0; i < N0; ++i){
					elementsStep_d[j * N0 + i].pP = elementsStep_d[35 * N0 + i].pP;
					elementsStep_d[j * N0 + i].pT = elementsStep_d[35 * N0 + i].pT;
					elementsStep_d[j * N0 + i].pm = elementsStep_d[35 * N0 + i].pm;
					elementsStep_d[j * N0 + i].pe = elementsStep_d[35 * N0 + i].pe;
					elementsStep_d[j * N0 + i].pw = elementsStep_d[35 * N0 + i].pw;
				}
			}
		}

	}
}




//lbfgs
__global__ void gradstep(double4 *elementsAOld_d, double4 *elementsBOld_d, double4 *elementsTOld_d, elements10 *elementsL_d, elementsS *elementsStep_d, double4 *elementsP_d, const int N0, const int Ne, const int Nst, const int ittv){
	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	if(id < Nst - 1){

		double dx;
		int jj = id % Ne;
		int ii = id / Ne;


		//if(ittv == 1){
			double alpha = elementsStep_d[id * N0].alpha;
if(id == 0) printf("alpha %g\n", alpha);

			if(jj == 0){
				//P
				dx = elementsL_d[id * N0 + ii].P;
				for(int j = 0; j < Nst; ++j){
					elementsTOld_d[j * N0 + ii].z = elementsStep_d[j * N0 + ii].P0 + alpha * elementsStep_d[35 * N0 + ii].pP;
				}
				elementsTOld_d[id * N0 + ii].z += dx;
			}
			if(jj == 1){
				//T
				dx = elementsL_d[id * N0 + ii].T;
				for(int j = 0; j < Nst; ++j){
					elementsTOld_d[j * N0 + ii].x = elementsStep_d[j * N0 + ii].T0 + alpha * elementsStep_d[35 * N0 + ii].pT;
				}
				elementsTOld_d[id * N0 + ii].x += dx;
			}
			if(jj == 2){
				//m
				dx = elementsL_d[id * N0 + ii].m;
				for(int j = 0; j < Nst; ++j){
					elementsAOld_d[j * N0 + ii].w = elementsStep_d[j * N0 + ii].m0 + alpha * elementsStep_d[35 * N0 + ii].pm;
				}
				elementsAOld_d[id * N0 + ii].w += dx;
			}
			if(jj == 3){
				//e
				dx = elementsL_d[id * N0 + ii].e;
				for(int j = 0; j < Nst; ++j){
					elementsAOld_d[j * N0 + ii].y = elementsStep_d[j * N0 + ii].e0 + alpha * elementsStep_d[35 * N0 + ii].pe;
				}
				elementsAOld_d[id * N0 + ii].y += dx;
			}
			if(jj == 4){
				//w
				dx = elementsL_d[id * N0 + ii].w;
				for(int j = 0; j < Nst; ++j){
					elementsBOld_d[j * N0 + ii].y = elementsStep_d[j * N0 + ii].w0 + alpha * elementsStep_d[35 * N0 + ii].pw;
				}
				elementsBOld_d[id * N0 + ii].y += dx;
			}
		//}
		
	}
}

//RMSprop with hyperparmaters optimization
__global__ void rmsprop2(double4 *elementsAOld_d, double4 *elementsBOld_d, double4 *elementsTOld_d, elements10 *elementsL_d, elements8 *elementsG_d, elements8 *elementsGh_d, elements8 *elementsD_d, double4 *elementsP_d, const int N0, const int Ne, const int Nst){
	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	int iid = id / (Ne * N0 + 1);	//simulation index, consisting with all the (35) gradient points, plus one point without gradient (total 36)
	int jjd = id % (Ne * N0 + 1);	// map to 0 - 35


	if(id < Nst && jjd < Ne * N0){
		double dx, dx1, gx, Gx;

		double eta = 0.01;
		double eps = 1.0e-6;
		
		double beta = 0.9;

		int nne = iid * (Ne * N0 + 1);			//corresponds to 0, 36, 72,...
		int nne0 = nne + (Ne * N0);			//corresponds to 35, 71, 107,...

		int jj = jjd % Ne;
		int ii = jjd / Ne;

//printf("nne0 %d %d %d %d %d %d\n", id, iid, jjd, nne0, jj, ii);

		if(jj == 0){
			dx = elementsL_d[id * N0 + ii].P;
			gx = elementsD_d[nne0 * N0 + ii].P;
			Gx = beta * elementsG_d[id * N0 + ii].P + (1.0 - beta) * gx * gx;
			elementsG_d[id * N0 + ii].P = Gx;
			dx1 = -eta / sqrt(Gx + eps*dx) * gx * elementsGh_d[id * N0 + ii].P;
			if(gx == 0.0 || dx == 0) dx1 = 0.0;
			//elementsL_d[id * N0 + ii].P = fmax(fmin(fabs(dx1), dx), 1.0e-16);
			for(int j = 0; j < Ne * N0 + 1; ++j){
				if(j % Ne == 0 && j / Ne == ii){
					elementsTOld_d[(j + nne) * N0 + ii].z = elementsTOld_d[nne0 * N0 + ii].z + 1.1 * dx1;
				}
				else{
					elementsTOld_d[(j + nne) * N0 + ii].z = elementsTOld_d[nne0 * N0 + ii].z + dx1;
				}
			}
//printf("dx P %d %d %d %.20g %.20g %g | %g %g %g %g %g\n", id, jjd, ii, 2.0 * elementsP_d[id].z, 2.0 * elementsP_d[nne0].z, elementsTOld_d[id * N0 + ii].z, dx, gx, sqrt(Gx + eps*dx), dx1, eta * elementsGh_d[id * N0 + ii].P);
		}
		if(jj == 1){
			dx = elementsL_d[id * N0 + ii].T;
			gx = elementsD_d[nne0 * N0 + ii].T;
			Gx = beta * elementsG_d[id * N0 + ii].T + (1.0 - beta) * gx * gx;
			elementsG_d[id * N0 + ii].T = Gx;
			dx1 = -eta / sqrt(Gx + eps*dx) * gx * elementsGh_d[id * N0 + ii].T;
			if(gx == 0.0 || dx == 0) dx1 = 0.0;
			//elementsL_d[id * N0 + ii].T = fmax(fmin(fabs(dx1), dx), 1.0e-16);
			for(int j = 0; j < Ne * N0 + 1; ++j){
				if(j % Ne == 1 && j / Ne == ii){
					elementsTOld_d[(j + nne) * N0 + ii].x = elementsTOld_d[nne0 * N0 + ii].x + 1.1 * dx1;
				}
				else{
					elementsTOld_d[(j + nne) * N0 + ii].x = elementsTOld_d[nne0 * N0 + ii].x + dx1;
				}
			}
//printf("dx T %d %d %d %.20g %.20g %g | %g %g %g %g %g\n", id, jjd, ii, 2.0 * elementsP_d[id].z, 2.0 * elementsP_d[nne0].z, elementsTOld_d[id * N0 + ii].x, dx, gx, sqrt(Gx + eps*dx), dx1, eta * elementsGh_d[id * N0 + ii].T);
		}
		if(jj == 2){
			dx = elementsL_d[id * N0 + ii].m;
			gx = elementsD_d[nne0 * N0 + ii].m;
			Gx = beta * elementsG_d[id * N0 + ii].m + (1.0 - beta) * gx * gx;
			elementsG_d[id * N0 + ii].m = Gx;
			dx1 = -eta / sqrt(Gx + eps*dx) * gx * elementsGh_d[id * N0 + ii].m;
			if(gx == 0.0 || dx == 0) dx1 = 0.0;
			//elementsL_d[id * N0 + ii].m = fmax(fmin(fabs(dx1), dx), 1.0e-16);
			for(int j = 0; j < Ne * N0 + 1; ++j){
				if(j % Ne == 2 && j / Ne == ii){
					elementsAOld_d[(j + nne) * N0 + ii].w = elementsAOld_d[nne0 * N0 + ii].w + 1.1 * dx1;
				}
				else{
					 elementsAOld_d[(j + nne) * N0 + ii].w = elementsAOld_d[nne0 * N0 + ii].w + dx1;
				}
			}
//printf("dx m %d %d %d %.20g %.20g %g | %g %g %g %g %g\n", id, jjd, ii, 2.0 * elementsP_d[id].x, 2.0 * elementsP_d[nne0].x, elementsAOld_d[id * N0 + ii].w, dx, gx, sqrt(Gx + eps*dx), dx1, eta * elementsGh_d[id * N0 + ii].m);
		}
		if(jj == 3){
			dx = elementsL_d[id * N0 + ii].e;
			gx = elementsD_d[nne0 * N0 + ii].e;
			Gx = beta * elementsG_d[id * N0 + ii].e + (1.0 - beta) * gx * gx;
			elementsG_d[id * N0 + ii].e = Gx;
			dx1 = -eta / sqrt(Gx + eps*dx) * gx * elementsGh_d[id * N0 + ii].e;
			if(gx == 0.0 || dx == 0) dx1 = 0.0;
			//elementsL_d[id * N0 + ii].e = fmax(fmin(fabs(dx1), dx), 1.0e-16);
			for(int j = 0; j < Ne * N0 + 1; ++j){
				if(j % Ne == 3 && j / Ne == ii){
					elementsAOld_d[(j + nne) * N0 + ii].y = elementsAOld_d[nne0 * N0 + ii].y + 1.1 * dx1;
				}
				else{
					elementsAOld_d[(j + nne) * N0 + ii].y = elementsAOld_d[nne0 * N0 + ii].y + dx1;
				}
			}
//printf("dx e %d %d %d %.20g %.20g %g | %g %g %g %g %g\n", id, jjd, ii, 2.0 * elementsP_d[id].z, 2.0 * elementsP_d[nne0].z, elementsAOld_d[id * N0 + ii].y, dx, gx, sqrt(Gx + eps*dx), dx1, eta * elementsGh_d[id * N0 + ii].e);
		}
		if(jj == 4){
			dx = elementsL_d[id * N0 + ii].w;
			gx = elementsD_d[nne0 * N0 + ii].w;
			Gx = beta * elementsG_d[id * N0 + ii].w + (1.0 - beta) * gx * gx;
			elementsG_d[id * N0 + ii].w = Gx;
			dx1 = -eta / sqrt(Gx + eps*dx) * gx * elementsGh_d[id * N0 + ii].w;
			if(gx == 0.0 || dx == 0) dx1 = 0.0;
			//elementsL_d[id * N0 + ii].w = fmax(fmin(fabs(dx1), dx), 1.0e-16);
			for(int j = 0; j < Ne * N0 + 1; ++j){
				if(j % Ne == 4 && j / Ne == ii){
					elementsBOld_d[(j + nne) * N0 + ii].y = elementsBOld_d[nne0 * N0 + ii].y + 1.1 * dx1;
				}
				else{
					elementsBOld_d[(j + nne) * N0 + ii].y = elementsBOld_d[nne0 * N0 + ii].y + dx1;
				}
			}
//printf("dx w %d %d %d %.20g %.20g %g | %g %g %g %g %g\n", id, jjd, ii, 2.0 * elementsP_d[id].z, 2.0 * elementsP_d[nne0].z, elementsBOld_d[id * N0 + ii].y, dx, gx, sqrt(Gx + eps*dx), dx1, eta * elementsGh_d[id * N0 + ii].w);
		}
	

	}
}


//RMSprop with hyperparmaters optimization and svgd
__global__ void SVGD(double4 *elementsAOld_d, double4 *elementsBOld_d, double4 *elementsTOld_d, elements10 *elementsL_d, elements8 *elementsD_d, double4 *elementsP_d, elements8 *elementsVar_d, const int N0, const int Ne, const int Nst){
	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	int iid = id * (Ne * N0 + 1) + (Ne * N0);			//corresponds to 35, 71,

	//scale variables
	double scaleP = 1.0e-5;
	double scaleT = 1.0e-5;
	double scalem = 1.0e-5;
	double scalee = 0.1;
	double scalew = 2.0*M_PI;
	//When using normalization


	//size factor
	double f1 = 10000.0;

	if(id < Nst / (Ne * N0 + 1)){
//printf("id %d %d\n", id, iid);


		//set gradients to zero
		for(int ii = 0; ii < N0; ++ii){
			elementsD_d[iid * N0 + ii].P = 0.0;
			elementsD_d[iid * N0 + ii].T = 0.0;
			elementsD_d[iid * N0 + ii].m = 0.0;
			elementsD_d[iid * N0 + ii].e = 0.0;
			elementsD_d[iid * N0 + ii].w = 0.0;
		}

		for(int jjd = Ne * N0; jjd < Nst; jjd += Ne * N0 + 1){ 
			//calculate kernel
			double dsq = 0.0;
			//P
			for(int ii = 0; ii < N0; ++ii){
				scaleP = elementsVar_d[ii].P;
				double dd = (elementsTOld_d[iid * N0 + ii].z - elementsTOld_d[jjd * N0 + ii].z) / scaleP;
				dsq += dd * dd;
//printf("ddP %d %d %d %g\n", iid, jjd, ii, dd);
			}
			//T
			for(int ii = 0; ii < N0; ++ii){
				scaleT = elementsVar_d[ii].T;
				double dd = (elementsTOld_d[iid * N0 + ii].x - elementsTOld_d[jjd * N0 + ii].x) / scaleT;
				dsq += dd * dd;
//printf("ddT %d %d %d %g\n", iid, jjd, ii, dd);
			}
			//m
			for(int ii = 0; ii < N0; ++ii){
				scalem = elementsVar_d[ii].m;
				double dd = (elementsAOld_d[iid * N0 + ii].w - elementsAOld_d[jjd * N0 + ii].w) / scalem;
				dsq += dd * dd;
//printf("ddm %d %d %d %g\n", iid, jjd, ii, dd);
			}
//printf("ddm %d %d %g\n", iid, jjd, dsq);
			//e
			for(int ii = 0; ii < N0; ++ii){
				scalee = elementsVar_d[ii].e;
				double dd = (elementsAOld_d[iid * N0 + ii].y - elementsAOld_d[jjd * N0 + ii].y) / scalew;
				dsq += dd * dd;
//printf("dde %d %d %d %g\n", iid, jjd, ii, dd);
			}
//printf("dde %d %d %g\n", iid, jjd, dsq);
			//w
			for(int ii = 0; ii < N0; ++ii){
				scalew = elementsVar_d[ii].w;
				double dd = (elementsBOld_d[iid * N0 + ii].y - elementsBOld_d[jjd * N0 + ii].y) / scalew;
				dsq += dd * dd;
//printf("ddw %d %d %d %g\n", iid, jjd, ii, dd);
			}
//printf("ddw %d %d %g\n", iid, jjd, dsq);
			double kij = exp(-1.0 * dsq);
	//kij = 1.0;
	//if(jjd != iid) kij = 0.0;


//printf("kij %d %d %g %g\n", iid, jjd, kij, dsq);

			//calculate gradient

			//gradients are P0, T0, m0, e0, w0, P1, T1, m1, e1, w1, ... Pn, Tn, mn, en, wn, NoGrad.
			double dx, gx, dd;
			//P
			for(int ii = 0; ii < N0; ++ii){
				dx = elementsL_d[jjd * N0 + ii].P;
				gx = -2.0 * (elementsP_d[jjd].z - elementsP_d[jjd - Ne * N0 + ii * Ne].z) / dx;
//if(iid == jjd && iid == 35) printf("dP %d %d %d %g\n", ii, jjd, jjd - Ne * N0 + ii * Ne, dx);
				elementsD_d[iid * N0 + ii].P += gx * kij;
				scaleP = elementsVar_d[ii].P;

				dd = (elementsTOld_d[iid * N0 + ii].z - elementsTOld_d[jjd * N0 + ii].z) / (scaleP * scaleP);
				elementsD_d[iid * N0 + ii].P += 2.0 * dd * kij * f1;
			}
			//T
			for(int ii = 0; ii < N0; ++ii){
				dx = elementsL_d[jjd * N0 + ii].T;
				gx = -2.0 * (elementsP_d[jjd].z - elementsP_d[jjd - Ne * N0 + ii * Ne + 1].z) / dx;
//if(iid == jjd && iid == 35) printf("dT %d %d %d %g\n", ii, jjd, jjd - Ne * N0 + ii * Ne + 1, dx);

				elementsD_d[iid * N0 + ii].T += gx * kij;
				scaleT = elementsVar_d[ii].T;

				dd = (elementsTOld_d[iid * N0 + ii].x - elementsTOld_d[jjd * N0 + ii].x) / (scaleT * scaleT);
				elementsD_d[iid * N0 + ii].T += 2.0 * dd * kij * f1;
			}
			//m
			for(int ii = 0; ii < N0; ++ii){
				dx = elementsL_d[jjd * N0 + ii].m;
				gx = -2.0 * (elementsP_d[jjd].z - elementsP_d[jjd - Ne * N0 + ii * Ne + 2].z) / dx;
//if(iid == jjd && iid == 35) printf("dm %d %d %d %g\n", ii, jjd, jjd - Ne * N0 + ii * Ne + 2, dx);

				elementsD_d[iid * N0 + ii].m += gx * kij;
				scalem = elementsVar_d[ii].m;

				dd = (elementsAOld_d[iid * N0 + ii].w - elementsAOld_d[jjd * N0 + ii].w) / (scalem * scalem);
				elementsD_d[iid * N0 + ii].m += 2.0 * dd * kij * f1;
			}
			//e
			for(int ii = 0; ii < N0; ++ii){
				dx = elementsL_d[jjd * N0 + ii].e;
				gx = -2.0 * (elementsP_d[jjd].z - elementsP_d[jjd - Ne * N0 + ii * Ne + 3].z) / dx;
//if(iid == jjd && iid == 35) printf("de %d %d %d %g\n", ii, jjd, jjd - Ne * N0 + ii * Ne + 3, dx);

				elementsD_d[iid * N0 + ii].e += gx * kij;
				scalee = elementsVar_d[ii].e;

				dd = (elementsAOld_d[iid * N0 + ii].y - elementsAOld_d[jjd * N0 + ii].y) / (scalee * scalee);
				elementsD_d[iid * N0 + ii].e += 2.0 * dd * kij * f1;
//printf("dde %d %d %g %g %g\n", iid, jjd, kij, gx, 2.0 * dd);
			}
			//w
			for(int ii = 0; ii < N0; ++ii){
				dx = elementsL_d[jjd * N0 + ii].w;
				gx = -2.0 * (elementsP_d[jjd].z - elementsP_d[jjd - Ne * N0 + ii * Ne + 4].z) / dx;
//if(iid == jjd && iid == 35) printf("dw %d %d %d %g\n", ii, jjd, jjd - Ne * N0 + ii * Ne + 4, dx);

				elementsD_d[iid * N0 + ii].w += gx * kij;
				scalew = elementsVar_d[ii].w;

				dd = (elementsBOld_d[iid * N0 + ii].y - elementsBOld_d[jjd * N0 + ii].y) / (scalew * scalew);
				elementsD_d[iid * N0 + ii].w += 2.0 * dd * kij * f1;
//printf("ddw %d %d %g %g %g\n", iid, jjd, kij, gx, 2.0 * dd);
			}
		}
	
		for(int ii = 0; ii < N0; ++ii){
			elementsD_d[iid * N0 + ii].P /= Nst / double(Ne * N0 + 1);
			elementsD_d[iid * N0 + ii].T /= Nst / double(Ne * N0 + 1);
			elementsD_d[iid * N0 + ii].m /= Nst / double(Ne * N0 + 1);
			elementsD_d[iid * N0 + ii].e /= Nst / double(Ne * N0 + 1);
			elementsD_d[iid * N0 + ii].w /= Nst / double(Ne * N0 + 1);
		}

	}
}


//adamW
__global__ void adam(double4 *elementsAOld_d, double4 *elementsBOld_d, double4 *elementsTOld_d, elements10 *elementsL_d, elements8 *elementsG_d, elements8 *elementsD_d, elements8 *elementsGh_d, double4 *elementsP_d, const int N0, const int Ne, const int Nst, const int ittv){
	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	if(id < Nst - 1){
		double dx, dx1, gx, Gx, Dx;

		double beta1 = 0.9;
		double beta2 = 0.999;
		double beta1t = pow(beta1, ittv);
		double beta2t = pow(beta2, ittv);
		double eta = 0.1;
		double eps = 1.0e-9;

		int jj = id % Ne;
		int ii = id / Ne;
		if(jj == 0){
			dx = elementsL_d[id * N0 + ii].P;
			gx = -2.0 * (elementsP_d[35].z - elementsP_d[id].z) / dx;
			Gx = beta2 * elementsG_d[id * N0 + ii].P + (1.0 - beta2) * gx * gx;
			Dx = beta1 * elementsD_d[id * N0 + ii].P + (1.0 - beta1) * gx;
			double DDx = Dx / (1.0 - beta1t);
			double GGx = Gx / (1.0 - beta2t);
			elementsG_d[id * N0 + ii].P = Gx;
			elementsD_d[id * N0 + ii].P = Dx;
			dx1 = -eta / sqrt(GGx + eps) * DDx;
			if(gx == 0.0 || dx == 0) dx1 = 0.0;
			//elementsL_d[id * N0 + ii].P = fmax(fmin(fabs(dx1), dx), 1.0e-16);
			for(int j = 0; j < Nst; ++j){
				elementsTOld_d[j * N0 + ii].z = elementsTOld_d[35 * N0 + ii].z + dx1 * elementsGh_d[j].P;
			}
printf("dx P %d %d %g %g | %g %g %g %g\n", id, ii, 2.0 * elementsP_d[id].x, 2.0 * elementsP_d[id].z, dx, gx, sqrt(Gx + eps), dx1);
		}
		if(jj == 1){
			dx = elementsL_d[id * N0 + ii].T;
			gx = -2.0 * (elementsP_d[35].z - elementsP_d[id].z) / dx;
			Gx = beta2 * elementsG_d[id * N0 + ii].T + (1.0 - beta2) * gx * gx;
			Dx = beta1 * elementsD_d[id * N0 + ii].T + (1.0 - beta1) * gx;
			double DDx = Dx / (1.0 - beta1t);
			double GGx = Gx / (1.0 - beta2t);
			elementsG_d[id * N0 + ii].T = Gx;
			elementsD_d[id * N0 + ii].T = Dx;
			dx1 = -eta / sqrt(GGx + eps) * DDx;
			if(gx == 0.0 || dx == 0) dx1 = 0.0;
			//elementsL_d[id * N0 + ii].T = fmax(fmin(fabs(dx1), dx), 1.0e-16);
			for(int j = 0; j < Nst; ++j){
				elementsTOld_d[j * N0 + ii].x = elementsTOld_d[35 * N0 + ii].x + dx1 * elementsGh_d[j].T;
			}
printf("dx T %d %d %g %g | %g %g %g %g\n", id, ii, 2.0 * elementsP_d[id].x, 2.0 * elementsP_d[id].z, dx, gx, sqrt(Gx + eps), dx1);
		}
		if(jj == 2){
			dx = elementsL_d[id * N0 + ii].m;
			gx = -2.0 * (elementsP_d[35].z - elementsP_d[id].z) / dx;
			Gx = beta2 * elementsG_d[id * N0 + ii].m + (1.0 - beta2) * gx * gx;
			Dx = beta1 * elementsD_d[id * N0 + ii].m + (1.0 - beta1) * gx;
			double DDx = Dx / (1.0 - beta1t);
			double GGx = Gx / (1.0 - beta2t);
			elementsG_d[id * N0 + ii].m = Gx;
			elementsD_d[id * N0 + ii].m = Dx;
			dx1 = -eta / sqrt(GGx + eps) * DDx;
			if(gx == 0.0 || dx == 0) dx1 = 0.0;
			//elementsL_d[id * N0 + ii].m = fmax(fmin(fabs(dx1), dx), 1.0e-16);
			for(int j = 0; j < Nst; ++j){
				elementsAOld_d[j * N0 + ii].w = elementsAOld_d[35 * N0 + ii].w + dx1 * elementsGh_d[j].m;
			}
printf("dx m %d %d %g %g | %g %g %g %g\n", id, ii, 2.0 * elementsP_d[id].x, 2.0 * elementsP_d[id].z, dx, gx, sqrt(Gx + eps), dx1);
		}
		if(jj == 3){
			dx = elementsL_d[id * N0 + ii].e;
			gx = -2.0 * (elementsP_d[35].z - elementsP_d[id].z) / dx;
			Gx = beta2 * elementsG_d[id * N0 + ii].e + (1.0 - beta2) * gx * gx;
			Dx = beta1 * elementsD_d[id * N0 + ii].e + (1.0 - beta1) * gx;
			double DDx = Dx / (1.0 - beta1t);
			double GGx = Gx / (1.0 - beta2t);
			elementsG_d[id * N0 + ii].e = Gx;
			elementsD_d[id * N0 + ii].e = Dx;
			dx1 = -eta / sqrt(GGx + eps) * DDx;
			if(gx == 0.0 || dx == 0) dx1 = 0.0;
			//elementsL_d[id * N0 + ii].e = fmax(fmin(fabs(dx1), dx), 1.0e-16);
			for(int j = 0; j < Nst; ++j){
				elementsAOld_d[j * N0 + ii].y = elementsAOld_d[35 * N0 + ii].y + dx1 * elementsGh_d[j].e;
			}
printf("dx e %d %d %g %g | %g %g %g %g\n", id, ii, 2.0 * elementsP_d[id].x, 2.0 * elementsP_d[id].z, dx, gx, sqrt(Gx + eps), dx1);
		}
		if(jj == 4){
			dx = elementsL_d[id * N0 + ii].w;
			gx = -2.0 * (elementsP_d[35].z - elementsP_d[id].z) / dx;
			Gx = beta2 * elementsG_d[id * N0 + ii].w + (1.0 - beta2) * gx * gx;
			Dx = beta1 * elementsD_d[id * N0 + ii].w + (1.0 - beta1) * gx;
			double DDx = Dx / (1.0 - beta1t);
			double GGx = Gx / (1.0 - beta2t);
			elementsG_d[id * N0 + ii].w = Gx;
			elementsD_d[id * N0 + ii].w = Dx;
			dx1 = -eta / sqrt(GGx + eps) * DDx;
			if(gx == 0.0 || dx == 0) dx1 = 0.0;
			//elementsL_d[id * N0 + ii].w = fmax(fmin(fabs(dx1), dx), 1.0e-16);
			for(int j = 0; j < Nst; ++j){
				elementsBOld_d[j * N0 + ii].y = elementsBOld_d[35 * N0 + ii].y + dx1 * elementsGh_d[j].w;
			}
printf("dx w %d %d %g %g | %g %g %g %g\n", id, ii, 2.0 * elementsP_d[id].x, 2.0 * elementsP_d[id].z, dx, gx, sqrt(Gx + eps), dx1);
		}


	}
}


__global__ void findMin(double4 *elementsAOld_d, double4 *elementsBOld_d, double4 *elementsTOld_d, elements8 *elementsGh_d, double4 *elementsP_d, const int Nst, const int N0){
	
	double min = 1.0e300;
	int imin = 0;
	for(int i = 0; i < Nst; ++i){
		if(elementsP_d[i].z < min){
			imin = i;
			min = elementsP_d[i].z;
		}	
printf("min %d %d %g %g \n", i, imin, min, elementsP_d[i].z);

	}
	imin = max(imin - 2, 0);

	for(int i = 0; i < Nst; ++i){
		for(int j = 0; j < N0; ++j){
			elementsTOld_d[i * N0 + j].z = elementsTOld_d[imin * N0 + j].z;
			elementsTOld_d[i * N0 + j].x = elementsTOld_d[imin * N0 + j].x;
			elementsAOld_d[i * N0 + j].w = elementsAOld_d[imin * N0 + j].w;
			elementsAOld_d[i * N0 + j].y = elementsAOld_d[imin * N0 + j].y;
			elementsBOld_d[i * N0 + j].y = elementsBOld_d[imin * N0 + j].y;

		}
		elementsGh_d[i].P = elementsGh_d[imin].P;
		elementsGh_d[i].T = elementsGh_d[imin].T;
		elementsGh_d[i].m = elementsGh_d[imin].m;
		elementsGh_d[i].e = elementsGh_d[imin].e;
		elementsGh_d[i].w = elementsGh_d[imin].w;
		elementsP_d[i].z = elementsP_d[imin].z;
	}
}


//This kernel modifies the non-gradient simulations with random numbers. Used to initialize multiple gradient runs.
__global__ void rmsPropRand(curandState *random_d, double4 *elementsAOld_d, double4 *elementsBOld_d, double4 *elementsTOld_d, elements10 *elementsL_d, const int N0, const int Ne, const int Nst){
#if USE_RANDOM == 1
	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	if(id < Nst / (Ne * N0 + 1)){
		
		int iid = id * (Ne * N0 + 1);			//corresponds to 0, 36, 72,...

		curandState random;
		random = random_d[id];
		double rd;

		for(int ii = 0; ii < N0; ++ii){
			if(Ne > 0){
				rd = curand_normal(&random);
				elementsTOld_d[iid * N0 + ii].z += rd * elementsL_d[iid * N0 + ii].P;       //scale to standard deviation of tuning length
			}
			if(Ne > 1){
				rd = curand_normal(&random);
				elementsTOld_d[iid * N0 + ii].x += rd * elementsL_d[iid * N0 + ii].T;
			}
			if(Ne > 2){
				rd = curand_normal(&random);
				elementsAOld_d[iid * N0 + ii].w += rd * elementsL_d[iid * N0 + ii].m;
			}
			if(Ne > 3){
				rd = curand_normal(&random);
				elementsAOld_d[iid * N0 + ii].y += rd * elementsL_d[iid * N0 + ii].e;
			}
			if(Ne > 4){
				rd = curand_normal(&random);
				elementsBOld_d[iid * N0 + ii].y += rd * elementsL_d[iid * N0 + ii].w;
			}
		}

		for(int j = 0; j < Ne * N0 + 1; ++j){
			for(int ii = 0; ii < N0; ++ii){
				elementsTOld_d[(j + iid) * N0 + ii].z = elementsTOld_d[iid * N0 + ii].z; //P
				elementsTOld_d[(j + iid) * N0 + ii].x = elementsTOld_d[iid * N0 + ii].x; //T
				elementsAOld_d[(j + iid) * N0 + ii].w = elementsAOld_d[iid * N0 + ii].w; //m
				elementsAOld_d[(j + iid) * N0 + ii].y = elementsAOld_d[iid * N0 + ii].y; //e
				elementsBOld_d[(j + iid) * N0 + ii].y = elementsBOld_d[iid * N0 + ii].y; //w
			}
		}
//printf("dh P %d %d %d %.20g %.20g | %g %g\n", id, jjd, ii, 2.0 * elementsP_d[nne0].z, 2.0 * elementsP_d[id].z, elementsGh_d[id * N0 + ii].P, elementsTOld_d[id * N0 + ii].z);
		random_d[id] = random;
	}
#endif
}

__global__ void tuneHyperParameters(double4 *elementsAOld_d, double4 *elementsBOld_d, double4 *elementsTOld_d, elements8 *elementsGh_d, double4 *elementsP_d, const int N0, const int Ne, const int Nst, const int ittv){
	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	int iid = id / (Ne * N0 + 1);	//simulation index, consisting with all the (35) gradient points, plus one point without gradient (total 36)
	int jjd = id % (Ne * N0 + 1);	// map to 0 - 35

	if(id < Nst && jjd < Ne * N0){

		int nne = iid * (Ne * N0 + 1);			//corresponds to 0, 36, 72,...
		int nne0 = nne + (Ne * N0);			//corresponds to 35, 71, 107,...

		int jj = jjd % Ne;
		int ii = jjd / Ne;

		if(jj == 0){
			if(elementsP_d[id].z < elementsP_d[nne0].z){
				elementsGh_d[id * N0 + ii].P *= 1.1;
			}
			if(elementsP_d[id].z > elementsP_d[nne0].z){
				elementsGh_d[id * N0 + ii].P *= 0.9;
			}
			for(int j = 0; j < Ne * N0 + 1; ++j){
				elementsTOld_d[(j + nne) * N0 + ii].z = elementsTOld_d[nne0 * N0 + ii].z;
			}
//printf("dh P %d %d %d %.20g %.20g | %g %g\n", id, jjd, ii, 2.0 * elementsP_d[nne0].z, 2.0 * elementsP_d[id].z, elementsGh_d[id * N0 + ii].P, elementsTOld_d[id * N0 + ii].z);
		}
		if(jj == 1){
			if(elementsP_d[id].z < elementsP_d[nne0].z){
				elementsGh_d[id * N0 + ii].T *= 1.1;
			}
			if(elementsP_d[id].z > elementsP_d[nne0].z){
				elementsGh_d[id * N0 + ii].T *= 0.9;
			}
			for(int j = 0; j < Ne * N0 + 1; ++j){
				elementsTOld_d[(j + nne) * N0 + ii].x = elementsTOld_d[nne0 * N0 + ii].x;
			}
//printf("dh T %d %d %d %.20g %.20g | %g %g\n", id, jjd, ii, 2.0 * elementsP_d[nne0].z, 2.0 * elementsP_d[id].z, elementsGh_d[id * N0 + ii].T, elementsTOld_d[id * N0 + ii].x);
		}
		if(jj == 2){
			if(elementsP_d[id].z < elementsP_d[nne0].z){
				elementsGh_d[id * N0 + ii].m *= 1.1;
			}
			if(elementsP_d[id].z > elementsP_d[nne0].z){
				elementsGh_d[id * N0 + ii].m *= 0.9;
			}
			for(int j = 0; j < Ne * N0 + 1; ++j){
				elementsAOld_d[(j + nne) * N0 + ii].w = elementsAOld_d[nne0 * N0 + ii].w;
			}
//printf("dh m %d %d %d %.20g %.20g | %g %g\n", id, jjd, ii, 2.0 * elementsP_d[nne0].z, 2.0 * elementsP_d[id].z, elementsGh_d[id * N0 + ii].m, elementsAOld_d[id * N0 + ii].w);
		}
		if(jj == 3){
			if(elementsP_d[id].z < elementsP_d[nne0].z){
				elementsGh_d[id * N0 + ii].e *= 1.1;
			}
			if(elementsP_d[id].z > elementsP_d[nne0].z){
				elementsGh_d[id * N0 + ii].e *= 0.9;
			}
			for(int j = 0; j < Ne * N0 + 1; ++j){
				elementsAOld_d[(j + nne) * N0 + ii].y = elementsAOld_d[nne0 * N0 + ii].y;
			}
//printf("dh e %d %d %d %.20g %.20g | %g %g\n", id, jjd, ii, 2.0 * elementsP_d[nne0].z, 2.0 * elementsP_d[id].z, elementsGh_d[id * N0 + ii].e, elementsAOld_d[id * N0 + ii].y);
		}
		if(jj == 4){
			if(elementsP_d[id].z < elementsP_d[nne0].z){
				elementsGh_d[id * N0 + ii].w *= 1.1;
			}
			if(elementsP_d[id].z > elementsP_d[nne0].z){
				elementsGh_d[id * N0 + ii].w *= 0.9;
			}
			for(int j = 0; j < Ne * N0 + 1; ++j){
				elementsBOld_d[(j + nne) * N0 + ii].y = elementsBOld_d[nne0 * N0 + ii].y;
			}
//printf("dh w %d %d %d %.20g %.20g | %g %g\n", id, jjd, ii, 2.0 * elementsP_d[nne0].z, 2.0 * elementsP_d[id].z, elementsGh_d[id * N0 + ii].w, elementsBOld_d[id * N0 + ii].y);
		}
		elementsP_d[id].z = elementsP_d[nne0].z;

	}
}

//Nelder Mead
__global__ void nelderMead(double4 *elementsAOld_d, double4 *elementsBOld_d, double4 *elementsTOld_d, elements10 *elementsL_d, double4 *elementsP_d, elements *Symplex_d, int *SymplexCount_d, const int N0, const int Ne, const int Nst, const int ittv){
	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	double alpha = 1.0;
	double gamma = 2.0;
	double beta = 0.5;
	double sigma = 0.5;

	if(ittv == 1) SymplexCount_d[id] = 0;


	if(id < Nst){
	
		if(SymplexCount_d[id] < N0 * Ne + 1){	
printf("store f %d %.20g\n", SymplexCount_d[id], 2.0 * elementsP_d[id].z);
			for(int i = 0; i < N0; ++i){
				Symplex_d[SymplexCount_d[id] * (Nst * N0) + id * N0 + i].f = 2.0 * elementsP_d[id].z;
			}
		}


		if(SymplexCount_d[id] == 1000){
			SymplexCount_d[id] = 2000;
printf("store reflect %d %.20g\n", N0 * Ne + 1, 2.0 * elementsP_d[id].z);
			for(int i = 0; i < N0; ++i){
				Symplex_d[(N0 * Ne + 1) * (Nst * N0) + id * N0 + i].f = 2.0 * elementsP_d[id].z;
			}
		}

		if(SymplexCount_d[id] == N0 * Ne) SymplexCount_d[id] = 1000;

		//build very first symplex
		if(ittv < N0 * Ne + 1){
			for(int i = 0; i < N0; ++i){
				if(ittv == 1){
					Symplex_d[(0) * (Nst * N0) + id * N0 + i].P = elementsTOld_d[id * N0 + i].z;
					Symplex_d[(0) * (Nst * N0) + id * N0 + i].T = elementsTOld_d[id * N0 + i].x;
					Symplex_d[(0) * (Nst * N0) + id * N0 + i].m = elementsAOld_d[id * N0 + i].w;
					Symplex_d[(0) * (Nst * N0) + id * N0 + i].e = elementsAOld_d[id * N0 + i].y;
					Symplex_d[(0) * (Nst * N0) + id * N0 + i].w = elementsBOld_d[id * N0 + i].y;
				}
				else{
					//copy from first symplex
					elementsTOld_d[id * N0 + i].z = Symplex_d[id * N0 + i].P;
					elementsTOld_d[id * N0 + i].x = Symplex_d[id * N0 + i].T;
					elementsAOld_d[id * N0 + i].w = Symplex_d[id * N0 + i].m;
					elementsAOld_d[id * N0 + i].y = Symplex_d[id * N0 + i].e;
					elementsBOld_d[id * N0 + i].y = Symplex_d[id * N0 + i].w;
				}
			}

			double dx;
			
			int jj = (ittv - 1) % Ne;
			int ii = (ittv - 1) / Ne;
			if(jj == 0){
				//P
				dx = elementsL_d[id * N0 + ii].P;
				elementsTOld_d[id * N0 + ii].z += dx;
printf("dx P %d %d %d %.20g %.20g\n", ittv, id, ii, elementsTOld_d[id * N0 + ii].z, dx);
			}
			if(jj == 1){
				//T
				dx = elementsL_d[id * N0 + ii].T;
				elementsTOld_d[id * N0 + ii].x += dx;
printf("dx T %d %d %d %.20g %.20g\n", ittv, id, ii, elementsTOld_d[id * N0 + ii].x, dx);
			}
			if(jj == 2){
				//m
				dx = elementsL_d[id * N0 + ii].m;
				elementsAOld_d[id * N0 + ii].w += dx;
printf("dx m %d %d %d %.20g %.20g\n", ittv, id, ii, elementsAOld_d[id * N0 + ii].w, dx);
			}
			if(jj == 3){
				//e
				dx = elementsL_d[id * N0 + ii].e;
				elementsAOld_d[id * N0 + ii].y += dx;
printf("dx e %d %d %d %.20g %.20g\n", ittv, id, ii, elementsAOld_d[id * N0 + ii].y, dx);
			}
			if(jj == 4){
				//w
				dx = elementsL_d[id * N0 + ii].w;
				elementsBOld_d[id * N0 + ii].y += dx;
printf("dx w %d %d %d %.20g %.20g\n", ittv, id, ii, elementsBOld_d[id * N0 + ii].y, dx);
			}

			for(int i = 0; i < N0; ++i){
				Symplex_d[(ittv) * (Nst * N0) + id * N0 + i].P = elementsTOld_d[id * N0 + i].z;
				Symplex_d[(ittv) * (Nst * N0) + id * N0 + i].T = elementsTOld_d[id * N0 + i].x;
				Symplex_d[(ittv) * (Nst * N0) + id * N0 + i].m = elementsAOld_d[id * N0 + i].w;
				Symplex_d[(ittv) * (Nst * N0) + id * N0 + i].e = elementsAOld_d[id * N0 + i].y;
				Symplex_d[(ittv) * (Nst * N0) + id * N0 + i].w = elementsBOld_d[id * N0 + i].y;
			}
			++SymplexCount_d[id];
		}



		if(SymplexCount_d[id] == 3000){
					
			int iM = 0;	//best solution
			int im = 0;	//worst solution
			int im2 = 0;	//second worst solution


			for(int i = 1; i < N0 * Ne + 1; ++i){
				if(Symplex_d[(i) * (Nst * N0) + id * N0].f < Symplex_d[(iM) * (Nst * N0) + id * N0].f){
					iM = i;
				}
			}
			for(int i = 1; i < N0 * Ne + 1; ++i){
				if(Symplex_d[(i) * (Nst * N0) + id * N0].f > Symplex_d[(im) * (Nst * N0) + id * N0].f){
					im = i;
				}
			}

			if(im == 0) im2 = 1;
			for(int i = 1; i < N0 * Ne + 1; ++i){
				if(Symplex_d[(i) * (Nst * N0) + id * N0].f > Symplex_d[(im2) * (Nst * N0) + id * N0].f && i != im){
					im2 = i;
				}
			}
printf("min maxB %d %d %d %d %.20g %.20g %.20g | %.20g %.20g\n", ittv, iM, im2, im, Symplex_d[(iM) * (Nst * N0) + id * N0].f, Symplex_d[(im2) * (Nst * N0) + id * N0].f, Symplex_d[(im) * (Nst * N0) + id * N0].f, Symplex_d[(N0 * Ne + 1) * (Nst * N0) + id * N0].f, 2.0 * elementsP_d[id].z);

			if(Symplex_d[(N0 * Ne + 1) * (Nst * N0) + id * N0].f < Symplex_d[iM * (Nst * N0) + id * N0].f){

				if(2.0 * elementsP_d[id].z < Symplex_d[(N0 * Ne + 1) * (Nst * N0) + id * N0].f){
					//replace xN with e
printf("expandB replace with e\n");
					for(int i = 0; i < N0; ++i){
						Symplex_d[(im) * (Nst * N0) + id * N0 + i].P = elementsTOld_d[id * N0 + i].z;
						Symplex_d[(im) * (Nst * N0) + id * N0 + i].T = elementsTOld_d[id * N0 + i].x;
						Symplex_d[(im) * (Nst * N0) + id * N0 + i].m = elementsAOld_d[id * N0 + i].w;
						Symplex_d[(im) * (Nst * N0) + id * N0 + i].e = elementsAOld_d[id * N0 + i].y;
						Symplex_d[(im) * (Nst * N0) + id * N0 + i].w = elementsBOld_d[id * N0 + i].y;
						Symplex_d[(im) * (Nst * N0) + id * N0 + i].f = 2.0 * elementsP_d[id].z;

					}
				}
				else{
printf("expandB replace with r\n");
					//replace xN with r
					for(int i = 0; i < N0; ++i){
						Symplex_d[(im) * (Nst * N0) + id * N0 + i] = Symplex_d[(N0 * Ne + 1) * (Nst * N0) + id * N0 + i];
					}
				}
			}
			else if(Symplex_d[(N0 * Ne + 1) * (Nst * N0) + id * N0].f < Symplex_d[im2 * (Nst * N0) + id * N0].f){
printf("replaceB replace with r\n");
					//replace xN with r
					for(int i = 0; i < N0; ++i){
						Symplex_d[(im) * (Nst * N0) + id * N0 + i] = Symplex_d[(N0 * Ne + 1) * (Nst * N0) + id * N0 + i];
					}

			}
			else if(2.0 * elementsP_d[id].z < Symplex_d[im * (Nst * N0) + id * N0].f){
printf("contractB replace with c\n");
				//replace xN with c
				for(int i = 0; i < N0; ++i){
					Symplex_d[(im) * (Nst * N0) + id * N0 + i].P = elementsTOld_d[id * N0 + i].z;
					Symplex_d[(im) * (Nst * N0) + id * N0 + i].T = elementsTOld_d[id * N0 + i].x;
					Symplex_d[(im) * (Nst * N0) + id * N0 + i].m = elementsAOld_d[id * N0 + i].w;
					Symplex_d[(im) * (Nst * N0) + id * N0 + i].e = elementsAOld_d[id * N0 + i].y;
					Symplex_d[(im) * (Nst * N0) + id * N0 + i].w = elementsBOld_d[id * N0 + i].y;
					Symplex_d[(im) * (Nst * N0) + id * N0 + i].f = 2.0 * elementsP_d[id].z;
				}

			}
			else{
				//shrink
//restart simplex build
				for(int j = 0; j < N0 * Ne + 1; ++j){
					for(int i = 0; i < N0; ++i){
						Symplex_d[(j) * (Nst * N0) + id * N0 + i].P = sigma * Symplex_d[(iM) * (Nst * N0) + id * N0 + i].P + (1.0 - sigma) * Symplex_d[(j) * (Nst * N0) + id * N0 + i].P;
						Symplex_d[(j) * (Nst * N0) + id * N0 + i].T = sigma * Symplex_d[(iM) * (Nst * N0) + id * N0 + i].T + (1.0 - sigma) * Symplex_d[(j) * (Nst * N0) + id * N0 + i].T;
						Symplex_d[(j) * (Nst * N0) + id * N0 + i].m = sigma * Symplex_d[(iM) * (Nst * N0) + id * N0 + i].m + (1.0 - sigma) * Symplex_d[(j) * (Nst * N0) + id * N0 + i].m;
						Symplex_d[(j) * (Nst * N0) + id * N0 + i].e = sigma * Symplex_d[(iM) * (Nst * N0) + id * N0 + i].e + (1.0 - sigma) * Symplex_d[(j) * (Nst * N0) + id * N0 + i].e;
						Symplex_d[(j) * (Nst * N0) + id * N0 + i].w = sigma * Symplex_d[(iM) * (Nst * N0) + id * N0 + i].w + (1.0 - sigma) * Symplex_d[(j) * (Nst * N0) + id * N0 + i].w;
					}
				}
printf("shrink\n");
				SymplexCount_d[id] = -1;

			}
			if(SymplexCount_d[id] == 3000) SymplexCount_d[id] = 1000;

		}

		//rebuild symplex, after shrinking
		if(ittv >= N0 * Ne + 2 && SymplexCount_d[id] < N0 * Ne){
			int sc = SymplexCount_d[id] + 1;
			for(int i = 0; i < N0; ++i){
				elementsTOld_d[id * N0 + i].z = Symplex_d[sc * (Nst * N0) + id * N0 + i].P;
				elementsTOld_d[id * N0 + i].x = Symplex_d[sc * (Nst * N0) + id * N0 + i].T;
				elementsAOld_d[id * N0 + i].w = Symplex_d[sc * (Nst * N0) + id * N0 + i].m;
				elementsAOld_d[id * N0 + i].y = Symplex_d[sc * (Nst * N0) + id * N0 + i].e;
				elementsBOld_d[id * N0 + i].y = Symplex_d[sc * (Nst * N0) + id * N0 + i].w;
			}
			++SymplexCount_d[id];
printf("rebuild symplex %d\n", SymplexCount_d[id]);

		}

		if(SymplexCount_d[id] == 1000 || SymplexCount_d[id] == 2000){
			
			int iM = 0;	//best solution
			int im = 0;	//worst solution
			int im2 = 0;	//second worst solution


			for(int i = 1; i < N0 * Ne + 1; ++i){
				if(Symplex_d[(i) * (Nst * N0) + id * N0].f < Symplex_d[(iM) * (Nst * N0) + id * N0].f){
					iM = i;
				}
			}
			for(int i = 1; i < N0 * Ne + 1; ++i){
				if(Symplex_d[(i) * (Nst * N0) + id * N0].f > Symplex_d[(im) * (Nst * N0) + id * N0].f){
					im = i;
				}
			}

			if(im == 0) im2 = 1;
			for(int i = 1; i < N0 * Ne + 1; ++i){
				if(Symplex_d[(i) * (Nst * N0) + id * N0].f > Symplex_d[(im2) * (Nst * N0) + id * N0].f && i != im){
					im2 = i;
				}
			}
printf("min max %d %d %d %d %.20g %.20g %.20g | %.20g \n", ittv, iM, im2, im, Symplex_d[(iM) * (Nst * N0) + id * N0].f, Symplex_d[(im2) * (Nst * N0) + id * N0].f, Symplex_d[(im) * (Nst * N0) + id * N0].f, Symplex_d[(N0 * Ne + 1) * (Nst * N0) + id * N0].f);

			for(int i = 0; i < N0; ++i){
				double T = 0.0;
				double P = 0.0;
				double m = 0.0;
				double e = 0.0;
				double w = 0.0;
				//calcualate average expect of worst point
				for(int j = 0; j < N0 * Ne + 1; ++j){
					if(j != im){
						P += Symplex_d[j * (Nst * N0) + id * N0 + i].P;
						T += Symplex_d[j * (Nst * N0) + id * N0 + i].T;
						m += Symplex_d[j * (Nst * N0) + id * N0 + i].m;
						e += Symplex_d[j * (Nst * N0) + id * N0 + i].e;
						w += Symplex_d[j * (Nst * N0) + id * N0 + i].w;
						
					}
				}
				P = P / (double(N0 * Ne));
				T = T / (double(N0 * Ne));
				m = m / (double(N0 * Ne));
				e = e / (double(N0 * Ne));
				w = w / (double(N0 * Ne));
//printf("average %d %.20g %.20g %.20g %.20g %.20g\n", i, P, T, m, e, w);
				if(SymplexCount_d[id] == 1000){
					//reflect worst point
					elementsTOld_d[id * N0 + i].z = (1.0 + alpha) * P - alpha * Symplex_d[(im) * (Nst * N0) + id * N0 + i].P;
					elementsTOld_d[id * N0 + i].x = (1.0 + alpha) * T - alpha * Symplex_d[(im) * (Nst * N0) + id * N0 + i].T;
					elementsAOld_d[id * N0 + i].w = (1.0 + alpha) * m - alpha * Symplex_d[(im) * (Nst * N0) + id * N0 + i].m;
					elementsAOld_d[id * N0 + i].y = (1.0 + alpha) * e - alpha * Symplex_d[(im) * (Nst * N0) + id * N0 + i].e;
					elementsBOld_d[id * N0 + i].y = (1.0 + alpha) * w - alpha * Symplex_d[(im) * (Nst * N0) + id * N0 + i].w;
if(i == 0) printf("reflect\n");
//printf("reflect %d %.20g %.20g %.20g %.20g %.20g\n", i, elementsTOld_d[id * N0 + i].z, elementsTOld_d[id * N0 + i].x, elementsAOld_d[id * N0 + i].w, elementsAOld_d[id * N0 + i].y, elementsBOld_d[id * N0 + i].y);

					Symplex_d[(N0 * Ne + 1) * (Nst * N0) + id * N0 + i].P = elementsTOld_d[id * N0 + i].z;
					Symplex_d[(N0 * Ne + 1) * (Nst * N0) + id * N0 + i].T = elementsTOld_d[id * N0 + i].x;
					Symplex_d[(N0 * Ne + 1) * (Nst * N0) + id * N0 + i].m = elementsAOld_d[id * N0 + i].w;
					Symplex_d[(N0 * Ne + 1) * (Nst * N0) + id * N0 + i].e = elementsAOld_d[id * N0 + i].y;
					Symplex_d[(N0 * Ne + 1) * (Nst * N0) + id * N0 + i].w = elementsBOld_d[id * N0 + i].y;
				}
				if(SymplexCount_d[id] == 2000){
					SymplexCount_d[id] = 3000;
					if(2.0 * elementsP_d[id].z < Symplex_d[iM * (Nst * N0) + id * N0].f){
						//expand
						elementsTOld_d[id * N0 + i].z = (1.0 + gamma) * P - gamma * Symplex_d[(im) * (Nst * N0) + id * N0 + i].P;
						elementsTOld_d[id * N0 + i].x = (1.0 + gamma) * T - gamma * Symplex_d[(im) * (Nst * N0) + id * N0 + i].T;
						elementsAOld_d[id * N0 + i].w = (1.0 + gamma) * m - gamma * Symplex_d[(im) * (Nst * N0) + id * N0 + i].m;
						elementsAOld_d[id * N0 + i].y = (1.0 + gamma) * e - gamma * Symplex_d[(im) * (Nst * N0) + id * N0 + i].e;
						elementsBOld_d[id * N0 + i].y = (1.0 + gamma) * w - gamma * Symplex_d[(im) * (Nst * N0) + id * N0 + i].w;
if(i == 0) printf("expand\n");
//printf("expand %d %.20g %.20g %.20g %.20g %.20g\n", i, elementsTOld_d[id * N0 + i].z, elementsTOld_d[id * N0 + i].x, elementsAOld_d[id * N0 + i].w, elementsAOld_d[id * N0 + i].y, elementsBOld_d[id * N0 + i].y);
					}
					else if(2.0 * elementsP_d[id].z < Symplex_d[im2 * (Nst * N0) + id * N0].f){
						//replace
if(i == 0) printf("replace\n");
//printf("replace %d %.20g %.20g %.20g %.20g %.20g\n", i, elementsTOld_d[id * N0 + i].z, elementsTOld_d[id * N0 + i].x, elementsAOld_d[id * N0 + i].w, elementsAOld_d[id * N0 + i].y, elementsBOld_d[id * N0 + i].y);
					}
					else{
						//contract
						if(2.0 * elementsP_d[id].z < Symplex_d[im * (Nst * N0) + id * N0].f){
							elementsTOld_d[id * N0 + i].z = beta * P + (1.0 - beta) * Symplex_d[(N0 * Ne + 1) * (Nst * N0) + id * N0 + i].P;
							elementsTOld_d[id * N0 + i].x = beta * T + (1.0 - beta) * Symplex_d[(N0 * Ne + 1) * (Nst * N0) + id * N0 + i].T;
							elementsAOld_d[id * N0 + i].w = beta * m + (1.0 - beta) * Symplex_d[(N0 * Ne + 1) * (Nst * N0) + id * N0 + i].m;
							elementsAOld_d[id * N0 + i].y = beta * e + (1.0 - beta) * Symplex_d[(N0 * Ne + 1) * (Nst * N0) + id * N0 + i].e;
							elementsBOld_d[id * N0 + i].y = beta * w + (1.0 - beta) * Symplex_d[(N0 * Ne + 1) * (Nst * N0) + id * N0 + i].w;
if(i == 0) printf("contract1\n");
//printf("contract1 %d %.20g %.20g %.20g %.20g %.20g\n", i, elementsTOld_d[id * N0 + i].z, elementsTOld_d[id * N0 + i].x, elementsAOld_d[id * N0 + i].w, elementsAOld_d[id * N0 + i].y, elementsBOld_d[id * N0 + i].y);
						}
						else{
							elementsTOld_d[id * N0 + i].z = beta * P + (1.0 - beta) * Symplex_d[(im) * (Nst * N0) + id * N0 + i].P;
							elementsTOld_d[id * N0 + i].x = beta * T + (1.0 - beta) * Symplex_d[(im) * (Nst * N0) + id * N0 + i].T;
							elementsAOld_d[id * N0 + i].w = beta * m + (1.0 - beta) * Symplex_d[(im) * (Nst * N0) + id * N0 + i].m;
							elementsAOld_d[id * N0 + i].y = beta * e + (1.0 - beta) * Symplex_d[(im) * (Nst * N0) + id * N0 + i].e;
							elementsBOld_d[id * N0 + i].y = beta * w + (1.0 - beta) * Symplex_d[(im) * (Nst * N0) + id * N0 + i].w;
if(i == 0) printf("contract2\n");
//printf("contract2 %d %.20g %.20g %.20g %.20g %.20g\n", i, elementsTOld_d[id * N0 + i].z, elementsTOld_d[id * N0 + i].x, elementsAOld_d[id * N0 + i].w, elementsAOld_d[id * N0 + i].y, elementsBOld_d[id * N0 + i].y);
						}
					}
				}
			}
		}
	}
}

//use Jacoby mass
//EE = -1: no change
//EE = 0: first step
//EE = 4: DEMCMC
//EE = 5: ADAGRAD
//EE = 10 Refine
__global__ void modifyElementsJ2(curandState *random_d, double4 *x4_d, double4 *v4_d, double4 *spin_d, double4 *elementsA_d, double4 *elementsB_d, double4 *elementsT_d, double4 *elementsSpin_d, double4 *elementsAOld_d, double4 *elementsBOld_d, double4 *elementsTOld_d, double4 *elementsSpinOld_d, elements10 *elementsL_d, double4 *elementsP_d, int4 *elementsI_d, int2 *elementsC_d, double2 *Msun_d, double time, int *N_d, int Nst, int ittv, int mcmcNE, int mcmcRestart, int EE){
#if USE_RANDOM == 1

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;
	curandState random;	

	if(id < Nst){
		elementsC_d[0].x = 0;

		int N0 = N_d[0];
		int ne = mcmcNE;
		double Msun = Msun_d[id].x;

		double eps = 0.0; //range of epsilon, random modification range
		//double eps = 1.0e-6; //range of epsilon, random modification range
		double eb = 0.05;    //range of e in stretch move
		double sc = 0.0;    //scaling factor for external update lengths


		int jjd = id % (ne * N0 + 1);   // map to 0 - 35

		double z = elementsP_d[id].z;
		if(ittv % (10 * MCMC_NQ) == 0){
			z =1.0;
			eb = 0.0;
		}

//use this for decoupled chains, pure MCMC with a covariance matrix
 #if MCMC_NCOV > 0
		z = 0;
		eps = 0.0;
		sc = elementsP_d[id].w;  //scaling factor
 #endif
		if(EE == 0 || EE == -1){
			z = 0.0;
			eps = 0.0;
			eb = 0.0;
		}
		if(EE == 5){
			//update only one parameter per chain, used for gradient calculation
			z = 0;
			eps = 0.0;
			eb = 0.0;
		}


		int st0 = id;
		int st1 = elementsI_d[id].y;
		int st2 = elementsI_d[id].z;
		
		//Remove outliers, but do that only in burn in phase
		//move st0 to st1
		/*
		if(ittv % (10 * MCMC_NQ) == 5){
			if(elementsP_d[st0].x > 2.0 * elementsP_d[st1].x){
				st2 = st0;
				z = 0.99;
			}
		}
		*/

		double mJ = 0.0;
		double mJ0 = 0.0;
		random = random_d[id];	
		
		for(int ii = 0; ii < N0; ++ii){

			//loop around planets 
			double a = elementsAOld_d[st0 * N0 + ii].x;		//semi major axis
			double e = elementsAOld_d[st0 * N0 + ii].y;		//eccentricity
			double inc = elementsAOld_d[st0 * N0 + ii].z;		//inclination
			double m = elementsAOld_d[st0 * N0 + ii].w;		//mass
			double Omega = elementsBOld_d[st0 * N0 + ii].x;		//longitude of ascending node
			double w = elementsBOld_d[st0 * N0 + ii].y;		//argument of periapsis
			double M = elementsBOld_d[st0 * N0 + ii].z;		//mean anomaly
			double r = elementsBOld_d[st0 * N0 + ii].w;		//radius
			double P = elementsTOld_d[st0 * N0 + ii].z;		//period
			double T = elementsTOld_d[st0 * N0 + ii].x;		//time of first transit
			double Sy = elementsSpinOld_d[st0 * N0 + ii].y;		//Spiny
			if(EE == 5 || EE == -1){
				P = fmax(P, 1.0e-16);
				e = fmax(e, 1.0e-16);
				m = fmax(m, 1.0e-16);
			}

			if(EE == 10){
			//Adjust M iteratively
				a = elementsA_d[st0 * N0 + ii].x;		//semi major axis
				e = elementsA_d[st0 * N0 + ii].y;		//eccentricity
				inc = elementsA_d[st0 * N0 + ii].z;		//inclination
				m = elementsA_d[st0 * N0 + ii].w;		//mass
				Omega = elementsB_d[st0 * N0 + ii].x;		//longitude of ascending node
				w = elementsB_d[st0 * N0 + ii].y;		//argument of periapsis
				M = elementsB_d[st0 * N0 + ii].z;		//mean anomaly
				r = elementsB_d[st0 * N0 + ii].w;		//radius
				P = elementsT_d[st0 * N0 + ii].z;		//period
				T = elementsT_d[st0 * N0 + ii].x;		//time of first transit
				Sy = elementsSpin_d[st0 * N0 + ii].y;		//Spiny
			}
//printf("Modify %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g| %g %d\n", ii, m, r, P, e, inc, Omega, w, T, a, M, z, ne);

			double rd;
			if(EE == 0){
				if(ne > 0){
					rd = curand_normal(&random);
					P += rd * elementsL_d[st0 * N0 + ii].P;       //scale to standard deviation of tuning length
				}
				if(ne > 1){
					rd = curand_normal(&random);
					T += rd * elementsL_d[st0 * N0 + ii].T;
				}
				if(ne > 2){
					rd = curand_normal(&random);
					m += rd * elementsL_d[st0 * N0 + ii].m;
				}
				if(ne > 3){
					rd = curand_normal(&random);
					e += rd * elementsL_d[st0 * N0 + ii].e;
				}
				if(ne > 4){
					rd = curand_normal(&random);
					w += rd * elementsL_d[st0 * N0 + ii].w;
				}
				if(ne > 5){
					rd = curand_normal(&random);
					inc += rd * elementsL_d[st0 * N0 + ii].inc;
					rd = curand_normal(&random);
					Omega += rd * elementsL_d[st0 * N0 + ii].O;
				}
				if(ne > 7){
					//rd = curand_normal(&random);
					//r += rd * elementsL_d[st0 * N0 + ii].r;
					rd = curand_normal(&random);
					Sy += rd * elementsL_d[st0 * N0 + ii].r;
				}
			}
			double P0 = P;
			double P1 = elementsTOld_d[st1 * N0 + ii].z;		
			double P2 = elementsTOld_d[st2 * N0 + ii].z;		

			double m0 = m;		
			double m1 = elementsAOld_d[st1 * N0 + ii].w;		
			double m2 = elementsAOld_d[st2 * N0 + ii].w;		

			if(ne > 0){
				rd = curand_uniform(&random) * 2.0 * eb;
				P += z * (1.0 - eb + rd) * (P1 - P2);
				rd = curand_normal(&random) * eps * elementsL_d[st0 * N0 + ii].P;
				P += P0 * rd;

				P += elementsL_d[st0 * N0 + ii].P * sc;
				if(EE == 5 && jjd / ne == ii && jjd % ne == 0) {
					 P += elementsL_d[st0 * N0 + ii].P;
//printf("GRAD P %d %d %d %d %d %.20g\n", id, jjd, ii, jjd / ne, jjd % ne, P);
				}
			}
//printf("ModifyA %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g | %g\n", ii, m, r, P, e, inc, Omega, w, T, a, z);

			if(ne > 2){
				//modify m
				rd = curand_uniform(&random) * 2.0 * eb;
				m += z * (1.0 - eb + rd) * (m1 - m2);
				rd = curand_normal(&random) * eps * elementsL_d[st0 * N0 + ii].m;
				m += m0 * rd;

				m += elementsL_d[st0 * N0 + ii].m * sc;
				if(EE == 5 && jjd / ne == ii && jjd % ne == 2) {
					 m += elementsL_d[st0 * N0 + ii].m;
//printf("GRAD m %d %d %d %d %d %.20g\n", id, jjd, ii, jjd / ne, jjd % ne, m);
				}
			}
			//Jacoby mass
			mJ += m;
			double mu = def_ksq * (Msun + mJ);
//printf("Modify m %d %.20g %.20g %.20g %.20g | %.20g %.20g %g\n", st0 * N0 + ii, m0, m1, m2, m, mJ, mu, z);

			volatile double a3 = P * P * dayUnit * dayUnit * mu / (4.0 * M_PI * M_PI);
			a = cbrt(a3);
//printf("Modify a %d %.30g %.30g %.30g %.30g %.30g\n", st0 * N0 + ii, P, mu, a, P * P * dayUnit * dayUnit * mu, a3);
	
			double e0 = e;
			double e1 = elementsAOld_d[st1 * N0 + ii].y;		//eccentricity
			double e2 = elementsAOld_d[st2 * N0 + ii].y;		//eccentricity


			double w0 = w;
			double w1 = elementsBOld_d[st1 * N0 + ii].y;		//eccentricity
			double w2 = elementsBOld_d[st2 * N0 + ii].y;		//eccentricity

/*
			double xx0 = sqrt(e0) * cos(w0);
			double xx1 = sqrt(e1) * cos(w1);
			double xx2 = sqrt(e2) * cos(w2);
			double yy0 = sqrt(e0) * sin(w0);
			double yy1 = sqrt(e1) * sin(w1);
			double yy2 = sqrt(e2) * sin(w2);
			double xx = xx0;
			double yy = yy0;

			if(ne > 3){
				xx += z * (xx1 - xx2);
				yy += z * (yy1 - yy2);
				e = xx * xx + yy * yy;
			}
	
			if(ne > 4){
				w = acos(xx / sqrt(e));
			}
			if(yy < 0.0) w = 2.0 * M_PI - w;


*/
			if(ne > 3){
				rd = curand_uniform(&random) * 2.0 * eb;
				e += z * (1.0 - eb + rd) * (e1 - e2);
				rd = curand_normal(&random) * eps * elementsL_d[st0 * N0 + ii].e;
				e += e0 * rd;

				e += elementsL_d[st0 * N0 + ii].e * sc;
				if(EE == 5 && jjd / ne == ii && jjd % ne == 3) {
					 e += elementsL_d[st0 * N0 + ii].e;
//printf("GRAD e %d %d %d %d %d %.20g\n", id, jjd, ii, jjd / ne, jjd % ne, e);
				}
			}
			if(ne > 4){
				rd = curand_uniform(&random) * 2.0 * eb;
				w += z * (1.0 - eb + rd) * (w1 - w2);
				rd = curand_normal(&random) * eps * elementsL_d[st0 * N0 + ii].w;
				w += w0 * rd;

				w += elementsL_d[st0 * N0 + ii].w * sc;
				if(EE == 5 && jjd / ne == ii && jjd % ne == 4) {
					 w += elementsL_d[st0 * N0 + ii].w;
//printf("GRAD w %d %d %d %d %d %.20g\n", id, jjd, ii, jjd / ne, jjd % ne, w);
				}
			}


			if(ne > 5){
				double inc0 = inc;
				double inc1 = elementsAOld_d[st1 * N0 + ii].z;		//eccentricity
				double inc2 = elementsAOld_d[st2 * N0 + ii].z;		//eccentricity
				rd = curand_uniform(&random) * 2.0 * eb;
				inc += z * (1.0 - eb + rd) * (inc1 - inc2);
				rd = curand_normal(&random) * eps * elementsL_d[st0 * N0 + ii].inc;
				inc += inc0 * rd;
	
				double Omega0 = Omega;
				double Omega1 = elementsBOld_d[st1 * N0 + ii].x;		//eccentricity
				double Omega2 = elementsBOld_d[st2 * N0 + ii].x;		//eccentricity
				rd = curand_uniform(&random) * 2.0 * eb;
				Omega += z * (1.0 - eb + rd) * (Omega1 - Omega2);
				rd = curand_normal(&random) * eps * elementsL_d[st0 * N0 + ii].O;
				Omega += Omega0 * rd;

			}
		
			/*	
			double r0 = r;		
			double r1 = elementsBOld_d[st1 * N0 + ii].w;		
			double r2 = elementsBOld_d[st2 * N0 + ii].w;		
			if(ne > 7){
				//modify r
				rd = curand_uniform(&random) * 2.0 * eb;
				r += z * (1.0 - eb + rd) * (r1 - r2);
				rd = curand_normal(&random) * eps * elementsL_d[st0 * N0 + ii].r;
				r += r0 * rd;

				r += elementsL_d[st0 * N0 + ii].r * sc;
			}
			*/
			double Sy0 = r;		
			double Sy1 = elementsSpinOld_d[st1 * N0 + ii].y;		
			double Sy2 = elementsSpinOld_d[st2 * N0 + ii].y;		
			if(ne > 7){
				//modify Sy
				rd = curand_uniform(&random) * 2.0 * eb;
				Sy += z * (1.0 - eb + rd) * (Sy1 - Sy2);
				rd = curand_normal(&random) * eps * elementsL_d[st0 * N0 + ii].r;
				Sy += Sy0 * rd;

				Sy += elementsL_d[st0 * N0 + ii].r * sc;
			}

			if(e <= 0) w = 0;
//printf("ModifyB %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g | %g\n", ii, m, r, P, e, inc, Omega, w, T, a, z);
	
			double T0 = T;
			double T1 = elementsTOld_d[st1 * N0 + ii].x;
			double T2 = elementsTOld_d[st2 * N0 + ii].x;

			if(ne > 1){
				rd = curand_uniform(&random) * 2.0 * eb;
				T += z * (1.0 - eb + rd) * (T1 - T2);
				rd = curand_normal(&random) * eps * elementsL_d[st0 * N0 + ii].T;
				T += T0 * rd;

				T += elementsL_d[st0 * N0 + ii].T * sc;
				//M += z * (M1 - M2);
				if(EE == 5 && jjd / ne == ii && jjd % ne == 1) {
					 T += elementsL_d[st0 * N0 + ii].T;
//printf("GRAD T %d %d %d %d %d %.20g\n", id, jjd, ii, jjd / ne, jjd % ne, T);
				}
			}
			
			for(int i = 0; i < 1000; ++i){
				if(T < time) T += P;
				else break;
			}

			for(int i = 0; i < 1000; ++i){
				if(T > time + P) T -= P;
				else break;
			}
//printf("ModifyT %d %.20g %.20g %.20g %.20g %.20g %g\n", ii, T0, T1, T2, T, P, M);

			double nu = M_PI * 0.5 - w;	//true anomaly at first transit
			double ee2 = e * e;
			double ee4 = ee2 * ee2;
			//compute Mean Anomaly of the first transit
			double Mt = nu - 2.0 * e * sin(nu) + (3.0 * 0.25 * ee2 + 0.125 * ee4) * sin(2.0 * nu) - 1.0 / 3.0 * e * ee2 * sin(3.0 * nu) + 5.0/32.0 * ee4 * sin(4.0 * nu);
			M = -(T - time) / P * 2.0 * M_PI + Mt;
			M = fmod(M, 2.0 * M_PI);
			if(M < 0.0) M+= 2.0 * M_PI;

			//first transit time, measured
/*			if(EE == 10){
				double Tobs = elementsT_d[id * N0 + ii].y;
				double Pobs = elementsT_d[id * N0 + ii].w;
				double dM = -(T - Tobs) / P * 2.0 * M_PI;					    //M(T + deltaT)
				M += dM;
				double da = cbrt(mu / (4.0 * M_PI * M_PI * P * dayUnit)) * 2.0 / 3.0 * (P - Pobs) * dayUnit;  //first order taylor expansion a(P + deltaP) = a + da/dP * deltaP
				double da2 = -cbrt(mu / (4.0 * M_PI * M_PI * P * P * P * P * dayUnit * dayUnit * dayUnit * dayUnit)) * 1.0 / 9.0 * (P - Pobs) * (P - Pobs) * dayUnit * dayUnit; //second order
if(id == 0) printf("TT %d T %.20g P %.20g M %g %g %g %g %.20g %.20g %.20g %.20g %.20g %.20g\n", id * N0 + ii, T, P, M, dM, da, da2, Tobs, Pobs, a, a + da, a + da + da2, P - Pobs);
				a += da + da2;
				
			}
*/
			inc = fmod(inc, 2.0*M_PI);
			Omega = fmod(Omega, 2.0*M_PI);
			w = fmod(w, 2.0*M_PI);
	
			if(inc < 0.0) inc = 2.0 * M_PI + inc;
			if(inc >= 2.0 * M_PI) inc = inc - 2.0 * M_PI;

			if(Omega < 0.0) Omega = 2.0 * M_PI + Omega;
			if(Omega >= 2.0 * M_PI) Omega = Omega - 2.0 * M_PI;

			if(w < 0.0) w = 2.0 * M_PI + w;
			if(w >= 2.0 * M_PI) w = w - 2.0 * M_PI;

			if(M < 0.0) M = 2.0 * M_PI + M;
			if(M >= 2.0 * M_PI) M = M - 2.0 * M_PI;

			if(mcmcRestart == 1 && EE < 3){
				//a = elementsAOld_d[st0 * N0 + ii].x;		//semi major axis
				e = elementsAOld_d[st0 * N0 + ii].y;		//eccentricity
				inc = elementsAOld_d[st0 * N0 + ii].z;		//inclination
				m = elementsAOld_d[st0 * N0 + ii].w;		//mass
				Omega = elementsBOld_d[st0 * N0 + ii].x;	//longitude of ascending node
				w = elementsBOld_d[st0 * N0 + ii].y;		//argument of periapsis
				//M = elementsBOld_d[st0 * N0 + ii].z;		//mean anomaly
				r = elementsBOld_d[st0 * N0 + ii].w;		//radius
				P = elementsTOld_d[st0 * N0 + ii].z;		//periode
				T = elementsTOld_d[st0 * N0 + ii].x;		//time of first transit
				Sy = elementsSpinOld_d[st0 * N0 + ii].y;	//Spiny
			
				//Jacoby mass
				mJ0 += m;
				mu = def_ksq * (Msun + mJ0);

				//a
				volatile double a3 = P * P * dayUnit * dayUnit * mu / (4.0 * M_PI * M_PI);
				a = cbrt(a3);
				//M
				double nu = M_PI * 0.5 - w;	//true anomaly at first transit
				double ee2 = e * e;
				double ee4 = ee2 * ee2;
				//compute Mean Anomaly of the first transit
				double Mt = nu - 2.0 * e * sin(nu) + (3.0 * 0.25 * ee2 + 0.125 * ee4) * sin(2.0 * nu) - 1.0 / 3.0 * e * ee2 * sin(3.0 * nu) + 5.0/32.0 * ee4 * sin(4.0 * nu);
				M = -(T - time) / P * 2.0 * M_PI + Mt;
				M = fmod(M, 2.0 * M_PI);
				if(M < 0.0) M+= 2.0 * M_PI;

//printf("ModifyR %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g | %g\n", ii, m, r, P, e, inc, Omega, w, T, a, M, z);
			}


			//set acceptance probability to zero for inpossible parameters
			if(e < 0){
				e = 0.0;
				elementsP_d[id].y = 1.0e100;

			}
			if(e > 0.3){
				e = 0.4;
				elementsP_d[id].y = 1.0e100;

			}
			if(m < 0){
				m = 0.0;
				elementsP_d[id].y = 1.0e100;

			}
			if(r < 0){
				r = 0.0;
				elementsP_d[id].y = 1.0e100;

			}
			if(a < 0){
				a = 1.0;
				elementsP_d[id].y = 1.0e100;

			}

			elementsA_d[st0 * N0 + ii].x = a;
			elementsA_d[st0 * N0 + ii].y = e;
			elementsA_d[st0 * N0 + ii].z = inc;
			elementsA_d[st0 * N0 + ii].w = m;
			elementsB_d[st0 * N0 + ii].x = Omega;
			elementsB_d[st0 * N0 + ii].y = w;
			elementsB_d[st0 * N0 + ii].z = M;
			elementsB_d[st0 * N0 + ii].w = r;
			elementsT_d[st0 * N0 + ii].z = P;
			elementsT_d[st0 * N0 + ii].x = T;
			elementsSpin_d[st0 * N0 + ii].y = Sy;
//printf("MJ %d %.20g %d %d %d\n", st0 * N0 + ii, a, st0, st1, st2);
//printf("ModifyC %d %d %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g| %g\n", id, jjd, ii, m, r, P, e, inc, Omega, w, T, a, M, z);

			//Convert to Cartesian Coordinates
			
			//Eccentric Anomaly
			double E = M + e * 0.5;
			double Eold = E;
			for(int j = 0; j < 32; ++j){
				E = E - (E - e * sin(E) - M) / (1.0 - e * cos(E));
				if(fabs(E - Eold) < 1.0e-15) break;
				Eold = E;
			}

			double cw = cos(w);
			double sw = sin(w);
			double cOmega = cos(Omega);
			double sOmega = sin(Omega);
			double ci = cos(inc);
			double si = sin(inc);

			double Px = cw * cOmega - sw * ci * sOmega;
			double Py = cw * sOmega + sw * ci * cOmega;
			double Pz = sw * si;

			double Qx = -sw * cOmega - cw * ci * sOmega;
			double Qy = -sw * sOmega + cw * ci * cOmega;
			double Qz = cw * si;

			double cE = cos(E);
			double sE = sin(E);
			double t1 = a * (cE - e);
			double t2 = a * sqrt(1.0 - e * e) * sE;

			double4 x4i, v4i;

			x4i.x =  t1 * Px + t2 * Qx;
			x4i.y =  t1 * Py + t2 * Qy;
			x4i.z =  t1 * Pz + t2 * Qz;
			x4i.w = m;

			double t0 = 1.0 / (1.0 - e * cE) * sqrt(mu / a);
			t1 = -sE;
			t2 = sqrt(1.0 - e * e) * cE;

			v4i.x = t0 * (t1 * Px + t2 * Qx);
			v4i.y = t0 * (t1 * Py + t2 * Qy);
			v4i.z = t0 * (t1 * Pz + t2 * Qz);
			v4i.w = r;

			x4_d[st0 * N0 + ii] = x4i;
			v4_d[st0 * N0 + ii] = v4i;
			spin_d[st0 * N0 + ii].y = Sy;

//printf("ModifyD %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g\n", ii, x4i.w, v4i.w, x4i.x, x4i.y, x4i.z, v4i.x, v4i.y, v4i.z);

		}// end of loop around planets
		random_d[id] = random;
	}
#endif
}

//quadratic estimation for period
__global__ void modifyElementsPQ(curandState *random_d, double4 *x4_d, double4 *v4_d, double4 *elementsA_d, double4 *elementsB_d, double4 *elementsAOld2_d, double4 *elementsBOld2_d, elements10 *elementsL_d, double4 *elementsP_d, int4 *elementsI_d, int2 *elementsC_d, double2 *Msun_d, double time, int *N_d, int Nst, int ittv, int mcmcNE, int mcmcRestart, int EE, double ff, int AA){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	if(id < Nst){
		elementsC_d[0].x = 0;
		elementsP_d[id].y = 1.0e400;	//not update the Old variables in the ttvstep

		int N0 = N_d[0];

		int st0 = id % (Nst / 3);

//printf("%d %d Px %g Pz %g %d\n", id, st0, elementsP_d[id].x, elementsP_d[id].z, EE);	
		double Msun = Msun_d[st0].x;
		double mJ = 0.0;	
		for(int ii = 0; ii < N0; ++ii){
			double a = elementsAOld2_d[st0 * N0 + ii].x;		//semi major axis
			double e = elementsAOld2_d[st0 * N0 + ii].y;		//eccentricity
			double inc = elementsAOld2_d[st0 * N0 + ii].z;		//inclination
			double m = elementsAOld2_d[st0 * N0 + ii].w;		//inclination
			double Omega = elementsBOld2_d[st0 * N0 + ii].x;		//longitude of ascending node
			double w = elementsBOld2_d[st0 * N0 + ii].y;		//argument of periapsis
			double M = elementsBOld2_d[st0 * N0 + ii].z;		//mean anomaly
			double r = elementsBOld2_d[st0 * N0 + ii].w;		//radius
			double x1, x2, x3;
				
			if(AA == 0){		
				//x1 = a - elementsL_d[st0 * N0 + ii].a * ff;
				//x2 = a;
				//x3 = a + elementsL_d[st0 * N0 + ii].a * ff;
				x1 = a - 5.0e-7;
				x2 = a;
				x3 = a + 5.0e-7;
			}
			if(AA == 1){		
				x1 = M - elementsL_d[st0 * N0 + ii].M * ff;
				x2 = M;
				x3 = M + elementsL_d[st0 * N0 + ii].M * ff;
			}
			if(ii != EE){
				if(AA == 0){
					x1 = a;
					x2 = a;
					x3 = a;
				}
				if(AA == 1){
					x1 = M;
					x2 = M;
					x3 = M;
				}
			}

			double pOld = elementsP_d[st0].z;
			elementsP_d[id].z = pOld;
			double x;

			if(id / (Nst / 3) == 0){
				x = x1;
//printf("a1 %d %.20g %.20g\n", id, a, elementsL_d[st0 * N0 + ii].a);
			}
			if(id / (Nst / 3) == 1){
				x = x2;
//printf("a2 %d %.20g %.20g\n", id, a, elementsL_d[st0 * N0 + ii].a);
				elementsP_d[id].z = pOld;
			}
			if(id / (Nst / 3) == 2){
				x = x3;
//printf("a3 %d %.20g %.20g\n", id, a, elementsL_d[st0 * N0 + ii].a);
				elementsP_d[id].z = pOld;
			}

			if(AA == 0) a = x;
			if(AA == 1) M = x;
			elementsA_d[id * N0 + ii].x = a;
			elementsA_d[id * N0 + ii].y = e;
			elementsA_d[id * N0 + ii].z = inc;
			elementsA_d[id * N0 + ii].w = m;
			elementsB_d[id * N0 + ii].x = Omega;
			elementsB_d[id * N0 + ii].y = w;
			elementsB_d[id * N0 + ii].z = M;
			elementsB_d[id * N0 + ii].w = r;

			
			//Eccentric Anomaly
			double E = M + e * 0.5;
			double Eold = E;
			for(int j = 0; j < 32; ++j){
				E = E - (E - e * sin(E) - M) / (1.0 - e * cos(E));
				if(fabs(E - Eold) < 1.0e-15) break;
				Eold = E;
			}

			double cw = cos(w);
			double sw = sin(w);
			double cOmega = cos(Omega);
			double sOmega = sin(Omega);
			double ci = cos(inc);
			double si = sin(inc);

			double Px = cw * cOmega - sw * ci * sOmega;
			double Py = cw * sOmega + sw * ci * cOmega;
			double Pz = sw * si;

			double Qx = -sw * cOmega - cw * ci * sOmega;
			double Qy = -sw * sOmega + cw * ci * cOmega;
			double Qz = cw * si;

			double cE = cos(E);
			double sE = sin(E);
			double t1 = a * (cE - e);
			double t2 = a * sqrt(1.0 - e * e) * sE;

			double4 x4i, v4i;

			x4i.x =  t1 * Px + t2 * Qx;
			x4i.y =  t1 * Py + t2 * Qy;
			x4i.z =  t1 * Pz + t2 * Qz;
			x4i.w = m;
	
			mJ += m;
			double mu = def_ksq * (Msun + mJ);

			double t0 = 1.0 / (1.0 - e * cE) * sqrt(mu / a);
			t1 = -sE;
			t2 = sqrt(1.0 - e * e) * cE;

			v4i.x = t0 * (t1 * Px + t2 * Qx);
			v4i.y = t0 * (t1 * Py + t2 * Qy);
			v4i.z = t0 * (t1 * Pz + t2 * Qz);
			v4i.w = r;

			x4_d[id * N0 + ii] = x4i;
			v4_d[id * N0 + ii] = v4i;
		}
	}
}

//quadratic estimation for period
__global__ void modifyElementsPQ2(curandState *random_d, double4 *x4_d, double4 *v4_d, double4 *elementsA_d, double4 *elementsB_d, double4 *elementsAOld2_d, double4 *elementsBOld2_d, elements10 *elementsL_d, double4 *elementsP_d, int4 *elementsI_d, int2 *elementsC_d, double2 *Msun_d, double time, int *N_d, int Nst, int ittv, int mcmcNE, int mcmcRestart, int EE, double ff, int AA){

#if USE_RANDOM == 1
	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	if(id < Nst){
		elementsC_d[0].x = 0;


		int N0 = N_d[0];
		int st0 = id % (Nst / 3);
		int st1 = st0 + (Nst / 3);
		int st2 = st1 + (Nst / 3);

		if(ff > 0.0 || EE < N0 - 1){
			elementsP_d[id].y = 1.0e400;	//not update the Old variables in the ttvstep
		}
		else{
			curandState random;
			random = random_d[id];
			double rd = curand_uniform(&random);
			if(id / (Nst / 3) == 0){
				elementsP_d[st0].y = rd;
				elementsP_d[st1].y = rd;
				elementsP_d[st2].y = rd;
			}
			random_d[id] = random;
if(id == 0) printf("random %d %g\n", id, rd);
		}
		

//printf("%d %d %d %d Px %g Pz %g %d\n", id, st0, st1, st2, elementsP_d[id].x, elementsP_d[id].z, EE);

		double Msun = Msun_d[st0].x;
		double mJ = 0.0;	
		for(int ii = 0; ii < N0; ++ii){
			double a = elementsAOld2_d[id * N0 + ii].x;		//semi major axis
			double e = elementsAOld2_d[id * N0 + ii].y;		//eccentricity
			double inc = elementsAOld2_d[id * N0 + ii].z;	//inclination
			double m = elementsAOld2_d[id * N0 + ii].w;		//inclination
			double Omega = elementsBOld2_d[id * N0 + ii].x;	//longitude of ascending node
			double w = elementsBOld2_d[id * N0 + ii].y;		//argument of periapsis
			double M = elementsBOld2_d[id * N0 + ii].z;		//mean anomaly
			double r = elementsBOld2_d[id * N0 + ii].w;		//radius

			double x1, x2, x3;
			double p1, p2, p3;

			if(AA == 0){
				//a
				x1 = elementsAOld2_d[st0 * N0 + ii].x; 
				x2 = elementsAOld2_d[st1 * N0 + ii].x;
				x3 = elementsAOld2_d[st2 * N0 + ii].x;
			}
			if(AA == 1){
				//M
				x1 = elementsBOld2_d[st0 * N0 + ii].z; 
				x2 = elementsBOld2_d[st1 * N0 + ii].z;
				x3 = elementsBOld2_d[st2 * N0 + ii].z;
			}

		//	double am = a2;
		
			p1 = elementsP_d[st0].x;
			p2 = elementsP_d[st1].x;
			p3 = elementsP_d[st2].x;

			//quadratic estimation
			//p(a) = b0 + b1 * ( a - a1) + b2* (a - a1)* (a - a2)


			double b0 = p1;
			double b1 = (p2 - p1) / (x2 - x1);
			double b2 = 1.0 / (x3 - x2) * ((p3 - p1) / (x3 - x1) - b1);

			double xx = (x1 + x2) * 0.5 - b1 / (2.0 * b2);
if(!(p1 > p2 && p2 < p3)){
	if(p1 <= p2 && p1 <= p3) xx = x1;
	if(p2 <= p3 && p2 <= p1) xx = x2;
	if(p3 <= p1 && p3 <= p2) xx = x3;

if(p1 != p2) printf("****** %d %g %g %g %g %g %g %g\n", ii, p1, p2, p3, b0, b1, b2, xx);
}
//printf("a %d %.20g %.20g %.20g %.20g %.20g %.20g | %g %g %g %.20g\n", id * N0 + ii, x1, x2, x3, p1, p2, p3, b0, b1, b2, xx);

	
			x1 = xx;
			x2 = xx;
			x3 = xx;
			
			if(ii != EE){
				if(AA == 0){
					x1 = a;
					x2 = a;
					x3 = a;
				}
				if(AA == 1){
					x1 = M;
					x2 = M;
					x3 = M;
				}
			}

			double deltaa = 0.0;
			double deltaM = 0.0;
			double deltaa1 = 0.0;
			double deltaM1 = 0.0;
			double deltaa2 = 0.0;
			double deltaM2 = 0.0;
			double deltaa3 = 0.0;
			double deltaM3 = 0.0;

			int EEE = EE + 1;
			if (EE == N0 - 1) EEE = 0;

			if(ii == EEE){
				if(AA == 0 && EE < N0 - 1){
					//deltaa1 =-elementsL_d[st0 * N0 + ii].a * ff;
					//deltaa2 = 0.0;
					//deltaa3 = elementsL_d[st0 * N0 + ii].a * ff;
					if(EE == 0){
						deltaa1 =-5.0e-7;
						deltaa2 = 0.0;
						deltaa3 = 5.0e-7;
					}
					if(EE == 1){
						deltaa1 =-5.0e-6;
						deltaa2 = 0.0;
						deltaa3 = 5.0e-6;
					}
					if(EE == 2){
						deltaa1 =-5.0e-6;
						deltaa2 = 0.0;
						deltaa3 = 5.0e-6;
					}
					if(EE == 3){
						deltaa1 =-1.0e-5;
						deltaa2 = 0.0;
						deltaa3 = 1.0e-5;
					}
					if(EE == 4){
						deltaa1 =-5.0e-6;
						deltaa2 = 0.0;
						deltaa3 = 5.0e-6;
					}
					if(EE == 5){
						deltaa1 =-5.0e-6;
						deltaa2 = 0.0;
						deltaa3 = 5.0e-6;
					}
					if(EE == 6){
						deltaa1 =-5.0e-6;
						deltaa2 = 0.0;
						deltaa3 = 5.0e-6;
					}
					deltaa1 *= ff;
					deltaa2 *= ff;
					deltaa3 *= ff;
				}
				else{
					//deltaM1 =-elementsL_d[st0 * N0 + ii].M * ff;
					//deltaM2 = 0.0;
					//deltaM3 = elementsL_d[st0 * N0 + ii].M * ff;
					if(EE == 0){
						deltaM1 =-0.01;
						deltaM2 = 0.0;
						deltaM3 = 0.01;
					}
					if(EE == 1){
						deltaM1 =-0.01;
						deltaM2 = 0.0;
						deltaM3 = 0.01;
					}
					if(EE == 2){
						deltaM1 =-0.01;
						deltaM2 = 0.0;
						deltaM3 = 0.01;
					}
					if(EE == 3){
						deltaM1 =-0.01;
						deltaM2 = 0.0;
						deltaM3 = 0.01;
					}
					if(EE == 4){
						deltaM1 =-0.01;
						deltaM2 = 0.0;
						deltaM3 = 0.01;
					}
					if(EE == 5){
						deltaM1 =-0.01;
						deltaM2 = 0.0;
						deltaM3 = 0.01;
					}
					if(EE == 6){
						deltaM1 =-0.01;
						deltaM2 = 0.0;
						deltaM3 = 0.01;
					}
					deltaM1 *= ff;
					deltaM2 *= ff;
					deltaM3 *= ff;
				}
			}

			double x;

			if(id / (Nst / 3) == 0){
				x = x1;
				deltaa = deltaa1;
				deltaM = deltaM1;
//printf("a1 %d %d %.20g %.20g %g %g %g\n", ii, id, x, elementsL_d[st0 * N0 + ii].a, ff, deltaa, deltaM);
			}
			if(id / (Nst / 3) == 1){
				x = x2;
				deltaa = deltaa2;
				deltaM = deltaM2;
//printf("a2 %d %d %.20g %.20g %g %g %g\n", ii, id, x, elementsL_d[st0 * N0 + ii].a, ff, deltaa, deltaM);
			}
			if(id / (Nst / 3) == 2){
				x = x3;
				deltaa = deltaa3;
				deltaM = deltaM3;
//printf("a3 %d %d %.20g %.20g %g %g %g\n", ii, id, x, elementsL_d[st0 * N0 + ii].a, ff, deltaa, deltaM);
			}

			if(AA == 0) a = x;
			if(AA == 1) M = x;

			a += deltaa;
			M += deltaM;

			elementsA_d[id * N0 + ii].x = a;
			elementsA_d[id * N0 + ii].y = e;
			elementsA_d[id * N0 + ii].z = inc;
			elementsA_d[id * N0 + ii].w = m;
			elementsB_d[id * N0 + ii].x = Omega;
			elementsB_d[id * N0 + ii].y = w;
			elementsB_d[id * N0 + ii].z = M;
			elementsB_d[id * N0 + ii].w = r;

			
			//Eccentric Anomaly
			double E = M + e * 0.5;
			double Eold = E;
			for(int j = 0; j < 32; ++j){
				E = E - (E - e * sin(E) - M) / (1.0 - e * cos(E));
				if(fabs(E - Eold) < 1.0e-15) break;
				Eold = E;
			}

			double cw = cos(w);
			double sw = sin(w);
			double cOmega = cos(Omega);
			double sOmega = sin(Omega);
			double ci = cos(inc);
			double si = sin(inc);

			double Px = cw * cOmega - sw * ci * sOmega;
			double Py = cw * sOmega + sw * ci * cOmega;
			double Pz = sw * si;

			double Qx = -sw * cOmega - cw * ci * sOmega;
			double Qy = -sw * sOmega + cw * ci * cOmega;
			double Qz = cw * si;

			double cE = cos(E);
			double sE = sin(E);
			double t1 = a * (cE - e);
			double t2 = a * sqrt(1.0 - e * e) * sE;

			double4 x4i, v4i;

			x4i.x =  t1 * Px + t2 * Qx;
			x4i.y =  t1 * Py + t2 * Qy;
			x4i.z =  t1 * Pz + t2 * Qz;
			x4i.w = m;
	
			mJ += m;
			double mu = def_ksq * (Msun + mJ);

			double t0 = 1.0 / (1.0 - e * cE) * sqrt(mu / a);
			t1 = -sE;
			t2 = sqrt(1.0 - e * e) * cE;

			v4i.x = t0 * (t1 * Px + t2 * Qx);
			v4i.y = t0 * (t1 * Py + t2 * Qy);
			v4i.z = t0 * (t1 * Pz + t2 * Qz);
			v4i.w = r;

			x4_d[id * N0 + ii] = x4i;
			v4_d[id * N0 + ii] = v4i;
		}
	}
#endif
}
__global__ void setJ_kernel(double4 *elementsP_d, int Nst){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	if(id < Nst){

		elementsP_d[id].x = elementsP_d[id].z;
	}

}
__global__ void setJ_kernel(curandState *random_d, double4 *elementsP_d, int4 *elementsI_d, int2 *elementsC_d, int Nst, int N0, double2 *Msun_d, double *elementsM_d, int ittv, int mcmcNE, int EE){

#if USE_RANDOM == 1
	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;
	curandState random;	

	int iT = id / (Nst / MCMC_NT);			//index of temperature in parallel tempering

	if(id < Nst){
		random = random_d[id];	
		if(EE < 3){		
			if(elementsC_d[id + MCMC_NT].x == 0){
				elementsP_d[iT].x = elementsP_d[id].z; //set pOld to the accepted p
			}
		}
		if(EE == 4){
//			elementsM_d[id] = Msun_d[id].x;
//			//update Msun
//			Msun_d[id].x = 0.089 + curand_normal(&random) * 0.012;
			
			elementsP_d[id].x = elementsP_d[id].z; //set pOld to the accepted p

			int st0 = id;
			int st1 = id;
			int st2 = id;
			while(st0 == id){			
				st0 = (int)(curand_uniform(&random) * Nst);
			}
			while(st1 == id){			
				st1 = (int)(curand_uniform(&random) * Nst);
			}
			while(st2 == id || st2 == st1){			
				st2 = (int)(curand_uniform(&random) * Nst);
			}
			elementsI_d[id].x = st0;
			elementsI_d[id].y = st1;
			elementsI_d[id].z = st2;
			//compute z
			//double sigma = 0.01;

			//double z = curand_normal(&random) * sigma;
			//double g0 = 2.38 / sqrt(2.0 * N0 * mcmcNE);
			//double gamma = g0 * (1.0 + z);
			int dd = mcmcNE * N0;//max(elementsI_d[0].w, 1);
			//int dd = mcmcNE;//max(elementsI_d[0].w, 1);
			//int dd = 1;//max(elementsI_d[0].w, 1);
			double gamma = 2.38 / sqrt(2.0 * dd);
			double w = elementsP_d[id].w;

			/*
			// ************************************************
			//adapt global acceptance rate
			if(elementsC_d[0].x < 0.2 * Nst) w *= 0.9;
			if(elementsC_d[0].x > 0.3 * Nst) w *= 1.111;
			// ************************************************
			*/
			// ************************************************
			//adapt acceptance rate for each walker individually
			if(ittv % 10 == 0 && ittv > 0){
				if(elementsC_d[id].y / N0 / 10.0 < 0.2) w *= 0.9; 
				if(elementsC_d[id].y / N0 / 10.0 > 0.3) w *= 1.111; 
				elementsC_d[id].y = 0;

			}
			// ************************************************

			elementsP_d[id].w = w;
if(id == 0) printf("z %d %g %d %g %d\n", id, gamma,  elementsC_d[0].x, w, dd);
			
			elementsP_d[id].z = gamma * w;
			if(id == 0){
				elementsI_d[0].w = N0 * mcmcNE;
			}

		}
		elementsC_d[id + MCMC_NT].x = -1;
		double rd = curand_uniform(&random);
		elementsP_d[id].y = rd;
		random_d[id] = random;
		if(EE < 3){		
			if(id < MCMC_NT){
				elementsC_d[id].x = 0;
			}
		}
	}
#endif
}

__global__ void Mix_kernel(double4 *elementsA_d, double4 *elementsB_d, double4 *elementsAOld_d, double4 *elementsBOld_d, int2 *elementsC_d, double4 *elementsP_d, int Nst, int N, int N0){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	if(id < N){

		int i = id;
		int j = id + N;
		if(id % (N0 * 2) >= N0){
			i = id + N;
			j = id;
		}

		if(id % N0 == 0){
			double4 Pi = elementsP_d[i / N0];
			double4 Pj = elementsP_d[j / N0];
			elementsP_d[id / N0] = Pi;
			elementsP_d[(id + N) / N0] = Pj;
		}
		double4 Ai = elementsAOld_d[i];
		double4 Bi = elementsBOld_d[i];
		double4 Aj = elementsAOld_d[j];
		double4 Bj = elementsBOld_d[j];

		elementsAOld_d[id] = Ai;
		elementsBOld_d[id] = Bi;
		elementsAOld_d[id + N] = Aj;
		elementsBOld_d[id + N] = Bj;

		Ai = elementsA_d[i];
		Bi = elementsB_d[i];
		Aj = elementsA_d[j];
		Bj = elementsB_d[j];

		elementsA_d[id] = Ai;
		elementsB_d[id] = Bi;
		elementsA_d[id + N] = Aj;
		elementsB_d[id + N] = Bj;


		int2 Ci = elementsC_d[i + MCMC_NT];
		int2 Cj = elementsC_d[j + MCMC_NT];

		elementsC_d[id + MCMC_NT] = Ci;
		elementsC_d[id + N + MCMC_NT] = Cj;
	}

}



__global__ void TSwap_kernel(curandState *random_d, double4 *elementsP_d, double4 *elementsAOld_d, double4 *elementsBOld_d, double *elementsSA_d, int N0, int Nst){

#if USE_RANDOM == 1
	curandState random;
	random = random_d[0];	

	for(int k = 0; k < 50; ++k){

		int i = int (curand_uniform(&random) * MCMC_NT);
		int j = int (curand_uniform(&random) * MCMC_NT);

		if(i == j){
			j = int (curand_uniform(&random) * MCMC_NT);
		}

		double pi = elementsP_d[i].x;
		double pj = elementsP_d[j].x;
		double Ti = elementsSA_d[i * Nst / MCMC_NT];
		double Tj = elementsSA_d[j * Nst / MCMC_NT];

		double q = exp((pi - pj) * (1.0 / Ti - 1.0 / Tj));
		double rd = curand_uniform(&random);
		
		if(q > rd && i != j){

printf("accept swap     %3d %3d %g %g %g %g %g %g\n", i, j, pi, pj, Ti, Tj, q, rd);

			for(int ii = 0; ii < N0; ++ii){
				double4 elementsAOldi = elementsAOld_d[i * N0 + ii];
				double4 elementsBOldi = elementsBOld_d[i * N0 + ii];
				double4 elementsAOldj = elementsAOld_d[j * N0 + ii];
				double4 elementsBOldj = elementsBOld_d[j * N0 + ii];

				elementsAOld_d[i * N0 + ii] = elementsAOldj;
				elementsBOld_d[i * N0 + ii] = elementsBOldj;
				elementsAOld_d[j * N0 + ii] = elementsAOldi;
				elementsBOld_d[j * N0 + ii] = elementsBOldi;

				elementsP_d[i].x = pj;
				elementsP_d[j].x = pi;

			}
			break;
		}
		else{
printf("not accept swap %3d %3d %g %g %g %g %g %g\n", i, j, pi, pj, Ti, Tj, q, rd);
		
		}
	}
	random_d[0] = random;
#endif
}


__global__ void sigma_kernel(double4 *elementsAOld_d, double4 *elementsBOld_d, elements10 *elementsL_d, double time, double Msun, int N0, int Nst){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;
;
	double4 elementsAOld = {0.0, 0.0, 0.0, 0.0};
	double4 elementsBOld = {0.0, 0.0, 0.0, 0.0};
	double4 elementsAOld2 = {0.0, 0.0, 0.0, 0.0};
	double4 elementsBOld2 = {0.0, 0.0, 0.0, 0.0};

	if(id < N0){

		for(int i = 0; i < Nst; ++i){


			double a = elementsAOld_d[id + i *N0].x;
			double e = elementsAOld_d[id + i *N0].y;
			double inc = elementsAOld_d[id + i *N0].z;
			double m = elementsAOld_d[id + i *N0].w;
			double Omega = elementsBOld_d[id + i *N0].x;
			double w = elementsBOld_d[id + i *N0].y;
			double M = elementsBOld_d[id + i *N0].z;
			double r = elementsBOld_d[id + i *N0].w;

			double mu = def_ksq * (Msun + m);

			double P = 2.0 * M_PI * sqrt(a * a * a / mu) / dayUnit;

			double nu = M_PI * 0.5 - w;	//true anomaly at first transit

			double e2 = e * e;
			double e4 = e2 * e2;
			double Mt = nu - 2.0 * e * sin(nu) + (3.0 * 0.25 * e2 + 0.125 * e4) * sin(2.0 * nu) - 1.0 / 3.0 * e * e2 * sin(3.0 * nu) + 5.0/32.0 * e4 * sin(4.0 * nu);
			
			if(M > Mt) Mt += 2.0 * M_PI;

			double T = time + P /(2.0 * M_PI) * (Mt - M);
		
			double xx = sqrt(e) * cos(w);
			double yy = sqrt(e) * sin(w);


			elementsAOld.x += P;
			elementsAOld.y += xx;
			elementsAOld.z += inc;
			elementsAOld.w += m;
			elementsBOld.x += Omega;
			elementsBOld.y += yy;
			elementsBOld.z += T;
			elementsBOld.w += r;

			elementsAOld2.x += P * P;
			elementsAOld2.y += xx * xx;
			elementsAOld2.z += inc * inc;
			elementsAOld2.w += m * m;
			elementsBOld2.x += Omega * Omega;
			elementsBOld2.y += yy * yy;
			elementsBOld2.z += T * T;
			elementsBOld2.w += r * r;

		}	
		elementsAOld.x /= ((double)(Nst));
		elementsAOld.y /= ((double)(Nst));
		elementsAOld.z /= ((double)(Nst));
		elementsAOld.w /= ((double)(Nst));
		elementsBOld.x /= ((double)(Nst));
		elementsBOld.y /= ((double)(Nst));
		elementsBOld.z /= ((double)(Nst));
		elementsBOld.w /= ((double)(Nst));
		elementsAOld2.x /= ((double)(Nst));
		elementsAOld2.y /= ((double)(Nst));
		elementsAOld2.z /= ((double)(Nst));
		elementsAOld2.w /= ((double)(Nst));
		elementsBOld2.x /= ((double)(Nst));
		elementsBOld2.y /= ((double)(Nst));
		elementsBOld2.z /= ((double)(Nst));
		elementsBOld2.w /= ((double)(Nst));

 printf("S1 %d %15.10g %15.10g %15.10g %15.10g %15.10g %15.10g %15.10g %15.10g\n", id, elementsAOld.x, elementsAOld.y, elementsAOld.z, elementsAOld.w, elementsBOld.x, elementsBOld.y, elementsBOld.z, elementsBOld.w);

		elementsL_d[id].P = sqrt(fmax(elementsAOld2.x - elementsAOld.x * elementsAOld.x, 0.0));
		elementsL_d[id].e = sqrt(fmax(elementsAOld2.y - elementsAOld.y * elementsAOld.y, 0.0));
		elementsL_d[id].inc = sqrt(fmax(elementsAOld2.z - elementsAOld.z * elementsAOld.z, 0.0));
		elementsL_d[id].m = sqrt(fmax(elementsAOld2.w - elementsAOld.w * elementsAOld.w, 0.0));

		elementsL_d[id].O = sqrt(fmax(elementsBOld2.x - elementsBOld.x * elementsBOld.x, 0.0));
		elementsL_d[id].w = sqrt(fmax(elementsBOld2.y - elementsBOld.y * elementsBOld.y, 0.0));
		elementsL_d[id].T = sqrt(fmax(elementsBOld2.z - elementsBOld.z * elementsBOld.z, 0.0));
		elementsL_d[id].r = sqrt(fmax(elementsBOld2.w - elementsBOld.w * elementsBOld.w, 0.0));


 printf("S %d %15.10g %15.10g %15.10g %15.10g %15.10g %15.10g %15.10g %15.10g\n", id, elementsL_d[id].P, elementsL_d[id].e, elementsL_d[id].inc, elementsL_d[id].m, elementsL_d[id].O, elementsL_d[id].w, elementsL_d[id].T, elementsL_d[id].r);
	}	

}
__host__ void Data::modifyElementsCall(int ittv, int EE){
	setNtransits <<< (NT + 127) / 128, 128 >>> (NtransitsT_d, NT);
#if def_RV == 1 
	setNtransits <<< (Nst + 127) / 128, 128 >>> (NRVT_d, Nst);
#endif

#if MCMC_Q == 0
	//reduce sampling temperature
	SetSA_kernel <<< (Nst + 127) / 128, 128 >>> (elementsSA_d, Nst);

 #if MCMC_NCOV > 0
	setCovarianceRandom1 <<< Nst, ((N_h[0] + 31) / 32) * 32 >>> (random_d, elementsL_d, Nst, N_h[0]); 
	setCovarianceRandom <<< Nst, ((N_h[0] + 31) / 32) * 32 >>> (elementsCOV_d, elementsL_d, Nst, N_h[0]); 
 #endif
	modifyElementsJ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, spin_d, elementsA_d, elementsB_d, elementsT_d, elementsSpin_d, elementsAOld_d, elementsBOld_d, elementsTOld_d, elementsSpinOld_d, elementsL_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, EE);

#endif
#if MCMC_Q == 2
	modifyElementsJ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, spin_d, elementsA_d, elementsB_d, elementsT_d, elementsSpin_d, elementsAOld_d, elementsBOld_d, elementsTOld_d, elementsSpinOld_d, elementsL_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, 0, P.mcmcRestart, 10); 
#endif
#if MCMC_Q == 1

	if(ittv % 16 == 0) modifyElementsJ <<< (Nst / 3 + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld_d, elementsBOld_d, elementsL_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst / 3, ittv, P.mcmcNE, P.mcmcRestart, EE);

	cudaMemcpy(elementsAOld2_d, elementsA_d, sizeof(double4) * NconstT, cudaMemcpyDeviceToDevice);
	cudaMemcpy(elementsBOld2_d, elementsB_d, sizeof(double4) * NconstT, cudaMemcpyDeviceToDevice);

	if(ittv % 16 == 1) modifyElementsPQ <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsL_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 0, 1.0, 0);
	if(ittv % 16 == 2) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsL_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 0, 1.0, 0);
	if(ittv % 16 == 3) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsL_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 1, 1.0, 0);
	if(ittv % 16 == 4) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsL_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 2, 1.0, 0);
	if(ittv % 16 == 5) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsL_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 3, 1.0, 0);
	if(ittv % 16 == 6) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsL_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 4, 1.0, 0);
	if(ittv % 16 == 7) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsL_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 5, 1.0, 0);
	if(ittv % 16 == 8) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsL_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 6, 1.0, 0);
	if(ittv % 16 == 9) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsL_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 0, 1.0, 1);
	if(ittv % 16 == 10) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsL_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 1, 1.0, 1);
	if(ittv % 16 == 11) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsL_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 2, 1.0, 1);
	if(ittv % 16 == 12) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsL_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 3, 1.0, 1);
	if(ittv % 16 == 13) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsL_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 4, 1.0, 1);
	if(ittv % 16 == 14) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsL_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 5, 1.0, 1);
	if(ittv % 16 == 15){
			modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsA_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 6, 0.0, 1);
			setJ_kernel <<< (Nst + 127) / 128, 128 >>> (elementsP_d, Nst);

	}
/*
	if(ittv % 16 == 1) modifyElementsPQ <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsL_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 0, 1.0, 0);
	if(ittv % 16 == 2) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsL_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 0, 1.0, 0);
	if(ittv % 16 == 3) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsL_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 1, 1.0, 0);
	if(ittv % 16 == 4) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsL_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 2, 1.0, 0);
	if(ittv % 16 == 5) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsL_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 3, 1.0, 0);
	if(ittv % 16 == 6) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsL_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 4, 1.0, 0);
	if(ittv % 16 == 7) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsL_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 5, 1.0, 0);
	if(ittv % 16 == 8) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsL_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 6, 1.0, 0);
	if(ittv % 16 == 9) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsL_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 0, 1.0, 0);
	if(ittv % 16 == 10) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsL_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 1, 1.0, 0);
	if(ittv % 16 == 11) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsL_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 2, 1.0, 0);
	if(ittv % 16 == 12) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsL_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 3, 1.0, 0);
	if(ittv % 16 == 13) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsL_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 4, 1.0, 0);
	if(ittv % 16 == 14) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsL_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 5, 1.0, 0);
	if(ittv % 16 == 15){
			modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsL_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 6, 0.0, 0);
			setJ_kernel <<< (Nst + 127) / 128, 128 >>> (elementsP_d, Nst);

	}
*/
#endif

}


//For def_TTV == 2
//set the time and timestep for a new simulation
__global__ void setTimeTTV_kernel(double *time_d, double *dt_d, double *lastTransitTime_d, int *transitIndex_d, int2 *EpochCount_d, double *TTV_d, double time, double dt0, int Nst, int N){

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < Nst){
		dt_d[id] = dt0 * dayUnit;
		time_d[id] = time;
		transitIndex_d[id] = 100000;
	}
	if(id < N){
		lastTransitTime_d[id] = time - 10.0 * dt0;
		EpochCount_d[id].x = 0;
		EpochCount_d[id].y = 0;
		TTV_d[id] = 0.0;
	}
}


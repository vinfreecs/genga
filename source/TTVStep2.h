#include "Host2.h"

// **************************************
//This function converts heliocentric coordinares to democratic coordinates.
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
		}
	}
}
//This function converts heliocentric coordinares to barycentric coordinates.
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
		int NtObs = NtransitsTObs_d[id];
//if(id <= 8) printf("NtObs %d\n", NtObs);
		double pp = 0.0;
		int Epoch = 0;
		int setEpoch = 0;

		for(int EpochObs = 0; EpochObs <= NtObs; ++EpochObs){
			double T = TransitTime_d[id * def_NtransitTimeMax + Epoch + 1];
			double T1 = TransitTime_d[id * def_NtransitTimeMax + Epoch + 2];	//next transit
			double2 Tobs = TransitTimeObs_d[id * def_NtransitTimeMax + EpochObs + 1];
//if(id % N_d[0] == 2) printf("--------- %d %.20g %.20g | %g %g %g | %d %d %d\n", id, T, Tobs.x, Tobs.x - T, Tobs.x - T1, T - Tobs.x, Epoch, EpochObs, id * def_NtransitTimeMax + Epoch + 1);

			if(setEpoch == 0 && fabs(Tobs.x - T) < fabs(Tobs.x - T1) && T != 0.0 && Tobs.x != 0.0){
				 setEpoch = 1;
//if(id % N_d[0] == 2) printf("set Epoch %d %.20g %.20g %d %d\n", id, T, Tobs.x, Epoch, EpochObs);
			}

			if(setEpoch == 0 && T != 0 && Tobs.x != 0 && fabs(Tobs.x - T) < fabs(Tobs.x - T1)){
//if(id % N_d[0] == 2) printf("********* %d %.20g %.20g %d %d\n", id, T, Tobs.x, Epoch, EpochObs);
				++EpochObs;
				if(EpochObs >= NtObs) break;
				Tobs = TransitTimeObs_d[id * def_NtransitTimeMax + EpochObs + 1];
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
				int NtObs = NtransitsTObs_d[id];
				pp = 0.0;
				int Epoch = 0;
				int setEpoch = 0;
				double P = TransitTimeObs_d[id * def_NtransitTimeMax].y; //period
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
					double2 Tobs = TransitTimeObs_d[id * def_NtransitTimeMax + EpochObs + 1];

					if(setEpoch == 0 && fabs(Tobs.x - T) < fabs(Tobs.x - T1) && T != 0.0 && Tobs.x != 0.0){
						 setEpoch = 1;
					}

					if(setEpoch == 0 && T != 0 && Tobs.x != 0 && fabs(Tobs.x - T) < fabs(Tobs.x - T1)){
						++EpochObs;
						if(EpochObs >= NtObs) break;
						Tobs = TransitTimeObs_d[id * def_NtransitTimeMax + EpochObs + 1];
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

// ********************************************************************************************
// This kernel performs a parallel summation of the values p*p / 2, it is also parallel for multi simulations.
//
// At the end, this kernel cheks, if the mcmc step is accepted or not.
// elementsP_d.y contains a random number
//
// Author: Simon Grimm
// February 2020
// *******************************************************************************************
__global__ void TTVstep3(int *index_d, double4 *elementsA_d, double4 *elementsB_d, double4 *elementsT_d, double4 *elementsAOld_d, double4 *elementsBOld_d, double4 *elementsTOld_d, double4 *elementsP_d, double *elementsSA_d, int2 *elementsC_d, int2 *NtransitsT_d, double4 *Msun_d, double *elementsM_d, int NT, int N0, int Nst, int mcmcNE){

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

// The kernels overwrites tuning lengths elementsLA and elementsLB
//Date: February 2020
//Author: Simon Grimm
// *******************************************************
__global__ void setCovarianceRandom1(curandState *random_d, double4 *elementsLA_d, double4 *elementsLB_d, int Nst, int N0){
	int idy = threadIdx.x;
	int idx = blockIdx.x;
	curandState random;	

	if(idx < Nst){

		if(idy < N0){
			random = random_d[idx * N0 + idy];
			//generate random number vector X
			//P
			double rd = curand_normal(&random);
			elementsLA_d[idx * N0 + idy].x = rd;
#if MCMC_NCOV > 1
			//T
			rd = curand_normal(&random);
			elementsLB_d[idx * N0 + idy].z = rd;
#endif
#if MCMC_NCOV > 2
			//m
			rd = curand_normal(&random);
			elementsLA_d[idx * N0 + idy].w = rd;
#endif
#if MCMC_NCOV > 3
			//e
			rd = curand_normal(&random);
			elementsLA_d[idx * N0 + idy].y = rd;
#endif
#if MCMC_NCOV > 4
			//w
			rd = curand_normal(&random);
			elementsLB_d[idx * N0 + idy].y = rd;
#endif
			random_d[idx * N0 + idy] = random;

		}
	}
}

__global__ void setCovarianceRandom(double *elementsCOV_d, double4 *elementsLA_d, double4 *elementsLB_d, int Nst, int N0){
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
						if(k == 0) rd = elementsLA_d[idx * N0 + j].x; //P
						if(k == 1) rd = elementsLB_d[idx * N0 + j].z; //T
						if(k == 3) rd = elementsLA_d[idx * N0 + j].w; //m
						if(k == 4) rd = elementsLA_d[idx * N0 + j].y; //e
						if(k == 5) rd = elementsLB_d[idx * N0 + j].y; //w


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
				
			if(MCMC_NCOV > 0) elementsLA_d[idx * N0 + idy].x = zP;
			if(MCMC_NCOV > 1) elementsLB_d[idx * N0 + idy].z = zT;
			if(MCMC_NCOV > 2) elementsLA_d[idx * N0 + idy].w = zm;
			if(MCMC_NCOV > 3) elementsLA_d[idx * N0 + idy].y = ze;
			if(MCMC_NCOV > 4) elementsLB_d[idx * N0 + idy].y = zw;
//printf("COV %d %d %g %g %g\n", idx * N0 + idy, idy, zP, elementsCOV_d[idy * N0 + 0], elementsCOV_d[(idx * N0 + idy) * N0 * MCMC_NCOV + 1]);
		}
	}
}


//adagrad
__global__ void adagrad(double4 *elementsAOld_d, double4 *elementsBOld_d, double4 *elementsLA_d, double4 *elementsLB_d, double4 *elementsGA_d, double4 *elementsGB_d, double4 *elementsP_d, const int N0, const int Ne, const int Nst){
	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	if(id < Nst){
		double dx, dx1, gx, Gx;

		//double eta = 0.1;
		double eta = 0.2;
		double eps = 1.0e-6;
		int jj = id % Ne;
		int ii = id / Ne;
		if(jj == 0){
			dx = elementsLA_d[id * N0 + ii].x;
			gx = -2.0 * (elementsP_d[id].z - elementsP_d[id].x) / dx;
			Gx = elementsGA_d[id * N0 + ii].x + gx * gx;
			elementsGA_d[id * N0 + ii].x = Gx;
			dx1 = -eta / sqrt(Gx + eps*dx) * gx *1.0e-9;
			elementsLA_d[id * N0 + ii].x = fmax(fmin(fabs(dx1), dx), 1.0e-16);
			for(int j = 0; j < Nst; ++j){
				elementsAOld_d[j * N0 + ii].x += dx1;
			}
printf("dx P %d %d %g %g | %g %g %g %g\n", id, ii, 2.0 * elementsP_d[id].x, 2.0 * elementsP_d[id].z, dx, gx, sqrt(Gx + eps*dx), dx1);
		}
		if(jj == 1){
			dx = elementsLB_d[id * N0 + ii].z;
			gx = -2.0 * (elementsP_d[id].z - elementsP_d[id].x) / dx;
			Gx = elementsGB_d[id * N0 + ii].z + gx * gx;
			elementsGB_d[id * N0 + ii].z = Gx;
			dx1 = -eta / sqrt(Gx + eps*dx) * gx *1.0e-9;
			elementsLB_d[id * N0 + ii].z = fmax(fmin(fabs(dx1), dx), 1.0e-16);
			for(int j = 0; j < Nst; ++j){
				elementsBOld_d[j * N0 + ii].z += dx1;
			}
printf("dx T %d %d %g %g | %g %g %g %g\n", id, ii, 2.0 * elementsP_d[id].x, 2.0 * elementsP_d[id].z, dx, gx, sqrt(Gx + eps*dx), dx1);
		}
		if(jj == 2){
			dx = elementsLA_d[id * N0 + ii].w;
			gx = -2.0 * (elementsP_d[id].z - elementsP_d[id].x) / dx;
			Gx = elementsGA_d[id * N0 + ii].w + gx * gx;
			elementsGA_d[id * N0 + ii].w = Gx;
			dx1 = -eta / sqrt(Gx + eps*dx) * gx * 1.0e-9;
			elementsLA_d[id * N0 + ii].w = fmax(fmin(fabs(dx1), dx), 1.0e-16);
			for(int j = 0; j < Nst; ++j){
				elementsAOld_d[j * N0 + ii].w += dx1;
			}
printf("dx m %d %d %g %g | %g %g %g %g\n", id, ii, 2.0 * elementsP_d[id].x, 2.0 * elementsP_d[id].z, dx, gx, sqrt(Gx + eps*dx), dx1);
		}
		if(jj == 3){
			dx = elementsLA_d[id * N0 + ii].y;
			gx = -2.0 * (elementsP_d[id].z - elementsP_d[id].x) / dx;
			Gx = elementsGA_d[id * N0 + ii].y + gx * gx;
			elementsGA_d[id * N0 + ii].y = Gx;
			dx1 = -eta / sqrt(Gx + eps*dx) * gx * 1.0e-6;
			elementsLA_d[id * N0 + ii].y = fmax(fmin(fabs(dx1), dx), 1.0e-16);
			for(int j = 0; j < Nst; ++j){
				elementsAOld_d[j * N0 + ii].y += dx1;
			}
printf("dx e %d %d %g %g | %g %g %g %g\n", id, ii, 2.0 * elementsP_d[id].x, 2.0 * elementsP_d[id].z, dx, gx, sqrt(Gx + eps*dx), dx1);
		}
		if(jj == 4){
			dx = elementsLB_d[id * N0 + ii].y;
			gx = -2.0 * (elementsP_d[id].z - elementsP_d[id].x) / dx;
			Gx = elementsGB_d[id * N0 + ii].y + gx * gx;
			elementsGB_d[id * N0 + ii].y = Gx;
			dx1 = -eta / sqrt(Gx + eps*dx) * gx * 1.0e-6;
			elementsLB_d[id * N0 + ii].y = fmax(fmin(fabs(dx1), dx), 1.0e-16);
			for(int j = 0; j < Nst; ++j){
				elementsBOld_d[j * N0 + ii].y += dx1;
			}
printf("dx w %d %d %g %g | %g %g %g %g\n", id, ii, 2.0 * elementsP_d[id].x, 2.0 * elementsP_d[id].z, dx, gx, sqrt(Gx + eps*dx), dx1);
		}


	}
}

//adadelta
__global__ void adadelta(double4 *elementsAOld_d, double4 *elementsBOld_d, double4 *elementsLA_d, double4 *elementsLB_d, double4 *elementsGA_d, double4 *elementsGB_d, double4 *elementsDA_d, double4 *elementsDB_d, double4 *elementsP_d, const int N0, const int Ne, const int Nst){
	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	if(id < Nst){
		double dx, dx1, gx, Gx, Dx;

		double beta = 0.9;
		double eps = 1.0e-6;
		int jj = id % Ne;
		int ii = id / Ne;
		if(jj == 0){
			dx = elementsLA_d[id * N0 + ii].x;
			//eps = 1.0e-11 * dx;
			eps = 1.0e-12 * dx;
			gx = -2.0 * (elementsP_d[id].z - elementsP_d[id].x) / dx;
			Gx = beta * elementsGA_d[id * N0 + ii].x + (1.0 - beta) * gx * gx;
			Dx = elementsDA_d[id * N0 + ii].x;
			elementsGA_d[id * N0 + ii].x = Gx;
			dx1 = -sqrt(Dx + eps) / sqrt(Gx + eps) * gx;
			elementsLA_d[id * N0 + ii].x = fmax(fmin(fabs(dx1), dx), 1.0e-16);
			elementsDA_d[id * N0 + ii].x = beta * Dx + (1.0 - beta) * dx1 * dx1;
			for(int j = 0; j < Nst; ++j){
				elementsAOld_d[j * N0 + ii].x += dx1;
			}
printf("dx P %d %d %g %g | %g %g %g %g\n", id, ii, 2.0 * elementsP_d[id].x, 2.0 * elementsP_d[id].z, dx, gx, sqrt(Gx + eps*dx), dx1);
		}
		if(jj == 1){
			dx = elementsLB_d[id * N0 + ii].z;
			eps = 1.0e-12 * dx;
			gx = -2.0 * (elementsP_d[id].z - elementsP_d[id].x) / dx;
			Gx = beta * elementsGB_d[id * N0 + ii].z + (1.0 - beta) * gx * gx;
			Dx = elementsDB_d[id * N0 + ii].z;
			elementsGB_d[id * N0 + ii].z = Gx;
			dx1 = -sqrt(Dx + eps) / sqrt(Gx + eps) * gx;
			elementsLB_d[id * N0 + ii].z = fmax(fmin(fabs(dx1), dx), 1.0e-16);
			elementsDB_d[id * N0 + ii].z = beta * Dx + (1.0 - beta) * dx1 * dx1;
			for(int j = 0; j < Nst; ++j){
				elementsBOld_d[j * N0 + ii].z += dx1;
			}
printf("dx T %d %d %g %g | %g %g %g %g\n", id, ii, 2.0 * elementsP_d[id].x, 2.0 * elementsP_d[id].z, dx, gx, sqrt(Gx + eps*dx), dx1);
		}
		if(jj == 2){
			dx = elementsLA_d[id * N0 + ii].w;
			eps = 1.0e-12 * dx;
			gx = -2.0 * (elementsP_d[id].z - elementsP_d[id].x) / dx;
			Gx = beta * elementsGA_d[id * N0 + ii].w + (1.0 - beta) * gx * gx;
			Dx = elementsDA_d[id * N0 + ii].w;
			elementsGA_d[id * N0 + ii].w = Gx;
			dx1 = -sqrt(Dx + eps) / sqrt(Gx + eps) * gx;
			elementsLA_d[id * N0 + ii].w = fmax(fmin(fabs(dx1), dx), 1.0e-16);
			elementsDA_d[id * N0 + ii].w = beta * Dx + (1.0 - beta) * dx1 * dx1;
			for(int j = 0; j < Nst; ++j){
				elementsAOld_d[j * N0 + ii].w += dx1;
			}
printf("dx m %d %d %g %g | %g %g %g %g\n", id, ii, 2.0 * elementsP_d[id].x, 2.0 * elementsP_d[id].z, dx, gx, sqrt(Gx + eps*dx), dx1);
		}
		if(jj == 3){
			dx = elementsLA_d[id * N0 + ii].y;
			eps = 1.0e-12 * dx;
			gx = -2.0 * (elementsP_d[id].z - elementsP_d[id].x) / dx;
			Gx = beta * elementsGA_d[id * N0 + ii].y + (1.0 - beta) * gx * gx;
			Dx = elementsDA_d[id * N0 + ii].y;
			elementsGA_d[id * N0 + ii].y = Gx;
			dx1 = -sqrt(Dx + eps) / sqrt(Gx + eps) * gx;
			elementsLA_d[id * N0 + ii].y = fmax(fmin(fabs(dx1), dx), 1.0e-16);
			elementsDA_d[id * N0 + ii].y = beta * Dx + (1.0 - beta) * dx1 * dx1;
			for(int j = 0; j < Nst; ++j){
				elementsAOld_d[j * N0 + ii].y += dx1;
			}
printf("dx e %d %d %g %g | %g %g %g %g\n", id, ii, 2.0 * elementsP_d[id].x, 2.0 * elementsP_d[id].z, dx, gx, sqrt(Gx + eps*dx), dx1);
		}
		if(jj == 4){
			dx = elementsLB_d[id * N0 + ii].y;
			eps = 1.0e-12 * dx;
			gx = -2.0 * (elementsP_d[id].z - elementsP_d[id].x) / dx;
			Gx = beta * elementsGB_d[id * N0 + ii].y + (1.0 - beta) * gx * gx;
			Dx = elementsDB_d[id * N0 + ii].y;
			elementsGB_d[id * N0 + ii].y = Gx;
			dx1 = -sqrt(Dx + eps) / sqrt(Gx + eps) * gx;
			elementsLB_d[id * N0 + ii].y = fmax(fmin(fabs(dx1), dx), 1.0e-16);
			elementsDB_d[id * N0 + ii].y = beta * Dx + (1.0 - beta) * dx1 * dx1;
			for(int j = 0; j < Nst; ++j){
				elementsBOld_d[j * N0 + ii].y += dx1;
			}
printf("dx w %d %d %g %g | %g %g %g %g\n", id, ii, 2.0 * elementsP_d[id].x, 2.0 * elementsP_d[id].z, dx, gx, sqrt(Gx + eps*dx), dx1);
		}


	}
}

//adam
__global__ void adam(double4 *elementsAOld_d, double4 *elementsBOld_d, double4 *elementsLA_d, double4 *elementsLB_d, double4 *elementsGA_d, double4 *elementsGB_d, double4 *elementsDA_d, double4 *elementsDB_d, double4 *elementsP_d, const int N0, const int Ne, const int Nst){
	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	if(id < Nst){
		double dx, dx1, gx, Gx, Dx;

		double beta1 = 0.9;
		double beta2 = 0.999;
		double eta = 0.01;
		double eps = 1.0e-9;
		int jj = id % Ne;
		int ii = id / Ne;
		if(jj == 0){
			dx = elementsLA_d[id * N0 + ii].x;
			gx = -2.0 * (elementsP_d[id].z - elementsP_d[id].x) / dx;
			Gx = beta2 * elementsGA_d[id * N0 + ii].x + (1.0 - beta2) * gx * gx;
			Dx = beta1 * elementsDA_d[id * N0 + ii].x + (1.0 - beta1) * gx;
			elementsGA_d[id * N0 + ii].x = Gx;
			elementsDA_d[id * N0 + ii].x = Dx;
			dx1 = -eta / sqrt(Gx + eps) * Dx * 1.0e-8;
			elementsLA_d[id * N0 + ii].x = fmax(fmin(fabs(dx1), dx), 1.0e-16);
			for(int j = 0; j < Nst; ++j){
				elementsAOld_d[j * N0 + ii].x += dx1;
			}
printf("dx P %d %d %g %g | %g %g %g %g\n", id, ii, 2.0 * elementsP_d[id].x, 2.0 * elementsP_d[id].z, dx, gx, sqrt(Gx + eps), dx1);
		}
		if(jj == 1){
			dx = elementsLB_d[id * N0 + ii].z;
			gx = -2.0 * (elementsP_d[id].z - elementsP_d[id].x) / dx;
			Gx = beta2 * elementsGB_d[id * N0 + ii].z + (1.0 - beta2) * gx * gx;
			Dx = beta1 * elementsDB_d[id * N0 + ii].z + (1.0 - beta1) * gx;
			elementsGB_d[id * N0 + ii].z = Gx;
			elementsDB_d[id * N0 + ii].z = Dx;
			dx1 = -eta / sqrt(Gx + eps) * Dx * 1.0e-9;
			elementsLB_d[id * N0 + ii].z = fmax(fmin(fabs(dx1), dx), 1.0e-16);
			for(int j = 0; j < Nst; ++j){
				elementsBOld_d[j * N0 + ii].z += dx1;
			}
printf("dx T %d %d %g %g | %g %g %g %g\n", id, ii, 2.0 * elementsP_d[id].x, 2.0 * elementsP_d[id].z, dx, gx, sqrt(Gx + eps), dx1);
		}
		if(jj == 2){
			dx = elementsLA_d[id * N0 + ii].w;
			gx = -2.0 * (elementsP_d[id].z - elementsP_d[id].x) / dx;
			Gx = beta2 * elementsGA_d[id * N0 + ii].w + (1.0 - beta2) * gx * gx;
			Dx = beta1 * elementsDA_d[id * N0 + ii].w + (1.0 - beta1) * gx;
			elementsGA_d[id * N0 + ii].w = Gx;
			elementsDA_d[id * N0 + ii].w = Dx;
			dx1 = -eta / sqrt(Gx + eps) * Dx * 1.0e-7;
			elementsLA_d[id * N0 + ii].w = fmax(fmin(fabs(dx1), dx), 1.0e-16);
			for(int j = 0; j < Nst; ++j){
				elementsAOld_d[j * N0 + ii].w += dx1;
			}
printf("dx m %d %d %g %g | %g %g %g %g\n", id, ii, 2.0 * elementsP_d[id].x, 2.0 * elementsP_d[id].z, dx, gx, sqrt(Gx + eps), dx1);
		}
		if(jj == 3){
			dx = elementsLA_d[id * N0 + ii].y;
			gx = -2.0 * (elementsP_d[id].z - elementsP_d[id].x) / dx;
			Gx = beta2 * elementsGA_d[id * N0 + ii].y + (1.0 - beta2) * gx * gx;
			Dx = beta1 * elementsDA_d[id * N0 + ii].y + (1.0 - beta1) * gx;
			elementsGA_d[id * N0 + ii].y = Gx;
			elementsDA_d[id * N0 + ii].y = Dx;
			dx1 = -eta / sqrt(Gx + eps) * Dx * 1.0e-4;
			elementsLA_d[id * N0 + ii].y = fmax(fmin(fabs(dx1), dx), 1.0e-16);
			for(int j = 0; j < Nst; ++j){
				elementsAOld_d[j * N0 + ii].y += dx1;
			}
printf("dx e %d %d %g %g | %g %g %g %g\n", id, ii, 2.0 * elementsP_d[id].x, 2.0 * elementsP_d[id].z, dx, gx, sqrt(Gx + eps), dx1);
		}
		if(jj == 4){
			dx = elementsLB_d[id * N0 + ii].y;
			gx = -2.0 * (elementsP_d[id].z - elementsP_d[id].x) / dx;
			Gx = beta2 * elementsGB_d[id * N0 + ii].y + (1.0 - beta2) * gx * gx;
			Dx = beta1 * elementsDB_d[id * N0 + ii].y + (1.0 - beta1) * gx;
			elementsGB_d[id * N0 + ii].y = Gx;
			elementsDB_d[id * N0 + ii].y = Dx;
			dx1 = -eta / sqrt(Gx + eps) * Dx * 1.0e-2;
			elementsLB_d[id * N0 + ii].y = fmax(fmin(fabs(dx1), dx), 1.0e-16);
			for(int j = 0; j < Nst; ++j){
				elementsBOld_d[j * N0 + ii].y += dx1;
			}
printf("dx w %d %d %g %g | %g %g %g %g\n", id, ii, 2.0 * elementsP_d[id].x, 2.0 * elementsP_d[id].z, dx, gx, sqrt(Gx + eps), dx1);
		}


	}
}

//adaMax
__global__ void adaMax(double4 *elementsAOld_d, double4 *elementsBOld_d, double4 *elementsLA_d, double4 *elementsLB_d, double4 *elementsGA_d, double4 *elementsGB_d, double4 *elementsDA_d, double4 *elementsDB_d, double4 *elementsP_d, const int N0, const int Ne, const int Nst){
	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	if(id < Nst){
		double dx, dx1, gx, Gx, Dx;

		double beta1 = 0.9;
		double beta2 = 0.999;
		double eta = 0.01;
		int jj = id % Ne;
		int ii = id / Ne;
		if(jj == 0){
			dx = elementsLA_d[id * N0 + ii].x;
			gx = -2.0 * (elementsP_d[id].z - elementsP_d[id].x) / dx;
			Gx = fmax(beta2 * elementsGA_d[id * N0 + ii].x, fabs(gx));
			Dx = beta1 * elementsDA_d[id * N0 + ii].x + (1.0 - beta1) * gx;
			elementsGA_d[id * N0 + ii].x = Gx;
			elementsDA_d[id * N0 + ii].x = Dx;
			dx1 = -eta / Gx * Dx * 1.0e-7;
			if(gx == 0.0) dx1 = 0.0;
			elementsLA_d[id * N0 + ii].x = fmax(fmin(fabs(dx1), dx), 1.0e-16);
			for(int j = 0; j < Nst; ++j){
				elementsAOld_d[j * N0 + ii].x += dx1;
			}
printf("dx P %d %d %g %g | %g %g %g %g\n", id, ii, 2.0 * elementsP_d[id].x, 2.0 * elementsP_d[id].z, dx, gx, Gx, dx1);
		}
		if(jj == 1){
			dx = elementsLB_d[id * N0 + ii].z;
			gx = -2.0 * (elementsP_d[id].z - elementsP_d[id].x) / dx;
			Gx = fmax(beta2 * elementsGB_d[id * N0 + ii].z, fabs(gx));
			Dx = beta1 * elementsDB_d[id * N0 + ii].z + (1.0 - beta1) * gx;
			elementsGB_d[id * N0 + ii].z = Gx;
			elementsDB_d[id * N0 + ii].z = Dx;
			dx1 = -eta / Gx * Dx * 1.0e-9;
			elementsLB_d[id * N0 + ii].z = fmax(fmin(fabs(dx1), dx), 1.0e-16);
			for(int j = 0; j < Nst; ++j){
				elementsBOld_d[j * N0 + ii].z += dx1;
			}
printf("dx T %d %d %g %g | %g %g %g %g\n", id, ii, 2.0 * elementsP_d[id].x, 2.0 * elementsP_d[id].z, dx, gx, Gx, dx1);
		}
		if(jj == 2){
			dx = elementsLA_d[id * N0 + ii].w;
			gx = -2.0 * (elementsP_d[id].z - elementsP_d[id].x) / dx;
			Gx = fmax(beta2 * elementsGA_d[id * N0 + ii].w, fabs(gx));
			Dx = beta1 * elementsDA_d[id * N0 + ii].w + (1.0 - beta1) * gx;
			elementsGA_d[id * N0 + ii].w = Gx;
			elementsDA_d[id * N0 + ii].w = Dx;
			dx1 = -eta / Gx * Dx * 1.0e-7;
			elementsLA_d[id * N0 + ii].w = fmax(fmin(fabs(dx1), dx), 1.0e-16);
			for(int j = 0; j < Nst; ++j){
				elementsAOld_d[j * N0 + ii].w += dx1;
			}
printf("dx m %d %d %g %g | %g %g %g %g\n", id, ii, 2.0 * elementsP_d[id].x, 2.0 * elementsP_d[id].z, dx, gx, Gx, dx1);
		}
		if(jj == 3){
			dx = elementsLA_d[id * N0 + ii].y;
			gx = -2.0 * (elementsP_d[id].z - elementsP_d[id].x) / dx;
			Gx = fmax(beta2 * elementsGA_d[id * N0 + ii].y, fabs(gx));
			Dx = beta1 * elementsDA_d[id * N0 + ii].y + (1.0 - beta1) * gx;
			elementsGA_d[id * N0 + ii].y = Gx;
			elementsDA_d[id * N0 + ii].y = Dx;
			dx1 = -eta / Gx * Dx * 1.0e-3;
			elementsLA_d[id * N0 + ii].y = fmax(fmin(fabs(dx1), dx), 1.0e-16);
			for(int j = 0; j < Nst; ++j){
				elementsAOld_d[j * N0 + ii].y += dx1;
			}
printf("dx e %d %d %g %g | %g %g %g %g\n", id, ii, 2.0 * elementsP_d[id].x, 2.0 * elementsP_d[id].z, dx, gx, Gx, dx1);
		}
		if(jj == 4){
			dx = elementsLB_d[id * N0 + ii].y;
			gx = -2.0 * (elementsP_d[id].z - elementsP_d[id].x) / dx;
			Gx = fmax(beta2 * elementsGB_d[id * N0 + ii].y, fabs(gx));
			Dx = beta1 * elementsDB_d[id * N0 + ii].y + (1.0 - beta1) * gx;
			elementsGB_d[id * N0 + ii].y = Gx;
			elementsDB_d[id * N0 + ii].y = Dx;
			dx1 = -eta / Gx * Dx * 1.0e-2;
			elementsLB_d[id * N0 + ii].y = fmax(fmin(fabs(dx1), dx), 1.0e-16);
			for(int j = 0; j < Nst; ++j){
				elementsBOld_d[j * N0 + ii].y += dx1;
			}
printf("dx w %d %d %g %g | %g %g %g %g\n", id, ii, 2.0 * elementsP_d[id].x, 2.0 * elementsP_d[id].z, dx, gx, Gx, dx1);
		}


	}
}


//use Jacoby mass
//EE = -1: no change
//EE = 0: first step
//EE = 4: DEMCMC
//EE = 5: ADAGRAD
//EE = 10 Refine
__global__ void modifyElementsJ2(curandState *random_d, double4 *x4_d, double4 *v4_d, double4 *elementsA_d, double4 *elementsB_d, double4 *elementsT_d, double4 *elementsAOld_d, double4 *elementsBOld_d, double4 *elementsTOld_d, double4 *elementsLA_d, double4 *elementsLB_d, double4 *elementsP_d, int4 *elementsI_d, int2 *elementsC_d, double4 *Msun_d, double time, int *N_d, int Nst, int ittv, int mcmcNE, int mcmcRestart, int EE){

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

#if def_TTV == 2
Msun = 0.0;
#endif
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
			}
//printf("Modify %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g| %g %d\n", ii, m, r, P, e, inc, Omega, w, T, a, M, z, ne);

			double rd;
			if(EE == 0){
				if(ne > 0){
					rd = curand_normal(&random);
					P += rd * elementsLA_d[st0 * N0 + ii].x;       //scale to standard deviation of tuning length
				}
				if(ne > 1){
					rd = curand_normal(&random);
					T += rd * elementsLB_d[st0 * N0 + ii].z;
				}
				if(ne > 2){
					rd = curand_normal(&random);
					m += rd * elementsLA_d[st0 * N0 + ii].w;
				}
				if(ne > 3){
					rd = curand_normal(&random);
					e += rd * elementsLA_d[st0 * N0 + ii].y;
				}
				if(ne > 4){
					rd = curand_normal(&random);
					w += rd * elementsLB_d[st0 * N0 + ii].y;
					//jump to other modes
					//rd = curand_uniform(&random);
					//if(rd < 0.5) w += M_PI;
				}
				if(ne > 5){
					rd = curand_normal(&random);
					inc += rd * elementsLA_d[st0 * N0 + ii].z;
					rd = curand_normal(&random);
					Omega += rd * elementsLB_d[st0 * N0 + ii].x;
				}
				if(ne > 7){
					rd = curand_normal(&random);
					r += rd * elementsLB_d[st0 * N0 + ii].w;
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
				rd = curand_normal(&random) * eps * elementsLA_d[st0 * N0 + ii].x;
				P += P0 * rd;

				P += elementsLA_d[st0 * N0 + ii].x * sc;
				if(EE == 5 && id / ne == ii && id % ne == 0) {
					 P -= elementsLA_d[st0 * N0 + ii].x;
printf("GRAD P %d %d %d %d\n", id, ii, id / ne, id % ne);
				}
			}
//printf("ModifyA %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g | %g\n", ii, m, r, P, e, inc, Omega, w, T, a, z);

			if(ne > 2){
				//modify m
				rd = curand_uniform(&random) * 2.0 * eb;
				m += z * (1.0 - eb + rd) * (m1 - m2);
				rd = curand_normal(&random) * eps * elementsLA_d[st0 * N0 + ii].w;
				m += m0 * rd;

				m += elementsLA_d[st0 * N0 + ii].w * sc;
				if(EE == 5 && id / ne == ii && id % ne == 2) {
					 m -= elementsLA_d[st0 * N0 + ii].w;
printf("GRAD m %d %d %d %d\n", id, ii, id / ne, id % ne);
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
				rd = curand_normal(&random) * eps * elementsLA_d[st0 * N0 + ii].y;
				e += e0 * rd;

				e += elementsLA_d[st0 * N0 + ii].y * sc;
				if(EE == 5 && id / ne == ii && id % ne == 3) {
					 e -= elementsLA_d[st0 * N0 + ii].y;
printf("GRAD e %d %d %d %d\n", id, ii, id / ne, id % ne);
				}
			}
			if(ne > 4){
				rd = curand_uniform(&random) * 2.0 * eb;
				w += z * (1.0 - eb + rd) * (w1 - w2);
				rd = curand_normal(&random) * eps * elementsLB_d[st0 * N0 + ii].y;
				w += w0 * rd;

				w += elementsLB_d[st0 * N0 + ii].y * sc;
				if(EE == 5 && id / ne == ii && id % ne == 4) {
					 w -= elementsLB_d[st0 * N0 + ii].y;
printf("GRAD w %d %d %d %d\n", id, ii, id / ne, id % ne);
				}
			}


			if(ne > 5){
				double inc0 = inc;
				double inc1 = elementsAOld_d[st1 * N0 + ii].z;		//eccentricity
				double inc2 = elementsAOld_d[st2 * N0 + ii].z;		//eccentricity
				rd = curand_uniform(&random) * 2.0 * eb;
				inc += z * (1.0 - eb + rd) * (inc1 - inc2);
				rd = curand_normal(&random) * eps * elementsLA_d[st0 * N0 + ii].z;
				inc += inc0 * rd;
	
				double Omega0 = Omega;
				double Omega1 = elementsBOld_d[st1 * N0 + ii].x;		//eccentricity
				double Omega2 = elementsBOld_d[st2 * N0 + ii].x;		//eccentricity
				rd = curand_uniform(&random) * 2.0 * eb;
				Omega += z * (1.0 - eb + rd) * (Omega1 - Omega2);
				rd = curand_normal(&random) * eps * elementsLB_d[st0 * N0 + ii].x;
				Omega += Omega0 * rd;

			}

			if(e <= 0) w = 0;
//printf("ModifyB %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g | %g\n", ii, m, r, P, e, inc, Omega, w, T, a, z);
	
			double T0 = T;
			double T1 = elementsTOld_d[st1 * N0 + ii].x;
			double T2 = elementsTOld_d[st2 * N0 + ii].x;

			if(ne > 1){
				rd = curand_uniform(&random) * 2.0 * eb;
				T += z * (1.0 - eb + rd) * (T1 - T2);
				rd = curand_normal(&random) * eps * elementsLB_d[st0 * N0 + ii].z;
				T += T0 * rd;

				T += elementsLB_d[st0 * N0 + ii].z * sc;
				//M += z * (M1 - M2);
				if(EE == 5 && id / ne == ii && id % ne == 1) {
					 T -= elementsLB_d[st0 * N0 + ii].z;
printf("GRAD T %d %d %d %d\n", id, ii, id / ne, id % ne);
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

//printf("ModifyR %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g | %g\n", ii, m, r, P, e, inc, Omega, w, T, a, z);
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
//printf("MJ %d %.20g %d %d %d\n", st0 * N0 + ii, a, st0, st1, st2);
//printf("ModifyC %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g| %g\n", ii, m, r, P, e, inc, Omega, w, T, a, M, z);

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
#if def_TTV == 2
if(ii == 0){
//set cenctral body to be heliocentric
	x4i.x = 0.0;
	x4i.y = 0.0;
	x4i.z = 0.0;
	v4i.x = 0.0;
	v4i.y = 0.0;
	v4i.z = 0.0;
}
#endif

			x4_d[st0 * N0 + ii] = x4i;
			v4_d[st0 * N0 + ii] = v4i;

//printf("ModifyD %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g\n", ii, x4i.w, v4i.w, x4i.x, x4i.y, x4i.z, v4i.x, v4i.y, v4i.z);

		}// end of loop around planets
		 random_d[id] = random;
	}
}

//quadratic estimation for period
__global__ void modifyElementsPQ(curandState *random_d, double4 *x4_d, double4 *v4_d, double4 *elementsA_d, double4 *elementsB_d, double4 *elementsAOld2_d, double4 *elementsBOld2_d, double4 *elementsLA_d, double4 *elementsLB_d, double4 *elementsP_d, int4 *elementsI_d, int2 *elementsC_d, double4 *Msun_d, double time, int *N_d, int Nst, int ittv, int mcmcNE, int mcmcRestart, int EE, double ff, int AA){

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
				//x1 = a - elementsLA_d[st0 * N0 + ii].x * ff;
				//x2 = a;
				//x3 = a + elementsLA_d[st0 * N0 + ii].x * ff;
				x1 = a - 5.0e-7;
				x2 = a;
				x3 = a + 5.0e-7;
			}
			if(AA == 1){		
				x1 = M - elementsLB_d[st0 * N0 + ii].z * ff;
				x2 = M;
				x3 = M + elementsLB_d[st0 * N0 + ii].z * ff;
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
//printf("a1 %d %.20g %.20g\n", id, a, elementsLA_d[st0 * N0 + ii].x);
			}
			if(id / (Nst / 3) == 1){
				x = x2;
//printf("a2 %d %.20g %.20g\n", id, a, elementsLA_d[st0 * N0 + ii].x);
				elementsP_d[id].z = pOld;
			}
			if(id / (Nst / 3) == 2){
				x = x3;
//printf("a3 %d %.20g %.20g\n", id, a, elementsLA_d[st0 * N0 + ii].x);
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
__global__ void modifyElementsPQ2(curandState *random_d, double4 *x4_d, double4 *v4_d, double4 *elementsA_d, double4 *elementsB_d, double4 *elementsAOld2_d, double4 *elementsBOld2_d, double4 *elementsLA_d, double4 *elementsLB_d, double4 *elementsP_d, int4 *elementsI_d, int2 *elementsC_d, double4 *Msun_d, double time, int *N_d, int Nst, int ittv, int mcmcNE, int mcmcRestart, int EE, double ff, int AA){

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
					//deltaa1 =-elementsLA_d[st0 * N0 + ii].x * ff;
					//deltaa2 = 0.0;
					//deltaa3 = elementsLA_d[st0 * N0 + ii].x * ff;
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
					//deltaM1 =-elementsLB_d[st0 * N0 + ii].z * ff;
					//deltaM2 = 0.0;
					//deltaM3 = elementsLB_d[st0 * N0 + ii].z * ff;
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
//printf("a1 %d %d %.20g %.20g %g %g %g\n", ii, id, x, elementsLA_d[st0 * N0 + ii].x, ff, deltaa, deltaM);
			}
			if(id / (Nst / 3) == 1){
				x = x2;
				deltaa = deltaa2;
				deltaM = deltaM2;
//printf("a2 %d %d %.20g %.20g %g %g %g\n", ii, id, x, elementsLA_d[st0 * N0 + ii].x, ff, deltaa, deltaM);
			}
			if(id / (Nst / 3) == 2){
				x = x3;
				deltaa = deltaa3;
				deltaM = deltaM3;
//printf("a3 %d %d %.20g %.20g %g %g %g\n", ii, id, x, elementsLA_d[st0 * N0 + ii].x, ff, deltaa, deltaM);
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
}
__global__ void setJ_kernel(double4 *elementsP_d, int Nst){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	if(id < Nst){

		elementsP_d[id].x = elementsP_d[id].z;
	}

}
__global__ void setJ_kernel(curandState *random_d, double4 *elementsP_d, int4 *elementsI_d, int2 *elementsC_d, int Nst, int N0, double4 *Msun_d, double *elementsM_d, int ittv, int mcmcNE, int EE){

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
}


__global__ void sigma_kernel(double4 *elementsAOld_d, double4 *elementsBOld_d, double4 *elementsLA_d, double4 *elementsLB_d, double time, double Msun, int N0, int Nst){

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

		elementsLA_d[id].x = sqrt(fmax(elementsAOld2.x - elementsAOld.x * elementsAOld.x, 0.0));
		elementsLA_d[id].y = sqrt(fmax(elementsAOld2.y - elementsAOld.y * elementsAOld.y, 0.0));
		elementsLA_d[id].z = sqrt(fmax(elementsAOld2.z - elementsAOld.z * elementsAOld.z, 0.0));
		elementsLA_d[id].w = sqrt(fmax(elementsAOld2.w - elementsAOld.w * elementsAOld.w, 0.0));

		elementsLB_d[id].x = sqrt(fmax(elementsBOld2.x - elementsBOld.x * elementsBOld.x, 0.0));
		elementsLB_d[id].y = sqrt(fmax(elementsBOld2.y - elementsBOld.y * elementsBOld.y, 0.0));
		elementsLB_d[id].z = sqrt(fmax(elementsBOld2.z - elementsBOld.z * elementsBOld.z, 0.0));
		elementsLB_d[id].w = sqrt(fmax(elementsBOld2.w - elementsBOld.w * elementsBOld.w, 0.0));


 printf("S %d %15.10g %15.10g %15.10g %15.10g %15.10g %15.10g %15.10g %15.10g\n", id, elementsLA_d[id].x, elementsLA_d[id].y, elementsLA_d[id].z, elementsLA_d[id].w, elementsLB_d[id].x, elementsLB_d[id].y, elementsLB_d[id].z, elementsLB_d[id].w);
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
	setCovarianceRandom1 <<< Nst, ((N_h[0] + 31) / 32) * 32 >>> (random_d, elementsLA_d, elementsLB_d, Nst, N_h[0]); 
	setCovarianceRandom <<< Nst, ((N_h[0] + 31) / 32) * 32 >>> (elementsCOV_d, elementsLA_d, elementsLB_d, Nst, N_h[0]); 
 #endif
	modifyElementsJ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsT_d, elementsAOld_d, elementsBOld_d, elementsTOld_d, elementsLA_d, elementsLB_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, EE);

#endif
#if MCMC_Q == 2
	modifyElementsJ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsT_d, elementsAOld_d, elementsBOld_d, elementsTOld_d, elementsLA_d, elementsLB_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, 0, P.mcmcRestart, 10); 
#endif
#if MCMC_Q == 1

	if(ittv % 16 == 0) modifyElementsJ <<< (Nst / 3 + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld_d, elementsBOld_d, elementsLA_d, elementsLB_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst / 3, ittv, P.mcmcNE, P.mcmcRestart, EE);

	cudaMemcpy(elementsAOld2_d, elementsA_d, sizeof(double4) * NconstT, cudaMemcpyDeviceToDevice);
	cudaMemcpy(elementsBOld2_d, elementsB_d, sizeof(double4) * NconstT, cudaMemcpyDeviceToDevice);

	if(ittv % 16 == 1) modifyElementsPQ <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsLA_d, elementsLB_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 0, 1.0, 0);
	if(ittv % 16 == 2) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsLA_d, elementsLB_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 0, 1.0, 0);
	if(ittv % 16 == 3) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsLA_d, elementsLB_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 1, 1.0, 0);
	if(ittv % 16 == 4) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsLA_d, elementsLB_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 2, 1.0, 0);
	if(ittv % 16 == 5) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsLA_d, elementsLB_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 3, 1.0, 0);
	if(ittv % 16 == 6) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsLA_d, elementsLB_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 4, 1.0, 0);
	if(ittv % 16 == 7) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsLA_d, elementsLB_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 5, 1.0, 0);
	if(ittv % 16 == 8) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsLA_d, elementsLB_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 6, 1.0, 0);
	if(ittv % 16 == 9) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsLA_d, elementsLB_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 0, 1.0, 1);
	if(ittv % 16 == 10) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsLA_d, elementsLB_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 1, 1.0, 1);
	if(ittv % 16 == 11) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsLA_d, elementsLB_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 2, 1.0, 1);
	if(ittv % 16 == 12) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsLA_d, elementsLB_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 3, 1.0, 1);
	if(ittv % 16 == 13) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsLA_d, elementsLB_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 4, 1.0, 1);
	if(ittv % 16 == 14) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsLA_d, elementsLB_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 5, 1.0, 1);
	if(ittv % 16 == 15){
			modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsLA_d, elementsLB_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 6, 0.0, 1);
			setJ_kernel <<< (Nst + 127) / 128, 128 >>> (elementsP_d, Nst);

	}
/*
	if(ittv % 16 == 1) modifyElementsPQ <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsLA_d, elementsLB_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 0, 1.0, 0);
	if(ittv % 16 == 2) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsLA_d, elementsLB_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 0, 1.0, 0);
	if(ittv % 16 == 3) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsLA_d, elementsLB_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 1, 1.0, 0);
	if(ittv % 16 == 4) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsLA_d, elementsLB_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 2, 1.0, 0);
	if(ittv % 16 == 5) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsLA_d, elementsLB_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 3, 1.0, 0);
	if(ittv % 16 == 6) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsLA_d, elementsLB_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 4, 1.0, 0);
	if(ittv % 16 == 7) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsLA_d, elementsLB_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 5, 1.0, 0);
	if(ittv % 16 == 8) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsLA_d, elementsLB_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 6, 1.0, 0);
	if(ittv % 16 == 9) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsLA_d, elementsLB_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 0, 1.0, 0);
	if(ittv % 16 == 10) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsLA_d, elementsLB_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 1, 1.0, 0);
	if(ittv % 16 == 11) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsLA_d, elementsLB_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 2, 1.0, 0);
	if(ittv % 16 == 12) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsLA_d, elementsLB_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 3, 1.0, 0);
	if(ittv % 16 == 13) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsLA_d, elementsLB_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 4, 1.0, 0);
	if(ittv % 16 == 14) modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsLA_d, elementsLB_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 5, 1.0, 0);
	if(ittv % 16 == 15){
			modifyElementsPQ2 <<< (Nst + 127) / 128, 128 >>> (random_d, x4_d, v4_d, elementsA_d, elementsB_d, elementsAOld2_d, elementsBOld2_d, elementsLA_d, elementsLB_d, elementsP_d, elementsI_d, elementsC_d, Msun_d, time_h[0] - dt_h[0] / dayUnit, N_d, Nst, ittv, P.mcmcNE, P.mcmcRestart, 6, 0.0, 0);
			setJ_kernel <<< (Nst + 127) / 128, 128 >>> (elementsP_d, Nst);

	}
*/
#endif

}



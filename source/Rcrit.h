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
__global__ void Rcritb_kernel(double4 *__restrict__ x4_d, double4 *__restrict__ v4_d, double4 * __restrict__ x4b_d, double4 *__restrict__ v4b_d, double4 *__restrict__ spin_d, double4 *__restrict__ spinb_d, double iMsun3, double *__restrict__ rcrit_d, double *__restrict__ rcritb_d, double *__restrict__ rcritv_d, double *__restrict__ rcritvb_d, int * __restrict__ index_d, int * __restrict__  indexb_d, double dt, double n1, double n2, double *time_d, double time, int *EjectionFlag_d, const int N, const int NconstT, const int SLevels, const int f){
	
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	
	double4 x4i;
	double4 v4i;
	
	double rcrit, rcritv;
	double rsq, vsq, r, v;
	
	if(id == 0) time_d[0] = time;
	
	if(id < N){
		
		if(StopAtCollision_c[0] != 0 || CollTshift_c[0] != 1.0){
			//printf("Rcrit %d %g %g\n", StopAtCollision_c[0], StopMinMass_c[0], CollTshift_c[0]);
			if(f == 0){
				x4i = x4_d[id];
				v4i = v4_d[id];
				
				//store coordinates backup		
				x4b_d[id] = x4i;
				v4b_d[id] = v4i;
				spinb_d[id] = spin_d[id];
				for(int l = 0; l < SLevels; ++l){
					rcritb_d[id + l * NconstT] = rcrit_d[id + l * NconstT];
					rcritvb_d[id + l * NconstT] = rcritv_d[id + l * NconstT];
				}
				rcritb_d[id] = rcrit_d[id];
				rcritvb_d[id] = rcritv_d[id];
				indexb_d[id] = index_d[id];
			}
			else{
				//restore old coordinates
				x4i = x4b_d[id];
				v4i = v4b_d[id];
				
				x4_d[id] = x4i;
				v4_d[id] = v4i;
				spin_d[id] = spinb_d[id];
				for(int l = 0; l < SLevels; ++l){
					rcrit_d[id + l * NconstT] = rcritb_d[id + l * NconstT];
					rcritv_d[id + l * NconstT] = rcritvb_d[id + l * NconstT];
				}
				index_d[id] = indexb_d[id];
			}
		}
		else{
			x4i = x4_d[id];
			v4i = v4_d[id];
			#if def_TTV > 0
			v4b_d[id] = v4i;
			#endif
		}
		rsq = x4i.x*x4i.x + x4i.y*x4i.y + x4i.z*x4i.z + 1.0e-30;
		vsq = v4i.x*v4i.x + v4i.y*v4i.y + v4i.z*v4i.z + 1.0e-30;
		
		r = sqrt(rsq);
		v = sqrt(vsq);
		
		rcrit = n1 * r * cbrt(x4i.w * iMsun3);
		
		if(WriteEncounters_c[0] > 0){
			//in scales of planetary Radius
			double writeRadius = WriteEncountersRadius_c[0] * v4i.w;
			rcrit = fmax(rcrit, writeRadius);
		}
		
		if(StopAtEncounter_c[0] > 0){
			//rescale to non n2 rcrit 
			rcrit = StopAtEncounterRadius_c[0] * rcrit / n1;
		}

		double rc2 = n2 * fabs(dt) * v;
		rcritv = fmax(rcrit, rc2);
		
		rcrit_d[id] = rcrit;
		//the following prevents from too large critical radii for highly eccentric massive planetes
		if(rc2 > rcrit){
			rcritv_d[id] = fmax(rcritv, rcritv_d[id]);
		}
		else{
			rcritv_d[id] = rcritv;
		}
		//if(id == 3) printf("%d %g %g %g %g %g %g\n", id, rcritv_d[id], rcrit_d[id], rcritv, rc2, rcrit, r);
		//Check for Ejections or too small distances to the Sun
		if((rsq > Rcut_c[0] * Rcut_c[0] || rsq < RcutSun_c[0] * RcutSun_c[0]) && x4_d[id].w >= 0.0){
			EjectionFlag_d[0] = 1;
		}
	}
}

__global__ void Rcrit_kernel(double4 *__restrict__ x4_d, double4 *__restrict__ v4_d, double4 * __restrict__ x4b_d, double4 *__restrict__ v4b_d, double4 *__restrict__ spin_d, double4 *__restrict__ spinb_d, double iMsun3, double *__restrict__ rcrit_d, double *__restrict__ rcritb_d, double *__restrict__ rcritv_d, double *__restrict__ rcritvb_d, int * __restrict__ index_d, int * __restrict__  indexb_d, double dt, double n1, double n2, double *time_d, double time, int *EjectionFlag_d, const int N, const int NconstT, const int SLevels, const int f){
	
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	
	double4 x4i;
	double4 v4i;
	
	double rcrit, rcritv;
	double rsq, vsq, r, v;
	
	if(id == 0) time_d[0] = time;
	
	if(id < N){
		
		if(StopAtCollision_c[0] != 0 || CollTshift_c[0] != 1.0){
			//printf("Rcrit %d %g %g\n", StopAtCollision_c[0], StopMinMass_c[0], CollTshift_c[0]);
			if(f == 0){
				x4i = x4_d[id];
				v4i = v4_d[id];
				
				//store coordinates backup		
				x4b_d[id] = x4i;
				v4b_d[id] = v4i;
				spinb_d[id] = spin_d[id];
				for(int l = 0; l < SLevels; ++l){
					rcritb_d[id + l * NconstT] = rcrit_d[id + l * NconstT];
					rcritvb_d[id + l * NconstT] = rcritv_d[id + l * NconstT];
				}
				indexb_d[id] = index_d[id];
			}
			else{
				//restore old coordinates
				x4i = x4b_d[id];
				v4i = v4b_d[id];
				
				x4_d[id] = x4i;
				v4_d[id] = v4i;
				spin_d[id] = spinb_d[id];
				for(int l = 0; l < SLevels; ++l){
					rcrit_d[id + l * NconstT] = rcritb_d[id + l * NconstT];
					rcritv_d[id + l * NconstT] = rcritvb_d[id + l * NconstT];
				}
				index_d[id] = indexb_d[id];
			}
		}
		else{
			x4i = x4_d[id];
			v4i = v4_d[id];
			#if def_TTV > 0
			v4b_d[id] = v4i;
			#endif
		}
		rsq = x4i.x*x4i.x + x4i.y*x4i.y + x4i.z*x4i.z + 1.0e-30;
		vsq = v4i.x*v4i.x + v4i.y*v4i.y + v4i.z*v4i.z + 1.0e-30;
		
		r = sqrt(rsq);
		v = sqrt(vsq);
		
		rcrit = n1 * r * cbrt(x4i.w * iMsun3);
		
		if(WriteEncounters_c[0] > 0){
			//in scales of planetary Radius
			double writeRadius = WriteEncountersRadius_c[0] * v4i.w;
			rcrit = fmax(rcrit, writeRadius);
		}
		
		if(StopAtEncounter_c[0] > 0){
			//rescale to non n2 rcrit 
			rcrit = StopAtEncounterRadius_c[0] * rcrit / n1;
		}
		
		double rc2 = n2 * fabs(dt) * v;
		rcritv = fmax(rcrit, rc2);
		
		rcrit_d[id] = rcrit;
		//the following prevents from too large critical radii for highly eccentric massive planets
		if(rc2 > rcrit){
			rcritv_d[id] = fmax(rcritv, rcritv_d[id]);
		}
		else{
			rcritv_d[id] = rcritv;
		}
//if(id < 10) printf("rcrit %d %g %.20g %.20g %g %g %g %g %g %d\n", id, time, x4i.x, v4i.x, x4i.w, x4b_d[id].x, x4b_d[id].w, rcritv_d[id], rcritvb_d[id], f);
		//Check for Ejections or too small distances to the Sun
		if((rsq > Rcut_c[0] * Rcut_c[0] || rsq < RcutSun_c[0] * RcutSun_c[0]) && x4_d[id].w >= 0.0){
			EjectionFlag_d[0] = 1;
		}
	}
}

//device function version of the Rcrit_kernel
/*
__device__ void Rcrit(double4 *__restrict__ x4_d, double4 *__restrict__ v4_d, double4 * __restrict__ x4b_d, double4 *__restrict__ v4b_d, double4 *__restrict__ spin_d, double4 *__restrict__ spinb_d, double iMsun3, double *__restrict__ rcrit_d, double *__restrict__ rcritb_d, double *__restrict__ rcritv_d, double *__restrict__ rcritvb_d, int * __restrict__ index_d, int * __restrict__  indexb_d, double dt, double n1, double n2, int *EjectionFlag_d, const int N, const int NconstT, const int SLevels, const int f){
	
	double4 x4i;
	double4 v4i;
	
	double rcrit, rcritv;
	double rsq, vsq, r, v;
	
	
	if(id < N){
		
		if(StopAtCollision_c[0] != 0 || CollTshift_c[0] != 1.0){
			//printf("Rcrit %d %g %g\n", StopAtCollision_c[0], StopMinMass_c[0], CollTshift_c[0]);
			if(f == 0){
				x4i = x4_d[id];
				v4i = v4_d[id];
				
				//store coordinates backup		
				x4b_d[id] = x4i;
				v4b_d[id] = v4i;
				spinb_d[id] = spin_d[id];
				for(int l = 0; l < SLevels; ++l){
					rcritb_d[id + l * NconstT] = rcrit_d[id + l * NconstT];
					rcritvb_d[id + l * NconstT] = rcritv_d[id + l * NconstT];
				}
				indexb_d[id] = index_d[id];
			}
			else{
				//restore old coordinates
				x4i = x4b_d[id];
				v4i = v4b_d[id];
				
				x4_d[id] = x4i;
				v4_d[id] = v4i;
				spin_d[id] = spinb_d[id];
				for(int l = 0; l < SLevels; ++l){
					rcrit_d[id + l * NconstT] = rcritb_d[id + l * NconstT];
					rcritv_d[id + l * NconstT] = rcritvb_d[id + l * NconstT];
				}
				index_d[id] = indexb_d[id];
			}
		}
		else{
			x4i = x4_d[id];
			v4i = v4_d[id];
		}
		rsq = x4i.x*x4i.x + x4i.y*x4i.y + x4i.z*x4i.z + 1.0e-30;
		vsq = v4i.x*v4i.x + v4i.y*v4i.y + v4i.z*v4i.z + 1.0e-30;
		
		r = sqrt(rsq);
		v = sqrt(vsq);
		
		rcrit = n1 * r * cbrt(x4i.w * iMsun3);
		
		if(WriteEncounters_c[0] > 0){
			//in scales of planetary Radius
			double writeRadius = WriteEncountersRadius_c[0] * v4i.w;
			rcrit = fmax(rcrit, writeRadius);
		}
		
		if(StopAtEncounter_c[0] > 0){
			//rescale to non n2 rcrit 
			rcrit = StopAtEncounterRadius_c[0] * rcrit / n1;
		}
		
		double rc2 = n2 * fabs(dt) * v;
		rcritv = fmax(rcrit, rc2);
		
		rcrit_d[id] = rcrit;
		//the following prevents from too large critical radii for highly eccentric massive planetes
		if(rc2 > rcrit){
			rcritv_d[id] = fmax(rcritv, rcritv_d[id]);
		}
		else{
			rcritv_d[id] = rcritv;
		}
		//printf("rcrit %d %.20g %.20g %g %g %g %g %g %d\n", id, x4i.x, v4i.x, x4i.w, x4b_d[id].x, x4b_d[id].w, rcritv_d[id], rcritvb_d[id], f);
		//Check for Ejections or too small distances to the Sun
		if((rsq > Rcut_c[0] * Rcut_c[0] || rsq < RcutSun_c[0] * RcutSun_c[0]) && x4_d[id].w >= 0.0){
			EjectionFlag_d[0] = 1;
		}
	}

}
*/

// *****************************************************
// Version of the Rcrit kernel which is called from the recursive symplectic sub step method
//
// Author: Simon Grimm
// January 2019
// ********************************************************
__global__ void RcritS_kernel(double4 *__restrict__ x4_d, double4 *__restrict__ v4_d, double iMsun3, double *__restrict__ rcrit_d, double *__restrict__ rcritv_d, double dt, double n1, double n2, const int N, int *Nencpairs_d, int *Nencpairs2_d, int *Nencpairs3_d, int *Encpairs3_d, const int NencMax, const int NconstT, const int SLevel){
	
	int idd = blockIdx.x * blockDim.x + threadIdx.x;
	
	double4 x4i;
	double4 v4i;
	
	double rcrit, rcritv ;
	double rsq, vsq, r, v;
	if(idd == 0){
		Nencpairs_d[0] = 0;
		Nencpairs2_d[0] = 0;
	}
	if(idd < Nencpairs3_d[0]){
		int id = Encpairs3_d[idd * NencMax + 1];
		if(id >= 0 && id < N){
			
			x4i = x4_d[id];
			v4i = v4_d[id];
			//printf("RcritS %d %d %.20g %.20g %g %g\n", idd, id, x4i.x, v4i.y, rcrit_d[id + SLevel * NconstT], rcritv_d[id + SLevel * NconstT]);			
			rsq = x4i.x*x4i.x + x4i.y*x4i.y + x4i.z*x4i.z + 1.0e-30;
			vsq = v4i.x*v4i.x + v4i.y*v4i.y + v4i.z*v4i.z + 1.0e-30;
			
			r = sqrt(rsq);
			v = sqrt(vsq);
			
			rcrit = n1 * r * cbrt(x4i.w * iMsun3);
#if def_SLn1 == 1
			rcrit = fmax(rcrit, 3.0 * v4i.w);
#endif
			double rc2 = n2 * fabs(dt) * v;
			rcritv = fmax(rcrit, rc2);
			
			if(rc2 > rcrit){
				rcritv_d[id + SLevel * NconstT] = fmax(rcritv, rcritv_d[id + SLevel * NconstT]);
			}
			else{
				rcritv_d[id + SLevel * NconstT] = rcritv;
			}
			
			rcrit_d[id + SLevel * NconstT] = rcrit;
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
//
//Author: Simon Grimm
//November 2016
// ****************************************
__global__ void RcritM_kernel(double4 * __restrict__ x4_d, double4 * __restrict__ v4_d, double4 * __restrict__ x4b_d, double4 * __restrict__ v4b_d, double4 *__restrict__ spin_d, double4 *__restrict__ spinb_d, double2 *Msun_d, double *rcrit_d, double *rcritb_d, double *rcritv_d, double *rcritvb_d, double *dt_d, double *n1_d, double *n2_d, double *Rcut_d, double *RcutSun_d, int *EjectionFlag_d, int * __restrict__ index_d, int * __restrict__ indexb_d, const int Nst, const int NT, double *time_d, double *idt_d, double *ict_d, long long *delta_d, long long timeStep, int *StopFlag_d, const int NconstT, const int SLevels, const int f, const int Nstart){
	
	int id = blockIdx.x * blockDim.x + threadIdx.x + Nstart;
	int st = 0;
	
	if(id < NT + Nstart) st = index_d[id] / def_MaxIndex;
	if(id < Nst + Nstart) time_d[id - Nstart] = timeStep * idt_d[id - Nstart] + ict_d[id - Nstart] * 365.25;
	
	double4 x4i;
	double4 v4i;
	
	double rcrit, rcritv;
	double rsq, vsq, r, v;
	
	if(id < NT + Nstart){
//printf("Rcrit %d %d %.20e %.20e %.20e %.20e %.20e %.20e %g\n", id, index_d[id], x4_d[id].x, x4_d[id].y, x4_d[id].z, v4_d[id].x, v4_d[id].y, v4_d[id].z, n1_d[st]);
		double Msun = Msun_d[st].x;
		double n1 = n1_d[st];
		double n2 = n2_d[st];
		double Rcut = Rcut_d[st];
		double RcutSun = RcutSun_d[st];
		double dt = dt_d[st];
//printf("Rcrit %d %g %g %g %g %g %g %g %g\n", id, Msun, n1, n2, Rcut, RcutSun, dt, rcrit_d[id], rcritv_d[id]);		
		if(StopAtCollision_c[0] != 0 || CollTshift_c[0] != 1.0){
			if(f == 0){
				x4i = x4_d[id];
				v4i = v4_d[id];
				
				x4b_d[id] = x4i;
				v4b_d[id] = v4i;
				spinb_d[id] = spin_d[id];
				for(int l = 0; l < SLevels; ++l){
					rcritb_d[id + l * NconstT] = rcrit_d[id + l * NconstT];
					rcritvb_d[id + l * NconstT] = rcritv_d[id + l * NconstT];
				}
				indexb_d[id] = index_d[id];
			}
			else{
				x4i = x4b_d[id];
				v4i = v4b_d[id];
				
				x4_d[id] = x4i;
				v4_d[id] = v4i;
				spin_d[id] = spinb_d[id];
				for(int l = 0; l < SLevels; ++l){
					rcrit_d[id + l * NconstT] = rcritb_d[id + l * NconstT];
					rcritv_d[id + l * NconstT] = rcritvb_d[id + l * NconstT];
				}
				index_d[id] = indexb_d[id];
			}
		}
		else{
			x4i = x4_d[id];
			v4i = v4_d[id];
			#if def_TTV > 0
			v4b_d[id] = v4i;
			#endif
		}
		
		__syncthreads();
//printf("Rcrit %d %d %.20g %.20g %.20g %.20g %.20g %.20g %g\n", id, index_d[id], x4_d[id].x, x4_d[id].y, x4_d[id].z, v4_d[id].x, v4_d[id].y, v4_d[id].z, n1_d[st]);
		
		rsq = x4i.x*x4i.x + x4i.y*x4i.y + x4i.z*x4i.z + 1.0e-30;
		vsq = v4i.x*v4i.x + v4i.y*v4i.y + v4i.z*v4i.z + 1.0e-30;
		r = sqrt(rsq);
		v = sqrt(vsq);
		
		rcrit = n1 * r * cbrt(x4i.w  / ( Msun * 3.0));
		if(WriteEncounters_c[0] > 0){
			//in scales of planetary Radius
			double writeRadius = WriteEncountersRadius_c[0] * v4i.w;
			rcrit = fmax(rcrit, writeRadius);
		}
		
		if(StopAtEncounter_c[0] > 0){
			//rescale to non n2 rcrit 
			rcrit = StopAtEncounterRadius_c[0] * rcrit / n1;
		}
		
		double rc2 = n2 * fabs(dt) * v;
		rcritv = fmax(rcrit, rc2);
		
		rcrit_d[id] = rcrit;
		//the following prevents from too large critical radii for highly eccentric massive planetes
		if(rc2 > rcrit){
			rcritv_d[id] = fmax(rcritv, rcritv_d[id]);
		}
		else{
			rcritv_d[id] = rcritv;
		}
		//Check for Ejections or too small distances to the Sun
		if((rsq > Rcut * Rcut || rsq < RcutSun * RcutSun) && x4_d[id].w >= 0.0){
			EjectionFlag_d[st + 1] = 1;
			EjectionFlag_d[0] = 1;
		}
		#if def_TTV == 0
		if(timeStep >= delta_d[st]){
			StopFlag_d[0] = 1;
		}
		#endif
	}
}


__host__ void Data::firstStep(int noColl){
#if def_CPU == 0
	if(Nst > 1){
		#if def_TTV == 2

		#else
		if(UseM3 == 0){
			firstKick_M(0, noColl);
		}
		else{
			firstKick_M3(0, noColl);
		}
		#endif
	}
	else{
		if(P.UseTestParticles > 0){
			firstKick_small(noColl);
		}
		else{
			if(NB[0] <= WarpSize){
				firstKick_16(noColl);
			}
			else{
				firstKick_largeN(noColl);
			}
		}
	}
#else
	if(P.UseTestParticles > 0){
		firstKick_small_cpu(noColl);
	}
	else{
		firstKick_cpu(noColl);
	}
#endif
}

__host__ int Data::step(int noColl){
	int er;
#if def_CPU == 0
	//Multi simulation mode
	if(MultiSim == 1){
		#if def_TTV == 2
		  er = ttv_step();
		#else

		  #if def_NoEncounters == 0
			if(UseM3 == 0){
				er = step_M(noColl);
			}
			else{
				er = step_M3(noColl);
			}
	 	  #else
			er = step_MSimple();
		  #endif
		#endif
		if(er == 0) return 0;
	}
	else{
		//Test particles
		if(P.UseTestParticles > 0){
			er = step_small(noColl);
			if(er == 0) return 0;
		}
		//check the number of massive particles
		else{
			if(NB[0] <= WarpSize){
				er =  step_16(noColl);
			}
			else{
				er = step_largeN(noColl);
			}
			if(er == 0) return 0;
		}
	}
#else
	if(P.UseTestParticles > 0){
		er = step_small_cpu(noColl);
		if(er == 0) return 0;
	}
	else{
		//step1_cpu();
		//er = 1;
		er = step_cpu(noColl);
		if(er == 0) return 0;
	}
#endif
	return 1;
}

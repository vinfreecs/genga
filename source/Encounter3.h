#ifndef ENCOUNTER_H
#define ENCOUNTER_H
#include "Orbit2.h"

#if G3 ==1
#include "Encounter3G3.h"
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
//E = 2: Used for Critical Radius with Test Particles
//E = 3: Used for Critial Radius in multi simlation mode
//Code is adapted from Mercury
//
// Authors: Simon Grimm
// April 2016
//
// ****************************************
template<int E>
__device__ int encounter(double4 x4i, double4 v4i, double4 x4oldi, double4 v4oldi, double4 x4j, double4 v4j, double4 x4oldj, double4 v4oldj, double rcriti, double rcritj, double rcritvi, double rcritvj, double dt, int i, int j, double *test_d, int2 *encpairs, int &Nenc, int N, double &time, int writeEncounters, double writeEncountersRadius, double MinMass){

//if((E == 0 || E >= 2))printf("E %d %d %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", i ,j, E, N, x4oldi.x, x4oldi.y, x4oldi.z, v4oldi.x, v4oldi.y, v4oldi.z);
//if(E == 1 && i != j) printf("E1  %d %d %g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g\n", i, j, time, x4i.w, v4i.w, x4i.x, x4i.y, x4i.z, x4j.w, v4j.w, x4j.x, x4j.y, x4j.z);
//if(E == 1 && i < j ) printf("E1o %d %d %g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g\n", i, j, time, x4oldi.w, v4oldi.w, x4oldi.x, x4oldi.y, x4oldi.z, x4oldj.w, v4oldj.w, x4oldj.x, x4oldj.y, x4oldj.z);
	int Enc = 0;
	if(i != j && (x4i.w > MinMass || x4j.w > MinMass) && x4i.w >= 0.0 && x4j.w >= 0.0){
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
			f = def_cef;
		}
		if(E == 1){
			rcrit = 0.0;
			rcritv = rcriti + rcritj;
			f = 1.0;
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
		else delta1 = 100.0;
		if(0 <= t2 && t2 <= 1){
			t22 = t2*t2;
			tt2 = 1.0-t2;
			tt22 = tt2*tt2;
			delta2 = tt22*(1.0 + 2.0*t2)*d0 + t22*(3.0 - 2.0*t2)*d1 + t2*tt22*c - t22*tt2*cc;
		}
		else delta2 = 100.0;

		delta = min(delta1,delta2);
		if(delta < 0) delta = 0.0;
		
		delta = fmin(delta, d1);
		delta = fmin(delta, d0);

		if(delta < f * rcritv*rcritv){
			Enc = 2;
//if((E == 0 || E >= 2))printf("EE %d %d %g %g %.40g %.40g %.40g %.40g %g %g %d\n", i, j, x4i.w, x4j.w, x4i.x, x4j.x, v4i.x, v4j.x, v4i.w, v4j.w, E);
//if (E == 1)printf("EE1 %d %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g\n", i, j, x4i.x, x4j.x, x4i.y, x4j.y, x4i.z, x4j.z, delta, rcritv*rcritv, d0, d1);
			if(E < 2){ 
				Ni = atomicAdd(&Nenc, 1);
				if(E == 1 && Nenc >= MaxColl) Ni = MaxColl - 1;
				if(x4i.w >= x4j.w){
					encpairs[Ni].x = i;
					encpairs[Ni].y = j;
				}
				else{
					encpairs[Ni].x = j;
					encpairs[Ni].y = i;
				}

// *****************
//dont group test particles
/*				if(x4i.w == 0.0){
					encpairs[Ni].x = i;
					encpairs[Ni].y = i;
				}
				if(x4j.w == 0.0){
					encpairs[Ni].x = j;
					encpairs[Ni].y = j;
				}
*/
// *****************
			}
		}
		else Enc = 0;
		if(delta < rcrit*rcrit){
			Enc = 1;
		}


		if(writeEncounters > 0 && E == 1){
			double writeRadius = 0.0;
			if(writeEncounters == 1){
				//in scales of planetary Radius
				writeRadius = writeEncountersRadius * fmax(v4i.w, v4j.w);

			}
			if(delta < writeRadius * writeRadius){

				double t = 0.0;
				if(0 <= t1 && t1 <= 1) t = t1;
				if(0 <= t2 && t2 <= 1 && delta2 < delta1) t = t2;

				if(t > 0.0 && t < 1.0){
					time = t;
//printf("Enc %g %g %g %d %d\n", t, writeRadius, sqrt(delta), i, j);	
//					writeEnc_d[ne * 25 + 0] = (time + dt * fmin(t1, t2)/0.01720209895) / 365.25;
				}
			}
		}


		return Enc;
	}
	else return 0;
}
template<int E>
__device__ int encounterb(double4 x4i, double4 v4i, double4 x4oldi, double4 v4oldi, double4 x4j, double4 v4j, double4 x4oldj, double4 v4oldj, double rcriti, double rcritj, double rcritvi, double rcritvj, double dt, int i, int j, double *test_d, int2 *encpairs, int &Nenc, int N, double &Ki, double &Kj, double &Kiold, double &Kjold, double &time, int writeEncounters, double writeEncountersRadius, double MinMass){

//if((E == 0 || E >= 2))printf("E %d %d %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", i ,j - N, E, N, x4oldi.x, x4oldi.y, x4oldi.z, v4oldi.x, v4oldi.y, v4oldi.z);
//if(E == 1 && i < j ) printf("E1  %d %d %g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g\n", i, j, time, x4i.w, v4i.w, x4i.x, x4i.y, x4i.z, x4j.w, v4j.w, x4j.x, x4j.y, x4j.z);
//if(E == 1 && i < j ) printf("E1o %d %d %g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g\n", i, j, time, x4oldi.w, v4oldi.w, x4oldi.x, x4oldi.y, x4oldi.z, x4oldj.w, v4oldj.w, x4oldj.x, x4oldj.y, x4oldj.z);

	int Enc = 0;
	if(i < j && (x4i.w > MinMass || x4j.w > MinMass) && x4i.w >= 0.0 && x4j.w >= 0.0){
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
			f = def_cef;
		}
		if(E == 1){
			rcrit = 0.0;
			rcritv = rcriti + rcritj;
			f = 1.0;
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
		else delta1 = 100.0;
		if(0 <= t2 && t2 <= 1){
			t22 = t2*t2;
			tt2 = 1.0-t2;
			tt22 = tt2*tt2;
			delta2 = tt22*(1.0 + 2.0*t2)*d0 + t22*(3.0 - 2.0*t2)*d1 + t2*tt22*c - t22*tt2*cc;
		}
		else delta2 = 100.0;

		delta = min(delta1,delta2);
		if(delta < 0) delta = 0.0;
		
		delta = fmin(delta, d1);
		delta = fmin(delta, d0);

//if((E == 0 || E >= 2))printf("d %d %d %.20g %.20g\n", i, j - N, delta, rcritv);
	
		Kiold = Ki;
		Kjold = Kj;
		Ki = 1.0;
		Kj = 1.0;


		if(delta < f * rcritv*rcritv || Kiold < 1.0){
			Enc = 2;
//if((E == 0 || E >= 2))printf("EE %d %d %g %g %.40g %.40g %.40g %.40g %d\n", i, j - N, x4i.w, x4j.w, x4i.x, x4j.x, v4i.x, v4j.x, E);
//if (E == 1)printf("EE1 %d %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g\n", i, j, x4i.x, x4j.x, x4i.y, x4j.y, x4i.z, x4j.z, delta, rcritv*rcritv, d0, d1);
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

			if(delta <= 0.01 * rcritv*rcritv){
		 		Ki = 0.0;
		 		Kj = 0.0;
			
			}
			else{
				double y = (sqrt(delta) - 0.1 * rcritv)/(0.9*rcritv);
				double yy = y * y;
				Ki = yy / (2.0*yy - 2.0*y + 1.0);
				Kj = yy / (2.0*yy - 2.0*y + 1.0);
			}

		}
		else Enc = 0;
		if(delta < rcrit*rcrit){
			Enc = 1;
		}

//printf("Enc %d %d %g %g\n", i, j, Ki, Kiold);
		if(writeEncounters > 0 && E == 1){
			double writeRadius = 0.0;
			if(writeEncounters == 1){
				//in scales of planetary Radius
				writeRadius = writeEncountersRadius * fmax(v4i.w, v4j.w);

			}
			if(delta < writeRadius * writeRadius){

				double t = 0.0;
				if(0 <= t1 && t1 <= 1) t = t1;
				if(0 <= t2 && t2 <= 1 && delta2 < delta1) t = t2;

				if(t > 0.0 && t < 1.0){
					time = t;
//printf("Enc %g %g %g %d %d\n", t, writeRadius, sqrt(delta), i, j);	
/*					writeEnc_d[ne * 25 + 0] = (time + dt * fmin(t1, t2)/0.01720209895) / 365.25;
*/				}
			}
		}


		return Enc;
	}
	else return 0;
}

// **************************************
//For the multi simulation mode
//This reads all encounter pairs from the prechecker, and calls the encounter function
//to detect close encounter pairs.
//All close encounter pairs are stored in the array Encpairs2_d. 
//The number of close encounter pairs is stored in Nencpairs2_d.
//
// Authors: Simon Grimm
// July  2016
//
// ****************************************
template < int Nmax >
__global__ void encounterM_kernel(double4 *x4_d, double4 *v4_d, double4 *xold_d, double4 *vold_d, double *rcrit_d, double *rcritv_d, double *dt_d, int *Nencpairs_d, int2 *Encpairs_d, int *Nencpairs2_d, int2 *Encpairs2_d, double *test_d, int *index_d, int *NBS_d, int *N_d, int *enccount_d, int si, double FGt, int Nst, double* time_d, int writeEncounters, double writeEncountersRadius, int *StopFlag_d, double MinMass){
	int idy = threadIdx.x;
	int idx = blockIdx.x;
	int id = idx * blockDim.x + idy;

	int ii = 0;
	int jj = 0;

	int st = 0;
	int NBS = 0;
	double dt = 0.0;
	double time = 0.0;

	if(id < Nencpairs_d[0]){
		ii = Encpairs_d[id].x;
		jj = Encpairs_d[id].y;
		if(ii >= 0 && jj >= 0){
			st = index_d[ii] / 100;
			NBS = NBS_d[st];
			dt = dt_d[st];
			time = time_d[st];
		}
	}
	__syncthreads();

	if(id < Nencpairs_d[0] && ii >= 0 && jj >= 0 && st < Nst){
		int enccount = encounter<3>(x4_d[ii], v4_d[ii], xold_d[ii], vold_d[ii], x4_d[jj], v4_d[jj], xold_d[jj], vold_d[jj], rcrit_d[ii], rcrit_d[jj], rcritv_d[ii], rcritv_d[jj], dt * FGt, ii, jj , test_d, Encpairs2_d, Nencpairs2_d[st], 0, time, writeEncounters, writeEncountersRadius, MinMass);
//printf("enc %d %d %d %d %d\n", ii, jj, enccount, st, Nencpairs2_d[st + 1]);
		if(enccount > 0){
			int Ne = atomicAdd(&Nencpairs2_d[st + 1], 1);
#if StopAtEncounter > 0 
			if(enccount == 1){
				N_d[st] = 0;
				StopFlag_d[0] = 1;
			}
#endif
			if(Ne == 0){
				//write a list with simulations containing close encounters
				int NT = atomicAdd(Nencpairs2_d, 1);
				Encpairs_d[NT].y = st;
			}
			if(x4_d[ii].w >= x4_d[jj].w){
				Encpairs2_d[Ne + NBS * Nmax].x = ii;
				Encpairs2_d[Ne + NBS * Nmax].y = jj;
			}
			else{
				Encpairs2_d[Ne + NBS * Nmax].x = jj;
				Encpairs2_d[Ne + NBS * Nmax].y = ii;
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
//
//Authors: Simon Grimm, Joachim Stadel
//March 2014
//
// ****************************************
__global__ void encounter_kernel(double4 *x4_d, double4 *v4_d, double4 *xold_d, double4 *vold_d, double4 *x4G3_d, double4 *v4G3_d, double *rcrit_d, double *rcritv_d, double dt, int *Nencpairs_d, int2 *Encpairs_d, int *Nencpairs2_d, int2 *Encpairs2_d, double *test_d, int *enccount_d, int si, double *K_d, double *Kold_d, double4 *StopTime_d, int NB, double time, int writeEncounters, double writeEncountersRadius, double MinMass){

	int idy = threadIdx.x;
	int idx = blockIdx.x;
	int id = idx * blockDim.x + idy;

	int ii = 0;
	int jj = 0;
	if(id < *Nencpairs_d){
		ii = Encpairs_d[id].x;
		jj = Encpairs_d[id].y;
//printf("%d %d %d\n", ii, jj, id);
	}
	__syncthreads();
	int enccount = 0;
	
	if(id < *Nencpairs_d){
#if G3 == 0
		enccount = encounter<0>(x4_d[ii], v4_d[ii], xold_d[ii], vold_d[ii], x4_d[jj], v4_d[jj], xold_d[jj], vold_d[jj], rcrit_d[ii], rcrit_d[jj], rcritv_d[ii], rcritv_d[jj], dt, ii, jj , test_d, Encpairs2_d, *Nencpairs2_d, 0, time, writeEncounters, writeEncountersRadius, MinMass);
#elif G3 == 1
		enccount = encounterb<0>(x4_d[ii], v4_d[ii], xold_d[ii], vold_d[ii], x4_d[jj], v4_d[jj], xold_d[jj], vold_d[jj], rcrit_d[ii], rcrit_d[jj], rcritv_d[ii], rcritv_d[jj], dt, ii, jj , test_d, Encpairs2_d, *Nencpairs2_d, 0, K_d[ii * NB + jj], K_d[jj * NB + ii], Kold_d[ii * NB + jj], Kold_d[jj * NB + ii], time, writeEncounters, writeEncountersRadius, MinMass);
#else
//change here ii and jj to index[ii], index[jj]
		enccount = encounterG3<0>(x4_d[ii], v4_d[ii], xold_d[ii], vold_d[ii], x4G3_d[ii], v4G3_d[ii], x4_d[jj], v4_d[jj], xold_d[jj], vold_d[jj], x4G3_d[jj], v4G3_d[jj], rcrit_d[ii], ii, jj, rcrit_d[jj], rcritv_d[ii], rcritv_d[jj], dt, ii, jj , test_d, Encpairs2_d, *Nencpairs2_d, 0, K_d[ii * NB + jj], K_d[jj * NB + ii], Kold_d[ii * NB + jj], Kold_d[jj * NB + ii], StopTime_d[ii * NB + jj], StopTime_d[jj * NB + ii], time, MinMass);
#endif
		if(si == 0 && enccount > 0){
			atomicAdd(&enccount_d[ii], 1);
			atomicAdd(&enccount_d[jj], 1);
		}
	}
}


__global__ void groupsmall1_kernel(int *Nencpairssmall_d, int2 *Encpairssmall_d, const int NencMax, int Nsmall){
	int idx = blockIdx.x;
	int id = idx * blockDim.x + threadIdx.x;

	if(id == 0) *Nencpairssmall_d = 0;

	if(id < Nsmall){
		Encpairssmall_d[id * NencMax].y = 1;
		Encpairssmall_d[id * NencMax].x = id;
	}
}

__global__ void groupsmall2_kernel(int *Nencpairssmall2_d, int2 *Encpairssmall2_d, int *Nencsmall_d, int2 *Encpairssmall_d, const int NencMax){
	int idx = blockIdx.x;
	int id = idx * blockDim.x + threadIdx.x;
	if(id < *Nencpairssmall2_d){
		int i = Encpairssmall2_d[id].x;
		int j = Encpairssmall2_d[id].y;
		int Nj = atomicAdd(&Encpairssmall_d[j * NencMax].y, 1);
		Encpairssmall_d[j * NencMax + Nj].x = i;
		if(Nj == 1){
			int Ne = atomicAdd(&Nencsmall_d[0], 1);
			Encpairssmall_d[Ne * NencMax + 1].y = j;
		}
	}
}

__global__ void groupsmall3_kernel(int *Nencsmall_d, int2 *Encpairssmall_d, int2 *Encpairssmall2_d, const int NencMax){
	int idx = blockIdx.x;
	int id = idx * blockDim.x + threadIdx.x;

	if(id < Nencsmall_d[0]){
		int i = Encpairssmall_d[id * NencMax + 1].y;
		int nn = Encpairssmall_d[i * NencMax].y;
		volatile int ne2 = 2;
		if(nn > 0){
			for(volatile int ii = 0; ii < 11; ++ii){
				if(nn <= ne2){
					int Ne = atomicAdd(&Nencsmall_d[ii + 1], 1);
					Encpairssmall2_d[Ne * NencMax + ii].y = i;
					break;
				}
				else{
					ne2 *= 2;
				}
			}
		}
	}
}

// **************************************
//This Kernel sorts all close encounter pairs into independent groups, using a 
//parallel sorting algorithm. 
//this kernel works for the following cases:
// E = 1: less than 512 bodies and less than 512 close encounter pairs
// E = 2: less than 512 bodies and more than 512 close encounter pairs
// E = 3: more than 512 bodies and less than 512 close encounter pairs
// E = 4: more than 512 bodies and more than 512 close encounter pairs
//This kernel works only in the case of more than 512 close encounter pairs, and 
//less than 512 Bodies.
//It classifies the groups into sets of equal sizes.
//The size of group i is stored in Encpairs2_d[i].y, the elements j of the 
//group i are stored in Encpairs2_d[i * BN + j].x
//In Nenc_d[0] is stored the total number of groups.
//in Nenc_d[i] is stored the number of groups with: 2^(2-1) < size of group < 2^(2+1)
//
//This Kernel must be launched only with one block!.
//
//Authors: Simon Grimm
//March  2016
//
// ****************************************
template <int bn, int Bl, int E>
__global__ void group_kernel(int *Nenc_d, double *test_d, int *Nencpairs2_d, int2 *Encpairs2_d, int2 *Encpairs_d, int NencMax, int *groupIndex_d, int BN, int N){

	int idy = threadIdx.x;

	__shared__ volatile int T_s;
	__shared__ int Nenc_s[def_GMax];
	__shared__ int start_s[1];

	int Ne = *Nencpairs2_d;
	int BN2 = BN * BN -1;
	if(BN > 46340) BN2 = 2147483647;	//prevent from overflow

	int2 *A;
	int2 *encpairs;

	int2 *B;
	int2 *B2;

	if(E == 1 || E == 3){//16 b or 1024 b
		__shared__ int2 A_s[Bl];
		__shared__ int2 encpairs_s[Bl];

		if(idy < Ne){
			encpairs_s[idy] = Encpairs2_d[idy];
			A_s[idy] = encpairs_s[idy];
//printf("%d %d %d\n", idy, encpairs_s[idy].x, encpairs_s[idy].y);
		}
		/*encpairs_s[idy] contains the two close encounter pairs*/
		else{
			encpairs_s[idy].x = -1;
			encpairs_s[idy].y = -1;
			A_s[idy] = encpairs_s[idy];
		}

		A = A_s;
		encpairs = encpairs_s;
	}
	if(E == 2 || E == 4){ // 16 c or 1024 c
		A = &Encpairs2_d[Ne];
		encpairs = Encpairs2_d;
		for(int i = 0; i < Ne; i += Bl){
			if(idy + i < Ne){
				A[idy + i] = encpairs[idy + i];
			}
		}
	}

	if(E == 1 || E == 2){ //16
		__shared__ int2 B_s[bn];
		__shared__ int2 B2_s[bn];
		if(idy < bn){
			B_s[idy].x = 0;
			B2_s[idy].x = 0;
			B_s[idy].y = BN2;
			B2_s[idy].y = BN2;
			Encpairs_d[idy].y = 0;
		
		}
		B = B_s;
		B2 = B2_s;
	}
	if(E == 3 || E == 4){ //1024
		B = &Encpairs_d[2 * BN];
		B2 = &Encpairs_d[3 * BN];
		for(int i = 0; i < BN; i += Bl){
			if(idy + i < BN){
				B[idy + i].y = BN2;
				B2[idy + i].y = BN2;
				Encpairs_d[idy + i].y = 0;
			}
		}
	}

	if(idy == 0){
		T_s = 1;
	}
	if(idy < def_GMax) Nenc_s[idy] = 0;
	if(idy == 0) start_s[0] = 0;

	__syncthreads();
	for(int i = 0; i < Ne; i += Bl){
		if(idy + i < Ne){
			//create list of direct close encounter pairs
			volatile int ii = encpairs[idy + i].x;
			volatile int jj = encpairs[idy + i].y;
			volatile int Ni = 0;
			volatile int Nj = 0;
			if(jj < N){
				Ni = atomicAdd(&Encpairs_d[ii].y, 1);
				Encpairs_d[ii * NencMax + Ni].x = jj;
			}
			if(ii < N){
				Nj = atomicAdd(&Encpairs_d[jj].y, 1);
				Encpairs_d[jj * NencMax + Nj].x = ii;
			}
			//Encpairs_d[i].y contains the number of direct encounter paris of body i
			//Encpairs_d[i * NencMax + j].x contains the indeces j of the direct encounter pairs
//printf("%d %d %d %d\n", ii, jj, Ni, Nj);
		}
	}
	__syncthreads();
#if SERIAL_GROUPING == 1
	for(int i = 0; i < BN; i += Bl){
		if(idy + i< BN){
			int Ni = Encpairs_d[idy + i].y;
			int stop = 0;
			while(stop == 0){
				stop = 1;
				for(int j = 0; j < Ni - 1; ++j){
					int jj = Encpairs_d[(idy + i) * NencMax + j].x;
					int jjnext = Encpairs_d[(idy + i) * NencMax + j + 1].x;
				
					if(jjnext < jj){
						//swap
						Encpairs_d[(idy + i) * NencMax + j].x = jjnext;
						Encpairs_d[(idy + i) * NencMax + j + 1].x = jj;
						stop = 0;
					}
				}
			}
			stop = 0;
		}
	}
#endif
	__syncthreads();

	for(int tt = 0; tt < 100; ++tt){ 
		T_s = 0;
		for(int i = 0; i < Ne; i += Bl){
			if(idy + i < Ne){
				int Am = min(A[idy + i].x, A[idy + i].y);
				atomicMin(&B[A[idy + i].y].y, Am);
				atomicMin(&B[A[idy + i].x].y, Am);
			}
		}
		__syncthreads();

		for(int i = 0; i < BN; i += Bl){
			if(idy + i< BN){
				if(B[idy + i].y < BN2) B2[idy + i].y = B[B[idy + i].y].y;
			}
		}
		__syncthreads();
		for(int i = 0; i < Ne; i += Bl){
			if(idy + i < Ne){
				A[idy + i].x = B2[encpairs[idy + i].x].y;
				A[idy + i].y = B2[encpairs[idy + i].y].y;
				if(A[idy + i].x != A[idy + i].y) T_s = 1;
			}
		}
		for(int i = 0; i < BN; i += Bl){
			if(idy + i < BN){
				B[idy + i].y = B2[idy + i].y;
			}
		}
		__syncthreads();
		if(T_s == 0){
//if(idy == 0) printf("%d\n", tt);
			 break;
		}
		__syncthreads();

	}
	// *At this point B[idy] contains the smallest index of the group* /
	__syncthreads();

	for(int i = 0; i < BN; i += Bl){
		if(idy + i < BN) B2[idy + i].y = -1;
//printf("B %d %d\n", idy + i, B[idy + i].y);
	}
	__syncthreads();
	// *Check now for new groups and increase the total number of groups* /
	for(int i = 0; i < BN; i += Bl){
		if(idy + i < BN){
			if(B[idy + i].y == idy + i){
				B2[idy + i].y = atomicAdd(&Nenc_s[0],1);
			}		
		}
	}
	__syncthreads();
	// *Transform now the smallest index of the group into a consecutive group index* /
	for(int i = 0; i < BN; i += Bl){
		if(idy + i < BN){
			if(B[idy + i].y < BN2) B[idy + i].y = B2[B[idy + i].y].y;
			Encpairs2_d[idy + i].y = 0;
		}
	}
	// *At this point B[idy] contains a consecutive group index* /
	__syncthreads();

//for(int i = 0; i < BN; i += Bl){
//	if(idy + i < BN){
//		if(B[idy + i].y < BN2){
//printf("B %d %d %d\n", idy + i, B[idy + i].y, B2[idy + i].y);
//		}		
//	}
//}

#if G3 > 0
	for(int i = 0; i < BN; i += Bl){
		if(idy + i < BN){
			groupIndex_d[idy + i] = B[idy + i].y;
		}
	}
#endif
#if SERIAL_GROUPING == 0
	for(int i = 0; i < BN; i += Bl){
		if(idy + i < BN){
			if(B[idy + i].y < BN2){
				int Ns = atomicAdd(&Encpairs2_d[B[idy + i].y].y,1);
				B2[idy + i].y = Ns; //index in the group
				Encpairs_d[BN + idy + i].y = B2[idy + i].y;
			}
		// *At this point Encpairs2_d.x contains now line by line the members of the groups, Encpairs2_s.y contains the sizes of the groups* /
		}
	}
#endif
#if SERIAL_GROUPING == 1
	if(idy == 0){
		for(int i = BN - 1; i >=0; --i){
			if(B[i].y < BN2){
				int Ns = atomicAdd(&Encpairs2_d[B[i].y].y,1);
				B2[i].y = Ns;   //index in the group
				Encpairs_d[BN + i].y = B2[i].y;
			}
		}
	}
#endif
	__syncthreads();
	for(int i = 0; i < Nenc_s[0]; i += Bl){
		if(idy + i < Nenc_s[0]){
			if(Encpairs2_d[idy + i].y > 0){
				int start = atomicAdd(&start_s[0], Encpairs2_d[idy + i].y);
				Encpairs2_d[BN + idy + i].y = start; //starting points of te groups
//printf("start %d %d %d %d\n", idy + i, Encpairs2_d[idy + i].y, start_s[0], start);
			}
		}
	}
	__syncthreads();
	for(int i = 0; i < BN; i += Bl){
		if(idy + i < BN){
			if(B[idy + i].y < BN2){
				int n = B2[idy + i].y;
				int start = Encpairs2_d[BN + B[idy + i].y].y;
				Encpairs2_d[start + n].x = idy + i;
			}
		// *At this point Encpairs2_d.x contains now members of the groups, Encpairs2_d.y contains the sizes of the groups* /
		}
	}
	__syncthreads();

	for(int i = 0; i < BN; i += Bl){
		if(idy + i < BN){
			int nn = Encpairs2_d[idy + i].y;
			volatile int ne2 = 2;
			if(nn > 0){
				for(volatile int ii = 0; ii < def_GMax - 1; ++ii){
					if(nn <= ne2){
						int Ns = atomicAdd(&Nenc_s[ii + 1],1);
//printf("G %d %d\n", ii + 1, Nenc_s[ii + 1]);
						Encpairs2_d[ (ii+2) * BN + Ns].y = idy + i;
						break;
					} 
					else{
						ne2 *= 2;
					}
				}
			}
		}
	}
	__syncthreads();

	if(idy < def_GMax){
		Nenc_d[idy] = Nenc_s[idy];
	}

}


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

	if(idy == 0){
		T_s = 1;
		Nenc_s = 0;
		
	}
	__syncthreads();

	if(idy < Ne){ 
		encpairs_s[idy].x = Encpairs2_d[idy + NBS * NmaxM].x - NBS;
		encpairs_s[idy].y = Encpairs2_d[idy + NBS * NmaxM].y - NBS;
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
			B2_s[idy] = atomicAdd(&Nenc_s,1);
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

__global__ void groupM2_kernel(int2 *Encpairs_d, int2 *Encpairs2_d, int *Nenc_d, int *NBS_d, int *N_d, int Nst){

	int idy = threadIdx.x;
	int idx = blockIdx.x;

	int st = Encpairs_d[idx].y;

	int NBS = NBS_d[st];
	int N = N_d[st];

	if(idy < N){

		int nn = Encpairs_d[idy + NBS + Nst].y;
//printf("n %d %d %d %d\n", st, idy, nn, NBS);

		volatile int ne2 = 2;
		if(nn > 0){
//printf("nn %d %d %d %d %d %d\n", st, idy, nn, Encpairs_d[(idy + NBS)* 16].x, Encpairs_d[((idy  + NBS)* 16)+ 1].x, Encpairs_d[((idy + NBS) * 16) + 2].x);
			for(volatile int ii = 0; ii < def_GMax - 1; ++ii){
				if(nn <= ne2){
					Encpairs2_d[ (ii+1) + NmaxM * atomicAdd(&Nenc_d[ii + 1],1)].y = idy + NBS;
					break;
				} 
				else{
					ne2 *= 2;
				}
			}
		}
	}
}
#endif

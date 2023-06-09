#ifndef ENCOUNTER_H
#define ENCOUNTER_H
#include "Orbit2.h"

#if def_G3 ==1
#include "Encounter3G3.h"
#endif


__global__ void setNencpairs_kernel(int *Nencpairs_d, const int N){

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < N){
		Nencpairs_d[id] = 0;
	}
}
__global__ void setNencpairs2_kernel(int2 *Nencpairs_d, const int N){

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < N){
		Nencpairs_d[id].x = 0;
		Nencpairs_d[id].y = 0;
	}
}


// **************************************
//This function estimates the minimal separation of two bodies
//during a time step, using a third order interpolation. 
//
//The interpolation scheme is based on the mercury code from Chambers.
//
//If the minimal separation is less than the critical radius, a
//close encounter is reported.
//
//
// Authors: Simon Grimm
// April 2016
//
// ****************************************
__device__ int encounter(const double4 x4i, const double4 v4i, const double4 x4oldi, const double4 v4oldi, const double4 x4j, const double4 v4j, const double4 x4oldj, const double4 v4oldj, const double rcriti, const double rcritj, const double rcritvi, const double rcritvj, const double dt, const int i, const int j, double &time, const double MinMass){

//printf("E0 %d %d %.20g %.20g %.20g %.20g %.20g %.20g | %.20g %.20g %.20g %.20g %.20g %.20g | m %.20g %.20g\n", i ,j, x4oldi.x, x4oldi.y, x4oldi.z, v4oldi.x, v4oldi.y, v4oldi.z, x4oldj.x, x4oldj.y, x4oldj.z, v4oldj.x, v4oldj.y, v4oldj.z, x4oldi.w, x4oldj.w);
//printf("E1 %d %d %.20g %.20g %.20g %.20g %.20g %.20g | %.20g %.20g %.20g %.20g %.20g %.20g | m %.20g %.20g\n", i ,j, x4i.x, x4i.y, x4i.z, v4i.x, v4i.y, v4i.z, x4j.x, x4j.y, x4j.z, v4j.x, v4j.y, v4j.z, x4i.w, x4j.w);
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
		double f;
	
		rcrit = fmax(rcriti, rcritj);
		rcritv = fmax(rcritvi, rcritvj);
		f = def_cef;

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

//printf("dt %d %d %g %g %g\n", i, j, t1, t2, dt);

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

		delta = fmin(delta1,delta2);
		if(delta < 0) delta = 0.0;
		
		delta = fmin(delta, d1);
		delta = fmin(delta, d0);
//printf("EE %d %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %g %g %g %g %g %g\n", i, j, time, x4i.w, x4j.w, x4i.x, x4i.y, x4i.z, x4j.x, x4j.y, x4j.z, delta, rcritv*rcritv, d0, d1, delta1, delta2, t1, t2);

		if(delta < f * rcritv*rcritv){
			Enc = 2;
//printf("EE %d %d %g %g %.40g %.40g %.40g %.40g %g %g\n", i, j, x4i.w, x4j.w, x4i.x, x4i.y, v4i.z, v4j.x, v4j.y, v4j.z);
		}
		else Enc = 0;

		if(delta < rcrit*rcrit){
			Enc = 1;
		}

		return Enc;
	}
	else return 0;
}
__device__ int encounterb(const double4 x4i, const double4 v4i, const double4 x4oldi, const double4 v4oldi, const double4 x4j, const double4 v4j, const double4 x4oldj, const double4 v4oldj, const double rcriti, const double rcritj, const double rcritvi, const double rcritvj, const double dt, const int i, const int j, double &Ki, double &Kj, double &Kiold, double &Kjold, double &time, const double MinMass){

//printf("E %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", i , j, x4oldi.x, x4oldi.y, x4oldi.z, v4oldi.x, v4oldi.y, v4oldi.z);

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
		double f;
	
		rcrit = fmax(rcriti, rcritj);
		rcritv = fmax(rcritvi, rcritvj);
		f = def_cef;

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

		delta = fmin(delta1,delta2);
		if(delta < 0) delta = 0.0;
		
		delta = fmin(delta, d1);
		delta = fmin(delta, d0);

//printf("d %d %d %.20g %.20g\n", i, j, delta, rcritv);
	
		Kiold = Ki;
		Kjold = Kj;
		Ki = 1.0;
		Kj = 1.0;


		if(delta < f * rcritv*rcritv || Kiold < 1.0){
			Enc = 2;
//printf("EE %d %d %g %g %.40g %.40g %.40g %.40g\n", i, j, x4i.w, x4j.w, x4i.x, x4j.x, v4i.x, v4j.x);

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

		return Enc;
	}
	else return 0;
}


// This function returns delta, the square of the minimal distanz
// It sets colt, the collision time  0 < colt < 1.
// It sets enct, the time of smallest distance. 0 < enct < 1
__device__ double encounter1(const double4 x4i, const double4 v4i, const double4 x4oldi, const double4 v4oldi, const double4 x4j, const double4 v4j, const double4 x4oldj, const double4 v4oldj, const double rcrit, const double dt, const int i, const int j, double &enct, double &colt, const double MinMass, const int noColl){

//printf("E1o %d %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g\n", i, j, x4oldi.w, v4oldi.w, x4oldi.x, x4oldi.y, x4oldi.z, x4oldj.w, v4oldj.w, x4oldj.x, x4oldj.y, x4oldj.z);
	if(i != j && (x4i.w > MinMass || x4j.w > MinMass) && x4i.w >= 0.0 && x4j.w >= 0.0){

		double d0, d1, dd0, dd1;
		double3 r;
		double3 rd;

		r.x = x4j.x - x4i.x;
		r.y = x4j.y - x4i.y;
		r.z = x4j.z - x4i.z;
		d1 = r.x*r.x + r.y*r.y+ r.z*r.z;

		rd.x = v4j.x - v4i.x;
		rd.y = v4j.y - v4i.y;
		rd.z = v4j.z - v4i.z;

		dd1 = (r.x*rd.x + r.y*rd.y+ r.z*rd.z) * 2.0;

		r.x = x4oldj.x - x4oldi.x;
		r.y = x4oldj.y - x4oldi.y;
		r.z = x4oldj.z - x4oldi.z;
		d0 = r.x*r.x + r.y*r.y+ r.z*r.z;
			
		rd.x = v4oldj.x - v4oldi.x;
		rd.y = v4oldj.y - v4oldi.y;
		rd.z = v4oldj.z - v4oldi.z;

		dd0 = (r.x*rd.x + r.y*rd.y+ r.z*rd.z) * 2.0;

		double t1, t2;
		t1 = 6.0 *(d0-d1); 
		double a = t1 + 3.0 * dt * (dd0 + dd1);
		double b = -t1 - 2.0 * dt * (2.0 * dd0 + dd1);
		double c = dt*dd0;

		double sgnb = 1.0;
		if(b < 0){
			sgnb = -1.0;
		}
		t1 = 0.0;
		t2 = 0.0;

		double w = b*b - 4.0*a*c;
		if(w < 0.0) w = 0.0;
		if( b != 0){
			double q = -0.5 * (b + sgnb * sqrt(w));
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
//if(i == 12888 && j == 11191) printf("d0d1  %d %d %g %g | %.12g %.12g %.12g | noColl: %d\n", i, j, t1, t2, rcrit, sqrt(d0), sqrt(d1), noColl);
//if(i == 4969 && j == 530) printf("d0d1  %d %d %g %g | %.12g %.12g %.12g | noColl: %d\n", i, j, t1, t2, rcrit, sqrt(d0), sqrt(d1), noColl);
		double delta = 100.0;
		if(0 <= t1 && t1 <= 1){
			double t12 = t1*t1;
			double tt1 = 1.0-t1;
			double tt12 = tt1*tt1;
			double delta1 = tt12*(1.0 + 2.0*t1)*d0 + t12*(3.0 - 2.0*t1)*d1 + t1*tt12*dt*dd0 - t12*tt1*dt*dd1;
			delta = fmin(delta, delta1);
			enct = t1;
		}
		if(0 <= t2 && t2 <= 1){
			double t22 = t2*t2;
			double tt2 = 1.0-t2;
			double tt22 = tt2*tt2;
			double delta2 = tt22*(1.0 + 2.0*t2)*d0 + t22*(3.0 - 2.0*t2)*d1 + t2*tt22*dt*dd0 - t22*tt2*dt*dd1;
			delta = fmin(delta, delta2);
			enct = t2;
		}
		if(delta < 0) delta = 0.0;
	
		delta = fmin(delta, d1);
		delta = fmin(delta, d0);

//if(enct >= 0.0 && enct <= 1.0) printf("dt %d %d %g %g %g %g %g\n", i, j, sqrt(delta), enct, t1, t2, rcrit);
		double rcritsq = rcrit * rcrit;

		if(delta < rcritsq){
//printf("EEa %d %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %g %g %g %g %g %g\n", i, j, x4i.w, x4j.w, x4i.x, x4i.y, x4i.z, x4j.x, x4j.y, x4j.z, delta, rcritsq, d0, d1, delta, t1, t2, MinMass);
			if(d0 == d1){
				colt = 0.0;
			}
			else{
				double coltime = (rcritsq - d0) / (d1 - d0);
				
				if(noColl == 1){
					//only report entering collisions, not leaving collisions
					if(d1 <= rcritsq && d0 > rcritsq){
						//linear interpolation of the collision time
						colt = coltime;
					}
		
					//find collision in between of time steps	
					if(d0 >= rcritsq && d1 >= rcritsq){
						colt = fmin(t1, t2);
					}

					if(d0 < 0.95 * rcritsq && d1 < 0.95 * rcritsq){
//printf("%d %d are already overlapping %g %g %g\n", i, j, sqrt(d0), sqrt(d1), rcrit);
						colt = 200.0;
						return 100.0;
					}

				}
				else if(noColl == -1){
					//only report leaving collisions, not entering collisions
					if(d0 <= rcritsq && d1 > rcritsq){
						//linear interpolation of the collision time
						colt = coltime;
					}
				}
				else{
					if((d0 >= rcritsq && d1 < rcritsq) || (d1 >= rcritsq && d0 < rcritsq)){
						//linear interpolation of the collision time
						colt = coltime;
					}


					//find collision in between of time steps	
					if(d0 >= rcritsq && d1 >= rcritsq){
						if(t1 < 0.0) t1 = t2;
						if(t2 < 0.0) t2 = t1;
						if(t1 >= 0.0){
							colt = fmin(t1, t2);
						}
					}


				}
//printf("colt  %d %d %.12g %.12g %.12g | noColl: %d\n", i, j, coltime, colt, enct, noColl);
			}
		}
		return delta;

	}
	else return 100.0;
}

// **************************************
// For the multi simulation mode
// This reads all encounter pairs from the prechecker, and calls the encounter function
// to detect close encounter pairs.
// All close encounter pairs are stored in the array Encpairs2_d. 
// The number of close encounter pairs is stored in Nencpairs2_d.
// Creates a list of simulations containing close encounters
//
// Authors: Simon Grimm
// July  2016
//
// ****************************************
__global__ void encounterM_kernel(double4 *x4_d, double4 *v4_d, double4 *xold_d, double4 *vold_d, double *rcrit_d, double *rcritv_d, double *dt_d, int *Nencpairs_d, int2 *Encpairs_d, int *Nencpairs2_d, int2 *Encpairs2_d, int *index_d, int *NBS_d, unsigned int *enccount_d, const int si, const double FGt, const int Nst, double* time_d, const int StopAtEncounter, int *Ncoll_d, double *n1_d, const double MinMass, const int NencMax){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

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
			st = index_d[ii] / def_MaxIndex;
			NBS = NBS_d[st];
			dt = dt_d[st];
			time = time_d[st];
//printf("encA %d %d %d %d %d %d\n", ii, jj, st, index_d[ii], index_d[jj], NBS);
		}
	}
	__syncthreads();

	if(id < Nencpairs_d[0] && ii >= 0 && jj >= 0 && st < Nst){
		int enccount = encounter(x4_d[ii], v4_d[ii], xold_d[ii], vold_d[ii], x4_d[jj], v4_d[jj], xold_d[jj], vold_d[jj], rcrit_d[ii], rcrit_d[jj], rcritv_d[ii], rcritv_d[jj], dt * FGt, ii, jj , time, MinMass);
//printf("enc %d %d %d %d %d\n", ii, jj, enccount, st, Nencpairs2_d[st + 1]);
		if(enccount > 0){
			int Ne = atomicAdd(&Nencpairs2_d[st + 1], 1);
//printf("encB %d %d %d %d %d %d %d\n", ii, jj, st, index_d[ii], index_d[jj], NBS, Ne);

			if(StopAtEncounter > 0){ 
				if(enccount == 1){
					Ncoll_d[0] = 1;
					n1_d[st] = -1.0;
				}
			}
			if(Ne == 0){
				//write a list with simulations containing close encounters
				int NT = atomicAdd(Nencpairs2_d, 1);
				Encpairs_d[NT].y = st;
//printf("NT %d %d\n", st, NT);
			}
			if(x4_d[ii].w >= x4_d[jj].w){
				Encpairs2_d[Ne + NBS * NencMax].x = ii;
				Encpairs2_d[Ne + NBS * NencMax].y = jj;
			}
			else{
				Encpairs2_d[Ne + NBS * NencMax].x = jj;
				Encpairs2_d[Ne + NBS * NencMax].y = ii;
			}
		}
		if(si == 0 && enccount > 0){
			atomicAdd(&enccount_d[ii], 1);
			atomicAdd(&enccount_d[jj], 1);
		}
	}
}

// **************************************
// Does not creates a list of simulations containing close encounters
// All simulations call the group kernel
// That is needed because in that way the full grouping algorithm can be applied, not just M1 grouping
//
// Authors: Simon Grimm
// March  2023
//
// ****************************************
__global__ void encounterM3_kernel(double4 *x4_d, double4 *v4_d, double4 *xold_d, double4 *vold_d, double *rcrit_d, double *rcritv_d, double *dt_d, int Nencpairs, int *Nencpairs_d, int2 *Encpairs_d, int *Nencpairs2_d, int2 *Encpairs2_d, int *index_d, int *NBS_d, unsigned int *enccount_d, const int si, const double FGt, const int Nst, double* time_d, const int StopAtEncounter, int *Ncoll_d, double *n1_d, const double MinMass, const int NencMax){

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	int ii = 0;
	int jj = 0;

	int st = 0;
	int NBS = 0;
	double dt = 0.0;
	double time = 0.0;

	if(id < Nencpairs){
		ii = Encpairs_d[id].x;
		jj = Encpairs_d[id].y;
		if(ii >= 0 && jj >= 0){
			st = index_d[ii] / def_MaxIndex;
			NBS = NBS_d[st];
			dt = dt_d[st];
			time = time_d[st];
//printf("encA %d %d %d %d %d %d\n", ii, jj, st, index_d[ii], index_d[jj], NBS);
		}
	}
	__syncthreads();

	if(id < Nencpairs && ii >= 0 && jj >= 0 && st < Nst){
		int enccount = encounter(x4_d[ii], v4_d[ii], xold_d[ii], vold_d[ii], x4_d[jj], v4_d[jj], xold_d[jj], vold_d[jj], rcrit_d[ii], rcrit_d[jj], rcritv_d[ii], rcritv_d[jj], dt * FGt, ii, jj , time, MinMass);
//printf("enc %d %d %d %d %d\n", ii, jj, enccount, st, Nencpairs2_d[st + 1]);
		if(enccount > 0){
			int Ne = atomicAdd(&Nencpairs2_d[st + 1], 1);
//printf("encB %d %d %d %d %d %d %d\n", ii, jj, st, index_d[ii], index_d[jj], NBS, Ne);

			if(Ne == 0){
				atomicAdd(Nencpairs2_d, 1);
			}

			if(StopAtEncounter > 0){ 
				if(enccount == 1){
					Ncoll_d[0] = 1;
					n1_d[st] = -1.0;
				}
			}
			if(x4_d[ii].w >= x4_d[jj].w){
				Encpairs2_d[Ne + NBS * NencMax].x = ii - NBS;
				Encpairs2_d[Ne + NBS * NencMax].y = jj - NBS;
			}
			else{
				Encpairs2_d[Ne + NBS * NencMax].x = jj - NBS;
				Encpairs2_d[Ne + NBS * NencMax].y = ii - NBS;
			}
		}
		if(si == 0 && enccount > 0){
			atomicAdd(&enccount_d[ii], 1);
			atomicAdd(&enccount_d[jj], 1);
		}
		if(id == 0){
			Nencpairs_d[0] = 0;
		}
	}
}

// **************************************
//This kernels reads all encounter pairs from the prechecker, and calls the encounter function
//to detect close encounter pairs.
//All close encounter pairs are stored in the array Encpairs2_d. 
//The number of close encounter pairs is stored in Nencpairs2_d.
//
//Authors: Simon Grimm
//April 2019
//
// ****************************************
__global__ void encounter_kernel(double4 *x4_d, double4 *v4_d, double4 *xold_d, double4 *vold_d, double *rcrit_d, double *rcritv_d, const double dt, int Nencpairs, int *Nencpairs_d, int2 *Encpairs_d, int *Nencpairs2_d, int2 *Encpairs2_d, unsigned int *enccount_d, const int si, const int NB, double time, const int StopAtEncounter, int *Ncoll_d, const double MinMass){

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < Nencpairs){
		int ii = Encpairs_d[id].x;
		int jj = Encpairs_d[id].y;
//printf("%d %d %d\n", ii, jj, id);
		int enccount = 0;	

#if def_G3 == 0
		enccount = encounter(x4_d[ii], v4_d[ii], xold_d[ii], vold_d[ii], x4_d[jj], v4_d[jj], xold_d[jj], vold_d[jj], rcrit_d[ii], rcrit_d[jj], rcritv_d[ii], rcritv_d[jj], dt, ii, jj , time, MinMass);
#elif def_G3 == 1
		enccount = encounterb(x4_d[ii], v4_d[ii], xold_d[ii], vold_d[ii], x4_d[jj], v4_d[jj], xold_d[jj], vold_d[jj], rcrit_d[ii], rcrit_d[jj], rcritv_d[ii], rcritv_d[jj], dt, ii, jj , K_d[ii * NB + jj], K_d[jj * NB + ii], Kold_d[ii * NB + jj], Kold_d[jj * NB + ii], time, MinMass);
#else
//change here ii and jj to index[ii], index[jj]
		enccount = encounterG3(x4_d[ii], v4_d[ii], xold_d[ii], vold_d[ii], x4G3_d[ii], v4G3_d[ii], x4_d[jj], v4_d[jj], xold_d[jj], vold_d[jj], x4G3_d[jj], v4G3_d[jj], rcrit_d[ii], ii, jj, rcrit_d[jj], rcritv_d[ii], rcritv_d[jj], dt, ii, jj , Encpairs2_d, *Nencpairs2_d, 0, K_d[ii * NB + jj], K_d[jj * NB + ii], Kold_d[ii * NB + jj], Kold_d[jj * NB + ii], time, MinMass);
#endif
		if(enccount > 0){
#if def_CPU == 0
			int Ne = atomicAdd(Nencpairs2_d, 1);
#else
			int Ne;
			#pragma omp atomic capture
			Ne = Nencpairs2_d[0]++;
#endif
			if(StopAtEncounter > 0){
				if(enccount == 1){
					Ncoll_d[0] = 1;
				}
			}

			if(x4_d[ii].w >= x4_d[jj].w){
				Encpairs2_d[Ne].x = ii;
				Encpairs2_d[Ne].y = jj;
			}
			else{
				Encpairs2_d[Ne].x = jj;
				Encpairs2_d[Ne].y = ii;
			}

// *****************
//dont group test particles
/*
			if(x4_d[ii].w == 0.0){
				Encpairs2_d[Ne].x = ii;
				Encpairs2_d[Ne].y = jj;
			}
			if(x4_d[jj].w == 0.0){
				Encpairs2_d[Ne].x = jj;
				Encpairs2_d[Ne].y = ii;
			}
*/
// *****************

		}
		if(si == 0 && enccount > 0){
#if def_CPU == 0
			atomicAdd(&enccount_d[ii], 1);
			atomicAdd(&enccount_d[jj], 1);
#else
			#pragma omp atomic
			enccount_d[ii]++;
			#pragma omp atomic
			enccount_d[jj]++;
#endif
		}
		if(id == 0){
			Nencpairs_d[0] = 0;
		}
	}
}

// **************************************
// This kernels reads all encounter pairs between test particles. It calls the encounter function
// to detect close encounter pairs. Close encounters are reported in the writeencounters file.
//
// Authors: Simon Grimm
// October 2022
// ****************************************
__global__ void encounter_small_kernel(double4 *x4_d, double4 *v4_d, double4 *xold_d, double4 *vold_d, int *index_d, double4 *spin_d, const double dt, int Nencpairs2, int2 *Encpairs2_d, int *NWriteEnc_d, double *writeEnc_d, double time){

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < Nencpairs2){

		int ii = Encpairs2_d[id].x;
		int jj = Encpairs2_d[id].y;
		if(ii < jj){
			int swap = jj;
			jj = ii;
			ii = swap;
		}
//printf("%d %d %d\n", ii, jj, id);


		double delta = 1000.0;
		double enct = 100.0;
		double colt = 100.0;
		double rcrit = WriteEncountersRadius_c[0] * fmax(v4_d[ii].w, v4_d[jj].w); //writeradius

		delta = encounter1(x4_d[ii], v4_d[ii], xold_d[ii], vold_d[ii], x4_d[jj], v4_d[jj], xold_d[jj], vold_d[jj], rcrit, dt, ii, jj, enct, colt, 0.0, 0);

		if(delta < rcrit * rcrit){
			if(enct > 0.0 && enct < 1.0){
//printf("Write Enc %g %g %g %g %g %d %d\n", dt / dayUnit, rcrit, sqrt(delta), enct, colt, ii, jj);
#if def_CPU == 0
				int ne = atomicAdd(NWriteEnc_d, 1);
#else
				int ne;
				#pragma omp atomic capture
				ne = NWriteEnc_d[0]++;
#endif
				if(ne >= def_MaxWriteEnc - 1) ne = def_MaxWriteEnc - 1;
				if(colt > 1.0) colt = 1.0;
				storeEncounters(x4_d, v4_d, ii, jj, ii, jj, index_d, ne, writeEnc_d, time + (colt * dt - dt) / dayUnit, spin_d);
			}
		}
	}
}


#if def_CPU == 1
void group_cpu(int *Nenc_m, int *Nencpairs2_h, int2 *Encpairs2_h, int2 *Encpairs_h, const int NencMax, const int NT, const int N, const int SERIAL_GROUPING){


	int T_s;
	int start_s[1];

	int Ne = *Nencpairs2_d;
	

	int BN2 = NT * NT -1;
	if(NT > 46340) BN2 = 2147483647;	//prevent from overflow

	int2 *A;
	int2 *encpairs;

	int2 *B;
	int2 *B2;

	A = &Encpairs2_d[Ne];
	encpairs = Encpairs2_d;
	for(int i = 0; i < Ne; ++i){
		A[i] = encpairs[i];
	}

	B = &Encpairs_d[2 * NT];
	B2 = &Encpairs_d[3 * NT];
	for(int i = 0; i < NT; ++i){
		B[i].y = BN2;
		B2[i].y = BN2;
		Encpairs_d[i].y = 0;
	}

	T_s = 1;
	start_s[0] = 0;

	for(int i = 0; i < def_GMax; ++i){
		Nenc_m[i] = 0;
	}

	for(int i = 0; i < Ne; ++i){
		//create list of direct close encounter pairs
		volatile int ii = encpairs[i].x;
		volatile int jj = encpairs[i].y;
		volatile int Ni = 0;
		volatile int Nj = 0;
		if(jj < N){
			Ni = Encpairs_d[ii].y++;
			Encpairs_d[ii * NencMax + Ni].x = jj;
		}
		if(ii < N){
			Nj = Encpairs_d[jj].y++;
			Encpairs_d[jj * NencMax + Nj].x = ii;
		}
		//Encpairs_d[i].y contains the number of direct encounter pairs of body i
		//Encpairs_d[i * NencMax + j].x contains the indeces j of the direct encounter pairs
//printf("Encpairs %d %d %d %d\n", ii, jj, Ni, Nj);
	}
	if(SERIAL_GROUPING == 1){
		for(int i = 0; i < NT; ++i){
			int Ni = Encpairs_d[i].y;
			int stop = 0;
			while(stop == 0){
				stop = 1;
				for(int j = 0; j < Ni - 1; ++j){
					int jj = Encpairs_d[i * NencMax + j].x;
					int jjnext = Encpairs_d[i * NencMax + j + 1].x;
				
					if(jjnext < jj){
						//swap
						Encpairs_d[i * NencMax + j].x = jjnext;
						Encpairs_d[i * NencMax + j + 1].x = jj;
						stop = 0;
					}
				}
			}
			stop = 0;
		}
	}

	for(int tt = 0; tt < 100; ++tt){ 
		T_s = 0;
		for(int i = 0; i < Ne; ++i){
			int Am = min(A[i].x, A[i].y);
			B[A[i].y].y = min(B[A[i].y].y, Am);			
			B[A[i].x].y = min(B[A[i].x].y, Am);			
		}

		for(int i = 0; i < NT; ++i){
			if(B[i].y < BN2) B2[i].y = B[B[i].y].y;
		}
		for(int i = 0; i < Ne; ++i){
			A[i].x = B2[encpairs[i].x].y;
			A[i].y = B2[encpairs[i].y].y;
			if(A[i].x != A[i].y) T_s = 1;
		}
		for(int i = 0; i < NT; ++i){
			B[i].y = B2[i].y;
		}
		if(T_s == 0){
			 break;
		}

	}
	// At this point B[i] contains the smallest index of the group

	for(int i = 0; i < NT; ++i){
		B2[i].y = -1;
//printf("B %d %d\n", i, B[i].y);
	}
	// Check now for new groups and increase the total number of groups
	for(int i = 0; i < NT; ++i){
		if(B[i].y == i){
			B2[i].y = Nenc_m[0]++;
		}		
	}
	// Transform now the smallest index of the group into a consecutive group index
	for(int i = 0; i < NT; ++i){
		if(B[i].y < BN2) B[i].y = B2[B[i].y].y;
		Encpairs2_d[i].y = 0;
	}
	// At this point B[i] contains a consecutive group index

//for(int i = 0; i < NT; ++i){
//	if(B[i].y < BN2){
//printf("B %d %d %d\n", i, B[i].y, B2[i].y);
//	}		
//}

	if(SERIAL_GROUPING == 0){
		for(int i = 0; i < NT; ++i){
			if(B[i].y < BN2){
				int Ns = Encpairs2_d[B[i].y].y++;
				B2[i].y = Ns; //index in the group
				Encpairs_d[NT + i].y = B2[i].y;
			}
			// At this point Encpairs2_d.x contains now line by line the members of the groups, Encpairs2_s.y contains the sizes of the groups
		}
	}
	if(SERIAL_GROUPING == 1){
		for(int i = NT - 1; i >=0; --i){
			if(B[i].y < BN2){
				int Ns = Encpairs2_d[B[i].y].y++;
				B2[i].y = Ns;	//index in the group
				Encpairs_d[NT + i].y = B2[i].y;
			}
		}
	}
	for(int i = 0; i < Nenc_m[0]; ++i){
		if(Encpairs2_d[i].y > 0){
			int start = start_s[0];
			start_s[0] += Encpairs2_d[i].y;
			Encpairs2_d[NT + i].y = start; //starting points of te groups
//printf("start %d %d %d %d\n", i, Encpairs2_d[i].y, start_s[0], start);
		}
	}
	for(int i = 0; i < NT; ++i){
		if(B[i].y < BN2){
			int n = B2[i].y;
			int start = Encpairs2_d[NT + B[i].y].y;
			Encpairs2_d[start + n].x = i;
//printf("members %d %d %d\n", start, n, i);
		}
		// At this point Encpairs2_d.x contains now members of the groups, Encpairs2_d.y contains the sizes of the groups/
	}

	for(int i = 0; i < NT; ++i){
		int nn = Encpairs2_d[i].y;
//if(nn > 0) printf("n %d %d\n", i, nn);
		volatile int ne2 = 2;
		if(nn > 0){
			for(volatile int ii = 0; ii < def_GMax - 1; ++ii){
				if(nn <= ne2){
					int Ns = Nenc_m[ii + 1]++;
//printf("G %d %d\n", ii + 1, Nenc_m[ii + 1]);
					Encpairs2_d[ (ii+2) * NT + Ns].y = i;
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

// **************************************
//This Kernel sorts all close encounter pairs into independent groups, using a 
//parallel sorting algorithm. 
//this kernel works for the following cases:
// E = 1: less than 512 bodies and less than 512 close encounter pairs
// E = 2: less than 512 bodies and more than 512 close encounter pairs
// E = 3: more than 512 bodies and less than 512 close encounter pairs
// E = 4: more than 512 bodies and more than 512 close encounter pairs
//It classifies the groups into sets of equal sizes.
//The size of group i is stored in Encpairs2_d[i].y, the elements j of the 
//group i are stored in Encpairs2_d[i * N + j].x
//In Nenc_d[0] is stored the total number of groups.
//in Nenc_d[i] is stored the number of groups with: 2^(2-1) < size of group < 2^(2+1)
//
//This Kernel must be launched with only one block!.
//
//Author: Simon Grimm
//March  2016
// ****************************************
template <int bn, int Bl>
__global__ void group_kernel(int *Nenc_d, int *Nencpairs2_d, int2 *Encpairs2_d, int2 *Encpairs_d, const int NencMax, const int NT, const int N, const int SERIAL_GROUPING){

	int idy = threadIdx.x;

	__shared__ volatile int T_s;
	__shared__ int Nenc_s[def_GMax];
	__shared__ int start_s[1];

	int Ne = *Nencpairs2_d;
	
	int E;

	if(NT <= 512){
		if(Ne < Bl){
			E = 1;
		}
		else{
			E = 2;
		}
	}
	else{
		if(Ne < Bl){
			E = 3;
		}
		else{
			E = 4;
		}
	}
//printf("E %d %d %d\n", bn, Bl, E);

	int BN2 = NT * NT -1;
	if(NT > 46340) BN2 = 2147483647;	//prevent from overflow

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
		// encpairs_s[idy] contains the two close encounter pairs
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
		B = &Encpairs_d[2 * NT];
		B2 = &Encpairs_d[3 * NT];
		for(int i = 0; i < NT; i += Bl){
			if(idy + i < NT){
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
			//Encpairs_d[i].y contains the number of direct encounter pairs of body i
			//Encpairs_d[i * NencMax + j].x contains the indeces j of the direct encounter pairs
//printf("Encpairs %d %d %d %d\n", ii, jj, Ni, Nj);
		}
	}
	__syncthreads();
	if(SERIAL_GROUPING == 1){
		for(int i = 0; i < NT; i += Bl){
			if(idy + i< NT){
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
	}
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

		for(int i = 0; i < NT; i += Bl){
			if(idy + i < NT){
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
		__syncthreads();
		for(int i = 0; i < NT; i += Bl){
			if(idy + i < NT){
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
	// At this point B[idy] contains the smallest index of the group
	__syncthreads();

	for(int i = 0; i < NT; i += Bl){
		if(idy + i < NT){
			B2[idy + i].y = -1;
//printf("B %d %d\n", idy + i, B[idy + i].y);
		}
	}
	__syncthreads();
	// Check now for new groups and increase the total number of groups
	for(int i = 0; i < NT; i += Bl){
		if(idy + i < NT){
			if(B[idy + i].y == idy + i){
				B2[idy + i].y = atomicAdd(&Nenc_s[0],1);
			}		
		}
	}
	__syncthreads();
	// Transform now the smallest index of the group into a consecutive group index
	for(int i = 0; i < NT; i += Bl){
		if(idy + i < NT){
			if(B[idy + i].y < BN2) B[idy + i].y = B2[B[idy + i].y].y;
			Encpairs2_d[idy + i].y = 0;
		}
	}
	// At this point B[idy] contains a consecutive group index
	__syncthreads();

//for(int i = 0; i < NT; i += Bl){
//	if(idy + i < NT){
//		if(B[idy + i].y < BN2){
//printf("B %d %d %d\n", idy + i, B[idy + i].y, B2[idy + i].y);
//		}		
//	}
//}

	if(SERIAL_GROUPING == 0){
		for(int i = 0; i < NT; i += Bl){
			if(idy + i < NT){
				if(B[idy + i].y < BN2){
					int Ns = atomicAdd(&Encpairs2_d[B[idy + i].y].y,1);
					B2[idy + i].y = Ns; //index in the group
					Encpairs_d[NT + idy + i].y = B2[idy + i].y;
				}
			// At this point Encpairs2_d.x contains now line by line the members of the groups, Encpairs2_s.y contains the sizes of the groups
			}
		}
	}
	if(SERIAL_GROUPING == 1){
		if(idy == 0){
			for(int i = NT - 1; i >=0; --i){
				if(B[i].y < BN2){
					int Ns = atomicAdd(&Encpairs2_d[B[i].y].y,1);
					B2[i].y = Ns;	//index in the group
					Encpairs_d[NT + i].y = B2[i].y;
				}
			}
		}
	}
	__syncthreads();
	for(int i = 0; i < Nenc_s[0]; i += Bl){
		if(idy + i < Nenc_s[0]){
			if(Encpairs2_d[idy + i].y > 0){
				int start = atomicAdd(&start_s[0], Encpairs2_d[idy + i].y);
				Encpairs2_d[NT + idy + i].y = start; //starting points of te groups
//printf("start %d %d %d %d\n", idy + i, Encpairs2_d[idy + i].y, start_s[0], start);
			}
		}
	}
	__syncthreads();
	for(int i = 0; i < NT; i += Bl){
		if(idy + i < NT){
			if(B[idy + i].y < BN2){
				int n = B2[idy + i].y;
				int start = Encpairs2_d[NT + B[idy + i].y].y;
				Encpairs2_d[start + n].x = idy + i;
//printf("members %d %d %d\n", start, n, idy + i);
			}
		// At this point Encpairs2_d.x contains now members of the groups, Encpairs2_d.y contains the sizes of the groups/
		}
	}
	__syncthreads();

	for(int i = 0; i < NT; i += Bl){
		if(idy + i < NT){
			int nn = Encpairs2_d[idy + i].y;
//if(nn > 0) printf("n %d %d\n", idy + i, nn);
			volatile int ne2 = 2;
			if(nn > 0){
				for(volatile int ii = 0; ii < def_GMax - 1; ++ii){
					if(nn <= ne2){
						int Ns = atomicAdd(&Nenc_s[ii + 1],1);
//printf("G %d %d\n", ii + 1, Nenc_s[ii + 1]);
						Encpairs2_d[ (ii+2) * NT + Ns].y = idy + i;
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


template <int nb, int Bl>
__global__ void groupM1_kernel(int *Nencpairs2_d, int2 *Encpairs_d, int2 *Encpairs2_d, int *NBS_d, int *N_d, const int Nst, const int NencMax){

	int idy = threadIdx.x;
	int idx = blockIdx.x;	//The number of blocks corresponds to the number of simulations with close encounter groups

	int st = Encpairs_d[idx].y; //idx is maximally = Nst

	__shared__ int2 encpairs_s[Bl];
	__shared__ int A_s[Bl];
	__shared__ int AOld_s[Bl];
	__shared__ int B_s[nb];
	__shared__ int B2_s[nb];
	__shared__ volatile int T_s;
	__shared__ int Nenc_s;

	int NBS = NBS_d[st];
	int N = N_d[st];

	int Ne = Nencpairs2_d[st + 1];
	int BN2 = nb * nb - 1;

//if(idy == 0) printf("st %d %d %d | %d %d\n", idy, idx, st, NBS, Ne);

	__syncthreads();
	if(idy == 0){
		T_s = 1;
		Nenc_s = 0;
		
	}
	__syncthreads();

	if(idy < Ne){ 
		encpairs_s[idy].x = Encpairs2_d[idy + NBS * NencMax].x - NBS;
		encpairs_s[idy].y = Encpairs2_d[idy + NBS * NencMax].y - NBS;
		A_s[idy] = encpairs_s[idy].x;
//printf("encpairs %d %d %d %d %d\n", st, Ne, idy, encpairs_s[idy].x, encpairs_s[idy].y);
	}
	//encpairs_s[idy] contains the two close encounter pairs//
	else{
		encpairs_s[idy].x = -1;
		encpairs_s[idy].y = -1;
		A_s[idy] = -1;
	}
	if(idy < nb){
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
//printf("A %d %d %d %d %d\n", tt, st, idy, A_s[idy], B_s[encpairs_s[idy].x]);
		}
		__syncthreads();
		if(idy < Ne){
			if (A_s[idy] < B_s[encpairs_s[idy].y]) atomicMin(&B_s[encpairs_s[idy].y], A_s[idy]);
		}
		__syncthreads();

		if(idy < nb){
			if(B_s[idy] < BN2) B2_s[idy] = B_s[B_s[idy]];
		}
		__syncthreads();
		if(A_s[idy] > -1) A_s[idy] = min(B2_s[encpairs_s[idy].x], B2_s[encpairs_s[idy].y]);
		__syncthreads();
		if(AOld_s[idy] != A_s[idy]) T_s = 1;
		__syncthreads();
		if(idy < nb){
			B_s[idy] = B2_s[idy];
		}
		AOld_s[idy] = A_s[idy];
		__syncthreads();
		if(T_s == 0) break;
		__syncthreads();

	}
	//At this point B_s[idy] contains the smallest index of the group//
	__syncthreads();
	if(idy < nb){
//printf("B %d %d %d\n", st, idy, B_s[idy]);
		B2_s[idy] = -1;
	}
	__syncthreads();
	//Check now for new groups and increase the total number of groups//
	if(idy < nb){
		if(B_s[idy] == idy){
			B2_s[idy] = atomicAdd(&Nenc_s,1);
		}		
	}
	__syncthreads();
	//Transform now the smallest index of the group into a consecutive group index//
	if(idy < nb){
		if(B_s[idy] < BN2) B_s[idy] = B2_s[B_s[idy]];
		encpairs_s[idy].y = 0;
	}
	//At this point B_s[idy] contains a consecutive group index//
	__syncthreads();
	if(idy < nb){
		if(B_s[idy] < BN2){
			int ne = atomicAdd(&encpairs_s[B_s[idy]].y,1);
			Encpairs_d[(B_s[idy] + NBS) * NencMax + ne].x = idy + NBS;
		}

		//At this point Encpairs_d.x contains now line by line the members of the groups, encpairs_s.y contains the sizes of the groups//
	}
	__syncthreads();

	if(idy < N){
		//Nst is the maximum offset, where the array was used before, it is the number simulations containing close encounters
		//Each simulation can have maximally N/2 groups, therefore NBS is enough 
		Encpairs_d[idy + NBS + Nst].y = encpairs_s[idy].y;
//printf("S %d %d %d %d\n", idx, st, idy, encpairs_s[idy].y);
	}

}

__global__ void groupM2_kernel(int2 *Encpairs_d, int2 *Encpairs2_d, int *Nenc_d, int *NBS_d, int *N_d, const int Nst, const int NencMax){

	int idy = threadIdx.x;
	int idx = blockIdx.x;

	int st = Encpairs_d[idx].y;	//simulation index

	int NBS = NBS_d[st];
	int N = N_d[st];

	if(idy < N){

		int nn = Encpairs_d[idy + NBS + Nst].y;	//Sizes of the groups
//if(nn > 0) printf("n %d %d %d %d\n", st, idy, nn, NBS);

		volatile int ne2 = 2;
		if(nn > 0){
//printf("nn %d %d %d %d %d %d\n", st, idy, nn, Encpairs_d[(idy + NBS)* 16].x, Encpairs_d[((idy  + NBS)* 16)+ 1].x, Encpairs_d[((idy + NBS) * 16) + 2].x);
			for(volatile int ii = 0; ii < def_GMax - 1; ++ii){
				if(nn <= ne2){
					Encpairs2_d[ (ii+1) + NencMax * atomicAdd(&Nenc_d[ii + 1],1)].y = idy + NBS;
					break;
				} 
				else{
					ne2 *= 2;
				}
			}
		}
	}
}

// less that 512 bodies
template <int nb, int Bl>
__global__ void groupM3_kernel(int *Nenc_d, int *Nencpairs2_d, int2 *Encpairs_d, int2 *Encpairs2_d, int2 *groupIndex_d, int *NBS_d, int *N_d, const int NT, const int NencMax, const int SERIAL_GROUPING){
	
	int idy = threadIdx.x;
	int st = blockIdx.x;	//The number of blocks corresponds to Nst
	
	__shared__ volatile int T_s;
	
	int NBS = NBS_d[st];
	int N = N_d[st];
	
	int Ne = Nencpairs2_d[st + 1];

	int E;
	if(Ne < Bl){
		E = 1;
	}
	else{
		E = 2;
	}

	int BN2 = nb * nb - 1;

	int2 *A;
	int2 *encpairs;


	if(E == 1){
		__shared__ int2 A_s[Bl];
		__shared__ int2 encpairs_s[Bl];
		if(idy < Ne){
			encpairs_s[idy] = Encpairs2_d[NBS * NencMax + idy];
			A_s[idy] = encpairs_s[idy];

		}
		else{
			encpairs_s[idy].x = -1;
			encpairs_s[idy].y = -1;
			A_s[idy] = encpairs_s[idy];
		}

		A = A_s;
		encpairs = encpairs_s;
		
	}
	if(E == 2){
		encpairs = &Encpairs2_d[NBS * NencMax];
		A = &Encpairs2_d[NBS * NencMax + Ne]; 
		for(int i = 0; i < Ne; i += Bl){
			if(idy + i < Ne){ 
				A[idy + i] = encpairs[idy + i];
//printf("encpairs %d %d %d %d %d\n", st, Ne, idy + i, encpairs_s[idy + i].x, encpairs_s[idy + i].y);
			}
		}
	}
	


	__shared__ int2 B_s[nb];
	__shared__ int2 B2_s[nb];
	
	__syncthreads();
	if(idy == 0){
		T_s = 1;
	}
	__syncthreads();


	if(idy < nb){
		B_s[idy].x = 0;
		B2_s[idy].x = 0;
		B_s[idy].y = BN2;
		B2_s[idy].y = BN2;
	}
	for(int i = 0; i < N; i += Bl){
		if(idy + i < N){
			Encpairs_d[NBS + idy + i].y = 0;
		}
	}
	__syncthreads();

	for(int i = 0; i < Ne; i += Bl){
		if(idy + i < Ne){
			//create list of direct close encounter pairs
			volatile int ii = encpairs[idy + i].x;
			volatile int jj = encpairs[idy + i].y;
			volatile int Ni = 0;
			volatile int Nj = 0;
			if(jj < N){
				Ni = atomicAdd(&Encpairs_d[ii + NBS].y, 1);
				Encpairs_d[(ii + NBS) * NencMax + Ni].x = jj + NBS;
			}
			if(ii < N){
				Nj = atomicAdd(&Encpairs_d[jj + NBS].y, 1);
				Encpairs_d[(jj + NBS) * NencMax + Nj].x = ii + NBS;
			}
			//Encpairs_d[i].y contains the number of direct encounter pairs of body i
			//Encpairs_d[i * NencMax + j].x contains the indeces j of the direct encounter pairs
//printf("Encpairs %d %d %d %d %d NBS %d\n", st, ii, jj, Ni, Nj, NBS);
		}
	}
	__syncthreads();
	if(SERIAL_GROUPING == 1){
		for(int i = 0; i < N; i += Bl){
			if(idy + i< N){
				int Ni = Encpairs_d[NBS + idy + i].y;
				int stop = 0;
				while(stop == 0){
					stop = 1;
					for(int j = 0; j < Ni - 1; ++j){
						int jj = Encpairs_d[(NBS + idy + i) * NencMax + j].x;
						int jjnext = Encpairs_d[(NBS + idy + i) * NencMax + j + 1].x;

						if(jjnext < jj){
							//swap
							Encpairs_d[(NBS + idy + i) * NencMax + j].x = jjnext;
							Encpairs_d[(NBS + idy + i) * NencMax + j + 1].x = jj;
							stop = 0;
						}
					}
				}
				stop = 0;
			}
		}
	}
	__syncthreads();

	for(int tt = 0; tt < 100; ++ tt){
		T_s = 0;

		for(int i = 0; i < Ne; i += Bl){
			if(idy + i < Ne){
				int Am = min(A[idy + i].x, A[idy + i].y);
				atomicMin(&B_s[A[idy + i].y].y, Am);
				atomicMin(&B_s[A[idy + i].x].y, Am);
			}
		}
		__syncthreads();
		
		for(int i = 0; i < N; i += Bl){
			if(idy + i < N){
				if(B_s[idy + i].y < BN2) B2_s[idy + i].y = B_s[B_s[idy + i].y].y;
			}
		}
		__syncthreads();

		for(int i = 0; i < Ne; i += Bl){
			if(idy + i < Ne){
				A[idy + i].x = B2_s[encpairs[idy + i].x].y;
				A[idy + i].y = B2_s[encpairs[idy + i].y].y;
				if(A[idy + i].x != A[idy + i].y) T_s = 1;
			}
		}

		__syncthreads();
		for(int i = 0; i < N; i += Bl){
			if(idy + i < N){
				B_s[idy + i].y = B2_s[idy + i].y;
			}
		}
		__syncthreads();
		if(T_s == 0){
			break;
		}
		__syncthreads();
		
	}
	//At this point B_s[idy] contains the smallest index of the group//
	__syncthreads();

	for(int i = 0; i < N; i += Bl){
		if(idy + i < N){
//printf("B %d %d %d\n", st, idy + i, B_s[idy + i].y);
			B2_s[idy + i].y = -1;
		}
	}
	__syncthreads();
	//Check now for new groups and increase the total number of groups//
	for(int i = 0; i < N; i += Bl){
		if(idy + i < N){
			if(B_s[idy + i].y == idy + i){
				B2_s[idy + i].y = atomicAdd(&Nenc_d[0],1);
			}		
		}
	}
	__syncthreads();
	//Transform now the smallest index of the group into a consecutive group index//
	for(int i = 0; i < N; i += Bl){
		if(idy + i < N){
			if(B_s[idy + i].y < BN2){
				B_s[idy + i].y = B2_s[B_s[idy + i].y].y;
				groupIndex_d[NBS + idy + i].x = B_s[idy + i].y;
//printf("Group index %d %d %d %d\n", st, idy + i, NBS + idy + i, B_s[idy + i].y);
			}
			else{
				groupIndex_d[NBS + idy + i].x = BN2;
			}
		}
	}
	//At this point B_s[idy] contains a consecutive group index//
	__syncthreads();

	if(SERIAL_GROUPING == 0){
		for(int i = 0; i < N; i += Bl){
			if(idy + i < N){
				if(B_s[idy + i].y < BN2){
					int Ns = atomicAdd(&groupIndex_d[B_s[idy + i].y + 1].y,1);
					Encpairs_d[NT + NBS + idy + i].y = Ns;	//index in the group
//printf("index in the group %d %d %d %d\n", st, idy + i, NBS + idy + i, Ns);
				}
				//At this point Encpairs_d.x contains now line by line the members of the groups, groupIndex contains the sizes of the groups//
			}
		}
	}
	if(SERIAL_GROUPING == 1){
		if(idy == 0){
			for(int i = N - 1; i >=0; --i){
				if(B_s[i].y < BN2){
					int Ns = atomicAdd(&groupIndex_d[B_s[i].y + 1].y,1);
					Encpairs_d[NT + NBS + i].y = Ns;
				}
			}
		}
	}

	__syncthreads();
	for(int i = 0; i < N; i += Bl){
		if(idy + i < N){
			int g = B_s[idy + i].y;
			if(g < BN2 && Encpairs_d[NT + NBS + idy + i].y == 0){
				int start = atomicAdd(&groupIndex_d[0].y, groupIndex_d[g + 1].y);
				Encpairs2_d[NT + g].y = start; //starting points of the groups
//printf("start %d %d %d %d %d\n", st, idy + i, g, groupIndex_d[g + 1].y, start);
			}
		}
	}
}

__global__ void groupM3_2_kernel(int *Nenc_d, int *Nencpairs2_d, int2 *Encpairs_d, int2 *Encpairs2_d, int2 *groupIndex_d, int *NBS_d, const int NT, const int NBmax){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	int BN2 = NBmax * NBmax - 1;


	if(id < NT){
		int g = groupIndex_d[id].x; 			//group index
//printf("members %d %d\n", id, g);
		if(g < BN2){
			int n = Encpairs_d[NT + id].y;		//index in the group
			int start = Encpairs2_d[NT + g].y;
			Encpairs2_d[start + n].x = id;
//printf("members %d %d %d\n", id, g, n);
//printf("members %d %d %d %d\n", id, g, n, start);
		}
	}


	__syncthreads();

	if(id < NT - 1){
		int nn = groupIndex_d[id + 1].y;
//if(nn > 0) printf("n %d %d\n", id, nn);
		volatile int ne2 = 2;
		if(nn > 0){
			for(volatile int ii = 0; ii < def_GMax - 1; ++ii){
				if(nn <= ne2){ 
					int Ns = atomicAdd(&Nenc_d[ii + 1],1);
//printf("G %d %d\n", ii + 1, Nenc_d[ii + 1]);  
					Encpairs2_d[ (ii+2) * NT + Ns].y = id;
					break;
				} 
				else{   
					ne2 *= 2;
				}
			}
		}
	}

}



// **********************************************************
// This kernel writes a list of close encounter pairs needed for the symplectic sub step
// Date: March 2020
// Author: Simon Grimm
// **********************************************************
__global__ void setEnc3_kernel(int N, int *Nencpairs3_d, int *Encpairs3_d, int2 *scan_d, const int NencMax){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id == 0){
		Nencpairs3_d[0] = 0;
	}

	if(id < N){
		Encpairs3_d[id * NencMax] = 0;		//Encounter pairs per body
		Encpairs3_d[id * NencMax + 1] = -1;	//list of indices	 
		Encpairs3_d[id * NencMax + 2] = 0;	//number of pairs with real gravitational influence
		Encpairs3_d[id * NencMax + 3] = 0;	//helper array for stream compaction, replace with scan_d

		scan_d[id].x = 0;
		scan_d[id].y = 0;
	}
}


// **********************************************************
// This kernel writes lists of encounter pairs for each bodies.
// It prepares the helper array for stream compaction, for a list of all involved particles
// Date: March 2020
// Author: Simon Grimm
// **********************************************************
__global__ void groupS2_kernel(int *Nencpairs2_d, int2 *Encpairs2_d, int *Nencpairs3_d, int *Encpairs3_d, int2 *scan_d, const int NencMax, const int UseTestParticles, const int N, const int SLevel){

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	int Ne = Nencpairs2_d[0];
	if(id < Ne){
		int ii = Encpairs2_d[id].x;
		int jj = Encpairs2_d[id].y;

		//count encounter pairs per body
#if def_CPU == 0
		//int NI = atomicAdd(&Encpairs3_d[ii * NencMax], 1);
		//int NJ = atomicAdd(&Encpairs3_d[jj * NencMax], 1);
		atomicAdd(&Encpairs3_d[ii * NencMax], 1);
		atomicAdd(&Encpairs3_d[jj * NencMax], 1);
#else
		//int NI = Encpairs3_d[ii * NencMax]++;
		//int NJ = Encpairs3_d[jj * NencMax]++;
		#pragma omp atomic
		Encpairs3_d[ii * NencMax]++;
		#pragma omp atomic
		Encpairs3_d[jj * NencMax]++;

#endif
//printf("group S %d %d %d %d %d\n", id, ii, jj, NI, NJ);

		//fill helper array for stream compaction
		scan_d[ii].x = 1;
		scan_d[jj].x = 1;

		if(jj < N || (UseTestParticles == 2 && ii < N)){
#if def_CPU == 0
			int Ni = atomicAdd(&Encpairs3_d[ii * NencMax + 2], 1);
#else
			int Ni;
			#pragma omp atomic capture
			Ni = Encpairs3_d[ii * NencMax + 2]++;
#endif
			Encpairs3_d[ii * NencMax + Ni + 4] = jj;
		}

		if(ii < N || (UseTestParticles == 2 && jj < N)){
#if def_CPU == 0
			int Nj = atomicAdd(&Encpairs3_d[jj * NencMax + 2], 1);
#else
			int Nj;
			#pragma omp atomic capture
			Nj = Encpairs3_d[jj * NencMax + 2]++;

#endif
			Encpairs3_d[jj * NencMax + Nj + 4] = ii;
		}
	}
}

__host__ void Data::groupCall(){
	if(Nsmall_h[0] == 0){
		if(NB[0] <= 512){
			switch(NB[0]){
				case 16:{
					group_kernel < 16, 512 > <<< 1, 512 >>> (Nenc_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0], N_h[0], P.SERIAL_GROUPING);
				}
				break;
				case 32:{
					group_kernel < 32, 512 > <<< 1, 512 >>> (Nenc_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0], N_h[0], P.SERIAL_GROUPING);
				}
				break;
				case 64:{
					group_kernel < 64, 512 > <<< 1, 512 >>> (Nenc_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0], N_h[0], P.SERIAL_GROUPING);
				}
				break;
				case 128:{
					group_kernel < 128, 512> <<< 1, 512 >>> (Nenc_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0], N_h[0], P.SERIAL_GROUPING);
				}
				break;
				case 256:{
					group_kernel < 256, 512 > <<< 1, 512 >>> (Nenc_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0], N_h[0], P.SERIAL_GROUPING);
				}
				break;
				case 512:{
					group_kernel < 512, 512 > <<< 1, 512 >>> (Nenc_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0], N_h[0], P.SERIAL_GROUPING);
				}
				break;
			}
		}
		else{
			group_kernel < 1, 512 > <<< 1, 512 >>> (Nenc_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0], N_h[0], P.SERIAL_GROUPING);
		}
	}
	else{
		//assume here  E = 3 or E = 4
		if(P.UseTestParticles < 2){
			group_kernel < 512, 512 > <<< 1, 512 >>> (Nenc_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0] + Nsmall_h[0], N_h[0], P.SERIAL_GROUPING);
		}
		else{
			group_kernel < 512, 512 > <<< 1, 512 >>> (Nenc_d, Nencpairs2_d, Encpairs2_d, Encpairs_d, P.NencMax, N_h[0] + Nsmall_h[0], N_h[0] + Nsmall_h[0], P.SERIAL_GROUPING);
		}
	}
}

#endif

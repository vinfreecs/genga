#ifndef ENCOUNTER_H
#define ENCOUNTER_H
#include "Orbit2.h"

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
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
// ****************************************
template<int E>
__device__ int encounter(double4 x4i, double4 v4i, double4 x4oldi, double4 v4oldi, double4 x4j, double4 v4j, double4 x4oldj, double4 v4oldj, double rcriti, double rcritj, double rcritvi, double rcritvj, double dt, int i, int j, double *test_d, int2 *encpairs, int &Nenc, int N){

//if(E == 0 || E >= 2)printf("E %d %d\n", i,j);
//if(E == 1) printf("E1 %d %d %g %g\n", i, j, x4i.w, x4j.w);

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
// ************************************************
// This function is here only for testing
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
// ***********************************************
template<int E>
__device__ int encounterG3(double4 x4i, double4 v4i, double4 x4oldi, double4 v4oldi, double4 x4j, double4 v4j, double4 x4oldj, double4 v4oldj, double rcriti, double rcritj, double rcritvi, double rcritvj, double dt, int i, int j, double *test_d, int2 *encpairs, int &Nenc, int N, double &Ki, double &Kj, double &Kiold, double &Kjold, double &Bdd1oldi, double &Bdd1oldj, double time, int &groupIndexOldi, int &groupIndexOldj, double3 aiold, double3 ajold){

//if(E == 0 || E >= 2)printf("E %d %d\n", i,j);

	int Enc = 0;
	if(i < j && (x4i.w > 0.0 || x4j.w > 0.0) && x4i.w >= 0.0 && x4j.w >= 0.0){
		int Ni;

		//undo last kick
		double Bdd0, Bd0, B0;
		double4 x4it = x4oldi;
		double4 x4jt = x4oldj;
		double4 v4it = v4oldi;
		double4 v4jt = v4oldj;
double E0, E1;
		{
		
			double3 r;
			r.x = x4jt.x - x4it.x;
			r.y = x4jt.y - x4it.y;
			r.z = x4jt.z - x4it.z;
			double d = r.x*r.x + r.y*r.y + r.z*r.z;
			double rij = sqrt(d);

			double irij3 = 1.0 / (rij * d);

			//Kick
			v4it.x -= aiold.x;
			v4it.y -= aiold.y;
			v4it.z -= aiold.z;
			v4jt.x -= ajold.x;
			v4jt.y -= ajold.y;
			v4jt.z -= ajold.z;

			B0 = x4i.w * x4j.w / rij;
			double3 rd;
			rd.x = v4jt.x - v4it.x;
			rd.y = v4jt.y - v4it.y;
			rd.z = v4jt.z - v4it.z;
			double dd = (r.x*rd.x + r.y*rd.y+ r.z*rd.z) * 2.0;
			Bd0 = -B0 / d * 0.5 * dd;

			double ri = sqrt(x4oldi.x * x4oldi.x + x4oldi.y * x4oldi.y + x4oldi.z * x4oldi.z);
			double rj = sqrt(x4oldj.x * x4oldj.x + x4oldj.y * x4oldj.y + x4oldj.z * x4oldj.z);

			double iri3 = 1.0 / (ri * ri * ri);
			double irj3 = 1.0 / (rj * rj * rj);
			double3 ai;
			double3 aj;

			ai.x = x4j.w * irij3 * r.x - iri3 * x4i.x;
			ai.y = x4j.w * irij3 * r.y - iri3 * x4i.y;
			ai.z = x4j.w * irij3 * r.z - iri3 * x4i.z;
			aj.x = -x4i.w * irij3 * r.x - irj3 * x4j.x;
			aj.y = -x4i.w * irij3 * r.y - irj3 * x4j.y;
			aj.z = -x4i.w * irij3 * r.z - irj3 * x4j.z;

			double ud = rd.x * rd.x + r.x * (aj.x - ai.x) + rd.y * rd.y + r.y * (aj.y - ai.y) + rd.z * rd.z + r.z * (aj.z - ai.z);
			Bdd0 = -3.0 * Bd0 / d * 0.5 * dd - B0 / d * ud;

			double Ti = 0.5 * x4i.w * (v4it.x * v4it.x + v4it.y * v4it.y + v4it.z * v4it.z);
			double Tj = 0.5 * x4j.w * (v4jt.x * v4jt.x + v4jt.y * v4jt.y + v4jt.z * v4jt.z);

			double Msun = 1.0;
			double Vi = -Msun * x4it.w / ri;
			double Vj = -Msun * x4jt.w / rj;
			double Vij = -x4it.w * x4jt.w / rij;
			E0 = (Ti + Tj) + (Vi + Vj) + Vij;
//printf("Ei %g %.20g %.20g %.20g\n", time, Vi, Ti, Vij);
//printf("Ej %g %.20g %.20g %.20g\n", time, Vj, Tj, Vij);
//printf("%g %.20g %.20g %.20g\n",time,  Kg * (Ti + Tj), Kg *(Vi + Vj), Kg * Vij);

		}
//Bdd0 = Bdd1oldi;
		double Bdd1;
		double Bd1, B1;
		//do next step with RK 4
/*		{
			double3 r;
			r.x = x4jt.x - x4it.x;
			r.y = x4jt.y - x4it.y;
			r.z = x4jt.z - x4it.z;
			double d = r.x*r.x + r.y*r.y+ r.z*r.z;

			double rij = sqrt(d);

			double3 ai;
			double3 aj;

			double irij3 = 1.0 / (rij * d);
			double ri = sqrt(x4it.x * x4it.x + x4it.y * x4it.y + x4it.z * x4it.z);
			double rj = sqrt(x4jt.x * x4jt.x + x4jt.y * x4jt.y + x4jt.z * x4jt.z);

			double iri3 = 1.0 / (ri * ri * ri);
			double irj3 = 1.0 / (rj * rj * rj);

			ai.x = x4j.w * irij3 * r.x - iri3 * x4i.x;
			ai.y = x4j.w * irij3 * r.y - iri3 * x4i.y;
			ai.z = x4j.w * irij3 * r.z - iri3 * x4i.z;
			aj.x = -x4i.w * irij3 * r.x - irj3 * x4j.x;
			aj.y = -x4i.w * irij3 * r.y - irj3 * x4j.y;
			aj.z = -x4i.w * irij3 * r.z - irj3 * x4j.z;
	
			double3 l1i;
			double3 l1j;
			l1i.x = dt * v4it.x;
			l1i.y = dt * v4it.y;
			l1i.z = dt * v4it.z;
			l1j.x = dt * v4jt.x;
			l1j.y = dt * v4jt.y;
			l1j.z = dt * v4jt.z;

			double3 k1i;
			double3 k1j;
			k1i.x = dt * ai.x;
			k1i.y = dt * ai.y;
			k1i.z = dt * ai.z;
			k1j.x = dt * aj.x;
			k1j.y = dt * aj.y;
			k1j.z = dt * aj.z;

			// *****************  //

			double4 x4is;
			double4 x4js;

			x4is.x = x4it.x + 0.5 * k1i.x;
			x4is.y = x4it.y + 0.5 * k1i.y;
			x4is.z = x4it.z + 0.5 * k1i.z;
			x4js.x = x4jt.x + 0.5 * k1j.x;
			x4js.y = x4jt.y + 0.5 * k1j.y;
			x4js.z = x4jt.z + 0.5 * k1j.z;

			r.x = x4js.x - x4is.x;
			r.y = x4js.y - x4is.y;
			r.z = x4js.z - x4is.z;
			d = r.x*r.x + r.y*r.y+ r.z*r.z;

			rij = sqrt(d);

			irij3 = 1.0 / (rij * d);
			ri = sqrt(x4is.x * x4is.x + x4is.y * x4is.y + x4is.z * x4is.z);
			rj = sqrt(x4js.x * x4js.x + x4js.y * x4js.y + x4js.z * x4js.z);

			iri3 = 1.0 / (ri * ri * ri);
			irj3 = 1.0 / (rj * rj * rj);

			ai.x = x4j.w * irij3 * r.x - iri3 * x4i.x;
			ai.y = x4j.w * irij3 * r.y - iri3 * x4i.y;
			ai.z = x4j.w * irij3 * r.z - iri3 * x4i.z;
			aj.x = -x4i.w * irij3 * r.x - irj3 * x4j.x;
			aj.y = -x4i.w * irij3 * r.y - irj3 * x4j.y;
			aj.z = -x4i.w * irij3 * r.z - irj3 * x4j.z;

			double3 l2i;
			double3 l2j;
			l2i.x = dt * (v4it.x + 0.5 * l1i.x);
			l2i.y = dt * (v4it.y + 0.5 * l1i.y);
			l2i.z = dt * (v4it.z + 0.5 * l1i.z);
			l2j.x = dt * (v4jt.x + 0.5 * l1j.x);
			l2j.y = dt * (v4jt.y + 0.5 * l1j.y);
			l2j.z = dt * (v4jt.z + 0.5 * l1j.z);

			double3 k2i;
			double3 k2j;
			k2i.x = dt * ai.x;
			k2i.y = dt * ai.y;
			k2i.z = dt * ai.z;
			k2j.x = dt * aj.x;
			k2j.y = dt * aj.y;
			k2j.z = dt * aj.z;

			// *****************  //

			x4is.x = x4it.x + 0.5 * k2i.x;
			x4is.y = x4it.y + 0.5 * k2i.y;
			x4is.z = x4it.z + 0.5 * k2i.z;
			x4js.x = x4jt.x + 0.5 * k2j.x;
			x4js.y = x4jt.y + 0.5 * k2j.y;
			x4js.z = x4jt.z + 0.5 * k2j.z;

			r.x = x4js.x - x4is.x;
			r.y = x4js.y - x4is.y;
			r.z = x4js.z - x4is.z;
			d = r.x*r.x + r.y*r.y+ r.z*r.z;

			rij = sqrt(d);

			irij3 = 1.0 / (rij * d);
			ri = sqrt(x4is.x * x4is.x + x4is.y * x4is.y + x4is.z * x4is.z);
			rj = sqrt(x4js.x * x4js.x + x4js.y * x4js.y + x4js.z * x4js.z);

			iri3 = 1.0 / (ri * ri * ri);
			irj3 = 1.0 / (rj * rj * rj);

			ai.x = x4j.w * irij3 * r.x - iri3 * x4i.x;
			ai.y = x4j.w * irij3 * r.y - iri3 * x4i.y;
			ai.z = x4j.w * irij3 * r.z - iri3 * x4i.z;
			aj.x = -x4i.w * irij3 * r.x - irj3 * x4j.x;
			aj.y = -x4i.w * irij3 * r.y - irj3 * x4j.y;
			aj.z = -x4i.w * irij3 * r.z - irj3 * x4j.z;

			double3 l3i;
			double3 l3j;
			l3i.x = dt * (v4it.x + 0.5 * l2i.x);
			l3i.y = dt * (v4it.y + 0.5 * l2i.y);
			l3i.z = dt * (v4it.z + 0.5 * l2i.z);
			l3j.x = dt * (v4jt.x + 0.5 * l2j.x);
			l3j.y = dt * (v4jt.y + 0.5 * l2j.y);
			l3j.z = dt * (v4jt.z + 0.5 * l2j.z);

			double3 k3i;
			double3 k3j;
			k3i.x = dt * ai.x;
			k3i.y = dt * ai.y;
			k3i.z = dt * ai.z;
			k3j.x = dt * aj.x;
			k3j.y = dt * aj.y;
			k3j.z = dt * aj.z;

			// *****************  //

			x4is.x = x4it.x + k3i.x;
			x4is.y = x4it.y + k3i.y;
			x4is.z = x4it.z + k3i.z;
			x4js.x = x4jt.x + k3j.x;
			x4js.y = x4jt.y + k3j.y;
			x4js.z = x4jt.z + k3j.z;

			r.x = x4js.x - x4is.x;
			r.y = x4js.y - x4is.y;
			r.z = x4js.z - x4is.z;
			d = r.x*r.x + r.y*r.y+ r.z*r.z;

			rij = sqrt(d);

			irij3 = 1.0 / (rij * d);
			ri = sqrt(x4is.x * x4is.x + x4is.y * x4is.y + x4is.z * x4is.z);
			rj = sqrt(x4js.x * x4js.x + x4js.y * x4js.y + x4js.z * x4js.z);

			iri3 = 1.0 / (ri * ri * ri);
			irj3 = 1.0 / (rj * rj * rj);

			ai.x = x4j.w * irij3 * r.x - iri3 * x4i.x;
			ai.y = x4j.w * irij3 * r.y - iri3 * x4i.y;
			ai.z = x4j.w * irij3 * r.z - iri3 * x4i.z;
			aj.x = -x4i.w * irij3 * r.x - irj3 * x4j.x;
			aj.y = -x4i.w * irij3 * r.y - irj3 * x4j.y;
			aj.z = -x4i.w * irij3 * r.z - irj3 * x4j.z;

			double3 l4i;
			double3 l4j;
			l4i.x = dt * (v4it.x + l3i.x);
			l4i.y = dt * (v4it.y + l3i.y);
			l4i.z = dt * (v4it.z + l3i.z);
			l4j.x = dt * (v4jt.x + l3j.x);
			l4j.y = dt * (v4jt.y + l3j.y);
			l4j.z = dt * (v4jt.z + l3j.z);

			double3 k4i;
			double3 k4j;
			k4i.x = dt * ai.x;
			k4i.y = dt * ai.y;
			k4i.z = dt * ai.z;
			k4j.x = dt * aj.x;
			k4j.y = dt * aj.y;
			k4j.z = dt * aj.z;


			x4it.x += 1.0/6.0 * l1i.x + 1.0 / 3.0 * l2i.x + 1.0 / 3.0 * l3i.x + 1.0 / 6.0 * l4i.x;
			x4it.y += 1.0/6.0 * l1i.y + 1.0 / 3.0 * l2i.y + 1.0 / 3.0 * l3i.y + 1.0 / 6.0 * l4i.y;
			x4it.z += 1.0/6.0 * l1i.z + 1.0 / 3.0 * l2i.z + 1.0 / 3.0 * l3i.z + 1.0 / 6.0 * l4i.z;
			x4jt.x += 1.0/6.0 * l1j.x + 1.0 / 3.0 * l2j.x + 1.0 / 3.0 * l3j.x + 1.0 / 6.0 * l4j.x;
			x4jt.y += 1.0/6.0 * l1j.y + 1.0 / 3.0 * l2j.y + 1.0 / 3.0 * l3j.y + 1.0 / 6.0 * l4j.y;
			x4jt.z += 1.0/6.0 * l1j.z + 1.0 / 3.0 * l2j.z + 1.0 / 3.0 * l3j.z + 1.0 / 6.0 * l4j.z;

			v4it.x += 1.0/6.0 * k1i.x + 1.0 / 3.0 * k2i.x + 1.0 / 3.0 * k3i.x + 1.0 / 6.0 * k4i.x;
			v4it.y += 1.0/6.0 * k1i.y + 1.0 / 3.0 * k2i.y + 1.0 / 3.0 * k3i.y + 1.0 / 6.0 * k4i.y;
			v4it.z += 1.0/6.0 * k1i.z + 1.0 / 3.0 * k2i.z + 1.0 / 3.0 * k3i.z + 1.0 / 6.0 * k4i.z;
			v4jt.x += 1.0/6.0 * k1j.x + 1.0 / 3.0 * k2j.x + 1.0 / 3.0 * k3j.x + 1.0 / 6.0 * k4j.x;
			v4jt.y += 1.0/6.0 * k1j.y + 1.0 / 3.0 * k2j.y + 1.0 / 3.0 * k3j.y + 1.0 / 6.0 * k4j.y;
			v4jt.z += 1.0/6.0 * k1j.z + 1.0 / 3.0 * k2j.z + 1.0 / 3.0 * k3j.z + 1.0 / 6.0 * k4j.z;

			//New B
			r.x = x4jt.x - x4it.x;
			r.y = x4jt.y - x4it.y;
			r.z = x4jt.z - x4it.z;
			d = r.x*r.x + r.y*r.y+ r.z*r.z;

			rij = sqrt(d);
			B1 = x4i.w * x4j.w / rij;
			double3 rd;
			rd.x = v4jt.x - v4it.x;
			rd.y = v4jt.y - v4it.y;
			rd.z = v4jt.z - v4it.z;
			double dd = (r.x*rd.x + r.y*rd.y+ r.z*rd.z) * 2.0;
			Bd1 = -B1 / d * 0.5 * dd;

			ri = sqrt(x4it.x * x4it.x + x4it.y * x4it.y + x4it.z * x4it.z);
			rj = sqrt(x4jt.x * x4jt.x + x4jt.y * x4jt.y + x4jt.z * x4jt.z);

			iri3 = 1.0 / (ri * ri * ri);
			irj3 = 1.0 / (rj * rj * rj);

			ai.x = x4j.w * irij3 * r.x - iri3 * x4i.x;
			ai.y = x4j.w * irij3 * r.y - iri3 * x4i.y;
			ai.z = x4j.w * irij3 * r.z - iri3 * x4i.z;
			aj.x = -x4i.w * irij3 * r.x - irj3 * x4j.x;
			aj.y = -x4i.w * irij3 * r.y - irj3 * x4j.y;
			aj.z = -x4i.w * irij3 * r.z - irj3 * x4j.z;

			double ud = rd.x * rd.x + r.x * (aj.x - ai.x) + rd.y * rd.y + r.y * (aj.y - ai.y) + rd.z * rd.z + r.z * (aj.z - ai.z);
			Bdd1 = -3.0 * Bd1 / d * 0.5 * dd - B1 / d * ud;

		}
*/		//do next step with Kick FG Kick
{
			double3 r;
			volatile double d, rij, irij3;

			int aecount = 0;
			int Gridaecount = 0;
			float4 aelimits = {0.0f, 0.0f, 0.0f, 0.0f};
			double test = 0.0;
			double Msun = 1.0;

			double3 ai;
			double3 aj;


			double dtt = dt / 1.0;
			for(int st = 0; st < 1; ++st){
				r.x = x4jt.x - x4it.x;
				r.y = x4jt.y - x4it.y;
				r.z = x4jt.z - x4it.z;
				d = r.x*r.x + r.y*r.y+ r.z*r.z;

				rij = sqrt(d);

				irij3 = 1.0 / (rij * d);

				ai.x = x4jt.w * irij3 * r.x;
				ai.y = x4jt.w * irij3 * r.y;
				ai.z = x4jt.w * irij3 * r.z;
				aj.x = -x4it.w * irij3 * r.x;
				aj.y = -x4it.w * irij3 * r.y;
				aj.z = -x4it.w * irij3 * r.z;

				//Kick
				v4it.x += ai.x * dtt * 0.5;
				v4it.y += ai.y * dtt * 0.5;
				v4it.z += ai.z * dtt * 0.5;
				v4jt.x += aj.x * dtt * 0.5;
				v4jt.y += aj.y * dtt * 0.5;
				v4jt.z += aj.z * dtt * 0.5;

				//FG
				fgfull(x4it, v4it, dtt, ksq * Msun, test, test, Msun, aelimits, aecount, &Gridaecount, 1, i);
				fgfull(x4jt, v4jt, dtt, ksq * Msun, test, test, Msun, aelimits, aecount, &Gridaecount, 1, j);
				//BSSinglestep(x4it, v4it, Msun, dtt, test, test);
				//BSSinglestep(x4jt, v4jt, Msun, dtt, test, test);

				//Kick
				r.x = x4jt.x - x4it.x;
				r.y = x4jt.y - x4it.y;
				r.z = x4jt.z - x4it.z;
				d = r.x*r.x + r.y*r.y+ r.z*r.z;

				rij = sqrt(d);

				irij3 = 1.0 / (rij * d);

				double si = x4jt.w * irij3;
				double sj = x4it.w * irij3;

				ai.x = r.x * si;
				ai.y = r.y * si;
				ai.z = r.z * si;
				aj.x = -r.x * sj;
				aj.y = -r.y * sj;
				aj.z = -r.z * sj;

				v4it.x += ai.x * dtt * 0.5;
				v4it.y += ai.y * dtt * 0.5;
				v4it.z += ai.z * dtt * 0.5;
				v4jt.x += aj.x * dtt * 0.5;
				v4jt.y += aj.y * dtt * 0.5;
				v4jt.z += aj.z * dtt * 0.5;
			}
			//New B
			B1 = x4i.w * x4j.w / rij;
			double3 rd;
			rd.x = v4jt.x - v4it.x;
			rd.y = v4jt.y - v4it.y;
			rd.z = v4jt.z - v4it.z;
			double dd = (r.x*rd.x + r.y*rd.y+ r.z*rd.z) * 2.0;
			Bd1 = -B1 / d * 0.5 * dd;

			double ri = sqrt(x4it.x * x4it.x + x4it.y * x4it.y + x4it.z * x4it.z);
			double rj = sqrt(x4jt.x * x4jt.x + x4jt.y * x4jt.y + x4jt.z * x4jt.z);

			double iri3 = 1.0 / (ri * ri * ri);
			double irj3 = 1.0 / (rj * rj * rj);

			ai.x = x4j.w * irij3 * r.x - iri3 * x4i.x;
			ai.y = x4j.w * irij3 * r.y - iri3 * x4i.y;
			ai.z = x4j.w * irij3 * r.z - iri3 * x4i.z;
			aj.x = -x4i.w * irij3 * r.x - irj3 * x4j.x;
			aj.y = -x4i.w * irij3 * r.y - irj3 * x4j.y;
			aj.z = -x4i.w * irij3 * r.z - irj3 * x4j.z;

			double ud = rd.x * rd.x + r.x * (aj.x - ai.x) + rd.y * rd.y + r.y * (aj.y - ai.y) + rd.z * rd.z + r.z * (aj.z - ai.z);
			Bdd1 = -3.0 * Bd1 / d * 0.5 * dd - B1 / d * ud;

			double Ti = 0.5 * x4i.w * (v4it.x * v4it.x + v4it.y * v4it.y + v4it.z * v4it.z);
			double Tj = 0.5 * x4j.w * (v4jt.x * v4jt.x + v4jt.y * v4jt.y + v4jt.z * v4jt.z);
			double Vi = -Msun * x4it.w / ri;
			double Vj = -Msun * x4jt.w / rj;
			double Vij = -x4it.w * x4jt.w / rij;
			E1 = (Ti + Tj) + (Vi + Vj) + Vij;
//printf("%g %.20g %.20g %.20g\n",time,  Kg * (Ti + Tj), Kg *(Vi + Vj), Kg * Vij);
printf("%g %.20g %.20g\n", time, E0, E1);
		}

		int Encflag = 0;

		if(E == 0){


			Kiold = Ki;
			Kjold = Kj;

//printf("%.10g %d %d %g %g %g %g %g %g %g %g %g %g %g %g %g\n", time, i, j, 0.0, 0.0, 0.0, 0.0, 0.0, Bdd1, Bd1, B1, (Bdd1 - Bdd0) / dt, Kiold, Bdd0, Bd0, B0);

			if(Kiold < 1.0) Encflag = 1;

			int stop = 0;
			if((Bdd0 >= G3Limit && Kiold < 1.0) || Kiold == 0.0) {
//printf("%.10g %d %d %g %g %g %g %g %g close encounter\n", time, i, j, Bdd0, Bdd1, Bd0, Bd1, B0, B1);
				//close encounter
				Ki = 0.0;
				Kj = 0.0;
			}
			//if(Bdd0 < G3Limit && Bdd1 < G3Limit && Bdd0 >= Bdd1 && Bdd0 > 0.0 && Bd0 < 0.0 && Bd1 < 0.0){
			if(Bdd0 < G3Limit && Bdd1 < G3Limit && Kiold < 1.0 && Kiold > 0.0){
//printf("%.10g %d %d %g %g %g %g %g %g no CL\n", time, i, j, Bdd0, Bdd1, Bd0, Bd1, B0, B1);
				//no close encounter
				Ki = 1.0;
				Kj = 1.0;
			}
			if(Bdd0 < G3Limit && Bdd1 >= G3Limit && Kiold == 1.0){
//printf("%.10g %d %d %g %g %g %g %g %g start\n", time, i, j, Bdd0, Bdd1, Bd0, Bd1, B0, B1);
				//start close encounter
				double root = (G3Limit - Bdd0)/(Bdd1 - Bdd0);
//root = 0.03333;
				Ki = root;
				Kj = root;
//printf("Start %g %g\n", Ki, root);
			}

/*			if(Bdd0 >= G3Limit && Bdd1 < G3Limit && Kiold < 1.0){
//printf("%.10g %d %d %g %g %g %g %g %g  stop\n", time, i, j, Bdd0, Bdd1, Bd0, Bd1, B0, B1);
				//stop close encounter
				stop = 1;
				double root = (G3Limit - Bdd0)/(Bdd1 - Bdd0);
//proot = 0.583;
				Ki = (1.0 - root);// * 0.9;
				Kj = (1.0 - root);// * 0.9;
//printf("Stop %g %g\n", Ki, root);

			}
*/
double Bdd = 0.0;
//this function searches if there is a following peak
if(stop == 1){
	double4	x4it = x4i;
	double4	x4jt = x4j;
	double4	v4it = v4i;
	double4	v4jt = v4j;
	int aecount = 0;
	int Gridaecount = 0;
	float4 aelimits = {0.0f, 0.0f, 0.0f, 0.0f};
	double test = 0.0;
	double Msun = 1.0;

	double dtt = dt;
	int phase = 1;
	Bdd = 2.0 * G3Limit;
	double rsqi = x4i.x * x4i.x + x4i.y * x4i.y + x4i.z * x4i.z; 
	double rsqj = x4j.x * x4j.x + x4j.y * x4j.y + x4j.z * x4j.z; 
	double vsqi = v4i.x * v4i.x + v4i.y * v4i.y + v4i.z * v4i.z; 
	double vsqj = v4j.x * v4j.x + v4j.y * v4j.y + v4j.z * v4j.z; 

	double iai = 2.0 / sqrt(rsqi) - vsqi / (Msun);
	double iaj = 2.0 / sqrt(rsqj) - vsqj / (Msun);
	double a = fmax(1.0 / iai, 1.0 /iaj);
	double T = 2.0 * M_PI * sqrt(a * a * a / Msun);
	for(int bt = 1; bt < T / dtt; ++bt){
		fgfull(x4it, v4it, dtt, ksq * Msun, test, test, Msun, aelimits, aecount, &Gridaecount, 1, i);
		fgfull(x4jt, v4jt, dtt, ksq * Msun, test, test, Msun, aelimits, aecount, &Gridaecount, 1, j);
		double rx = x4jt.x - x4it.x;
		double ry = x4jt.y - x4it.y;
		double rz = x4jt.z - x4it.z;
		double d = rx * rx + ry * ry + rz * rz;
		double rij = sqrt(d);
		double B = x4it.w * x4jt.w / rij;
		double rdx = v4jt.x - v4it.x;
		double rdy = v4jt.y - v4it.y;
		double rdz = v4jt.z - v4it.z;

		double dd = (rx * rdx + ry * rdy + rz * rdz) * 2.0;
		double Bd = -B / d * 0.5 * dd;

		double ri = sqrt(x4it.x * x4it.x + x4it.y * x4it.y + x4it.z * x4it.z);
		double rj = sqrt(x4jt.x * x4jt.x + x4jt.y * x4jt.y + x4jt.z * x4jt.z);

		double3 ai;
		double3 aj;

		double irij3 = 1.0 / (rij * d);
		double iri3 = 1.0 / (ri * ri * ri);
		double irj3 = 1.0 / (rj * rj * rj);

		ai.x = x4jt.w * irij3 * rx - iri3 * x4it.x;
		ai.y = x4jt.w * irij3 * ry - iri3 * x4it.y;
		ai.z = x4jt.w * irij3 * rz - iri3 * x4it.z;
		aj.x = -x4it.w * irij3 * rx - irj3 * x4jt.x;
		aj.y = -x4it.w * irij3 * ry - irj3 * x4jt.y;
		aj.z = -x4it.w * irij3 * rz - irj3 * x4jt.z;

		double ud = rdx * rdx + rx * (aj.x - ai.x) + rdy * rdy + ry * (aj.y - ai.y) + rdz * rdz + rz * (aj.z - ai.z);
		double Bddold = Bdd;
		Bdd = -3.0 * Bd / d * 0.5 * dd - B / d * ud;
		if(phase == 1 && Bddold < Bdd){
//printf("%g %d %d stop to phase 2. %g %g %g\n",time, i, j, time + bt * dtt/0.01720209895, Bddold, Bdd);
			//entering acending node
			//search for the height of the peak
			phase = 2;
		}
		if(phase == 2 && (Bdd >= G3Limit)){
//printf("%g %d %d stop. next peak is at %g %g\n",time, i, j, time + bt * dtt/0.01720209895, Bdd);
			break;
		}
	}
}

			Bdd1oldi = Bdd1;	
			//remove invalid stop points
			if(Ki < 1.0) Encflag = 1;
			if(stop == 1 && Bdd >= 1.01 * G3Limit){ //the feactor is needed due to prediction uncertainties
				Encflag = 1;
				Ki = 0.0;
				Kj = 0.0;

			}
//if(Encflag == 1)printf("%g %g %g %g %g %g %g %g %g %g %g %g\n", time, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, Bdd0, Bd0, B0, (Bdd1 - Bdd0) / dt);
		// *****************
		}
		if(Encflag == 1){
			Enc = 2;
//if (E == 0 || E >= 2)printf("EE %d %d %.40g %.40g %.40g %.40g\n", i, j, x4i.x, x4j.x, v4i.x, v4j.x);
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
		return Enc;
	}
	else return 0;
}

// **************************************
// For test particles
//This reads all encounter pairs from the prechecker, and calls the encounter function
//to detect close encounter pairs.
//All close encounter pairs are stored in the array Encpairs2_d. 
//The number of close encounter pairs is stored in Nencpairs2_d.
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
// ****************************************
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
		enccount = encounter<0>(x4_d[ii], v4_d[ii], xold_d[ii], vold_d[ii], x4_d[jj], v4_d[jj], xold_d[jj], vold_d[jj], rcrit_d[ii], rcrit_d[jj], rcritv_d[ii], rcritv_d[jj], dt, ii, jj , test_d, Encpairs2_d, *Nencpairs2_d, 0);
		if(si == 0 && enccount > 0){
			atomicAdd(&enccount_d[ii], 1);
			atomicAdd(&enccount_d[jj], 1);
		}
	}
	else if(id < *Nencpairs_d + *Nencpairssmall_d){
		enccount = encounter<2>(x4small_d[ii], v4small_d[ii], xoldsmall_d[ii], voldsmall_d[ii], x4_d[jj], v4_d[jj], xold_d[jj], vold_d[jj], 0.0, rcrit_d[jj], 0.0, rcritv_d[jj], dt, jj, ii + N , test_d, Encpairssmall2_d, *Nencpairssmall2_d, N);
		if(si == 0 && enccount > 0){
			atomicAdd(&enccountsmall_d[ii], 1);
		}
	}

}
// **************************************
// For the multi simulation mode
//This reads all encounter pairs from the prechecker, and calls the encounter function
//to detect close encounter pairs.
//All close encounter pairs are stored in the array Encpairs2_d. 
//The number of close encounter pairs is stored in Nencpairs2_d.
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
// ****************************************
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
		int enccount = encounter<3>(x4_d[ii], v4_d[ii], xold_d[ii], vold_d[ii], x4_d[jj], v4_d[jj], xold_d[jj], vold_d[jj], rcrit_d[ii], rcrit_d[jj], rcritv_d[ii], rcritv_d[jj], dt * FGt, ii, jj , test_d, Encpairs2_d, Nencpairs2_d[st], 0);
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
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
// ****************************************
__global__ void encounter_kernel(double4 *x4_d, double4 *v4_d, double4 *xold_d, double4 *vold_d, double *rcrit_d, double *rcritv_d, double dt, int *Nencpairs_d, int2 *Encpairs_d, int *Nencpairs2_d, int2 *Encpairs2_d, double *test_d, int *enccount_d, int si, double *K_d, double *Kold_d, double *Bdd1old_d, int NB, double t, int *groupIndexOld_d, double3 *aold_d){
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
	int enccount = 0;
	
	if(id < *Nencpairs_d){
#if G3 == 0
		enccount = encounter<0>(x4_d[ii], v4_d[ii], xold_d[ii], vold_d[ii], x4_d[jj], v4_d[jj], xold_d[jj], vold_d[jj], rcrit_d[ii], rcrit_d[jj], rcritv_d[ii], rcritv_d[jj], dt, ii, jj , test_d, Encpairs2_d, *Nencpairs2_d, 0);
#else
		enccount = encounterG3<0>(x4_d[ii], v4_d[ii], xold_d[ii], vold_d[ii], x4_d[jj], v4_d[jj], xold_d[jj], vold_d[jj], rcrit_d[ii], rcrit_d[jj], rcritv_d[ii], rcritv_d[jj], dt, ii, jj , test_d, Encpairs2_d, *Nencpairs2_d, 0, K_d[ii * NB + jj], K_d[jj * NB + ii], Kold_d[ii * NB + jj], Kold_d[jj * NB + ii], Bdd1old_d[ii * NB + jj], Bdd1old_d[jj * NB + ii], t, groupIndexOld_d[ii], groupIndexOld_d[jj], aold_d[ii], aold_d[jj]);
#endif
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
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
// ****************************************
template <int BN, int Bl>
__global__ void group_kernel16(int *Nenc_d, double *test_d, int *Nencpairs2_d, int2 *Encpairs2_d, int *groupIndex_d){

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
#if G3 == 1
	if(idy < BN){
		groupIndex_d[idy] = B_s[idy];
//printf("G %d %d %d\n", idy, B_s[idy],  groupIndex_d[idy]);
	}
#endif
#if SERIAL_GROUPING == 0
	if(idy < BN){
		if(B_s[idy] < BN2){
			Encpairs2_d[B_s[idy] * BN + atomicAdd(&encpairs_s[B_s[idy]].y,1)].x = idy;
		}
		// *At this point Encpairs2_d.x contains now line by line the members of the groups, encpairs_s.y contains the sizes of the groups* /
	}
#endif
#if SERIAL_GROUPING == 1
	if(idy == 0){
		for(int i = BN - 1; i >=0; --i){
			if(B_s[i] < BN2){
				Encpairs2_d[B_s[i] * BN + atomicAdd(&encpairs_s[B_s[i]].y,1)].x = i;
			}
		}
	}
#endif
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


//**************************************
//not up to date

//This Kernel sorts a sub-set of close encounter pairs into independent groups, using a 
//parallel sorting algorithm. 
//This kernel works in the case of more than 512 close encounter pairs, and 
//less than 1025 Bodies.
//
//This Kernel is launched with Ne/512 blocks a 512 threads, with Ne the number of close encounters.
//A Fusion Kernel has to be called to merger the sub sets.
//
//Authors: Simon Grimm, Joachim Stadel
//March 2014
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
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
// ****************************************
template <int BN, int Bl>
__global__ void group1024_kernel(int *Nenc_d, double *test_d, int *Nencpairs2_d, int2 *Encpairs2_d, int2 *Encpairs_d, double4 *x4_d, double *rcrit_d, int *groupIndex_d){

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
#if G3 == 1
		for(int i = 0; i < BN; i += Bl){
			groupIndex_d[idy + i] = Encpairs_d[iy + i].x;
		}
#endif
#if SERIAL_GROUPING == 0
		for(int i = 0; i < BN; i += Bl){
			if(Encpairs_d[idy + i].x < BN2){
				Encpairs2_d[Encpairs_d[idy + i].x * BN + atomicAdd(&Encpairs2_d[Encpairs_d[idy + i].x].y,1)].x = idy + i;
			}
			/*At this point Encpairs2_d.x contains now line by line the members of the groups, Encpsirs2_d.y contains the sizes of the groups*/

			__syncthreads();
		}
#endif
#if SERIAL_GROUPING == 1
	if(idy == 0){
		for(int i = BN - 1; i >=0; --i){
			if(Encpairs_d[i].x < BN2){
				Encpairs2_d[Encpairs_d[i].x * BN + atomicAdd(&Encpairs2_d[Encpairs_d[i].x].y,1)].x = i;
			}
		}
	}
#endif

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

// **************************************
//This kernel merges differrent sub-sets of close encounter groups, usig
//a parallel sorting algorithm.
//It classifies the groups into sets of equal sizes.
//The size of group i is stored in Encpairs2_d[i].y, the elements j of the 
//group i are stored in Encpairs2_d[i * BN + j].x
//In Nenc_d[0] is stored the total number of groups.
//in Nenc_d[i] is stored the number of groups with: 2^(2-1) < size of group < 2^(2+1)
//
//Authors: Simon Grimm, Joachim Stadel
//March 2014
//
//This Kernel must be launched  with only one block!
// ****************************************
template <int BN, int Bl> 
__global__ void fusionB_kernel(int *Nenc_d, int2 *Encpairs_d, int2 *Encpairs2_d, int NBlock, double *test_d, double4 *x4_d, double *rcrit_d, int *groupIndex_d){

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
#if G3 == 1
	for(int i = 0; i < BN; i += Bl){
		groupIndex_d[idy + i] = encpairs_s[idy + i];
	}
#endif
#if SERIAL_GROUPING == 0
	for(int i = 0; i < BN; i += Bl){
		if(encpairs_s[idy + i] < BN2){
			Encpairs2_d[encpairs_s[idy + i] * BN + atomicAdd(&encpairs2_s[encpairs_s[idy + i]],1)].x = idy + i; 
		}
	}
#endif
#if SERIAL_GROUPING == 1
	if(idy == 0){
		for(int i = BN - 1; i >=0; --i){
			if(Encpairs_d[i].x < BN2){
				Encpairs2_d[encpairs_s[i] * BN + atomicAdd(&encpairs2_s[encpairs_s[i]],1)].x = i;
			}
		}
	}
#endif

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
//printf("%d %d\n", encpairs_s[idy].x, encpairs_s[idy].y);
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


// **************************************
//This kernel merges two differrenit sub-sets from a tree of close encounter groups, usig
//a parallel sorting algorithm.
//This kernel works in the case of more than 1025 close encounter pairs, and 
//less than 1025 bodies.
//
//This Kernel is launched with Ne/512 blocks, with Ne the number of close encounters.
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
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
//
//Authors: Simon Grimm, Joachim Stadel
//March 2014
//
// ****************************************
template <int BN, int Bl>
__global__ void fusion_kernel(int *Nenc_d, int2 *Encpairs_d, int2 *Encpairs2_d, int NBlock, double *test_d, int *groupIndex_d){

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
#if G3 == 1
	for(int i = 0; i < BN; i += Bl){
		groupIndex_d[idy + i] = encpairs_s[idy + i];
	}
#endif
#if SERIAL_GROUPING == 0
	for(int i = 0; i < BN; i += Bl){
		if(encpairs_s[idy + i] < BN2){
			Encpairs2_d[encpairs_s[idy + i] * BN + atomicAdd(&encpairs2_s[encpairs_s[idy + i]],1)].x = idy + i; 
		}
	}
#endif
#if SERIAL_GROUPING == 1
	if(idy == 0){
		for(int i = BN - 1; i >=0; --i){
			if(Encpairs_d[i].x < BN2){
				Encpairs2_d[encpairs_s[i] * BN + atomicAdd(&encpairs2_s[encpairs_s[i]],1)].x = i;
			}
		}
	}
#endif

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


// **************************************
//This kernel merges two differrenit sub-sets from a tree of close encounter groups, usig
//a parallel sorting algorithm.
//This kernel works in the case of more than 1025 close encounter pairs, and 
//more than 1024 bodies.
//
//This Kernel is launched with Ne/512 blocks, with Ne the number of close encounters.
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
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
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
// ****************************************
template <int BN, int Bl>
__global__ void fusion2_kernel(int *Nenc_d, int2 *Encpairs_d, int2 *Encpairs2_d, int NBlock, double *test_d, int *groupIndex_d){

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
#if G3 == 1
	for(int i = 0; i < BN; i += Bl){
		groupIndex_d[idy + i] = Encpairs_d[idy + i].x;
	}
#endif
#if SERIAL_GROUPING == 0
	for(int i = 0; i < BN; i += Bl){
		if(Encpairs_d[idy + i].x < BN2){
			Encpairs2_d[Encpairs_d[idy + i].x * BN + atomicAdd(&Encpairs2_d[Encpairs_d[idy + i].x].y,1)].x = idy + i; 
		}
	}
#endif
#if SERIAL_GROUPING == 1
	if(idy == 0){
		for(int i = BN - 1; i >=0; --i){
			if(Encpairs_d[i].x < BN2){
			Encpairs2_d[Encpairs_d[i].x * BN + atomicAdd(&Encpairs2_d[Encpairs_d[i].x].y,1)].x = i; 
			}
		}
	}
#endif
	
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
#endif

#ifndef FG2_H
#define FG2_H

#include "Orbit2.h"
#include "BSSingle.h"

//**************************************
// This function copies the aeGrid parameters to constant memory. This functions must be in
// the same file as the use of the constant memory
//
//Authors: Simon Grimm
//April 2015
//
//***************************************/
__host__ void Data::constantCopy(){
	float GridaeP[9] = {Gridae.amin, Gridae.amax, Gridae.emin, Gridae.emax, Gridae.imin, Gridae.imax, Gridae.deltaa, Gridae.deltae, Gridae.deltai};
	int GridaeN[3] = {Gridae.Na, Gridae.Ne, Gridae.Ni};
	cudaMemcpyToSymbol(Gridae_c, GridaeP, 9*sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(GridaeN_c, GridaeN, 3*sizeof(int), 0, cudaMemcpyHostToDevice);
}
//**************************************
// This function copies the use ae grid flag to constant memory. This functions must be in
// the same file as the use of the constant memory
//
//Authors: Simon Grimm
//Mai 2015
//
//***************************************/
__host__ void Data::constantCopy2(){

	cudaMemcpyToSymbol(UseaeGrid_c, &P.UseaeGrid, sizeof(int), 0, cudaMemcpyHostToDevice);

}



//**************************************
//based on a code
//from Joachim Stadel
//See Danby for f and g method
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
//***************************************/
__device__ __noinline__ void fgfull(double4 &x4i, double4 &v4i, double dt, double mu, double &test, double &test2, const double Msun, float4 aelimits, int &aecount, int *Gridaecount_d, int *Gridaicount_d, int si, int id){

	if(x4i.w >= 0){

		double f,g,fd,gd;                               /* Gauss's f, g, fdot and gdot */
		double rsq,vsq,ir;
		double u;                                       /* r v cos(phi) */
		double ia;                                      //inverse of a
		volatile double ria;
		double air_;
		volatile double e;                                        /* eccentricity */
		volatile double ec,es;                               /* e cos(E), e sin(E) */
		double ien;                                   /* inverse mean motion */
		volatile double en;                                    /* mean motion */
		volatile double dec;                                      /* delta E */
		volatile double dm;
		double mw;                                       /* minus function to zero */
		double wp;                                       /* first derivative */
		double iwp;
		volatile double wpp;                                      /* second derivative */
		volatile double wppp;                                     /* third derivative */
		volatile double dx;
		double s,c;
		double t1;
		double3 t;
		const double DOUBLE_EPS = 1.2e-16;
		double converge;
		double UP = 2*M_PI;
		double LOW = -2*M_PI;
		double next;
		int i;
		/*
		* Evaluate some orbital quantites.
		*/

		rsq = x4i.x*x4i.x + x4i.y*x4i.y + x4i.z*x4i.z;
		vsq = v4i.x*v4i.x + v4i.y*v4i.y + v4i.z*v4i.z;
		u =  x4i.x*v4i.x + x4i.y*v4i.y + x4i.z*v4i.z;
		ir = 1.0/sqrt(rsq);
		ia = 2.0*ir-vsq/mu;
		if(ia > 0){

			t1 = ia*ia;
			ria = rsq*ir*ia;
			en = sqrt(mu*t1*ia);
			ien = 1.0/en;
			ec = 1.0-ria;
			es = u*t1*ien;
			e = sqrt(ec*ec + es*es);
			double a = 1.0/ia;
			if(UseaeGrid_c[0] == 1){
				int na = (int)((a - Gridae_c[0]) / Gridae_c[6]);
				int ne = (int)((e - Gridae_c[2]) / Gridae_c[7]);
				if(si == 0 && na >= 0 && na < GridaeN_c[0] && ne >= 0 && ne < GridaeN_c[1]){
					atomicAdd(&Gridaecount_d[ne * GridaeN_c[0] + na], 1); 
		
				}
			
				//compute inclination
				double inc;
				double3 h3;
				h3.x = x4i.y * v4i.z - x4i.z * v4i.y;
				h3.y = -x4i.x * v4i.z + x4i.z * v4i.x;
				h3.z = x4i.x * v4i.y - x4i.y * v4i.x;

				double h2 = h3.x * h3.x + h3.y * h3.y + h3.z * h3.z;
				double h = sqrtf(h2);

				double t = h3.z / h;

				if(t <= -1){
					inc = M_PI;
				}
				else{
					if(t < 1){
						inc = acos(t);
					}
					else inc = 0.0;
				}
				int ni = (int)((inc - Gridae_c[4]) / Gridae_c[8]);

				if(si == 0 && na >= 0 && na < GridaeN_c[0] && ni >= 0 && ni < GridaeN_c[2]){
					atomicAdd(&Gridaicount_d[ni * GridaeN_c[0] + na], 1); 
				}
			}
			if(e >= aelimits.z && e <= aelimits.w){
				if(a >= aelimits.x && a <= aelimits.y){
					aecount = 1;
				}
			}
			dm = en * dt - es;
			if((es*cos(dm)+ec*sin(dm)) > 0){
				dec = __fma_rn(0.85, e, dm); //dm + 0.85*e;
			}
			else dec = __fma_rn(-0.85, e, dm); //dm - 0.85*e;
			converge = fabs(en * dt *DOUBLE_EPS);
			for(i = 0; i < 32; ++i) {

				//s = sin(dec);
				//c = cos(dec);
				sincos(dec, &s, &c);
				wpp = ec*s + es*c;
				wppp = ec*c - es*s;
				mw = dm - dec + wpp;
				if(mw < 0){
					UP = dec;
				}
				else LOW = dec;
				wp = 1.0 - wppp;
				wpp *= 0.5;
				dx = mw/wp;
				dx = mw/(wp + dx*wpp);
				dx = mw/(wp + dx*(wpp + (1.0/6.0)*dx*wppp));
				next = dec + dx;
				if (fabs(dx) <= converge) break;
				if(next > LOW && next < UP){
					dec = next;
				}
				else dec = 0.5*(LOW + UP);
				if (dec==LOW || dec==UP) break;
			}
			if(i < 31){
				iwp = 1.0/wp;
				air_ = -1.0/ria;
				t1 = (1.0-c);
				f = 1.0 + air_*t1;
				g = dt + (s-dec)*ien;
				fd = air_*iwp*s*en;
				gd = 1.0 - iwp*t1;

				t.x = f*x4i.x+g*v4i.x;
				t.y = f*x4i.y+g*v4i.y;
				t.z = f*x4i.z+g*v4i.z;

				v4i.x = fd*x4i.x+gd*v4i.x;
				v4i.y = fd*x4i.y+gd*v4i.y;
				v4i.z = fd*x4i.z+gd*v4i.z;

				x4i.x = t.x;
				x4i.y = t.y;
				x4i.z = t.z;

			}
			else{
				BSSinglestep(x4i, v4i, Msun, dt, test, id);
			}
		}
		else{
			BSSinglestep(x4i, v4i, Msun, dt, test, id);
		}
	}
}

//**************************************
//this functions is not jet fully working and not energy conserving
//
//based on a code
//from Joachim Stadel
//See Danby for f and g method
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
//***************************************/
__device__ void fastfg(double4 &x4i, double4 &v4i, double dt, double mu, double &test, const double Msun, float4 aelimits, int &aecount, int *Gridaecount_d, int si, int id){

	if(x4i.w >= 0){
		int ii,i,j,jnew;
		double sgn,dEj,f0,f1,f2,f3;
		double y,dy,y2,y4,A,B;

		double s,c,wp;
		double f,g,fd,gd;	  // Gauss's f, g, fdot and gdot
		double rsq, ir,vsq;
		double u;		  // r v cos(phi)
		double ia;		  // semi-major axis
		double ec,es;		  // e cos(E), e sin(E)
		double ien, en;		  // mean motion
		double dM;		  // delta mean anomoly
		double t1;
		double ria, air_, iwp;
		double3 t;

		s = 0.0;
		c = 0.0;

		int ok = 1;
		
		rsq = x4i.x*x4i.x + x4i.y*x4i.y + x4i.z*x4i.z;
		vsq = v4i.x*v4i.x + v4i.y*v4i.y + v4i.z*v4i.z;
		u =  x4i.x*v4i.x + x4i.y*v4i.y + x4i.z*v4i.z;
		ir = 1.0/sqrt(rsq);
		ia = 2.0*ir-vsq/mu;

		if(ia > 0){

			t1 = ia*ia;
			ria = rsq*ir*ia;
			en = sqrt(mu*t1*ia);
			ien = 1.0/en;
			ec = 1.0-ria;
			es = u*t1*ien;
			dM = en * dt;

			dEj = dM;
			sgn = dEj < 0 ? -1.0 : 1.0;
			dM -= es;
			j = (int)floor(sgn*dEj*N_PI) + 1;

			for(i = 0; i < 6; ++i){  //32

				f2 = es*C_c[j] + sgn*ec*S_c[j];
				f3 = -sgn*es*S_c[j] + ec*C_c[j];
				dEj = sgn*j*PI_N;
				f0 = dEj - dM - f2;
				f1 = 1.0 - f3;
				y = -f0/f1;
				dEj += y;
				jnew = (int)floor(sgn*dEj*N_PI) + 1;
				if (jnew == j) break;
				j = jnew;
			}
			if(i >= 5) ok = 0; //31

			dEj = sgn*j*PI_N;
			y = (-f0 + y*y*(0.5*f2 + (1.0/3.0)*f3*y))/(f1 + y*(f2 + 0.5*f3*y));

			for(ii = 0; ii < 6; ++ii){ //32
				y2 = y*y;
				y4 = y2*y2;
				B = f2*y*(1.0 - (1.0/6.0)*y2 + (1.0/120.0)*y4) + 0.5*f3*y2*(1.0 - (1.0/12.0)*y2 + (1.0/360.0)*y4);
				A = 0.5*f2*y*(1.0 - 0.25*y2 + (1.0/72.0)*y4) + (1.0/3.0)*f3*y2*(1.0 - 0.1*y2 + (1.0/280.0)*y4);	
				dy = (-f0 + y*A)/(f1 + B) - y;
				if (fabs(dy) < 1e-20) break;
				y += dy;
			}
			if(ii >= 5) ok = 0; //31
			dEj += y;
			sincos(dEj, &s, &c);

			air_ = -1.0/ria;
			t1 = 1.0 - c;
			wp = 1.0 - ec*c + es*s;
			iwp = 1.0/wp;
			f = 1.0 + air_ * t1;
			g = dt + (s-dEj)*ien;
			fd = air_ * iwp * en * s;
			gd = 1.0 - t1*iwp;

		}
		if(ia <= 0 || ok == 0){
//	printf("%g %d %d", ia, i, ii);
			//fgfull(x4i, v4i, dt, ksq * Msun, test, test, Msun, aelimits, aecount, Gridaecount_d, si, id);
			BSSinglestep(x4i, v4i, Msun, dt, test, id);
		}
		else{
			t.x = f*x4i.x+g*v4i.x;
			t.y = f*x4i.y+g*v4i.y;
			t.z = f*x4i.z+g*v4i.z;
			v4i.x = fd*x4i.x+gd*v4i.x;
			v4i.y = fd*x4i.y+gd*v4i.y;
			v4i.z = fd*x4i.z+gd*v4i.z;
			x4i.x = t.x;
			x4i.y = t.y;
			x4i.z = t.z;
		}
	}
}
// ******************************************
//This function calculates the Poincare surcafe of section
//It markes paricles crossing the section and set the Flag PFlag_d
//Authors: Simon Grimm, Joachim Stadel
//March 2014
//
// ******************************************
__global__ void PoincareSection(double4 *x4_d, double4 *v4_d, double4 *xold_d, double4 *vold_d, double Msun, int N, int si, int *PFlag_d){

        int idy = threadIdx.x;
        int id = blockIdx.x * blockDim.x + idy;

	double test;

	float4 aelimits;
	aelimits.x = 0.0f;
	aelimits.y = 0.0f;
	aelimits.z = 0.0f;
	aelimits.w = 0.0f;
	int aecount = 0;
	int aicount = 0;


	if(id < N && si == 0){
		double4 x4i = x4_d[id];
		double4 x4oldi = xold_d[id];
		double4 v4oldi = vold_d[id];
		if(x4oldi.y < 0.0 && x4i.y >= 0.0 && x4i.x > 0.0){
			PFlag_d[0] = 1;
			double dtt = -x4oldi.y / v4oldi.y;
			fgfull(x4oldi, v4oldi, dtt, ksq * Msun, test, test, Msun, aelimits, aecount, &aecount, &aicount, si, id);
//			printf("%g %g %g\n", x4oldi.x, x4oldi.y, v4oldi.x);
			xold_d[id] = x4oldi;
			vold_d[id] = v4oldi;
			vold_d[id].w *= -1.0;		//Flag particles
		}
	}


}


// **************************************
//The fg_kernel does a copy of the coordinates and calls the FG function to perform the Kepler drift.
//There are 2 different FG, and one Burlish Stoer function, fastest one is fastfg.
//This Kernel is launched wich NB/Bl blocks with Bl threads. NB is the next bigger number of N
//which is a power of two.
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
// *****************************************
template <int Bl>
__global__ void fg_kernel(double4 *x4_d, double4 *v4_d, double4 *xold_d, double4 *vold_d, double3 *a_d, int *groupIndex_d, int *groupIndexOld_d, double dt, const double Msun, double *test_d, int N, float4 *aelimits_d, int *aecount_d, int *Gridaecount_d, int *Gridaicount_d, int si){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	__shared__ double4 x4_s[Bl];
	__shared__ double4 v4_s[Bl];
	if(id < N){
		int aecount = 0;
		x4_s[idy] = x4_d[id];
		v4_s[idy] = v4_d[id];
		__syncthreads();
		xold_d[id] = x4_s[idy];
		vold_d[id] = v4_s[idy];
#if G3 == 1
		groupIndexOld_d[id] = groupIndex_d[id];
#endif

		a_d[id].x = 0.0;
		a_d[id].y = 0.0;
		a_d[id].z = 0.0;

#if G3 == 1
		groupIndexOld_d[id] = groupIndex_d[id];
#endif
		double test;
		float4 aelimits = aelimits_d[idy];
		//fastfg(x4_s[idy], v4_s[idy], dt, ksq * Msun, test, Msun, aelimits, aecount, Gridaecount_d, si, id);
		fgfull(x4_s[idy], v4_s[idy], dt, ksq * Msun, test, test, Msun, aelimits, aecount, Gridaecount_d, Gridaicount_d, si, id);
//		BSSinglestep(x4_s[idy], v4_s[idy], Msun, dt, test, test);
		__syncthreads();
		x4_d[id] = x4_s[idy];
		v4_d[id] = v4_s[idy];
#if G3 == 1
		groupIndex_d[id] = -1;
#endif
		if(si == 0){
		aecount_d[id] += aecount;
		}
	}
}

// **************************************
//for test particles
//The fg_kernel does a copy of the coordinates and calls the FG function to perform the Kepler drift.
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
// *****************************************
template <int Bl>
__global__ void fgsmall_kernel(double4 *x4_d, double4 *v4_d, double4 *xold_d, double4 *vold_d, double3 *a_d, double dt, const double Msun, double *test_d, double4 *x4small_d, double4 *v4small_d, double4 *xoldsmall_d, double4 *voldsmall_d, double3 *asmall_d, int Nsmall, int N, float4 *aelimits_d, float4 *aelimitssmall_d, int *aecount_d, int *aecountsmall_d, int *Gridaecount_d, int *Gridaicount_d, int si){

        int idy = threadIdx.x;
        int id = blockIdx.x * blockDim.x + idy;

        __shared__ double4 x4_s[Bl];
        __shared__ double4 v4_s[Bl];

	float4 aelimits;
	int aecount = 0;

	if(id < N){
        	x4_s[idy] = x4_d[id];
        	v4_s[idy] = v4_d[id];
		aelimits = aelimits_d[id];
	        xold_d[id] = x4_s[idy];
        	vold_d[id] = v4_s[idy];
		a_d[id].x = 0.0;
		a_d[id].y = 0.0;
		a_d[id].z = 0.0;
	}
	else if(id < N + Nsmall){
		x4_s[idy] = x4small_d[id - N];
		v4_s[idy] = v4small_d[id - N];
		aelimits = aelimitssmall_d[id - N];
	        xoldsmall_d[id - N] = x4_s[idy];
        	voldsmall_d[id - N] = v4_s[idy];
		asmall_d[id - N].x = 0.0;
		asmall_d[id - N].y = 0.0;
		asmall_d[id - N].z = 0.0;
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
		aelimits.x = 0.0f;
		aelimits.y = 0.0f;
		aelimits.z = 0.0f;
		aelimits.w = 0.0f;
		
	}
        __syncthreads();
        
	double test;

	if(id < N + Nsmall){
        	//fastfg(x4_s[idy], v4_s[idy], dt, ksq * Msun, test, Msun, aelimits, aecount, Gridaecount_d, si, id);
        	fgfull(x4_s[idy], v4_s[idy], dt, ksq * Msun, test, test, Msun, aelimits, aecount, Gridaecount_d, Gridaicount_d, si, id);
        	//BSSinglestep(x4_s[idy], v4_s[idy], Msun, dt, test, id);
        }
        __syncthreads();

	if(id < N){
        	x4_d[id] = x4_s[idy];
        	v4_d[id] = v4_s[idy];
		if(si == 0){
			aecount_d[id] += aecount;
		}
	}
	else if (id < N + Nsmall){
                x4small_d[id - N] = x4_s[idy];
                v4small_d[id - N] = v4_s[idy];
		if(si == 0){
			aecountsmall_d[id - N] += aecount;
		}
	}

}
// **************************************
//for multi simulation mode
//The fg_kernel does a copy of the coordinates and calls the FG function to perform the Kepler drift.
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
// *****************************************
template <int Bl>
__global__ void fgM_kernel(double4 *x4_d, double4 *v4_d, double4 *xold_d, double4 *vold_d, double *dt_d, const double *Msun_d, double *test_d, int *index_d, int NT, float4 *aelimits_d, int *aecount_d, int *Gridaecount_d, int *Gridaicount_d, double FGt, int si){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;
	int st = index_d[id] / 100;

	__shared__ double4 x4_s[Bl];
	__shared__ double4 v4_s[Bl]; 

	if(id < NT){
		int aecount = 0;
		x4_s[idy] = x4_d[id];
		v4_s[idy] = v4_d[id];
		__syncthreads();

		xold_d[id] = x4_s[idy];
		vold_d[id] = v4_s[idy];
		double test;
		double Msun = Msun_d[st];
		double dt = dt_d[st];
		float4 aelimits = aelimits_d[id];
		//fastfg(x4_s[idy], v4_s[idy], dt * FGt, ksq * Msun, test, Msun, aelimits, aecount, Gridaecount_d, si, id);
		fgfull(x4_s[idy], v4_s[idy], dt * FGt, ksq * Msun, test, test, Msun, aelimits, aecount, Gridaecount_d, Gridaicount_d, si, id);
		//BSSinglestep(x4_s[idy], v4_s[idy], Msun, dt * FGt, test, test);
		__syncthreads();
		x4_d[id] = x4_s[idy];
		v4_d[id] = v4_s[idy];

		if(si == 0){
			aecount_d[id] += aecount;
		}
	}
}
#endif

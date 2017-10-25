#include "define.h"

// **************************************
//This Kernels performs the Sun-Kick 1/Msun * Sum(p_i)^2 on all the bodies.
//It uses a parallel reduction fomula to calculate the sum in log(N) steps.
//
//It works for the case of more than 65 bodies.
//Each Kernel is launched with 3 blocks, one for each dimension.
//
//E = 1 : perform C Kick.
//E = 2 : perform C Kick + reset Nencpairs 
//
//Authors: Simon Grimm
//July 2016
//*****************************************
template <int Bl, int E>
__global__ void HC128b_kernel(double4 *x4_d, double4 *v4_d, const double dt, const double dtiMsun, int *Nencpairs_d, int *Nencpairs2_d, int *Nenc_d, int N, double t, int UseForce){

	int idy = threadIdx.x;
	int idx = blockIdx.x;

	if(E == 1){
		if(idy == 0 && idx == 0){
			Nencpairs2_d[0] = 0;		//this variable is needed in the Encounter kernel
	}
	if(idy < def_GMax && idx == 0) Nenc_d[idy] = 0;
	}
	if(E == 2){
		if(idy == 0 && idx == 0){
			Nencpairs_d[0] = 0;	//This variable is needed in the Kick_kernel
		}
	}
	__shared__ double a1_s[Bl];

	a1_s[idy] = 0.0;

	__syncthreads(); 
	for (int i = 0; i < N; i+= Bl){
		if(idy + i < N){
			double m = x4_d[idy + i].w;
			if(m > 0.0){
				if(idx == 0){
					a1_s[idy] += __dmul_rn(m, v4_d[idy + i].x);
				}
				if(idx == 1){
					a1_s[idy] += __dmul_rn(m, v4_d[idy + i].y);
				}
				if(idx == 2){
					a1_s[idy] += __dmul_rn(m, v4_d[idy + i].z);
				}
			}
		}
	}
	__syncthreads();

	if(Bl >= 512){
		if(idy < 256){
			a1_s[idy] += a1_s[idy + 256];

		}
	}
	__syncthreads();

	if(Bl >= 256){
		if(idy < 128){
			a1_s[idy] += a1_s[idy + 128];

		}
	}
	__syncthreads();

	if(Bl >= 128){
		if(idy < 64){
			a1_s[idy] += a1_s[idy + 64];
		}
	}
	__syncthreads();

	if(idy < 32){
		volatile double *a = a1_s;
		a[idy] += a[idy + 32];
		a[idy] += a[idy + 16];
		a[idy] += a[idy + 8];
		a[idy] += a[idy + 4];
		a[idy] += a[idy + 2];
		a[idy] += a[idy + 1];
	}
	__syncthreads();
	for(int i = 0; i < N; i += Bl){
		if(idy + i < N){
			if(idx == 0) x4_d[idy + i].x += __dmul_rn(a1_s[0], dtiMsun);
			if(idx == 1) x4_d[idy + i].y += __dmul_rn(a1_s[0], dtiMsun);
			if(idx == 2) x4_d[idy + i].z += __dmul_rn(a1_s[0], dtiMsun);
			if(UseForce & 1){
				double c2 = def_cm * def_cm;
				double4 v4 = v4_d[idy + i];
				double vsq = v4.x * v4.x + v4.y * v4.y + v4.z * v4.z;
				double vcdt = 2.0 * vsq / c2 * dt;
				if(idx == 0) x4_d[idy + i].x -= __dmul_rn(v4.x, vcdt);
				if(idx == 1) x4_d[idy + i].y -= __dmul_rn(v4.y, vcdt);
				if(idx == 2) x4_d[idy + i].z -= __dmul_rn(v4.z, vcdt);
			}
		}
	}
}

//**************************************
//This Kernels performs the Sun-Kick 1/Msun * Sum(p_i)^2 on all the bodies.
//It uses a parallel reduction fomula to calculate the sum in log(N) steps.
//
//It works for the case of less than 65 bodies.
//Each Kernel is launched with 3 blocks, one for each dimension.
//
//E = 0 : perform C Kick 
//E = 1 : perform C Kick + reset Nencpairs2
//E = 2 : perform C Kick + reset Nencpairs
//
//Authors: Simon Grimm
//November 2016
//  *****************************************
template < int Bl2, int E>
__global__ void HC32_kernel(double4 *x4_d, double4 *v4_d, const double dt, const double dtiMsun, int *Nencpairs_d, int *Nencpairs2_d, int *Nenc_d, int N, int UseForce){

	int idy = threadIdx.x;
	int idx = blockIdx.x;
	if(E == 1){
		if(idy == 0 && idx == 0){
			Nencpairs2_d[0] = 0;		//this variable is needed in the Encounter kernel
		}
		if(idy < def_GMax && idx == 0) Nenc_d[idy] = 0;
	}
	if(E == 2){
		if(idy == 0 && idx == 0){
			Nencpairs_d[0] = 0;		//This variable is needed in the Kick_kernel
		}
	}
	__shared__ double a1_s[Bl2];

	for(int i = 0; i < Bl2; i += blockDim.x){
		a1_s[idy + i] = 0.0;
	}
	__syncthreads();
	double m = -1.0e-12;
	if(idy < N){
		m = x4_d[idy].w;
		if(m > 0.0){
			if(idx == 0){
				a1_s[idy] = m * v4_d[idy].x;
			}
			if(idx == 1){
				a1_s[idy] = m * v4_d[idy].y;
			}
			if(idx == 2){
				a1_s[idy] = m * v4_d[idy].z;
			}
		}
	}

	__syncthreads();
	if(idy < 32){
		volatile double *a = a1_s;
		if(blockDim.x >= 64) a[idy] += a[idy + 32];
		if(blockDim.x >= 32) a[idy] += a[idy + 16];
		a[idy] += a[idy + 8];
		a[idy] += a[idy + 4];
		a[idy] += a[idy + 2];
		a[idy] += a[idy + 1];
	}
	__syncthreads();
	if(idy < N){
		if(idx == 0) x4_d[idy].x += a1_s[0] * dtiMsun;
		if(idx == 1) x4_d[idy].y += a1_s[0] * dtiMsun;
		if(idx == 2) x4_d[idy].z += a1_s[0] * dtiMsun;
//if(idx == 0) printf("HCx %d %.20e %.20e %.20e\n", idy, x4_d[idy].x, a1_s[0], dtiMsun);
//if(idx == 1) printf("HCy %d %.20e %.20e %.20e\n", idy, x4_d[idy].y, a1_s[0], dtiMsun);
//if(idx == 2) printf("HCz %d %.20e %.20e %.20e\n", idy, x4_d[idy].z, a1_s[0], dtiMsun);
			if(UseForce & 1){
				double c2 = def_cm * def_cm;
				double4 v4 = v4_d[idy];
				double vsq = v4.x * v4.x + v4.y * v4.y + v4.z * v4.z;
				double vcdt = 2.0 * vsq / c2 * dt;
				if(idx == 0) x4_d[idy].x -= __dmul_rn(v4.x, vcdt);
				if(idx == 1) x4_d[idy].y -= __dmul_rn(v4.y, vcdt);
				if(idx == 2) x4_d[idy].z -= __dmul_rn(v4.z, vcdt);
			}
	}
}

// **************************************
//Used for Multi Simulation Mode
//This Kernels performs the Sun-Kick 1/Msun * Sum(p_i)^2 on all the bodies.
//It uses a parallel reduction fomula to calculate the sum in log(N) steps.
//
//It works for the case of less than 16bodies.
//Each Kernel is launched with 3 blocks, one for each dimension.
//E = 1 : perform C Kick.
//E = 2 : perform C Kick + reset Nencpairs
//
//Authors: Simon Grimm
//JUly 2016
//
//*****************************************
template <int Bl, int Bl2, int Nmax, int E>
__global__ void HCM2_kernel(double4 *x4_d, double4 *v4_d, const double *dt_d, const double4 *Msun_d, int *index_d, int NT, double Ct, double *test_d, int *Nencpairs_d, int *Nencpairs2_d, int *Nenc_d, int Nst, int UseForce, int Nstart){

	int idy = threadIdx.x;
	int id = blockIdx.x * Bl2 + idy - Nmax + Nstart;
	__shared__ volatile double3 p_s[Bl + Nmax / 2];
	__shared__ int st_s[Bl + Nmax / 2];
	volatile double dtiMsun;
	volatile double dt;
	if(E == 1){
		if(id >= Nstart && id < Nst + 1 + Nstart){
			Nencpairs2_d[id - Nstart] = 0;		//This variable is needed in the Encounter_kernel
		}
		if(id >= Nstart && id < def_GMax + Nstart){
			Nenc_d[id - Nstart] = 0;
		}
	}
		
	if(E == 2){
		if(id >= Nstart && id < Nst + 1 + Nstart){
			Nencpairs_d[id - Nstart] = 0;		//This variable is needed in the Kick_kernel
		}
	}
	if(id < NT + Nstart && id >= Nstart){
		st_s[idy] = index_d[id] / 100;
		volatile double m = x4_d[id].w;
		p_s[idy].x = m * v4_d[id].x;
		p_s[idy].y = m * v4_d[id].y;
		p_s[idy].z = m * v4_d[id].z;
		dt = dt_d[st_s[idy]] * Ct;
		dtiMsun = dt / Msun_d[st_s[idy]].x;
	}
	else{
		st_s[idy] = -idy-1;
		p_s[idy].x = 0.0;
		p_s[idy].y = 0.0;
		p_s[idy].z = 0.0;
		dtiMsun = 0.0;
		dt = 0.0;
	}
	//halo
	if(idy < Nmax / 2){
		//right
		if(id + Bl < NT + Nstart){
			st_s[idy + Bl] = index_d[id + Bl] / 100;
			volatile double m = x4_d[id + Bl].w;
			p_s[idy + Bl].x = m * v4_d[id + Bl].x;
			p_s[idy + Bl].y = m * v4_d[id + Bl].y;
			p_s[idy + Bl].z = m * v4_d[id + Bl].z;
		}
		else{
			st_s[idy + Bl] = -idy-Bl-1;
			p_s[idy + Bl].x = 0.0;
			p_s[idy + Bl].y = 0.0;
			p_s[idy + Bl].z = 0.0;
		}
	}

	volatile int f;
	volatile double px;
	volatile double py;
	volatile double pz;
	if(Nmax >= 64){
		__syncthreads();
		f = ((st_s[idy] - st_s[idy + 32]) == 0);		//one if sti == stj, zero else
		px = (p_s[idy + 32].x) * f;	
		py = (p_s[idy + 32].y) * f;
		pz = (p_s[idy + 32].z) * f;

		__syncthreads();
	
		p_s[idy].x += px;
		p_s[idy].y += py;
		p_s[idy].z += pz;
	}

	if(Nmax >= 32){
		__syncthreads();
		f = ((st_s[idy] - st_s[idy + 16]) == 0);		//one if sti == stj, zero else
		px = (p_s[idy + 16].x) * f;	
		py = (p_s[idy + 16].y) * f;
		pz = (p_s[idy + 16].z) * f;

		__syncthreads();
	
		p_s[idy].x += px;
		p_s[idy].y += py;
		p_s[idy].z += pz;
	}

	if(Nmax >= 16){
		__syncthreads();
		f = ((st_s[idy] - st_s[idy + 8]) == 0);		//one if sti == stj, zero else
		px = (p_s[idy + 8].x) * f;	
		py = (p_s[idy + 8].y) * f;
		pz = (p_s[idy + 8].z) * f;

		__syncthreads();
	
		p_s[idy].x += px;
		p_s[idy].y += py;
		p_s[idy].z += pz;
	}

	if(Nmax >= 8){
		__syncthreads();
		f = ((st_s[idy] - st_s[idy + 4]) == 0);		//one if sti == stj, zero else
		px = (p_s[idy + 4].x) * f;
		py = (p_s[idy + 4].y) * f;
		pz = (p_s[idy + 4].z) * f;

		__syncthreads();

		p_s[idy].x += px;
		p_s[idy].y += py;
		p_s[idy].z += pz;
	}

	__syncthreads();

	f = ((st_s[idy] - st_s[idy + 2]) == 0);			//one if sti == stj, zero else
	px = (p_s[idy + 2].x) * f;
	py = (p_s[idy + 2].y) * f;
	pz = (p_s[idy + 2].z) * f;

	__syncthreads();

	p_s[idy].x += px;
	p_s[idy].y += py;
	p_s[idy].z += pz;

	__syncthreads();

	f = ((st_s[idy] - st_s[idy + 1]) == 0);			//one if sti == stj, zero else
	px = (p_s[idy + 1].x) * f;
	py = (p_s[idy + 1].y) * f;
	pz = (p_s[idy + 1].z) * f;

	__syncthreads();

	p_s[idy].x += px;
	p_s[idy].y += py;
	p_s[idy].z += pz;

	__syncthreads();
	//sum is complete, now distribute solution
	f = ((st_s[idy] - st_s[idy + 1]) == 0);
	px = (p_s[idy].x) * f + (1 - f) * p_s[idy + 1].x;
	py = (p_s[idy].y) * f + (1 - f) * p_s[idy + 1].y;
	pz = (p_s[idy].z) * f + (1 - f) * p_s[idy + 1].z;

	__syncthreads();
	p_s[idy + 1].x = px;
	p_s[idy + 1].y = py;
	p_s[idy + 1].z = pz;
	__syncthreads();

	f = ((st_s[idy] - st_s[idy + 2]) == 0);
	px = (p_s[idy].x) * f + (1 - f) * p_s[idy + 2].x;
	py = (p_s[idy].y) * f + (1 - f) * p_s[idy + 2].y;
	pz = (p_s[idy].z) * f + (1 - f) * p_s[idy + 2].z;

	__syncthreads();
	p_s[idy + 2].x = px;
	p_s[idy + 2].y = py;
	p_s[idy + 2].z = pz;
	__syncthreads();

	if(Nmax >= 8){
		f = ((st_s[idy] - st_s[idy + 4]) == 0);
		px = (p_s[idy].x) * f + (1 - f) * p_s[idy + 4].x;
		py = (p_s[idy].y) * f + (1 - f) * p_s[idy + 4].y;
		pz = (p_s[idy].z) * f + (1 - f) * p_s[idy + 4].z;

		__syncthreads();
		p_s[idy + 4].x = px;
		p_s[idy + 4].y = py;
		p_s[idy + 4].z = pz;
		__syncthreads();
	}

	if(Nmax >= 16){
		f = ((st_s[idy] - st_s[idy + 8]) == 0);
		px = (p_s[idy].x) * f + (1 - f) * p_s[idy + 8].x;
		py = (p_s[idy].y) * f + (1 - f) * p_s[idy + 8].y;
		pz = (p_s[idy].z) * f + (1 - f) * p_s[idy + 8].z;

		__syncthreads();
		p_s[idy + 8].x = px;
		p_s[idy + 8].y = py;
		p_s[idy + 8].z = pz;
		__syncthreads();
	}

	if(Nmax >= 32){
		f = ((st_s[idy] - st_s[idy + 16]) == 0);
		px = (p_s[idy].x) * f + (1 - f) * p_s[idy + 16].x;
		py = (p_s[idy].y) * f + (1 - f) * p_s[idy + 16].y;
		pz = (p_s[idy].z) * f + (1 - f) * p_s[idy + 16].z;

		__syncthreads();
		p_s[idy + 16].x = px;
		p_s[idy + 16].y = py;
		p_s[idy + 16].z = pz;
		__syncthreads();
	}

	if(Nmax >= 64){
		f = ((st_s[idy] - st_s[idy + 32]) == 0);
		px = (p_s[idy].x) * f + (1 - f) * p_s[idy + 32].x;
		py = (p_s[idy].y) * f + (1 - f) * p_s[idy + 32].y;
		pz = (p_s[idy].z) * f + (1 - f) * p_s[idy + 32].z;

		__syncthreads();
		p_s[idy + 32].x = px;
		p_s[idy + 32].y = py;
		p_s[idy + 32].z = pz;
		__syncthreads();
	}

	if(id < NT + Nstart && id >= Nstart && idy >= Nmax && idy < Bl - Nmax / 2 && x4_d[id].w >= 0){
		x4_d[id].x += p_s[idy].x * dtiMsun;
		x4_d[id].y += p_s[idy].y * dtiMsun;
		x4_d[id].z += p_s[idy].z * dtiMsun;
//printf("HCx %d %.20e %.20e %.20e\n", id, x4_d[id].x, p_s[idy].x, dtiMsun);
//printf("HCy %d %.20e %.20e %.20e\n", id, x4_d[id].y, p_s[idy].y, dtiMsun);
//printf("HCz %d %.20e %.20e %.20e\n", id, x4_d[id].z, p_s[idy].z, dtiMsun);
		if(UseForce & 1){// GR part depending on velocity only (see Saha & Tremaine 1994)
			double c2 = def_cm * def_cm;
			double4 v4 = v4_d[id];
			double vsq = v4.x * v4.x + v4.y * v4.y + v4.z * v4.z;
			double vcdt = 2.0*vsq/c2 * dt;
			x4_d[id].x -= v4.x * vcdt;
			x4_d[id].y -= v4.y * vcdt;
			x4_d[id].z -= v4.z * vcdt;
 		}
	}

}

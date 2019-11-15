#include "define.h"


//**************************************
//This Kernels performs the Sun-Kick 1/Msun * Sum(p_i)^2 on all the bodies.
//It uses a parallel reduction fomula to calculate the sum in log(N) steps.
//
//It works for the case of multiple blocks
//must be followed by HC32d2 and HC32d3
//
//Uses shuffle instructions
//Authors: Simon Grimm
//April 2019
//  *****************************************
__global__ void HC32d1_kernel(double4 *x4_d, double4 *v4_d, double3 *a_d, int N){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	int idx = blockIdx.y;

	double a = 0.0;
	double vi;

	__shared__ double a_s[32];
	int lane = threadIdx.x % warpSize;
	int warp = threadIdx.x / warpSize;

	if(warp == 0){
		a_s[threadIdx.x] = 0.0;
	}


	for(int i = 0; i < N; i += blockDim.x * gridDim.x){	//gridDim.x is for multiple bock reduce
		if(id + i < N){
			double m = x4_d[id + i].w;
			if(idx == 0){
				vi = v4_d[id + i].x;
			}
			if(idx == 1){
				vi = v4_d[id + i].y;
			}
			if(idx == 2){
				vi = v4_d[id + i].z;
			}
			if(m > 0.0){
				a += m * vi;
//if(idx == 0) printf("HCax %d %d %.20g\n", i, id, a);
//if(idx == 1) printf("HCay %d %d %.20g\n", i, id, a);
//if(idx == 2) printf("HCaz %d %d %.20g\n", i, id, a);
			}
		}
	}

	__syncthreads();

	for(int i = 16; i >= 1; i/=2){
#if OldShuffle == 0
		a += __shfl_xor_sync(0xffffffff, a, i, 32);
#else
		a += __shfld_xor(a, i);
#endif
//if(idx == 0) printf("HCbx %d %d %.20g\n", i, id, a);
//if(idx == 1) printf("HCby %d %d %.20g\n", i, id, a);
//if(idx == 2) printf("HCbz %d %d %.20g\n", i, id, a);
	}
	__syncthreads();

	if(blockDim.x > warpSize){
		//reduce across warps
		if(lane == 0){
			a_s[warp] = a;
		}
		__syncthreads();
		//reduce previous warp results in the first warp
		if(warp == 0){
			a = a_s[threadIdx.x];
//if(idx == 0) printf("HCcx %d %.20g %d %d\n", id, a, blockDim.x, warpSize);
//if(idx == 1) printf("HCcy %d %.20g %d %d\n", id, a, blockDim.x, warpSize);
//if(idx == 2) printf("HCcz %d %.20g %d %d\n", id, a, blockDim.x, warpSize);
			for(int i = 16; i >= 1; i/=2){
#if OldShuffle == 0
				a += __shfl_xor_sync(0xffffffff, a, i, 32);
#else
				a += __shfld_xor(a, i);
#endif
//if(idx == 0) printf("HCdx %d %d %.20g\n", i, id, a);
//if(idx == 1) printf("HCdy %d %d %.20g\n", i, id, a);
//if(idx == 2) printf("HCdz %d %d %.20g\n", i, id, a);
			}
		}
	}
	__syncthreads();
	if(threadIdx.x == 0){
		if(idx == 0){
			a_d[blockIdx.x].x = a;
		}
		if(idx == 1){
//if(idx == 1) printf("HCey %d %.20g\n", blockIdx.x, a);
			a_d[blockIdx.x].y = a;
		}
		if(idx == 2){
			a_d[blockIdx.x].z = a;
		}
	}
}
//single trhead block
__global__ void HC32d2_kernel(double3 *a_d, int N){

	int idy = threadIdx.x;
	int idx = blockIdx.x;

	double a = 0.0;

	__shared__ double a_s[32];
	int lane = threadIdx.x % warpSize;
	int warp = threadIdx.x / warpSize;

	if(warp == 0){
		a_s[threadIdx.x] = 0.0;
	}


	if(idy < N){
		if(idx == 0){
			a = a_d[idy].x;
		}
		if(idx == 1){
			a = a_d[idy].y;
		}
		if(idx == 2){
			a = a_d[idy].z;
		}
	}
//if(idx == 0) printf("HC2ax %d %.20g\n", idy, a);
//if(idx == 1) printf("HC2ay %d %.20g\n", idy, a);
//if(idx == 2) printf("HC2az %d %.20g\n", idy, a);

	__syncthreads();

	for(int i = 16; i >= 1; i/=2){
#if OldShuffle == 0
		a += __shfl_xor_sync(0xffffffff, a, i, 32);
#else
		a += __shfld_xor(a, i);
#endif
//if(idx == 0) printf("HC2bx %d %d %.20g\n", i, idy, a);
//if(idx == 1) printf("HC2by %d %d %.20g\n", i, idy, a);
//if(idx == 2) printf("HC2bz %d %d %.20g\n", i, idy, a);
	}
	__syncthreads();

	if(blockDim.x > warpSize){
		//reduce across warps
		if(lane == 0){
			a_s[warp] = a;
		}
		__syncthreads();
		//reduce previous warp results in the first warp
		if(warp == 0){
			a = a_s[threadIdx.x];
//if(idx == 0) printf("HC2cx %d %d %.20g %d %d\n", 0, idy, a, blockDim.x, warpSize);
//if(idx == 1) printf("HC2cy %d %d %.20g %d %d\n", 0, idy, a, blockDim.x, warpSize);
//if(idx == 2) printf("HC2cz %d %d %.20g %d %d\n", 0, idy, a, blockDim.x, warpSize);
			for(int i = 16; i >= 1; i/=2){
#if OldShuffle == 0
				a += __shfl_xor_sync(0xffffffff, a, i, 32);
#else
				a += __shfld_xor(a, i);
#endif
//if(idx == 0) printf("HC2dx %d %d %.20g\n", i, idy, a);
//if(idx == 1) printf("HC2dy %d %d %.20g\n", i, idy, a);
//if(idx == 2) printf("HC2dz %d %d %.20g\n", i, idy, a);
			}
		}
	}
	__syncthreads();
	if(threadIdx.x == 0){
//if(idx == 1) printf("HCey %d %.20g\n", idy, a);
		if(idx == 0){
			a_d[0].x = a;
		}
		if(idx == 1){
			a_d[0].y = a;
		}
		if(idx == 2){
			a_d[0].z = a;
		}
	}
}
__global__ void HC32d3_kernel(double4 *x4_d, double4 *v4_d, double3 *a_d, const double dt, const double dtiMsun, int N, int UseForce){

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id< N){
		double3 a = a_d[0];
//if(id == 0) printf("HC %.20g %.20g %.20g\n", a.x, a.y, a.z);
		x4_d[id].x += a.x * dtiMsun;
		x4_d[id].y += a.y * dtiMsun;
		x4_d[id].z += a.z * dtiMsun;
//printf("HC %d %.20e %.20e %.20e\n", id, x4_d[id].x, a1_s[0].x, dtiMsun);
		if(UseForce & 1){
			double c2 = def_cm * def_cm;
			double4 v4 = v4_d[id];
			double vsq = v4.x * v4.x + v4.y * v4.y + v4.z * v4.z;
			double vcdt = 2.0 * vsq / c2 * dt;
			x4_d[id].x -= __dmul_rn(v4.x, vcdt);
			x4_d[id].y -= __dmul_rn(v4.y, vcdt);
			x4_d[id].z -= __dmul_rn(v4.z, vcdt);
		}
	}
}

//**************************************
//This Kernels performs the Sun-Kick 1/Msun * Sum(p_i)^2 on all the bodies.
//It uses a parallel reduction fomula to calculate the sum in log(N) steps.
//
//It works for the case of multiple warps, but only 1 thread block
//
//Uses shuffle instructions
//Authors: Simon Grimm
//April 2019
//  *****************************************
__global__ void HC32a_kernel(double4 *x4_d, double4 *v4_d, double3 *a_d, const double dt, const double dtiMsun, int N, int UseForce){

	int idy = threadIdx.x;
	int idx = blockIdx.x;

	double a = 0.0;
	double vi;

	for(int i = 0; i < N; i += blockDim.x * gridDim.x){	//gridDim.x is for multiple bock reduce
		if(idy + i < N){
			double m = x4_d[idy + i].w;
			if(idx == 0){
				vi = v4_d[idy + i].x;
			}
			if(idx == 1){
				vi = v4_d[idy + i].y;
			}
			if(idx == 2){
				vi = v4_d[idy + i].z;
			}
			if(m > 0.0){
				a += m * vi;
//if(idx == 0) printf("HCax %d %d %.20g\n", i, idy, a);
//if(idx == 1) printf("HCay %d %d %.20g\n", i, idy, a);
//if(idx == 2) printf("HCaz %d %d %.20g\n", i, idy, a);
			}
		}
	}

	__syncthreads();

	for(int i = 16; i >= 1; i/=2){
#if OldShuffle == 0
		a += __shfl_xor_sync(0xffffffff, a, i, 32);
#else
		a += __shfld_xor(a, i);
#endif
//if(idx == 0) printf("HCbx %d %d %.20g\n", i, idy, a);
//if(idx == 1) printf("HCby %d %d %.20g\n", i, idy, a);
//if(idx == 2) printf("HCbz %d %d %.20g\n", i, idy, a);
	}
	__syncthreads();

	if(blockDim.x > warpSize){
		//reduce across warps
		__shared__ double a_s[32];
		int lane = threadIdx.x % warpSize;
		int warp = threadIdx.x / warpSize;
		if(warp == 0){
			a_s[threadIdx.x] = 0.0;
		}
		__syncthreads(); 

		if(lane == 0){
			a_s[warp] = a;
		}

		__syncthreads();
		//reduce previous warp results in the first warp
		if(warp == 0){
			a = a_s[threadIdx.x];
//if(idx == 0) printf("HCcx %d %d %.20g %d %d\n", 0, idy, a, blockDim.x, warpSize);
//if(idx == 1) printf("HCcy %d %d %.20g %d %d\n", 0, idy, a, blockDim.x, warpSize);
//if(idx == 2) printf("HCcz %d %d %.20g %d %d\n", 0, idy, a, blockDim.x, warpSize);
			for(int i = 16; i >= 1; i/=2){
#if OldShuffle == 0
				a += __shfl_xor_sync(0xffffffff, a, i, 32);
#else
				a += __shfld_xor(a, i);
#endif
//if(idx == 0) printf("HCdx %d %d %.20g\n", i, idy, a);
//if(idx == 1) printf("HCdy %d %d %.20g\n", i, idy, a);
//if(idx == 2) printf("HCdz %d %d %.20g\n", i, idy, a);
			}
			if(lane == 0){
				a_s[0] = a;
			}
		}
		__syncthreads();
		
		a = a_s[0];
//if(idx == 0) printf("HCex %d %.20g\n", idy, a);
//if(idx == 1) printf("HCey %d %.20g\n", idy, a);
//if(idx == 2) printf("HCez %d %.20g\n", idy, a);
	}
	__syncthreads();
	for(int i = 0; i < N; i += blockDim.x){
		if(idy + i < N){
			if(idx == 0) x4_d[idy + i].x += a * dtiMsun;
			if(idx == 1) x4_d[idy + i].y += a * dtiMsun;
			if(idx == 2) x4_d[idy + i].z += a * dtiMsun;
//if(idx == 0 && idy + i == 0) printf("HCx %d %.20e %.20g %.20e\n", idy + i, x4_d[idy + i].x, a, dtiMsun);
//if(idx == 1 && idy + i == 0) printf("HCy %d %.20e %.20g %.20e\n", idy + i, x4_d[idy + i].x, a, dtiMsun);
//if(idx == 2 && idy + i == 0) printf("HCz %d %.20e %.20g %.20e\n", idy + i, x4_d[idy + i].x, a, dtiMsun);
			if(UseForce & 1){
				double c2 = def_cm * def_cm;
				double4 v4 = v4_d[idy];
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
//It works for the case of only 1 single warp
//
//Uses shuffle instructions
//Authors: Simon Grimm
//April 2019
//  *****************************************
__global__ void HC32c_kernel(double4 *x4_d, double4 *v4_d, const double dt, const double dtiMsun, int N, int UseForce){

	int idy = threadIdx.x;

	double3 a = {0.0, 0.0, 0.0};
	double4 v4;

	if(idy < N){
		double m = x4_d[idy].w;
		v4 = v4_d[idy];
		if(m > 0.0){
			a.x += m * v4.x;
			a.y += m * v4.y;
			a.z += m * v4.z;
		}
	}
	__syncthreads();

	for(int i = 16; i >= 1; i/=2){
#if OldShuffle == 0
		a.x += __shfl_xor_sync(0xffffffff, a.x, i, 32);
		a.y += __shfl_xor_sync(0xffffffff, a.y, i, 32);
		a.z += __shfl_xor_sync(0xffffffff, a.z, i, 32);
#else
		a.x += __shfld_xor(a.x, i);
		a.y += __shfld_xor(a.y, i);
		a.z += __shfld_xor(a.z, i);
#endif
//printf("HC %d %d %.20g %.20g %.20g\n", i, idy, a.x, a.y, a.z);
	}		

	__syncthreads();

	if(idy< N){
		x4_d[idy].x += a.x * dtiMsun;
		x4_d[idy].y += a.y * dtiMsun;
		x4_d[idy].z += a.z * dtiMsun;
//printf("HC %d %.20e %.20e %.20e %.20e %.20e\n", idy, x4_d[idy].x, a.x, a.y, a.z, dtiMsun);
		if(UseForce & 1){
			double c2 = def_cm * def_cm;
			double vsq = v4.x * v4.x + v4.y * v4.y + v4.z * v4.z;
			double vcdt = 2.0 * vsq / c2 * dt;
			x4_d[idy].x -= __dmul_rn(v4.x, vcdt);
			x4_d[idy].y -= __dmul_rn(v4.y, vcdt);
			x4_d[idy].z -= __dmul_rn(v4.z, vcdt);
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
		st_s[idy] = index_d[id] / def_MaxIndex;
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
			st_s[idy + Bl] = index_d[id + Bl] / def_MaxIndex;
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
//printf("HCx %d %d %.20e %.20e %.20e\n", E, id, x4_d[id].x, p_s[idy].x, dtiMsun);
//printf("HCy %d %d %.20e %.20e %.20e\n", E, id, x4_d[id].y, p_s[idy].y, dtiMsun);
//printf("HCz %d %d %.20e %.20e %.20e\n", E, id, x4_d[id].z, p_s[idy].z, dtiMsun);
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

#include "define.h"


//This function is needed for the pseudovelocity conversion
//It is the right hand side of equation 32 from Saha & Tremaine 1994
//vv is pseudovelocity
__device__ void FPseudoV(double mu, double x, double y, double z, double vvx, double vvy, double vvz, volatile double &fx, volatile double &fy, volatile double &fz){

	double c2 = def_cm * def_cm;

	double vsq = vvx * vvx + vvy * vvy + vvz * vvz;
	double rsq = x * x + y * y + z * z;
	double r = sqrt(rsq);

	double t = 1.0 - 1.0/c2 * (vsq * 0.5 + 3.0 * mu / r);

	fx = vvx * t;
	fy = vvy * t;
	fz = vvz * t;
}


//This function converts pseudovelocities to true velocities
//See Saha & Tremaine 1994
__global__ void convertPseudovToV(double4 *x4_d, double4 *v4_d, double Msun, int N){

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(id < N){

		double c2 = def_cm * def_cm;

		double mu = def_ksq * (Msun + x4_d[id].w);
		//use here Jacoby masses from Saha Tremaine


		double vsq = v4_d[id].x * v4_d[id].x + v4_d[id].y * v4_d[id].y + v4_d[id].z * v4_d[id].z;
		double rsq = x4_d[id].x * x4_d[id].x + x4_d[id].y * x4_d[id].y + x4_d[id].z * x4_d[id].z;
		double r = sqrt(rsq);

		double t = 1.0 - 1.0/c2 * (vsq * 0.5 + 3.0 * mu / r);

//printf("%d %.20g %.20g %.20g | %.20g %.20g %.20g\n", i, vx[i], vy[i], vz[i], vx[i] * t, vy[i] * t, vz[i] * t);

		v4_d[id].x *= t;
		v4_d[id].y *= t;
		v4_d[id].z *= t;

	}
}

//This function converts velocities to pseudovelocities
//See Saha & Tremaine 1994
__global__ void convertVToPseidov(double4 *x4_d, double4 *v4_d, int *ErrorFlag_d, double Msun, int N){

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(id < N){

		double mu = def_ksq * (Msun + x4_d[id].w);
		//use here Jacoby masses from Saha Tremaine

		double xi = x4_d[id].x;
		double yi = x4_d[id].y;
		double zi = x4_d[id].z;

		double vxi = v4_d[id].x;
		double vyi = v4_d[id].y;
		double vzi = v4_d[id].z;

		//first guess of pseudovelocity
		double vvx0 = vxi;
		double vvy0 = vyi;
		double vvz0 = vzi;

		//second guess of pseudovelocity
		double vvx1 = vvx0 * 0.01;
		double vvy1 = vvy0 * 0.01;
		double vvz1 = vvz0 * 0.01;

		volatile double fx0;
		volatile double fy0;
		volatile double fz0;

		FPseudoV(mu, xi, yi, zi, vvx0, vvy0, vvz0, fx0, fy0, fz0);
		fx0 -= vxi;
		fy0 -= vyi;
		fz0 -= vzi;

		volatile double fx1;
		volatile double fy1;
		volatile double fz1;
		//without volatile, f*1 is not updated and the loop does not terminate

		FPseudoV(mu, xi, yi, zi, vvx1, vvy1, vvz1, fx1, fy1, fz1);
		fx1 -= vxi;
		fy1 -= vyi;
		fz1 -= vzi;

		//Newton Method
		int k;
		for(k = 0; k < 30; ++k){

			double tx = vvx1 - (vvx1 - vvx0) / (fx1 - fx0) * fx1;
			double ty = vvy1 - (vvy1 - vvy0) / (fy1 - fy0) * fy1;
			double tz = vvz1 - (vvz1 - vvz0) / (fz1 - fz0) * fz1;

			int Stop = 0;
			if(fabs(fx1 - fx0) < 1.0e-18){
				tx = vvx1;
				++Stop;
			}
			if(fabs(fy1 - fy0) < 1.0e-18){
				ty = vvy1;
				++Stop;
			}
			if(fabs(fz1 - fz0) < 1.0e-18){
				tz = vvz1;
				++Stop;
			}

			vvx0 = vvx1;
			vvy0 = vvy1;
			vvz0 = vvz1;

			fx0 = fx1;
			fy0 = fy1;
			fz0 = fz1;

			vvx1 = tx;
			vvy1 = ty;
			vvz1 = tz;

			if(Stop == 3){
				break;
			}
			FPseudoV(mu, xi, yi, zi, vvx1, vvy1, vvz1, fx1, fy1, fz1);
			fx1 -= vxi;
			fy1 -= vyi;
			fz1 -= vzi;
//if(k > 4) printf("%d %d %.20g %.20g %.20g | %.20g %.20g %.20g | %g %g %g\n", id, k, vxi, vyi, vzi, vvx1, vvy1, vvz1, fx1, fy1, fz1);

		}
		__syncthreads();
		if(k >= 29){
			ErrorFlag_d[0] = 1;
			printf("Warning: Newton Method in 'convertVToPseidov' did not convert. %d\n", id);
		}

		v4_d[id].x = vvx1;
		v4_d[id].y = vvy1;
		v4_d[id].z = vvz1;
	}
}

__global__ void convertPseudovToVM(double4 *x4_d, double4 *v4_d, int *index_d, double2 *Msun_d, int NT){

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	
	if(id < NT){
		int st = index_d[id] / def_MaxIndex;

		double c2 = def_cm * def_cm;

		double mu = def_ksq * (Msun_d[st].x + x4_d[id].w);
		//use here Jacoby masses from Saha Tremaine


		double vsq = v4_d[id].x * v4_d[id].x + v4_d[id].y * v4_d[id].y + v4_d[id].z * v4_d[id].z;
		double rsq = x4_d[id].x * x4_d[id].x + x4_d[id].y * x4_d[id].y + x4_d[id].z * x4_d[id].z;
		double r = sqrt(rsq);

		double t = 1.0 - 1.0/c2 * (vsq * 0.5 + 3.0 * mu / r);

//printf("%d %.20g %.20g %.20g | %.20g %.20g %.20g\n", i, vx[i], vy[i], vz[i], vx[i] * t, vy[i] * t, vz[i] * t);

		v4_d[id].x *= t;
		v4_d[id].y *= t;
		v4_d[id].z *= t;

	}
}

//This function converts velocities to pseudovelocities
//See Saha & Tremaine 1994
__global__ void convertVToPseidovM(double4 *x4_d, double4 *v4_d, int *index_d, int *ErrorFlag_d, double2 *Msun_d, int NT){

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(id < NT){

		int st = index_d[id] / def_MaxIndex;

		double mu = def_ksq * (Msun_d[st].x + x4_d[id].w);
		//use here Jacoby masses from Saha Tremaine

		double xi = x4_d[id].x;
		double yi = x4_d[id].y;
		double zi = x4_d[id].z;

		double vxi = v4_d[id].x;
		double vyi = v4_d[id].y;
		double vzi = v4_d[id].z;

		//first guess of pseudovelocity
		double vvx0 = vxi;
		double vvy0 = vyi;
		double vvz0 = vzi;

		//second guess of pseudovelocity
		double vvx1 = vvx0 * 0.01;
		double vvy1 = vvy0 * 0.01;
		double vvz1 = vvz0 * 0.01;

		volatile double fx0;
		volatile double fy0;
		volatile double fz0;

		FPseudoV(mu, xi, yi, zi, vvx0, vvy0, vvz0, fx0, fy0, fz0);
		fx0 -= vxi;
		fy0 -= vyi;
		fz0 -= vzi;

		volatile double fx1;
		volatile double fy1;
		volatile double fz1;

		FPseudoV(mu, xi, yi, zi, vvx1, vvy1, vvz1, fx1, fy1, fz1);
		fx1 -= vxi;
		fy1 -= vyi;
		fz1 -= vzi;

		int k;
		for(k = 0; k < 30; ++k){

			double tx = vvx1 - (vvx1 - vvx0) / (fx1 - fx0) * fx1;
			double ty = vvy1 - (vvy1 - vvy0) / (fy1 - fy0) * fy1;
			double tz = vvz1 - (vvz1 - vvz0) / (fz1 - fz0) * fz1;

			int Stop = 0;
			if(fabs(fx1 - fx0) < 1.0e-18){
				tx = vvx1;
				++Stop;
			}
			if(fabs(fy1 - fy0) < 1.0e-18){
				ty = vvy1;
				++Stop;
			}
			if(fabs(fz1 - fz0) < 1.0e-18){
				tz = vvz1;
				++Stop;
			}

			vvx0 = vvx1;
			vvy0 = vvy1;
			vvz0 = vvz1;

			fx0 = fx1;
			fy0 = fy1;
			fz0 = fz1;

			vvx1 = tx;
			vvy1 = ty;
			vvz1 = tz;

			if(Stop == 3) break;

			FPseudoV(mu, xi, yi, zi, vvx1, vvy1, vvz1, fx1, fy1, fz1);
			fx1 -= vxi;
			fy1 -= vyi;
			fz1 -= vzi;

//printf("%d %d %.20g %.20g %.20g | %.20g %.20g %.20g | %g %g %g\n", i, k, vx[i], vy[i], vz[i], vvx1, vvy1, vvz1, fx1, fy1, fz1);

		}
		__syncthreads();
		if(k >= 29){
			ErrorFlag_d[0] = 1;
			printf("Warning: Newton Method in 'convertVToPseidov' did not convert. %d\n", id);
		}

		v4_d[id].x = vvx1;
		v4_d[id].y = vvy1;
		v4_d[id].z = vvz1;
	}
}

//**************************************
//This Kernels performs the Sun-Kick 1/Msun * Sum(p_i)^2 on all the bodies.
//It uses a parallel reduction formula to calculate the sum in log(N) steps.
//
//It works for the case of multiple blocks
//must be followed by HC32d2 and HC32d3
//
//Uses shuffle instructions
//Authors: Simon Grimm
//April 2019
//  *****************************************
__global__ void HC32d1_kernel(double4 *x4_d, double4 *v4_d, double3 *a_d, const int N){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	int idx = blockIdx.y;

	double a = 0.0;
	double vi;

	extern __shared__ double HCd1_s[];
	double *a_s = HCd1_s;

	int lane = threadIdx.x % warpSize;
	int warp = threadIdx.x / warpSize;

	if(warp == 0){
		a_s[threadIdx.x] = 0.0;
	}


	for(int i = 0; i < N; i += blockDim.x * gridDim.x){	//gridDim.x is for multiple block reduce
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

	for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
		a += __shfl_xor_sync(0xffffffff, a, i, warpSize);
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
//if(idx == 0) printf("HCcx %d %.20g %d %d\n", id, a, int(blockDim.x), warpSize);
//if(idx == 1) printf("HCcy %d %.20g %d %d\n", id, a, blockDim.x, warpSize);
//if(idx == 2) printf("HCcz %d %.20g %d %d\n", id, a, blockDim.x, warpSize);
			for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
				a += __shfl_xor_sync(0xffffffff, a, i, warpSize);
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
//printf("HCex %d %.20g\n", int(blockIdx.x), a);
			a_d[blockIdx.x].x = a;
		}
		if(idx == 1){
//printf("HCey %d %.20g\n", blockIdx.x, a);
			a_d[blockIdx.x].y = a;
		}
		if(idx == 2){
//printf("HCez %d %.20g\n", blockIdx.x, a);
			a_d[blockIdx.x].z = a;
		}
	}
}

//**************************************
//This kernel reads the result from the multiple thread block kernel HC32d1
//and performs the last summation step in
// --a single thread block --
//
//must be followed by HC32d3
//
//Uses shuffle instructions
//Authors: Simon Grimm
//April 2019
//  *****************************************
__global__ void HC32d2_kernel(double3 *a_d, const int N){

	int idy = threadIdx.x;
	int idx = blockIdx.x;

	double a = 0.0;

	extern __shared__ double HCd2_s[];
	double *a_s = HCd2_s;

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

	for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
		a += __shfl_xor_sync(0xffffffff, a, i, warpSize);
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
//if(idx == 0) printf("HC2cx %d %d %.20g %d %d\n", 0, idy, a, int(blockDim.x), warpSize);
//if(idx == 1) printf("HC2cy %d %d %.20g %d %d\n", 0, idy, a, int(blockDim.x), warpSize);
//if(idx == 2) printf("HC2cz %d %d %.20g %d %d\n", 0, idy, a, int(blockDim.x), warpSize);
			for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
				a += __shfl_xor_sync(0xffffffff, a, i,  warpSize);
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
		if(idx == 0){
//printf("HC2ex %d %.20g\n", idy, a);
			a_d[0].x = a;
		}
		if(idx == 1){
//printf("HC2ey %d %.20g\n", idy, a);
			a_d[0].y = a;
		}
		if(idx == 2){
//printf("HC2ez %d %.20g\n", idy, a);
			a_d[0].z = a;
		}
	}
}

//**************************************
//This kernel distributes the result from the multiple thread block kernel
//HC32d1 and HC32d2.
// --a single thread block --
//
//
//Authors: Simon Grimm
//April 2019
//  *****************************************
__global__ void HC32d3_kernel(double4 *x4_d, double4 *v4_d, double3 *a_d, const double dt, const double dtiMsun, const int N, const int UseGR){

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < N && x4_d[id].w >= 0.0){
		double3 a = a_d[0];
//if(id == 0) printf("HC A %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g\n", id, x4_d[id].w, a.x, a.y, a.z, x4_d[id].x, x4_d[id].y, x4_d[id].z);
		x4_d[id].x += a.x * dtiMsun;
		x4_d[id].y += a.y * dtiMsun;
		x4_d[id].z += a.z * dtiMsun;
//if(id == 0) printf("HC B %d %.20g %.20g %.20g %.20g %.20g %.20g\n", id, a.x, a.y, a.z, x4_d[id].x, x4_d[id].y, x4_d[id].z);
		if(UseGR == 1){
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
__global__ void HC32a_kernel(double4 *x4_d, double4 *v4_d, const double dt, const double dtiMsun, const int N, const int UseGR){

	int idy = threadIdx.x;
	int idx = blockIdx.x;

	double a = 0.0;
	double vi;


	for(int i = 0; i < N; i += blockDim.x * gridDim.x){	//gridDim.x is for multiple block reduce
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

	for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
		a += __shfl_xor_sync(0xffffffff, a, i, warpSize);
#else
		a += __shfld_xor(a, i);
#endif
//if(i >= 16 && idx == 0) printf("HCbx %d %d %.20g\n", i, idy, a);
//if(idx == 1) printf("HCby %d %d %.20g\n", i, idy, a);
//if(idx == 2) printf("HCbz %d %d %.20g\n", i, idy, a);
	}
	__syncthreads();

	if(blockDim.x > warpSize){
		//reduce across warps
		extern __shared__ double HCa_s[];
 		double *a_s = HCa_s;

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
//if(idx == 0) printf("HCcx %d %d %.20g %d %d\n", 0, idy, a, int(blockDim.x), warpSize);
//if(idx == 1) printf("HCcy %d %d %.20g %d %d\n", 0, idy, a, blockDim.x, warpSize);
//if(idx == 2) printf("HCcz %d %d %.20g %d %d\n", 0, idy, a, blockDim.x, warpSize);
			for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
				a += __shfl_xor_sync(0xffffffff, a, i, warpSize);
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
		if(idy + i < N && x4_d[idy + i].w >= 0.0){
			if(idx == 0) x4_d[idy + i].x += a * dtiMsun;
			if(idx == 1) x4_d[idy + i].y += a * dtiMsun;
			if(idx == 2) x4_d[idy + i].z += a * dtiMsun;
//if(idx == 0 && idy + i == 0) printf("HCx %d %.20e %.20g %.20e\n", idy + i, x4_d[idy + i].x, a, dtiMsun);
//if(idx == 1 && idy + i == 0) printf("HCy %d %.20e %.20g %.20e\n", idy + i, x4_d[idy + i].x, a, dtiMsun);
//if(idx == 2 && idy + i == 0) printf("HCz %d %.20e %.20g %.20e\n", idy + i, x4_d[idy + i].x, a, dtiMsun);
			if(UseGR == 1){
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

__global__ void HC32aM_kernel(double4 *x4_d, double4 *v4_d, double *dt_d, double2 *Msun_d, int *N_d, int *NBS_d, const int Nst, const double Ct, const int UseGR, const int si){

	int st = blockIdx.x;
	int idy = threadIdx.x;				//must be in x dimension in order to be in the same warp

	if(st < Nst){

		double3 a = {0.0, 0.0, 0.0};
		double m;
		double4 v4i;

		int Ni = N_d[st];
		int NBS = NBS_d[st];

		double dt = dt_d[st] * Ct;
		double dtiMsun = dt / Msun_d[st].x;

		for(int i = 0; i < Ni; i += blockDim.x){	
			if(idy + i < Ni){
				m = x4_d[NBS + idy + i].w;
				v4i = v4_d[NBS + idy + i];
				if(m > 0.0){
					a.x += m * v4i.x;
					a.y += m * v4i.y;
					a.z += m * v4i.z;
				}
			}
		}
//printf("%d %d %.20g %.20g %.20g\n", st, idy, a.x, a.y, a.z);
		__syncthreads();

		for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
			a.x += __shfl_xor_sync(0xffffffff, a.x, i, warpSize);
			a.y += __shfl_xor_sync(0xffffffff, a.y, i, warpSize);
			a.z += __shfl_xor_sync(0xffffffff, a.z, i, warpSize);
#else
			a.x += __shfld_xor(a.x, i);
			a.y += __shfld_xor(a.y, i);
			a.z += __shfld_xor(a.z, i);
#endif
		}
		__syncthreads();

		if(blockDim.x > warpSize){
			//reduce across warps
			extern __shared__ double3 HCaM_s[];
			double3 *a_s = HCaM_s;

			int lane = threadIdx.x % warpSize;
			int warp = threadIdx.x / warpSize;
			if(warp == 0){
				a_s[threadIdx.x].x = 0.0;
				a_s[threadIdx.x].y = 0.0;
				a_s[threadIdx.x].z = 0.0;
			}
			__syncthreads(); 

			if(lane == 0){
				a_s[warp] = a;
			}

			__syncthreads();
			//reduce previous warp results in the first warp
			if(warp == 0){
				a = a_s[threadIdx.x];
				for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
					a.x += __shfl_xor_sync(0xffffffff, a.x, i, warpSize);
					a.y += __shfl_xor_sync(0xffffffff, a.y, i, warpSize);
					a.z += __shfl_xor_sync(0xffffffff, a.z, i, warpSize);
#else
					a.x += __shfld_xor(a.x, i);
					a.y += __shfld_xor(a.y, i);
					a.z += __shfld_xor(a.z, i);
#endif
				}
				if(lane == 0){
					a_s[0] = a;
				}
			}
			__syncthreads();
			
			a = a_s[0];
		}
		__syncthreads();
		for(int i = 0; i < Ni; i += blockDim.x){
			if(idy + i < Ni && x4_d[NBS + idy + i].w >= 0.0){
//printf("HC A %d %d %.20g %.20g %.20g %.20g %.20g %.20g %.29g %.20g %.20g %.20g %.20g\n", st, idy + i, a.x, a.y, a.z, x4_d[NBS + idy + i].x, x4_d[NBS + idy + i].y, x4_d[NBS + idy + i].z, v4_d[NBS + idy + i].x, v4_d[NBS + idy + i].y, v4_d[NBS + idy + 1].z, x4_d[NBS + idy + i].w, dtiMsun);
				if(si == 1){
					x4_d[NBS + idy + i].x += a.x * dtiMsun;
					x4_d[NBS + idy + i].y += a.y * dtiMsun;
					x4_d[NBS + idy + i].z += a.z * dtiMsun;

					if(UseGR == 1){
						double c2 = def_cm * def_cm;
						double4 v4 = v4_d[NBS + idy + i];
						double vsq = v4.x * v4.x + v4.y * v4.y + v4.z * v4.z;
						double vcdt = 2.0 * vsq / c2 * dt;
						x4_d[NBS + idy + i].x -= __dmul_rn(v4.x, vcdt);
						x4_d[NBS + idy + i].y -= __dmul_rn(v4.y, vcdt);
						x4_d[NBS + idy + i].z -= __dmul_rn(v4.z, vcdt);
					}
				}
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
__global__ void HC32c_kernel(double4 *x4_d, double4 *v4_d, const double dt, const double dtiMsun, const int N, const int UseGR){

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
//printf("HC1 %d %.20g %.20g %.20g\n", idy, a.x, a.y, a.z);
	}
	__syncthreads();

	for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
		a.x += __shfl_xor_sync(0xffffffff, a.x, i, warpSize);
		a.y += __shfl_xor_sync(0xffffffff, a.y, i, warpSize);
		a.z += __shfl_xor_sync(0xffffffff, a.z, i, warpSize);
#else
		a.x += __shfld_xor(a.x, i);
		a.y += __shfld_xor(a.y, i);
		a.z += __shfld_xor(a.z, i);
#endif
//if(i >= 16) printf("HCa %d %d %.20g %.20g %.20g\n", i, idy, a.x, a.y, a.z);
	}		

	__syncthreads();

	if(idy < N && x4_d[idy].w >= 0.0){
//printf("HC A %d %.20g %.20g %.20g %.20g %.20g %.20g %.29g %.20g %.20g %.20g %.20g\n", idy, a.x, a.y, a.z, x4_d[idy].x, x4_d[idy].y, x4_d[idy].z, v4_d[idy].x, v4_d[idy].y, v4_d[idy].z, x4_d[idy].w, dtiMsun);
		x4_d[idy].x += a.x * dtiMsun;
		x4_d[idy].y += a.y * dtiMsun;
		x4_d[idy].z += a.z * dtiMsun;
//if(idy == 12) printf("HC B %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g\n", idy, a.x, a.y, a.z, x4_d[idy].x, x4_d[idy].y, x4_d[idy].z, dtiMsun);
//printf("HC %d %.20e %.20e %.20e %.20e %.20e %.20e\n", idy, x4_d[idy].w, x4_d[idy].x, a.x, a.y, a.z, dtiMsun);
		if(UseGR == 1){
			double c2 = def_cm * def_cm;
			double vsq = v4.x * v4.x + v4.y * v4.y + v4.z * v4.z;
			double vcdt = 2.0 * vsq / c2 * dt;
			x4_d[idy].x -= __dmul_rn(v4.x, vcdt);
			x4_d[idy].y -= __dmul_rn(v4.y, vcdt);
			x4_d[idy].z -= __dmul_rn(v4.z, vcdt);
		}
	}
}


//First call f = 1;
//Second call f = -1;
__host__ void Data::HCCall(const double Ct, const int f){

	if(P.UseGR == 1 && f == 1){
		convertVToPseidov <<< (N_h[0] + Nsmall_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, ErrorFlag_m, Msun_h[0].x, N_h[0] + Nsmall_h[0]);
	}
	//HC
	if(N_h[0] + Nsmall_h[0] <= WarpSize){
		HC32c_kernel <<< 1, WarpSize >>> (x4_d, v4_d, dt_h[0] * Ct, dt_h[0] / Msun_h[0].x * Ct, N_h[0] + Nsmall_h[0], P.UseGR);
	}
	else if(N_h[0] + Nsmall_h[0] <= 512){
		int nn = (N_h[0] + Nsmall_h[0] + WarpSize - 1) / WarpSize;
		HC32a_kernel <<< 3, nn * WarpSize, WarpSize * sizeof(double)  >>> (x4_d, v4_d, dt_h[0] * Ct, dt_h[0] / Msun_h[0].x * Ct, N_h[0] + Nsmall_h[0], P.UseGR);
	}
	else{
		int nct = 512;
		int ncb = min((N_h[0] + Nsmall_h[0] + nct - 1) / nct, 1024);
		HC32d1_kernel <<< dim3(ncb, 3, 1), dim3(nct, 1, 1), WarpSize * sizeof(double) >>> (x4_d, v4_d, a_d, N_h[0] + Nsmall_h[0]);
		HC32d2_kernel <<< 3, ((ncb + WarpSize - 1) / WarpSize) * WarpSize, WarpSize * sizeof(double)  >>> (a_d, ncb);
		HC32d3_kernel <<<(N_h[0] + Nsmall_h[0] + FTX - 1)/FTX, FTX >>> (x4_d, v4_d, a_d, dt_h[0] * Ct, dt_h[0] / Msun_h[0].x * Ct, N_h[0] + Nsmall_h[0], P.UseGR);
	}

	if(P.UseGR == 1 && f == -1){
		convertPseudovToV <<< (N_h[0] + Nsmall_h[0] + 127) / 128, 128 >>> (x4_d, v4_d, Msun_h[0].x, N_h[0] + Nsmall_h[0]);
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
__global__ void HCM2_kernel(double4 *x4_d, double4 *v4_d, double *dt_d, double2 *Msun_d, int *index_d, const int NT, const double Ct, int *Nencpairs_d, int *Nencpairs2_d, int *Nenc_d, const int Nst, const int UseGR, const int Nstart){

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
		if(m > 0.0){
			p_s[idy].x = m * v4_d[id].x;
			p_s[idy].y = m * v4_d[id].y;
			p_s[idy].z = m * v4_d[id].z;
		}
		else{
			p_s[idy].x = 0.0;
			p_s[idy].y = 0.0;
			p_s[idy].z = 0.0;
		}
//printf("HC %d %d %g %g %g %g\n", id, idy, p_s[idy].x, p_s[idy].y, p_s[idy].z, m);
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

	if(id < NT + Nstart && id >= Nstart && idy >= Nmax && idy < Bl - Nmax / 2 && x4_d[id].w >= 0.0){
		x4_d[id].x += p_s[idy].x * dtiMsun;
		x4_d[id].y += p_s[idy].y * dtiMsun;
		x4_d[id].z += p_s[idy].z * dtiMsun;
//printf("HCx %d %d %.20e %.20e %.20e\n", E, id, x4_d[id].x, p_s[idy].x, dtiMsun);
//printf("HCy %d %d %.20e %.20e %.20e\n", E, id, x4_d[id].y, p_s[idy].y, dtiMsun);
//printf("HCz %d %d %.20e %.20e %.20e\n", E, id, x4_d[id].z, p_s[idy].z, dtiMsun);
		if(UseGR == 1){// GR part depending on velocity only (see Saha & Tremaine 1994)
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

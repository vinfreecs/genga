#include "Kick3.h"

template __device__ void  acc < 0 > (double3 &, double3 &, double4 &, double4 &, double , double, double, double, int *, int *, int2 *, int, int, int, double &, const int);
template __device__ void  acc < 1 > (double3 &, double3 &, double4 &, double4 &, double , double, double, double, int *, int *, int2 *, int, int, int, double &, const int);
template __device__ void  acc < 2 > (double3 &, double3 &, double4 &, double4 &, double , double, double, double, int *, int *, int2 *, int, int, int, double &, const int);
template __device__ void  acc < 20 > (double3 &, double3 &, double4 &, double4 &, double , double, double, double, int *, int *, int2 *, int, int, int, double &, const int);
template __device__ void  acc < 21 > (double3 &, double3 &, double4 &, double4 &, double , double, double, double, int *, int *, int2 *, int, int, int, double &, const int);
template __device__ void  acc < 22 > (double3 &, double3 &, double4 &, double4 &, double , double, double, double, int *, int *, int2 *, int, int, int, double &, const int);
template __device__ void  acc < 30 > (double3 &, double3 &, double4 &, double4 &, double , double, double, double, int *, int *, int2 *, int, int, int, double &, const int);
template __device__ void  acc < 31 > (double3 &, double3 &, double4 &, double4 &, double , double, double, double, int *, int *, int2 *, int, int, int, double &, const int);
template __device__ void  acc < 32 > (double3 &, double3 &, double4 &, double4 &, double , double, double, double, int *, int *, int2 *, int, int, int, double &, const int);


template __global__ void kick32A_kernel < 16 > (double4 *, double4 *, double3 *, double *, const double, int2 *, double *, int, int);
template __global__ void kick32A_kernel < 32 > (double4 *, double4 *, double3 *, double *, const double, int2 *, double *, int, int);
template __global__ void kickAsmall_kernel < 128 > (double4 *, double4 *, double3 *, double *, const double, int2 *, double *, int, double4 *, double4 *, int2 *, int, double3 *, int, const int);


template __global__ void kick16_kernel < 16, 40, 0 > (double4 *, double4 *, double3 *, double *, double *, const double, int *, int2 *, int2 *, double *, int);
template __global__ void kick32_kernel < 32, 64, 0 > (double4 *, double4 *, double3 *, double *, double *, const double, int *, int2 *, int2 *, double *, int);
template __global__ void kick32_kernel < 64, 64, 0 > (double4 *, double4 *, double3 *, double *, double *, const double, int *, int2 *, int2 *, double *, int);
template __global__ void kick128_kernel < 128, 0 > (double4 *, double4 *, double3 *, double *, double *, const double, int *, int2 *, int2 *, double *, int, int);
template __global__ void kick256_kernel < 128, 256,  0 >(double4 *, double4 *, double3 *, double *, double *, const double, int, int *, int2 *, int2 *, double *, int, int);
template __global__ void kick4_kernel < 256, 512,  0 > (double4 *, double4 *, double3 *, double *, double *, const double, int, int *, int2 *, int2 *, double *, int, int);
template __global__ void kick4_kernel < 256, 1024,  0 > (double4 *, double4 *, double3 *, double *, double *, const double, int, int *, int2 *, int2 *, double *, int, int);
template __global__ void kick4_kernel < 256, 2048,  0 > (double4 *, double4 *, double3 *, double *, double *, const double, int, int *, int2 *, int2 *, double *, int, int);
template __global__ void kicksmall_kernel < 128, 0 > (double4 *, double4 *, double3 *, double *, double *, const double, int, int *, int2 *, int2 *, double4 *, double4 *, double3 *, int, double *, int *, int2 *, int2 *, int, const int);
template __global__ void KickM2_kernel < KM_Bl, KM_Bl2, NmaxM, 0, 16 > (double4 *, double4 *, double3 *, double *, double *, int *, int2 *, double, int *, int, double *);
template __global__ void KickM2_kernel < KM_Bl, KM_Bl2, NmaxM, 1, 16 > (double4 *, double4 *, double3 *, double *, double *, int *, int2 *, double, int *, int, double *);
template __global__ void KickM2_kernel < KM_Bl, KM_Bl2, NmaxM, 2, 16 > (double4 *, double4 *, double3 *, double *, double *, int *, int2 *, double, int *, int, double *);

template __global__ void kick16_kernel < 16, 40, 1 > (double4 *, double4 *, double3 *, double *, double *, const double, int *, int2 *, int2 *, double *, int);
template __global__ void kick32_kernel < 32, 64, 1 > (double4 *, double4 *, double3 *, double *, double *, const double, int *, int2 *, int2 *, double *, int);
template __global__ void kick32_kernel < 64, 64, 1 > (double4 *, double4 *, double3 *, double *, double *, const double, int *, int2 *, int2 *, double *, int);
template __global__ void kick128_kernel < 128, 1 > (double4 *, double4 *, double3 *, double *, double *, const double, int *, int2 *, int2 *, double *, int, int);
template __global__ void kick256_kernel < 128, 256,  1 >(double4 *, double4 *, double3 *, double *, double *, const double, int, int *, int2 *, int2 *, double *, int, int);
template __global__ void kick4_kernel < 256, 512,  1 > (double4 *, double4 *, double3 *, double *, double *, const double, int, int *, int2 *, int2 *, double *, int, int);
template __global__ void kick4_kernel < 256, 1024,  1 > (double4 *, double4 *, double3 *, double *, double *, const double, int, int *, int2 *, int2 *, double *, int, int);
template __global__ void kick4_kernel < 256, 2048,  1 > (double4 *, double4 *, double3 *, double *, double *, const double, int, int *, int2 *, int2 *, double *, int, int);
template __global__ void kicksmall_kernel < 128, 1 > (double4 *, double4 *, double3 *, double *, double *, const double, int, int *, int2 *, int2 *, double4 *, double4 *, double3 *, int, double *, int *, int2 *, int2 *, int, const int);

template __global__ void kick16_kernel < 16, 40, 2 > (double4 *, double4 *, double3 *, double *, double *, const double, int *, int2 *, int2 *, double *, int);
template __global__ void kick32_kernel < 32, 64, 2 > (double4 *, double4 *, double3 *, double *, double *, const double, int *, int2 *, int2 *, double *, int);
template __global__ void kick32_kernel < 64, 64, 2 > (double4 *, double4 *, double3 *, double *, double *, const double, int *, int2 *, int2 *, double *, int);
template __global__ void kick128_kernel < 128, 2 > (double4 *, double4 *, double3 *, double *, double *, const double, int *, int2 *, int2 *, double *, int, int);
template __global__ void kick256_kernel < 128, 256,  2 >(double4 *, double4 *, double3 *, double *, double *, const double, int, int *, int2 *, int2 *, double *, int, int);
template __global__ void kick4_kernel < 256, 512,  2 > (double4 *, double4 *, double3 *, double *, double *, const double, int, int *, int2 *, int2 *, double *, int, int);
template __global__ void kick4_kernel < 256, 1024,  2 > (double4 *, double4 *, double3 *, double *, double *, const double, int, int *, int2 *, int2 *, double *, int, int);
template __global__ void kick4_kernel < 256, 2048,  2 > (double4 *, double4 *, double3 *, double *, double *, const double, int, int *, int2 *, int2 *, double *, int, int);
template __global__ void kicksmall_kernel < 128, 2 > (double4 *, double4 *, double3 *, double *, double *, const double, int, int *, int2 *, int2 *, double4 *, double4 *, double3 *, int, double *, int *, int2 *, int2 *, int, const int);



// **************************************
// This function computes the term a = mi/rij^3 * Kij
// ****************************************
__device__ void  accA(double3 &ac, double4 &x4i, double4 &x4j, double rcritvi, double rcritvj, int j, int i){
	if( i != j && x4i.w >= 0.0 && x4j.w > 0.0){
		double rsq, ir, ir3, s;
		double3 r3ij;
		double rcritv, rcritv2;
		double y, yy;

		r3ij.x = x4j.x - x4i.x;
		r3ij.y = x4j.y - x4i.y;
		r3ij.z = x4j.z - x4i.z;

		rsq = r3ij.x*r3ij.x + r3ij.y*r3ij.y + r3ij.z*r3ij.z;
		rcritv = fmax(rcritvi, rcritvj);

		rcritv2 = rcritv * rcritv;

		ir = 1.0/sqrt(rsq);
		ir3 = ir*ir*ir;

		if(rsq >= 1.0 * rcritv2){
			s = x4j.w * ir3;
		}
		else{
			if(rsq <= 0.01 * rcritv2){
				s = 0.0;
			}
			else{
				y = (rsq * ir - 0.1 * rcritv)/(0.9*rcritv);
				yy = y * y;
				s = ir3 * yy / (2.0*yy - 2.0*y + 1.0) * x4j.w;
			}
		}
		ac.x += __dmul_rn(r3ij.x, s);
		ac.y += __dmul_rn(r3ij.y, s);
		ac.z += __dmul_rn(r3ij.z, s);
	}
}

// **************************************
//This kernel performs the first kick of the time step, in the case of no close encounters.
//It reuses the values from the seccond kick in the previous time step.
// ****************************************
__global__ void kick32B_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	if(x4_d[id].w >= 0.0){
		v4_d[id].x += acck_d[id].x;
		v4_d[id].y += acck_d[id].y;
		v4_d[id].z += acck_d[id].z;
	}
}

__global__ void kickBsmall_kernel(double4 *x4_d, double4 *v4_d, double3 *acck_d, double4 *x4small_d, double4 *v4small_d, double3 *accksmall_d, int N, int Nsmall){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;
	if(id < N){
		if(x4_d[id].w >= 0.0){
			v4_d[id].x += acck_d[id].x;
			v4_d[id].y += acck_d[id].y;
			v4_d[id].z += acck_d[id].z;
		}
	}
	else if(id < Nsmall + N){
		if(x4small_d[id - N].w >= 0.0){
			v4small_d[id - N].x += accksmall_d[id - N].x;
			v4small_d[id - N].y += accksmall_d[id - N].y;
			v4small_d[id - N].z += accksmall_d[id - N].z;
		}
	}
}

__global__ void kickMB_kernel(double4 *v4_d, double3 *acck_d, double *test_d, int NT){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;	
	if(id < NT){
		v4_d[id].x += acck_d[id].x;
		v4_d[id].y += acck_d[id].y;
		v4_d[id].z += acck_d[id].z;
	}
}

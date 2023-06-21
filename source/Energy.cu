#include "Orbit2.h"

// ****************************************
//This function computes the terms m/r^3 between all pairs of bodies.
//
//Authors: Simon Grimm, Joachim Stadel
//March 2014

// ****************************************
__device__ double  PE(double4 x4i, double4 x4j, int i, int j){

	double3 r;
	double rsq, ir;
	double a = 0.0;
	if( i != j){
		r.x = x4j.x - x4i.x;
		r.y = x4j.y - x4i.y;
		r.z = x4j.z - x4i.z;
		rsq = r.x*r.x + r.y*r.y + r.z*r.z;

		if(rsq > 0.0){
			ir = 1.0/sqrt(rsq);
			a = -x4j.w * ir;
		}
	}
	return a;
}


// **************************************
//This function computes the potential energy from the Sun and body i.
//
//Authors: Simon Grimm, Joachim Stadel
//March 2014
//
//***************************************
__device__ inline double PESun(double4 x4i, double ksqMsun){

	double rsq, ir;
	double a = 0.0;

	rsq = x4i.x * x4i.x + x4i.y * x4i.y + x4i.z * x4i.z;
	if(rsq > 0.0){
		ir = 1.0/sqrt(rsq);
		a = -ksqMsun * x4i.w * ir;
	}
	return a;

}


// **************************************
//This Kernel computes the potential energy for the body i, in the case N >= 64.
//It uses a reduction formula to compute the sum over all bodies. 
//The Kernel is launched with N blocks. 
//
//Authors: Simon Grimm
//August 2016
// ****************************************
__global__ void potentialEnergy_kernel(double4 *x4_d, double4 *v4_d, const double Msun, double *EnergySum_d, const int st, const int N){

	int idy = threadIdx.x;
	int idx = blockIdx.x;

	double V = 0.0;

	if(idx < N){
		if(x4_d[idx].w > 0.0){
			for(int i = 0; i < N; i += blockDim.x){
				if(idy + i < N){
					if(x4_d[idy + i].w > 0.0){
						V += PE(x4_d[idx], x4_d[idy + i], idx, idy + i);
					}
				}
			}

			__syncthreads();
			for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
				V += __shfl_xor_sync(0xffffffff, V, i, warpSize);
#else
				V += __shfld_xor(V, i);
#endif
//if(i >= 16) printf("VEa %d %d %.20g\n", i, idy, V);
			
			}
			if(blockDim.x > warpSize){
				//reduce across warps
				extern __shared__ double VE_s[];

				int lane = threadIdx.x % warpSize;
				int warp = threadIdx.x / warpSize;
				if(warp == 0){
					VE_s[threadIdx.x] = 0.0;
				}
				__syncthreads();

				if(lane == 0){
					VE_s[warp] = V;
				}

				__syncthreads();
				//reduce previous warp results in the first warp
				if(warp == 0){
					V = VE_s[threadIdx.x];
//printf("VEc %d %d %.20g %d %d\n", 0, idy, V, int(blockDim.x), warpSize);
					for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
						V += __shfl_xor_sync(0xffffffff, V, i, warpSize);
#else
						V += __shfld_xor(V, i);
#endif
//printf("VEd %d %d %.20g\n", i, idy, V);
					}
					if(lane == 0){
						VE_s[0] = V;
					}
				}
				__syncthreads();

				V = VE_s[0];
//printf("VEe %d %.20g\n", idy, V);
			}
			__syncthreads();

			if(idy == 0){
				V *= 0.5 * def_ksq * x4_d[idx].w;
				V += PESun(x4_d[idx], def_ksq * Msun);

				EnergySum_d[idx] = V;
//printf("%d %.20g\n", idx, V);
			}
		}
		else{
			EnergySum_d[idx] = 0.0;
		}
	}
}

// **************************************
// This Kernel computes the energy change due to ejections
// It uses a reduction formula to compute the sum over all bodies. 
// The Kernel is launched with 1 block 
//
//Authors: Simon Grimm
//September 2019
// ****************************************
__global__ void EjectionEnergy_kernel(double4 *x4_d, double4 *v4_d, double4 *spin_d, double Msun, int idx, double *U_d, double *LI_d, double3 *vcom_d, const int N){

	int idy = threadIdx.x;

	double T = 0.0;
	double V = 0.0;
	double mtot = 0.0;
	double3 p, s, L;

	p.x = 0.0;
	p.y = 0.0;
	p.z = 0.0;

	s.x = 0.0;
	s.y = 0.0;
	s.z = 0.0;

	L.x = 0.0;
	L.y = 0.0;
	L.z = 0.0;

	extern __shared__ double TE_s[];
	double *T_s = TE_s;                             //size: warpSize
	double *V_s = (double*)&T_s[warpSize];          //size: warpSize
	double *m_s = (double*)&V_s[warpSize];          //size: warpSize
	double3 *p_s = (double3*)&m_s[warpSize];        //size: warpSize
	double3 *s_s = (double3*)&p_s[warpSize];        //size: warpSize
	double3 *L_s = (double3*)&s_s[warpSize];        //size: warpSize



	//--------------------------------------------
	//calculate s_s and p_s first
	//--------------------------------------------
	for(int i = 0; i < N; i += blockDim.x){
		if(idy + i < N){
			double m = x4_d[idy + i].w;
			if(m >= 0.0){
				s.x += m * x4_d[idy + i].x;
				s.y += m * x4_d[idy + i].y;
				s.z += m * x4_d[idy + i].z;
				p.x += m * v4_d[idy + i].x;
				p.y += m * v4_d[idy + i].y;
				p.z += m * v4_d[idy + i].z;
			}
		}
	}
	__syncthreads();

	for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
		s.x += __shfl_xor_sync(0xffffffff, s.x, i, warpSize);
		s.y += __shfl_xor_sync(0xffffffff, s.y, i, warpSize);
		s.z += __shfl_xor_sync(0xffffffff, s.z, i, warpSize);
		p.x += __shfl_xor_sync(0xffffffff, p.x, i, warpSize);
		p.y += __shfl_xor_sync(0xffffffff, p.y, i, warpSize);
		p.z += __shfl_xor_sync(0xffffffff, p.z, i, warpSize);
#else
		s.x += __shfld_xor(s.x, i);
		s.y += __shfld_xor(s.y, i);
		s.z += __shfld_xor(s.z, i);
		p.x += __shfld_xor(p.x, i);
		p.y += __shfld_xor(p.y, i);
		p.z += __shfld_xor(p.z, i);
#endif
	}
	__syncthreads();

	if(blockDim.x > warpSize){
		//reduce across warps

		int lane = threadIdx.x % warpSize;
		int warp = threadIdx.x / warpSize;
		if(warp == 0){
			s_s[threadIdx.x].x = 0.0;
			s_s[threadIdx.x].y = 0.0;
			s_s[threadIdx.x].z = 0.0;
			p_s[threadIdx.x].x = 0.0;
			p_s[threadIdx.x].y = 0.0;
			p_s[threadIdx.x].z = 0.0;
		}
		__syncthreads();

		if(lane == 0){
			s_s[warp] = s;
			p_s[warp] = p;
		}

		__syncthreads();
		//reduce previous warp results in the first warp
		if(warp == 0){
			s = s_s[threadIdx.x];
			p = p_s[threadIdx.x];
//printf("VEc %d %d %.20g %d %d\n", 0, idy, V, int(blockDim.x), warpSize);
			for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
				s.x += __shfl_xor_sync(0xffffffff, s.x, i, warpSize);
				s.y += __shfl_xor_sync(0xffffffff, s.y, i, warpSize);
				s.z += __shfl_xor_sync(0xffffffff, s.z, i, warpSize);
				p.x += __shfl_xor_sync(0xffffffff, p.x, i, warpSize);
				p.y += __shfl_xor_sync(0xffffffff, p.y, i, warpSize);
				p.z += __shfl_xor_sync(0xffffffff, p.z, i, warpSize);
#else
				s.x += __shfld_xor(s.x, i);
				s.y += __shfld_xor(s.y, i);
				s.z += __shfld_xor(s.z, i);
				p.x += __shfld_xor(p.x, i);
				p.y += __shfld_xor(p.y, i);
				p.z += __shfld_xor(p.z, i);
#endif
//printf("VEd %d %d %.20g\n", i, idy, V);
			}
			if(lane == 0){
				s_s[0] = s;
				p_s[0] = p;
			}
		}
		__syncthreads();

		s = s_s[0];
		p = p_s[0];
//printf("VEe %d %.20g\n", idy, V);
	}

	//--------------------------------------------
	__syncthreads();

	for(int i = 0; i < N; i += blockDim.x){
		if(idy + i < N){
			double m = x4_d[idy + i].w;
			if(m >= 0.0){
				mtot += m;
				V += PE(x4_d[idx], x4_d[idy + i], idx, idy + i);
				T += 0.5 * m * (v4_d[idy + i].x * v4_d[idy + i].x +  v4_d[idy + i].y * v4_d[idy + i].y + v4_d[idy + i].z * v4_d[idy + i].z);
				//convert to barycentric positions
				double3 x4h;
				x4h.x = x4_d[idy + i].x - s.x / Msun;
				x4h.y = x4_d[idy + i].y - s.y / Msun;
				x4h.z = x4_d[idy + i].z - s.z / Msun;
				L.x += m * (x4h.y * v4_d[idy + i].z - x4h.z * v4_d[idy + i].y) + spin_d[idy + i].x;
				L.y += m * (x4h.z * v4_d[idy + i].x - x4h.x * v4_d[idy + i].z) + spin_d[idy + i].y;
				L.z += m * (x4h.x * v4_d[idy + i].y - x4h.y * v4_d[idy + i].x) + spin_d[idy + i].z;
//printf("L ejection 1 %d %.20g %.20g %.20g\n", idy, L.x, L.y, L.z);

			}
		}
	}

	__syncthreads();
	for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
		mtot += __shfl_xor_sync(0xffffffff, mtot, i, warpSize);
		V += __shfl_xor_sync(0xffffffff, V, i, warpSize);
		T += __shfl_xor_sync(0xffffffff, T, i, warpSize);
		L.x += __shfl_xor_sync(0xffffffff, L.x, i, warpSize);
		L.y += __shfl_xor_sync(0xffffffff, L.y, i, warpSize);
		L.z += __shfl_xor_sync(0xffffffff, L.z, i, warpSize);
#else
		mtot += __shfld_xor(mtot, i);
		V += __shfld_xor(V, i);
		T += __shfld_xor(T, i);
		L.x += __shfld_xor(L.x, i);
		L.y += __shfld_xor(L.y, i);
		L.z += __shfld_xor(L.z, i);
#endif
	}
	__syncthreads();

	if(blockDim.x > warpSize){
		//reduce across warps

		int lane = threadIdx.x % warpSize;
		int warp = threadIdx.x / warpSize;
		if(warp == 0){
			m_s[threadIdx.x] = 0.0;
			V_s[threadIdx.x] = 0.0;
			T_s[threadIdx.x] = 0.0;
			L_s[threadIdx.x].x = 0.0;
			L_s[threadIdx.x].y = 0.0;
			L_s[threadIdx.x].z = 0.0;
		}
		__syncthreads();

		if(lane == 0){
			m_s[warp] = mtot;
			V_s[warp] = V;
			T_s[warp] = T;
			L_s[warp] = L;
		}

		__syncthreads();
		//reduce previous warp results in the first warp
		if(warp == 0){
			mtot = m_s[threadIdx.x];
			V = V_s[threadIdx.x];
			T = T_s[threadIdx.x];
			L = L_s[threadIdx.x];
//printf("VEc %d %d %.20g %d %d\n", 0, idy, V, int(blockDim.x), warpSize);
			for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
				mtot += __shfl_xor_sync(0xffffffff, mtot, i, warpSize);
				V += __shfl_xor_sync(0xffffffff, V, i, warpSize);
				T += __shfl_xor_sync(0xffffffff, T, i, warpSize);
				L.x += __shfl_xor_sync(0xffffffff, L.x, i, warpSize);
				L.y += __shfl_xor_sync(0xffffffff, L.y, i, warpSize);
				L.z += __shfl_xor_sync(0xffffffff, L.z, i, warpSize);
#else
				mtot += __shfld_xor(mtot, i);
				V += __shfld_xor(V, i);
				T += __shfld_xor(T, i);
				L.x += __shfld_xor(L.x, i);
				L.y += __shfld_xor(L.y, i);
				L.z += __shfld_xor(L.z, i);
#endif
//printf("VEd %d %d %.20g\n", i, idy, V);
			}
			if(lane == 0){
				m_s[0] = mtot;
				V_s[0] = V;
				T_s[0] = T;
				L_s[0] = L;
			}
		}
		__syncthreads();

		mtot = m_s[0];
		V = V_s[0];
		T = T_s[0];
		L = L_s[0];
//printf("VEe %d %.20g\n", idy, V);
	}
	__syncthreads();

	mtot = Msun + mtot - x4_d[idx].w;
	if(idy == 0){
		V *= def_ksq * x4_d[idx].w;

		V += PESun(x4_d[idx], def_ksq * Msun);
		double Tsun0 = 0.5 / Msun * ( p.x * p.x + p.y * p.y + p.z * p.z);
		
		double3 Vsun;
		Vsun.x = -p.x / Msun + x4_d[idx].w * v4_d[idx].x/mtot;
		Vsun.y = -p.y / Msun + x4_d[idx].w * v4_d[idx].y/mtot;
		Vsun.z = -p.z / Msun + x4_d[idx].w * v4_d[idx].z/mtot;
	
		double Tsun1 = 0.5 * Msun * (Vsun.x * Vsun.x + Vsun.y * Vsun.y + Vsun.z * Vsun.z);
		
		*U_d += -Tsun1 + Tsun0 + T + V;


		L.x += (s.y * p.z - s.z * p.y) / Msun;
		L.y += (s.z * p.x - s.x * p.z) / Msun;
		L.z += (s.x * p.y - s.y * p.x) / Msun;
		volatile double Ltot = sqrt(L.x * L.x + L.y * L.y + L.z * L.z);
//printf("Ltot ejection 1 %.20g %.20g %.20g\n", Ltot, LI_d[0], Ltot + LI_d[0]);
		LI_d[0] += Ltot;

	}
	__syncthreads();	


	s.x -= x4_d[idx].w * x4_d[idx].x;
	s.y -= x4_d[idx].w * x4_d[idx].y;
	s.z -= x4_d[idx].w * x4_d[idx].z;


	double3 vcom;
	vcom.x = x4_d[idx].w * v4_d[idx].x / mtot;
	vcom.y = x4_d[idx].w * v4_d[idx].y / mtot;
	vcom.z = x4_d[idx].w * v4_d[idx].z / mtot;


	if(idy == 0){
		vcom_d[0].x = vcom.x;
		vcom_d[0].y = vcom.y;
		vcom_d[0].z = vcom.z;
	}

	__syncthreads();
	
	for(int i = 0; i < N; i += blockDim.x){
		if(idy + i < N){
			v4_d[idy + i].x += vcom.x;
			v4_d[idy + i].y += vcom.y;
			v4_d[idy + i].z += vcom.z;
		}
	}
	
	__syncthreads();

	//mark here the particle as ghost particle	
	x4_d[idx].w = -1.0e-12;

	// ---------------------------------------------
	//redo p_s now
	// ---------------------------------------------

	p.x = 0.0;
	p.y = 0.0;
	p.z = 0.0;

	__syncthreads();

	for(int i = 0; i < N; i += blockDim.x){
		if(idy + i < N){
			double m = x4_d[idy + i].w;
			if(m >= 0.0){
				p.x += m * v4_d[idy + i].x;
				p.y += m * v4_d[idy + i].y;
				p.z += m * v4_d[idy + i].z;
			}
		}
	}

	__syncthreads();

	for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
		p.x += __shfl_xor_sync(0xffffffff, p.x, i, warpSize);
		p.y += __shfl_xor_sync(0xffffffff, p.y, i, warpSize);
		p.z += __shfl_xor_sync(0xffffffff, p.z, i, warpSize);
#else
		p.x += __shfld_xor(p.x, i);
		p.y += __shfld_xor(p.y, i);
		p.z += __shfld_xor(p.z, i);
#endif
	}
	__syncthreads();

	if(blockDim.x > warpSize){
		//reduce across warps

		int lane = threadIdx.x % warpSize;
		int warp = threadIdx.x / warpSize;
		if(warp == 0){
			p_s[threadIdx.x].x = 0.0;
			p_s[threadIdx.x].y = 0.0;
			p_s[threadIdx.x].z = 0.0;
		}
		__syncthreads();

		if(lane == 0){
			p_s[warp] = p;
		}

		__syncthreads();
		//reduce previous warp results in the first warp
		if(warp == 0){
			p = p_s[threadIdx.x];
//printf("VEc %d %d %.20g %d %d\n", 0, idy, V, int(blockDim.x), warpSize);
			for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
				p.x += __shfl_xor_sync(0xffffffff, p.x, i, warpSize);
				p.y += __shfl_xor_sync(0xffffffff, p.y, i, warpSize);
				p.z += __shfl_xor_sync(0xffffffff, p.z, i, warpSize);
#else
				p.x += __shfld_xor(p.x, i);
				p.y += __shfld_xor(p.y, i);
				p.z += __shfld_xor(p.z, i);
#endif
//printf("VEd %d %d %.20g\n", i, idy, V);
			}
			if(lane == 0){
				p_s[0] = p;
			}
		}
		__syncthreads();
		p = p_s[0];
	}

	__syncthreads();

	// ------------------------------------------------------

	// ------------------------------------------------------
	//redo now L calculation without the ejected particle
	// ------------------------------------------------------
	T = 0.0;
	L.x = 0.0;
	L.y = 0.0;
	L.z = 0.0;

	__syncthreads();

	for(int i = 0; i < N; i += blockDim.x){
		if(idy + i < N){
			double m = x4_d[idy + i].w;
			if(m >= 0.0){
				T += 0.5 *x4_d[idy + i].w * (v4_d[idy + i].x * v4_d[idy + i].x +  v4_d[idy + i].y * v4_d[idy + i].y + v4_d[idy + i].z * v4_d[idy + i].z);
				//convert to barycentric positions
				double3 x4h;
				x4h.x = x4_d[idy + i].x - s.x / Msun;
				x4h.y = x4_d[idy + i].y - s.y / Msun;
				x4h.z = x4_d[idy + i].z - s.z / Msun;
				L.x += m * (x4h.y * v4_d[idy + i].z - x4h.z * v4_d[idy + i].y) + spin_d[idy + i].x;
				L.y += m * (x4h.z * v4_d[idy + i].x - x4h.x * v4_d[idy + i].z) + spin_d[idy + i].y;
				L.z += m * (x4h.x * v4_d[idy + i].y - x4h.y * v4_d[idy + i].x) + spin_d[idy + i].z;
//printf("L ejection 2 %d %.20g %.20g %.20g\n", idy, L.x, L.y, L.z);

			}
		}
	}

	__syncthreads();
	for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
		T += __shfl_xor_sync(0xffffffff, T, i, warpSize);
		L.x += __shfl_xor_sync(0xffffffff, L.x, i, warpSize);
		L.y += __shfl_xor_sync(0xffffffff, L.y, i, warpSize);
		L.z += __shfl_xor_sync(0xffffffff, L.z, i, warpSize);
#else
		T += __shfld_xor(T, i);
		L.x += __shfld_xor(L.x, i);
		L.y += __shfld_xor(L.y, i);
		L.z += __shfld_xor(L.z, i);
#endif
	}
	__syncthreads();

	if(blockDim.x > warpSize){
		//reduce across warps

		int lane = threadIdx.x % warpSize;
		int warp = threadIdx.x / warpSize;
		if(warp == 0){
			T_s[threadIdx.x] = 0.0;
			L_s[threadIdx.x].x = 0.0;
			L_s[threadIdx.x].y = 0.0;
			L_s[threadIdx.x].z = 0.0;
		}
		__syncthreads();

		if(lane == 0){
			T_s[warp] = T;
			L_s[warp] = L;
		}

		__syncthreads();
		//reduce previous warp results in the first warp
		if(warp == 0){
			T = T_s[threadIdx.x];
			L = L_s[threadIdx.x];
//printf("VEc %d %d %.20g %d %d\n", 0, idy, V, int(blockDim.x), warpSize);
			for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
				T += __shfl_xor_sync(0xffffffff, T, i, warpSize);
				L.x += __shfl_xor_sync(0xffffffff, L.x, i, warpSize);
				L.y += __shfl_xor_sync(0xffffffff, L.y, i, warpSize);
				L.z += __shfl_xor_sync(0xffffffff, L.z, i, warpSize);
#else
				T += __shfld_xor(T, i);
				L.x += __shfld_xor(L.x, i);
				L.y += __shfld_xor(L.y, i);
				L.z += __shfld_xor(L.z, i);
#endif
//printf("VEd %d %d %.20g\n", i, idy, V);
			}
			if(lane == 0){
				T_s[0] = T;
				L_s[0] = L;
			}
		}
		__syncthreads();

		T = T_s[0];
		L = L_s[0];
//printf("VEe %d %.20g\n", idy, V);
	}

	__syncthreads();
	if(idy == 0){
		L.x += (s.y * p.z - s.z * p.y) / Msun;
		L.y += (s.z * p.x - s.x * p.z) / Msun;
		L.z += (s.x * p.y - s.y * p.x) / Msun;
		volatile double Ltot = sqrt(L.x * L.x + L.y * L.y + L.z * L.z);
//printf("Ltot ejection 2 %.20g %.20g %.20g\n", Ltot, LI_d[0], Ltot + LI_d[0]);
		LI_d[0] -= Ltot;
		*U_d -= T;

	}
}


// **************************************
//This Kernel computes the total energy of the system, in the case N >= 128.
//It computes the sum over the momenta p_i which is used to calculate the Kinetic 
//Energy from the sun.
//It computes the sum of the  potential and kinetic energy over all bodies.
//All sums are performed using a reduction formula.
//At the first call, the initial energy is stored in Energy0_d.
//At all other calls, it stores in Energy_d the following quantities:
//Total potential energy, total kinetic energy, Lost Angular Momentum at ejections, inner energy due to collisions + Ejections + Gas, 
//total energy, total Angular Momentum, relative energy Error (E-E0)/E0,  relative angular momentum Error (L-L0)/L0
//
//Authors: Simon Grimm
//August 2016
// ****************************************
__global__ void kineticEnergy_kernel(double4 *x4_d, double4 *v4_d, double4 *spin_d, double *EnergySum_d, double *Energy_d, double Msun, double4 *Spinsun_d, double *U_d, double *LI_d, double *Energy0_d, double *LI0_d, int st, int N, int EE){
	int idy = threadIdx.x;

	double T = 0.0;
	double V = 0.0;
	double E = 0.0;
	double3 p, s, L;

	p.x = 0.0;
	p.y = 0.0;
	p.z = 0.0;

	s.x = 0.0;
	s.y = 0.0;
	s.z = 0.0;

	L.x = 0.0;
	L.y = 0.0;
	L.z = 0.0;

	extern __shared__ double TE_s[];
	double *T_s = TE_s;				//size: warpSize
	double *V_s = (double*)&T_s[warpSize];		//size: warpSize
	double *E_s = (double*)&V_s[warpSize];		//size: warpSize
	double3 *p_s = (double3*)&E_s[warpSize];	//size: warpSize
	double3 *s_s = (double3*)&p_s[warpSize];	//size: warpSize
	double3 *L_s = (double3*)&s_s[warpSize];	//size: warpSize

	for(int i = 0; i < N; i += blockDim.x){
		if(idy + i < N){
			double m = x4_d[idy + i].w;
			if(m > 0.0){
				s.x += m * x4_d[idy + i].x;
				s.y += m * x4_d[idy + i].y;
				s.z += m * x4_d[idy + i].z;
				p.x += m * v4_d[idy + i].x;
				p.y += m * v4_d[idy + i].y;
				p.z += m * v4_d[idy + i].z;
			}
		}
	}
	__syncthreads();
	for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
		s.x += __shfl_xor_sync(0xffffffff, s.x, i, warpSize);
		s.y += __shfl_xor_sync(0xffffffff, s.y, i, warpSize);
		s.z += __shfl_xor_sync(0xffffffff, s.z, i, warpSize);
		p.x += __shfl_xor_sync(0xffffffff, p.x, i, warpSize);
		p.y += __shfl_xor_sync(0xffffffff, p.y, i, warpSize);
		p.z += __shfl_xor_sync(0xffffffff, p.z, i, warpSize);
#else
		s.x += __shfld_xor(s.x, i);
		s.y += __shfld_xor(s.y, i);
		s.z += __shfld_xor(s.z, i);
		p.x += __shfld_xor(p.x, i);
		p.y += __shfld_xor(p.y, i);
		p.z += __shfld_xor(p.z, i);
#endif
	}

	if(blockDim.x > warpSize){
		//reduce across warps

		int lane = threadIdx.x % warpSize;
		int warp = threadIdx.x / warpSize;
		if(warp == 0){
			s_s[threadIdx.x].x = 0.0;
			s_s[threadIdx.x].y = 0.0;
			s_s[threadIdx.x].z = 0.0;
			p_s[threadIdx.x].x = 0.0;
			p_s[threadIdx.x].y = 0.0;
			p_s[threadIdx.x].z = 0.0;
		}
		__syncthreads();

		if(lane == 0){
			s_s[warp] = s;
			p_s[warp] = p;
		}

		__syncthreads();
		//reduce previous warp results in the first warp
		if(warp == 0){
			s = s_s[threadIdx.x];
			p = p_s[threadIdx.x];
//printf("VEc %d %d %.20g %d %d\n", 0, idy, V, int(blockDim.x), warpSize);
			for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
				s.x += __shfl_xor_sync(0xffffffff, s.x, i, warpSize);
				s.y += __shfl_xor_sync(0xffffffff, s.y, i, warpSize);
				s.z += __shfl_xor_sync(0xffffffff, s.z, i, warpSize);
				p.x += __shfl_xor_sync(0xffffffff, p.x, i, warpSize);
				p.y += __shfl_xor_sync(0xffffffff, p.y, i, warpSize);
				p.z += __shfl_xor_sync(0xffffffff, p.z, i, warpSize);
#else
				s.x += __shfld_xor(s.x, i);
				s.y += __shfld_xor(s.y, i);
				s.z += __shfld_xor(s.z, i);
				p.x += __shfld_xor(p.x, i);
				p.y += __shfld_xor(p.y, i);
				p.z += __shfld_xor(p.z, i);
#endif
//printf("VEd %d %d %.20g\n", i, idy, V);
			}
			if(lane == 0){
				s_s[0] = s;
				p_s[0] = p;
			}
		}
		__syncthreads();

		s = s_s[0];
		p = p_s[0];
//printf("VEe %d %.20g\n", idy, V);
	}

	__syncthreads();

	for(int i = 0; i < N; i += blockDim.x){
		if(idy + i < N){
			V += EnergySum_d[idy + i];
			EnergySum_d[idy + i] = 0.0;
			double4 x4 = x4_d[idy + i];
			double4 v4 = v4_d[idy + i];
			if(x4.w > 0.0){
				T += 0.5 * x4.w * (v4.x * v4.x +  v4.y * v4.y + v4.z * v4.z);
			}
			//convert to barycentric positions
			double3 x4h;
			x4h.x = x4.x - s.x / Msun;
			x4h.y = x4.y - s.y / Msun;
			x4h.z = x4.z - s.z / Msun;
			L.x += x4.w * (x4h.y * v4.z - x4h.z * v4.y) + spin_d[idy + i].x;
			L.y += x4.w * (x4h.z * v4.x - x4h.x * v4.z) + spin_d[idy + i].y;
			L.z += x4.w * (x4h.x * v4.y - x4h.y * v4.x) + spin_d[idy + i].z;
//printf("VTa  %d %.20g %.20g\n", idy, V, T);
//printf("L %d %.20g %.20g %.20g\n", idy, L.x, L.y, L.z);
		}
	}
	E = V + T;
	__syncthreads();

	for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
		T += __shfl_xor_sync(0xffffffff, T, i, warpSize);
		V += __shfl_xor_sync(0xffffffff, V, i, warpSize);
		E += __shfl_xor_sync(0xffffffff, E, i, warpSize);
		L.x += __shfl_xor_sync(0xffffffff, L.x, i, warpSize);
		L.y += __shfl_xor_sync(0xffffffff, L.y, i, warpSize);
		L.z += __shfl_xor_sync(0xffffffff, L.z, i, warpSize);
#else
		T += __shfld_xor(T, i);
		V += __shfld_xor(V, i);
		E += __shfld_xor(E, i);
		L.x += __shfld_xor(L.x, i);
		L.y += __shfld_xor(L.y, i);
		L.z += __shfld_xor(L.z, i);
#endif
	}
//printf("VTb  %d %.20g %.20g\n", idy, V, T);
	if(blockDim.x > warpSize){
		//reduce across warps

		int lane = threadIdx.x % warpSize;
		int warp = threadIdx.x / warpSize;
		if(warp == 0){
			T_s[threadIdx.x] = 0.0;
			V_s[threadIdx.x] = 0.0;
			E_s[threadIdx.x] = 0.0;
			L_s[threadIdx.x].x = 0.0;
			L_s[threadIdx.x].y = 0.0;
			L_s[threadIdx.x].z = 0.0;
		}
		__syncthreads();

		if(lane == 0){
			T_s[warp] = T;
			V_s[warp] = V;
			E_s[warp] = E;
			L_s[warp] = L;
		}

		__syncthreads();
		//reduce previous warp results in the first warp
		if(warp == 0){
			T = T_s[threadIdx.x];
			V = V_s[threadIdx.x];
			E = E_s[threadIdx.x];
			L = L_s[threadIdx.x];
//printf("VEc %d %d %.20g %d %d\n", 0, idy, V, int(blockDim.x), warpSize);
			for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
				T += __shfl_xor_sync(0xffffffff, T, i, warpSize);
				V += __shfl_xor_sync(0xffffffff, V, i, warpSize);
				E += __shfl_xor_sync(0xffffffff, E, i, warpSize);
				L.x += __shfl_xor_sync(0xffffffff, L.x, i, warpSize);
				L.y += __shfl_xor_sync(0xffffffff, L.y, i, warpSize);
				L.z += __shfl_xor_sync(0xffffffff, L.z, i, warpSize);
#else
				T += __shfld_xor(T, i);
				V += __shfld_xor(V, i);
				E += __shfld_xor(E, i);
				L.x += __shfld_xor(L.x, i);
				L.y += __shfld_xor(L.y, i);
				L.z += __shfld_xor(L.z, i);
#endif
//printf("VEd %d %d %.20g\n", i, idy, V);
			}
			if(lane == 0){
				T_s[0] = T;
				V_s[0] = V;
				E_s[0] = E;
				L_s[0] = L;
			}
		}
		__syncthreads();

		T = T_s[0];
		V = V_s[0];
		E = E_s[0];
		L = L_s[0];
//printf("VEe %d %.20g\n", idy, V);
	}


	__syncthreads();
	if(idy == 0){
		volatile double Tsun = 0.5 / Msun * (p.x*p.x + p.y*p.y + p.z*p.z);  
		//Lsun
//printf("Lsum %d %.20g %.20g %.20g\n", idy, L.x, L.y, L.z);
		L.x += (s.y * p.z - s.z * p.y) / Msun;
		L.y += (s.z * p.x - s.x * p.z) / Msun;
		L.z += (s.x * p.y - s.y * p.x) / Msun;
//printf("LSun %.20g %.20g %.20g\n", (s.y * p.z - s.z * p.y) / Msun, (s.z * p.x - s.x * p.z) / Msun, (s.x * p.y - s.y * p.x) / Msun);
//printf("Lsum+ %d %.20g %.20g %.20g\n", idy, L.x, L.y, L.z);
		volatile double Ltot = sqrt(L.x * L.x + L.y * L.y + L.z * L.z);
//printf("Ltot %.20g %.20g %.20g\n", Ltot, LI_d[0], Ltot + LI_d[0]);

		double4 Spinsun4 = Spinsun_d[st];
		double Spinsun = sqrt(Spinsun4.x * Spinsun4.x + Spinsun4.y * Spinsun4.y + Spinsun4.z * Spinsun4.z);
//printf("Spinsun %g\n", Spinsun);
		Ltot += Spinsun;

		V *= def_Kg;
		T *= def_Kg;
		E *= def_Kg;
		Tsun *= def_Kg;
		Energy_d[0] = V;
		Energy_d[1] = T + Tsun;
		Energy_d[2] = LI_d[st] * dayUnit;
		Energy_d[3] = U_d[st] * def_Kg;
		Energy_d[4] = T + V + __dmul_rn(U_d[st], def_Kg) + Tsun;
		Energy_d[5] = (Ltot + LI_d[st]) * dayUnit;

		if(EE == 0){

			Energy0_d[st] = T + V + __dmul_rn(U_d[st], def_Kg) + Tsun;
			LI0_d[st] = (Ltot + LI_d[st]) * dayUnit;
			Energy_d[7] = 0.0;
			Energy_d[6] = 0.0;
		}
		if(EE == 1){
			Energy_d[6] = ((Ltot + LI_d[st]) * dayUnit - LI0_d[st]) / LI0_d[st]; 
			Energy_d[7] = ((T + V + __dmul_rn(U_d[st], def_Kg) + Tsun) - Energy0_d[st]) / Energy0_d[st];
		}
	}
}

// *************************************
//This function calls the Energy kernels
//
//Authors: Simon Grimm
//August 2016
// *************************************
__host__ void Data::EnergyCall(int st, int E){

	int NBS = NBS_h[st];
	int NE = NEnergy[st];
	int NN = N_h[st] + Nsmall_h[st];

	potentialEnergy_kernel  <<< NN, min(NB[st], 512), WarpSize * sizeof(double), hstream[st%16] >>> (x4_d + NBS , v4_d + NBS, Msun_h[st].x, EnergySum_d + NBS, st, NN);
	kineticEnergy_kernel <<< 1, min(NBT[st], 512), 12 * WarpSize * sizeof(double), hstream[st%16] >>> (x4_d + NBS, v4_d + NBS, spin_d + NBS, EnergySum_d + NBS, Energy_d + NE, Msun_h[st].x, Spinsun_d, U_d, LI_d, Energy0_d, LI0_d, st, NN, E);
}
// *************************************
//This function calls the EjectionEnergy kernels
//
//Authors: Simon Grimm
//April 2016
// *************************************
__host__ void Data::EjectionEnergyCall(int st, int i){

	int NBS = NBS_h[st];
	int NN = N_h[st] + Nsmall_h[st];

	EjectionEnergy_kernel <<<1, min(NBT[st], 512), 12 * WarpSize * sizeof(double) >>> (x4_d + NBS, v4_d + NBS, spin_d + NBS, Msun_h[st].x, i, U_d + st, LI_d + st, vcom_d + st, NN);
}


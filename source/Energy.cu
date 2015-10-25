#include "Orbit2.h"

// **************************************
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

		if(rsq > 0){
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
//****************************************/
__device__ inline double PESun(double4 x4i, double ksqMsun, double &test){

	double rsq, ir;
	double a = 0.0;

	rsq = x4i.x * x4i.x + x4i.y * x4i.y + x4i.z * x4i.z;
	if(rsq > 0 ){
		ir = 1.0/sqrt(rsq);
		a = -ksqMsun * x4i.w * ir;
	}
	return a;

}


// **************************************
//This Kernel computes the potential energy for the body i, in the case N >= 64.
//It uses a reduction formula to compute the sum over all bodies. 
//The Kernel is launched with NB blocks with Bl threads. NB is the next bigger number of N
//which is a power of two. Bl is the minimum of NB and 512.
//
//Authors: Simon Grimm, Joachim Stadel
//March 2015
//
// ****************************************/
template <int Bl>
__global__ void potentialEnergy64_kernel(double4 *x4_d, double4 *v4_d, double Msun, double *Energy_d, double *test_d, int st, int N){
	int idy = threadIdx.x;
	int idx = blockIdx.x;

	__shared__ volatile double V_s[Bl];
	double test;

	V_s[idy] = 0.0;
	__syncthreads();

	if(idx < N){
		if(x4_d[idx].w > 0){

			for (int i = 0; i < N; i += Bl){
				if(x4_d[idy + i].w > 0 && idy + i < N){
					V_s[idy] += PE(x4_d[idx], x4_d[idy + i], idx, idy + i);
				}
			}

			__syncthreads();
			int s = Bl/2;
			for(int i = 6; i < log2f(Bl); ++i){
				if( idy < s ) {
					V_s[idy] += V_s[idy + s];
				}
				__syncthreads();
				s /= 2;
			}

			if(idy < 32){
				if(Bl >= 64) V_s[idy] += V_s[idy + 32];
				if(Bl >= 32) V_s[idy] += V_s[idy + 16];
				if(Bl >= 16) V_s[idy] += V_s[idy + 8];
				if(Bl >= 8) V_s[idy] += V_s[idy + 4];
				if(Bl >= 4) V_s[idy] += V_s[idy + 2];
				if(Bl >= 2) V_s[idy] += V_s[idy + 1];
			}

			__syncthreads();

			if(idy == 0){
				V_s[0] *= 0.5 * ksq * x4_d[idx].w;

				V_s[0] += PESun(x4_d[idx], ksq * Msun, test);

				Energy_d[idx] = V_s[0];

			}
		}
		else Energy_d[idx] = 0.0;
	}
}

// **************************************
// This Kernel computes the energy change due to ejections,  in the case N >= 64.
// It uses a reduction formula to compute the sum over all bodies. 
// The Kernel is launched with 1 block with Bl threads. Bl is equal to NB, and Bl2 is the minimum of 2*NB and 64.
//
//Authors: Simon Grimm, Joachim Stadel
//March 2015
//
// ****************************************
template <int Bl>
__global__ void EjectionEnergy64_kernel(double4 *x4_d, double4 *v4_d, double3 *spin_d, double Msun, int idx, double *U_d, double *LI_d, double3 *vcom_d, int N){
	int idy = threadIdx.x;

	__shared__ volatile double V_s[Bl];
	__shared__ volatile double T_s[Bl];
	__shared__ volatile double3 p_s[Bl];
	__shared__ volatile double3 s_s[Bl];
	__shared__ volatile double m_s[Bl];
	__shared__ volatile double mtot;
	double test;
	
	mtot = 0.0;

	V_s[idy] = 0.0;
	T_s[idy] = 0.0;
	m_s[idy] = 0.0;
	p_s[idy].x = 0.0;
	p_s[idy].y = 0.0;
	p_s[idy].z = 0.0;
        s_s[idy].x = 0.0;
        s_s[idy].y = 0.0;
        s_s[idy].z = 0.0;

	__syncthreads();

	for (int i = 0; i < N; i += Bl){
		if(x4_d[idy + i].w > 0 && idy + i < N){
			m_s[idy] += x4_d[idy + i].w;
			V_s[idy] += PE(x4_d[idx], x4_d[idy + i], idx, idy + i);
			T_s[idy] += 0.5 *x4_d[idy + i].w * (v4_d[idy + i].x * v4_d[idy + i].x +  v4_d[idy + i].y * v4_d[idy + i].y + v4_d[idy + i].z * v4_d[idy + i].z);
			p_s[idy].x += x4_d[idy + i].w * v4_d[idy + i].x;
			p_s[idy].y += x4_d[idy + i].w * v4_d[idy + i].y;
			p_s[idy].z += x4_d[idy + i].w * v4_d[idy + i].z;
                        s_s[idy].x += x4_d[idy + i].w * x4_d[idy + i].x;
                        s_s[idy].y += x4_d[idy + i].w * x4_d[idy + i].y;
                        s_s[idy].z += x4_d[idy + i].w * x4_d[idy + i].z;
		}
	}

	__syncthreads();
	int s = Bl/2;
	for(int i = 6; i < log2f(Bl); ++i){
		if( idy < s ) {
			m_s[idy] += m_s[idy + s];
			V_s[idy] += V_s[idy + s];
			T_s[idy] += T_s[idy + s];
			p_s[idy].x += p_s[idy + s].x;
			p_s[idy].y += p_s[idy + s].y;
			p_s[idy].z += p_s[idy + s].z;
                        s_s[idy].x += s_s[idy + s].x;
                        s_s[idy].y += s_s[idy + s].y;
                        s_s[idy].z += s_s[idy + s].z;
		}
		__syncthreads();
		s /= 2;
	}

	if(idy < 32){
		V_s[idy] += V_s[idy + 32];
		V_s[idy] += V_s[idy + 16];
		V_s[idy] += V_s[idy + 8];
		V_s[idy] += V_s[idy + 4];
		V_s[idy] += V_s[idy + 2];
		V_s[idy] += V_s[idy + 1];
		
		T_s[idy] += T_s[idy + 32];
		T_s[idy] += T_s[idy + 16];
		T_s[idy] += T_s[idy + 8];
		T_s[idy] += T_s[idy + 4];
		T_s[idy] += T_s[idy + 2];
		T_s[idy] += T_s[idy + 1];
		
		m_s[idy] += m_s[idy + 32];
		m_s[idy] += m_s[idy + 16];
		m_s[idy] += m_s[idy + 8];
		m_s[idy] += m_s[idy + 4];
		m_s[idy] += m_s[idy + 2];
		m_s[idy] += m_s[idy + 1];
		
		p_s[idy].x += p_s[idy + 32].x;
		p_s[idy].x += p_s[idy + 16].x;
		p_s[idy].x += p_s[idy + 8].x;
		p_s[idy].x += p_s[idy + 4].x;
		p_s[idy].x += p_s[idy + 2].x;
		p_s[idy].x += p_s[idy + 1].x;

		p_s[idy].y += p_s[idy + 32].y;
		p_s[idy].y += p_s[idy + 16].y;
		p_s[idy].y += p_s[idy + 8].y;
		p_s[idy].y += p_s[idy + 4].y;
		p_s[idy].y += p_s[idy + 2].y;
		p_s[idy].y += p_s[idy + 1].y;

		p_s[idy].z += p_s[idy + 32].z;
		p_s[idy].z += p_s[idy + 16].z;
		p_s[idy].z += p_s[idy + 8].z;
		p_s[idy].z += p_s[idy + 4].z;
		p_s[idy].z += p_s[idy + 2].z;
		p_s[idy].z += p_s[idy + 1].z;

                s_s[idy].x += s_s[idy + 32].x;
                s_s[idy].x += s_s[idy + 16].x;
                s_s[idy].x += s_s[idy + 8].x;
                s_s[idy].x += s_s[idy + 4].x;
                s_s[idy].x += s_s[idy + 2].x;
                s_s[idy].x += s_s[idy + 1].x;

                s_s[idy].y += s_s[idy + 32].y;
                s_s[idy].y += s_s[idy + 16].y;
                s_s[idy].y += s_s[idy + 8].y;
                s_s[idy].y += s_s[idy + 4].y;
                s_s[idy].y += s_s[idy + 2].y;
                s_s[idy].y += s_s[idy + 1].y;

                s_s[idy].z += s_s[idy + 32].z;
                s_s[idy].z += s_s[idy + 16].z;
                s_s[idy].z += s_s[idy + 8].z;
                s_s[idy].z += s_s[idy + 4].z;
                s_s[idy].z += s_s[idy + 2].z;
                s_s[idy].z += s_s[idy + 1].z;

	}

	__syncthreads();
	if(idy == 0){
		V_s[0] *= ksq * x4_d[idx].w;

		V_s[0] += PESun(x4_d[idx], ksq * Msun, test);
		double Tsun0 = 0.5 / Msun * ( p_s[0].x * p_s[0].x + p_s[0].y * p_s[0].y + p_s[0].z * p_s[0].z);
		
		mtot = Msun + m_s[0] - x4_d[idx].w;
		
		double3 Vsun;
		Vsun.x = -p_s[0].x / Msun + x4_d[idx].w * v4_d[idx].x/mtot;
		Vsun.y = -p_s[0].y / Msun + x4_d[idx].w * v4_d[idx].y/mtot;
		Vsun.z = -p_s[0].z / Msun + x4_d[idx].w * v4_d[idx].z/mtot;
		
		double Tsun1 = 0.5 * Msun * (Vsun.x * Vsun.x + Vsun.y * Vsun.y + Vsun.z * Vsun.z);
		
		*U_d += -Tsun1 + Tsun0 + T_s[0] + V_s[0];

                //convert to barycentric positions
                double4 x4i;
                x4i.x = x4_d[idx].x - s_s[0].x / Msun;
                x4i.y = x4_d[idx].y - s_s[0].y / Msun;
                x4i.z = x4_d[idx].z - s_s[0].z / Msun;
                x4i.w = x4_d[idx].w;
                double3 Li;
                Li.x = x4i.w * (x4i.y * v4_d[idx].z - x4i.z * v4_d[idx].y) + spin_d[idx].x;
                Li.y = x4i.w * (x4i.z * v4_d[idx].x - x4i.x * v4_d[idx].z) + spin_d[idx].y;
                Li.z = x4i.w * (x4i.x * v4_d[idx].y - x4i.y * v4_d[idx].x) + spin_d[idx].z;

		*LI_d += sqrt(Li.x * Li.x + Li.y * Li.y + Li.z * Li.z);
	}
	__syncthreads();	

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
	
	for (int i = 0; i < N; i += Bl){
		if(idy + i < N){
			v4_d[idy + i].x += vcom.x;
			v4_d[idy + i].y += vcom.y;
			v4_d[idy + i].z += vcom.z;
		}
	}
	
	__syncthreads();
	
	x4_d[idx].w = -1.0e-12;
	T_s[idy] = 0.0;
	
	__syncthreads();
	for (int i = 0; i < N; i += Bl){
		if(x4_d[idy + i].w > 0 && idy + i < N){
			T_s[idy] += 0.5 *x4_d[idy + i].w * (v4_d[idy + i].x * v4_d[idy + i].x +  v4_d[idy + i].y * v4_d[idy + i].y + v4_d[idy + i].z * v4_d[idy + i].z);
		}
	}
	
	__syncthreads();
	
	s = Bl/2;
	for(int i = 6; i < log2f(Bl); ++i){
		if( idy < s ) {
			T_s[idy] += T_s[idy + s];
		}
		__syncthreads();
		s /= 2;
	}
	
	if(idy < 32){
		if(Bl >= 64) T_s[idy] += T_s[idy + 32];
		if(Bl >= 32) T_s[idy] += T_s[idy + 16];
		if(Bl >= 16) T_s[idy] += T_s[idy + 8];
		if(Bl >= 8) T_s[idy] += T_s[idy + 4];
		if(Bl >= 4) T_s[idy] += T_s[idy + 2];
		if(Bl >= 2) T_s[idy] += T_s[idy + 1];
	}
	__syncthreads();
	
	if(idy == 0){
		*U_d -= T_s[0];
	}
}


// **************************************
//This Kernel computes the potential energy for the body i, in the case N < 64.
//It uses a reduction formula to compute the sum over all bodies. 
//The Kernel is launched with NB blocks with Bl threads. NB is the next bigger number of N
//which is a power of two. Bl is equal to NB, and Bl2 is the minimum of 2*NB and 64.
//
//Authors: Simon Grimm, Joachim Stadel
//March 2014
//
// ****************************************
template<int Bl, int Bl2>
__global__ void potentialEnergy32_kernel(double4 *x4_d, double4 *v4_d, double Msun, double *Energy_d, double *test_d, int st, int N){
	int idy = threadIdx.x;
	int idx = blockIdx.x;

	__shared__ double V_s[Bl2];
	double test;

	V_s[idy] = 0.0;
	V_s[idy + Bl] = 0.0;

	__syncthreads();

	if(idx < N){
		if(x4_d[idx].w > 0){

			if(x4_d[idy].w > 0 && idy < N){
				V_s[idy] += PE(x4_d[idx], x4_d[idy], idx, idy);
			}		
		}
	}
	__syncthreads();

	if(idy < 32){
		volatile double *V = V_s;
		if(Bl >= 32) V[idy] += V[idy + 16];
		V[idy] += V[idy + 8];
		V[idy] += V[idy + 4];
		V[idy] += V[idy + 2];
		V[idy] += V[idy + 1];
	}

	__syncthreads();
	if(idx < N && idy == 0){
		if(x4_d[idx].w > 0){
			V_s[0] *= (0.5 * ksq) * x4_d[idx].w;
			V_s[0] += PESun(x4_d[idx], ksq * Msun, test);
			Energy_d[idx] = V_s[0];
		}
		else Energy_d[idx] = 0.0;
	}

}

// **************************************
//This Kernel computes the energy change due to ejections,  in the case N < 64.
//It uses a reduction formula to compute the sum over all bodies. 
//The Kernel is launched with 1 block with Bl threads. Bl is equal to NB, and Bl2 is the minimum of 2*NB and 64.
//
//Authors: Simon Grimm, Joachim Stadel
//March 2014
//
// ****************************************
template<int Bl, int Bl2>
__global__ void EjectionEnergy32_kernel(double4 *x4_d, double4 *v4_d, double3 *spin_d, double Msun, int idx, double *U_d, double *LI_d, double3 *vcom_d, int N){
	int idy = threadIdx.x;

	__shared__ volatile double V_s[Bl2];
	__shared__ volatile double T_s[Bl2];
	__shared__ volatile double3 p_s[Bl2];
	__shared__ volatile double3 s_s[Bl2];
	__shared__ volatile double m_s[Bl2];
	__shared__ volatile double mtot;
	double test;

	mtot = 0.0;
	
	V_s[idy] = 0.0;
	T_s[idy] = 0.0;
	m_s[idy] = 0.0;
	p_s[idy].x = 0.0;
	p_s[idy].y = 0.0;
	p_s[idy].z = 0.0;
        s_s[idy].x = 0.0;
        s_s[idy].y = 0.0;
        s_s[idy].z = 0.0;

	V_s[idy + Bl] = 0.0;
	T_s[idy + Bl] = 0.0;
	m_s[idy + Bl] = 0.0;
	p_s[idy + Bl].x = 0.0;
	p_s[idy + Bl].y = 0.0;
	p_s[idy + Bl].z = 0.0;
        s_s[idy + Bl].x = 0.0;
        s_s[idy + Bl].y = 0.0;
        s_s[idy + Bl].z = 0.0;

	if(x4_d[idy].w > 0 && idy < N){
		m_s[idy] = x4_d[idy].w;
		V_s[idy] += PE(x4_d[idx], x4_d[idy], idx, idy);
		T_s[idy] = 0.5 * m_s[idy] * (v4_d[idy].x * v4_d[idy].x + v4_d[idy].y * v4_d[idy].y + v4_d[idy].z * v4_d[idy].z);
	
		p_s[idy].x = m_s[idy] * v4_d[idy].x;
		p_s[idy].y = m_s[idy] * v4_d[idy].y;
		p_s[idy].z = m_s[idy] * v4_d[idy].z;
                s_s[idy].x = m_s[idy] * x4_d[idy].x;
                s_s[idy].y = m_s[idy] * x4_d[idy].y;
                s_s[idy].z = m_s[idy] * x4_d[idy].z;
	}
	
	__syncthreads();

	if(idy < 32){
		if(Bl >= 32) V_s[idy] += V_s[idy + 16];
		V_s[idy] += V_s[idy + 8];
		V_s[idy] += V_s[idy + 4];
		V_s[idy] += V_s[idy + 2];
		V_s[idy] += V_s[idy + 1];
		
		if(Bl >= 32) T_s[idy] += T_s[idy + 16];
		T_s[idy] += T_s[idy + 8];
		T_s[idy] += T_s[idy + 4];
		T_s[idy] += T_s[idy + 2];
		T_s[idy] += T_s[idy + 1];
		
		if(Bl >= 32) p_s[idy].x += p_s[idy + 16].x;
		p_s[idy].x += p_s[idy + 8].x;
		p_s[idy].x += p_s[idy + 4].x;
		p_s[idy].x += p_s[idy + 2].x;
		p_s[idy].x += p_s[idy + 1].x;
		
		if(Bl >= 32) p_s[idy].y += p_s[idy + 16].y;
		p_s[idy].y += p_s[idy + 8].y;
		p_s[idy].y += p_s[idy + 4].y;
		p_s[idy].y += p_s[idy + 2].y;
		p_s[idy].y += p_s[idy + 1].y;
		
		if(Bl >= 32) p_s[idy].z += p_s[idy + 16].z;
		p_s[idy].z += p_s[idy + 8].z;
		p_s[idy].z += p_s[idy + 4].z;
		p_s[idy].z += p_s[idy + 2].z;
		p_s[idy].z += p_s[idy + 1].z;

                if(Bl >= 32) s_s[idy].x += s_s[idy + 16].x;
                s_s[idy].x += s_s[idy + 8].x;
                s_s[idy].x += s_s[idy + 4].x;
                s_s[idy].x += s_s[idy + 2].x;
                s_s[idy].x += s_s[idy + 1].x;

                if(Bl >= 32) s_s[idy].y += s_s[idy + 16].y;
                s_s[idy].y += s_s[idy + 8].y;
                s_s[idy].y += s_s[idy + 4].y;
                s_s[idy].y += s_s[idy + 2].y;
                s_s[idy].y += s_s[idy + 1].y;

                if(Bl >= 32) s_s[idy].z += s_s[idy + 16].z;
                s_s[idy].z += s_s[idy + 8].z;
                s_s[idy].z += s_s[idy + 4].z;
                s_s[idy].z += s_s[idy + 2].z;
                s_s[idy].z += s_s[idy + 1].z;
		
		if(Bl >= 32) m_s[idy] += m_s[idy + 16];
		m_s[idy] += m_s[idy + 8];
		m_s[idy] += m_s[idy + 4];
		m_s[idy] += m_s[idy + 2];
		m_s[idy] += m_s[idy + 1];
	}

	__syncthreads();
	if(idy == 0){
		V_s[0] *= ksq * x4_d[idx].w;

		V_s[0] += PESun(x4_d[idx], ksq * Msun, test);
		double Tsun0 = 0.5 / Msun * ( p_s[0].x * p_s[0].x + p_s[0].y * p_s[0].y + p_s[0].z * p_s[0].z);
		
		mtot = Msun + m_s[0] - x4_d[idx].w;
		
		double3 Vsun;
		Vsun.x = -p_s[0].x / Msun + x4_d[idx].w * v4_d[idx].x/mtot;
		Vsun.y = -p_s[0].y / Msun + x4_d[idx].w * v4_d[idx].y/mtot;
		Vsun.z = -p_s[0].z / Msun + x4_d[idx].w * v4_d[idx].z/mtot;
		
		double Tsun1 = 0.5 * Msun * (Vsun.x * Vsun.x + Vsun.y * Vsun.y + Vsun.z * Vsun.z);
		
		*U_d += -Tsun1 + Tsun0 + T_s[0] + V_s[0];

		//convert to barycentric positions
		double4 x4i;
		x4i.x = x4_d[idx].x - s_s[0].x / Msun;
		x4i.y = x4_d[idx].y - s_s[0].y / Msun;
		x4i.z = x4_d[idx].z - s_s[0].z / Msun;
		x4i.w = x4_d[idx].w;
		double3 Li;
                Li.x = x4i.w * (x4i.y * v4_d[idx].z - x4i.z * v4_d[idx].y) + spin_d[idx].x;
                Li.y = x4i.w * (x4i.z * v4_d[idx].x - x4i.x * v4_d[idx].z) + spin_d[idx].y;
                Li.z = x4i.w * (x4i.x * v4_d[idx].y - x4i.y * v4_d[idx].x) + spin_d[idx].z;

		*LI_d += sqrt(Li.x * Li.x + Li.y * Li.y + Li.z * Li.z);
	}
	__syncthreads();

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

	if(idy < N ){	
		v4_d[idy].x += vcom.x;
		v4_d[idy].y += vcom.y;
		v4_d[idy].z += vcom.z;
	}
	
	x4_d[idx].w = -1.0e-12;
	T_s[idy] = 0.0;
	T_s[idy + Bl] = 0.0;
	
	__syncthreads();

	if(x4_d[idy].w > 0 && idy < N){	
		T_s[idy] = 0.5 * x4_d[idy].w * (v4_d[idy].x * v4_d[idy].x + v4_d[idy].y * v4_d[idy].y + v4_d[idy].z * v4_d[idy].z);
	}
	__syncthreads();

	if(idy < 32){
		if(Bl >= 32) T_s[idy] += T_s[idy + 16];
		T_s[idy] += T_s[idy + 8];
		T_s[idy] += T_s[idy + 4];
		T_s[idy] += T_s[idy + 2];
		T_s[idy] += T_s[idy + 1];
	}

	__syncthreads();
	
	if(idy == 0){
		*U_d -= T_s[0];
	}
}

__global__ void EjectionEnergysmall1_kernel(double4 *v4small_d, int Nsmall, double3 *vcom_d){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;	

	if(id < Nsmall){
		v4small_d[id].x += vcom_d[0].x;
		v4small_d[id].y += vcom_d[0].y;
		v4small_d[id].z += vcom_d[0].z;
	}
}

__global__ void EjectionEnergysmall2_kernel(double4 *x4small_d, int i){

	x4small_d[i].w = -1.0e-12;
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
//The kernel is launched with 1 Block with Bl threads. Bl is the minimum of NB and 256. NB is the next bigger number of N
//which is a power of two.
//
//Authors: Simon Grimm, Joachim Stadel
//March 2015
//
// ****************************************
template <int Bl>
__global__ void kineticEnergy128_kernel(double4 *x4_d, double4 *v4_d, double3 *spin_d, double *Energy_d, double Msun, double *U_d, double *LI_d, double *test_d, double *Energy0_d, double *LI0_d, int st, int N, int E){
	int idy = threadIdx.x;

	__shared__ double T_s[Bl];
	__shared__ double V_s[Bl];
	__shared__ double E_s[Bl];
	__shared__ double3 p_s[Bl];
	__shared__ double3 s_s[Bl];
	__shared__ double3 L_s[Bl];

	T_s[idy] = 0.0;
	V_s[idy] = 0.0;
	E_s[idy] = 0.0;
	L_s[idy].x = 0.0;
	L_s[idy].y = 0.0;
	L_s[idy].z = 0.0;

	s_s[idy].x = 0.0;
	s_s[idy].y = 0.0;
	s_s[idy].z = 0.0;
	p_s[idy].x = 0.0;
	p_s[idy].y = 0.0;
	p_s[idy].z = 0.0;

	for(int i = 0; i < N; i += Bl){
		if(x4_d[idy + i].w > 0 && idy + i < N){
			s_s[idy].x += x4_d[idy + i].w * x4_d[idy + i].x;
			s_s[idy].y += x4_d[idy + i].w * x4_d[idy + i].y;
			s_s[idy].z += x4_d[idy + i].w * x4_d[idy + i].z;
			p_s[idy].x += x4_d[idy + i].w * v4_d[idy + i].x;
			p_s[idy].y += x4_d[idy + i].w * v4_d[idy + i].y;
			p_s[idy].z += x4_d[idy + i].w * v4_d[idy + i].z;
		}
	}
	__syncthreads();

	if(Bl >= 512){
		if(idy < 256){
			s_s[idy].x += s_s[idy + 256].x;
			s_s[idy].y += s_s[idy + 256].y;
			s_s[idy].z += s_s[idy + 256].z;
			p_s[idy].x += p_s[idy + 256].x;
			p_s[idy].y += p_s[idy + 256].y;
			p_s[idy].z += p_s[idy + 256].z;
		}
	}
	__syncthreads();

	if(Bl >= 256){
		if(idy < 128){
			s_s[idy].x += s_s[idy + 128].x;
			s_s[idy].y += s_s[idy + 128].y;
			s_s[idy].z += s_s[idy + 128].z;
			p_s[idy].x += p_s[idy + 128].x;
			p_s[idy].y += p_s[idy + 128].y;
			p_s[idy].z += p_s[idy + 128].z;
		}
	}
	__syncthreads();

	if(Bl >= 128){
		if(idy < 64){
			s_s[idy].x += s_s[idy + 64].x;
			s_s[idy].y += s_s[idy + 64].y;
			s_s[idy].z += s_s[idy + 64].z;
			p_s[idy].x += p_s[idy + 64].x;
			p_s[idy].y += p_s[idy + 64].y;
			p_s[idy].z += p_s[idy + 64].z;
		}
	}
	__syncthreads();
	
	if(idy < 32){
		volatile double3 *p = p_s;
		volatile double3 *s = s_s;
		s[idy].x += s[idy + 32].x;
		s[idy].x += s[idy + 16].x;
		s[idy].x += s[idy + 8].x;
		s[idy].x += s[idy + 4].x;
		s[idy].x += s[idy + 2].x;
		s[idy].x += s[idy + 1].x;

		s[idy].y += s[idy + 32].y;
		s[idy].y += s[idy + 16].y;
		s[idy].y += s[idy + 8].y;
		s[idy].y += s[idy + 4].y;
		s[idy].y += s[idy + 2].y;
		s[idy].y += s[idy + 1].y;

		s[idy].z += s[idy + 32].z;
		s[idy].z += s[idy + 16].z;
		s[idy].z += s[idy + 8].z;
		s[idy].z += s[idy + 4].z;
		s[idy].z += s[idy + 2].z;
		s[idy].z += s[idy + 1].z;

		p[idy].x += p[idy + 32].x;
		p[idy].x += p[idy + 16].x;
		p[idy].x += p[idy + 8].x;
		p[idy].x += p[idy + 4].x;
		p[idy].x += p[idy + 2].x;
		p[idy].x += p[idy + 1].x;

		p[idy].y += p[idy + 32].y;
		p[idy].y += p[idy + 16].y;
		p[idy].y += p[idy + 8].y;
		p[idy].y += p[idy + 4].y;
		p[idy].y += p[idy + 2].y;
		p[idy].y += p[idy + 1].y;

		p[idy].z += p[idy + 32].z;
		p[idy].z += p[idy + 16].z;
		p[idy].z += p[idy + 8].z;
		p[idy].z += p[idy + 4].z;
		p[idy].z += p[idy + 2].z;
		p[idy].z += p[idy + 1].z;
	}
	__syncthreads();

	for(int i = 0; i < N; i += Bl){
		if(idy + i < N){
			V_s[idy] += Energy_d[idy + i];
			double4 x4 = x4_d[idy + i];
			double4 v4 = v4_d[idy + i];
			if(x4_d[idy + i].w > 0){
				T_s[idy] += 0.5 * x4.w * (v4.x * v4.x +  v4.y * v4.y + v4.z * v4.z);
			}
			//convert to barycentric positions
			double3 x4h;
			x4h.x = x4.x - s_s[0].x / Msun;
			x4h.y = x4.y - s_s[0].y / Msun;
			x4h.z = x4.z - s_s[0].z / Msun;
			L_s[idy].x += x4.w * (x4h.y * v4.z - x4h.z * v4.y) + spin_d[idy + i].x;
			L_s[idy].y += x4.w * (x4h.z * v4.x - x4h.x * v4.z) + spin_d[idy + i].y;
			L_s[idy].z += x4.w * (x4h.x * v4.y - x4h.y * v4.x) + spin_d[idy + i].z;

		}
	}
	E_s[idy] = V_s[idy] + T_s[idy];
	__syncthreads();

	if(Bl >= 512){
		if(idy < 256){
			T_s[idy] += T_s[idy + 256];
			V_s[idy] += V_s[idy + 256];
			E_s[idy] += E_s[idy + 256];
			L_s[idy].x += L_s[idy + 256].x;
			L_s[idy].y += L_s[idy + 256].y;
			L_s[idy].z += L_s[idy + 256].z;
		}
	}
	__syncthreads();

	if(Bl >= 256){
		if(idy < 128){
			T_s[idy] += T_s[idy + 128];
			V_s[idy] += V_s[idy + 128];
			E_s[idy] += E_s[idy + 128];
			L_s[idy].x += L_s[idy + 128].x;
			L_s[idy].y += L_s[idy + 128].y;
			L_s[idy].z += L_s[idy + 128].z;
		}
	}
	__syncthreads();

	if(Bl >= 128){
		if(idy < 64){
			T_s[idy] += T_s[idy + 64];
			V_s[idy] += V_s[idy + 64];
			E_s[idy] += E_s[idy + 64];
			L_s[idy].x += L_s[idy + 64].x;
			L_s[idy].y += L_s[idy + 64].y;
			L_s[idy].z += L_s[idy + 64].z;
		}
	}

	__syncthreads();
	if(idy < 32){
		volatile double *T = T_s;
		volatile double *V = V_s;
		volatile double *Es = E_s;
		volatile double3 *L = L_s;
		T[idy] += T[idy + 32];
		T[idy] += T[idy + 16];
		T[idy] += T[idy + 8];
		T[idy] += T[idy + 4];
		T[idy] += T[idy + 2];
		T[idy] += T[idy + 1];

		V[idy] += V[idy + 32];
		V[idy] += V[idy + 16];
		V[idy] += V[idy + 8];
		V[idy] += V[idy + 4];
		V[idy] += V[idy + 2];
		V[idy] += V[idy + 1];

		Es[idy] += Es[idy + 32];
		Es[idy] += Es[idy + 16];
		Es[idy] += Es[idy + 8];
		Es[idy] += Es[idy + 4];
		Es[idy] += Es[idy + 2];
		Es[idy] += Es[idy + 1];

		L[idy].x += L[idy + 32].x;
		L[idy].x += L[idy + 16].x;
		L[idy].x += L[idy + 8].x;
		L[idy].x += L[idy + 4].x;
		L[idy].x += L[idy + 2].x;
		L[idy].x += L[idy + 1].x;

		L[idy].y += L[idy + 32].y;
		L[idy].y += L[idy + 16].y;
		L[idy].y += L[idy + 8].y;
		L[idy].y += L[idy + 4].y;
		L[idy].y += L[idy + 2].y;
		L[idy].y += L[idy + 1].y;

		L[idy].z += L[idy + 32].z;
		L[idy].z += L[idy + 16].z;
		L[idy].z += L[idy + 8].z;
		L[idy].z += L[idy + 4].z;
		L[idy].z += L[idy + 2].z;
		L[idy].z += L[idy + 1].z;
	}

	__syncthreads();
	if(idy == 0){
		volatile double Tsun = 0.5 / Msun * (p_s[0].x*p_s[0].x + p_s[0].y*p_s[0].y + p_s[0].z*p_s[0].z);  
		//Lsun
		L_s[0].x += s_s[0].y * p_s[0].z - s_s[0].z * p_s[0].y;
		L_s[0].y += s_s[0].z * p_s[0].x - s_s[0].x * p_s[0].z;
		L_s[0].z += s_s[0].x * p_s[0].y - s_s[0].y * p_s[0].x;
		volatile double Ltot = sqrt(L_s[0].x * L_s[0].x + L_s[0].y * L_s[0].y + L_s[0].z * L_s[0].z);
		V_s[0] *= Kg;
		T_s[0] *= Kg;
		E_s[0] *= Kg;
		Tsun *= Kg;
		Energy_d[0] = V_s[0];
		Energy_d[1] = T_s[0] + Tsun;
		Energy_d[2] = LI_d[st] * Kg;
		Energy_d[3] = U_d[st] * Kg;
		Energy_d[4] = T_s[0] + V_s[0] + __dmul_rn(U_d[st], Kg) + Tsun;
		Energy_d[5] = (Ltot + LI_d[st]) * Kg;

		if(E == 0){ 
			Energy0_d[st] = T_s[0] + V_s[0] + __dmul_rn(U_d[st], Kg) + Tsun;
			LI0_d[st] = (Ltot + LI_d[st]) * Kg;
			Energy_d[7] = 0.0;
			Energy_d[6] = 0.0;
		}
		if(E == 1){
			Energy_d[6] = ((Ltot + LI_d[st]) * Kg - LI0_d[st]) / LI0_d[st]; 
			Energy_d[7] = ((T_s[0] + V_s[0] + __dmul_rn(U_d[st], Kg) + Tsun) - Energy0_d[st]) / Energy0_d[st];
		}
	}
}

// **************************************
//This Kernel computes the total energy of the system, in the case N < 128.
//It computes the sum over the momenta p_i which is used to calculate the Kinetic 
//Energy from the sun.
//It computes the sum of the  potential and kinetic energy over all bodies.
//All sums are performed using a reduction formula.
//At the first call, the initial energy is stored in Energy0_d.
//At all other calls, it stores in Energy_d the following quantities:
//Total potential energy, total kinetic energy, Lost Angular Momentum at ejections, inner energy due to collisions + Ejections + Gas, 
//total energy, total Angular Momentum, relative energy Error (E-E0)/E0,  relative angular momentum Error (L-L0)/L0
//The kernel is launched with 1 Block with Bl threads. Bl is equal to NB, Bl2 is the minimum of 2*NB and 64. NB is the next bigger number of N
//which is a power of two.
//
//Authors: Simon Grimm, Joachim Stadel
//March 2014
// ****************************************
template < int Bl, int Bl2>
__global__ void kineticEnergy32_kernel(double4 *x4_d, double4 *v4_d, double3 *spin_d, double *Energy_d, double Msun, double *U_d, double *LI_d, double *test_d, double *Energy0_d, double *LI0_d, int st, int N, int E){

	int idy = threadIdx.x;

	__shared__ double T_s[Bl2];
	__shared__ double V_s[Bl2];
	__shared__ double3 p_s[Bl2];
	__shared__ double3 s_s[Bl2];
	__shared__ double E_s[Bl2];
	__shared__ double3 L_s[Bl2];

	__shared__ double4 v4_s[Bl];
	__shared__ double4 x4_s[Bl];
	__shared__ double3 spin_s[Bl];

	T_s[idy] = 0.0;
	V_s[idy] = 0.0;
	E_s[idy] = 0.0;
	L_s[idy].x = 0.0;
	L_s[idy].y = 0.0;
	L_s[idy].z = 0.0;

	if(idy < N){
		x4_s[idy] = x4_d[idy];
		v4_s[idy] = v4_d[idy];
		spin_s[idy] = spin_d[idy];
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
		spin_s[idy].x = 0.0;
		spin_s[idy].y = 0.0;
		spin_s[idy].z = 0.0;
	}

	__syncthreads();

	if(x4_d[idy].w > 0 && idy < N){
		s_s[idy].x = x4_d[idy].w * x4_s[idy].x;
		s_s[idy].y = x4_d[idy].w * x4_s[idy].y;
		s_s[idy].z = x4_d[idy].w * x4_s[idy].z;
		p_s[idy].x = x4_d[idy].w * v4_s[idy].x;
		p_s[idy].y = x4_d[idy].w * v4_s[idy].y;
		p_s[idy].z = x4_d[idy].w * v4_s[idy].z;
	}
	else{
		s_s[idy].x = 0.0;
		s_s[idy].y = 0.0;
		s_s[idy].z = 0.0;
		p_s[idy].x = 0.0;
		p_s[idy].y = 0.0;
		p_s[idy].z = 0.0;
	}

	if(Bl <= 32){
		T_s[idy + Bl] = 0.0;
		V_s[idy + Bl] = 0.0;
		E_s[idy + Bl] = 0.0;
		L_s[idy + Bl].x = 0.0;
		L_s[idy + Bl].y = 0.0;
		L_s[idy + Bl].z = 0.0;

		p_s[idy + Bl].x = 0.0;
		p_s[idy + Bl].y = 0.0;
		p_s[idy + Bl].z = 0.0;
		s_s[idy + Bl].x = 0.0;
		s_s[idy + Bl].y = 0.0;
		s_s[idy + Bl].z = 0.0;
	}

	__syncthreads();

	if(idy < 32){
		volatile double3 *p = p_s;
		volatile double3 *s = s_s;
		if(Bl >= 64) s[idy].x += s[idy + 32].x;
		if(Bl >= 32) s[idy].x += s[idy + 16].x;
		if(Bl >= 16) s[idy].x += s[idy + 8].x;
		if(Bl >= 8) s[idy].x += s[idy + 4].x;
		if(Bl >= 4) s[idy].x += s[idy + 2].x;
		if(Bl >= 2) s[idy].x += s[idy + 1].x;

		if(Bl >= 64) s[idy].y += s[idy + 32].y;
		if(Bl >= 32) s[idy].y += s[idy + 16].y;
		if(Bl >= 16) s[idy].y += s[idy + 8].y;
		if(Bl >= 8) s[idy].y += s[idy + 4].y;
		if(Bl >= 4) s[idy].y += s[idy + 2].y;
		if(Bl >= 2) s[idy].y += s[idy + 1].y;

		if(Bl >= 64) s[idy].z += s[idy + 32].z;
		if(Bl >= 32) s[idy].z += s[idy + 16].z;
		if(Bl >= 16) s[idy].z += s[idy + 8].z;
		if(Bl >= 8) s[idy].z += s[idy + 4].z;
		if(Bl >= 4) s[idy].z += s[idy + 2].z;
		if(Bl >= 2) s[idy].z += s[idy + 1].z;

		if(Bl >= 64) p[idy].x += p[idy + 32].x;
		if(Bl >= 32) p[idy].x += p[idy + 16].x;
		if(Bl >= 16) p[idy].x += p[idy + 8].x;
		if(Bl >= 8) p[idy].x += p[idy + 4].x;
		if(Bl >= 4) p[idy].x += p[idy + 2].x;
		if(Bl >= 2) p[idy].x += p[idy + 1].x;

		if(Bl >= 64) p[idy].y += p[idy + 32].y;
		if(Bl >= 32) p[idy].y += p[idy + 16].y;
		if(Bl >= 16) p[idy].y += p[idy + 8].y;
		if(Bl >= 8) p[idy].y += p[idy + 4].y;
		if(Bl >= 4) p[idy].y += p[idy + 2].y;
		if(Bl >= 2) p[idy].y += p[idy + 1].y;

		if(Bl >= 64) p[idy].z += p[idy + 32].z;
		if(Bl >= 32) p[idy].z += p[idy + 16].z;
		if(Bl >= 16) p[idy].z += p[idy + 8].z;
		if(Bl >= 8) p[idy].z += p[idy + 4].z;
		if(Bl >= 4) p[idy].z += p[idy + 2].z;
		if(Bl >= 2) p[idy].z += p[idy + 1].z;
	}
	__syncthreads();
	if(idy < N){
		V_s[idy] = Energy_d[idy];
		if(x4_s[idy].w > 0){
			T_s[idy] = 0.5 * x4_s[idy].w * (v4_s[idy].x * v4_s[idy].x + v4_s[idy].y * v4_s[idy].y + v4_s[idy].z * v4_s[idy].z);
		}
		E_s[idy] =  V_s[idy] + T_s[idy];
test_d[idy] = E_s[idy];
	}
	__syncthreads();
	//convert to barycentric positions
	if(idy < N){
		x4_s[idy].x -= s_s[0].x / Msun;
		x4_s[idy].y -= s_s[0].y / Msun;
		x4_s[idy].z -= s_s[0].z / Msun;
		L_s[idy].x = x4_s[idy].w * (x4_s[idy].y * v4_s[idy].z - x4_s[idy].z * v4_s[idy].y) + spin_s[idy].x;
		L_s[idy].y = x4_s[idy].w * (x4_s[idy].z * v4_s[idy].x - x4_s[idy].x * v4_s[idy].z) + spin_s[idy].y;
		L_s[idy].z = x4_s[idy].w * (x4_s[idy].x * v4_s[idy].y - x4_s[idy].y * v4_s[idy].x) + spin_s[idy].z;
	}
	if(idy < 32){
		volatile double *T = T_s;
		volatile double *V = V_s;
		volatile double *Es = E_s;
		volatile double3 *L = L_s;
		if(Bl >= 64) T[idy] += T[idy + 32];
		if(Bl >= 32) T[idy] += T[idy + 16];
		if(Bl >= 16) T[idy] += T[idy + 8];
		if(Bl >= 8) T[idy] += T[idy + 4];
		if(Bl >= 4) T[idy] += T[idy + 2];
		if(Bl >= 2) T[idy] += T[idy + 1];

		if(Bl >= 64) V[idy] += V[idy + 32];
		if(Bl >= 32) V[idy] += V[idy + 16];
		if(Bl >= 16) V[idy] += V[idy + 8];
		if(Bl >= 8) V[idy] += V[idy + 4];
		if(Bl >= 4) V[idy] += V[idy + 2];
		if(Bl >= 2) V[idy] += V[idy + 1];

		if(Bl >= 64) Es[idy] += Es[idy + 32];
		if(Bl >= 32) Es[idy] += Es[idy + 16];
		if(Bl >= 16) Es[idy] += Es[idy + 8];
		if(Bl >= 8) Es[idy] += Es[idy + 4];
		if(Bl >= 4) Es[idy] += Es[idy + 2];
		if(Bl >= 2) Es[idy] += Es[idy + 1];

		if(Bl >= 64) L[idy].x += L[idy + 32].x;
		if(Bl >= 32) L[idy].x += L[idy + 16].x;
		if(Bl >= 16) L[idy].x += L[idy + 8].x;
		if(Bl >= 8) L[idy].x += L[idy + 4].x;
		if(Bl >= 4) L[idy].x += L[idy + 2].x;
		if(Bl >= 2) L[idy].x += L[idy + 1].x;

		if(Bl >= 64) L[idy].y += L[idy + 32].y;
		if(Bl >= 32) L[idy].y += L[idy + 16].y;
		if(Bl >= 16) L[idy].y += L[idy + 8].y;
		if(Bl >= 8) L[idy].y += L[idy + 4].y;
		if(Bl >= 4) L[idy].y += L[idy + 2].y;
		if(Bl >= 2) L[idy].y += L[idy + 1].y;

		if(Bl >= 64) L[idy].z += L[idy + 32].z;
		if(Bl >= 32) L[idy].z += L[idy + 16].z;
		if(Bl >= 16) L[idy].z += L[idy + 8].z;
		if(Bl >= 8) L[idy].z += L[idy + 4].z;
		if(Bl >= 4) L[idy].z += L[idy + 2].z;
		if(Bl >= 2) L[idy].z += L[idy + 1].z;
	}

	__syncthreads();
	if(idy == 0){
		volatile double Tsun = 0.5 / Msun * (p_s[0].x * p_s[0].x + p_s[0].y * p_s[0].y + p_s[0].z * p_s[0].z);
		//Lsun
		L_s[0].x += s_s[0].y * p_s[0].z - s_s[0].z * p_s[0].y;
		L_s[0].y += s_s[0].z * p_s[0].x - s_s[0].x * p_s[0].z;
		L_s[0].z += s_s[0].x * p_s[0].y - s_s[0].y * p_s[0].x;
		volatile double Ltot = sqrt(L_s[0].x * L_s[0].x + L_s[0].y * L_s[0].y + L_s[0].z * L_s[0].z);
		V_s[0] *= Kg;
		T_s[0] *= Kg;
		E_s[0] *= Kg;
		Tsun *= Kg;
		Energy_d[0] = V_s[0];
		Energy_d[1] = T_s[0] + Tsun;
		Energy_d[2] = LI_d[st] * Kg;
		Energy_d[3] = U_d[st] * Kg;
		Energy_d[4] = T_s[0] + V_s[0] + __dmul_rn(U_d[st], Kg) + Tsun;
		Energy_d[5] = (Ltot + LI_d[st]) * Kg;
		if(E == 0){
			Energy0_d[st] = T_s[0] + V_s[0] + __dmul_rn(U_d[st], Kg) + Tsun;
			Energy_d[7] = 0.0;
			LI0_d[st] = (Ltot + LI_d[st]) * Kg;
			Energy_d[6] = 0.0;
		}
		if(E == 1){
			Energy_d[7] = ((T_s[0] + V_s[0] + __dmul_rn(U_d[st], Kg) + Tsun) - Energy0_d[st]) / Energy0_d[st];
			Energy_d[6] = ((Ltot + LI_d[st]) * Kg - LI0_d[st]) / LI0_d[st];
		}
	}
}

// *************************************
//This function calls the Energy kernels
//
//Authors: Simon Grimm, Joachim Stadel
//March 2014
// *************************************3
__host__ void Data::EnergyCall(int NB, double4 *x4_d, double4 *v4_d, double3 *spin_d, double Msun, double* Energy_d, double *test_d, double *U_d, double *LI_d, double *Energy0_d, double *LI0_d, cudaStream_t hstream, int st, int N, int E){
	switch(NB){
		case 16:{
			potentialEnergy32_kernel <16, 32> <<< NB, 16, 0, hstream>>> (x4_d, v4_d, Msun, Energy_d, test_d, st, N);
			kineticEnergy32_kernel <16, 32> <<< 1, 16, 0, hstream>>> (x4_d, v4_d, spin_d, Energy_d, Msun, U_d, LI_d, test_d, Energy0_d, LI0_d, st, N, E);
		};
		break;
		case 32:{
			potentialEnergy32_kernel <32, 64> <<< NB, 32, 0, hstream>>> (x4_d, v4_d, Msun, Energy_d, test_d, st, N);
			kineticEnergy32_kernel <32, 64> <<< 1, 32, 0, hstream>>> (x4_d, v4_d, spin_d, Energy_d, Msun, U_d, LI_d, test_d, Energy0_d, LI0_d, st, N, E);
		};
		break;
		case 64:{
			potentialEnergy64_kernel <64> <<< NB, 64, 0, hstream>>> (x4_d, v4_d, Msun, Energy_d, test_d, st, N);
			kineticEnergy32_kernel <64, 64> <<< 1, 64, 0, hstream>>> (x4_d, v4_d, spin_d, Energy_d, Msun, U_d, LI_d, test_d, Energy0_d, LI0_d, st, N, E);
		};
		break;
		case 128:{
			potentialEnergy64_kernel <128> <<< NB, 128,  0, hstream>>> (x4_d, v4_d, Msun, Energy_d, test_d, st, N);
			kineticEnergy128_kernel <128> <<< 1, 128, 0, hstream>>> (x4_d, v4_d, spin_d, Energy_d, Msun, U_d, LI_d, test_d, Energy0_d, LI0_d, st, N, E);
		};
		break;
		case 256:{
			potentialEnergy64_kernel <256> <<< NB, 256, 0, hstream>>> (x4_d, v4_d, Msun, Energy_d, test_d, st, N);
			kineticEnergy128_kernel <256> <<< 1, 256, 0, hstream>>> (x4_d, v4_d, spin_d, Energy_d, Msun, U_d, LI_d, test_d, Energy0_d, LI0_d, st, N, E);
		};
		break;
		case 512:{
			potentialEnergy64_kernel <512> <<< NB, 512, 0, hstream>>> (x4_d, v4_d, Msun, Energy_d, test_d, st, N);
			kineticEnergy128_kernel <256> <<< 1, 256, 0, hstream>>> (x4_d, v4_d, spin_d, Energy_d, Msun, U_d, LI_d, test_d, Energy0_d, LI0_d, st, N, E);
		};
		break;
		case 1024:{
			potentialEnergy64_kernel <512> <<< NB, 512, 0, hstream>>> (x4_d, v4_d, Msun, Energy_d, test_d, st, N);
			kineticEnergy128_kernel <256> <<< 1, 256, 0, hstream>>> (x4_d, v4_d, spin_d, Energy_d, Msun, U_d, LI_d, test_d, Energy0_d, LI0_d, st, N, E);
		};
		break;
		case 2048:{
			potentialEnergy64_kernel <512> <<< NB, 512, 0, hstream>>> (x4_d, v4_d, Msun, Energy_d, test_d, st, N);
			kineticEnergy128_kernel <256> <<< 1 ,256, 0, hstream>>> (x4_d, v4_d, spin_d, Energy_d, Msun, U_d, LI_d, test_d, Energy0_d, LI0_d, st, N, E);
		};
		break;
	}
	if(NB > 2048){
		potentialEnergy64_kernel <512> <<< NB, 512, 0, hstream>>> (x4_d, v4_d, Msun, Energy_d, test_d, st, N);
		kineticEnergy128_kernel <256> <<< 1 ,256, 0, hstream>>> (x4_d, v4_d, spin_d, Energy_d, Msun, U_d, LI_d, test_d, Energy0_d, LI0_d, st, N, E);

	}
}
// *************************************
//This function calls the EjectionEnergy kernels
//
//Authors: Simon Grimm, Joachim Stadel
//March 2014
// *************************************3
__host__ void Data::EjectionEnergyCall(int NB, double4 *x4_d, double4 *v4_d, double3 *spin_d, double Msun, int i, double *U_d, double *LI_d, double3 *vcom_d, int N){
	switch(NB){
		case 16:{
			EjectionEnergy32_kernel <16, 32> <<<1, 16>>> (x4_d, v4_d, spin_d, Msun, i, U_d, LI_d, vcom_d, N);
		};
		break;
		case 32:{
			EjectionEnergy32_kernel <32, 64> <<<1, 32>>> (x4_d, v4_d, spin_d, Msun, i, U_d, LI_d, vcom_d, N);
		};
		break;
		case 64:{
			EjectionEnergy64_kernel <64> <<<1, 64>>> (x4_d, v4_d, spin_d, Msun, i, U_d, LI_d, vcom_d, N);
		};
		break;
		case 128:{
			EjectionEnergy64_kernel <128> <<<1, 128>>> (x4_d, v4_d, spin_d, Msun, i, U_d, LI_d, vcom_d, N);
		};
		break;
		case 256:{
			EjectionEnergy64_kernel <256> <<<1, 256>>> (x4_d, v4_d, spin_d, Msun, i, U_d, LI_d, vcom_d, N);
		};
		break;
		case 512:{
			EjectionEnergy64_kernel <512> <<<1, 512>>> (x4_d, v4_d, spin_d, Msun, i, U_d, LI_d, vcom_d, N);
		};
		break;
		case 1024:{
			EjectionEnergy64_kernel <512> <<<1, 512>>> (x4_d, v4_d, spin_d, Msun, i, U_d, LI_d, vcom_d, N);
		};
		break;
		case 2048:{
			EjectionEnergy64_kernel <512> <<<1, 512>>> (x4_d, v4_d, spin_d, Msun, i, U_d, LI_d, vcom_d, N);
		};
	}
	if(NB > 2048){
		EjectionEnergy64_kernel <512> <<<1, 512>>> (x4_d, v4_d, spin_d, Msun, i, U_d, LI_d, vcom_d, N);
	}
}
__host__ void Data::EjectionEnergysmallCall(double4 *v4small_d, int Nsmall, double3 *vcom_d){
	EjectionEnergysmall1_kernel <<< (Nsmall + 127) / 128, 128 >>> (v4small_d, Nsmall, vcom_d);
	
}
__host__ void Data::EjectionEnergysmall2Call(double4 *x4small_d, int i){
	EjectionEnergysmall2_kernel <<< 1, 1 >>> (x4small_d, i);
}


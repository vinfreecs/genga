#include "define.h"

// **************************************
//This Kernels performs the Sun-Kick 1/Msun * Sum(p_i)^2 on all the bodies.
//It uses a parallel reduction fomula to calculate the sum in log(N) steps.
//
//It works for the case of more than 65 bodies.
//Each Kernel is launched with 3 blocks, one for each dimension.
//
//c = 1 : perform C Kick.
//c = 2 : perform C Kick + reset Nencpairs + update time.
//c = 3 : perform C Kick + reset Nencpairs
//
//Authors: Simon Grimm, Joachim Stadel
//March 2014
//*****************************************
template <int Bl, int NB, int E>
__global__ void HC128_kernel(double4 *x4_d, double4 *v4_d, const double dti2Msun, int *Nencpairs_d, int *Nencpairs2_d, int *Nenc_d, int N, double t){

	int idy = threadIdx.x;
	int idx = blockIdx.x;

        if(E == 1){
                if(idy == 0 && idx == 0){
                        Nencpairs2_d[0] = 0;      //this variable is needed in the Encounter kernel
                }
                if(idy < 12 && idx == 0) Nenc_d[idy] = 0;
        }
        if(E == 2){
                if(idy == 0 && idx == 0){
                        Nencpairs_d[0] = 0;	//This variable is needed in the Kick_kernel
                }
        }
        if(E == 3){
                if(idy == 0 && idx == 0){
                        Nencpairs_d[0] = 0;     //This variable is needed in the Kick_kernel
                }
        }

	__shared__  double a1_s[Bl];

	a1_s[idy] = 0.0;

	__syncthreads(); 
	for (int i = 0; i < NB ; i+= Bl){
		if(x4_d[idy + i].w > 0 && idy < N){
			if(idx == 0){
				a1_s[idy] += x4_d[idy + i].w * v4_d[idy + i].x;
			}
			if(idx == 1){
				a1_s[idy] += x4_d[idy + i].w * v4_d[idy + i].y;
			}
			if(idx == 2){
				a1_s[idy] += x4_d[idy + i].w * v4_d[idy + i].z;
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
	for(int i = 0; i <NB; i +=  Bl){
		if(idy < N){
			if(idx == 0) x4_d[idy + i].x += __dmul_rn(a1_s[0], dti2Msun);
			if(idx == 1) x4_d[idy + i].y += __dmul_rn(a1_s[0], dti2Msun);
			if(idx == 2) x4_d[idy + i].z += __dmul_rn(a1_s[0], dti2Msun);
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
//c = 1 : perform C Kick.
//c = 2 : perform C Kick + reset Nencpairs + update time.
//c = 3 : perform C Kick + reset Nencpairs
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
//  *****************************************
template < int Bl, int Bl2, int E>
__global__ void HC32_kernel(double4 *x4_d, double4 *v4_d, const double dti2Msun, int *Nencpairs_d, int *Nencpairs2_d, int *Nenc_d, int N){

	int idy = threadIdx.x;
	int idx = blockIdx.x;
        if(E == 1){
                if(idy == 0 && idx == 0){
                        Nencpairs2_d[0] = 0;      //this variable is needed in the Encounter kernel
                }
                if(idy < 12 && idx == 0) Nenc_d[idy] = 0;
        }
	if(E == 2){
		if(idy == 0 && idx == 0){
			Nencpairs_d[0] = 0;		//This variable is needed in the Kick_kernel
		}
	}
        if(E == 3){
                if(idy == 0 && idx == 0){
                        Nencpairs_d[0] = 0;     //This variable is needed in the Kick_kernel
                }
        }
	__shared__ double a1_s[Bl2];

	if(x4_d[idy].w > 0.0 && idy < N){
		if(idx == 0){
			a1_s[idy] = x4_d[idy].w * v4_d[idy].x;
		}
		if(idx == 1){
			a1_s[idy] = x4_d[idy].w * v4_d[idy].y;
		}
		if(idx == 2){
			a1_s[idy] = x4_d[idy].w * v4_d[idy].z;
		}
	}
	else{
		a1_s[idy] = 0.0;
	}

	if(Bl <= 32){
		a1_s[idy + Bl] = 0.0;
	}

	__syncthreads();
	if(idy < 32){
		volatile double *a = a1_s;
		if(Bl >= 64) a[idy] += a[idy + 32];
		if(Bl >= 32) a[idy] += a[idy + 16];
		a[idy] += a[idy + 8];
		a[idy] += a[idy + 4];
		a[idy] += a[idy + 2];
		a[idy] += a[idy + 1];
	}

	__syncthreads();
	if(idy < N){
		if(idx == 0) x4_d[idy].x += a1_s[0] * dti2Msun;
		if(idx == 1) x4_d[idy].y += a1_s[0] * dti2Msun;
		if(idx == 2) x4_d[idy].z += a1_s[0] * dti2Msun;
	}
}

// **************************************
//This Kernels performs the Sun-Kick 1/Msun * Sum(p_i)^2 on all the bodies.
//It uses a parallel reduction fomula to calculate the sum in log(N) steps.
//Each Kernel is launched with 3 blocks, one for each dimension.
//
//c = 1 : perform C Kick.
//c = 2 : perform C Kick + reset Nencpairs + update time.
//c = 3 : perform C Kick + reset Nencpairs
//i
//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
// *****************************************/
template <int Bl, int E>
__global__ void HCsmall_kernel(double4 *x4_d, double4 *v4_d, const double dti2Msun, int *Nencpairs_d, int *Nencpairssmall_d, int *Nencpairs2_d, int *Nencpairssmall2_d, int *Nenc_d, int *Nencsmall_d, double4 *x4small_d, double4 *v4small_d, int Nsmall, int N){

	int idy = threadIdx.x;
	int idx = blockIdx.x;

        if(E == 1){
                if(idy == 0 && idx == 0){
                        Nencpairs2_d[0] = 0;      //this variable is needed in the Encounter kernel
			Nencpairssmall2_d[0] = 0;
                }
                if(idy < 12 && idx == 0){
			Nenc_d[idy] = 0;
			Nencsmall_d[idy] = 0;
		}
        }
        if(E == 2){
                if(idy == 0 && idx == 0){
                        Nencpairs_d[0] = 0;	//This variable is needed in the Kick_kernel
			Nencpairssmall_d[0] = 0;
                }
        }
        if(E == 3){
                if(idy == 0 && idx == 0){
                        Nencpairs_d[0] = 0;     //This variable is needed in the Kick_kernel
                }
        }

	__shared__  double a1_s[Bl];

	a1_s[idy] = 0.0;
	__syncthreads();

	for(int i = 0; i < N + Nsmall; i+= Bl){
		if(idy + i < N){
			if(x4_d[idy + i].w > 0){
				if(idx == 0) a1_s[idy] += x4_d[idy + i].w * v4_d[idy + i].x;
				if(idx == 1) a1_s[idy] += x4_d[idy + i].w * v4_d[idy + i].y;
				if(idx == 2) a1_s[idy] += x4_d[idy + i].w * v4_d[idy + i].z;
			}
		}
		else if(idy + i < Nsmall + N){
                        if(x4small_d[idy + i - N].w > 0){
                                if(idx == 0) a1_s[idy] += x4small_d[idy + i - N].w * v4small_d[idy + i - N].x;
                                if(idx == 1) a1_s[idy] += x4small_d[idy + i - N].w * v4small_d[idy + i - N].y;
                                if(idx == 2) a1_s[idy] += x4small_d[idy + i - N].w * v4small_d[idy + i - N].z;
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
  
	for(int i = 0; i < N + Nsmall; i +=  Bl){
		if(idy + i < N){
			if(idx == 0) x4_d[idy + i].x += a1_s[0] * dti2Msun;
			if(idx == 1) x4_d[idy + i].y += a1_s[0] * dti2Msun;
			if(idx == 2) x4_d[idy + i].z += a1_s[0] * dti2Msun;
		}
		else if(idy + i < N + Nsmall){
                        if(idx == 0) x4small_d[idy + i - N].x += a1_s[0] * dti2Msun;
                        if(idx == 1) x4small_d[idy + i - N].y += a1_s[0] * dti2Msun;
                        if(idx == 2) x4small_d[idy + i - N].z += a1_s[0] * dti2Msun;
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
//
//E = 1 : perform C Kick.
//E = 2 : perform C Kick + reset Nencpairs
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
//*****************************************
template <int Bl, int Bl2, int Nmax, int E>
__global__ void HCM2_kernel(double4 *x4_d, double4 *v4_d, const double *dti2Msun_d, int *index_d, int NT, double Ct, double *test_d, int *Nencpairs_d, int *Nencpairs2_d, int *Nenc_d, int Nst){

	int idy = threadIdx.x;
	int id = blockIdx.x * Bl2 + idy - Nmax;
	__shared__ volatile double3 p_s[Bl + Nmax / 2];
	__shared__ int st_s[Bl + Nmax / 2];
	volatile double dti2Msun;
	if(E == 1){
		if(id >= 0 && id < Nst + 1){
			Nencpairs2_d[id] = 0;           //This variable is needed in the Encounter_kernel
		}
		if(id >= 0 && id < 12){
			Nenc_d[id] = 0;
		}
	}
		
	if(E == 2){
		if(id < Nst + 1){
			Nencpairs_d[id] = 0;		//This variable is needed in the Kick_kernel
		}
	}
	if(id < NT && id >= 0){
		st_s[idy] = index_d[id] / 100;
		volatile double m = x4_d[id].w;
		p_s[idy].x = m * v4_d[id].x; //st_s[idy].x;
		p_s[idy].y = m * v4_d[id].y;
		p_s[idy].z = m * v4_d[id].z;
		dti2Msun = dti2Msun_d[st_s[idy]] * Ct;

	}
	else{
		st_s[idy] = -idy-1;
		p_s[idy].x = 0.0;
		p_s[idy].y = 0.0;
		p_s[idy].z = 0.0;
		dti2Msun = 0.0;
	}
	//halo
	if(idy < Nmax / 2){
		//right
		if(id + Bl < NT){
			st_s[idy + Bl] = index_d[id + Bl] / 100;
			volatile double m = x4_d[id + Bl].w;
			p_s[idy + Bl].x = m * v4_d[id + Bl].x; //st_s[idy + Bl].x;
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
		f = ((st_s[idy] - st_s[idy + 4]) == 0);            //one if sti == stj, zero else
		px = (p_s[idy + 4].x) * f;
		py = (p_s[idy + 4].y) * f;
		pz = (p_s[idy + 4].z) * f;

		__syncthreads();

		p_s[idy].x += px;
		p_s[idy].y += py;
		p_s[idy].z += pz;
	}

	__syncthreads();

	f = ((st_s[idy] - st_s[idy + 2]) == 0);            //one if sti == stj, zero else
	px = (p_s[idy + 2].x) * f;
	py = (p_s[idy + 2].y) * f;
	pz = (p_s[idy + 2].z) * f;

	__syncthreads();

	p_s[idy].x += px;
	p_s[idy].y += py;
	p_s[idy].z += pz;

	__syncthreads();

	f = ((st_s[idy] - st_s[idy + 1]) == 0);            //one if sti == stj, zero else
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


	if(id < NT && id >= 0 && idy >= Nmax && idy < Bl - Nmax / 2 && x4_d[id].w > 0){
		x4_d[id].x += p_s[idy].x * dti2Msun;
		x4_d[id].y += p_s[idy].y * dti2Msun;
		x4_d[id].z += p_s[idy].z * dti2Msun;
	}
}

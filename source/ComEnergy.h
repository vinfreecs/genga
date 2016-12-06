#ifndef COMENERGY_H
#define COMENERGY_H

// *********************************************************
//This kernel computes the kinetic energy of the center of mass
// it converts the velocities between heliocentric and democratic coordinates
//
//Author: Simon Grimm
//August 2016
// ***********************************************************
template < int Bl>
__global__ void com_kernel(double4 *x4_d, double4 *v4_d, double3 *vcom_d, double Msun, double *test_d, int N, int f){

	int idy = threadIdx.x;

	__shared__ double4 p_s[Bl];

	for(int i = 0; i < Bl; i += blockDim.x){
		p_s[idy + i].x = 0.0;
		p_s[idy + i].y = 0.0;
		p_s[idy + i].z = 0.0;
		p_s[idy + i].w = 0.0;
	}

	for(int i = 0; i < N; i += blockDim.x){
		if(idy + i < N){
			double m = x4_d[idy + i].w;
			if(m > 0.0){
				p_s[idy].x += m * v4_d[idy + i].x;
				p_s[idy].y += m * v4_d[idy + i].y;
				p_s[idy].z += m * v4_d[idy + i].z;
				p_s[idy].w += m;
			}
		}
	}
	__syncthreads();


	int s = blockDim.x/2;
	for(int i = 6; i < log2f(blockDim.x); ++i){
		if( idy < s ) {
			p_s[idy].x += p_s[idy + s].x;
			p_s[idy].y += p_s[idy + s].y;
			p_s[idy].z += p_s[idy + s].z;
			p_s[idy].w += p_s[idy + s].w;
		}
		__syncthreads();
		s /= 2;
	}

	if(idy < 32){
		volatile double4*p = p_s;
		if(blockDim.x >= 64) p[idy].x += p[idy + 32].x;
		if(blockDim.x >= 32) p[idy].x += p[idy + 16].x;
		if(blockDim.x >= 16) p[idy].x += p[idy + 8].x;
		if(blockDim.x >= 8) p[idy].x += p[idy + 4].x;
		if(blockDim.x >= 4) p[idy].x += p[idy + 2].x;
		if(blockDim.x >= 2) p[idy].x += p[idy + 1].x;

		if(blockDim.x >= 64) p[idy].y += p[idy + 32].y;
		if(blockDim.x >= 32) p[idy].y += p[idy + 16].y;
		if(blockDim.x >= 16) p[idy].y += p[idy + 8].y;
		if(blockDim.x >= 8) p[idy].y += p[idy + 4].y;
		if(blockDim.x >= 4) p[idy].y += p[idy + 2].y;
		if(blockDim.x >= 2) p[idy].y += p[idy + 1].y;

		if(blockDim.x >= 64) p[idy].z += p[idy + 32].z;
		if(blockDim.x >= 32) p[idy].z += p[idy + 16].z;
		if(blockDim.x >= 16) p[idy].z += p[idy + 8].z;
		if(blockDim.x >= 8) p[idy].z += p[idy + 4].z;
		if(blockDim.x >= 4) p[idy].z += p[idy + 2].z;
		if(blockDim.x >= 2) p[idy].z += p[idy + 1].z;

		if(blockDim.x >= 64) p[idy].w += p[idy + 32].w;
		if(blockDim.x >= 32) p[idy].w += p[idy + 16].w;
		if(blockDim.x >= 16) p[idy].w += p[idy + 8].w;
		if(blockDim.x >= 8) p[idy].w += p[idy + 4].w;
		if(blockDim.x >= 4) p[idy].w += p[idy + 2].w;
		if(blockDim.x >= 2) p[idy].w += p[idy + 1].w;
	}
	__syncthreads();

	double iMsun = 1.0 / Msun;

	if(idy == 0){
		if(f == 0){
			vcom_d[0].x = p_s[0].x;
			vcom_d[0].y = p_s[0].y;
			vcom_d[0].z = p_s[0].z;
		}
	}
	for(int i = 0; i < N; i += blockDim.x){
		if(idy + i < N){
			double m = x4_d[idy + i].w;
			if(m >= 0.0 && f == 1){
				//Convert to Heliocentric coordinates
				v4_d[idy + i].x += p_s[0].x * iMsun;
				v4_d[idy + i].y += p_s[0].y * iMsun;
				v4_d[idy + i].z += p_s[0].z * iMsun;
			}
			if(m >= 0.0 && f == -1){
				//Convert to Democratic coordinates
				double iMsunp = 1.0 / (Msun + p_s[0].w);
				v4_d[idy + i].x -= p_s[0].x * iMsunp;
				v4_d[idy + i].y -= p_s[0].y * iMsunp;
				v4_d[idy + i].z -= p_s[0].z * iMsunp;
			}
		}
	}
}

// *********************************************************
// This kernel computes the kinetic energy of the center of mass
//
// Authors: Simon Grimm, Joachim Stadel
// October  2015
// ***********************************************************
template <int Bl, int Bl2, int Nmax >
__global__ void comM_kernel(double4 *x4_d, double4 *v4_d, double3 *vcom_d, const double4 *Msun_d, int *index_d, int *NBS_d, int NT, double *test_d, int ff){

	int idy = threadIdx.x;
	int id = blockIdx.x * Bl2 + idy - Nmax;
	__shared__ volatile double4 p_s[Bl + Nmax / 2];
	__shared__ int st_s[Bl + Nmax / 2];
	volatile double Msun;
	int NBS;

	if(id < NT && id >= 0){
		st_s[idy] = index_d[id] / 100;
		volatile double m = x4_d[id].w;
		p_s[idy].x = m * v4_d[id].x;
		p_s[idy].y = m * v4_d[id].y;
		p_s[idy].z = m * v4_d[id].z;
		p_s[idy].w = m;
		Msun = Msun_d[st_s[idy]].x;
		NBS = NBS_d[st_s[idy]];

	}
	else{
		st_s[idy] = -idy-1;
		p_s[idy].x = 0.0;
		p_s[idy].y = 0.0;
		p_s[idy].z = 0.0;
		p_s[idy].w = 0.0;
		Msun = 0.0;
		NBS = -1;
	}
	//halo
	if(idy < Nmax / 2){
		//right
		if(id + Bl < NT){
			st_s[idy + Bl] = index_d[id + Bl] / 100;
			volatile double m = x4_d[id + Bl].w;
			p_s[idy + Bl].x = m * v4_d[id + Bl].x;
			p_s[idy + Bl].y = m * v4_d[id + Bl].y;
			p_s[idy + Bl].z = m * v4_d[id + Bl].z;
			p_s[idy + Bl].w = m;
		}
		else{
			st_s[idy + Bl] = -idy-Bl-1;
			p_s[idy + Bl].x = 0.0;
			p_s[idy + Bl].y = 0.0;
			p_s[idy + Bl].z = 0.0;
			p_s[idy + Bl].w = 0.0;
		}
	}

	volatile int f;
	volatile double px;
	volatile double py;
	volatile double pz;
	volatile double pw;	
	if(Nmax >= 64){
		__syncthreads();
		f = ((st_s[idy] - st_s[idy + 32]) == 0);		//one if sti == stj, zero else
		px = (p_s[idy + 32].x) * f;	
		py = (p_s[idy + 32].y) * f;
		pz = (p_s[idy + 32].z) * f;
		pw = (p_s[idy + 32].w) * f;

		__syncthreads();
	
		p_s[idy].x += px;
		p_s[idy].y += py;
		p_s[idy].z += pz;
		p_s[idy].w += pw;
	}

	if(Nmax >= 32){
		__syncthreads();
		f = ((st_s[idy] - st_s[idy + 16]) == 0);		//one if sti == stj, zero else
		px = (p_s[idy + 16].x) * f;	
		py = (p_s[idy + 16].y) * f;
		pz = (p_s[idy + 16].z) * f;
		pw = (p_s[idy + 16].w) * f;

		__syncthreads();
	
		p_s[idy].x += px;
		p_s[idy].y += py;
		p_s[idy].z += pz;
		p_s[idy].w += pw;
	}
	if(Nmax >= 16){
		__syncthreads();
		f = ((st_s[idy] - st_s[idy + 8]) == 0);		//one if sti == stj, zero else
		px = (p_s[idy + 8].x) * f;	
		py = (p_s[idy + 8].y) * f;
		pz = (p_s[idy + 8].z) * f;
		pw = (p_s[idy + 8].w) * f;

		__syncthreads();
	
		p_s[idy].x += px;
		p_s[idy].y += py;
		p_s[idy].z += pz;
		p_s[idy].w += pw;
	}

	if(Nmax >= 8){
		__syncthreads();
		f = ((st_s[idy] - st_s[idy + 4]) == 0);		//one if sti == stj, zero else
		px = (p_s[idy + 4].x) * f;
		py = (p_s[idy + 4].y) * f;
		pz = (p_s[idy + 4].z) * f;
		pw = (p_s[idy + 4].w) * f;

		__syncthreads();

		p_s[idy].x += px;
		p_s[idy].y += py;
		p_s[idy].z += pz;
		p_s[idy].w += pw;
	}

	__syncthreads();

	f = ((st_s[idy] - st_s[idy + 2]) == 0);			//one if sti == stj, zero else
	px = (p_s[idy + 2].x) * f;
	py = (p_s[idy + 2].y) * f;
	pz = (p_s[idy + 2].z) * f;
	pw = (p_s[idy + 2].w) * f;

	__syncthreads();

	p_s[idy].x += px;
	p_s[idy].y += py;
	p_s[idy].z += pz;
	p_s[idy].w += pw;

	__syncthreads();

	f = ((st_s[idy] - st_s[idy + 1]) == 0);			//one if sti == stj, zero else
	px = (p_s[idy + 1].x) * f;
	py = (p_s[idy + 1].y) * f;
	pz = (p_s[idy + 1].z) * f;
	pw = (p_s[idy + 1].w) * f;

	__syncthreads();

	p_s[idy].x += px;
	p_s[idy].y += py;
	p_s[idy].z += pz;
	p_s[idy].w += pw;

	__syncthreads();
	//sum is complete, now distribute solution
	f = ((st_s[idy] - st_s[idy + 1]) == 0);
	px = (p_s[idy].x) * f + (1 - f) * p_s[idy + 1].x;
	py = (p_s[idy].y) * f + (1 - f) * p_s[idy + 1].y;
	pz = (p_s[idy].z) * f + (1 - f) * p_s[idy + 1].z;
	pw = (p_s[idy].w) * f + (1 - f) * p_s[idy + 1].w;

	__syncthreads();
	p_s[idy + 1].x = px;
	p_s[idy + 1].y = py;
	p_s[idy + 1].z = pz;
	p_s[idy + 1].w = pw;
	__syncthreads();

	f = ((st_s[idy] - st_s[idy + 2]) == 0);
	px = (p_s[idy].x) * f + (1 - f) * p_s[idy + 2].x;
	py = (p_s[idy].y) * f + (1 - f) * p_s[idy + 2].y;
	pz = (p_s[idy].z) * f + (1 - f) * p_s[idy + 2].z;
	pw = (p_s[idy].w) * f + (1 - f) * p_s[idy + 2].w;

	__syncthreads();
	p_s[idy + 2].x = px;
	p_s[idy + 2].y = py;
	p_s[idy + 2].z = pz;
	p_s[idy + 2].w = pw;
	__syncthreads();

	if(Nmax >= 8){
		f = ((st_s[idy] - st_s[idy + 4]) == 0);
		px = (p_s[idy].x) * f + (1 - f) * p_s[idy + 4].x;
		py = (p_s[idy].y) * f + (1 - f) * p_s[idy + 4].y;
		pz = (p_s[idy].z) * f + (1 - f) * p_s[idy + 4].z;
		pw = (p_s[idy].w) * f + (1 - f) * p_s[idy + 4].w;

		__syncthreads();
		p_s[idy + 4].x = px;
		p_s[idy + 4].y = py;
		p_s[idy + 4].z = pz;
		p_s[idy + 4].w = pw;
		__syncthreads();
	}

	if(Nmax >= 16){
		f = ((st_s[idy] - st_s[idy + 8]) == 0);
		px = (p_s[idy].x) * f + (1 - f) * p_s[idy + 8].x;
		py = (p_s[idy].y) * f + (1 - f) * p_s[idy + 8].y;
		pz = (p_s[idy].z) * f + (1 - f) * p_s[idy + 8].z;
		pw = (p_s[idy].w) * f + (1 - f) * p_s[idy + 8].w;

		__syncthreads();
		p_s[idy + 8].x = px;
		p_s[idy + 8].y = py;
		p_s[idy + 8].z = pz;
		p_s[idy + 8].w = pw;
		__syncthreads();
	}

	if(Nmax >= 32){
		f = ((st_s[idy] - st_s[idy + 16]) == 0);
		px = (p_s[idy].x) * f + (1 - f) * p_s[idy + 16].x;
		py = (p_s[idy].y) * f + (1 - f) * p_s[idy + 16].y;
		pz = (p_s[idy].z) * f + (1 - f) * p_s[idy + 16].z;
		pw = (p_s[idy].w) * f + (1 - f) * p_s[idy + 16].w;

		__syncthreads();
		p_s[idy + 16].x = px;
		p_s[idy + 16].y = py;
		p_s[idy + 16].z = pz;
		p_s[idy + 16].w = pw;
		__syncthreads();
	}

	if(Nmax >= 64){
		f = ((st_s[idy] - st_s[idy + 32]) == 0);
		px = (p_s[idy].x) * f + (1 - f) * p_s[idy + 32].x;
		py = (p_s[idy].y) * f + (1 - f) * p_s[idy + 32].y;
		pz = (p_s[idy].z) * f + (1 - f) * p_s[idy + 32].z;
		pw = (p_s[idy].w) * f + (1 - f) * p_s[idy + 32].w;

		__syncthreads();
		p_s[idy + 32].x = px;
		p_s[idy + 32].y = py;
		p_s[idy + 32].z = pz;
		p_s[idy + 32].w = pw;
		__syncthreads();
	}

	double iMsun = 1.0 / Msun;
	//now the sum is complete
	if(id == NBS && NBS >= 0 && idy >= Nmax && idy < Bl - Nmax / 2){
		if(ff == 0){
			vcom_d[st_s[idy]].x = p_s[idy].x;
			vcom_d[st_s[idy]].y = p_s[idy].y;
			vcom_d[st_s[idy]].z = p_s[idy].z;
		}
	}
	if(id < NT && id >= 0 && idy >= Nmax && idy < Bl - Nmax / 2 && x4_d[id].w >= 0.0 && ff == 1){
		v4_d[id].x += p_s[idy].x * iMsun;
		v4_d[id].y += p_s[idy].y * iMsun;
		v4_d[id].z += p_s[idy].z * iMsun;
	}
	if(id < NT && id >= 0 && idy >= Nmax && idy < Bl - Nmax / 2 && x4_d[id].w >= 0.0 && ff == -1){
		double iMsunp = 1.0 / (Msun + p_s[idy].w);
		v4_d[id].x -= p_s[idy].x * iMsunp;
		v4_d[id].y -= p_s[idy].y * iMsunp;
		v4_d[id].z -= p_s[idy].z * iMsunp;
	}
}
#endif

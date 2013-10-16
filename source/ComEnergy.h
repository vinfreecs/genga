//This kernel computes the kinet energy of the center of mass
template < int Bl, int Bl2>
__global__ void com32_kernel(double4 *x4_d, double4 *v4_d, double* U_d, double Msun, double *test_d, int N, int f){

        int idy = threadIdx.x;

        __shared__ double3 p_s[Bl2];
        __shared__ double4 v4_s[Bl];

        if(idy < N){
                v4_s[idy] = v4_d[idy];
        }
        else{
                v4_s[idy].x = 0.0;
                v4_s[idy].y = 0.0;
                v4_s[idy].z = 0.0;
                v4_s[idy].w = 0.0;
        }

	double m = x4_d[idy].w;

        __syncthreads();

        if(m > 0 && idy < N){
                p_s[idy].x = m * v4_s[idy].x;
                p_s[idy].y = m * v4_s[idy].y;
                p_s[idy].z = m * v4_s[idy].z;
        }
        else{
                p_s[idy].x = 0.0;
                p_s[idy].y = 0.0;
                p_s[idy].z = 0.0;
        }

        if(Bl <= 32){
                p_s[idy + Bl].x = 0.0;
                p_s[idy + Bl].y = 0.0;
                p_s[idy + Bl].z = 0.0;
        }

        __syncthreads();
        if(idy < 32){
                volatile double3 *p = p_s;
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
	if(idy == 0){
                volatile double Tsun = 0.5 / Msun * ( p_s[0].x*p_s[0].x + p_s[0].y*p_s[0].y + p_s[0].z*p_s[0].z);
//printf("Tsun %.40g %d\n", Tsun, f);
		U_d[0] += f * Tsun;	
	}
}
template < int NB, int Bl>
__global__ void com128_kernel(double4 *x4_d, double4 *v4_d, double* U_d, double Msun, double *test_d, int N, int f){

	int idy = threadIdx.x;

        __shared__ double3 p_s[Bl];

        p_s[idy].x = 0.0;
        p_s[idy].y = 0.0;
        p_s[idy].z = 0.0;

        for(int i = 0; i < NB ;i += Bl){
		double m = x4_d[idy + i].w;
                if(m > 0 && idy + i < N){
                        p_s[idy].x += m * v4_d[idy + i].x;
                        p_s[idy].y += m * v4_d[idy + i].y;
                        p_s[idy].z += m * v4_d[idy + i].z;
                }
        }
        __syncthreads();

        if(Bl >= 512){
                if(idy < 256){
                        p_s[idy].x += p_s[idy + 256].x;
                        p_s[idy].y += p_s[idy + 256].y;
                        p_s[idy].z += p_s[idy + 256].z;
                }
        }
        __syncthreads();

        if(Bl >= 256){
                if(idy < 128){
                        p_s[idy].x += p_s[idy + 128].x;
                        p_s[idy].y += p_s[idy + 128].y;
                        p_s[idy].z += p_s[idy + 128].z;
                }
        }
        __syncthreads();

        if(Bl >= 128){
                if(idy < 64){
                        p_s[idy].x += p_s[idy + 64].x;
                        p_s[idy].y += p_s[idy + 64].y;
                        p_s[idy].z += p_s[idy + 64].z;
                }
        }
        __syncthreads();
	if(idy < 32){
                volatile double3 *p = p_s;
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
        if(idy == 0){
                volatile double Tsun = 0.5 / Msun * ( p_s[0].x*p_s[0].x + p_s[0].y*p_s[0].y + p_s[0].z*p_s[0].z);
//printf("Tsun %.40g %d\n", Tsun, f);
		U_d[0] += f * Tsun;
	}
}

template <int Bl, int Bl2, int Nmax >
__global__ void comM_kernel(double4 *x4_d, double4 *v4_d, const double *Msun_d, double *U_d, int *index_d, int NT, double *test_d, int ff){

	int idy = threadIdx.x;
	int id = blockIdx.x * Bl2 + idy - Nmax;
	__shared__ volatile double3 p_s[Bl + Nmax / 2];
	__shared__ int st_s[Bl + Nmax / 2];
	volatile double Msun;
	int index;

	if(id < NT && id >= 0){
		st_s[idy] = index_d[id] / 100;
		volatile double m = x4_d[id].w;
		p_s[idy].x = m * v4_d[id].x;
		p_s[idy].y = m * v4_d[id].y;
		p_s[idy].z = m * v4_d[id].z;
		Msun = Msun_d[st_s[idy]];
		index = index_d[id] % 100;

	}
	else{
		st_s[idy] = -idy-1;
		p_s[idy].x = 0.0;
		p_s[idy].y = 0.0;
		p_s[idy].z = 0.0;
		Msun = 0.0;
		index = -1;
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
			index = index_d[id + Bl] % 100;
		}
		else{
			st_s[idy + Bl] = -idy-Bl-1;
			p_s[idy + Bl].x = 0.0;
			p_s[idy + Bl].y = 0.0;
			p_s[idy + Bl].z = 0.0;
			index = -1;
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
	//now the sum is complete
	if(index == 0){
                volatile double Tsun = 0.5 / Msun * ( p_s[idy].x*p_s[idy].x + p_s[idy].y*p_s[idy].y + p_s[idy].z*p_s[idy].z);
//printf("Tsun %.40g %d\n", Tsun, f);
		U_d[st_s[idy]] += ff * Tsun;
	}
}

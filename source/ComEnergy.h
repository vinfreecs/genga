// *********************************************************
// This kernel computes the kinetic energy of the center of mass
// it converts the velocities between heliocentric and democratic coordinates
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
// ***********************************************************
template < int Bl, int Bl2>
__global__ void com32_kernel(double4 *x4_d, double4 *v4_d, double* U_d, double Msun, double *test_d, int N, int f){

        int idy = threadIdx.x;

        __shared__ double4 p_s[Bl2];
        __shared__ double4 v4_s[Bl];

	double m = x4_d[idy].w;

        if(idy < N){
                v4_s[idy] = v4_d[idy];
        }
        else{
                v4_s[idy].x = 0.0;
                v4_s[idy].y = 0.0;
                v4_s[idy].z = 0.0;
                v4_s[idy].w = 0.0;
        }


        __syncthreads();

        if(m > 0 && idy < N){
                p_s[idy].x = m * v4_s[idy].x;
                p_s[idy].y = m * v4_s[idy].y;
                p_s[idy].z = m * v4_s[idy].z;
		p_s[idy].w = m;
        }
        else{
                p_s[idy].x = 0.0;
                p_s[idy].y = 0.0;
                p_s[idy].z = 0.0;
		p_s[idy].w = 0.0;
        }

        if(Bl <= 32){
                p_s[idy + Bl].x = 0.0;
                p_s[idy + Bl].y = 0.0;
                p_s[idy + Bl].z = 0.0;
                p_s[idy + Bl].w = 0.0;
        }

//printf("p %d %.20g %.20g %.20g\n", idy, p_s[idy].x, p_s[idy].y, p_s[idy].z);
        __syncthreads();
        if(idy < 32){
                volatile double4 *p = p_s;
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

                if(Bl >= 64) p[idy].w += p[idy + 32].w;
                if(Bl >= 32) p[idy].w += p[idy + 16].w;
                if(Bl >= 16) p[idy].w += p[idy + 8].w;
                if(Bl >= 8) p[idy].w += p[idy + 4].w;
                if(Bl >= 4) p[idy].w += p[idy + 2].w;
                if(Bl >= 2) p[idy].w += p[idy + 1].w;
        }
        __syncthreads();
//printf("p0 %d %.20g %.20g %.20g\n", idy, p_s[idy].x, p_s[idy].y, p_s[idy].z);
	double iMsun = 1.0 / Msun;
	if(idy == 0){
                volatile double Tsun = 0.5 * iMsun * ( p_s[0].x*p_s[0].x + p_s[0].y*p_s[0].y + p_s[0].z*p_s[0].z);
//printf("Tsun %.40g %d\n", Tsun, f);
		U_d[0] += f * Tsun;
	}
	if(idy < N && f == 1){
		//Convert to Heliocentric coordinates
		v4_d[idy].x += p_s[0].x * iMsun;
		v4_d[idy].y += p_s[0].y * iMsun;
		v4_d[idy].z += p_s[0].z * iMsun;
//printf("v %d %.20g %.20g %.20g %.20g\n", idy, v4_d[idy].x, v4_d[idy].y, v4_d[idy].z, iMsun);
	}
	if(idy < N && f == -1){
		//Convert to Democratic coordinates
		double iMsunp = 1.0 / (Msun + p_s[0].w);
		v4_d[idy].x -= p_s[0].x * iMsunp;
		v4_d[idy].y -= p_s[0].y * iMsunp;
		v4_d[idy].z -= p_s[0].z * iMsunp;
//printf("v %d %.20g %.20g %.20g %.20g\n", idy, v4_d[idy].x, v4_d[idy].y, v4_d[idy].z, iMsunp);
	}
}
// *********************************************************
//This kernel computes the kinetic energy of the center of mass
// it converts the velocities between heliocentric and democratic coordinates
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
// ***********************************************************
template < int Bl>
__global__ void com128_kernel(double4 *x4_d, double4 *v4_d, double* U_d, double Msun, double *test_d, int N, int f){

	int idy = threadIdx.x;

        __shared__ double4 p_s[Bl];

        p_s[idy].x = 0.0;
        p_s[idy].y = 0.0;
        p_s[idy].z = 0.0;
        p_s[idy].w = 0.0;

        for(int i = 0; i < N; i += Bl){
		double m = x4_d[idy + i].w;
                if(m > 0 && idy + i < N){
                        p_s[idy].x += m * v4_d[idy + i].x;
                        p_s[idy].y += m * v4_d[idy + i].y;
                        p_s[idy].z += m * v4_d[idy + i].z;
                        p_s[idy].w += m;
                }
        }
        __syncthreads();

        if(Bl >= 512){
                if(idy < 256){
                        p_s[idy].x += p_s[idy + 256].x;
                        p_s[idy].y += p_s[idy + 256].y;
                        p_s[idy].z += p_s[idy + 256].z;
                        p_s[idy].w += p_s[idy + 256].w;
                }
        }
        __syncthreads();

        if(Bl >= 256){
                if(idy < 128){
                        p_s[idy].x += p_s[idy + 128].x;
                        p_s[idy].y += p_s[idy + 128].y;
                        p_s[idy].z += p_s[idy + 128].z;
                        p_s[idy].w += p_s[idy + 128].w;
                }
        }
        __syncthreads();

        if(Bl >= 128){
                if(idy < 64){
                        p_s[idy].x += p_s[idy + 64].x;
                        p_s[idy].y += p_s[idy + 64].y;
                        p_s[idy].z += p_s[idy + 64].z;
                        p_s[idy].w += p_s[idy + 64].w;
                }
        }
        __syncthreads();
	if(idy < 32){
                volatile double4*p = p_s;
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
    
	        p[idy].w += p[idy + 32].w;
                p[idy].w += p[idy + 16].w;
                p[idy].w += p[idy + 8].w;
                p[idy].w += p[idy + 4].w;
                p[idy].w += p[idy + 2].w;
                p[idy].w += p[idy + 1].w;
        }
        __syncthreads();

	double iMsun = 1.0 / Msun;

        if(idy == 0){
                volatile double Tsun = 0.5 * iMsun * ( p_s[0].x*p_s[0].x + p_s[0].y*p_s[0].y + p_s[0].z*p_s[0].z);
//printf("Tsun %.40g %d\n", Tsun, f);
		U_d[0] += f * Tsun;
	}
        for(int i = 0; i < N; i += Bl){
                double m = x4_d[idy + i].w;
                if(m > 0 && idy + i < N && f == 1){
			//Convert to Heliocentric coordinates
			v4_d[idy + i].x += p_s[0].x * iMsun;
			v4_d[idy + i].y += p_s[0].y * iMsun;
			v4_d[idy + i].z += p_s[0].z * iMsun;

		}
		if(m > 0 && idy + i < N && f == -1){
			//Convert to Democratic coordinates
			double iMsunp = 1.0 / (Msun + p_s[0].w);
			v4_d[idy + i].x -= p_s[0].x * iMsunp;
			v4_d[idy + i].y -= p_s[0].y * iMsunp;
			v4_d[idy + i].z -= p_s[0].z * iMsunp;
		}
	}
}

// *********************************************************
//This kernel computes the kinetic energy of the center of mass
// it converts the velocities between heliocentric and democratic coordinates
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
// ***********************************************************
template < int Bl>
__global__ void comsmall_kernel(double4 *x4_d, double4 *v4_d, double4 *x4small_d, double4 *v4small_d, double* U_d, double Msun, double *test_d, int N, int Nsmall, int f){

	int idy = threadIdx.x;

        __shared__ double4 p_s[Bl];

        p_s[idy].x = 0.0;
        p_s[idy].y = 0.0;
        p_s[idy].z = 0.0;
        p_s[idy].w = 0.0;

        for(int i = 0; i < N; i += Bl){
		double m = x4_d[idy + i].w;
                if(m > 0 && idy + i < N){
                        p_s[idy].x += m * v4_d[idy + i].x;
                        p_s[idy].y += m * v4_d[idy + i].y;
                        p_s[idy].z += m * v4_d[idy + i].z;
                        p_s[idy].w += m;
                }
        }
        __syncthreads();

        if(Bl >= 512){
                if(idy < 256){
                        p_s[idy].x += p_s[idy + 256].x;
                        p_s[idy].y += p_s[idy + 256].y;
                        p_s[idy].z += p_s[idy + 256].z;
                        p_s[idy].w += p_s[idy + 256].w;
                }
        }
        __syncthreads();

        if(Bl >= 256){
                if(idy < 128){
                        p_s[idy].x += p_s[idy + 128].x;
                        p_s[idy].y += p_s[idy + 128].y;
                        p_s[idy].z += p_s[idy + 128].z;
                        p_s[idy].w += p_s[idy + 128].w;
                }
        }
        __syncthreads();

        if(Bl >= 128){
                if(idy < 64){
                        p_s[idy].x += p_s[idy + 64].x;
                        p_s[idy].y += p_s[idy + 64].y;
                        p_s[idy].z += p_s[idy + 64].z;
                        p_s[idy].w += p_s[idy + 64].w;
                }
        }
        __syncthreads();
	if(idy < 32){
                volatile double4*p = p_s;
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
    
	        p[idy].w += p[idy + 32].w;
                p[idy].w += p[idy + 16].w;
                p[idy].w += p[idy + 8].w;
                p[idy].w += p[idy + 4].w;
                p[idy].w += p[idy + 2].w;
                p[idy].w += p[idy + 1].w;
        }
        __syncthreads();

	double iMsun = 1.0 / Msun;
	double iMsunp = 1.0 / (Msun + p_s[0].w);

        if(idy == 0){
                volatile double Tsun = 0.5 * iMsun * ( p_s[0].x*p_s[0].x + p_s[0].y*p_s[0].y + p_s[0].z*p_s[0].z);
//printf("Tsun %.40g %d\n", Tsun, f);
		U_d[0] += f * Tsun;
	}
        for(int i = 0; i < N; i += Bl){
                double m = x4_d[idy + i].w;
                if(m > 0 && idy + i < N && f == 1){
			//Convert to Heliocentric coordinates
			v4_d[idy + i].x += p_s[0].x * iMsun;
			v4_d[idy + i].y += p_s[0].y * iMsun;
			v4_d[idy + i].z += p_s[0].z * iMsun;

		}
                if(m > 0 && idy + i < N && f == -1){
			//Convert to Democratic coordinates
			v4_d[idy + i].x -= p_s[0].x * iMsunp;
			v4_d[idy + i].y -= p_s[0].y * iMsunp;
			v4_d[idy + i].z -= p_s[0].z * iMsunp;
		}
	}
        for(int i = 0; i < Nsmall; i += Bl){
                double m = x4small_d[idy + i].w;
                if(m >= 0 && idy + i < Nsmall && f == 1){
			//Convert to Heliocentric coordinates
			v4small_d[idy + i].x += p_s[0].x * iMsun;
			v4small_d[idy + i].y += p_s[0].y * iMsun;
			v4small_d[idy + i].z += p_s[0].z * iMsun;

		}
                if(m >= 0 && idy + i < Nsmall && f == -1){
			//Convert to Democratic coordinates
			v4small_d[idy + i].x -= p_s[0].x * iMsunp;
			v4small_d[idy + i].y -= p_s[0].y * iMsunp;
			v4small_d[idy + i].z -= p_s[0].z * iMsunp;
		}
	}
}
// *********************************************************
//This kernel computes the kinetic energy of the center of mass
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
// ***********************************************************
template <int Bl, int Bl2, int Nmax >
__global__ void comM_kernel(double4 *x4_d, double4 *v4_d, const double *Msun_d, double *U_d, int *index_d, int NT, double *test_d, int ff){

	int idy = threadIdx.x;
	int id = blockIdx.x * Bl2 + idy - Nmax;
	__shared__ volatile double4 p_s[Bl + Nmax / 2];
	__shared__ int st_s[Bl + Nmax / 2];
	volatile double Msun;
	int index;

	if(id < NT && id >= 0){
		st_s[idy] = index_d[id] / 100;
		volatile double m = x4_d[id].w;
		p_s[idy].x = m * v4_d[id].x;
		p_s[idy].y = m * v4_d[id].y;
		p_s[idy].z = m * v4_d[id].z;
		p_s[idy].w = m;
		Msun = Msun_d[st_s[idy]];
		index = index_d[id] % 100;

	}
	else{
		st_s[idy] = -idy-1;
		p_s[idy].x = 0.0;
		p_s[idy].y = 0.0;
		p_s[idy].z = 0.0;
		p_s[idy].w = 0.0;
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
			p_s[idy + Bl].w = m;
			index = index_d[id + Bl] % 100;
		}
		else{
			st_s[idy + Bl] = -idy-Bl-1;
			p_s[idy + Bl].x = 0.0;
			p_s[idy + Bl].y = 0.0;
			p_s[idy + Bl].z = 0.0;
			p_s[idy + Bl].w = 0.0;
			index = -1;
		}
	}
//printf("p %d %d %.20g %.20g %.20g\n", idy, id, p_s[idy].x, p_s[idy].y, p_s[idy].z);

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

//printf("p1 %d %.20g %.20g %.20g\n", idy, p_s[idy].x, p_s[idy].y, p_s[idy].z);
	double iMsun = 1.0 / Msun;
	//now the sum is complete
	if(index == 0 && idy >= Nmax && idy < Bl - Nmax / 2){
                volatile double Tsun = 0.5 * iMsun * ( p_s[idy].x*p_s[idy].x + p_s[idy].y*p_s[idy].y + p_s[idy].z*p_s[idy].z);
//printf("Tsun %.40g %d\n", Tsun, f);
		U_d[st_s[idy]] += ff * Tsun;
	}
	if(id < NT && id >= 0 && idy >= Nmax && idy < Bl - Nmax / 2 && ff == 1){
		v4_d[id].x += p_s[idy].x * iMsun;
		v4_d[id].y += p_s[idy].y * iMsun;
		v4_d[id].z += p_s[idy].z * iMsun;
//printf("v %d %.20g %.20g %.20g %.20g %d\n", id, v4_d[id].x, v4_d[id].y, v4_d[id].z, iMsun, index_d[id]);
	}
	if(id < NT && id >= 0 && idy >= Nmax && idy < Bl - Nmax / 2 && ff == -1){
		double iMsunp = 1.0 / (Msun + p_s[idy].w);
		v4_d[id].x -= p_s[idy].x * iMsunp;
		v4_d[id].y -= p_s[idy].y * iMsunp;
		v4_d[id].z -= p_s[idy].z * iMsunp;
//printf("v %d %.20g %.20g %.20g %.20g\n", id, v4_d[id].x, v4_d[id].y, v4_d[id].z, iMsunp);
	}
}

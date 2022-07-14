#include "Orbit2.h"

__global__ void create1_kernel(curandState *random_d, double4 *x4_d, double4 *v4_d, double4 *spin_d, double3 *love_d, int *index_d, int NN, double dt, double Msun, double time, int MaxIndex, double *Fragments_d, int *nFragments_d, int NconstT){

        int idy = threadIdx.x;
        int id = blockIdx.x * blockDim.x + idy;

	if(id < NN){

		curandState random = random_d[id];
		curand_uniform(&random);
		random_d[id] = random;


		double p = 2.0;				//probability of creating a new particle, per year.
		p = p / 365.25 * dt / dayUnit;          //probability per time step and per thread

		double rd = curand_uniform(&random);
		if(rd < p){

			double amin = 2.5;
			double amax = 3.5;
			double emin = 0.0;
			double emax = 0.1;
			double incmin = 0.0;
			double incmax = 0.1;


			double a = curand_uniform(&random) * (amax - amin) + amin;   //in AU
			double e = curand_uniform(&random) * (emax - emin) + emin;
			double inc = curand_uniform(&random) * (incmax - incmin) + incmin;

			double Omega = curand_uniform(&random) * 2.0 * M_PI;
			double w = curand_uniform(&random) * 2.0 * M_PI;
			double M = curand_uniform(&random) * 2.0 * M_PI;

			double m = 0.0;
			double r = 1.0e-10;	//in AU

			double4 spin4 = {0.0, 0.0, 0.0, 0.4};
			double3 love3 = {0.0, 0.0, 0.0};

			double4 x4i, v4i;

			x4i.w = m;
			v4i.w = r;

			x4i.x = a;
			x4i.y = e;
			x4i.z = inc;
			v4i.x = Omega;
			v4i.y = w;
			v4i.z = M;
			KepToCart_M(x4i, v4i, Msun);

			int nf = atomicAdd(&nFragments_d[0], 1);

			if(NN + nf < NconstT){
printf("Create particle, %d\n", MaxIndex + nf);
				x4_d[NN + nf] = x4i;
				v4_d[NN + nf] = v4i;
				index_d[NN + nf] = MaxIndex + nf;
				spin_d[NN + nf] = spin4;
				love_d[NN + nf] = love3;

				Fragments_d[nf * 25 + 0] = time/365.25;
				Fragments_d[nf * 25 + 1] = (double)(MaxIndex + nf);
				Fragments_d[nf * 25 + 2] = x4i.w;
				Fragments_d[nf * 25 + 3] = v4i.w;
				Fragments_d[nf * 25 + 4] = x4i.x;
				Fragments_d[nf * 25 + 5] = x4i.y;
				Fragments_d[nf * 25 + 6] = x4i.z;
				Fragments_d[nf * 25 + 7] = v4i.x;
				Fragments_d[nf * 25 + 8] = v4i.y;
				Fragments_d[nf * 25 + 9] = v4i.z;
				Fragments_d[nf * 25 + 10] = spin4.x;
				Fragments_d[nf * 25 + 11] = spin4.y;
				Fragments_d[nf * 25 + 12] = spin4.z;

			}
		}
	}

}



__host__ int Data::create1Call(){
	int st = 0;
	nFragments_m[0] = 0;

	create1_kernel <<< 1, 1 >>> (random_d, x4_d, v4_d, spin_d, love_d, index_d, N_h[0] + Nsmall_h[0], dt_h[0], Msun_h[0].x, time_h[0], MaxIndex, Fragments_d, nFragments_d, NconstT);

	cudaDeviceSynchronize();
	if(nFragments_m[0] > 0){
		Nsmall_h[st] += nFragments_m[0];
		MaxIndex += nFragments_m[0];

		if(N_h[0] + Nsmall_h[0] >= NconstT){
			printf("Error, create1Call failed, too many particles\n");
			return 0;
		}

	}
	return 1;
}

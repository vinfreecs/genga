__global__ void collision_model1_kernel(curandState *random_d, double4 *x4_d, double4 *v4_d, double4 *spin_d, double3 *love_d, int *index_d, double *t1_d, double *rcrit_d, double *rcritv_d, const int NN, double *Coll_d, int MaxIndex, double *Fragments_d, int *nFragments_d, int *Nencpairs2_d, const double n1, const double n2, const double iMsun3, const double dt, const int Nc0, const int Nc, const int NT){

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int jj = blockIdx.y;

	//int NF = 87;
	int NF = 400;

	if(id == 0 && jj == 0){
		Nencpairs2_d[0] = 0;
	}

	if(jj < Nc){
		int nc = jj + Nc0;
		if(id < NF){

			double Vmin = 1.0;
			double Vmax = 1.005;


			curandState random = random_d[id];


			double time = Coll_d[nc * def_NColl];
			int indexi = int(Coll_d[nc * def_NColl + 2]); 
			int indexj = int(Coll_d[nc * def_NColl + 15]); 
			int i = int(Coll_d[nc * def_NColl + 1]); 
			//int j = int(Coll_d[nc * def_NColl + 14]); 

//printf("Resolve collision between body %d %d, time %.20g\n", i, j, time);
			double t1 = t1_d[i];


			//Coordinates of the parent body 
			double4 xp, vp;

			xp.w = Coll_d[nc * def_NColl + 3];
			vp.w = Coll_d[nc * def_NColl + 4];
			xp.x = Coll_d[nc * def_NColl + 5];
			xp.y = Coll_d[nc * def_NColl + 6];
			xp.z = Coll_d[nc * def_NColl + 7];
			vp.x = Coll_d[nc * def_NColl + 8];
			vp.y = Coll_d[nc * def_NColl + 9];
			vp.z = Coll_d[nc * def_NColl + 10];



			//new radius of parent body
			xp = x4_d[i];
			vp = v4_d[i];

			//Escape velocity at 1.1 times the physical radius of the parent body
			double vesc = sqrt(2.0 * def_ksq * xp.w / (1.1 * vp.w));


			double v = (curand_uniform_double(&random) * (Vmax - Vmin) + Vmin) * vesc;

			//direction 
			double u = curand_uniform_double(&random);
			double theta = curand_uniform_double(&random) * 2.0 * M_PI;

			//sign
			double s = curand_uniform_double(&random);

			//move the new particle 1.1 times the physical radius of the parent body away
			double x = 1.1 * vp.w * sqrt(1.0 - u * u) * cos(theta);
			double y = 1.1 * vp.w * sqrt(1.0 - u * u) * sin(theta);
			double z = 1.1 * vp.w * u;

			volatile double vx = v * sqrt(1.0 - u * u) * cos(theta);
			volatile double vy = v * sqrt(1.0 - u * u) * sin(theta);
			volatile double vz = v * u;

			if( s > 0.5){
				z *= -1.0;
				vz *= -1.0;
			}


			double4 spin4 = {0.0, 0.0, 0.0, 0.4};
			double3 love3 = {0.0, 0.0, 0.0};

			double4 x4i, v4i;
			x4i.w = 1.0e-8;
			v4i.w = 1.0e-10;

			x4i.x = xp.x + x;
			x4i.y = xp.y + y;
			x4i.z = xp.z + z;
			v4i.x = vp.x + vx;
			v4i.y = vp.y + vy;
			v4i.z = vp.z + vz;
//printf("%d %g %g %g %g %g\n", id, vp.w, x, y, z, sqrt(x*x + y*y + z*z));



			//Calculate critical radius
			double rsq = x4i.x*x4i.x + x4i.y*x4i.y + x4i.z*x4i.z + 1.0e-30;
			double vsq = v4i.x*v4i.x + v4i.y*v4i.y + v4i.z*v4i.z + 1.0e-30;

			double r = sqrt(rsq);
			double v1 = sqrt(vsq);

			double rcrit = n1 * r * cbrt(x4i.w * iMsun3);

			double rc2 = n2 * fabs(dt) * v1;
			double rcritv = (rcrit > rc2) ? rcrit : rc2;


#if def_CPU == 0
			int nf = atomicAdd(&nFragments_d[0], 1);
#else
			int nf;
			#pragma omp atomic capture
			nf = nFragments_d[0]++;
#endif
printf("Create particle, %d %d %d %d %.20g \n", id, i, nf, MaxIndex + nf + 1, time);

			if(NN + nf < NT){
//printf("Create particle, %d %d %d %d %d\n", id, i, nf, MaxIndex, MaxIndex + nf + 1);
				x4_d[NN + nf] = x4i;
				v4_d[NN + nf] = v4i;
				index_d[NN + nf] = MaxIndex + nf + 1;
				spin_d[NN + nf] = spin4;
				love_d[NN + nf] = love3;
				t1_d[NN + nf] = t1;

				rcrit_d[NN + nf] = rcrit;
				rcritv_d[NN + nf] = rcritv;


				Fragments_d[nf * def_NColl + 0] = time;
				Fragments_d[nf * def_NColl + 1] = (double)(MaxIndex + nf + 1);
				Fragments_d[nf * def_NColl + 2] = x4i.w;
				Fragments_d[nf * def_NColl + 3] = v4i.w;
				Fragments_d[nf * def_NColl + 4] = x4i.x;
				Fragments_d[nf * def_NColl + 5] = x4i.y;
				Fragments_d[nf * def_NColl + 6] = x4i.z;
				Fragments_d[nf * def_NColl + 7] = v4i.x;
				Fragments_d[nf * def_NColl + 8] = v4i.y;
				Fragments_d[nf * def_NColl + 9] = v4i.z;
				Fragments_d[nf * def_NColl + 10] = spin4.x;
				Fragments_d[nf * def_NColl + 11] = spin4.y;
				Fragments_d[nf * def_NColl + 12] = spin4.z;
				Fragments_d[nf * def_NColl + 13] = double(indexi);
				Fragments_d[nf * def_NColl + 14] = double(indexj);


			}

		}

	}
}



// This kernel creates a new encounter pairs list after collision fragments are created.
// Only the bodies are considered with an unfinished BS integration (t1 < 1.0)

__global__ void encounter_fragments_kernel(double4 *x4_d, double *rcritv_d, double *t1_d, int2 *Encpairs2_d, int *Nencpairs2_d, int *EncFlag_d, const int NencMax, const int N){


	int idy = threadIdx.x;
	int jj = blockIdx.x;


	if(jj < N){

		double4 x4j = x4_d[jj];
		double t1j = t1_d[jj];
		double rcritvj = rcritv_d[jj];

		__shared__ int Ni_s[1];

		if(idy == 0){
			Ni_s[0] = 0;
		}
		__syncthreads();


		for(int i = 0; i < N; i += blockDim.x){

			int ii = idy + i;

			if(ii < N){

				double4 x4i = x4_d[ii];
				double t1i = t1_d[ii];
				double rcritvi = rcritv_d[ii];

				if(ii != jj && x4i.w >= 0.0 && x4j.w >= 0.0){

					if(t1i == t1j && t1i < 1.0){
						double3 r3ij;

						r3ij.x = x4j.x - x4i.x;
						r3ij.y = x4j.y - x4i.y;
						r3ij.z = x4j.z - x4i.z;

						double rsq = r3ij.x*r3ij.x + r3ij.y*r3ij.y + r3ij.z*r3ij.z;
						double rcritv = (rcritvi > rcritvj) ? rcritvi : rcritvj;


						if(rsq < def_pc * rcritv * rcritv && (x4i.w > 0.0 || x4j.w > 0.0)){


#if def_CPU == 0
							int Ni = atomicAdd(&Ni_s[0], 1);
							if(Ni >= NencMax){
								atomicMax(&EncFlag_d[0], Ni);
							}
#else
							int Ni = Ni_s[0]++;
							if(Ni >= NencMax){
								EncFlag_m[0] = max(EncFlag_m[0], Ni);
							}
#endif


							if(ii < jj && Ni < NencMax){
#if def_CPU == 0
								int Ne = atomicAdd(Nencpairs2_d, 1);
#else
								int Ne = Nencpairs2_h[0]++;
#endif
								Encpairs2_d[Ne].x = ii;
								Encpairs2_d[Ne].y = jj;
//printf("Precheck %d %d %d %d\n", i, j, Ne, EE);
							}
						}
					}
				}
			}
		}
	}
}



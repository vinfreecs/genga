#include "Orbit2.h"
#define facrho 1.0/sqrt(2.0 * M_PI)


double Gas_dTau_diss;
double Gas_alpha;
double *Gas_rg_h, *Gas_rg_d;
double *Gas_zg_h, *Gas_zg_d;
double *Gas_rho_h, *Gas_rho_d;
double3 *GasDisk_h, *GasDisk_d;
double3 *GasAcc_h, *GasAcc_d;

__host__ void Data::GasAlloc(){

	Gas_dTau_diss = P.G_dTau_diss;
	Gas_alpha = P.G_alpha;

	Gas_rg_h = (double*)malloc(def_Gasnr_g * sizeof(double));
	Gas_zg_h = (double*)malloc(def_Gasnr_g * def_Gasnz_g * sizeof(double));
	Gas_rho_h = (double*)malloc(def_Gasnr_g * def_Gasnz_g * sizeof(double));
	GasDisk_h = (double3*)malloc(def_Gasnr_p * sizeof(double3));
	GasAcc_h = (double3*)malloc(def_Gasnr_p * def_Gasnz_p * sizeof(double3));

	cudaMalloc((void **) &Gas_rg_d, def_Gasnr_g * sizeof(double));
	cudaMalloc((void **) &Gas_zg_d, def_Gasnr_g * def_Gasnz_g * sizeof(double));
	cudaMalloc((void **) &Gas_rho_d, def_Gasnr_g * def_Gasnz_g * sizeof(double));
	cudaMalloc((void **) &GasDisk_d, def_Gasnr_p * sizeof(double3));
	cudaMalloc((void **) &GasAcc_d, def_Gasnr_p * def_Gasnz_p * sizeof(double3));
}


// *************************************************
// This function corresponds to the msrGasTable function in the file master.c in pkdgrav_planets.
//
// ****************************************************
__host__ void Data::GasDisk(double *Gas_rg_h, double *Gas_zg_h, double *Gas_rho_h, double dTau_diss, double G_alpha, double G_Sigma_10){

	double dr1 = 0.1;
	double dr2 = 0.5;

	double drg, h, Sigma, zh, rg, zg;

	double Sigma10 = G_Sigma_10;
//	double rin, ro;
//	if(uniform != 1){
//		rin = 0.1 + dTime/(2.0 * M_PI * dTau_diss); // time scale for the inner edge to move 1AU
//	}

	for(int ig = 0; ig < def_Gasnr_g; ++ig){
		if(ig < 149){
			drg = dr1;
			rg = 0.15 + drg * ig;
		}
		else{
			drg = dr2;
			rg = 15.25 + (ig - 149.0) * drg;
		}

		Gas_rg_h[ig] = rg;

		h = def_h_1 * rg * pow(rg, 0.25); //beta = 0.25 comes from Temperature profile
		Sigma = Sigma10 * pow(rg, -G_alpha);
	
//			if(uniform != 1){
//				ro = rg + 0.5 * drg; // radius of the outer cel boundary
//				if(rin > ro){
//					Sigma = 0.0;
//				}
//				else if(fabs(rin - rg) < 0.5 * drg){
//					//if((ro - rin) > drg) send error
//					Sigma *= (ro - rin)/drg;
//				}
//			}
		for(int jg = 0; jg < def_Gasnz_g; ++jg){
			zg = 0.01 * (0.5 + jg) * rg;
			zh = zg / h;
			Gas_zg_h[ig * def_Gasnz_g + jg] = zg;
			Gas_rho_h[ig * def_Gasnz_g + jg] = facrho * (Sigma / h) * exp(-0.5 * zh * zh);
//printf("%g %g %g\n", rg, zg, Gas_rho_h[ig * def_Gasnz_g + jg];
		}
	}
}

// **********************************************
// First Kind elliptic integral
// This function corresponds to the rf function in the file master.c in pkdgrav_planets.
// It is based on numerical recipes and the paper from Carlson 
// **********************************************
__device__ double rf(double x, double y, double z){
	double lambda, mu, imu, X, Y, Z;
	double E2, E3, sqrtx, sqrty, sqrtz;
	double errtol = 0.008;
	double third = 1.0/3.0;

	double xt, yt, zt;

	xt = x;
	yt = y;
	zt = z;

//	if(fmin(fmin(x, y), z) < 0.0 || fmin(fmin(x + y, x + z), y + z) < 1.5e-38 || fmax(fmax(x, y), z) > 3.0e37){
//		printf("invalid arguments in first elliptical integral.");
//	}

	for(int i = 0; i < 10000; ++i){
		sqrtx = sqrt(xt);	
		sqrty = sqrt(yt);
		sqrtz = sqrt(zt);

		lambda = sqrtx * (sqrty + sqrtz) + sqrty * sqrtz;

		xt = 0.25 * (xt + lambda);
		yt = 0.25 * (yt + lambda);
		zt = 0.25 * (zt + lambda);

		mu = third * (xt + yt + zt);
		imu = 1.0 / mu;
		X = (mu - xt) * imu;
		Y = (mu - yt) * imu;
		Z = (mu - zt) * imu;

		if(fmax(fmax(fabs(X), fabs(Y)), fabs(Z)) <= errtol) break;
	}


	E2 = X * Y - Z * Z;
	E3 = X * Y * Z;

	return (1.0 + E2*(E2/24.0 - E3 * 3.0/44.0 - 0.1) + E3/14.0) / sqrt(mu);
}


// **********************************************
// Second Kind elliptic integral
// This function corresponds to the rf function in the file master.c in pkdgrav_planets.
// It is based on numerical recipes and the paper from Carlson 
// **********************************************
__device__ double rd(double x, double y, double z){
	double lambda, mu, imu, X, Y, Z;
	double EA, EB, EC, ED, EE, sqrtx, sqrty, sqrtz;
	double errtol = 0.008;
	double sum = 0.0;
	double fac = 1.0;

	double xt, yt, zt;

	xt = x;
	yt = y;
	zt = z;

//	if(fmin(x, y) < 0.0 || fmin(x + y, z) < 1.0e-25 || fmax(fmax(x, y), z) > 4.5e21){
//		printf("invalid arguments in second elliptical integral.");
//	}

	for(int i = 0; i < 10000; ++i){
		sqrtx = sqrt(xt);
		sqrty = sqrt(yt);
		sqrtz = sqrt(zt);

		lambda = sqrtx * (sqrty + sqrtz) + sqrty * sqrtz;
		sum += fac / (sqrtz * (zt + lambda));	//difference from master.c and numerical recipes
//sum += fac / (sqrtz * zt + lambda);
		fac *= 0.25;
		xt = 0.25 * (xt + lambda);
		yt = 0.25 * (yt + lambda);
		zt = 0.25 * (zt + lambda);

		mu = 0.2 * (xt + yt + 3.0 * zt);
		imu = 1.0 / mu;
		X = (mu - xt) * imu;
		Y = (mu - yt) * imu;
		Z = (mu - zt) * imu;

		if(fmax(fmax(fabs(X), fabs(Y)), fabs(Z)) <= errtol) break;
	}

	EA = X * Y;
	EB = Z * Z;
	EC = EA - EB;
	ED = EA - 6.0 * EB;
	EE = ED + EC + EC;


	return 3.0 * sum + fac * (1.0 + ED * (ED * 9.0/88.0 - Z * EE * 4.5/26.0 - 3.0/14.0) + Z * (EE / 6.0 + Z *(EC * -9.0/22.0 + Z * EA * 3.0/26.0))) /(mu * sqrt(mu));
}

// *************************************************
// This function corresponds to the ERRORFUNCT function in the file pkd.h in pkdgrav_planets.
//
// ****************************************************
__device__ double ERFUNC(double zz){
	double z, t, erfcc;
	z = fabs(zz);
	t = 1.0/(1.0 + 0.5 * z);
	erfcc = t*exp(-z*z-1.26551223+t*(1.00002368+t*(0.37409196+t*(0.09678418+t*(-0.18628806+t*(0.27886807+t*(-1.13520398+t*(1.48851587+t*(-0.82215223+t*0.17087277)))))))));
	double erfc = 1.0-erfcc;
	if(zz < 0)erfc *= -1;

	return erfc;
}

// *************************************************
// This function corresponds to the msrGasTable function in the file master.c in pkdgrav_planets.
//
// ****************************************************
__global__ void gasTabel_kernel(double *Gas_rg_d, double *Gas_zg_d, double *Gas_rho_d, double3 *GasDisk_d, double3 *GasAcc_d, double G_alpha, double G_Sigma_10){

	int ip = blockIdx.x * blockDim.x + threadIdx.x; // r
	int jp = blockIdx.y * blockDim.y + threadIdx.y; // z


	double dr1 = 0.1;
	double dr2 = 0.5;

	volatile double ar, az;
	double rp, zp, drg;
	double h;
	double ellf, elle;

	if(ip < def_Gasnr_p){	
		rp = 0.1 + dr1 * ip;

		h = def_h_1 * rp * pow(rp, 0.25);
		double Sigma = G_Sigma_10 * pow(rp, -G_alpha);
//		if(uniform != 1){
//			ro = rp + 0.5 * dr1; // radius of the outer cel boundary
//			if(rin > rp + 0.5 * dr1){
//				Sigma = 0.0;
//			}
//			else if(fabs(rin - rp) < 0.5 * dr1){
//				//if((ro - rin) > dr1) send error
//				Sigma *= (ro - rin)/dr1;
//			}
//		}

		if(jp == 0){
			GasDisk_d[ip].x = Sigma;
			GasDisk_d[ip].y = h;
			GasDisk_d[ip].z = rp;
//printf("%d %g %g %g\n", ip, Sigma, h, rp);
		}
	}
	__syncthreads();

	if(ip < def_Gasnr_p && jp < def_Gasnz_p){

		zp = (0.03 * jp) * rp;
		ar = 0.0;
		az = 0.0;
		for(int ig = 0; ig < def_Gasnr_g; ++ig){
			if(ig < 150) drg = dr1;
			else drg = dr2;
			double rgas = Gas_rg_d[ig];
			double dzg = 0.03 * rgas;
			
			for(int jg = 0; jg < def_Gasnz_g; ++jg){

				double zgas = Gas_zg_d[ig * def_Gasnz_g + jg];
				double rho_gas = Gas_rho_d[ig * def_Gasnz_g + jg];

				volatile double rpzm = (zp - zgas) * (zp - zgas);
				volatile double rmzm = rpzm + (rp - rgas) * (rp - rgas);
				rpzm += (rp + rgas) * (rp + rgas);

				double k2 = (4.0 * rp * rgas) / rpzm;
				ellf = rf(0.0, 1.0 - k2, 1.0);
				elle = rf(0.0, 1.0 - k2, 1.0) - k2 * rd(0.0, 1.0 - k2, 1.0) / 3.0;

				volatile double temp = -2.0 * (rho_gas / sqrt(rpzm)) * rgas * drg * dzg;
				elle /= rmzm;

				ar += (temp/rp) * (elle * (rp * rp - rgas * rgas - (zp - zgas) * (zp - zgas)) + ellf);
				az += 2.0 * temp * (zp - zgas) * elle;

				zgas = -zgas;
				rpzm = (zp - zgas) * (zp - zgas);
				rmzm = rpzm + (rp - rgas) * (rp - rgas);
				rpzm += (rp + rgas) * (rp + rgas);

				k2 = (4.0 * rp * rgas) / rpzm;
				ellf = rf(0.0, 1.0 - k2, 1.0);
				elle = rf(0.0, 1.0 - k2, 1.0) - k2 * rd(0.0, 1.0 - k2, 1.0) / 3.0;
		
				temp = -2.0 * (rho_gas / sqrt(rpzm)) * rgas * drg * dzg;
				elle /= rmzm;

				ar += (temp/rp) * (elle * (rp * rp - rgas * rgas - (zp - zgas) * (zp - zgas)) + ellf);
				az += 2.0 * temp * (zp - zgas) * elle;

			}
		}
		GasAcc_d[ip * def_Gasnz_p + jp].x = ar;
		GasAcc_d[ip * def_Gasnz_p + jp].y = az;
		GasAcc_d[ip * def_Gasnz_p + jp].z = zp;
	}
}

__host__ int Data::setGasDisk(){
	cudaError_t error;
	GasDisk(Gas_rg_h, Gas_zg_h, Gas_rho_h, P.G_dTau_diss, P.G_alpha, P.G_Sigma_10);

	cudaMemcpy(Gas_rg_d, Gas_rg_h, def_Gasnr_g * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Gas_zg_d, Gas_zg_h, def_Gasnr_g * def_Gasnz_g * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Gas_rho_d, Gas_rho_h, def_Gasnr_g * def_Gasnz_g * sizeof(double), cudaMemcpyHostToDevice);

	dim3 NTGasTabel(32, 1, 1);
	dim3 NBGasTabel((def_Gasnr_p + 31) / 32, def_Gasnz_p, 1);

	gasTabel_kernel <<< NBGasTabel, NTGasTabel >>>(Gas_rg_d, Gas_zg_d, Gas_rho_d, GasDisk_d, GasAcc_d, P.G_alpha, P.G_Sigma_10);

	cudaDeviceSynchronize();
	error = cudaGetLastError();
	fprintf(masterfile, "Gas Table error = %d = %s\n",error, cudaGetErrorString(error));
	if(error != 0){
		printf("Gas Table error = %d = %s\n",error, cudaGetErrorString(error));
		return 0;
	}
	else return 1;
}

// *************************************************
// This kernel corresponds to the pkdGasAccel function in the file pkd.c in pkdgrav_planets.
//
// ****************************************************
__global__ void GasAcc(double4 *x4_d, double4 *v4_d, int *index_d, double3 *GasDisk_d, double3 *GasAcc_d, double *time_d, double4 *Msun_d, double *dt_d, int N, double *Energy_d, double dTau_diss, double G_alpha, double G_Sigma_10, int Nst, double Ct, int UsegasPotential, int UsegasEnhance, int UsegasDrag, int UsegasTidalDamping, double Mgiant, int Nstart){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy + Nstart;
	int st = 0;

	if(Nst > 1 && id < N + Nstart) st = index_d[id] / def_MaxIndex;
	double dt = dt_d[st] * Ct;
	double dTime = time_d[st] / 365.25;

	double Msun = Msun_d[st].x;
	double U = 0.0;

	double3 v_rel3;
	double v_rel_r, v_rel_th;
	double big = 1.0e7;

	double depfac = exp(-dTime/(dTau_diss));

	double4 x4;
	double4 v4;

	if(id < N + Nstart){
		x4 = x4_d[id];
		v4 = v4_d[id];
	}
	else{
		x4.x = 0.0;
		x4.y = 0.0;
		x4.z = 0.0;
		x4.w = 0.0;
		v4.x = 0.0;
		v4.y = 0.0;
		v4.z = 0.0;
		v4.w = 0.0;
	}

	double r1 = x4.x * x4.x + x4.y * x4.y;
	double rsq = r1 + x4.z * x4.z;
	double vsq = v4.x * v4.x + v4.y * v4.y + v4.z * v4.z;
	r1 = sqrt(r1);

//r1 = id * 0.01;
	double r = sqrt(rsq);

	volatile double a_r = 0.0;
	volatile double a_th = 0.0;
	volatile double a_x = 0.0;
	volatile double a_y = 0.0;
	volatile double a_z = 0.0;

	__syncthreads();
	
	if(id < N + Nstart && r1 > 0.1 && r1 < 35.0 && x4.z/r1 < 1.5){ //otherwise there is no gas

		double h = def_h_1 * r1 * pow(r1, 0.25);
		//double Sigma = G_Sigma_10 * pow(r1, -G_alpha);
		double Sigma = G_Sigma_10 / r1;
		if(G_alpha == 2.0) Sigma *= 3.5/(2.0794*r1);

		Sigma *= depfac;
//if(id < 100) printf("%d %g %g %g\n", id, r1, Sigma, h);

		int ip = floor(10*r1);

		double zh = x4.z / h;

		double m = x4.w;
		if(m == 0.0) m = def_MgasSmall;

		//gas potential
		if(UsegasPotential == 1 || (UsegasPotential == 2 && m < Mgiant)){

			if(r1 >= 15.0){
				a_r += -2.0 * M_PI * Sigma;	
				double erfc = ERFUNC(zh);
				a_z += a_r * erfc;
				if(G_alpha == 2.0) a_r *= log(10.0 * r1);
			}
			else{
				double rr0 = GasDisk_d[ip - 1].z;
				double rr1 = GasDisk_d[ip].z;

				int jp0 = floor(fabs(x4.z)/(rr0 * 0.03));
				int jp1 = floor(fabs(x4.z)/(rr1 * 0.03));

				double zz00 = GasAcc_d[(ip - 1) * def_Gasnz_p + jp0].z;
				double zz01 = GasAcc_d[(ip - 1) * def_Gasnz_p + jp0 + 1].z;
				double zz10 = GasAcc_d[ip * def_Gasnz_p + jp1].z;
				double zz11 = GasAcc_d[ip * def_Gasnz_p + jp1 + 1].z;

				double dr00 = __dmul_rn((x4.z - zz00) , (x4.z - zz00)) + __dmul_rn((r1 - rr0) , (r1 - rr0));
				double dr01 = __dmul_rn((x4.z - zz01) , (x4.z - zz01)) + __dmul_rn((r1 - rr0) , (r1 - rr0));
				double dr10 = __dmul_rn((x4.z - zz10) , (x4.z - zz10)) + __dmul_rn((r1 - rr1) , (r1 - rr1));
				double dr11 = __dmul_rn((x4.z - zz11) , (x4.z - zz11)) + __dmul_rn((r1 - rr1) , (r1 - rr1));
				
				dr00 = fmin(1.0/sqrt(dr00), big);
				dr01 = fmin(1.0/sqrt(dr01), big);
				dr10 = fmin(1.0/sqrt(dr10), big);
				dr11 = fmin(1.0/sqrt(dr11), big);
				double drtotal = dr00 + dr01 + dr10 + dr11;


				double a_r_t0 = GasAcc_d[(ip - 1) * def_Gasnz_p + jp0].x * dr00 + GasAcc_d[(ip - 1) * def_Gasnz_p + jp0 + 1].x * dr01 + GasAcc_d[ip * def_Gasnz_p + jp1].x * dr10 + GasAcc_d[ip * def_Gasnz_p + jp1 + 1].x * dr11;

				double a_z_t0 = GasAcc_d[(ip - 1) * def_Gasnz_p + jp0].y * dr00 + GasAcc_d[(ip - 1) * def_Gasnz_p + jp0 + 1].y * dr01 + GasAcc_d[ip * def_Gasnz_p + jp1].y * dr10 + GasAcc_d[ip * def_Gasnz_p + jp1 + 1].y * dr11;
				a_r_t0 /= drtotal;
				a_z_t0 /= drtotal;


				a_r += a_r_t0 * depfac;
				if(x4.z >= 0.0){
					a_z += a_z_t0 * depfac;
				}
				else a_z -= a_z_t0 * depfac;

			}

//printf("%d %g %g %g\n", id, r1, Sigma, a_r);
		}


		if(Sigma > 0.0){
			
			//Enhanced Drag
			double Soft;
			if(UsegasEnhance == 1 || (UsegasEnhance == 2 && m < Mgiant)){
				
				if(m < def_M_Enhance && m > def_MgasSmall){
					double pid = log(m/def_fMass_min) / log(def_M_Enhance/def_fMass_min);
					double jc = def_M_Enhance/def_Mass_pl;
					m = pow(jc, pid);
					m *= def_Mass_pl;
					Soft = v4.w * pow(m/x4.w, 1.0/3.0);
				}
				else{
				
					Soft = v4.w;
				}
			}
			else{
				Soft = v4.w;

			}
			
			//vKep
			double v_kep = 0.0;
			if(UsegasDrag > 0 || UsegasTidalDamping > 0){
				v_kep = sqrt(Msun * def_ksq / r1 - a_r * r1); 
			}
	
			//gas drag
			if(UsegasDrag == 1 || (UsegasDrag == 2 && m < Mgiant)){
				double eta = 0.5 * ((G_alpha + 1.75) * h * h + 0.5 * x4.z * x4.z) / (r1 * r1);
				double v_gas = v_kep * (1.0 - eta);	//Change that to v_kep * sqrt(1.0 - 2.0 * eta);

				double rho = facrho * Sigma / h * exp(-0.5 * zh * zh);
				v_rel3.x = v4.x + v_gas * x4.y / r1; 
				v_rel3.y = v4.y - v_gas * x4.x / r1;
				v_rel3.z = v4.z;

				v_rel_r = (x4.x * v_rel3.x + x4.y * v_rel3.y) / r1;
				v_rel_th = (x4.x * v_rel3.y - x4.y * v_rel3.x) / r1;
				
				double v_rel = v_rel3.x * v_rel3.x + v_rel3.y * v_rel3.y + v_rel3.z * v_rel3.z;
				v_rel = sqrt(v_rel);
				if(m > 0.0) v_rel *= M_PI / m * Soft * Soft * rho;
				else v_rel = 0.0;

				a_x += -v_rel * v_rel3.x;
				a_y += -v_rel * v_rel3.y;
				a_z += -v_rel * v_rel3.z;
			}

			//tidal damping
			if(UsegasTidalDamping == 1 || (UsegasTidalDamping == 2 && m < Mgiant)){
				v_rel3.x = v4.x + v_kep * x4.y / r1;
				v_rel3.y = v4.y - v_kep * x4.x / r1;
				v_rel3.z = v4.z;
				v_rel_r = (x4.x * v_rel3.x + x4.y * v_rel3.y) / r1;
				v_rel_th = (x4.x * v_rel3.y - x4.y * v_rel3.x) / r1;

				double Mtot = Msun + m;
				double Etot = 0.5 * vsq - def_ksq * Mtot / r;
				double3 L;
				L.x = x4.y * v4.z - x4.z * v4.y;
				L.y = x4.z * v4.x - x4.x * v4.z;
				L.z = x4.x * v4.y - x4.y * v4.x;

				double Lsq = L.x * L.x + L.y * L.y + L.z * L.z;
				double esq = 1.0 + 2.0 * Etot * Lsq /(Mtot * Mtot);
				if(esq < 0.0) esq = 0.0;
				double isq = 1.0 - L.z * L.z / Lsq;
				double a = -0.5/Etot;

				double r1h2 = r1 * r1 / (h * h);
				
				double chi = 0.5 * a * sqrt(esq + isq) / h;
				double chi3 = chi * chi * chi;

				double iTau_tid1 = m * (Sigma * r1 * r1) * r1h2	/ (r1 * sqrt(r1)); //a = ri because of circular orbit
				double iTauwave_gwave = iTau_tid1 * r1h2 / (1.0 + 0.25 * chi3);
				iTau_tid1 *= 3.8 * (1.0 - 0.683 * chi3 * chi) / (1.0 + 0.269329 * chi3 * chi * chi);

				a_th += -0.5 * iTau_tid1 * v_kep; //circular

				a_r += (0.104 * v_rel_th + 0.176 * v_rel_r) * iTauwave_gwave;
				a_th += (-1.736 * v_rel_th + 0.325 * v_rel_r) * iTauwave_gwave;
				a_z += (-1.088 * v_rel3.z - 0.871 * v_kep * x4.z / r1) * iTauwave_gwave;
			}
		}
		a_x += (x4.x * a_r - x4.y * a_th) / r1;
		a_y += (x4.y * a_r + x4.x * a_th) / r1;

	}
	__syncthreads();

	if(id < N + Nstart && x4.w >= 0.0){
		double v2 = v4.x * v4.x + v4.y * v4.y + v4.z * v4.z;
		//Kick
		v4.x += a_x * dt;
		v4.y += a_y * dt;
		v4.z += a_z * dt;
//printf("Gas 2 %d %g %g %g %g %g %g %g\n", id, v4.x, v4.y, v4.z, a_x, a_y, a_z, x4.w);
		double v2B = v4.x * v4.x + v4.y * v4.y + v4.z * v4.z;

		v4_d[id] = v4;
	
		if(x4.w > 0.0){
			U = 0.5 * x4.w * (v2 - v2B);
			Energy_d[id] += U;
		}
	}

}

// *************************************************
// This kernel corresponds to the pkdGasAccel function in the file pkd.c in pkdgrav_planets.
//
// ****************************************************
__global__ void GasAcc2(double4 *x4_d, double4 *v4_d, int *index_d, double *time_d, double4 *Msun_d, double *dt_d, int N, double *Energy_d, int Nst, double Ct, int nr, double2 GasDatatime, double4 *GasData_d, double G_alpha, int UsegasPotential, int UsegasEnhance, int UsegasDrag, int UsegasTidalDamping, double Mgiant, int Nstart){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy + Nstart;
	int st = 0;

	double rin = 0.25;
	double rout = 1000.0;
	double rs = log(rout / rin) / ((double)(nr - 1)); //slope in distance profile 

	if(Nst > 1 && id < N + Nstart) st = index_d[id] / def_MaxIndex;
	double dt = dt_d[st] * Ct;
	double dTime = time_d[st] / 365.25;

//if(id == 0) printf("rs %g %g %g %g\n", rs, GasDatatime.x, GasDatatime.y, dTime);

	double Msun = Msun_d[st].x;
	double U = 0.0;

	double3 v_rel3;
	double v_rel_r, v_rel_th;

	double4 x4;
	double4 v4;

	if(id < N + Nstart){
		x4 = x4_d[id];
		v4 = v4_d[id];
	}
	else{
		x4.x = 0.0;
		x4.y = 0.0;
		x4.z = 0.0;
		x4.w = 0.0;
		v4.x = 0.0;
		v4.y = 0.0;
		v4.z = 0.0;
		v4.w = 0.0;
	}

	double r1 = x4.x * x4.x + x4.y * x4.y;
	double rsq = r1 + x4.z * x4.z;
	double vsq = v4.x * v4.x + v4.y * v4.y + v4.z * v4.z;
	r1 = sqrt(r1);
	double r = sqrt(rsq);

	volatile double a_r = 0.0;
	volatile double a_th = 0.0;
	volatile double a_x = 0.0;
	volatile double a_y = 0.0;
	volatile double a_z = 0.0;

	__syncthreads();
	
	if(id < N + Nstart && r1 > rin && r1 < rout){ //otherwise there is no gas

		int ri = (int)(log(r1 / rin) / rs);
		double ri0 = rin * exp(rs * ri);
		double ri1 = rin * exp(rs * (ri + 1));

		double4 GasData0 = GasData_d[ri];
		double4 GasData1 = GasData_d[ri + 1];

		double tr = (r1 - ri0) / (ri1 - ri0);
		double Sigma0 = (GasData1.x - GasData0.x) * tr + GasData0.x;
		double h0 = (GasData1.y - GasData0.y) * tr + GasData0.y;
		double Sigma1 = (GasData1.z - GasData0.z) * tr + GasData0.z;
		double h1 = (GasData1.w - GasData0.w) * tr + GasData0.w;
		double tt = (dTime - GasDatatime.x) / (GasDatatime.y - GasDatatime.x);
		double Sigma = (Sigma1 - Sigma0) * tt + Sigma0;
		Sigma *= 1.49598*1.49598/1.98892*1.0e-7;
		double h = ((h1 - h0) * tt + h0) * r1;

//if(id < 100) printf("%d %g %g %g %g %g\n", id, r1, Sigma, h, GasData1.x, GasData0.x);

		double zh = x4.z / h;

		double m = x4.w;
		if(m == 0.0) m = def_MgasSmall;
		if(Sigma > 0.0){
		
			//Enhanced Drag
			double Soft;
			if(UsegasEnhance == 1 || (UsegasEnhance == 2 && m < Mgiant)){
				
				if(m < def_M_Enhance && m > def_MgasSmall){
					double pid = log(m/def_fMass_min) / log(def_M_Enhance/def_fMass_min);
					double jc = def_M_Enhance/def_Mass_pl;
					m = pow(jc, pid);
					m *= def_Mass_pl;
					Soft = v4.w * pow(m/x4.w, 1.0/3.0);
				}
				else{
				
					Soft = v4.w;
				}
			}
			else{
				Soft = v4.w;

			}

			//vKep
			double v_kep = 0.0;
			if(UsegasDrag > 0 || UsegasTidalDamping > 0){
				v_kep = sqrt(Msun * def_ksq / r1 - a_r * r1); 
			}

			//gas drag
			if(UsegasDrag == 1 || (UsegasDrag == 2 && m < Mgiant)){
				double eta = 0.5 * ((G_alpha + 1.75) * h * h + 0.5 * x4.z * x4.z) / (r1 * r1);
				double v_gas = v_kep * (1.0 - eta);	//Change that to v_kep * sqrt(1.0 - 2.0 * eta);

				double rho = facrho * Sigma / h * exp(-0.5 * zh * zh);
				v_rel3.x = v4.x + v_gas * x4.y / r1; 
				v_rel3.y = v4.y - v_gas * x4.x / r1;
				v_rel3.z = v4.z;

				v_rel_r = (x4.x * v_rel3.x + x4.y * v_rel3.y) / r1;
				v_rel_th = (x4.x * v_rel3.y - x4.y * v_rel3.x) / r1;
				
				double v_rel = v_rel3.x * v_rel3.x + v_rel3.y * v_rel3.y + v_rel3.z * v_rel3.z;
				v_rel = sqrt(v_rel);
				if(m > 0.0) v_rel *= M_PI / m * Soft * Soft * rho;
				else v_rel = 0.0;

				a_x += -v_rel * v_rel3.x;
				a_y += -v_rel * v_rel3.y;
				a_z += -v_rel * v_rel3.z;
			}

			//tidal damping
			if(UsegasTidalDamping == 1 || (UsegasTidalDamping == 2 && m < Mgiant)){
				v_rel3.x = v4.x + v_kep * x4.y / r1;
				v_rel3.y = v4.y - v_kep * x4.x / r1;
				v_rel3.z = v4.z;

				v_rel_r = (x4.x * v_rel3.x + x4.y * v_rel3.y) / r1;
				v_rel_th = (x4.x * v_rel3.y - x4.y * v_rel3.x) / r1;

				double Mtot = Msun + m;
				double Etot = 0.5 * vsq - def_ksq * Mtot / r;
				double3 L;
				L.x = x4.y * v4.z - x4.z * v4.y;
				L.y = x4.z * v4.x - x4.x * v4.z;
				L.z = x4.x * v4.y - x4.y * v4.x;

				double Lsq = L.x * L.x + L.y * L.y + L.z * L.z;
				double esq = 1.0 + 2.0 * Etot * Lsq / (Mtot * Mtot);
				if(esq < 0.0) esq = 0.0;
				double isq = 1.0 - L.z * L.z / Lsq;
				double a = -0.5 / Etot;

				double r1h2 = r1 * r1 / (h * h);
				
				double chi = 0.5 * a * sqrt(esq + isq) / h;
				double chi3 = chi * chi * chi;

				double iTau_tid1 = m * (Sigma * r1 * r1) * r1h2	/ (r1 * sqrt(r1)); //a = ri because of circular orbit
				double iTauwave_gwave = iTau_tid1 * r1h2 / (1.0 + 0.25 * chi3);
				iTau_tid1 *= 3.8 * (1.0 - 0.683 * chi3 * chi) / (1.0 + 0.269329 * chi3 * chi * chi);

				a_th += -0.5 * iTau_tid1 * v_kep; //circular

				a_r += (0.104 * v_rel_th + 0.176 * v_rel_r) * iTauwave_gwave;
				a_th += (-1.736 * v_rel_th + 0.325 * v_rel_r) * iTauwave_gwave;
				a_z += (-1.088 * v_rel3.z - 0.871 * v_kep * x4.z / r1) * iTauwave_gwave;
			}
		}
		a_x += (x4.x * a_r - x4.y * a_th) / r1;
		a_y += (x4.y * a_r + x4.x * a_th) / r1;
	}
	__syncthreads();

	if(id < N + Nstart && x4.w >= 0.0){

		double v2 = v4.x * v4.x + v4.y * v4.y + v4.z * v4.z;
		//Kick
		v4.x += a_x * dt;
		v4.y += a_y * dt;
		v4.z += a_z * dt;

		double v2B = v4.x * v4.x + v4.y * v4.y + v4.z * v4.z;

		v4_d[id] = v4;
		if(x4.w > 0.0){
			U = 0.5 * x4.w * (v2 - v2B);
			Energy_d[id] += U;
		}
	}
}


// ************************************
//this kernel sums up all the Energy loss due to the Gas Disc and adds to the internal Energy
//Author: Simon Grimm
//August 2016
// *************************************
template < int Bl2 >
__global__ void gasEnergy_kernel(double *Energy_d, double *U_d, double *test_d, int st, int N){

	int idy = threadIdx.x;

	__shared__ volatile double U_s[Bl2];

	for(int i = 0; i < Bl2; i += blockDim.x){
		U_s[idy + i] = 0.0;
	}

	__syncthreads();

	for(int i = 0; i < N ;i += blockDim.x){
		if(idy + i < N){
			U_s[idy] += Energy_d[idy + i];
		}
	}
//printf("%d %g\n", idy, U_s[idy]);
	__syncthreads();
	int s = blockDim.x/2;
	for(int i = 6; i < log2f(blockDim.x); ++i){
		if( idy < s ) {
			U_s[idy] += U_s[idy + s];
		}
		__syncthreads();
		s /= 2;
	}

	if(idy < 32){
		if(blockDim.x >= 64) U_s[idy] += U_s[idy + 32];
		if(blockDim.x >= 32) U_s[idy] += U_s[idy + 16];
		if(blockDim.x >= 16) U_s[idy] += U_s[idy + 8];
		if(blockDim.x >= 8) U_s[idy] += U_s[idy + 4];
		if(blockDim.x >= 4) U_s[idy] += U_s[idy + 2];
		if(blockDim.x >= 2) U_s[idy] += U_s[idy + 1];
	}

	__syncthreads();

	if(idy == 0){
		U_d[0] += U_s[0];
	}
	for(int i = 0; i < N ;i += blockDim.x){
		if(idy + i < N){
			Energy_d[idy + i] = 0.0;
		}
	}

}

//This function calls the Gas Energy kernel
__host__ void Data::gasEnergyCall(int NB, double* Energy_d, double *test_d, double *U_d, cudaStream_t hstream, int st, int N, int Nsmall){

	if(Nsmall == 0){
		switch(NB){
			case 16:{
				gasEnergy_kernel< 32> <<< 1, 16, 0, hstream>>> (Energy_d, U_d, test_d, st, N);
			};
			break;
			case 32:{
				gasEnergy_kernel< 64> <<< 1, 32, 0, hstream>>> (Energy_d, U_d, test_d, st, N);
			};
			break;
			case 64:{
				gasEnergy_kernel< 64> <<< 1, 64, 0, hstream>>> (Energy_d, U_d, test_d, st, N);
			};
			break;
			case 128:{
				gasEnergy_kernel< 128> <<< 1, 128, 0, hstream>>> (Energy_d, U_d, test_d, st, N);
			};
			break;
			case 256:{
				gasEnergy_kernel< 256> <<< 1, 256, 0, hstream>>> (Energy_d, U_d, test_d, st, N);
			};
			break;
			case 512:{
				gasEnergy_kernel< 512> <<< 1, 512, 0, hstream>>> (Energy_d, U_d, test_d, st, N);
			};
			break;
			case 1024:{
				gasEnergy_kernel< 512> <<< 1, 512, 0, hstream>>> (Energy_d, U_d, test_d, st, N);
			};
			break;
			case 2048:{
				gasEnergy_kernel< 512> <<< 1, 512, 0, hstream>>> (Energy_d, U_d, test_d, st, N);
			};
			break;
		}
		if(NB > 2048){
			gasEnergy_kernel< 512> <<< 1, 512, 0, hstream>>> (Energy_d, U_d, test_d, st, N);
		}
	}
	else{
			gasEnergy_kernel< 512> <<< 1, 512, 0, hstream>>> (Energy_d, U_d, test_d, st, N + Nsmall);
	}
}
__host__ void Data::gasEnergyMCall(int NB, double* Energy_d, double *test_d, double *U_d, cudaStream_t hstream, int st, int N){
	gasEnergy_kernel < 2 * NmaxM > <<< 1, NmaxM, 0, hstream>>> (Energy_d, U_d, test_d, st, N);
}

__host__ void Data::GasAccCall(double *time_d, double *dt_d, double Ct){
	int nt = min(32, NB[0]);
	GasAcc <<< (N_h[0] + nt - 1) / nt , nt >>> (x4_d, v4_d, index_d, GasDisk_d, GasAcc_d, time_d, Msun_d, dt_d, N_h[0], Energy_d, P.G_dTau_diss, P.G_alpha, P.G_Sigma_10, Nst, Ct, P.UsegasPotential, P.UsegasEnhance, P.UsegasDrag, P.UsegasTidalDamping, P.G_Mgiant, 0);
}
__host__ void Data::GasAccCall_small(double *time_d, double *dt_d, double Ct){
	if(Nsmall_h[0] > 0) GasAcc <<<(Nsmall_h[0] + 127)/128, 128 >>> (x4_d + N_h[0], v4_d + N_h[0], index_d + N_h[0], GasDisk_d, GasAcc_d, time_d, Msun_d, dt_d, Nsmall_h[0], Energy_d, P.G_dTau_diss, P.G_alpha, P.G_Sigma_10, Nst, Ct, P.UsegasPotential, P.UsegasEnhance, P.UsegasDrag, P.UsegasTidalDamping, P.G_Mgiant, 0);
}
__host__ void Data::GasAccCall_M(double *time_d, double *dt_d, double Ct){
	GasAcc <<< (NT + 127) / 128, 128 >>> (x4_d, v4_d, index_d, GasDisk_d, GasAcc_d, time_d, Msun_d, dt_d, NT, Energy_d, P.G_dTau_diss, P.G_alpha, P.G_Sigma_10, Nst, Ct, P.UsegasPotential, P.UsegasEnhance, P.UsegasDrag, P.UsegasTidalDamping, P.G_Mgiant, Nstart);
}
__host__ void Data::GasAccCall2_small(double *time_d, double *dt_d, double Ct){
	if(Nsmall_h[0] > 0) GasAcc2 <<<(Nsmall_h[0] + 127)/128, 128 >>> (x4_d + N_h[0], v4_d + N_h[0], index_d + N_h[0], time_d, Msun_d, dt_d, Nsmall_h[0], Energy_d, Nst, Ct, GasDatanr, GasDatatime, GasData_d, P.G_alpha, P.UsegasPotential, P.UsegasEnhance, P.UsegasDrag, P.UsegasTidalDamping, P.G_Mgiant, 0);
}

__host__ int Data::freeGas(){
	cudaError_t error;
	
	free(Gas_rg_h);
	free(Gas_zg_h);
	free(Gas_rho_h);
	free(GasDisk_h);
	free(GasAcc_h);

	cudaFree(Gas_rg_d);
	cudaFree(Gas_zg_d);
	cudaFree(Gas_rho_d);
	cudaFree(GasDisk_d);
	cudaFree(GasAcc_d);
	error = cudaGetLastError();
	if(error != 0){
		printf("Cuda Gas free error = %d = %s\n",error, cudaGetErrorString(error));
		fprintf(masterfile, "Cuda Gas free error = %d = %s\n",error, cudaGetErrorString(error));
		return 0;
	}
	return 1;

}


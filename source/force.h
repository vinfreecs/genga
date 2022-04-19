#include "Host2.h"
// *******************************************************
// This is a template function for additional forces
// The velocities in this kernel are already converted to heliocentric coordinates
//
// non canonical perturbations are treated in a symplectic way by the implicicit midpoint
// method, according to Mikkola 1997
//
// si = 0 is used in tunig step, and no global variables are updated
//
// Use vold_d as temporary storage for Spinsun
//
// October 2021
// Authors: Simon Grimm, Jean-Baptiste Delisle
// **********************************************************
__global__ void force(double4 *x4_d, double4 *v4_d, int *index_d, double4 *spin_d, double3 *love_d, double2 *Msun_d, double4 *Spinsun_d, double3 *Lovesun_d, double2 *J2_d, double4 *vold_d, double *dt_d, double Kt, double *time_d, int N, int Nst, int UseGR, int UseTides, int UseRotationalDeformation, int Nstart, int si){
	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy + Nstart;

	double3 T3sun = {0.0, 0.0, 0.0};

	int st = 0;
	double dt = 0.0;

	if(id < N + Nstart){
	
		int index = index_d[id];
		if(Nst > 1) st = index / def_MaxIndex;	//st is the sub simulation index

		double4 x4 = x4_d[id];
		double4 v4 = v4_d[id];
		double4 Spin = spin_d[id];
		double Msun = Msun_d[st].x;			//This is the mass of the central star
		double4 Spinsun = Spinsun_d[st];		//This is the spin of the central star and the moment of inertia
		double3 Lovesun = Lovesun_d[st];		//This is the Love number, fluid Love numer and time lag
		double2 J2s = J2_d[st];			//This is the J2 value for additional gravitational harmonics forces and the mean radius
		dt = dt_d[st] * Kt;			//This is the time step to do
//		double time = time_d[st] / 365.25;		//This is the time in years

		double3 a3;
		a3.x = 0.0; 	
		a3.y = 0.0;
		a3.z = 0.0;

		double3 T3;	//torque
		T3.x = 0.0;
		T3.y = 0.0;
		T3.z = 0.0;

		// **********************************************************
		// prepare first all values which dont depend on the velocity
		// **********************************************************

		double rsq = (x4.x * x4.x + x4.y * x4.y + x4.z * x4.z);
		double ir = 1.0 / sqrt(rsq);

		double A, B;

		if(UseGR == 1 && x4.w >= 0.0){
			// GR symplectic
			// GR part depending on position only (see Saha & Tremaine 1994)
			double mu = def_ksq * (Msun + x4.w);
			A = mu/(rsq * def_cm);
			B = 2.0 * A * A;
			a3.x -= B * x4.x;
			a3.y -= B * x4.y;
			a3.z -= B * x4.z;
		}

		if(UseGR == 3 && x4.w >= 0.0){
			// GR force
			// GR  see Fabrycky 2010 equation 2
			double csq = def_cm * def_cm;

			A = def_ksq * (Msun + x4.w) * ir;
			B = A * ir / csq;
			double eta = Msun * x4.w / ((Msun + x4.w) * (Msun + x4.w));
			double vsq = (v4.x * v4.x + v4.y * v4.y + v4.z * v4.z);
			double rd = (x4.x * v4.x + x4.y * v4.y + x4.z * v4.z) * ir; 

			double C = 2.0 * (2.0 - eta) * rd;
			double D = (1.0 + 3.0 * eta) * vsq - 1.5 * eta * rd * rd - 2.0 * (2.0 + eta) * A;
			a3.x += B * (C * v4.x - D * x4.x * ir); 	
			a3.y += B * (C * v4.y - D * x4.y * ir);
			a3.z += B * (C * v4.z - D * x4.z * ir);
		}

		double eta;
		if(UseGR == 2 && x4.w >= 0.0){
			// GR  see Fabrycky 2010 equation 2
			//first part of implicit function
			double csq = def_cm * def_cm;

			A = def_ksq * (Msun + x4.w) * ir;
			B = A * ir / csq;
			eta = Msun * x4.w / ((Msun + x4.w) * (Msun + x4.w));
		}



		if((UseTides == 2) && x4.w > 0.0){
			//Tidal force see Fabrycky 2010 equation 3
			double R2 = v4.w * v4.w;
			double R5 = R2 * R2 * v4.w;
			double ir3 = ir * ir * ir;
			double ir7 = ir3 * ir3 * ir;

			double E = -3.0 * love_d[id].x * def_ksq * Msun * Msun * R5 / x4.w * ir7;
//printf("%d %g %g\n", id, E * ir, E * x4.x * ir);
			a3.x += E * x4.x * ir;
			a3.y += E * x4.y * ir;	
			a3.z += E * x4.z * ir;
		}

		if(UseRotationalDeformation == 2){
			//Rotational Force see Fabrycky 2010 equation 4
			double Rsun2 = Msun_d[st].y * Msun_d[st].y;
			double Rsun5 = Rsun2 * Rsun2 * Msun_d[st].y;
			double lovesun = Lovesun.x;
			double ir2 = ir * ir;
			double ir4 = ir2 * ir2;

			//compute rotation vector from spin vector
			double Icsun = Spinsun.w;
			double iI = 1.0 / (Icsun * Msun * Rsun2); // inverse Moment of inertia of a solid sphere in 1/ (Solar Masses AU^2)
			double3 omegasun3;
			omegasun3.x = Spinsun.x * iI;
			omegasun3.y = Spinsun.y * iI;
			omegasun3.z = Spinsun.z * iI;

			double omegasun2 = omegasun3.x * omegasun3.x + omegasun3.y * omegasun3.y + omegasun3.z * omegasun3.z; 	//angular velocity in 1 / day * 0.017

			double F = -0.5 * lovesun * omegasun2 * Rsun5 * ir4;

			a3.x += F * x4.x * ir;
			a3.y += F * x4.y * ir;	
			a3.z += F * x4.z * ir;
		}

		double3 omega3;
		double3 omegasun3;
		double Rsun2, Rsun3, Rsun5;
		double R2, R3, R5;
		double ir2, ir3, ir5, ir7, ir8;
		double F1;
		double P, Psun;
		double3 t2, t3;

		double iIsun, iI;

		if((UseTides == 1 || UseRotationalDeformation == 1 || J2s.x != 0.0) && x4.w >= 0.0){
			Rsun2 = Msun_d[st].y * Msun_d[st].y;
			Rsun3 = Rsun2 * Msun_d[st].y;
			Rsun5 = Rsun3 * Rsun2;

			R2 = v4.w * v4.w;
			R3 = R2 * v4.w;
			R5 = R3 * R2;

			//compute rotation vector from spin vector
			double Icsun = Spinsun.w;
			iIsun = 1.0 / (Icsun * Msun * Rsun2); // inverse Moment of inertia of a solid sphere in 1/ (Solar Masses AU^2)
			omegasun3.x = Spinsun.x * iIsun;
			omegasun3.y = Spinsun.y * iIsun;
			omegasun3.z = Spinsun.z * iIsun;
//printf("omegaS %d %g %g %g\n", id, Spinsun.z, 1.0 / omegasun3.z / dayUnit, 1.0/iIsun, omegasun3.z * dayUnit);

			//compute rotation vector from spin vector
			if(x4.w > 0.0){
				double Ic = Spin.w;
				iI = 1.0 / (Ic * x4.w * R2); // inverse Moment of inertia of a solid sphere in 1/ (Solar Masses AU^2)
				omega3.x = Spin.x * iI;
				omega3.y = Spin.y * iI;
				omega3.z = Spin.z * iI;
			}
			else{
				iI = 0.0;
				omega3.x = 0.0;
				omega3.y = 0.0;
				omega3.z = 0.0;
			}
//printf("omegaP %d %g %g %g\n", id, omega3.x * dayUnit, omega3.y * dayUnit, omega3.z * dayUnit);

			ir2 = ir * ir;
			ir3 = ir2 * ir;
			ir5 = ir3 * ir2;
			ir7 = ir5 * ir2;
			ir8 = ir5 * ir3;
		}
		if((UseTides == 1) && x4.w > 0.0){
			//Tidal Force see Bolmont et al 2015 equation 5
			double Msun2 = Msun * Msun;
			double lovesun = Lovesun.x;
			double tausun = Lovesun.z;

			double m2 = x4.w * x4.w;
			double love = love_d[id].x;
			double tau = love_d[id].z;

			volatile double tsun = 3.0 * def_ksq * m2 * Rsun5 * ir8 * lovesun / x4.w;
			volatile double t = 3.0 * def_ksq * Msun2 * R5 * ir8 * love / x4.w;
//printf("tidal %d %g %g | %g %g | %g %g\n", id, love, tau, lovesun, tausun, tsun, t);


//volatile double tt = 3.0 * def_ksq * Msun2 * R5 * ir7 * love * tau;

//in SI units
//double Msun2si = Msun2 * def_Solarmass * def_Solarmass;
//double tausi = tau * 60.0 * 60.0 * 24.0 / 0.017;
//double G = 6.674E-11;
//volatile double ttsi = 3.0 * G * Msun2si * R5 * ir7 * love * tausi / (def_AU * def_AU);	//R5 * AU^5 * ir7 / AU^7 = R5 * ir7 / AU^2
//printf("%d %g %g %g %g %g %g %g %g\n", id, tausi, Msun, v4.w, 1.0/ir, love, tau, tt, ttsi);

			Psun = tsun * tausun;
			P = t * tau;
			F1 = -tsun - t;

			//F1 * x4i.xyz is the nondissipative radial part of the acceleration

		}

		double mred = Msun / (x4.w + Msun);

		// **********************************************************
		// Now add the velocity dependent terms and iterate the 
		// implicit midpoint method
		// **********************************************************


		if((UseTides == 1 || UseRotationalDeformation == 1 || UseGR == 2 || J2s.x != 0.0) && x4.w >= 0.0){
			double3 a3t, a3told;
			double4 v4t = v4;
			a3told.x = 0.0;
			a3told.y = 0.0;
			a3told.z = 0.0;

			double3 T3t, T3sunt;	//torque
			double3 T3told, T3suntold;	//torque
			T3told.x = 0.0;
			T3told.y = 0.0;
			T3told.z = 0.0;
			T3suntold.x = 0.0;
			T3suntold.y = 0.0;
			T3suntold.z = 0.0;


			double4 Spint = Spin;
			double4 Spinsunt = Spinsun;

			
			for(int k = 0; k < 30; ++k){
			//for(int k = 0; k < 1; ++k){
			
				a3t.x = 0.0;
				a3t.y = 0.0;
				a3t.z = 0.0;
				T3t.x = 0.0;
				T3t.y = 0.0;
				T3t.z = 0.0;
				T3sunt.x = 0.0;
				T3sunt.y = 0.0;
				T3sunt.z = 0.0;
	
				if(UseGR == 2){
					// GR  see Fabrycky 2010 equation 2
					double vsq = (v4t.x * v4t.x + v4t.y * v4t.y + v4t.z * v4t.z);
					double rd = (x4.x * v4t.x + x4.y * v4t.y + x4.z * v4t.z) * ir; 

					double C = 2.0 * (2.0 - eta) * rd;
					double D = (1.0 + 3.0 * eta) * vsq - 1.5 * eta * rd * rd - 2.0 * (2.0 + eta) * A;

					a3t.x += B * (C * v4t.x - D * x4.x * ir); 	
					a3t.y += B * (C * v4t.y - D * x4.y * ir);
					a3t.z += B * (C * v4t.z - D * x4.z * ir);
				}
	
				if(UseTides == 1 && x4.w > 0.0){
					//Tidal Force see Bolmont et al 2015 equation 6
			
					t2.x = ( omega3.y * x4.z) - (omega3.z * x4.y);
					t2.y = (-omega3.x * x4.z) + (omega3.z * x4.x);
					t2.z = ( omega3.x * x4.y) - (omega3.y * x4.x);
					
					t3.x = ( omegasun3.y * x4.z) - (omegasun3.z * x4.y);
					t3.y = (-omegasun3.x * x4.z) + (omegasun3.z * x4.x);
					t3.z = ( omegasun3.x * x4.y) - (omegasun3.y * x4.x);
//printf("tidal %d %g %g %g %g %g %g\n", id, t2.x, t2.y, t2.z, t3.x, t3.y, t3.z);

					double rv = x4.x * v4t.x + x4.y * v4t.y + x4.z * v4t.z;
					double F2 = F1 - 2.0 * rv * ir2 * (Psun + P);  // -3 + 1 = -2

					a3t.x += (F2 * x4.x + P * t2.x + Psun * t3.x - (P + Psun) * v4t.x);
					a3t.y += (F2 * x4.y + P * t2.y + Psun * t3.y - (P + Psun) * v4t.y);
					a3t.z += (F2 * x4.z + P * t2.z + Psun * t3.z - (P + Psun) * v4t.z);

					//spin evolution
					double fdx = (P * t2.x - P * v4t.x) * x4.w;
					double fdy = (P * t2.y - P * v4t.y) * x4.w;
					double fdz = (P * t2.z - P * v4t.z) * x4.w;
//printf("P %d %d %g %g %g %g %g\n", id, k, P, t2.x, v4t.x, x4.x, mred);

					T3t.x += -mred * ( x4.y * fdz - x4.z * fdy);
					T3t.y += -mred * (-x4.x * fdz + x4.z * fdx);
					T3t.z += -mred * ( x4.x * fdy - x4.y * fdx);

					fdx = (Psun * t3.x - Psun * v4t.x) * x4.w;
					fdy = (Psun * t3.y - Psun * v4t.y) * x4.w;
					fdz = (Psun * t3.z - Psun * v4t.z) * x4.w;

					T3sunt.x += -mred * ( x4.y * fdz - x4.z * fdy);
					T3sunt.y += -mred * (-x4.x * fdz + x4.z * fdx);
					T3sunt.z += -mred * ( x4.x * fdy - x4.y * fdx);

//printf("T %d %d %g %g %g | %g %g %g\n", id, k, T3t.x, T3t.y, T3t.z, T3sunt.x, T3sunt.y, T3sunt.z);
				}
				if((UseRotationalDeformation == 1) && x4.w > 0.0){
					//Rotational Force see Bolmont et al 2015 equation 15
					double lovesunf = Lovesun.y;
					double lovef = love_d[st].y;

					double omegasun2 = omegasun3.x * omegasun3.x + omegasun3.y * omegasun3.y + omegasun3.z * omegasun3.z; 	//angular velocity in 1 / day * 0.017

					double omega2 = omega3.x * omega3.x + omega3.y * omega3.y + omega3.z * omega3.z; 	//angular velocity in 1 / day * 0.017
					volatile double Csun = x4.w * lovesunf * omegasun2 * Rsun5 / 6.0;
					volatile double Cp = Msun * lovef * omega2 * R5 / 6.0;
//double J2 = lovef * omega2 * R3 / (3.0 * x4.w);
//double J2sun = lovesunf * omegasun2 * Rsun3 / (3.0 * Msun);
//printf("J2 %d %g %g\n", id, J2, J2sun);


					volatile double r_omegasun = x4.x * omegasun3.x + x4.y * omegasun3.y + x4.z * omegasun3.z;
					volatile double r_omega = x4.x * omega3.x + x4.y * omega3.y + x4.z * omega3.z;

					volatile double F1 = -3.0 * ir5 * (Csun + Cp);
					if(omegasun2 != 0.0){
						F1 += 15.0 * ir7 * Csun * r_omegasun * r_omegasun / omegasun2;
					}
					if(omega2 != 0.0){
						F1 += 15.0 * ir7 * Cp * r_omega * r_omega / omega2; 
					}

					volatile double F2 = 0.0;
					volatile double F3 = 0.0;
					if(omegasun2 != 0.0){
						F2 = -6.0 * ir5 * Csun * r_omegasun / omegasun2;
					}
					if(omega2 != 0.0){
						F3 = -6.0 * ir5 * Cp * r_omega / omega2;
					}
//printf("F %d %g %g %g %g %g %g\n", id, F3, F2, Cp, Csun, r_omegasun, r_omega);
					

					a3t.x += (F1 * x4.x + F2 * omegasun3.x + F3 * omega3.x) / x4.w;
					a3t.y += (F1 * x4.y + F2 * omegasun3.y + F3 * omega3.y) / x4.w;
					a3t.z += (F1 * x4.z + F2 * omegasun3.z + F3 * omega3.z) / x4.w;


					//spin evolution
					T3t.x += -mred * F3 * ( x4.y * omega3.z - x4.z * omega3.y);
					T3t.y += -mred * F3 * (-x4.x * omega3.z + x4.z * omega3.x);
					T3t.z += -mred * F3 * ( x4.x * omega3.y - x4.y * omega3.x);

					T3sunt.x += -mred * F2 * ( x4.y * omegasun3.z - x4.z * omegasun3.y);
					T3sunt.y += -mred * F2 * (-x4.x * omegasun3.z + x4.z * omegasun3.x);
					T3sunt.z += -mred * F2 * ( x4.x * omegasun3.y - x4.y * omegasun3.x);
//printf("T %d %d %g %g |  %g %g %g | %g %g %g\n", id, 0, F3, F2, T3.x, T3.y, T3.z, T3sun.x, T3sun.y, T3sun.z);
				}

				if(J2s.x != 0.0){
					//Additional J2 for secular evolution of planets, See Zderic and Madigan 2020, eq 1 and 2
					double D = 0.5 * Msun * J2s.x * J2s.y * J2s.y;
					volatile double F4 = -3.0 * D * ir5 + 15.0 * D * ir7 * x4.z * x4.z;
				
					a3t.x += F4 * x4.x;
					a3t.y += F4 * x4.y;
					a3t.z += (F4 * x4.z - 6.0 * D * ir5) * x4.z;
				}

				v4t.x = v4.x + 0.5 * dt * a3t.x;
				v4t.y = v4.y + 0.5 * dt * a3t.y;
				v4t.z = v4.z + 0.5 * dt * a3t.z;

				Spint.x = Spin.x + 0.5 * dt * T3t.x;
				Spint.y = Spin.y + 0.5 * dt * T3t.y;
				Spint.z = Spin.z + 0.5 * dt * T3t.z;

//sum over T3sun here

				Spinsunt.x = Spinsun.x + 0.5 * dt * T3sunt.x;
				Spinsunt.y = Spinsun.y + 0.5 * dt * T3sunt.y;
				Spinsunt.z = Spinsun.z + 0.5 * dt * T3sunt.z;

				omega3.x = Spint.x * iI;
				omega3.y = Spint.y * iI;
				omega3.z = Spint.z * iI;
	
				omegasun3.x = Spinsunt.x * iIsun;
				omegasun3.y = Spinsunt.y * iIsun;
				omegasun3.z = Spinsunt.z * iIsun;


				int stop = 1;
				if(fabs(a3t.x - a3told.x) >= 1.0e-15) stop = 0;
				if(fabs(a3t.y - a3told.y) >= 1.0e-15) stop = 0;
				if(fabs(a3t.z - a3told.z) >= 1.0e-15) stop = 0;

				if(fabs(T3t.x - T3told.x) >= 1.0e-15) stop = 0;
				if(fabs(T3t.y - T3told.y) >= 1.0e-15) stop = 0;
				if(fabs(T3t.z - T3told.z) >= 1.0e-15) stop = 0;

				if(fabs(T3sunt.x - T3suntold.x) >= 1.0e-15) stop = 0;
				if(fabs(T3sunt.y - T3suntold.y) >= 1.0e-15) stop = 0;
				if(fabs(T3sunt.z - T3suntold.z) >= 1.0e-15) stop = 0;

				//if(fabs(a3t.x - a3told.x) < 1.0e-15 && fabs(a3t.y - a3told.y) < 1.0e-15 && fabs(a3t.z - a3told.z) < 1.0e-15){
				if(stop == 1){
//if(k > 1) printf("k %d %d\n", id, k);
					break;
				}

				a3told = a3t;
				T3told = T3t;
				T3suntold = T3sunt;
//printf("tidal2 %d %g %g %g %g\n", id, x4.w, a3t.x, a3t.y, a3t.z);
			}

			a3.x += a3t.x;
			a3.y += a3t.y;
			a3.z += a3t.z;

			T3.x += T3t.x;
			T3.y += T3t.y;
			T3.z += T3t.z;

			T3sun.x += T3sunt.x;
			T3sun.y += T3sunt.y;
			T3sun.z += T3sunt.z;
		}

		//apply the Kick
		v4.x += a3.x * dt;
		v4.y += a3.y * dt;
		v4.z += a3.z * dt;

		Spin.x += T3.x * dt;
		Spin.y += T3.y * dt;
		Spin.z += T3.z * dt;


//printf("Force %d %g %g %g %g\n", id, x4.w, a3.x, a3.y, a3.z);
// printf("Force %d %g %g %g %g\n", id, x4.w, v4.x, v4.y, v4.z);
		if(si == 1){
			v4_d[id] = v4;
			spin_d[id] = Spin;
		}
	}

	__syncthreads();

	//Sum up all torques of the star
	if(UseTides > 0 || UseRotationalDeformation > 0){
//printf("A %d %d %g %g %g\n", id, 0, T3sun.x * dt, T3sun.y * dt, T3sun.z * dt);
		if(Nst == 1){
			for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
				T3sun.x += __shfl_xor_sync(0xffffffff, T3sun.x, i, warpSize);
				T3sun.y += __shfl_xor_sync(0xffffffff, T3sun.y, i, warpSize);
				T3sun.z += __shfl_xor_sync(0xffffffff, T3sun.z, i, warpSize);
#else
				T3sun.x += __shfld_xor(T3sun.x, i);
				T3sun.y += __shfld_xor(T3sun.y, i);
				T3sun.z += __shfld_xor(T3sun.z, i);
#endif
//printf("A1 %d %d %g %g %g\n", id, i, T3sun.x, T3sun.y, T3sun.z);
			}
			__syncthreads();
//printf("A1 %d %g %g %g\n", id, T3sun.x * dt, T3sun.y * dt, T3sun.z * dt);

			if(blockDim.x > warpSize){
				//reduce across warps
				extern __shared__ double3 ForceT_s[];
				double3 *T3sun_s = ForceT_s;

				int lane = threadIdx.x % warpSize;
				int warp = threadIdx.x / warpSize;
				if(warp == 0){
					T3sun_s[threadIdx.x].x = 0.0;
					T3sun_s[threadIdx.x].y = 0.0;
					T3sun_s[threadIdx.x].z = 0.0;
				}
				__syncthreads();

				if(lane == 0){
					T3sun_s[warp] = T3sun;
				}

				__syncthreads();
				//reduce previous warp results in the first warp
				if(warp == 0){
					T3sun = T3sun_s[threadIdx.x];
					for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
						T3sun.x += __shfl_xor_sync(0xffffffff, T3sun.x, i, warpSize);
						T3sun.y += __shfl_xor_sync(0xffffffff, T3sun.y, i, warpSize);
						T3sun.z += __shfl_xor_sync(0xffffffff, T3sun.z, i, warpSize);
#else
						T3sun.x += __shfld_xor(T3sun.x, i);
						T3sun.y += __shfld_xor(T3sun.y, i);
						T3sun.z += __shfld_xor(T3sun.z, i);
#endif
					}
					if(lane == 0){
						T3sun_s[0] = T3sun;
					}
//printf("B1 %d %g %g %g\n", id, T3sun.x * dt, T3sun.y * dt, T3sun.z * dt);
				}
				__syncthreads();

				T3sun = T3sun_s[0];
			}
			__syncthreads();


			if(N <= blockDim.x && id == 0){
				if(si == 0) printf("B %d %.20g %.20g %.20g\n", id, T3sun.x * dt, T3sun.y * dt, T3sun.z * dt);
				if(si == 1){
					Spinsun_d[st].x += T3sun.x * dt;
					Spinsun_d[st].y += T3sun.y * dt;
					Spinsun_d[st].z += T3sun.z * dt;
				}
			}
			else if(threadIdx.x == 0){
				vold_d[blockIdx.x].x = T3sun.x * dt;
				vold_d[blockIdx.x].y = T3sun.y * dt;
				vold_d[blockIdx.x].z = T3sun.z * dt;
			}

		}
		else{ //Nst > 1
			vold_d[id].x = T3sun.x * dt;
			vold_d[id].y = T3sun.y * dt;
			vold_d[id].z = T3sun.z * dt;
		}
	}
}

__global__ void forced2_kernel(double4 *vold_d, double4 *Spinsun_d, const int N, int si){

	int idy = threadIdx.x;

	double3 p = {0.0, 0.0, 0.0};

	extern __shared__ double3 forced2_s[];
	double3 *p_s = forced2_s;

	int lane = threadIdx.x % warpSize;
	int warp = threadIdx.x / warpSize;

	if(warp == 0){
		p_s[threadIdx.x].x = 0.0;
		p_s[threadIdx.x].y = 0.0;
		p_s[threadIdx.x].z = 0.0;
	}


	for(int i = 0; i < N; i += blockDim.x){
		if(idy + i < N){
			p.x += vold_d[idy + i].x;
			p.y += vold_d[idy + i].y;
			p.z += vold_d[idy + i].z;
		}
	}

	__syncthreads();
//printf("AA %d %.20g %.20g %.20g\n", idy, p.x, p.y, p.z);

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
//if(i >= 16) printf("d2A %d %d %.20g\n", idy, i, p.x);
	}
	__syncthreads();

	if(blockDim.x > warpSize){
		//reduce across warps
		if(lane == 0){
			p_s[warp] = p;
		}
		__syncthreads();
		//reduce previous warp results in the first warp
		if(warp == 0){
			p = p_s[threadIdx.x];
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
//if(i >= 16) printf("d2B %d %d %.20g\n", idy, i, p.x);
			}
		}
	}
	__syncthreads();

	if(threadIdx.x == 0 && si == 0) printf("BB %d %.20g %.20g %.20g\n", idy, p.x, p.y, p.z);

	if(threadIdx.x == 0 && si == 1){
		Spinsun_d[0].x += p.x;
		Spinsun_d[0].y += p.y;
		Spinsun_d[0].z += p.z;
	}
}



template <int Bl, int Bl2, int Nmax >
__global__ void forceM_kernel(double4 *vold_d, int *index_d, double4 *Spinsun_d, int *NBS_d, int NT, int Nstart){

	int idy = threadIdx.x;
	int id = blockIdx.x * Bl2 + idy - Nmax + Nstart;
	__shared__ volatile double3 p_s[Bl + Nmax / 2];
	__shared__ int st_s[Bl + Nmax / 2];
	int NBS;

	if(id < NT + Nstart && id >= Nstart){
		st_s[idy] = index_d[id] / def_MaxIndex;
		p_s[idy].x = vold_d[id].x;
		p_s[idy].y = vold_d[id].y;
		p_s[idy].z = vold_d[id].z;
		NBS = NBS_d[st_s[idy]];
//printf("TA1 %d %d %g %g %g\n", id, idy, p_s[idy].x, p_s[idy].y, p_s[idy].z);

	}
	else{
		st_s[idy] = -idy-1;
		p_s[idy].x = 0.0;
		p_s[idy].y = 0.0;
		p_s[idy].z = 0.0;
		NBS = -1;
	}
	//halo

	if(idy < Nmax / 2){
		//right
		if(id + Bl < NT + Nstart){
			st_s[idy + Bl] = index_d[id + Bl] / def_MaxIndex;
			p_s[idy + Bl].x = vold_d[id + Bl].x;
			p_s[idy + Bl].y = vold_d[id + Bl].y;
			p_s[idy + Bl].z = vold_d[id + Bl].z;
		}
		else{
			st_s[idy + Bl] = -idy-Bl-1;
			p_s[idy + Bl].x = 0.0;
			p_s[idy + Bl].y = 0.0;
			p_s[idy + Bl].z = 0.0;
		}
//printf("TA2 %d %d %g %g %g\n", id, idy, p_s[idy + Bl].x, p_s[idy + Bl].y, p_s[idy + Bl].z);
	}
	volatile int f;
	volatile double px;
	volatile double py;
	volatile double pz;
	if(Nmax >= 64){
		__syncthreads();
		f = ((st_s[idy] - st_s[idy + 32]) == 0);		//one if sti == stj, zero else
		px = (p_s[idy + 32].x) * f;	
		py = (p_s[idy + 32].y) * f;
		pz = (p_s[idy + 32].z) * f;

		__syncthreads();
	
		p_s[idy].x += px;
		p_s[idy].y += py;
		p_s[idy].z += pz;
//printf("TA3 %d %d %g %g %g\n", id, idy, p_s[idy].x, p_s[idy].y, p_s[idy].z);
	}

	if(Nmax >= 32){
		__syncthreads();
		f = ((st_s[idy] - st_s[idy + 16]) == 0);		//one if sti == stj, zero else
		px = (p_s[idy + 16].x) * f;	
		py = (p_s[idy + 16].y) * f;
		pz = (p_s[idy + 16].z) * f;

		__syncthreads();
	
		p_s[idy].x += px;
		p_s[idy].y += py;
		p_s[idy].z += pz;
	}

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
	//sum is complete
//printf("T %d %d %g %g %g\n", id, idy, p_s[idy].x, p_s[idy].y, p_s[idy].z);

	if(id == NBS && NBS >= Nstart && idy >= Nmax && idy < Bl - Nmax / 2){
//printf("TT %d %d %g %g %g\n", id, idy, p_s[idy].x, p_s[idy].y, p_s[idy].z);
		Spinsun_d[st_s[idy]].x += p_s[idy].x;
		Spinsun_d[st_s[idy]].y += p_s[idy].y;
		Spinsun_d[st_s[idy]].z += p_s[idy].z;
	}
}



__constant__ int setElementsNumbers_c[3];
__constant__ int setElements_c[25];
//**************************************
// This function copies the setElements parameters to constant memory. This functions must be in
// the same file as the use of the constant memory
//
//June 2015
//Authors: Simon Grimm
//***************************************/
__host__ void Host::constantCopy3(int *Elements, int nelements, int nbodies, int nlines){
	int setElementsNumbers[3] = {nelements, nbodies, nlines};	
	cudaMemcpyToSymbol(setElements_c, Elements, 25 * sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(setElementsNumbers_c, setElementsNumbers, 3 * sizeof(int), 0, cudaMemcpyHostToDevice);
}

// ***************************************************************
// This kernel converts the heliocentric coordinates into Keplerian elemtnts,
// modifies the Keplerian elements according to the setElementsData_d data and
// converts back to heliocentric coordinates.
//
// EE 0 = only x v, 1 Kepler elements + m, r
// March 2017
// Authors: Simon Grimm
// *****************************************************************
__global__ void setElements(double4 *x4_d, double4 *v4_d, int *index_d, double *setElementsData_d, int *setElementsLine_d, double2 *Msun_d, double *dt_d, double *time_d, int N, int Nst, int EE){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	int line = setElementsLine_d[0];
	int nelements = setElementsNumbers_c[0];
	int nbodies = setElementsNumbers_c[1];
	int nlines = setElementsNumbers_c[2];


	if(id < nbodies){

//printf("id %d, line %d, nelements %d, nbodies %d EE %d\n", id, line, nelements, nbodies, EE);

		//Compute the Kepler Elements
		int st = 0;

		if(Nst > 1 && id < N) st = index_d[id] / def_MaxIndex;	//st is the sub simulation index


		//check if one of the Keplerian elements will be modified
		int doConversion = 0;
		for(int i = 0; i < nelements; ++i){
//printf("elements %d %d\n", i, setElements_c[i]);
			if(setElements_c[i] == 3){
				//a
				doConversion = 1;
				break;
			}
			if(setElements_c[i] == 4){
				//e
				doConversion = 1;
				break;
			}
			if(setElements_c[i] == 5){
				//i
				doConversion = 1;
				break;
			}
			if(setElements_c[i] == 6){
				//O
				doConversion = 1;
				break;
			}
			if(setElements_c[i] == 7){
				//w
				doConversion = 1;
				break;
			}
			if(setElements_c[i] == 10){
				//T
				doConversion = 1;
				break;
			}
		}


		double4 x4i = x4_d[id];
		double4 v4i = v4_d[id];

		if(x4i.w >= 0.0){

			//int index = index_d[id];
			double Msun = Msun_d[st].x;
			double dt = dt_d[st];
			double time;
			if(EE == 0){
				time = (time_d[st] - dt / dayUnit) / 365.25;//time at beginning of the time step in years
			}
			else{
				time = time_d[st] / 365.25;//time at end of the time step in years
			}
			double mu = def_ksq * (Msun + x4i.w);

			double a, e, inc, Omega, w, Theta, E;
			double x, y, z;
			double vx, vy, vz;
			if(doConversion == 1){

				double rsq = x4i.x * x4i.x + x4i.y * x4i.y + x4i.z * x4i.z;
				double vsq = v4i.x * v4i.x + v4i.y * v4i.y + v4i.z * v4i.z;
				double u =  x4i.x * v4i.x + x4i.y * v4i.y + x4i.z * v4i.z;
				double ir = 1.0 / sqrt(rsq);
				double ia = 2.0 * ir - vsq / mu;

				a = 1.0 / ia;

				//inclination
				double3 h3;
				h3.x = ( x4i.y * v4i.z) - (x4i.z * v4i.y);
				h3.y = (-x4i.x * v4i.z) + (x4i.z * v4i.x);
				h3.z = ( x4i.x * v4i.y) - (x4i.y * v4i.x);

				double h = sqrt(h3.x * h3.x + h3.y * h3.y + h3.z * h3.z);

				double t = h3.z / h;
				if(t < -1.0) t = -1.0;
				if(t > 1.0) t = 1.0;
			
				inc = acos(t);

				//longitude of ascending node
				double n = sqrt(h3.x * h3.x + h3.y * h3.y);
				Omega = acos(-h3.y / n);
				if(h3.x < 0.0){
					Omega = 2.0 * M_PI - Omega;
				}

				if(inc < 1.0e-10 || n == 0) Omega = 0.0;

				//argument of periapsis
				double3 e3;
				e3.x = ( v4i.y * h3.z - v4i.z * h3.y) / mu - x4i.x * ir;
				e3.y = (-v4i.x * h3.z + v4i.z * h3.x) / mu - x4i.y * ir;
				e3.z = ( v4i.x * h3.y - v4i.y * h3.x) / mu - x4i.z * ir;
			
				e = sqrt(e3.x * e3.x + e3.y * e3.y + e3.z * e3.z); 

				t = (-h3.y * e3.x + h3.x * e3.y) / (n * e);
				if(t < -1.0) t = -1.0;
				if(t > 1.0) t = 1.0;
				w = acos(t);
				if(e3.z < 0.0) w = 2.0 * M_PI - w;
				if(n == 0) w = 0.0;

				//True Anomaly
				t = (e3.x * x4i.x + e3.y * x4i.y + e3.z * x4i.z) / e * ir;
				if(t < -1.0) t = -1.0;
				if(t > 1.0) t = 1.0;
				Theta = acos(t);
				if(u < 0.0){
					if(e < 1.0 - 1.0e-10){
						//elliptic
						Theta = 2.0 * M_PI - Theta;
					}
					else if(e > 1.0 + 1.0e-10){
						//hyperbolic
						Theta = -Theta;
					}
					else{
						//parabolic
						Theta = - Theta;
					}
				}

				//Non circular, equatorial orbit
				if(e > 1.0e-10 && inc < 1.0e-10){
					Omega = 0.0;
					w = acos(e3.x / e);
					if(e3.y < 0.0) w = 2.0 * M_PI - w;
				}
				
				//circular, inclinded orbit
				if(e < 1.0e-10 && inc > 1.0e-11){
					w = 0.0;
				}
				
				//circular, equatorial orbit
				if(e < 1.0e-10 && inc < 1.0e-11){
					w = 0.0;
					Omega = 0.0;
				}


				if(w == 0 && Omega != 0.0){
					t = (-h3.y * x4i.x + h3.x * x4i.y) / n * ir;
					if(t < -1.0) t = -1.0;
					if(t > 1.0) t = 1.0;
					Theta = acos(t);
					if(x4i.z < 0.0){
						if(e < 1.0 - 1.0e-10){
							//elliptic
							Theta = 2.0 * M_PI - Theta;
						}
						else if(e > 1.0 + 1.0e-10){
							//hyperbolic
							Theta = -Theta;
						}
						else{
							//parabolic
							Theta = -Theta;
						}
					}
				}
				if(w == 0 && Omega == 0.0){
					Theta = acos(x4i.x * ir);
					if(x4i.y < 0.0){
						if(e < 1.0 - 1.0e-10){
							//elliptic
							Theta = 2.0 * M_PI - Theta;
						}
						else if(e > 1.0 + 1.0e-10){
							//hyperbolic
							Theta = -Theta;
						}
						else{
							//parabolic
							Theta = -Theta;
						}
					}
				}

				if(e < 1.0 - 1.0e-10){
					//Eccentric Anomaly
					E = acos((e + cos(Theta)) / (1.0 + e * cos(Theta)));
					if(M_PI < Theta && Theta < 2.0 * M_PI) E = 2.0 * M_PI - E;
				}
				else if(e > 1.0 + 1.0e-10){
					//Hyperbolic Anomaly
					//named still E instead of H or F
					E = acosh((e + t) / (1.0 + e * t));
					if(Theta < 0.0) E = - E;
				}
				else{
					//Parabolic Anomaly
					E = tan(Theta * 0.5);
					if(E > M_PI) E = E - 2.0 * M_PI;
					//use a to store q
					a = h * h / mu * 0.5;
				}


//printf("K0 %.10g %.10g %g %g %g %g %g %g\n", x4i.w, a, e, inc, Omega, w, E, Theta);
			}
			//modify Elements

			int line1 = line + nbodies;
			int line2 = line1 + nbodies;
			int line3 = line2 + nbodies;
			double time0, time1, time2, time3;
			double T, M;
			double xx0, xx1, xx2, xx3;


			for(int i = 0; i < nelements; ++i){
				if(setElements_c[i] == 1){
					time0 = setElementsData_d[line * nelements + i];
					time1 = setElementsData_d[line1 * nelements + i];
					time2 = setElementsData_d[line2 * nelements + i];
					time3 = setElementsData_d[line3 * nelements + i];
//printf("i %d id %d time0 %.10g time1 %.10g time2 %.10g time3 %.10g | time %.10g | line0 %d line1 %d line2 %d line3 %d\n", i, id, time0, time1, time2, time3, time, line, line1, line2, line3);
					if(time >= time2 && line3 < nlines - nbodies){
						line += nbodies;
						line1 += nbodies;
						line2 += nbodies;
						line3 += nbodies;
						i -= 1;
					}
				}

			}

			setElementsLine_d[0] = line;
			


			for(int i = 0; i < nelements; ++i){
		
				//keep that inside for loop because of boundary corrections
				double xx = (time - time1) / (time2 - time1);
				if(time1 - time0 == 0 || time < time0) xx = 0.0;

				xx0 = setElementsData_d[(line + id) * nelements + i];
				xx1 = setElementsData_d[(line1 + id) * nelements + i];
				xx2 = setElementsData_d[(line2 + id) * nelements + i];
				xx3 = setElementsData_d[(line3 + id) * nelements + i];

				double f0 = (time2 - time1) / (time2 - time0);
				double f1 = (time2 - time1) / (time3 - time1);

//if(i == 0) printf("id %d xxA %g %g %g %g | %g | %g %g\n", id, xx0, xx1, xx2, xx3, xx, f0, f1);

				if(xx < 0){
					//first point
					xx3 = xx2;
					xx2 = xx1;
					xx1 = xx0;
					xx = (time - time0) / (time1 - time0);
					f0 = (time1 - time0) / (2.0 * (time1 - time0));
					f1 = (time1 - time0) / (time2 - time0);
				}
				if(xx > 1){
					//last point
					xx0 = xx1;
					xx1 = xx2;
					xx2 = xx3;
					xx = (time - time2) / (time3 - time2);
					f0 = (time3 - time2) / (time3 - time1);
					f1 = (time3 - time2) / (2.0 * (time3 - time2));
				}
//if(i == 0) printf("id %d xxB %g %g %g %g | %g | %g %g\n", id, xx0, xx1, xx2, xx3, xx, f0, f1);


//f0 = 0.5;
//f1 = 0.5;				

				//cubic interpolation
				double aa = -f0 * xx0 + 2.0 * xx1 - f1 * xx1 - 2.0 * xx2 + f0 * xx2 + f1 * xx3;
				double bb = 2.0 * f0 * xx0 - 3.0 * xx1 + f1 * xx1 + 3.0 * xx2 - 2.0 * f0 * xx2 - f1 * xx3;
				double cc = -f0 * xx0 + f0 * xx2;
				double dd = xx1;

				double xx22 = xx * xx;

				double f = aa * xx22 * xx + bb * xx22 + cc * xx + dd;

/*
if(setElements_c[i] == 8){
	printf("id %d %d m0 %g m1 %g m2 %g m3 %g %g\n", id, i, xx0, xx1, xx2, xx3, f);
}
if(setElements_c[i] == 9){
	printf("id %d %d r0 %g r1 %g r2 %g r3 %g %g\n", id, i, xx0, xx1, xx2, xx3, f);
}
if(setElements_c[i] == 11){
	printf("id %d %d x0 %g x1 %g x2 %g x3 %g %g\n", id, i, xx0, xx1, xx2, xx3, f);
}
if(setElements_c[i] == 12){
	printf("id %d %d y0 %g y1 %g y2 %g y3 %g %g\n", id, i, xx0, xx1, xx2, xx3, f);
}
if(setElements_c[i] == 13){
	printf("id %d %d z0 %g z1 %g z2 %g z3 %g %g\n", id, i, xx0, xx1, xx2, xx3, f);
}
*/
				if(setElements_c[i] == 3 && EE == 1){
					a = f;
				}
				if(setElements_c[i] == 4 && EE == 1){
					e = f;
				}
				if(setElements_c[i] == 5 && EE == 1){
					inc = f;
				}
				if(setElements_c[i] == 6 && EE == 1){
					Omega = f;
				}
				if(setElements_c[i] == 7 && EE == 1){
					w = f;
				}
				if(setElements_c[i] == 8 && EE == 1){
					x4i.w = f;
				}
				if(setElements_c[i] == 9 && EE == 1){
					v4i.w = f;
				}
				if(setElements_c[i] == 10 && EE == 1){
					T = f * dayUnit;	//t0 epoch time day to day'
					M = T * sqrt(mu / (a * a * a));		//Mean anomaly
					M = fmod(M, 2.0*M_PI);
				}
				//do  x y z after conversion from Kepler elements
				if(setElements_c[i] == 11 && EE == 0){
					x = f;
				}
				if(setElements_c[i] == 12 && EE == 0){
					y = f;
				}
				if(setElements_c[i] == 13 && EE == 0){
					z = f;
				}
				if((setElements_c[i] == 15 || setElements_c[i] == 18) && EE == 0){	//heliocentric or barycentric
					vx = f;
				}
				if((setElements_c[i] == 16 || setElements_c[i] == 19) && EE == 0){
					vy = f;
				}
				if((setElements_c[i] == 17 || setElements_c[i] == 20) && EE == 0){
					vz = f;
				}
			}
			for(int i = 0; i < nelements; ++i){
				if(setElements_c[i] == 10){
					E = M + e * 0.5;
					double Eold = E;
					for(int j = 0; j < 32; ++j){
						E = E - (E - e * sin(E) - M) / (1.0 - e * cos(E));
						if(fabs(E - Eold) < 1.0e-15) break;
						Eold = E;
					}
				}
			}
			mu = def_ksq * (Msun + x4i.w);

			if(doConversion == 1){
//printf("K1 %.10g %.10g %g %g %g %g %g %g\n", x4i.w, a, e, inc, Omega, w, E, Theta);
				//Convert to Cartesian Coordinates

				double cw = cos(w);
				double sw = sin(w);
				double cOmega = cos(Omega);
				double sOmega = sin(Omega);
				double ci = cos(inc);
				double si = sin(inc);

				double Px = cw * cOmega - sw * ci * sOmega;
				double Py = cw * sOmega + sw * ci * cOmega;
				double Pz = sw * si;

				double Qx = -sw * cOmega - cw * ci * sOmega;
				double Qy = -sw * sOmega + cw * ci * cOmega;
				double Qz = cw * si;

				double cE = cos(E);
				double sE = sin(E);

				double t0, t1, t2;

				if(e < 1.0 - 1.0e-10){
					//elliptic
					t1 = a * (cE - e);
					t2 = a * sqrt(1.0 - e * e) * sE;
				}
				else if(e > 1.0 + 1.0e-10){
					//hyperbolic
					//double r = a * (1.0 - e*e)/(1.0 + e *cos(Theta));
					//or
					//double r = a * ( 1.0 - e * cosh(E));
					//t1 = r * cos(Theta); 
					//t2 = r * sin(Theta); 
					t1 = a * (cosh(E) - e);
					t2 = -a * sqrt(e * e - 1.0) * sinh(E);
				}
				else{
					//parabolic
					// a is assumed to be q, p = 2q, p = h^2/mu
					double r = 2 * a /(1.0 + cos(Theta));
					t1 = r * cos(Theta);
					t2 = r * sin(Theta);
				}


				x4i.x =  t1 * Px + t2 * Qx;
				x4i.y =  t1 * Py + t2 * Qy;
				x4i.z =  t1 * Pz + t2 * Qz;

				if(e < 1.0 - 1.0e-10){
					//elliptic
					t0 = 1.0 / (1.0 - e * cE) * sqrt(mu / a);
					t1 = -sE;
					t2 = sqrt(1.0 - e * e) * cE;
				}
				else if(e > 1.0 + 1.0e-10){
					//hyperbolic
					//double r = a * (1.0 - e*e)/(1.0 + e *cos(Theta));
					double r = a * ( 1.0 - e * cosh(E));
					t0 = sqrt(-mu * a) / r;
					t1 = -sinh(E);
					t2 = sqrt(e * e - 1.0) * cosh(E);
				}
				else{
				//parabolic
					t0 = mu / sqrt(2.0 * a * mu);
					t1 = -sin(Theta);
					t2 = 1.0 +  cos(Theta);
				}


				v4i.x = t0 * (t1 * Px + t2 * Qx);
				v4i.y = t0 * (t1 * Py + t2 * Qy);
				v4i.z = t0 * (t1 * Pz + t2 * Qz);
			}

			for(int i = 0; i < nelements; ++i){
				if(setElements_c[i] == 11){
					x4i.x = x;
				}
				if(setElements_c[i] == 12){
					x4i.y = y;
				}
				if(setElements_c[i] == 13){
					x4i.z = z;
				}
				if(setElements_c[i] == 15 || setElements_c[i] == 18){
					v4i.x = vx;
				}
				if(setElements_c[i] == 16 || setElements_c[i] == 19){
					v4i.y = vy;
				}
				if(setElements_c[i] == 17 || setElements_c[i] == 20){
					v4i.z = vz;
				}
			}

			x4_d[id] = x4i;
			v4_d[id] = v4i;
//printf("SE %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g\n", id, x4i.x, x4i.y, x4i.z, x4i.w, v4i.x, v4i.y, v4i.z, v4i.w);
		}	
	}
}


// ***************************************************************
// This kernel calulates the probability of a collisional induced
// rotation reset. 
// Fragmentation events are reportend in the Fragments_d array.
//
// March 2017
// Authors: Simon Grimm, Matthias Meier
// *****************************************************************
__global__ void rotation_kernel(curandState *random_d, double4 *x4_d, double4 *v4_d, double4 *spin_d, int *index_d, int *N_d, int *Nsmall_d, double *dt_d, int st, double *Fragments_d, double time, int *nFragments_d){

	int N = N_d[st];

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy + N;

	int Nsmall = Nsmall_d[st];
	double dt = dt_d[st];

	if(id < Nsmall + N){

		volatile double4 x4 = x4_d[id];
		volatile double4 v4 = v4_d[id];
		double4 spin = spin_d[id];
		curandState random = random_d[id];
	
		if(x4.w >= 0.0){

			//mass of the parent body
			double RR = v4.w * def_AU;	//convert radius in m
			double M = x4.w;
			if(x4.w == 0.0){
				M = Asteroid_rho_c[0] * 4.0 / 3.0 * M_PI * v4.w * v4.w * v4.w * def_AU * def_AU * def_AU; 	//mass in kg;
				M /= def_Solarmass;								//mass im Solar masses
			}

			//compute rotation vector from spin vector
			double Ic = spin.w;
			double iI = 1.0 / (Ic * M * v4.w * v4.w); // inverse Moment of inertia of a solid sphere in 1/ (Solar Masses AU^2)
			double3 omega3;
			omega3.x = spin.x * iI;
			omega3.y = spin.y * iI;
			omega3.z = spin.z * iI;

			double omega = sqrt(omega3.x * omega3.x + omega3.y * omega3.y + omega3.z * omega3.z);	//angular velocity in 1 / day * 0.017
			omega *= 2.0 * M_PI * dayUnit / (24.0 * 3600.0);					//in 1 / s

			//compute probability of rotation reset
			double t1 = 2.0 * sqrt(2.0) * omega / (5.0 * Asteroid_V_c[0]);
			double p = 1.0e-18 / cbrt(RR * RR * RR * RR) * pow(t1, -5.0/6.0);	//probability per second
			p = p * 3600.0 * 24.0 * dt / dayUnit;					//probability per time step
#if USE_RANDOM == 1
			double rd = curand_uniform(&random);
			int accept = -2;
			if(rd < p && omega > 0.0) {
				accept = atomicMax(&nFragments_d[0], 0);
printf("rotation reset %d %d %g %g %g\n", id, index_d[id], time/365.25, rd, p);
printf("rA %g %d %g %g %g %g\n", time, id, RR, omega, p, rd);
			}
			if(accept == -1){
				//reset the rotation rate and spin vector
				rd = curand_uniform(&random);
				double omega = 1.0/((rd * 35 + 1.0) * RR); //rotations per s
printf("rB %g %d %g %g %g %g\n", time, id, RR, omega, p, rd);
				omega = omega / dayUnit * 24.0 * 3600.0;  //rotation in 1 / day'

				double S = Ic * M * v4.w * v4.w * omega;
				double u = curand_uniform(&random);
				double theta = curand_uniform(&random) * 2.0 * M_PI;
				//sign
				double s = curand_uniform(&random);;

				double t2 = S * sqrt(1.0 - u * u);
				spin.x = t2 * cos(theta);
				spin.y = t2 * sin(theta);
				spin.z = S * u;

				if( s > 0.5){
					spin.z *= -1.0;
				}
				spin_d[id] = spin;

				Fragments_d[0] = time/365.25;
				Fragments_d[1] = (double)(index_d[id]);
				Fragments_d[2] = x4_d[id].w;
				Fragments_d[3] = v4_d[id].w;
				Fragments_d[4] = x4_d[id].x;
				Fragments_d[5] = x4_d[id].y;
				Fragments_d[6] = x4_d[id].z;
				Fragments_d[7] = v4_d[id].x;
				Fragments_d[8] = v4_d[id].y;
				Fragments_d[9] = v4_d[id].z;
				Fragments_d[10] = spin_d[id].x;
				Fragments_d[11] = spin_d[id].y;
				Fragments_d[12] = spin_d[id].z;

				atomicMax(&nFragments_d[0], 1);
			}
#endif
		}
		random_d[id] = random;
	}
}
// ***************************************************************
// This kernel calulates the probability of Asteroid Collisions
// generates fragment kernels. 
// Fragmentation events are reportend in the Fragments_d array.
//
// March 2017
// Authors: Simon Grimm, Matthias Meier
// *****************************************************************
__global__ void fragment_kernel(curandState *random_d, double4 *x4_d, double4 *v4_d, double4 *spin_d, int *index_d, int *N_d, int *Nsmall_d, double *dt_d, int NconstT, int MaxIndex, int st, double *Fragments_d, double time, int *nFragments_d){
#if USE_RANDOM == 1
	int N = N_d[st];

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy + N;

	int Nsmall = Nsmall_d[st];
	double dt = dt_d[st];

	if(id < Nsmall + N){ 
		
		volatile double4 x4 = x4_d[id];
		volatile double4 v4 = v4_d[id];
		curandState random = random_d[id];
		curand_uniform(&random);
		random_d[id] = random;
		
		if(x4.w >= 0.0){

			//mass of the parent body
			double RR = v4.w * def_AU;	//convert radius in m
			double M = x4.w * def_Solarmass; //mass in kg
			if(x4.w == 0.0){
				M = Asteroid_rho_c[0] * 4.0 / 3.0 * M_PI * v4.w * v4.w * v4.w * def_AU * def_AU * def_AU; 	//mass in kg;
			}

			double p = 1.0 / (2.0e7 * sqrt(RR));	//probability per year per body
			p = p / 365.25 * dt / dayUnit;  	//probability per time step per body
			double rd = curand_uniform(&random);
			volatile int accept = -2;
			if(rd < p) {
				accept = atomicMax(&nFragments_d[0], 0);
printf("fragment %d %d %d %g %g %g %g %g %d\n", id, index_d[id], accept, time/365.25, rd, p, M, RR, MaxIndex);
			}
			if(accept == -1){
				double x0 = Asteroid_rmin_c[0];	//m
				double x1 = RR;		//m

				volatile int ii;
				double vscaleT = 0.0;
				for(ii = 0; ii < 10000; ++ii){

					//mass
					double n = -1.5;
					double u = curand_uniform(&random);
					double r = pow((pow(x1,n+1.0) - pow(x0,n+1.0)) * u + pow(x0, n+1.0), 1.0/(n+1.0));
					double m = Asteroid_rho_c[0] * 4.0 / 3.0 * M_PI * r * r * r; //mass in kg;

					M -= m;
					if(M <= 0.0){
						m += M;
						M = 0.0;
						r = cbrt(m * 3.0 / (Asteroid_rho_c[0] * 4.0 * M_PI));
					}

					double vscale = curand_uniform(&random) * 2.0 * 0.2 + (1.0 - 0.2) * pow(m, -1.0/6.0);
					vscaleT += vscale;
					//velocity
					double v = 31.0 * vscale; //m/s

					//direction 
					u = curand_uniform(&random);
					double theta = curand_uniform(&random) * 2.0 * M_PI;

					//sign
					double s = curand_uniform(&random);

					double x = 3 * RR * sqrt(1.0 - u * u) * cos(theta);
					double y = 3 * RR * sqrt(1.0 - u * u) * sin(theta);
					double z = 3 * RR * u;

					volatile double vx = v * sqrt(1.0 - u * u) * cos(theta);
					volatile double vy = v * sqrt(1.0 - u * u) * sin(theta);
					volatile double vz = v * u;
printf("fA %d %g %g %g %g %g %g %g %g %g\n", ii, M, RR, m, r, v, vx, vy, vz, v4.x);

					if( s > 0.5){
						z *= -1.0;
						vz *= -1.0;
					}

					//rotation rate and spin vector
					rd = curand_uniform(&random);
					double omega = 1.0/((rd * 35 + 1.0) * r);	//rotations per s
					omega = omega / dayUnit * 24.0 * 3600.0;  //rotation in 1 / day'

					x /= def_AU;
					y /= def_AU;
					z /= def_AU;
					r /= def_AU;

					vx = vx / def_AU * 3600.0 * 24.0 / dayUnit;
					vy = vy / def_AU * 3600.0 * 24.0 / dayUnit;
					vz = vz / def_AU * 3600.0 * 24.0 / dayUnit;
printf("fB %d %g %g %g %g %g %g %g %g %g\n", ii, M, RR, m, r, v, vx, vy, vz, v4.x);

					m /= def_Solarmass;

					x4_d[ii + N + Nsmall].x = x4.x + x;
					x4_d[ii + N + Nsmall].y = x4.y + y;
					x4_d[ii + N + Nsmall].z = x4.z + z;
					x4_d[ii + N + Nsmall].w = m;

					v4_d[ii + N + Nsmall].x = vx;
					v4_d[ii + N + Nsmall].y = vy;
					v4_d[ii + N + Nsmall].z = vz;
					v4_d[ii + N + Nsmall].w = r;

					double4 spin;
					spin.w = spin_d[id].w;

					double S = spin.w * m * r * r * omega;
					u = curand_uniform(&random);
					theta = curand_uniform(&random) * 2.0 * M_PI;
					//sign
					s = curand_uniform(&random);;

					double t2 = S * sqrt(1.0 - u * u);
					spin.x = t2 * cos(theta);
					spin.y = t2 * sin(theta);
					spin.z = S * u;

					if( s > 0.5){
						spin.z *= -1.0;
					}
					spin_d[ii + N + Nsmall] = spin;
					index_d[ii + N + Nsmall] = MaxIndex + ii + 1;


//printf("%.20g %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g 0\n", time/365.25, index_d[ii + N + Nsmall], x4_d[ii + N + Nsmall].w, v4_d[ii + N + Nsmall].w, x4_d[ii + N + Nsmall].x, x4_d[ii + N + Nsmall].y, x4_d[ii + N + Nsmall].z, v4_d[ii + N + Nsmall].x, v4_d[ii + N + Nsmall].y, v4_d[ii + N + Nsmall].z, spin_d[ii + N + Nsmall].x, spin_d[ii + N + Nsmall].y, spin_d[ii + N + Nsmall].z);

					if(M == 0.0) break;
					if(N + Nsmall + ii >= NconstT){
						Nsmall_d[st] += ii;
						atomicMax(&nFragments_d[0], ii);
						break;
					}
				}
				//rescale the velocity
				for(int i = 0; i < ii; ++i){
					
					double vx = v4_d[i + N + Nsmall].x / vscaleT;
					double vy = v4_d[i + N + Nsmall].y / vscaleT;
					double vz = v4_d[i + N + Nsmall].z / vscaleT;

					v4_d[i + N + Nsmall].x = v4.x + vx;
					v4_d[i + N + Nsmall].y = v4.y + vy;
					v4_d[i + N + Nsmall].z = v4.z + vz;


					Fragments_d[(i + 1) * 25 + 0] = time/365.25;
					Fragments_d[(i + 1) * 25 + 1] = (double)(index_d[i + N + Nsmall]);
					Fragments_d[(i + 1) * 25 + 2] = x4_d[i + N + Nsmall].w;
					Fragments_d[(i + 1) * 25 + 3] = v4_d[i + N + Nsmall].w;
					Fragments_d[(i + 1) * 25 + 4] = x4_d[i + N + Nsmall].x;
					Fragments_d[(i + 1) * 25 + 5] = x4_d[i + N + Nsmall].y;
					Fragments_d[(i + 1) * 25 + 6] = x4_d[i + N + Nsmall].z;
					Fragments_d[(i + 1) * 25 + 7] = v4_d[i + N + Nsmall].x;
					Fragments_d[(i + 1) * 25 + 8] = v4_d[i + N + Nsmall].y;
					Fragments_d[(i + 1) * 25 + 9] = v4_d[i + N + Nsmall].z;
					Fragments_d[(i + 1) * 25 + 10] = spin_d[i + N + Nsmall].x;
					Fragments_d[(i + 1) * 25 + 11] = spin_d[i + N + Nsmall].y;
					Fragments_d[(i + 1) * 25 + 12] = spin_d[i + N + Nsmall].z;

//printf("%.20g %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g 0\n", time/365.25, index_d[i + N + Nsmall], x4_d[i + N + Nsmall].w, v4_d[i + N + Nsmall].w, x4_d[i + N + Nsmall].x, x4_d[i + N + Nsmall].y, x4_d[i + N + Nsmall].z, v4_d[i + N + Nsmall].x, v4_d[i + N + Nsmall].y, v4_d[i + N + Nsmall].z, spin_d[i + N + Nsmall].x, spin_d[i + N + Nsmall].y, spin_d[i + N + Nsmall].z);

					//remove too small particles
					double r = v4_d[i + N + Nsmall].w;
					if(r * def_AU < Asteroid_rdel_c[0]){
printf("Remove Fragment %d %g\n", i + N + Nsmall, r * def_AU);
						x4_d[i + N + Nsmall].x = 1.0;
						x4_d[i + N + Nsmall].y = 0.0;
						x4_d[i + N + Nsmall].z = 0.0;
						x4_d[i + N + Nsmall].w = -1.0e-12;

						v4_d[i + N + Nsmall].x = 0.0;
						v4_d[i + N + Nsmall].y = 0.0;
						v4_d[i + N + Nsmall].z = 0.0;
						v4_d[i + N + Nsmall].w = 0.0;

					}
				}
			
			
//printf("%d %g %g %g %d %d %d\n", id, p, RR, rd, ii, N + Nsmall, NconstT);
				Nsmall_d[st] += ii;
				atomicMax(&nFragments_d[0], ii);

				Fragments_d[0] = time/365.25;
				Fragments_d[1] = (double)(index_d[id]);
				Fragments_d[2] = x4_d[id].w;
				Fragments_d[3] = v4_d[id].w;
				Fragments_d[4] = x4_d[id].x;
				Fragments_d[5] = x4_d[id].y;
				Fragments_d[6] = x4_d[id].z;
				Fragments_d[7] = v4_d[id].x;
				Fragments_d[8] = v4_d[id].y;
				Fragments_d[9] = v4_d[id].z;
				Fragments_d[10] = spin_d[id].x;
				Fragments_d[11] = spin_d[id].y;
				Fragments_d[12] = spin_d[id].z;

				x4_d[id].x = 0.0;
				x4_d[id].y = 1.0;
				x4_d[id].z = 0.0;
				x4_d[id].w = -1.0e-12;

				v4_d[id].x = 0.0;
				v4_d[id].y = 0.0;
				v4_d[id].z = 0.0;
				v4_d[id].w = 0.0;

				spin_d[id].x = 0.0;
				spin_d[id].y = 0.0;
				spin_d[id].z = 0.0;

				index_d[id] = -1;
			}
			random_d[id] = random;
		}
	}
#endif
}

__host__ void fragmentCall(curandState *random_d, double4 *x4_d, double4 *v4_d, double4 *spin_d, int *index_d, int *N_h, int *N_d, int *Nsmall_h, int *Nsmall_d, double *dt_d, int Nst, int NconstT, double *Fragments_d, double time, int *nFragments_m, int *nFragments_d, int &MaxIndex){
	if(Nsmall_h[0] > 0.0){
		int st = 0;
		nFragments_m[0] = -1;
		fragment_kernel <<< (Nsmall_h[0] + 255) / 256, 256 >>> (random_d, x4_d, v4_d, spin_d, index_d, N_d, Nsmall_d, dt_d, NconstT, MaxIndex, st, Fragments_d, time, nFragments_d);
		cudaDeviceSynchronize();
		if(nFragments_m[0] > 0){
			Nsmall_h[st] += nFragments_m[0];
			MaxIndex += nFragments_m[0];
		}
	}
}
__host__ void rotationCall(curandState *random_d, double4 *x4_d, double4 *v4_d, double4 *spin_d, int *index_d, int *N_h, int *N_d, int *Nsmall_h, int *Nsmall_d, double *dt_d, int Nst, double *Fragments_d, double time, int *nFragments_m, int *nFragments_d){
	if(Nsmall_h[0] > 0.0){
		int st = 0;
		nFragments_m[0] = -1;
		rotation_kernel <<< (Nsmall_h[0] + 255) / 256, 256 >>> (random_d, x4_d, v4_d, spin_d, index_d, N_d, Nsmall_d, dt_d, st, Fragments_d, time, nFragments_d);
		cudaDeviceSynchronize();
	}
}

// ***************************************************************
// This kernel computes the seasonal and diurnal Yarkovsky effect.
// it computes the yarkovsky acceleration and performs a velocity kick

// See VOKROUHLICKY, MILANI, AND CHESLEY 2000
// See Appendix B from VOKROUHLICKYY & FARINELLA 1999

// March 2017
// Authors: Simon Grimm, Matthias Meier
// *****************************************************************
__global__ void CallYarkovsky2(double4 *x4_d, double4 *v4_d, double4 *spin_d, int *index_d, double2 *Msun_d, double *dt_d, double Kt, int N, int Nst, int Nstart){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy + Nstart;

	int st = 0;

//if(id == 0) printf("Asteroid %g %g %g %g %g %g\n", Asteroid_eps_c[0], Asteroid_rho_c[0], Asteroid_C_c[0], Asteroid_A_c[0], Asteroid_K_c[0], Asteroid_V_c[0]);
	if(id < N + Nstart){

		if(Nst > 1) st = index_d[id] / def_MaxIndex;	//st is the sub simulation index

		double4 x4i = x4_d[id];
		double4 v4i = v4_d[id];
		double4 spin = spin_d[id];

		if(x4i.w >= 0.0){

			//material constants

			double Gamma = sqrt(Asteroid_K_c[0] * Asteroid_rho_c[0] * Asteroid_C_c[0]);	//surface thermal intertia 
			double RR = v4i.w * def_AU;		//covert radius in m 

			//int index = index_d[id];
			double Msun = Msun_d[st].x;
			double dt = dt_d[st] * Kt;
			double m = x4i.w;
			if(m == 0.0){
				m = Asteroid_rho_c[0] * 4.0 / 3.0 * M_PI * RR * RR * RR; 	//mass in kg;
				m /= def_Solarmass;						//mass im Solar masses
			}
			double mu = def_ksq * (Msun + m);
	
			double rsq = x4i.x * x4i.x + x4i.y * x4i.y + x4i.z * x4i.z;
			double vsq = v4i.x * v4i.x + v4i.y * v4i.y + v4i.z * v4i.z;
			double u =  x4i.x * v4i.x + x4i.y * v4i.y + x4i.z * v4i.z;
			double ir = 1.0 / sqrt(rsq);
			double ia = 2.0 * ir - vsq / mu;

			double a = 1.0 / ia;

			double3 h3;
			h3.x = ( x4i.y * v4i.z) - (x4i.z * v4i.y);
			h3.y = (-x4i.x * v4i.z) + (x4i.z * v4i.x);
			h3.z = ( x4i.x * v4i.y) - (x4i.y * v4i.x);

			double h = sqrt(h3.x * h3.x + h3.y * h3.y + h3.z * h3.z);

	
			//longitude of ascending node
			double nn = sqrt(h3.x * h3.x + h3.y * h3.y);

			//argument of periapsis
			double3 e3;
			e3.x = ( v4i.y * h3.z - v4i.z * h3.y) / mu - x4i.x * ir;
			e3.y = (-v4i.x * h3.z + v4i.z * h3.x) / mu - x4i.y * ir;
			e3.z = ( v4i.x * h3.y - v4i.y * h3.x) / mu - x4i.z * ir;
		
			double e = sqrt(e3.x * e3.x + e3.y * e3.y + e3.z * e3.z); 

			double n;
			if(e < 1.0 - 1.0e-10){
				//Elliptic
				n = sqrt(mu / (a * a * a)); //mean motion in 1 / day * 0.017 
			}
			else if(e > 1.0 + 1.0e-10){
				//hyperbolic
				n = sqrt(mu / (-a * a * a)); //mean motion in 1 / day * 0.017 
			}
			else{
				//parabolic
				n = sqrt(mu); //mean motion in 1 / day * 0.017 
			}
			n *= dayUnit / (24.0 * 3600.0);  //mean motion  in 1 / s;

			//compute rotation vector from spin vector
			double Ic = spin.w;
			double iI = 1.0 / (Ic * m * v4i.w * v4i.w); // inverse Moment of inertia of a solid sphere in 1/ (Solar Masses AU^2)
			double3 omega3;
			omega3.x = spin.x * iI;
			omega3.y = spin.y * iI;
			omega3.z = spin.z * iI;

			double omega = sqrt(omega3.x * omega3.x + omega3.y * omega3.y + omega3.z * omega3.z); 	//angular velocity in 1 / day * 0.017
	
		//Normalize spin vector
			omega3.x /= omega;
			omega3.y /= omega;
			omega3.z /= omega;

			double sp, sq;
			//True Anomaly
			double Theta;
			double t;
			if(e > 1.0e-10){
				t = (e3.x * x4i.x + e3.y * x4i.y + e3.z * x4i.z) / e * ir;
				if(t < -1.0) t = -1.0;
				if(t > 1.0) t = 1.0;
				Theta = acos(t);
				if(u < 0.0){
					if(e < 1.0 - 1.0e-10){
						//elliptic
						Theta = 2.0 * M_PI - Theta;
					}
					else if(e > 1.0 + 1.0e-10){
						//hyperbolic
						Theta = -Theta;
					}
					else{
						//parabolic
						Theta = - Theta;
					}
				}
	
				sp = (omega3.x * e3.x + omega3.y * e3.y + omega3.z * e3.z) / e;
				double3 q3;
				q3.x = ( h3.y * e3.z) - (h3.z * e3.y);
				q3.y = (-h3.x * e3.z) + (h3.z * e3.x);
				q3.z = ( h3.x * e3.y) - (h3.y * e3.x);
				sq = (omega3.x * q3.x + omega3.y * q3.y + omega3.z * q3.z) / (e * h);
			}
			else{
			//circular inclined orbit
				if(h3.z < h * (1.0 - 1.0e-11)){
					t = (-h3.y * x4i.x + h3.x * x4i.y) / nn * ir;
					if(t < -1.0) t = -1.0;
					if(t > 1.0) t = 1.0;
					Theta = acos(t);
					if(x4i.z < 0.0) Theta = 2.0 * M_PI - Theta;
		
					sp = (omega3.x * -h3.y + omega3.y * h3.x) / nn;
					double3 q3;
					q3.x = 0.0;
					q3.y = 0.0;
					q3.z = ( h3.x * h3.x) - (h3.y * -h3.y);
					sq = (omega3.x * q3.x + omega3.y * q3.y + omega3.z * q3.z) / (e * nn);
				}
			//circular equatorial orbit
				else{
					t = x4i.x * ir;
					Theta = acos(t);
					if(x4i.y < 0.0) Theta = 2.0 * M_PI - Theta;
		
					sp = (omega3.x);
					double3 q3;
					q3.x = 0.0;
					q3.y = h3.z;
					q3.z = h3.y;
					sq = (omega3.x * q3.x + omega3.y * q3.y + omega3.z * q3.z) / h;
				}
			}

			if(omega == 0){
				sp = 0.0;
				sq = 0.0;
			}

			double E, M;
			if(e < 1.0 - 1.0e-10){
				//Eccentric Anomaly
				E = acos((e + t) / (1.0 + e * t));
				if(M_PI < Theta && Theta < 2.0 * M_PI) E = 2.0 * M_PI - E;

				//Mean Anomaly
				double M = E - e * sin(E);
			}
			else if(e > 1.0 + 1.0e-10){
				//Hyperbolic Anomaly
				//named still E instead of H or F
				E = acosh((e + t) / (1.0 + e * t));
				if(Theta < 0.0) E = - E;

				M = e * sinh(E) - E;
			}
			else{
				//Parabolic Anomaly
				E = tan(Theta * 0.5);
				if(E > M_PI) E = E - 2.0 * M_PI;

				M = E + E * E * E / 3.0;

				//use a to store q
				a = h * h / mu * 0.5;
			}


//printf("a %d %g %g %g %g %g %g %g %g %g %g %g\n", id, a, e, m, RR, omega, v4i.x, v4i.y, v4i.z,  Theta, E, M);

			double3 rs3;
			rs3.x = (( x4i.y * omega3.z) - (x4i.z * omega3.y)) * ir;
			rs3.y = ((-x4i.x * omega3.z) + (x4i.z * omega3.x)) * ir;
			rs3.z = (( x4i.x * omega3.y) - (x4i.y * omega3.x)) * ir;

			double3 srs3;
			srs3.x = (( omega3.y * rs3.z) - (omega3.z * rs3.y));
			srs3.y = ((-omega3.x * rs3.z) + (omega3.z * rs3.x));
			srs3.z = (( omega3.x * rs3.y) - (omega3.y * rs3.x));


			omega *= 2.0 * M_PI * dayUnit / (24.0 * 3600.0); 						//in 1 / s

			double d = a * (1.0 + e*e * 0.5);//time averaged heliocentric distance in AU
			double F = SolarConstant_c[0] / (d * d);		//scaled heliocentric distance, F = SEarth * (aEarth/a)^2

			double Ts4 = (1.0 - Asteroid_A_c[0]) * F / (Asteroid_eps_c[0] * def_sigma);
			double Ts = sqrt(sqrt(Ts4));

			double t1 = Gamma / (Asteroid_eps_c[0] * def_sigma * Ts * Ts * Ts);
			double t2 = (1.0 - Asteroid_A_c[0]) * 3.0 * F / (9.0 * Asteroid_rho_c[0] * RR * def_c);

			double s2 = sqrt(2.0);

			double3 a3;
			a3.x = 0.0;
			a3.y = 0.0;
			a3.z = 0.0;

			//Diurnal 
			// See VOKROUHLICKY, MILANI, AND CHESLEY 2000
			{
			double ilD = sqrt(Asteroid_rho_c[0] * Asteroid_C_c[0] * omega / Asteroid_K_c[0]);
			double ThetaD = t1 * sqrt(omega);
			double X = s2 * RR * ilD;
			double lamda = ThetaD / X;
			double L = lamda / (1.0 + lamda);
			double cX = cos(X);
			double sX = sin(X);

			double X2cX = (X - 2.0) * cX;
			double X2sX = (X - 2.0) * sX;
			double eX = exp(-X);

			double Ax = -eX * (X + 2.0) - (X2cX - X * sX);
			double Bx = -eX * X - (X * cX + X2sX);
			double Cx = Ax + L * (eX * 3.0 * (X + 2.0) + (3.0 * X2cX + X * (X - 3.0) * sX));
			double Dx = Bx + L * (eX * X * (X + 3.0) - (X * (X - 3.0) * cX - 3.0 * X2sX));

			double iC2D2 = 1.0 / (Cx * Cx + Dx * Dx);

			double Gcd = (Ax * Cx + Bx * Dx) * iC2D2;
			double Gsd = (Bx * Cx - Ax * Dx) * iC2D2;

			double WD = t2 / (1.0 + lamda);

			if(omega != 0.0 && e < 1.0){
				a3.x += WD * (Gsd * rs3.x + Gcd * srs3.x);
				a3.y += WD * (Gsd * rs3.y + Gcd * srs3.y);
				a3.z += WD * (Gsd * rs3.z + Gcd * srs3.z);
			}
//printf("D %d %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g\n", id, a3.x, a3.y, a3.z, srs3.x, srs3.y, srs3.z, Gcd, Gsd, WD, lamda);
			}
			
			//seasonal
			//See Appendix B from VOKROUHLICKYY & FARINELLA 1999
			{
			double ilS = sqrt(Asteroid_rho_c[0] * Asteroid_C_c[0] * n / Asteroid_K_c[0]);
			double ThetaS = t1 * sqrt(n);
			double eta = sqrt(1.0 - e * e);
			if(e >= 1.0) eta = 1.0;
		int k = 1;
	
			double e2 = e * e;
			double e3 = e * e2;
			double e4 = e2 * e2;
			//double e5 = e2 * e3;
			double e6 = e3 * e3;

			double alpha = 1.0 - 0.375 * e2 + 5.0 / 6.0 * 0.25 * e4 - 7.0 / 72.0 / 128.0;
			double beta = 1.0 - e2 / 8.0 + e4 / 192.0 - e6 / 9216.0;
	
			double X = s2 * RR * ilS;
			double lamda = ThetaS / X * sqrt(sqrt(eta * eta * eta));
			double L = lamda / (1.0 + lamda);

			double cX = cos(X);
			double sX = sin(X);

			double X2cX = (X - 2.0) * cX;
			double X2sX = (X - 2.0) * sX;

			double eX = exp(-X);
			// A B C D are multiplied by e^-X, which cancelles out later
			double Ax = -eX * (X + 2.0) - (X2cX - X * sX);
			double Bx = -eX * X - (X * cX + X2sX);
			double Cx = Ax + L * (eX * 3.0 * (X + 2.0) + (3.0 * X2cX + X * (X - 3.0) * sX));
			double Dx = Bx + L * (eX * X * (X + 3.0) - (X * (X - 3.0) * cX - 3.0 * X2sX));

			double iC2D2 = 1.0 / (Cx * Cx + Dx * Dx);

			double Gcd = (Ax * Cx + Bx * Dx) * iC2D2;
			double Gsd = (Bx * Cx - Ax * Dx) * iC2D2;
			double cM = cos(k * M); 
			double sM = sin(k * M); 

			double WS = (sp * alpha * (cM * Gcd - sM * Gsd) + sq * beta * (sM * Gcd + cM * Gsd)) / (1.0 + lamda);
			double aS = t2 * WS;
			if(omega != 0.0 && e < 1.0){
				a3.x += aS * omega3.x;
				a3.y += aS * omega3.y;
				a3.z += aS * omega3.z;
			}
//printf("S %d %g %g %g %.10g %.10g %.10g %.10g %.10g %g %g %g %g %g\n", id, a3.x, a3.y, a3.z, sp, sq, sp * sp + sq * sq, RR, Gcd, Gsd, n, lamda, sM, cM);
			}

			a3.x *= 24.0 * 3600.0 * 24.0 * 3600.0 / (def_AU * dayUnit * dayUnit); //in AU /day^2 * 0.017^2
			a3.y *= 24.0 * 3600.0 * 24.0 * 3600.0 / (def_AU * dayUnit * dayUnit);
			a3.z *= 24.0 * 3600.0 * 24.0 * 3600.0 / (def_AU * dayUnit * dayUnit);

		//printf("%d %g %g %g %g %g %g %g %g\n", id, m, RR, a, omega, n, a3.x, a3.y, a3.z);

			v4i.x += a3.x * dt;
			v4i.y += a3.y * dt;
			v4i.z += a3.z * dt;

			v4_d[id] = v4i;

//printf("Y %g %g %g %g %g %g %g %g\n", x4i.x, x4i.y, x4i.z, x4i.w, v4i.x, v4i.y, v4i.z, v4i.w);
		}	

	}
}



// Yarkovski
/*
__device__ void alpha(double e){
	
	double e2 = e * e;
	double e3 = e * e2;
	double e4 = e2 * e2;
	double e5 = e2 * e3;
	double e6 = e3 * e3;

	double alpha1 = 1.0 - 0.375 * e2 + 5.0 / 6.0 * 0.25 * e4 - 7.0 / 72.0 / 128.0;
	double alpha2 = 4.0 * (0.5 * e - 3.0 *  e3 + 0.0625 * e5);
	double alpha3 = 9.0 * (0.375 * e2 - 11.25 / 32.0 * e4 + 567 / 5120.0 * e6);
	double alpha4 = 16.0 * (1.0 / 3.0 * e3 - 0.4 * e5);
	double alpha5 = 25.0 * (125.0 / 384.0 * e4 - 4375.0 / 9216.0 * e6);
	double alpha6 = 36.0 * (108.0 / 320.0 * e5);
	double alpha7 = 49.0 * (16807.0 / 46080.0 * e6);

	double beta1 = 1.0 - e2 / 8.0 + e4 / 192.0 - e6 / 9216.0;
	double beta2 = 2.0 * e * (1.0 - e2 / 3.0 + e4 / 24.0);
	double beta3 = 27.0 / 8.0 * e2 * (1.0 - 9.0 / 16.0 * e2 + 81.0 / 640.0 * e4);
	double beta4 = 16.0 / 3.0 * e3 * (1.0 - 0.8 * e2);
	double beta5 = 25.0 * 125.0 / 384.0 * e4 * (1.0 - 25.0 / 24.0 * e2);
	double beta6 = 972.0 / 80.0 * e5;
	double beta7 = 823543.0 / 46080.0 * e6;  

}

*/

// ***************************************************************
// This kernel computes the Poynting-Robertson drag.
// it computes the PR drag drit rates da/dt and de/de and modifies the Keplerian elements

// BURNS, LAMY, AND SOTER, 1979 (Radiation Forces on Small Particles in the Solar System)

// January 2019
// Authors: Simon Grimm, Matthias Meier
// *****************************************************************
__global__ void PoyntingRobertsonDrag(double4 *x4_d, double4 *v4_d, int *index_d, double2 *Msun_d, double *dt_d, double Kt, int N, int Nst, int Nstart){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy + Nstart;

	//Compute the Kepler Elements

	if(id < N + Nstart){
		if(x4_d[id].w >= 0.0){

			int st = 0;
			if(Nst > 1) st = index_d[id] / def_MaxIndex;	//st is the sub simulation index

			double4 x4i = x4_d[id];
			double4 v4i = v4_d[id];

			double Msun = Msun_d[st].x;
			double dt = dt_d[st] * Kt;
			double m = x4i.w;
			double RR = v4i.w * def_AU;					//covert radius in m	

			if(m == 0.0){
				m = Asteroid_rho_c[0] * 4.0 / 3.0 * M_PI * RR * RR * RR; 	//mass in kg;
				m /= def_Solarmass;					//mass im Solar masses
			}
			double mu = def_ksq * (Msun + m);

			//double eta = 2.53e8 / (Asteroid_rho_c[0] * RR);			//m^2 / s
			double eta = SolarConstant_c[0] * def_AU * def_AU * RR * RR * M_PI / (m * def_Solarmass * def_c * def_c);			//m^2 / s
			eta = eta /(def_AU * def_AU * dayUnit) * 24.0 * 3600.0;		//AU^2 /day * 0.017

			double a, e, inc, Omega, w, Theta, E;
		

			double rsq = x4i.x * x4i.x + x4i.y * x4i.y + x4i.z * x4i.z;
			double vsq = v4i.x * v4i.x + v4i.y * v4i.y + v4i.z * v4i.z;
			double u =  x4i.x * v4i.x + x4i.y * v4i.y + x4i.z * v4i.z;
			double ir = 1.0 / sqrt(rsq);
			double ia = 2.0 * ir - vsq / mu;

			a = 1.0 / ia;

			//inclination
			double3 h3;
			h3.x = ( x4i.y * v4i.z) - (x4i.z * v4i.y);
			h3.y = (-x4i.x * v4i.z) + (x4i.z * v4i.x);
			h3.z = ( x4i.x * v4i.y) - (x4i.y * v4i.x);

			double h = sqrt(h3.x * h3.x + h3.y * h3.y + h3.z * h3.z);

			double t = h3.z / h;
			if(t < -1.0) t = -1.0;
			if(t > 1.0) t = 1.0;
		
			inc = acos(t);

			//longitude of ascending node
			double n = sqrt(h3.x * h3.x + h3.y * h3.y);
			Omega = acos(-h3.y / n);
			if(h3.x < 0.0){
				Omega = 2.0 * M_PI - Omega;
			}

			if(inc < 1.0e-10 || n == 0) Omega = 0.0;

			//argument of periapsis
			double3 e3;
			e3.x = ( v4i.y * h3.z - v4i.z * h3.y) / mu - x4i.x * ir;
			e3.y = (-v4i.x * h3.z + v4i.z * h3.x) / mu - x4i.y * ir;
			e3.z = ( v4i.x * h3.y - v4i.y * h3.x) / mu - x4i.z * ir;
		
			e = sqrt(e3.x * e3.x + e3.y * e3.y + e3.z * e3.z); 
			if(e < 1.0){

				t = (-h3.y * e3.x + h3.x * e3.y) / (n * e);
				if(t < -1.0) t = -1.0;
				if(t > 1.0) t = 1.0;
				w = acos(t);
				if(e3.z < 0.0) w = 2.0 * M_PI - w;
				if(n == 0) w = 0.0;

				//True Anomaly
				t = (e3.x * x4i.x + e3.y * x4i.y + e3.z * x4i.z) / e * ir;
				if(t < -1.0) t = -1.0;
				if(t > 1.0) t = 1.0;
				Theta = acos(t);
				if(u < 0.0) Theta = 2.0 * M_PI - Theta;

				//Non circular, equatorial orbit
				if(e > 1.0e-10 && inc < 1.0e-10){
					Omega = 0.0;
					w = acos(e3.x / e);
					if(e3.y < 0.0) w = 2.0 * M_PI - w;
				}
				
				//circular, inclinded orbit
				if(e < 1.0e-10 && inc > 1.0e-11){
					w = 0.0;
				}
				
				//circular, equatorial orbit
				if(e < 1.0e-10 && inc < 1.0e-11){
					w = 0.0;
					Omega = 0.0;
				}

				if(w == 0 && Omega != 0.0){
					t = (-h3.y * x4i.x + h3.x * x4i.y) / n * ir;
					if(t < -1.0) t = -1.0;
					if(t > 1.0) t = 1.0;
					Theta = acos(t);
					if(x4i.z < 0.0) Theta = 2.0 * M_PI - Theta;
				}
				if(w == 0 && Omega == 0.0){
					Theta = acos(x4i.x * ir);
					if(x4i.y < 0.0) Theta = 2.0 * M_PI - Theta;

				}

				//Eccentric Anomaly
				E = acos((e + cos(Theta)) / (1.0 + e * cos(Theta)));
				if(M_PI < Theta && Theta < 2.0 * M_PI) E = 2.0 * M_PI - E;

				if(e >= 1){
					E = acosh((e + t) / (1.0 + e * t));
					if(M_PI < Theta && Theta < 2.0 * M_PI) E = 2.0 * M_PI - E;
				}


//if(id < 10) printf("K1 %d %g %g %g %g %g %g %g %g %g\n", id, m, RR, a, e, inc, Omega, w, E, Theta);

				//modify elements
				//BURNS, LAMY, AND SOTER, 1979 equation 47 and 48
				double tt1 = 1.0 - e * e;
				double tt2 = sqrt(tt1);
				double dadt = -(eta * ia) * Qpr_c[0] * (2.0 + 3.0 * e * e) / (tt1 * tt2);
				double dedt = -2.5 * (eta * ia * ia) * Qpr_c[0] * e / tt2;

				a += dadt * dt;
				e += dedt * dt;
//if(id < 10) printf("K2 %d %g %g %g %g %g %g %g %g %g | %g %g %g %g\n", id, m, RR, a, e, inc, Omega, w, E, Theta, Qpr_c[0], eta, dadt, dedt);

				//Convert to Cartesian Coordinates

				double cw = cos(w);
				double sw = sin(w);
				double cOmega = cos(Omega);
				double sOmega = sin(Omega);
				double ci = cos(inc);
				double si = sin(inc);

				double3 P3;
				P3.x = cw * cOmega - sw * ci * sOmega;
				P3.y = cw * sOmega + sw * ci * cOmega;
				P3.z = sw * si;

				double3 Q3;
				Q3.x = -sw * cOmega - cw * ci * sOmega;
				Q3.y = -sw * sOmega + cw * ci * cOmega;
				Q3.z = cw * si;

				double cE = cos(E);
				double sE = sin(E);
				double t1 = a * (cE - e);
				double t2 = a * sqrt(1.0 - e * e) * sE;

				x4i.x =  t1 * P3.x + t2 * Q3.x;
				x4i.y =  t1 * P3.y + t2 * Q3.y;
				x4i.z =  t1 * P3.z + t2 * Q3.z;

				double t0 = 1.0 / (1.0 - e * cE) * sqrt(mu / a);
				t1 = -sE;
				t2 = sqrt(1.0 - e * e) * cE;
				v4i.x = t0 * (t1 * P3.x + t2 * Q3.x);
				v4i.y = t0 * (t1 * P3.y + t2 * Q3.y);
				v4i.z = t0 * (t1 * P3.z + t2 * Q3.z);

				x4_d[id] = x4i;
				v4_d[id] = v4i;
			}
//if(id < 10) printf("PR %g %g %g %g %g %g %g %g\n", x4i.x, x4i.y, x4i.z, x4i.w, v4i.x, v4i.y, v4i.z, v4i.w);
		}
	}	
}
// ***************************************************************
// This kernel computes the Poynting-Robertson drag.
// it computes the PR drag  acceleration and performs a velocity kick

// BURNS, LAMY, AND SOTER, 1979 (Radiation Forces on Small Particles in the Solar System)

// January 2019
// Authors: Simon Grimm, Matthias Meier
// *****************************************************************
__global__ void PoyntingRobertsonDrag2(double4 *x4_d, double4 *v4_d, int *index_d, double *dt_d, double Kt, int N, int Nst, int Nstart){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy + Nstart;

	if(id < N + Nstart){
		if(x4_d[id].w >= 0.0){
				
			int st = 0;
			if(Nst > 1) st = index_d[id] / def_MaxIndex;	//st is the sub simulation index

			double4 x4i = x4_d[id];
			double4 v4i = v4_d[id];
			double4 v4it = v4i;
			double3 a3t;


			double dt = dt_d[st] * Kt;
			double RR = v4i.w * def_AU;					//covert radius in m	
			double m = x4i.w;
		
			if(m == 0.0){
				m = Asteroid_rho_c[0] * 4.0 / 3.0 * M_PI * RR * RR * RR; 	//mass in kg;
				m /= def_Solarmass;					//mass im Solar masses
			}
		
			//double eta = 2.53e8 / (Asteroid_rho_c[0] * RR);			//m^2 / s
			double eta = SolarConstant_c[0] * def_AU * def_AU * RR * RR * M_PI / (m * def_Solarmass * def_c * def_c);			//m^2 / s
			eta = eta /(def_AU * def_AU * dayUnit) * 24.0 * 3600.0;		//AU^2 /day * 0.017

			//BURNS, LAMY, AND SOTER, 1979 equation 2

			double rsq = x4i.x * x4i.x + x4i.y * x4i.y + x4i.z * x4i.z;
			double ir = 1.0 / sqrt(rsq);
		
			double t1 = eta * ir * ir * Qpr_c[0];

			//v dependen part with implicit midpoint method
			for(int k = 0; k < 3; ++k){	
			
				double rd = (x4i.x * v4it.x + x4i.y * v4it.y + x4i.z * v4it.z) * ir; 
				double t2 = (def_cm - rd);
//if(id < 10) printf("%d %.20g %.20g %.20g %.20g %.20g %.20g\n", id, RR, eta, rsq, t1, rd, t2);


				a3t.x = t1 * (t2 * x4i.x * ir - v4it.x);
				a3t.y = t1 * (t2 * x4i.y * ir - v4it.y);
				a3t.z = t1 * (t2 * x4i.z * ir - v4it.z);

				v4it.x = v4i.x + 0.5 * dt * a3t.x;
				v4it.y = v4i.y + 0.5 * dt * a3t.y;
				v4it.z = v4i.z + 0.5 * dt * a3t.z;

			}
			//apply the Kick
			v4i.x += a3t.x * dt;
			v4i.y += a3t.y * dt;
			v4i.z += a3t.z * dt;

			x4_d[id] = x4i;
			v4_d[id] = v4i;
//printf("PR %g %g %g %g %g %g %g %g\n", x4i.x, x4i.y, x4i.z, x4i.w, v4i.x, v4i.y, v4i.z, v4i.w);
		}
	}	
}


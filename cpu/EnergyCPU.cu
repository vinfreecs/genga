#include "Orbit2CPU.h"

//copy of ../source/Energy.cu
//replace __device__ with __host__
//remove __inline__
//new kernels 
//replace _d with _h

// **************************************
//This function computes the terms m/r^3 between all pairs of bodies.
//
//Authors: Simon Grimm, Joachim Stadel
//March 2014

// ****************************************
double PE(double4 x4i, double4 x4j, int i, int j){

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
//****************************************/
double PESun(double4 x4i, double ksqMsun){

	double rsq, ir;
	double a = 0.0;

	rsq = x4i.x * x4i.x + x4i.y * x4i.y + x4i.z * x4i.z;
	if(rsq > 0.0){
		ir = 1.0/sqrt(rsq);
		a = -ksqMsun * x4i.w * ir;
	}
	return a;

}

//remove idy parallelism and do in serial
//only parallelize idx
//remove reduction sum and shared memory
//replace idx if condition with for loop
void potentialEnergy_cpu(double4 *x4_h, double4 *v4_h, const double Msun, double *EnergySum_h, const int N){

	for(int idx = 0; idx < N; ++idx){
		double V = 0.0;
		if(x4_h[idx].w > 0.0){
			for(int i = 0; i < N; ++i){
				if(x4_h[i].w > 0.0){
					V += PE(x4_h[idx], x4_h[i], idx, i);
				}	
			}


			V *= 0.5 * def_ksq * x4_h[idx].w;
			V += PESun(x4_h[idx], def_ksq * Msun);

			EnergySum_h[idx] = V;

		}
		else{
			EnergySum_h[idx] = 0.0;
		}
	}
}

//no paralell cpu version, could be done later with reduction
void EjectionEnergy_cpu(double4 *x4_h, double4 *v4_h, double4 *spin_h, double Msun, int idx, double *U_h, double *LI_h, double3 *vcom_h, const int N){


	//--------------------------------------------
	//calculate s_s and p_s first
	//--------------------------------------------
	double3 p = {0.0, 0.0, 0.0};
	double3 s = {0.0, 0.0, 0.0};


	for(int i = 0; i < N; ++i){
		double m = x4_h[i].w;
		if(m >= 0.0){
			p.x += m * v4_h[i].x;
			p.y += m * v4_h[i].y;
			p.z += m * v4_h[i].z;
			s.x += m * x4_h[i].x;
			s.y += m * x4_h[i].y;
			s.z += m * x4_h[i].z;
		}
	}
	//--------------------------------------------
	double V = 0.0;
	double T = 0.0;
	double ms = 0.0;
	double mtot = 0.0;
	double3 L = {0.0, 0.0, 0.0};


	for(int i = 0; i < N; ++i){
		double m = x4_h[i].w;
		if(m >= 0.0){
			ms += m;
			V += PE(x4_h[idx], x4_h[i], idx, i);
			T += 0.5 * m * (v4_h[i].x * v4_h[i].x +  v4_h[i].y * v4_h[i].y + v4_h[i].z * v4_h[i].z);
			//convert to barycentric positions
			double3 x4h;
			x4h.x = x4_h[i].x - s.x / Msun;
			x4h.y = x4_h[i].y - s.y / Msun;
			x4h.z = x4_h[i].z - s.z / Msun;
			L.x += m * (x4h.y * v4_h[i].z - x4h.z * v4_h[i].y) + spin_h[i].x;
			L.y += m * (x4h.z * v4_h[i].x - x4h.x * v4_h[i].z) + spin_h[i].y;
			L.z += m * (x4h.x * v4_h[i].y - x4h.y * v4_h[i].x) + spin_h[i].z;
//printf("L ejection 1 %d %.20g %.20g %.20g\n", 0, L.x, L.y, L.z);

		}
	}


	V *= def_ksq * x4_h[idx].w;

	V += PESun(x4_h[idx], def_ksq * Msun);
	double Tsun0 = 0.5 / Msun * ( p.x * p.x + p.y * p.y + p.z * p.z);
	
	mtot = Msun + ms - x4_h[idx].w;
	
	double3 Vsun;
	Vsun.x = -p.x / Msun + x4_h[idx].w * v4_h[idx].x/mtot;
	Vsun.y = -p.y / Msun + x4_h[idx].w * v4_h[idx].y/mtot;
	Vsun.z = -p.z / Msun + x4_h[idx].w * v4_h[idx].z/mtot;
	
	double Tsun1 = 0.5 * Msun * (Vsun.x * Vsun.x + Vsun.y * Vsun.y + Vsun.z * Vsun.z);
	
	*U_h += -Tsun1 + Tsun0 + T + V;


	L.x += (s.y * p.z - s.z * p.y) / Msun;
	L.y += (s.z * p.x - s.x * p.z) / Msun;
	L.z += (s.x * p.y - s.y * p.x) / Msun;
	double Ltot = sqrt(L.x * L.x + L.y * L.y + L.z * L.z);
//printf("Ltot ejection 1 %.20g %.20g %.20g\n", Ltot, LI_h[0], Ltot + LI_h[0]);
	LI_h[0] += Ltot;



	s.x -= x4_h[idx].w * x4_h[idx].x;
	s.y -= x4_h[idx].w * x4_h[idx].y;
	s.z -= x4_h[idx].w * x4_h[idx].z;


	double3 vcom;
	vcom.x = x4_h[idx].w * v4_h[idx].x / mtot;
	vcom.y = x4_h[idx].w * v4_h[idx].y / mtot;
	vcom.z = x4_h[idx].w * v4_h[idx].z / mtot;


	vcom_h[0].x = vcom.x;
	vcom_h[0].y = vcom.y;
	vcom_h[0].z = vcom.z;
	
	
	for (int i = 0; i < N; ++i){
		v4_h[i].x += vcom.x;
		v4_h[i].y += vcom.y;
		v4_h[i].z += vcom.z;
	}
	

	//mark here the particle as ghost particle	
	x4_h[idx].w = -1.0e-12;

	// ---------------------------------------------
	//redo p_s now
	// ---------------------------------------------

	p = {0.0, 0.0, 0.0};


	for(int i = 0; i < N; ++i){
		double m = x4_h[i].w;
		if(m >= 0.0){
			p.x += m * v4_h[i].x;
			p.y += m * v4_h[i].y;
			p.z += m * v4_h[i].z;
		}
	}

	// ------------------------------------------------------
	//redo now L calculation without the ejected particle
	// ------------------------------------------------------

	T = 0.0;
	L = {0.0, 0.0, 0.0};


	for(int i = 0; i < N; ++i){
		double m = x4_h[i].w;
		if(m >= 0.0){
			T += 0.5 *x4_h[i].w * (v4_h[i].x * v4_h[i].x +  v4_h[i].y * v4_h[i].y + v4_h[i].z * v4_h[i].z);
			//convert to barycentric positions
			double3 x4h;
			x4h.x = x4_h[i].x - s.x / Msun;
			x4h.y = x4_h[i].y - s.y / Msun;
			x4h.z = x4_h[i].z - s.z / Msun;
			L.x += m * (x4h.y * v4_h[i].z - x4h.z * v4_h[i].y) + spin_h[i].x;
			L.y += m * (x4h.z * v4_h[i].x - x4h.x * v4_h[i].z) + spin_h[i].y;
			L.z += m * (x4h.x * v4_h[i].y - x4h.y * v4_h[i].x) + spin_h[i].z;
//printf("L ejection 2 %d %.20g %.20g %.20g\n", i L.x, L.y, L.z);

		}
	}

	L.x += (s.y * p.z - s.z * p.y) / Msun;
	L.y += (s.z * p.x - s.x * p.z) / Msun;
	L.z += (s.x * p.y - s.y * p.x) / Msun;
	Ltot = sqrt(L.x * L.x + L.y * L.y + L.z * L.z);
//printf("Ltot ejection 2 %.20g %.20g %.20g\n", Ltot, LI_h[0], Ltot + LI_h[0]);
	LI_h[0] -= Ltot;
	*U_h -= T;

}

//no paralell cpu version, could be done later with reduction
void kineticEnergy_cpu(double4 *x4_h, double4 *v4_h, double4 *spin_h, double *EnergySum_h, double *Energy_h, double Msun, double *U_h, double *LI_h, double *Energy0_h, double *LI0_h, int st, int N, int EE){

	 double T = 0.0;
	 double V = 0.0;
	 double E = 0.0;
	 double3 p = {0.0, 0.0, 0.0};
	 double3 s = {0.0, 0.0, 0.0};
	 double3 L = {0.0, 0.0, 0.0};

	for(int i = 0; i < N; ++i){
		double m = x4_h[i].w;
		if(m > 0.0){
			s.x += m * x4_h[i].x;
			s.y += m * x4_h[i].y;
			s.z += m * x4_h[i].z;
			p.x += m * v4_h[i].x;
			p.y += m * v4_h[i].y;
			p.z += m * v4_h[i].z;
		}
	}


	for(int i = 0; i < N; ++i){
		V += EnergySum_h[i];
		EnergySum_h[i] = 0.0;
		double4 x4 = x4_h[i];
		double4 v4 = v4_h[i];
		if(x4.w > 0.0){
			T += 0.5 * x4.w * (v4.x * v4.x + v4.y * v4.y + v4.z * v4.z);
		}
		//convert to barycentric positions
		double3 x4h;
		x4h.x = x4.x - s.x / Msun;
		x4h.y = x4.y - s.y / Msun;
		x4h.z = x4.z - s.z / Msun;
		L.x += x4.w * (x4h.y * v4.z - x4h.z * v4.y) + spin_h[i].x;
		L.y += x4.w * (x4h.z * v4.x - x4h.x * v4.z) + spin_h[i].y;
		L.z += x4.w * (x4h.x * v4.y - x4h.y * v4.x) + spin_h[i].z;
	}
//printf("L %d %.20g %.20g %.20g\n", i, L.x, L.y, L.z);
	E = V + T;

	double Tsun = 0.5 / Msun * (p.x*p.x + p.y*p.y + p.z*p.z);  
	//Lsun
//printf("Lsum %d %.20g %.20g %.20g\n", 0, L.x, L.y, L.z);
	L.x += (s.y * p.z - s.z * p.y) / Msun;
	L.y += (s.z * p.x - s.x * p.z) / Msun;
	L.z += (s.x * p.y - s.y * p.x) / Msun;
//printf("LSun %.20g %.20g %.20g\n", (s.y * p.z - s.z * p.y) / Msun, (s.z * p.x - s.x * p.z) / Msun, (s.x * p.y - s.y * p.x) / Msun);
//printf("Lsum+ %d %.20g %.20g %.20g\n", 0, L.x, L.y, L.z);
	double Ltot = sqrt(L.x * L.x + L.y * L.y + L.z * L.z);
//printf("Ltot %.20g %.20g %.20g\n", Ltot, LI_h[0], Ltot + LI_h[0]);
	V *= def_Kg;
	T *= def_Kg;
	E *= def_Kg;
	Tsun *= def_Kg;
	Energy_h[0] = V;
	Energy_h[1] = T + Tsun;
	Energy_h[2] = LI_h[st] * def_Kg;
	Energy_h[3] = U_h[st] * def_Kg;
	Energy_h[4] = T + V + (U_h[st] * def_Kg) + Tsun;
	Energy_h[5] = (Ltot + LI_h[st]) * def_Kg;

	if(EE == 0){

		Energy0_h[st] = T + V + (U_h[st] * def_Kg) + Tsun;
		LI0_h[st] = (Ltot + LI_h[st]) * def_Kg;
		Energy_h[7] = 0.0;
		Energy_h[6] = 0.0;
	}
	if(EE == 1){
		Energy_h[6] = ((Ltot + LI_h[st]) * def_Kg - LI0_h[st]) / LI0_h[st]; 
		Energy_h[7] = ((T + V + (U_h[st] * def_Kg) + Tsun) - Energy0_h[st]) / Energy0_h[st];
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


	potentialEnergy_cpu  (x4_h + NBS, v4_h + NBS, Msun_h[st].x, EnergySum_h + NBS, NN);
	kineticEnergy_cpu  (x4_h + NBS, v4_h + NBS, spin_h + NBS, EnergySum_h + NBS, Energy_h + NE, Msun_h[st].x, U_h, LI_h, Energy0_h, LI0_h, st, NN, E);
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

	EjectionEnergy_cpu (x4_h + NBS, v4_h + NBS, spin_h + NBS, Msun_h[st].x, i, U_h + st, LI_h + st, vcom_h + st, NN);
}


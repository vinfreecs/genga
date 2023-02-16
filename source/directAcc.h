#ifndef DIRECTACC_H
#define DIRECTACC_H


//**************************************
//This function computes the term a = mj/rij^3 * (1 - Kij).
//
//Authors: Simon Grimm, Joachim Stadel
//March 2014
//
// ****************************************
__device__ void accEnc(double4 x4i, double4 x4j, double3 &ac, volatile double *rcritv_, double &test, const int i, const int j, const int NN, const double MinMass, const int UseTestParticles, const int SLevels){

	int c = 0;
	if(UseTestParticles == 0 && x4i.w >= 0.0 && x4j.w >= 0.0) c = 1;
	if(UseTestParticles == 1 && x4i.w >= 0.0 && x4j.w > MinMass) c = 1;
	if(UseTestParticles == 2 && (x4i.w >= 0.0 && x4j.w >= 0.0) && (x4i.w > MinMass || x4j.w > MinMass)) c = 1;

	if(c == 1 && i != j){

		double3 r3;
		double rsq;
		double ir, ir3;
		double s;
		double y, yy;
		double rcritv, rcritv2;

		r3.x = x4j.x - x4i.x;
		r3.y = x4j.y - x4i.y;
		r3.z = x4j.z - x4i.z;

		rsq = r3.x*r3.x + r3.y*r3.y + r3.z*r3.z + 1.0e-30;
		ir = 1.0/sqrt(rsq);
		ir3 = ir * ir * ir;
		s = x4j.w * ir3 * def_ksq;

		for(int l = 0; l < SLevels; ++l){		

			double rcritvi = rcritv_[i + l * NN];
			double rcritvj = rcritv_[j + l * NN];
			rcritv = fmax(rcritvi, rcritvj);

			rcritv2 = rcritv * rcritv;

			if(rsq <  1.0 * rcritv2){
				if(rsq <= 0.01 * rcritv2){
					s *= 1.0;
				}
				else{
					y = (rsq * ir - 0.1 * rcritv)/(0.9*rcritv);
					yy = y * y;
					s *= (1.0 - yy / (2.0*yy - 2.0*y + 1.0));
				}
			}
			else{
				s = 0.0;
			}
//printf("%.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g\n", x4i.w, x4i.x, x4i.y, x4i.z, x4j.w, x4j.x, x4j.y, x4j.z);
		}
		ac.x += __dmul_rn(r3.x, s);
		ac.y += __dmul_rn(r3.y, s);
		ac.z += __dmul_rn(r3.z, s);
	}

}
//**************************************
//This function computes the term a = mj/rij^3.
//
//Authors: Simon Grimm
//December 2016
// ****************************************
__device__ void accEncFull(double4 x4i, double4 x4j, double3 &ac, double &test, int i, int j, double MinMass, int UseTestParticles){

	int c = 0;
	if(UseTestParticles == 0 && x4i.w >= 0.0 && x4j.w >= 0.0) c = 1;
	if(UseTestParticles == 1 && x4i.w >= 0.0 && x4j.w > MinMass) c = 1;
	if(UseTestParticles == 2 && (x4i.w >= 0.0 && x4j.w >= 0.0) && (x4i.w > MinMass || x4j.w > MinMass)) c = 1;

	if(c == 1 && i != j){

		double3 r3;
		double rsq;
		double ir, ir3;
		double s;

		r3.x = x4j.x - x4i.x;
		r3.y = x4j.y - x4i.y;
		r3.z = x4j.z - x4i.z;

		rsq = r3.x*r3.x + r3.y*r3.y + r3.z*r3.z + 1.0e-30;

		ir = 1.0 / sqrt(rsq);
		ir3 = ir * ir * ir;
		s = x4j.w * ir3 * def_ksq;

		ac.x += __dmul_rn(r3.x, s);
		ac.y += __dmul_rn(r3.y, s);
		ac.z += __dmul_rn(r3.z, s);
//printf("%.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g\n", x4i.w, x4i.x, x4i.y, x4i.z, x4j.w, x4j.x, x4j.y, x4j.z);
	}
}
//**************************************
//This function is only here for testing
//
//Authors: Simon Grimm, Joachim Stadel
//March 2014
//
// ****************************************
__device__ inline void accEncG3(double4 x4i, double4 x4j, double3 &ac, double &test, int i, int j, double time, double K, double MinMass, int UseTestParticles){

	int c = 0;
	if(UseTestParticles == 0 && x4i.w >= 0.0 && x4j.w >= 0.0) c = 1;
	if(UseTestParticles == 1 && x4i.w >= 0.0 && x4j.w > MinMass) c = 1;
	if(UseTestParticles == 2 && (x4i.w >= 0.0 && x4j.w >= 0.0) && (x4i.w > MinMass || x4j.w > MinMass)) c = 1;

	if(c == 1 && i != j){

		double3 r3;
		double rsq;
		double ir, ir3;
		double s;

		r3.x = x4j.x - x4i.x;
		r3.y = x4j.y - x4i.y;
		r3.z = x4j.z - x4i.z;

		rsq = r3.x*r3.x + r3.y*r3.y + r3.z*r3.z + 1.0e-30;

		ir = 1.0/sqrt(rsq);
		ir3 = ir * ir * ir;

		s = ir3 * def_ksq * (1.0 - K) * x4j.w;

		ac.x += __dmul_rn(r3.x, s);
		ac.y += __dmul_rn(r3.y, s);
		ac.z += __dmul_rn(r3.z, s);
//if(s != 0.0) printf("%.20g %d %d %.20g %g accEnc\n", time, i, j, s, K);

	}
}

// **************************************
//This function computes the acceleration between the sun and body i.
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
// ****************************************
__device__ void accEncSun(double4 x4i, double3 &ac, const double ksqMsun){

	if(x4i.w >= 0.0){
		double rsq;
		double ir, ir3;
		double s;

		rsq = x4i.x*x4i.x + x4i.y*x4i.y + x4i.z*x4i.z + 1.0e-30;
		ir = 1.0/sqrt(rsq);
		ir3 = ir * ir * ir;
		s = - ksqMsun * ir3;

		ac.x += s * x4i.x;
		ac.y += s * x4i.y;
		ac.z += s * x4i.z;
	}
}
// ***************************************************
//This fuction corrects the first kick of the time step
//
//Authors: Simon Grimm, Joachim Stadel
//March 2014
//
// **************************************************
__device__ inline void CorrectKick(double4 x4i, double4 x4j, double3 &ac, double K, double Kold, double &test, int i, int j, double time, int NB, double MinMass, int UseTestParticles){

	int c = 0;
	if(UseTestParticles == 0 && x4i.w >= 0.0 && x4j.w >= 0.0) c = 1;
	if(UseTestParticles == 1 && x4i.w >= 0.0 && x4j.w > MinMass) c = 1;
	if(UseTestParticles == 2 && (x4i.w >= 0.0 && x4j.w >= 0.0) && (x4i.w > MinMass || x4j.w > MinMass)) c = 1;

	if(c == 1 && i != j){

		double3 r3;
		double rsq;
		double ir, ir3;
		double s;

		r3.x = x4j.x - x4i.x;
		r3.y = x4j.y - x4i.y;
		r3.z = x4j.z - x4i.z;

		rsq = r3.x*r3.x + r3.y*r3.y + r3.z*r3.z;

		ir = 1.0/sqrt(rsq);
		ir3 = ir * ir * ir;
		s = 0.0;

		//correct
#if def_G3 == 2
		if(Kold < 1.0) Kold = 0.0;
#endif
		s = (K - Kold) * x4j.w * ir3 * def_ksq;

		ac.x += __dmul_rn(r3.x, s);
		ac.y += __dmul_rn(r3.y, s);
		ac.z += __dmul_rn(r3.z, s);
/*if(s != 0.0)*/ printf("%.20g %d %d %.20g %.20g %.20g %.20g correct %g %g %g\n", time, i, j, s, Kold, K, ac.x, Kold * x4j.w * ir3 * def_ksq, Kold * x4j.w * ir3 * def_ksq * r3.x, K * x4j.w * ir3 * def_ksq * r3.x);
	}
}
// ***********************************************
//This fuction corrects the second kick of the time step
//
//Authors: Simon Grimm, Joachim Stadel
//March 2014
//*************************************************
__device__ inline void CorrectKick2(double4 x4i, double4 x4j, double3 &ac, double K, double Kold, double &test, int i, int j, double time, int E, double MinMass, int UseTestParticles){
	
	int c = 0;
	if(UseTestParticles == 0 && x4i.w >= 0.0 && x4j.w >= 0.0) c = 1;
	if(UseTestParticles == 1 && x4i.w >= 0.0 && x4j.w > MinMass) c = 1;
	if(UseTestParticles == 2 && (x4i.w >= 0.0 && x4j.w >= 0.0) && (x4i.w > MinMass || x4j.w > MinMass)) c = 1;

	if(c == 1 && i != j){
		double3 r3;
		double rsq;
		double ir, ir3;
		double s;

		r3.x = x4j.x - x4i.x;
		r3.y = x4j.y - x4i.y;
		r3.z = x4j.z - x4i.z;

		rsq = r3.x*r3.x + r3.y*r3.y + r3.z*r3.z;

		ir = 1.0/sqrt(rsq);
		ir3 = ir * ir * ir;

		s = K * x4j.w * ir3 * def_ksq; 
		ac.x += __dmul_rn(r3.x, s);
		ac.y += __dmul_rn(r3.y, s);
		ac.z += __dmul_rn(r3.z, s);
/*if(s != 0.0)*/ printf("%.20g %d %d %.20g %.20g %.20g correct2 %d\n", time, i, j, s, K, ac.x, E);
	}
}

// **************************************
//This function performs a merger of two bodies i and j.
//It also calculates the amount of energy which will be lost due
//to the collision: U = 0.5 * mi* mj/(mi + mj) * vij^2 - G * mi * mj / rij
//The index of the new bodie is the index of the more massiv one. If both bodies
//have an equal mass then the new index is the smaller one.
//
// rd can be used to generate random numbers for more complex collision models.
//
//Authors: Simon Grimm, Joachim Stadel
//March 2014
//
//****************************************
__device__ void collide(curandState &random, volatile double4 *x4, volatile double4 *v4, const int i, const int j, const int indexi, const int indexj, const double Msun, double *U_d, double &test, int *index, const int nc, double *Coll, double time, double4 *spin, double3 *love, int *createFlag, volatile double *rcritv, double *rcrit_d, const int NN, const int NconstT, float4 *aelimits, unsigned int *aecount, unsigned int *enccount, unsigned long long *aecountT, unsigned long long *enccountT, const int SLevels, const int noColl){

#if USE_RANDOM == 1
//	double rd = curand_uniform(&random);
	//This is a random number that can be used on a more complex collision model
#endif


	if(noColl != 1 && noColl != -1){
//printf("collide %d %d %g %g\n", (index[indexi]), (index[indexj]), time, rd);
//printf("collide %d %d %g CollisionModel: %d\n", (index[indexi]), (index[indexj]), time, CollisionModel_c[0]);

		Coll[nc * def_NColl + 0] = time/365.25;
		Coll[nc * def_NColl + 1] = (double)(index[indexi]);
		Coll[nc * def_NColl + 2] = x4[i].w;
		Coll[nc * def_NColl + 3] = v4[i].w;
		Coll[nc * def_NColl + 4] = x4[i].x;
		Coll[nc * def_NColl + 5] = x4[i].y;
		Coll[nc * def_NColl + 6] = x4[i].z;
		Coll[nc * def_NColl + 7] = v4[i].x;
		Coll[nc * def_NColl + 8] = v4[i].y;
		Coll[nc * def_NColl + 9] = v4[i].z;
		Coll[nc * def_NColl + 10] = spin[indexi].x;
		Coll[nc * def_NColl + 11] = spin[indexi].y;
		Coll[nc * def_NColl + 12] = spin[indexi].z;
		Coll[nc * def_NColl + 13] = (double)(index[indexj]);
		Coll[nc * def_NColl + 14] = x4[j].w;
		Coll[nc * def_NColl + 15] = v4[j].w;
		Coll[nc * def_NColl + 16] = x4[j].x;
		Coll[nc * def_NColl + 17] = x4[j].y;
		Coll[nc * def_NColl + 18] = x4[j].z;
		Coll[nc * def_NColl + 19] = v4[j].x;
		Coll[nc * def_NColl + 20] = v4[j].y;
		Coll[nc * def_NColl + 21] = v4[j].z;
		Coll[nc * def_NColl + 22] = spin[indexj].x;
		Coll[nc * def_NColl + 23] = spin[indexj].y;
		Coll[nc * def_NColl + 24] = spin[indexj].z;
	}

	//if(CollisionModel_c[0] != 0){
		//Here another collision model can be implemented
		//CollisionModel == 0 means perfect accretion
	//}
	//else{ ..... } 

	double3 vij;
	double3 rij;
	double3 L;

	double mimj = x4[i].w * x4[j].w;
	double mtot = x4[i].w + x4[j].w;

	rij.x = x4[j].x - x4[i].x;
	rij.y = x4[j].y - x4[i].y;
	rij.z = x4[j].z - x4[i].z;

	vij.x = v4[j].x - v4[i].x;
	vij.y = v4[j].y - v4[i].y;
	vij.z = v4[j].z - v4[i].z;

	L.x = mimj/mtot * ( rij.y * vij.z - rij.z * vij.y);
	L.y = mimj/mtot * (-rij.x * vij.z + rij.z * vij.x);
	L.z = mimj/mtot * ( rij.x * vij.y - rij.y * vij.x);

	double rsq = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z + 1.0e-30;
	double vsq = vij.x * vij.x + vij.y * vij.y + vij.z * vij.z + 1.0e-30;

	if(noColl == 0){
		*U_d += 0.5 * mimj / mtot * vsq - mimj * def_ksq / sqrt(rsq);
	}

	x4[i].x = (x4[i].x * x4[i].w + x4[j].x * x4[j].w) / mtot;
	x4[i].y = (x4[i].y * x4[i].w + x4[j].y * x4[j].w) / mtot;
	x4[i].z = (x4[i].z * x4[i].w + x4[j].z * x4[j].w) / mtot;

	v4[i].x = (v4[i].x * x4[i].w + v4[j].x * x4[j].w) / mtot;
	v4[i].y = (v4[i].y * x4[i].w + v4[j].y * x4[j].w) / mtot;
	v4[i].z = (v4[i].z * x4[i].w + v4[j].z * x4[j].w) / mtot;

	for(int l = 0; l < SLevels; ++l){
		rcritv[i + l * NN] = fmax(rcritv[i + l * NN], rcritv[j + l * NN]);
		rcritv[j + l * NN] = 0.0;
	
		rcrit_d[indexi + l * NconstT] = fmax(rcrit_d[indexi + l * NconstT], rcrit_d[indexj + l * NconstT]);
		rcrit_d[indexj + l * NconstT] = 0.0;
	}


	spin[indexi].x += spin[indexj].x + L.x;
	spin[indexi].y += spin[indexj].y + L.y;
	spin[indexi].z += spin[indexj].z + L.z;

	v4[j].x = 0.0;
	v4[j].y = 0.0;
	v4[j].z = 0.0;

	x4[j].x = 0.0;
	x4[j].y = 1.0;
	x4[j].z = 0.0;

	spin[indexj].x = 0.0;
	spin[indexj].y = 0.0;
	spin[indexj].z = 0.0;
	

	double Ic = (x4[i].w * spin[indexi].w + x4[j].w * spin[indexj].w) / mtot;
	spin[indexi].w = Ic;
	spin[indexj].w = 0.0;
	
	double k2 = (x4[i].w * love[indexi].x + x4[j].w * love[indexj].x) / mtot;
	love[indexi].x = k2;
	love[indexj].x = 0.0;

	double k2f = (x4[i].w * love[indexi].y + x4[j].w * love[indexj].y) / mtot;
	love[indexi].y = k2f;
	love[indexj].y = 0.0;

	double tau = (x4[i].w * love[indexi].z + x4[j].w * love[indexj].z) / mtot;
	love[indexi].z = tau;
	love[indexj].z = 0.0;

	if(CreateParticlesParameters_c[0] > 0){
		createFlag[indexi] = max(createFlag[indexi], createFlag[indexj]);
		createFlag[indexj] = 0;
	}

	if(x4[i].w < x4[j].w){
		index[indexi] = index[indexj];
		
		if(noColl == 0){
			aelimits[indexi] = aelimits[indexj];
			aecount[indexi] = aecount[indexj];
			enccount[indexi] = enccount[indexj];
			aecountT[indexi] = aecountT[indexj];
			enccountT[indexi] = enccountT[indexj];
		}
	}
	if(x4[i].w == x4[j].w){
		index[indexi] = min(index[indexi], index[indexj]);
		if(noColl == 0){
			aelimits[indexi] = aelimits[min(indexi, indexj)];
			aecount[indexi] = aecount[min(indexi, indexj)];
			enccount[indexi] = enccount[min(indexi, indexj)];
			aecountT[indexi] = aecountT[min(indexi, indexj)];
			enccountT[indexi] = enccountT[min(indexi, indexj)];
		}
	}
	index[indexj] = -1;

//	if((StopAtCollision_c[0] == 0 && CollTshift_c[0] == 1.0)){
		x4[i].w = mtot;
//	}
//	else{
		//prevent from following collisions in the same time step
//		if(x4[j].w >= StopMinMass_c[0] && x4[i].w >= StopMinMass_c[0]){
//			x4[i].w = -1.0e-12;
//printf("Stop At Collision %d %g %g\n", StopAtCollision_c[0], StopMinMass_c[0], time);
//		}
//		else{
//			x4[i].w = mtot;
//		}
//	}
	x4[j].w = -1.0e-12;
	//radius
	v4[i].w = cbrt(v4[i].w * v4[i].w * v4[i].w + v4[j].w * v4[j].w * v4[j].w);
	v4[j].w = 0.0;
}


// **************************************
// This function stores the details of close encounters
//
// Authors: Simon Grimm
// April 2016
//
//****************************************
__device__ void storeEncounters(volatile double4 *x4, volatile double4 *v4, int i, int j, int indexi, int indexj, int *index, int nc, double *Coll, double time, double4 *spin){

//printf("Enc %d %d %d %d %d %d\n", i, j, indexi, indexj, index[indexj], index[indexi]);
	Coll[nc * def_NColl + 0] = time/365.25;
	Coll[nc * def_NColl + 1] = (double)(index[indexj]);
	Coll[nc * def_NColl + 2] = x4[j].w;
	Coll[nc * def_NColl + 3] = v4[j].w;
	Coll[nc * def_NColl + 4] = x4[j].x;
	Coll[nc * def_NColl + 5] = x4[j].y;
	Coll[nc * def_NColl + 6] = x4[j].z;
	Coll[nc * def_NColl + 7] = v4[j].x;
	Coll[nc * def_NColl + 8] = v4[j].y;
	Coll[nc * def_NColl + 9] = v4[j].z;
	Coll[nc * def_NColl + 10] = spin[indexj].x;
	Coll[nc * def_NColl + 11] = spin[indexj].y;
	Coll[nc * def_NColl + 12] = spin[indexj].z;
	Coll[nc * def_NColl + 13] = (double)(index[indexi]);
	Coll[nc * def_NColl + 14] = x4[i].w;
	Coll[nc * def_NColl + 15] = v4[i].w;
	Coll[nc * def_NColl + 16] = x4[i].x;
	Coll[nc * def_NColl + 17] = x4[i].y;
	Coll[nc * def_NColl + 18] = x4[i].z;
	Coll[nc * def_NColl + 19] = v4[i].x;
	Coll[nc * def_NColl + 20] = v4[i].y;
	Coll[nc * def_NColl + 21] = v4[i].z;
	Coll[nc * def_NColl + 22] = spin[indexi].x;
	Coll[nc * def_NColl + 23] = spin[indexi].y;
	Coll[nc * def_NColl + 24] = spin[indexi].z;
}
#endif

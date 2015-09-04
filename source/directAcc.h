#ifndef DIRECTACC_H
#define DIRECTACC_H

//**************************************
//This function computes the term a = mj/rij^3 * (1 - Kij).
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
// ****************************************
__device__ void accEnc(double4 x4i, double4 x4j, double3 &ac, double rcritvi, double rcritvj, double &test, int i, int j){
	if(x4i.w >= 0 && x4j.w > 0 && i != j){

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
		rcritv = fmax(rcritvi, rcritvj);

		rcritv2 = rcritv * rcritv;

		if(rsq <  1.0 * rcritv2){
			ir = 1.0/sqrt(rsq);
			ir3 = ir * ir * ir;
			if(rsq <= 0.01 * rcritv2){
				s = x4j.w * ir3 * ksq;
			}
			else{
				y = (rsq * ir - 0.1 * rcritv)/(0.9*rcritv);
				yy = y * y;
				s = ir3 * ksq * (1.0 - yy / (2.0*yy - 2.0*y + 1.0)) * x4j.w;
			}
                        ac.x += __dmul_rn(r3.x, s);
                        ac.y += __dmul_rn(r3.y, s);
                        ac.z += __dmul_rn(r3.z, s);
		}
	}
}
//**************************************
//This function is only here for testing
//
//Authors: Simon Grimm, Joachim Stadel
////March 2014
//
// ****************************************
__device__ inline void accEncG3(double4 x4i, double4 x4j, double3 &ac, double &test, int i, int j, double time, double K){
	if(x4i.w >= 0 && x4j.w > 0 && i != j){

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

		s = ir3 * ksq * (1.0 - K) * x4j.w;

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

	if(x4i.w >= 0){
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
__device__ inline void CorrectKick(double4 x4i, double4 x4j, double3 &ac, int groupIndexOldi, int groupIndexOldj, double K, double Kold, double &test, int i, int j, double time, int NB){
        if(x4i.w >= 0 && x4j.w > 0 && i != j){

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
		if(Kold < 1.0) Kold = 0.0;

		s = (K - Kold) * x4j.w * ir3 * ksq;


//if(s != 0.0) printf("%.20g %d %d %.20g %.20g %.20g correct\n", time, i, j, s, Kold, K);
		ac.x += __dmul_rn(r3.x, s);
		ac.y += __dmul_rn(r3.y, s);
		ac.z += __dmul_rn(r3.z, s);
	}
}
// ***********************************************
//This fuction corrects the second kick of the time step
//
//Authors: Simon Grimm, Joachim Stadel
//March 2014
//*************************************************
__device__ inline void CorrectKick2(double4 x4i, double4 x4j, double3 &ac, double K, double Kold, double &test, int i, int j, double time, int E){
        if(x4i.w >= 0 && x4j.w > 0 && i != j){
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

		s = K * x4j.w * ir3 * ksq; 
//if(s != 0.0) printf("%.20g %d %d %.20g correct2 %d\n", time, i, j, K, E);
		ac.x += __dmul_rn(r3.x, s);
		ac.y += __dmul_rn(r3.y, s);
		ac.z += __dmul_rn(r3.z, s);
	}
}

// **************************************
//This function performs a merger of two bodies i and j.
//It also calculates the amount of energy which will be lost due
//to the collision: U = 0.5 * mi* mj/(mi + mj) * vij^2 - G * mi * mj / rij
//The index of the new bodie is the index of the more massiv one. If both bodies
//have an equal mass then the new index is the smaller one.
//
//Authors: Simon Grimm, Joachim Stadel
//March 2014
//
//****************************************
__device__ void collide(double4 *x4, double4 *v4, int i, int j, int indexi, int indexj, const double Msun, double *U_d, double &test, int *index, int nc, double *Coll, double time, double3 *spin, double *rcritv, double *rcrit_d, int Nst, int NBS, float4 *aelimits, int *aecount, int *enccount, long long *aecountT, long long *enccountT){

        if(Nst == 1){
		printf("Collision between body %d and %d\n", index[indexi], index[indexj]);
	}
	double3 vij;
	double3 rij;
	double rsq, vsq, ir, mimj, mtot;
	double3 p, s;
	double3 L;
	double rcrit;


	Coll[nc * 25 + 0] = time/365.25;
	Coll[nc * 25 + 1] = (double)(index[indexi]);
	Coll[nc * 25 + 2] = x4[i].w;
	Coll[nc * 25 + 3] = v4[i].w;
	Coll[nc * 25 + 4] = x4[i].x;
	Coll[nc * 25 + 5] = x4[i].y;
        Coll[nc * 25 + 6] = x4[i].z;
        Coll[nc * 25 + 7] = v4[i].x;
        Coll[nc * 25 + 8] = v4[i].y;
	Coll[nc * 25 + 9] = v4[i].z;
	Coll[nc * 25 + 10] = spin[indexi].x;
	Coll[nc * 25 + 11] = spin[indexi].y;
	Coll[nc * 25 + 12] = spin[indexi].z;
	Coll[nc * 25 + 13] = (double)(index[indexj]);
        Coll[nc * 25 + 14] = x4[j].w;
	Coll[nc * 25 + 15] = v4[j].w;
        Coll[nc * 25 + 16] = x4[j].x;
        Coll[nc * 25 + 17] = x4[j].y;
        Coll[nc * 25 + 18] = x4[j].z;
        Coll[nc * 25 + 19] = v4[j].x;
        Coll[nc * 25 + 20] = v4[j].y;
        Coll[nc * 25 + 21] = v4[j].z;
      	Coll[nc * 25 + 22] = spin[indexj].x;
       	Coll[nc * 25 + 23] = spin[indexj].y;
       	Coll[nc * 25 + 24] = spin[indexj].z;

	s.x = x4[i].x * x4[i].w + x4[j].x * x4[j].w;
	s.y = x4[i].y * x4[i].w + x4[j].y * x4[j].w;
	s.z = x4[i].z * x4[i].w + x4[j].z * x4[j].w;

	p.x = v4[i].x * x4[i].w + v4[j].x * x4[j].w;
	p.y = v4[i].y * x4[i].w + v4[j].y * x4[j].w;
	p.z = v4[i].z * x4[i].w + v4[j].z * x4[j].w;

	mimj = x4[i].w * x4[j].w;
	mtot = x4[i].w + x4[j].w;

	rij.x = x4[j].x - x4[i].x;
	rij.y = x4[j].y - x4[i].y;
	rij.z = x4[j].z - x4[i].z;

	vij.x = v4[j].x - v4[i].x;
	vij.y = v4[j].y - v4[i].y;
	vij.z = v4[j].z - v4[i].z;

	L.x = mimj/mtot * ( rij.y * vij.z - rij.z * vij.y);
	L.y = mimj/mtot * (-rij.x * vij.z + rij.z * vij.x);
	L.z = mimj/mtot * ( rij.x * vij.y - rij.y * vij.x);

	rsq = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z + 1.0e-30;
	ir = 1.0/sqrt(rsq);

	vsq = vij.x * vij.x + vij.y * vij.y + vij.z * vij.z + 1.0e-30;

	*U_d += 0.5 * mimj / mtot * vsq - mimj * ksq * ir;

	x4[i].x = s.x / mtot;
	x4[i].y = s.y / mtot;
	x4[i].z = s.z / mtot;

	v4[i].x = p.x / mtot;
	v4[i].y = p.y / mtot;
	v4[i].z = p.z / mtot;

	rcritv[i] = fmax(rcritv[i], rcritv[j]);
	rcritv[j] = 0.0;
	rcrit = fmax(rcrit_d[indexi], rcrit_d[indexj]);

	rcrit_d[indexi] = rcrit;
	rcrit_d[indexj] = 0.0;

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

        if(x4[i].w < x4[j].w){
                index[indexi] = index[indexj];
		aelimits[indexi] = aelimits[indexj];
		aecount[indexi] = aecount[indexj];
		enccount[indexi] = enccount[indexj];
		aecountT[indexi] = aecountT[indexj];
		enccountT[indexi] = enccountT[indexj];
        }
        if(x4[i].w == x4[j].w){
                index[indexi] = min(index[indexi], index[indexj]);
		aelimits[indexi] = aelimits[min(indexi, indexj)];
		aecount[indexi] = aecount[min(indexi, indexj)];
		enccount[indexi] = enccount[min(indexi, indexj)];
		aecountT[indexi] = aecountT[min(indexi, indexj)];
		enccountT[indexi] = enccountT[min(indexi, indexj)];
        }
        index[indexj] = -1;

        x4[i].w = mtot;
        x4[j].w = -1.0e-12;

        v4[i].w = cbrt(v4[i].w * v4[i].w * v4[i].w + v4[j].w * v4[j].w * v4[j].w);
        v4[j].w = 0.0;

}
// **************************************
// For test particles
//This function performs a merger of two bodies i and j.
//It also calculates the amount of energy which will be lost due
//to the collision: U = 0.5 * mi* mj/(mi + mj) * vij^2 - G * mi * mj / rij
//The index of the new bodie is the index of the more massiv one. If both bodies
//have an equal mass then the new index is the smaller one.
//
//Authors: Simon Grimm, Joachim Stadel
//March 2014
//
//****************************************
__device__ void collidesmall(double4 *x4, double4 *v4, int i, int j, int indexi, int indexj, int *index, int *indexsmall, int nc, double *Coll, double time, double3 *spin, double3 *spinsmall, double *rcritv){

	if(x4[i].w > 0.0 && x4[j].w > 0.0){

		double3 s, p;
		double mtot;

		s.x = x4[i].x * x4[i].w + x4[j].x * x4[j].w;
		s.y = x4[i].y * x4[i].w + x4[j].y * x4[j].w;
		s.z = x4[i].z * x4[i].w + x4[j].z * x4[j].w;

		p.x = v4[i].x * x4[i].w + v4[j].x * x4[j].w;
		p.y = v4[i].y * x4[i].w + v4[j].y * x4[j].w;
		p.z = v4[i].z * x4[i].w + v4[j].z * x4[j].w;

		mtot = x4[i].w + x4[j].w;

		x4[i].x = s.x / mtot;
		x4[i].y = s.y / mtot;
		x4[i].z = s.z / mtot;

		v4[i].x = p.x / mtot;
		v4[i].y = p.y / mtot;
		v4[i].z = p.z / mtot;

		rcritv[i] = fmax(rcritv[i], rcritv[j]);
		rcritv[j] = 0.0;

		v4[j].x = 0.0;
		v4[j].y = 0.0;
		v4[j].z = 0.0;

		x4[j].x = 0.0;
		x4[j].y = 1.0;
		x4[j].z = 0.0;

		x4[i].w = mtot;
		x4[j].w = -1.0e-12;

		v4[i].w = cbrt(v4[i].w * v4[i].w * v4[i].w + v4[j].w * v4[j].w * v4[j].w);
		v4[j].w = 0.0;

	}
	else if(x4[i].w >= 0.0 && x4[j].w >= 0.0){
		printf("Collision between body %d and Test particle %d\n", index[indexi], indexsmall[indexj]);

		Coll[nc * 25 + 0] = time/365.25;
		Coll[nc * 25 + 1] = (double)(index[indexi]);
		Coll[nc * 25 + 2] = x4[i].w;
		Coll[nc * 25 + 3] = v4[i].w;
		Coll[nc * 25 + 4] = x4[i].x;
		Coll[nc * 25 + 5] = x4[i].y;
		Coll[nc * 25 + 6] = x4[i].z;
		Coll[nc * 25 + 7] = v4[i].x;
		Coll[nc * 25 + 8] = v4[i].y;
		Coll[nc * 25 + 9] = v4[i].z;
		Coll[nc * 25 + 10] = spin[indexi].x;
		Coll[nc * 25 + 11] = spin[indexi].y;
		Coll[nc * 25 + 12] = spin[indexi].z;
		Coll[nc * 25 + 13] = (double)(indexsmall[indexj]);
		Coll[nc * 25 + 14] = x4[j].w;
		Coll[nc * 25 + 15] = v4[j].w;
		Coll[nc * 25 + 16] = x4[j].x;
		Coll[nc * 25 + 17] = x4[j].y;
		Coll[nc * 25 + 18] = x4[j].z;
		Coll[nc * 25 + 19] = v4[j].x;
		Coll[nc * 25 + 20] = v4[j].y;
		Coll[nc * 25 + 21] = v4[j].z;
		Coll[nc * 25 + 22] = spinsmall[indexj].x;
		Coll[nc * 25 + 23] = spinsmall[indexj].y;
		Coll[nc * 25 + 24] = spinsmall[indexj].z;


		x4[j].w = -1.0e-12;

		v4[j].w = 0.0;

		v4[j].x = 0.0;
		v4[j].y = 0.0;
		v4[j].z = 0.0;

		x4[j].x = 0.0;
		x4[j].y = 1.0;
		x4[j].z = 0.0;
	}
}
// **************************************
// This function stores the details of close encounters
//
// Authors: Simon Grimm
// July 2015
//
//****************************************
__device__ void storeEncounters(double4 *x4, double4 *v4, int i, int j, int indexi, int indexj, int *index, int nc, double *Coll, double time, double3 *spin){

	Coll[nc * 25 + 0] = time/365.25;
	Coll[nc * 25 + 1] = (double)(index[indexj]);
	Coll[nc * 25 + 2] = x4[j].w;
	Coll[nc * 25 + 3] = v4[j].w;
	Coll[nc * 25 + 4] = x4[j].x;
	Coll[nc * 25 + 5] = x4[j].y;
	Coll[nc * 25 + 6] = x4[j].z;
	Coll[nc * 25 + 7] = v4[j].x;
	Coll[nc * 25 + 8] = v4[j].y;
	Coll[nc * 25 + 9] = v4[j].z;
	Coll[nc * 25 + 10] = spin[indexj].x;
	Coll[nc * 25 + 11] = spin[indexj].y;
	Coll[nc * 25 + 12] = spin[indexj].z;
	Coll[nc * 25 + 13] = (double)(index[indexi]);
	Coll[nc * 25 + 14] = x4[i].w;
	Coll[nc * 25 + 15] = v4[i].w;
	Coll[nc * 25 + 16] = x4[i].x;
	Coll[nc * 25 + 17] = x4[i].y;
	Coll[nc * 25 + 18] = x4[i].z;
	Coll[nc * 25 + 19] = v4[i].x;
	Coll[nc * 25 + 20] = v4[i].y;
	Coll[nc * 25 + 21] = v4[i].z;
	Coll[nc * 25 + 22] = spin[indexi].x;
	Coll[nc * 25 + 23] = spin[indexi].y;
	Coll[nc * 25 + 24] = spin[indexi].z;
}
// **************************************
// This function stores the details of close encounters
//
// Authors: Simon Grimm
// July 2015
//
//****************************************
__device__ void storeEncounterssmall(double4 *x4, double4 *v4, int i, int j, int indexi, int indexj, int *index, int *indexsmall, int nc, double *Coll, double time, double3 *spin, double3 *spinsmall){


printf("S Enc %d %d %d %d %d %d\n", i, j, indexi, indexj, index[indexj], indexsmall[indexi]);
	Coll[nc * 25 + 0] = time/365.25;
	Coll[nc * 25 + 1] = (double)(index[indexj]);
	Coll[nc * 25 + 2] = x4[j].w;
	Coll[nc * 25 + 3] = v4[j].w;
	Coll[nc * 25 + 4] = x4[j].x;
	Coll[nc * 25 + 5] = x4[j].y;
	Coll[nc * 25 + 6] = x4[j].z;
	Coll[nc * 25 + 7] = v4[j].x;
	Coll[nc * 25 + 8] = v4[j].y;
	Coll[nc * 25 + 9] = v4[j].z;
	Coll[nc * 25 + 10] = spin[indexj].x;
	Coll[nc * 25 + 11] = spin[indexj].y;
	Coll[nc * 25 + 12] = spin[indexj].z;
	Coll[nc * 25 + 13] = (double)(indexsmall[indexi]);
	Coll[nc * 25 + 14] = x4[i].w;
	Coll[nc * 25 + 15] = v4[i].w;
	Coll[nc * 25 + 16] = x4[i].x;
	Coll[nc * 25 + 17] = x4[i].y;
	Coll[nc * 25 + 18] = x4[i].z;
	Coll[nc * 25 + 19] = v4[i].x;
	Coll[nc * 25 + 20] = v4[i].y;
	Coll[nc * 25 + 21] = v4[i].z;
	Coll[nc * 25 + 22] = spinsmall[indexi].x;
	Coll[nc * 25 + 23] = spinsmall[indexi].y;
	Coll[nc * 25 + 24] = spinsmall[indexi].z;
}
#endif

#include "define.h"

#ifndef COMENERGYCPU_H
#define COMENERGYCPU_H

//serial version 
void Data::comCall_1(const int f){

	int N = N_h[0] + Nsmall_h[0];
	double Msun = Msun_h[0].x;

	double4 p = {0.0, 0.0, 0.0, 0.0};

	for(int id = 0; id < N; ++id){
		double m = x4_h[id].w;
		if(m > 0.0){
			p.x += m * v4_h[id].x;
			p.y += m * v4_h[id].y;
			p.z += m * v4_h[id].z;
			p.w += m;
		}
	}

	double iMsun = 1.0 / Msun;

	if(f == 0){
		vcom_h[0].x = p.x;
		vcom_h[0].y = p.y;
		vcom_h[0].z = p.z;
	}

	for(int id = 0; id < N; ++id){
		double m = x4_h[id].w;
		if(m >= 0.0 && f == 1){
			//Convert to Heliocentric coordinates
			v4_h[id].x += p.x * iMsun;
			v4_h[id].y += p.y * iMsun;
			v4_h[id].z += p.z * iMsun;
		}
		if(m >= 0.0 && f == -1){
			//Convert to Democratic coordinates
			double iMsunp = 1.0 / (Msun + p.w);
			v4_h[id].x -= p.x * iMsunp;
			v4_h[id].y -= p.y * iMsunp;
			v4_h[id].z -= p.z * iMsunp;
		}
	}

}
//parallel version
void Data::comCall(const int f){

	int N = N_h[0] + Nsmall_h[0];
	double Msun = Msun_h[0].x;

	for(int k = 0; k < Nomp; ++k){
		vold_h[k].x = 0.0;
		vold_h[k].y = 0.0;
		vold_h[k].z = 0.0;
		vold_h[k].w = 0.0;
	}
	#pragma omp parallel for schedule(static)
	for(int id = 0; id < N; ++id){
		double m = x4_h[id].w;
		if(m > 0.0){
			int k = omp_get_thread_num();
			vold_h[k].x += m * v4_h[id].x;
			vold_h[k].y += m * v4_h[id].y;
			vold_h[k].z += m * v4_h[id].z;
			vold_h[k].w += m;
		}
	}
	for(int k = 1; k < Nomp; ++k){
		vold_h[0].x += vold_h[k].x;
		vold_h[0].y += vold_h[k].y;
		vold_h[0].z += vold_h[k].z;
		vold_h[0].w += vold_h[k].w;
	}

	double iMsun = 1.0 / Msun;

	if(f == 0){
		vcom_h[0].x = vold_h[0].x;
		vcom_h[0].y = vold_h[0].y;
		vcom_h[0].z = vold_h[0].z;
	}

	#pragma omp parallel for schedule(static)
	for(int id = 0; id < N; ++id){
		double m = x4_h[id].w;
		if(m >= 0.0 && f == 1){
			//Convert to Heliocentric coordinates
			v4_h[id].x += vold_h[0].x * iMsun;
			v4_h[id].y += vold_h[0].y * iMsun;
			v4_h[id].z += vold_h[0].z * iMsun;
		}
		if(m >= 0.0 && f == -1){
			//Convert to Democratic coordinates
			double iMsunp = 1.0 / (Msun + vold_h[0].w);
			v4_h[id].x -= vold_h[0].x * iMsunp;
			v4_h[id].y -= vold_h[0].y * iMsunp;
			v4_h[id].z -= vold_h[0].z * iMsunp;
		}
	}
}


#endif

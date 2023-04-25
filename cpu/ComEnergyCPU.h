#include "define.h"

#ifndef COMENERGYCPU_H
#define COMENERGYCPU_H


void Data::comCall(const int f){

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


#endif

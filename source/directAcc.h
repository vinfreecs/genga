#ifndef DIRECTACC_H
#define DIRECTACC_H

#include "Host2.h"
#include "Orbit2.h"
#endif

extern __device__ void accEnc(double4, double4, double3 &, double, double, double &, int, int);
extern __device__ void accEncSun(double4, double3 &, const double);
extern __device__ void collide(double4 *, double4 *, int, int, int, int, const double, double *, double &, int *, int, double *, double, double3 *, double *, double *, int, int, float4 *, int *, int *, long long *, long long *);
extern __device__ void collidesmall(double4 *, double4 *, int, int, int, int, int *, int *, int, double *, double, double3 *, double3 *);

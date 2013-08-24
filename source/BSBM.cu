#include "BSBM.h"

template __global__ void BSBMStep_kernel < 2, 2 > (double4 *, double4 *, double4 *, double4 *, double *, double *, int2 *, int2 *, double, double *, double *, int, int *, int *, double *, double, double3 *, int, int *, float4 *, int *, int *, long long *, long long *, int);
template __global__ void BSBMStep_kernel < 4, 4 > (double4 *, double4 *, double4 *, double4 *, double *, double *, int2 *, int2 *, double, double *, double *, int, int *, int *, double *, double, double3 *, int, int *, float4 *, int *, int *, long long *, long long *, int);
template __global__ void BSBMStep_kernel < 8, 8 > (double4 *, double4 *, double4 *, double4 *, double *, double *, int2 *, int2 *, double, double *, double *, int, int *, int *, double *, double, double3 *, int, int *, float4 *, int *, int *, long long *, long long *, int);
template __global__ void BSBMStep_kernel < 16, 16 > (double4 *, double4 *, double4 *, double4 *, double *, double *, int2 *, int2 *, double, double *, double *, int, int *, int *, double *, double, double3 *, int, int *, float4 *, int *, int *, long long *, long long *, int);


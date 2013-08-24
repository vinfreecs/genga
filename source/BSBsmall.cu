#include "BSBsmall.h"

template __global__ void BSBStepsmall_kernel < 2, 2 > (double4 *, double4 *, double4 *, double4 *, double4 *, double4 *, double *, double *, int2 *, int2 *, double, double, double *, int, int *, int *, int *, double *, double, double3 *, double3 *, const int);
template __global__ void BSBStepsmall_kernel < 4, 4 > (double4 *, double4 *, double4 *, double4 *, double4 *, double4 *, double *, double *, int2 *, int2 *, double, double, double *, int, int *, int *, int *, double *, double, double3 *, double3 *, const int);
template __global__ void BSBStepsmall_kernel < 8, 8 > (double4 *, double4 *, double4 *, double4 *, double4 *, double4 *, double *, double *, int2 *, int2 *, double, double, double *, int, int *, int *, int *, double *, double, double3 *, double3 *, const int);
template __global__ void BSBStepsmall_kernel < 16, 16 > (double4 *, double4 *, double4 *, double4 *, double4 *, double4 *, double *, double *, int2 *, int2 *, double, double, double *, int, int *, int *, int *, double *, double, double3 *, double3 *, const int); 

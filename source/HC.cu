#include "HC.h"

template __global__ void HC32_kernel <16, 32, 1> (double4 *, double4 *, const double, int *, int *, int *, int);
template __global__ void HC32_kernel <32, 64, 1> (double4 *, double4 *, const double, int *, int *, int *, int);
template __global__ void HC32_kernel <64, 64, 1> (double4 *, double4 *, const double, int *, int *, int *, int);

template __global__ void HC32_kernel <16, 32, 2> (double4 *, double4 *, const double, int *, int *, int *, int);
template __global__ void HC32_kernel <32, 64, 2> (double4 *, double4 *, const double, int *, int *, int *, int);
template __global__ void HC32_kernel <64, 64, 2> (double4 *, double4 *, const double, int *, int *, int *, int);

template __global__ void HC32_kernel <16, 32, 3> (double4 *, double4 *, const double, int *, int *, int *, int);
template __global__ void HC32_kernel <32, 64, 3> (double4 *, double4 *, const double, int *, int *, int *, int);
template __global__ void HC32_kernel <64, 64, 3> (double4 *, double4 *, const double, int *, int *, int *, int);

template __global__ void HC128_kernel < 128, 128, 1 > (double4 *, double4 *, const double, int *, int *, int *, int);
template __global__ void HC128_kernel < 256, 256, 1 > (double4 *, double4 *, const double, int *, int *, int *, int);
template __global__ void HC128_kernel < 512, 512, 1 > (double4 *, double4 *, const double, int *, int *, int *, int);
template __global__ void HC128_kernel < 512, 1024, 1 > (double4 *, double4 *, const double, int *, int *, int *, int);
template __global__ void HC128_kernel < 512, 2048, 1 > (double4 *, double4 *, const double, int *, int *, int *, int);

template __global__ void HC128_kernel < 128, 128, 2 > (double4 *, double4 *, const double, int *, int *, int *, int);
template __global__ void HC128_kernel < 256, 256, 2 > (double4 *, double4 *, const double, int *, int *, int *, int);
template __global__ void HC128_kernel < 512, 512, 2 > (double4 *, double4 *, const double, int *, int *, int *, int);
template __global__ void HC128_kernel < 512, 1024, 2 > (double4 *, double4 *, const double, int *, int *, int *, int);
template __global__ void HC128_kernel < 512, 2048, 2 > (double4 *, double4 *, const double, int *, int *, int *, int);

template __global__ void HC128_kernel < 128, 128, 3 > (double4 *, double4 *, const double, int *, int *, int *, int);
template __global__ void HC128_kernel < 256, 256, 3 > (double4 *, double4 *, const double, int *, int *, int *, int);
template __global__ void HC128_kernel < 512, 512, 3 > (double4 *, double4 *, const double, int *, int *, int *, int);
template __global__ void HC128_kernel < 512, 1024, 3 > (double4 *, double4 *, const double, int *, int *, int *, int);
template __global__ void HC128_kernel < 512, 2048, 3 > (double4 *, double4 *, const double, int *, int *, int *, int);

template __global__ void HCsmall_kernel < 256, 1> (double4 *, double4 *, const double, int *, int *, int *, int *, int *, int *, double4 *, double4 *, int, int);
template __global__ void HCsmall_kernel < 256, 2> (double4 *, double4 *, const double, int *, int *, int *, int *, int *, int *, double4 *, double4 *, int, int);
template __global__ void HCsmall_kernel < 256, 3> (double4 *, double4 *, const double, int *, int *, int *, int *, int *, int *, double4 *, double4 *, int, int);

template __global__ void HCM2_kernel < HCM_Bl, HCM_Bl2, NmaxM, 1 > (double4 *, double4 *, const double *, int *, int, double, double *, int *, int *, int *, int);
template __global__ void HCM2_kernel < HCM_Bl, HCM_Bl2, NmaxM, 2 > (double4 *, double4 *, const double *, int *, int, double, double *, int *, int *, int *, int);

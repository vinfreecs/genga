#ifndef ENERGY_H
#define ENERGY_H

#include "Host2.h"

#endif
extern __host__ void Energy(int, double4 *, double4 *, double, double *, double *, double *, double *, cudaStream_t, int, int, int);
extern __host__ void EjectionEnergyCall(int, double4 *, double4 *, double, int, double *, double3 *, int);
extern __host__ void EjectionEnergysmallCall(double4 *, int, double3 *);
extern __host__ void EjectionEnergysmall2Call(double4 *, int);


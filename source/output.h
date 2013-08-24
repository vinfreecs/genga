#include <sys/time.h>
#include <sys/resource.h>

#ifndef OUTPUT_H
#define OUTPUT_H

#include "Host2.h"
#include "Orbit2.h"
#include "Energy.h"
#include "gas.h"
#endif


extern __host__ int firstoutput();
extern __host__ void printOutput(double4 *, double4 *, int *, double *, double, long long, int, FILE *, double, double3 *, double4 *, double4 *, double3 *, int *, int, int, float4 *, float4 *, int *, int *, int *, int *, long long *, long long *, long long *, long long *, int);
extern __host__ void firstInfo();
extern __host__ void EnergyOutput(long long, double);
extern __host__ void CoordinateOutput(long long, double);
extern __host__ void GridaeOutput(long long);
extern __host__ void printfMaxColl(long long);
extern __host__ int MaxGroups(long long, double);
extern __host__ void setStartTime();
extern __host__ void printTime(long long);
extern __host__ void LastInfo();
extern __host__ void printLastTime();
extern __host__ void printCollisions();

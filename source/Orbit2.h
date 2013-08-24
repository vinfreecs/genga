#ifndef HOST_H
#define HOST_H

#include "Host2.h"
#include "Energy.h"
#endif

extern __constant__ float Gridae_c[6];
extern __constant__ int GridaeN_c[2];
extern __constant__ double S_c[FGN + 1];
extern __constant__ double C_c[FGN + 1];

extern double4 *x4_h, *x4_d;
extern double4 *v4_h, *v4_d;
extern double4 *xold_h, *xold_d;
extern double4 *vold_h, *vold_d;
extern int *index_h, *index_d;
extern double3 *spin_h, *spin_d;
extern double3 *a_d;
extern double *rcrit_h, *rcrit_d;
extern double *rcritv_d;
extern int *Nencpairs_h, *Nencpairs_d;
extern int *Nencpairs2_h, *Nencpairs2_d;
extern int2 *Encpairs_d;
extern int2 *Encpairs2_d;
extern int *Nenc_m, *Nenc_d;
extern float4 *aelimits_h, *aelimits_d;
extern int *aecount_h, *aecount_d;
extern int *enccount_h, *enccount_d;
extern long long *aecountT_h, *aecountT_d;
extern long long *enccountT_h, *enccountT_d;
extern int *Gridaecount_h, *Gridaecount_d;
extern long long *GridaecountS_h;
extern long long *GridaecountT_h;

extern double4 *x4small_h, *x4small_d;
extern double4 *v4small_h, *v4small_d;
extern double4 *xoldsmall_h, *xoldsmall_d;
extern double4 *voldsmall_h, *voldsmall_d;
extern int *indexsmall_h, *indexsmall_d;
extern double3 *spinsmall_h, *spinsmall_d;
extern double3 *asmall_d;
extern double *rcritvsmall_h, *rcritvsmall_d;
extern int *Nencpairssmall_h, *Nencpairssmall_d;
extern int *Nencpairssmall2_h, *Nencpairssmall2_d;
extern int2 *Encpairssmall_d;
extern int2 *Encpairssmall2_d;
extern int *Nencsmall_m, *Nencsmall_d;
extern double3 *vcomsmall_h, *vcomsmall_d;
extern float4 *aelimitssmall_h, *aelimitssmall_d;
extern int *aecountsmall_h, *aecountsmall_d;
extern int *enccountsmall_h, *enccountsmall_d;
extern long long *aecountsmallT_h, *aecountsmallT_d;
extern long long *enccountsmallT_h, *enccountsmallT_d;

extern double *U_h, *U_d;
extern double *Energy_h, *Energy_d;
extern double *Energy0_h, *Energy0_d;
extern int *Ncoll_m, *Ncoll_d;
extern int *EjectionFlag_d, *EjectionFlag_m;
extern double *Coll_h, *Coll_d;
extern double *test_h, *test_d;


extern __host__ void AllocateOrbitt();
extern __host__ int CMallocateOrbit();
extern __host__ int GridaeAlloc();
extern __host__ int FGAlloc();
extern __host__ int readGridae();
extern __host__ int copyGridae(long long);
extern __host__ int init();
extern __host__ int ic();
extern __host__ int readic(int);
extern __host__ void HelioToDemo(double4 *, double4 *, double, int, double4 *, double4 *, int);
extern __host__ void DemoToHelio(double4 *, double4 *, double, int, double4 *, double4 *, int);
extern __host__ int remove();
extern __host__ void resize(int &, int &, int &, int &);
extern __host__ void stopSimulations();
extern __host__ void Ejection(double);
extern __host__ void Ejectionsmall(double);
extern __host__ int freeOrbit();

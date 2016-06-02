#ifndef ORBIT_H
#define ORBIT_H

#include "define.h"
#include "Host2.h"

#if USE_NAF == 1
#include "naf2.h"
#endif


// *************************************
// Authors: Simon Grimm, Joachim Stadel
// March 2014
//
// *************************************
class Data : public Host{
public:

	long long timeStep;

	double4 *x4_h, *x4_d;
	double4 *v4_h, *v4_d;
	double4 *xold_h, *xold_d;
	double4 *vold_h, *vold_d;
	int *index_h, *index_d;
	double3 *spin_h, *spin_d;
	double3 *love_h, *love_d;
	double3 *a_d;
	double *rcrit_h, *rcrit_d;
	double *rcritv_d;
	int *groupIndex_d;
	int *Nencpairs_h, *Nencpairs_d;
	int *Nencpairs2_h, *Nencpairs2_d;
	int *groupIterate_h, *groupIterate_d;
	int2 *Encpairs_d;
	int2 *Encpairs2_d;
	bool *Encpairsb_d;
	int *Nenc_m, *Nenc_d;
	float4 *aelimits_h, *aelimits_d;
	int *aecount_h, *aecount_d;
	int *enccount_h, *enccount_d;
	long long *aecountT_h, *aecountT_d;
	long long *enccountT_h, *enccountT_d;
	int *Gridaecount_h, *Gridaecount_d;
	long long *GridaecountS_h;
	long long *GridaecountT_h;
	int *Gridaicount_h, *Gridaicount_d;
	long long *GridaicountS_h;
	long long *GridaicountT_h;

	//arrays for BSA
	double4 *xt_d;
	double4 *vt_d;
	double4 *xp_d;
	double4 *vp_d;
	double3 *dx_d;
	double3 *dv_d;
	double *dt1_d;
	double *t1_d;
	int *BSAstop_h, *BSAstop_d;

	// G3 Data
	double *K_d;
	double *Kold_d;
	double4 *StopTime_d;
	double4 *x4G3_d;
	double4 *v4G3_d;

	double3 *vcom_h, *vcom_d;

#if USE_RANDOM
	curandState *random_d;
#endif

	double *U_h, *U_d;	//internal Energy
	double *LI_h, *LI_d;	//internal Angular Momentum
	double *Energy_h, *Energy_d;
	double *Energy0_h, *Energy0_d;
	double *LI0_h, *LI0_d;
	int *Ncoll_m, *Ncoll_d;
	int *EjectionFlag_d, *EjectionFlag_m;
	int *EncFlag_d, *EncFlag_m;
	int CollisionFlag;
	double *Coll_h, *Coll_d;
	double *writeEnc_h, *writeEnc_d;
	double *Fragments_h, *Fragments_d;
	int *nFragments_m, *nFragments_d;
	int *NWriteEnc_m, *NWriteEnc_d;
	double *test_h, *test_d;

	double *coordinateBuffer_h, *coordinateBuffer_d;
	double *coordinateBufferIrr_d;
	int *timestepBuffer;
	int *timestepBufferIrr;
	int2 *NBuffer;
	int2 *NBufferIrr;

#if USE_NAF == 1
	NAF naf;
#endif

	cudaError_t error;

	__host__ Data(long long);

	__host__ void AllocateOrbitt();
	__host__ int CMallocateOrbit();
	__host__ int GridaeAlloc();
	__host__ int FGAlloc();
	__host__ int readGridae();
	__host__ int copyGridae();
	__host__ int init();
	__host__ int ic();
	__host__ void HelioToDemo(double4 *, double4 *, double, int);
	__host__ void DemoToHelio(double4 *, double4 *, double, int);
	__host__ int remove();
	__host__ void stopSimulations();
	__host__ void Ejection();
	__host__ int freeOrbit();

	//FG2
	__host__ void constantCopy();
	__host__ void constantCopy2();
	__host__ void constantCopySC(double *, double *);
	//output
	cudaStream_t *hstream;
	__host__ int firstoutput();
	__host__ void firstInfo();
	__host__ void LastInfo();
	__host__ void setStartTime();
	__host__ void printLastTime();
	__host__ void printTime();
	__host__ void CoordinateOutput(int);
	__host__ void CoordinateOutputBuffer(int);
	__host__ int MaxGroups();
	__host__ void GridaeOutput();
	__host__ void printCollisions();
	__host__ int printEncounters();
	__host__ int printFragments(int);
	__host__ int printRotation();
	__host__ int firstEnergy();
	__host__ void EnergyOutput();
	__host__ void CoordinateToBuffer(int, int);

	//Energy
	__host__ void EnergyCall(int, double4 *, double4 *, double3 *, double, double *, double *, double *, double *, double *, double *, cudaStream_t, int, int, int);
	__host__ void EjectionEnergyCall(int, double4 *, double4 *, double3 *, double, int, double *, double *, double3 *, int, int);

	//integrator
	__host__ void SymplecticP(int);
	__host__ void IrregularStep(double);
	__host__ void firstKick_16();
	__host__ void firstKick_32();
	__host__ void firstKick_64();
	__host__ void firstKick_128();
	__host__ void firstKick_256();
	__host__ void firstKick_512();
	__host__ void firstKick_1024();
	__host__ void firstKick_2048();
	__host__ void firstKick_largeN();
	__host__ void firstKick_small();
	__host__ void firstKick_M(long long);

	__host__ int step();
	__host__ int step_16();
	__host__ int step_32();
	__host__ int step_64();
	__host__ int step_128();
	__host__ int step_256();
	__host__ int step_512();
	__host__ int step_1024();
	__host__ int step_2048();
	__host__ int step_largeN();
	__host__ int step_small();
	__host__ int step_M();

	__host__ int CollisionCall();
	__host__ int CollisionMCall();
	__host__ int RemoveCall();
	__host__ int writeEncCall();
	__host__ int writeEncMCall();
	__host__ int EjectionCall();
	__host__ void EjectionMCall();
	__host__ void BSCall(int, int, double);
	__host__ void BSsmallCall(int, double);
	__host__ void BSBMCall(int);

	__host__ void BSACall(int, int, int, int, double, double);
	__host__ void BSAsmallCall(int, int, int, int, double, double);

	//gas
	__host__ void GasAlloc();
	__host__ void GasDisk(double *, double *, double *, double, int);
	__host__ int setGasDisk();
	__host__ int freeGas();
	__host__ void gasEnergyCall(int, double *, double *, double *, cudaStream_t, int, int);
	__host__ void gasEnergyMCall(int, double *, double *, double *, cudaStream_t, int, int);
	__host__ void GasAccCall_16(double *, double *, double);
	__host__ void GasAccCall_32(double *, double *, double);
	__host__ void GasAccCall_64(double *, double *, double);
	__host__ void GasAccCall_128(double *, double *, double);
	__host__ void GasAccCall_256(double *, double *, double);
	__host__ void GasAccCall_512(double *, double *, double);
	__host__ void GasAccCall_1024(double *, double *, double);
	__host__ void GasAccCall_2048(double *, double *, double);
	__host__ void GasAccCall_largeN(double *, double *, double);
	__host__ void GasAccCall_small(double *, double *, double);
	__host__ void GasAccCall_M(double *, double *, double);

#if poincareFlag == 1
	int *PFlag_h;
	int *PFlag_d;
	char poincarefilename[160];
	FILE *poincarefile;
	__host__ int PoincareSectionCall(int, double);
# endif
private:
	//Total sizes
	int GridNae;
	int GridNai;
	__host__ int readic(int);
	__host__ void resize(int &, int &, int &, int &);

	//output
	__host__ void printOutput(double4 *, double4 *, int *, double *, double, long long, int, FILE *, double, double3 *, int, int, float4 *, int *, int *, long long *, long long *, int, int);

};
#endif

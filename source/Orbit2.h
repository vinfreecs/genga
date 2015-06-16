#ifndef ORBIT_H
#define ORBIT_H

#include "define.h"
#include "Host2.h"

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
	double3 *a_d;
	double *rcrit_h, *rcrit_d;
	double *rcritv_d;
	int *groupIndex_d;
	int *groupIndexOld_d;
	int *Nencpairs_h, *Nencpairs_d;
	int *Nencpairs2_h, *Nencpairs2_d;
	int2 *Encpairs_d;
	int2 *Encpairs2_d;
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


	// G3 Data
	double *K_d;
	double *Kold_d;
	int *BddSign_d;
	double4 *StopTime_d;
	double4 *x4G3_d;
	double4 *v4G3_d;

	double4 *x4small_h, *x4small_d;
	double4 *v4small_h, *v4small_d;
	double4 *xoldsmall_h, *xoldsmall_d;
	double4 *voldsmall_h, *voldsmall_d;
	int *indexsmall_h, *indexsmall_d;
	double3 *spinsmall_h, *spinsmall_d;
	double3 *asmall_d;
	double *rcritvsmall_h, *rcritvsmall_d;
	int *groupIndexsmall_d;
	int *groupIndexsmallOld_d;
	int *Nencpairssmall_h, *Nencpairssmall_d;
	int *Nencpairssmall2_h, *Nencpairssmall2_d;
	int2 *Encpairssmall_d;
	int2 *Encpairssmall2_d;
	int *Nencsmall_m, *Nencsmall_d;
	double3 *vcomsmall_h, *vcomsmall_d;
	float4 *aelimitssmall_h, *aelimitssmall_d;
	int *aecountsmall_h, *aecountsmall_d;
	int *enccountsmall_h, *enccountsmall_d;
	long long *aecountsmallT_h, *aecountsmallT_d;
	long long *enccountsmallT_h, *enccountsmallT_d;

	double *U_h, *U_d;	//internal Energy
	double *LI_h, *LI_d;	//internal Angular Momentum
	double *Energy_h, *Energy_d;
	double *Energy0_h, *Energy0_d;
	double *LI0_h, *LI0_d;
	int *Ncoll_m, *Ncoll_d;
	int *EjectionFlag_d, *EjectionFlag_m;
	double *Coll_h, *Coll_d;
	double *test_h, *test_d;

	double *coordinateBuffer_h, *coordinateBuffer_d;
	double *coordinateBufferIrr_d;
	int *timestepBuffer;
	int *timestepBufferIrr;


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
	__host__ void HelioToDemo(double4 *, double4 *, double, int, double4 *, double4 *, int);
	__host__ void DemoToHelio(double4 *, double4 *, double, int, double4 *, double4 *, int);
	__host__ int remove();
	__host__ void stopSimulations();
	__host__ void Ejection();
	__host__ void Ejectionsmall();
	__host__ int freeOrbit();

	//FG2
	__host__ void constantCopy();
	__host__ void constantCopy2();
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
	__host__ void printMaxColl();
	__host__ void GridaeOutput();
	__host__ void printCollisions();
	__host__ int firstEnergy();
	__host__ void EnergyOutput(int);
	__host__ void CoordinateToBuffer(int, int);

	//Energy
	__host__ void EnergyCall(int, double4 *, double4 *, double3 *, double, double *, double *, double *, double *, double *, double *, cudaStream_t, int, int, int);
	__host__ void EjectionEnergyCall(int, double4 *, double4 *, double3 *, double, int, double *, double *, double3 *, int);
	__host__ void EjectionEnergysmallCall(double4 *, int, double3 *);
	__host__ void EjectionEnergysmall2Call(double4 *, int);

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
	__host__ void CollisionMCall();
	__host__ int EjectionCall();
	__host__ void EjectionMCall();
	__host__ void BSCall(int, int, double);
	__host__ void BSsmallCall(int, double);
	__host__ void BSBMCall(int);

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
	__host__ void printOutput(double4 *, double4 *, int *, double *, double, long long, int, FILE *, double, double3 *, double4 *, double4 *, double3 *, int *, int, int, float4 *, float4 *, int *, int *, int *, int *, long long *, long long *, long long *, long long *, int, int);

};
#endif

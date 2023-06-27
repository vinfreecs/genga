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
	double4 *x4_d1;				//used for multiGPUs
	double4 *x4_d2;				//used for multiGPUs
	double4 *x4_d3;				//used for multiGPUs
	double4 *v4_h, *v4_d;
	double4 *v4Helio_h;
	double4 *xold_d;
	double4 *vold_d;
	int *index_h, *index_d;
	double4 *spin_h, *spin_d;
	double3 *love_h, *love_d;
	double3 *migration_h, *migration_d;	//artificial migration force
	int *createFlag_h, *createFlag_d;
	double3 *a_d;
	double *rcrit_h, *rcrit_d;
	double *rcritv_d;
	double *rcritv_d1;			//used for multiGPUs
	double *rcritv_d2;			//used for multiGPUs
	double *rcritv_d3;			//used for multiGPUs
	int2 *groupIndex_d;
	int *Nencpairs_h, *Nencpairs_d;
	int *Nencpairs2_h, *Nencpairs2_d;
	int *Nencpairs3_h, *Nencpairs3_d;
	int *Nencpairs_d1;			//used for multiGPUs
	int *Nencpairs_d2;			//used for multiGPUs
	int *Nencpairs_d3;			//used for multiGPUs
	int *groupIterate_h, *groupIterate_d;
	int2 *Encpairs_d;
	int2 *Encpairs_d1;			//used for multiGPUs
	int2 *Encpairs_d2;			//used for multiGPUs
	int2 *Encpairs_d3;			//used for multiGPUs
	int2 *Encpairs2_d;
	int2 *Encpairs2_d1;			//used for multiGPUs
	int2 *Encpairs2_d2;			//used for multiGPUs
	int2 *Encpairs2_d3;			//used for multiGPUs
	int *Encpairs3_d;
	int2 *scan_d;
	int *Nenc_m, *Nenc_d;
	float4 *aelimits_h, *aelimits_d;
	unsigned int *aecount_h, *aecount_d;
	unsigned int *enccount_h, *enccount_d;
	unsigned long long *aecountT_h, *aecountT_d;
	unsigned long long *enccountT_h, *enccountT_d;
	unsigned int *Gridaecount_h, *Gridaecount_d;
	unsigned long long *GridaecountS_h;
	unsigned long long *GridaecountT_h;
	unsigned int *Gridaicount_h, *Gridaicount_d;
	unsigned long long *GridaicountS_h;
	unsigned long long *GridaicountT_h;

	//arrays for backup
	double4 *x4b_d;
	double4 *v4b_d;
	double4 *x4bb_d;
	double4 *v4bb_d;
	double3 *ab_d;
	int *indexb_d;
	int *indexbb_d;
	double *rcritb_d, *rcritvb_d;
	double *rcritbb_d, *rcritvbb_d;
	double4 *spinb_d;
	double4 *spinbb_d;

	//arrays for BSA
	double4 *xt_d;
	double4 *vt_d;
	double4 *xp_d;
	double4 *vp_d;
	double3 *dx_d;
	double3 *dv_d;
	double *dt1_d;
	double *t1_d;
	double *dtgr_d;
	int *BSAstop_h, *BSAstop_d;
	int *BSstop_d;
	double *Coltime_d;

	// G3 Data
	double *K_d;
	double *Kold_d;
	double4 *x4G3_d;
	double4 *v4G3_d;

	double3 *vcom_d;

	curandState *random_d;

	double *U_h, *U_d;	//internal Energy
	double *LI_h, *LI_d;	//internal Angular Momentum
	double *Energy_h, *Energy_d;
	double *Energy0_h, *Energy0_d;
	double *EnergySum_d;
	double *LI0_h, *LI0_d;
	int *Ncoll_m, *Ncoll_d;
	int *EjectionFlag_d, *EjectionFlag_m;
	int *EncFlag_d, *EncFlag_m;
	int CollisionFlag;
	int *StopFlag_d, *StopFlag_m;
	int *ErrorFlag_d, *ErrorFlag_m;
	double *Coll_h, *Coll_d;
	double *writeEnc_h, *writeEnc_d;
	double *Fragments_h, *Fragments_d;
	int *nFragments_m, *nFragments_d;
	int *NWriteEnc_m, *NWriteEnc_d;
	double *test_h, *test_d;

	//TTV
	double4 *elementsA_h, *elementsA_d;
	double4 *elementsB_h, *elementsB_d;
	double4 *elementsT_h, *elementsT_d;
	double4 *elementsSpin_h, *elementsSpin_d;
	double4 *elementsAOld_d, *elementsAOld2_d;
	double4 *elementsBOld_d, *elementsBOld2_d;
	double4 *elementsTOld_d, *elementsTOld2_d;
	double4 *elementsSpinOld_d, *elementsSpinOld2_d;
	elements10 *elementsL_h, *elementsL_d;			//tuning lenghts
	elements8 *elementsG_d;					//gradient for adadelta
	elements8 *elementsD_d;					//gradient for adadelta		
	elements8 *elementsMean_d;				//mean
	elements8 *elementsVar_d;				//variance		
	elements8 *elementsGh_d;				//hyperparameter learning rates					
	elementsS *elementsStep_d;
	elementsH *elementsHist_d;				//History for Hessian matrix

	int2 *elementsC_h, *elementsC_d;			//current count, total count
	double4 *elementsP_h, *elementsP_d;			//probability, random number, old probability, global tuning factor
	double *elementsSA_h, *elementsSA_d;			//Temperature for simulated annealing
	int4 *elementsI_h, *elementsI_d;			//indizes for proposals
	double *elementsM_h, *elementsM_d;			//Stellar mass
	double *elementsCOV_h, *elementsCOV_d;			//Covariance matrix 
	elements *Symplex_d;					//Nelder Mead downhill symplex
	int *SymplexCount_d;

	int *Ntransit_m, *Ntransit_d;		//Number of transits per time step
	int *Transit_d;				//contains indexes of transiting objects per time step
	__host__ void modifyElementsCall(int, int);
	__host__ void BSTTVCall(int);
	FILE *MCMCRestartFile;

	//TTV2
	double *timeold_d;
	double *lastTransitTime_d;
	int *transitIndex_d;
	int2 *EpochCount_d;
	double *TTV_d;

	
	//Buffer
	double *coordinateBuffer_h, *coordinateBuffer_d;
	double *coordinateBufferIrr_h, *coordinateBufferIrr_d;
	long long int *timestepBuffer;
	long long int *timestepBufferIrr;
	int2 *NBuffer;
	int2 *NBufferIrr;


	//BVH
	unsigned int *morton_d;
	unsigned int *sortRank_d;
	unsigned int *sortCount_d;
	int2 *sortIndex_d;
	Node *leafNodes_d;
	Node *internalNodes_d;

#if USE_NAF == 1
	NAF naf;
#endif

#if def_CPU == 0
	cudaError_t error;
	cudaStream_t copyStream;
	cudaStream_t BSStream[12];
	cudaStream_t hstream[16];
	cudaEvent_t KickEvent;
#endif

#if def_CPU == 1
	int error;

	double4 *xold_h;
	double4 *vold_h;
	double3 *a_h;
	double3 *b_h;
	double4 *x4b_h;
	double4 *v4b_h;
	double4 *x4bb_h;
	double4 *v4bb_h;
	double3 *ab_h;
	int *indexb_h;
	int *indexbb_h;

	double *rcritv_h;
	double *rcritb_h, *rcritvb_h;
	double *rcritbb_h, *rcritvbb_h;

	double4 *spinb_h;
	double4 *spinbb_h;

	double3 *vcom_h;
	double *EnergySum_h;
	int2 *Encpairs_h;
	int2 *Encpairs2_h;
	int *Encpairs3_h;
	int2 *scan_h;

	int2 *groupIndex_h;

	double4 *xt_h;
	double4 *vt_h;
	double4 *xp_h;
	double4 *vp_h;
	double3 *dx_h;
	double3 *dv_h;
	double *dt1_h;
	double *t1_h;
	double *dtgr_h;
	double *Coltime_h;
	int *BSstop_h;

	double *K_h;
	double *Kold_h;

	curandState *random_h;

	//BVH
	unsigned int *morton_h;
	unsigned int *sortRank_h;
	unsigned int *sortCount_h;
	int2 *sortIndex_h;
	Node *leafNodes_h;
	Node *internalNodes_h;

#endif

	__host__ Data(long long);
	__host__ int beforeTimeStepLoop1();
	__host__ int beforeTimeStepLoop(int);
	__host__ int timeStepLoop(int, int);
	__host__ int Remaining();
	__host__ int AllocateOrbit();
	__host__ int CMallocateOrbit();
	__host__ int GridaeAlloc();
	__host__ int FGAlloc();
	__host__ int readGridae();
	__host__ int copyGridae();
	__host__ int readMCMC_COV();
	__host__ int init();
	__host__ int ic();
	__host__ void KepToCart(double4 &, double4 &, double);
	__host__ void HelioToDemo(double4 *, double4 *, double, int);
	__host__ void HelioToBary(double4 *, double4 *, double, int);
	__host__ void DemoToHelio(double4 *, double4 *, double4 *, double, int);
	__host__ void BaryToHelio(double4 *, double4 *, double, int);
	__host__ int remove();
	__host__ void stopSimulations();
	__host__ void Ejection();
	__host__ int freeOrbit();

	__host__ void constantCopyDirectAcc();
	__host__ void constantCopyBS();



	//FG2
	__host__ void constantCopy();
	__host__ void constantCopy2();
	__host__ void constantCopySC(double *, double *);
	//output
	__host__ int firstoutput(int);
	__host__ void firstInfo();
	__host__ void firstInfoB();
	__host__ void LastInfo();
	__host__ void setStartTime();
	__host__ void printLastTime(int);
	__host__ int printTime(int);
	__host__ void CoordinateOutput(int);
	__host__ void CoordinateOutputBuffer(int);
	__host__ int MaxGroups();
	__host__ void GridaeOutput();
	__host__ int printCollisions();
	__host__ void printCollisionsTshift();
	__host__ int printEncounters();
	__host__ int printFragments(int);
	__host__ int printRotation();
	__host__ int printCreateparticle(int);
	__host__ int printTransits();
	__host__ int printRV();
	__host__ int printRV2();
	__host__ void printMCMC(int);
	__host__ int firstEnergy();
	__host__ int EnergyOutput(int);
	__host__ void CoordinateToBuffer(int, int, double);

	//Energy
	__host__ void EnergyCall(int, int);
	__host__ void EjectionEnergyCall(int, int);

	//integrator
	__host__ void SymplecticP(int);
	__host__ void IrregularStep(double);
	__host__ int tuneFG(int &);
	__host__ int tuneRcrit(int &);
	__host__ int tuneKick(int, int &, int &, int &);
	__host__ int tuneForce(int &);
	__host__ int tuneKickM3(int &);
	__host__ int tuneHCM3(int &);
	__host__ int tuneBVH(int &);
	__host__ int tuneBS();
	__host__ int tuneBS2();

	__host__ void firstKick_16(int);
	__host__ void firstKick_largeN(int);
	__host__ void firstKick_small(int);
	__host__ void firstKick_M(long long, int);
	__host__ void firstKick_M3(long long, int);

	__host__ void SEnc(double &, int, double, int, int);
	__host__ int bStep(int);
	__host__ int bStepM(int);
	__host__ int bStepM3(int);

	__host__ void firstStep(int);
	__host__ int step(int);
	__host__ int step_1kernel(int);
	__host__ int step_16(int);
	__host__ int step_largeN(int);
	__host__ int step_small(int);
	__host__ int step_M(int);
	__host__ int step_M3(int);
	__host__ int step_MSimple();
	__host__ int ttv_step();
	
	__host__ void comCall(const int);
	__host__ void HCCall(const double, const int);
	__host__ void groupCall();

	__host__ int CollisionCall(int);
	__host__ int CollisionMCall(int);
	__host__ int RemoveCall();
	__host__ int writeEncCall();
	__host__ int writeEncMCall();
	__host__ int EjectionCall();
	__host__ int StopAtEncounterCall();
	__host__ void BSCall(int, double, int, double);
	__host__ void BSBMCall(int, int, double);
	__host__ void BSBM3Call(int, int, double);

	__host__ void BSACall(int, int, int, int, double, double, int);

	__host__ int KickfirstndevCall(int);
	__host__ int KickndevCall(int, int);

	//gas
	__host__ void GasAlloc();
	__host__ int setGasDisk();
	__host__ int freeGas();
	__host__ void gasEnergyCall();
	__host__ void gasEnergyMCall(int);
	__host__ void GasAccCall(double *, double *, double);
	__host__ void GasAccCall_small(double *, double *, double);
	__host__ void GasAccCall2_small(double *, double *, double);
	__host__ void GasAccCall_M(double *, double *, double);


	//force
	__host__ void fragmentCall();
	__host__ void rotationCall();

	//createparticle
	__host__ int createReadFile1();
	__host__ int createReadFile2();
	__host__ int createCall();

	//BVH
	__host__ void BVHCall1();
	__host__ void BVHCall2();

#if def_CPU == 1
	void ElapsedTime(float *, timeval, timeval);
	void firstKick_cpu(int);
	void firstKick_small_cpu(int);
	void step1_cpu();
	int step_cpu(int);
	int step_small_cpu(int);
	int tuneKickCPU(int);
	void SEncCPU(double &, int, double, int, int);
	void acc4D_cpu();
	void acc4Df_cpu();
	void acc4Dsmall_cpu();
	void acc4Dfsmall_cpu();
	void acc4E_cpu();
	void acc4Ef_cpu();
	void acc4Esmall_cpu();
	void acc4Efsmall_cpu();
#endif


#if def_poincareFlag == 1
	int *PFlag_h;
	int *PFlag_d;
	char poincarefilename[160];
	FILE *poincarefile;
	__host__ int PoincareSectionCall(double);
# endif
private:
	//Total sizes
	int GridNae;
	int GridNai;
	__host__ int readic(int);
	__host__ void resize(int, int &, int);

	//output
	__host__ void printOutput(double4 *, double4 *, double4 *, int *, double *, double, long long, int, FILE *, double, double4 *, double3 *, double3 *, double *, int, int, float4 *, unsigned int *, unsigned int *, unsigned long long *, unsigned long long *, int, int);

};
#endif

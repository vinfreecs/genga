#include "Orbit2.h"

//Constructor
__host__ Data::Data(long long Restart): Host(Restart){


}

//Allocate orbit data
__host__ void Data::AllocateOrbitt(){

	//allocate memory on host//
	rcrit_h = (double*)malloc(NT*sizeof(double));
	x4_h = (double4*)malloc(NT*sizeof(double4));
	v4_h = (double4*)malloc(NT*sizeof(double4));
	xold_h = (double4*)malloc(NT*sizeof(double4));
	vold_h = (double4*)malloc(NT*sizeof(double4));
	index_h = (int*)malloc(NT*sizeof(int));
	spin_h = (double3*)malloc(NT*sizeof(double3));
	U_h = (double*)malloc(Nst*sizeof(double));
	LI_h = (double*)malloc(Nst*sizeof(double));
	Energy_h = (double*)malloc(NEnergyT*sizeof(double));
	Energy0_h = (double*)malloc(Nst*sizeof(double));
	LI0_h = (double*)malloc(Nst*sizeof(double));
	Coll_h = (double*)malloc(25*MaxColl*Nst*sizeof(double));
	aelimits_h = (float4*)malloc(NT*sizeof(float4));
	aecount_h = (int*)malloc(NT*sizeof(int));
	enccount_h = (int*)malloc(NT*sizeof(int));
	aecountT_h = (long long*)malloc(NT*sizeof(long long));
	enccountT_h = (long long*)malloc(NT*sizeof(long long));

	x4small_h = (double4*)malloc(NsmallT*sizeof(double4));
	v4small_h = (double4*)malloc(NsmallT*sizeof(double4));
	xoldsmall_h = (double4*)malloc(NsmallT*sizeof(double4));
	voldsmall_h = (double4*)malloc(NsmallT*sizeof(double4));
	indexsmall_h = (int*)malloc(NsmallT*sizeof(int));
	spinsmall_h = (double3*)malloc(NsmallT*sizeof(double3));
	rcritvsmall_h = (double*)malloc(NsmallT*sizeof(double));
	vcomsmall_h = (double3*)malloc(Nst*sizeof(double3));
	aelimitssmall_h = (float4*)malloc(NsmallT*sizeof(float4));
	aecountsmall_h = (int*)malloc(NsmallT*sizeof(int));
	enccountsmall_h = (int*)malloc(NsmallT*sizeof(int));
	aecountsmallT_h = (long long*)malloc(NsmallT*sizeof(long long));
	enccountsmallT_h = (long long*)malloc(NsmallT*sizeof(long long));

	cudaHostAlloc((void **)&test_h, NT*sizeof(double), cudaHostAllocDefault);
#if poincareFlag == 1
	PFlag_h = (int*)malloc(sizeof(int));
	PFlag_h[0] = 0;
#endif
	//allocate pinned memory on host//
	cudaHostAlloc((void **)&Nencpairs_h, (Nst + 1)*sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void **)&Nencpairs2_h, (Nst + 1)*sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void **)&Nencpairssmall_h, (Nst + 1)*sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void **)&Nencpairssmall2_h, (Nst + 1)*sizeof(int), cudaHostAllocDefault);

	//allocate memory on device//
	cudaMalloc((void **) &x4_d,NT*sizeof(double4));
	cudaMalloc((void **) &v4_d,NT*sizeof(double4));
	cudaMalloc((void **) &xold_d,NT*sizeof(double4));
	cudaMalloc((void **) &vold_d,NT*sizeof(double4));
	cudaMalloc((void **) &rcrit_d,NT*sizeof(double));
	cudaMalloc((void **) &rcritv_d,NT*sizeof(double));
	cudaMalloc((void **) &test_d,NT*sizeof(double));
	cudaMalloc((void **) &index_d,NT*sizeof(int));
	cudaMalloc((void **) &spin_d,NT*sizeof(double3));
	cudaMalloc((void **) &U_d,Nst*sizeof(double));
	cudaMalloc((void **) &LI_d,Nst*sizeof(double));
	cudaMalloc((void **) &a_d, NT*sizeof(double3));
	cudaMalloc((void **) &Energy_d, NEnergyT*sizeof(double));
	cudaMalloc((void **) &Energy0_d, Nst*sizeof(double));
	cudaMalloc((void **) &LI0_d, Nst*sizeof(double));
	cudaMalloc((void **) &Nenc_d, 12*sizeof(int));
	cudaMalloc((void **) &Ncoll_d, sizeof(int));
	cudaMalloc((void **) &Nencpairs_d, (Nst + 1)*sizeof(int));
	cudaMalloc((void **) &Nencpairs2_d, (Nst + 1)*sizeof(int));
	cudaMalloc((void **) &Encpairs_d, sizeof(int2)*NB2T);
	cudaMalloc((void **) &Encpairs2_d, sizeof(int2)*NB2T);
	cudaMalloc((void **) &Coll_d, sizeof(double)*Nst*25*MaxColl);
	cudaMalloc((void **) &aelimits_d,NT*sizeof(float4));
	cudaMalloc((void **) &aecount_d,NT*sizeof(int));
	cudaMalloc((void **) &enccount_d,NT*sizeof(int));
	cudaMalloc((void **) &aecountT_d,NT*sizeof(long long));
	cudaMalloc((void **) &enccountT_d,NT*sizeof(long long));

#if G3 == 1
	cudaMalloc((void **) &K_d, NT * NT * sizeof(double));
	cudaMalloc((void **) &Kold_d, NT * NT * sizeof(double));
	cudaMalloc((void **) &BddSign_d, NT * NT * sizeof(int));
	cudaMalloc((void **) &groupIndex_d,NT*sizeof(int));
	cudaMalloc((void **) &groupIndexOld_d,NT*sizeof(int));
	cudaMalloc((void **) &groupIndexsmall_d,NsmallT*sizeof(int));
	cudaMalloc((void **) &groupIndexsmallOld_d,NsmallT*sizeof(int));
	cudaMalloc((void **) &StopTime_d, NT * NT * sizeof(double4));
	cudaMalloc((void **) &x4G3_d, NT * sizeof(double4));
	cudaMalloc((void **) &v4G3_d, NT * sizeof(double4));
#else
	K_d = NULL;
	Kold_d = NULL;
	BddSign_d = NULL;
	groupIndex_d = NULL;
	groupIndexOld_d = NULL;
	groupIndexsmall_d = NULL;
	groupIndexsmallOld_d = NULL;
	StopTime_d = NULL;
	x4G3_d = NULL;
	v4G3_d = NULL;
	
#endif

	cudaMalloc((void **) &x4small_d,NsmallT*sizeof(double4));
	cudaMalloc((void **) &v4small_d,NsmallT*sizeof(double4));
	cudaMalloc((void **) &xoldsmall_d,NsmallT*sizeof(double4));
	cudaMalloc((void **) &voldsmall_d,NsmallT*sizeof(double4));
	cudaMalloc((void **) &indexsmall_d,NsmallT*sizeof(int));
	cudaMalloc((void **) &spinsmall_d,NsmallT*sizeof(double3));
	cudaMalloc((void **) &rcritvsmall_d,NsmallT*sizeof(double));
	cudaMalloc((void **) &asmall_d, NsmallT*sizeof(double3));
	cudaMalloc((void **) &Nencsmall_d, 12*sizeof(int));
	cudaMalloc((void **) &Nencpairssmall_d, (Nst + 1)*sizeof(int));
	cudaMalloc((void **) &Nencpairssmall2_d, (Nst + 1)*sizeof(int));
	cudaMalloc((void **) &Encpairssmall_d, sizeof(int2)*Nsmall2T);
	cudaMalloc((void **) &Encpairssmall2_d, sizeof(int2)*Nsmall2T);
	cudaMalloc((void **) &vcomsmall_d, Nst*sizeof(double3));
	cudaMalloc((void **) &aelimitssmall_d,NsmallT*sizeof(float4));
	cudaMalloc((void **) &aecountsmall_d,NsmallT*sizeof(int));
	cudaMalloc((void **) &enccountsmall_d,NsmallT*sizeof(int));
	cudaMalloc((void **) &aecountsmallT_d,NsmallT*sizeof(long long));
	cudaMalloc((void **) &enccountsmallT_d,NsmallT*sizeof(long long));

#if poincareFlag == 1
	cudaMalloc((void **) &PFlag_d,sizeof(int));
	cudaMemcpy(PFlag_d, PFlag_h, sizeof(int), cudaMemcpyHostToDevice);
#endif
};


//This function allocates mapped memory
__host__ int Data::CMallocateOrbit(){

	cudaError_t error;
	error = cudaGetLastError();
	fprintf(masterfile,"CudaMalloc error = %d = %s\n",error, cudaGetErrorString(error));
	if(error != 0) return 0;

	cudaHostAlloc((void **)&Nenc_m, 12*sizeof(int), cudaHostAllocMapped);
	error = cudaGetLastError();
	fprintf(masterfile,"CudaHostAlloc error = %d = %s\n",error, cudaGetErrorString(error));
	if(error != 0) return 0;

	cudaHostGetDevicePointer((void **)&Nenc_d, (void *)Nenc_m, 0);
	error = cudaGetLastError();
	fprintf(masterfile,"mapping error = %d = %s\n",error, cudaGetErrorString(error));
	if(error != 0) return 0;

	cudaHostAlloc((void **)&Ncoll_m, sizeof(int), cudaHostAllocMapped);

	cudaHostGetDevicePointer((void **)&Ncoll_d, (void *)Ncoll_m, 0);

	cudaHostAlloc((void **)&EjectionFlag_m, (Nst + 1)*sizeof(int), cudaHostAllocMapped);
	cudaHostGetDevicePointer((void **)&EjectionFlag_d, (void *)EjectionFlag_m, 0);

	cudaHostAlloc((void **)&Nencsmall_m, 12*sizeof(int), cudaHostAllocMapped);
	cudaHostGetDevicePointer((void **)&Nencsmall_d, (void *)Nencsmall_m, 0);

	error = cudaGetLastError();
	fprintf(masterfile,"mapping error = %d = %s\n",error, cudaGetErrorString(error));
	if(error != 0) return 0;

	return 1;

}


//This function allocates the Grida and set values to zero
__host__ int Data::GridaeAlloc(){
	cudaError_t error;
	GridN = Gridae.Na * Gridae.Ne;
	cudaMalloc((void **) &Gridaecount_d, GridN * sizeof(int));
	Gridaecount_h = (int*)malloc(GridN * sizeof(int));
	GridaecountT_h = (long long*)malloc(GridN * sizeof(long long));
	GridaecountS_h = (long long*)malloc(GridN * sizeof(long long));

	for(int i = 0; i < GridN; ++i){
	      Gridaecount_h[i] = 0;
	      GridaecountT_h[i] = 0;
	      GridaecountS_h[i] = 0;
	}
	cudaMemcpy(Gridaecount_d, Gridaecount_h, sizeof(int)*GridN, cudaMemcpyHostToDevice);

	float GridaeP[6] = {Gridae.amin, Gridae.amax, Gridae.emin, Gridae.emax, Gridae.deltaa, Gridae.deltae};
	int GridaeN[2] = {Gridae.Na, Gridae.Ne};
	cudaMemcpyToSymbol(Gridae_c, GridaeP, 6*sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(GridaeN_c, GridaeN, 2*sizeof(int), 0, cudaMemcpyHostToDevice);

	error = cudaGetLastError();
	fprintf(masterfile,"GrieaeAlloc  error = %d = %s\n",error, cudaGetErrorString(error));
	if(error != 0) return 0;

	return 1;
}


__host__ int Data::FGAlloc(){
	cudaError_t error;
        double S_h[FGN + 1];
        double C_h[FGN + 1];

        //Table for fastfg//
        for (int j = 0; j<= FGN; ++j) {
                double dEj = j*PI_N;
                S_h[j] = sin(dEj);
                C_h[j] = cos(dEj);
        }
        cudaMemcpyToSymbol(S_c, S_h, sizeof(S_h), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(C_c, C_h, sizeof(C_h), 0, cudaMemcpyHostToDevice);
	error = cudaGetLastError();
	fprintf(masterfile,"FGAlloc  error = %d = %s\n",error, cudaGetErrorString(error));
	if(error != 0){
		printf("FGAlloc  error = %d = %s\n",error, cudaGetErrorString(error));
		return 0;
	}
	return 1;
}


//This function reads at a restart the corrspondent Gridae file
__host__ int Data::readGridae(){
	if(P.tRestart > 0){
		sprintf(Gridae.filename, "aeCount%s_%.12lld.dat", Gridae.X, P.tRestart);
		Gridae.file = fopen(Gridae.filename, "r");
		if(Gridae.file == NULL){
			fprintf(masterfile, "Error: aeGrid file not found: aeCount%s_%.12lld.dat\n", Gridae.X, P.tRestart);
			printf("Error: aeGrid file not found: aeCount%s_%.12lld.dat\n", Gridae.X, P.tRestart);
			return 0;
		}
		for(int i = 0; i < Gridae.Ne; ++i){
			for(int j = 0; j < Gridae.Na; ++j){
				fscanf(Gridae.file, "%lld",&GridaecountT_h[i * Gridae.Na + j]);
			}
		}
		fclose(Gridae.file);
	}
	return 1;
}

//This function copies values from the current Gridae to the total and summing host Grid
__host__ int Data::copyGridae(long long ts){
	cudaError_t error;

	cudaMemcpy(Gridaecount_h, Gridaecount_d, sizeof(int) * GridN, cudaMemcpyDeviceToHost);
	for(int i = 0; i < Gridae.Ne; ++i){
		for(int j = 0; j < Gridae.Na; ++j){
			if(ts > Gridae.Start){
				GridaecountS_h[i * Gridae.Na + j] += Gridaecount_h[i * Gridae.Na + j];
				GridaecountT_h[i * Gridae.Na + j] += Gridaecount_h[i * Gridae.Na + j];
			}
		}
	}
	cudaMemset(Gridaecount_d, 0, sizeof(int)*GridN);
        error = cudaGetLastError();
        fprintf(masterfile,"Grieae copy error = %d = %s\n",error, cudaGetErrorString(error));
        if(error != 0) return 0;

        return 1;
}


//This function initialized the data
__host__ int Data::init(){

	Ncoll_m[0] = 0;
	for(int i = 0; i < 12; ++i){
		Nenc_m[i] = 0;
		Nencsmall_m[i] = 0;
	}
	EjectionFlag_m[0] = 0;
	for(int st = 0; st < Nst; ++st){
		EjectionFlag_m[st + 1] = 0;
		for(int i = 0; i < N_h[st]; ++i){
			index_h[NBS_h[st] + i] = i+st*100;
		}
		for(int i = 0; i < Nsmall_h[st]; ++i){
			indexsmall_h[NsmallS_h[st] + i] = i + N_h[st] + st*100;
		}
	}
	for(int i = 0; i < NT; ++i){
		rcrit_h[i] = 0.0;
		x4_h[i].x = 1.0;
		x4_h[i].y = 0.0;
		x4_h[i].z = 0.0; 
		v4_h[i].x = 0.0;
		v4_h[i].y = 0.0;
		v4_h[i].z = 0.0;
		x4_h[i].w = -1.0e-12;
		v4_h[i].w = 0.0;
		xold_h[i].x = x4_h[i].x;
		xold_h[i].y = x4_h[i].y;
		xold_h[i].z = x4_h[i].z;
		xold_h[i].w = -1.0e-12;
		vold_h[i].x = v4_h[i].x;
		vold_h[i].y = v4_h[i].y;
		vold_h[i].z = v4_h[i].z;
		vold_h[i].w = 0.0;
		test_h[i] = -1.0;
		spin_h[i].x = 0.0;
		spin_h[i].y = 0.0;
		spin_h[i].z = 0.0;
		aelimits_h[i].x = 0.0f;
		aelimits_h[i].y = (float)(Rcut);
		aelimits_h[i].z = 0.0f;
		aelimits_h[i].w = 1.0f;
		aecount_h[i] = 0;
		enccount_h[i] = 0;
		aecountT_h[i] = 0;
		enccountT_h[i] = 0;
	}
	for(int i = 0; i < NEnergyT; ++i){
		Energy_h[i] = 0.0;
	}

	for(int i = 0; i < NsmallT; ++i){
		rcritvsmall_h[i] = 0.0;
		x4small_h[i].x = 1.0;
		x4small_h[i].y = 0.0;
		x4small_h[i].z = 0.0;
		v4small_h[i].x = 0.0;
		v4small_h[i].y = 0.0;
		v4small_h[i].z = 0.0;
		x4small_h[i].w = -1.0e-12;
		v4small_h[i].w = 0.0;
		xoldsmall_h[i].x = x4small_h[i].x;
		xoldsmall_h[i].y = x4small_h[i].y;
		xoldsmall_h[i].z = x4small_h[i].z;
		xoldsmall_h[i].w = -1.0e-12;
		voldsmall_h[i].x = v4small_h[i].x;
		voldsmall_h[i].y = v4small_h[i].y;
		voldsmall_h[i].z = v4small_h[i].z;
		voldsmall_h[i].w = 0.0;
		spinsmall_h[i].x = 0.0;
		spinsmall_h[i].y = 0.0;
		spinsmall_h[i].z = 0.0;
		aelimitssmall_h[i].x = 0.0f;
		aelimitssmall_h[i].y = (float)(Rcut);
		aelimitssmall_h[i].z = 0.0f;
		aelimitssmall_h[i].w = 1.0f;
		aecountsmall_h[i] = 0;
		enccountsmall_h[i] = 0;
		aecountsmallT_h[i] = 0;
		enccountsmallT_h[i] = 0;
	}

	for(int i = 0; i < Nst * 25 * MaxColl; ++i){
		Coll_h[i] = 0.0;
	}

	for(int st = 0; st < Nst + 1; ++st){    
		Nencpairs_h[st] = 0;
		Nencpairs2_h[st] = 0;
		Nencpairssmall_h[st] = 0;
		Nencpairssmall2_h[st] = 0;
	}
	for(int st = 0; st < Nst; ++st){
		U_h[st] = 0.0;
		LI_h[st] = 0.0;
		Energy0_h[st] = 1.0;
		LI0_h[st] = 1.0;
	}


	return 1;
}


//This function calls the readic function and copies the data to the GPU.
__host__ int Data::ic(){

	for(int st = 0; st < Nst; ++st){
		if(N_h[st] + Nsmall_h[st] > 0){
			GSF[st].logfile = fopen(GSF[st].logfilename, "a");
			int NBS = NBS_h[st];
			int NsmallS = NsmallS_h[st];
			fprintf(GSF[st].logfile, "\n************* Read initial conditions ****************\n \n");
			int icerr = 0;
			icerr = readic(st);
			if(icerr == 0){
				fprintf(GSF[st].logfile, "Error: Could not read initial conditions\n");
				fprintf(masterfile, "Error in Simulation %s\n", GSF[st].path);
				return 0;
			}
			fclose(GSF[st].logfile);
			HelioToDemo(x4_h + NBS, v4_h + NBS, Msun_h[st], N_h[st], x4small_h + NsmallS, v4small_h + NsmallS, Nsmall_h[st]);
		}
	}
	//Copy memory to device//
	cudaMemcpy(x4_d, x4_h, sizeof(double4)*NT, cudaMemcpyHostToDevice);
	cudaMemcpy(v4_d, v4_h, sizeof(double4)*NT, cudaMemcpyHostToDevice);
	cudaMemcpy(xold_d, x4_h, sizeof(double4)*NT, cudaMemcpyHostToDevice);
	cudaMemcpy(vold_d, v4_h, sizeof(double4)*NT, cudaMemcpyHostToDevice);
	cudaMemcpy(rcrit_d, rcrit_h, sizeof(double)*NT, cudaMemcpyHostToDevice);
	cudaMemcpy(rcritv_d, rcrit_h, sizeof(double)*NT, cudaMemcpyHostToDevice);
	cudaMemcpy(U_d, U_h, Nst*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(LI_d, LI_h, Nst*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Energy0_d, Energy0_h, Nst*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(LI0_d, LI0_h, Nst*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Energy_d, Energy_h, sizeof(double)*NEnergyT, cudaMemcpyHostToDevice);
	cudaMemcpy(test_d, test_h, sizeof(double)*NT, cudaMemcpyHostToDevice);
	cudaMemcpy(index_d, index_h, sizeof(int)*NT, cudaMemcpyHostToDevice);
	cudaMemcpy(spin_d, spin_h, sizeof(double3)*NT, cudaMemcpyHostToDevice);
	cudaMemcpy(N_d, N_h, Nst*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Nencpairs_d, Nencpairs_h, (Nst + 1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Nencpairs2_d, Nencpairs2_h, (Nst + 1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Coll_d, Coll_h, sizeof(double)*Nst*25*MaxColl, cudaMemcpyHostToDevice);
	cudaMemcpy(aelimits_d, aelimits_h, sizeof(float4)*NT, cudaMemcpyHostToDevice);
	cudaMemcpy(aecount_d, aecount_h, sizeof(int)*NT, cudaMemcpyHostToDevice);
	cudaMemcpy(enccount_d, enccount_h, sizeof(int)*NT, cudaMemcpyHostToDevice);
	cudaMemcpy(aecountT_d, aecountT_h, sizeof(long long)*NT, cudaMemcpyHostToDevice);
	cudaMemcpy(enccountT_d, enccountT_h, sizeof(long long)*NT, cudaMemcpyHostToDevice);

	cudaMemcpy(x4small_d, x4small_h, sizeof(double4)*NsmallT, cudaMemcpyHostToDevice);
	cudaMemcpy(v4small_d, v4small_h, sizeof(double4)*NsmallT, cudaMemcpyHostToDevice);
	cudaMemcpy(xoldsmall_d, xoldsmall_h, sizeof(double4)*NsmallT, cudaMemcpyHostToDevice);
	cudaMemcpy(voldsmall_d, voldsmall_h, sizeof(double4)*NsmallT, cudaMemcpyHostToDevice);
	cudaMemcpy(spinsmall_d, spinsmall_h, sizeof(double3)*NsmallT, cudaMemcpyHostToDevice);
	cudaMemcpy(indexsmall_d, indexsmall_h, sizeof(int)*NsmallT, cudaMemcpyHostToDevice);
	cudaMemcpy(rcritvsmall_d, rcritvsmall_h, sizeof(double)*NsmallT, cudaMemcpyHostToDevice);
	cudaMemcpy(Nsmall_d, Nsmall_h, Nst*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Nencpairssmall_d, Nencpairssmall_h, (Nst + 1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Nencpairssmall2_d, Nencpairssmall2_h, (Nst + 1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(aelimitssmall_d, aelimitssmall_h, sizeof(float4)*NsmallT, cudaMemcpyHostToDevice);
	cudaMemcpy(aecountsmall_d, aecountsmall_h, sizeof(int)*NsmallT, cudaMemcpyHostToDevice);
	cudaMemcpy(enccountsmall_d, enccountsmall_h, sizeof(int)*NsmallT, cudaMemcpyHostToDevice);
	cudaMemcpy(aecountsmallT_d, aecountsmallT_h, sizeof(long long)*NsmallT, cudaMemcpyHostToDevice);
	cudaMemcpy(enccountsmallT_d, enccountsmallT_h, sizeof(long long)*NsmallT, cudaMemcpyHostToDevice);

	cudaError_t error;

	cudaMemcpy(NBS_d, NBS_h, Nst*sizeof(int), cudaMemcpyHostToDevice);
	error = cudaGetLastError();
	fprintf(masterfile,"cudaMemcopy error = %d = %s\n",error, cudaGetErrorString(error));
	if(error != 0) return 0;

	return 1;
}

// ************************************** //
//This function reads the initial conditions from the IC file.
//Authors: Simon Grimm, Joachim Stadel
//March 2014
// *****************************************
__host__ int Data::readic(int st){

	int N = N_h[st];
	int Nsmall = Nsmall_h[st];
	int NsmallS = NsmallS_h[st];
	int NBS = NBS_h[st];

	FILE *infile;	
	double ttest;

	double AU = 1.49597870700e13; //in cm
	double Solarmass = 1.98892e33; //in g

	if(FormatP == 1 || P.tRestart == 0) infile = fopen(GSF[st].inputfilename, "r");

	int ii = 0;
	int iismall = 0;
	
	double skip;
	double4 x, v;
	double3 spin;
	int index;
	float4 aelimits;

	if(P.tRestart == 0){
		for(int i = 0; i < N + Nsmall; ++i){
			if(i < N){
				x = x4_h[i + NBS];
				v = v4_h[i + NBS];
				spin = spin_h[i + NBS];
				index = index_h[i + NBS];
				aelimits = aelimits_h[i + NBS];
			}
			else{
				x = x4small_h[i-N + NsmallS];
				v = v4small_h[i-N + NsmallS];
				spin = spinsmall_h[i-N + NsmallS];
				index = indexsmall_h[i-N + NsmallS];
				aelimits = aelimitssmall_h[i-N + NsmallS];
			}

			for(int f = 0; f < 22; ++f){
				if(GSF[st].informat[f] == 1) fscanf (infile, "%lf",&x.x);
				else if (GSF[st].informat[f] == 2) fscanf (infile, "%lf",&x.y);
				else if (GSF[st].informat[f] == 3) fscanf (infile, "%lf",&x.z);
				else if (GSF[st].informat[f] == 4) fscanf (infile, "%lf",&x.w);
				else if (GSF[st].informat[f] == 5) fscanf (infile, "%lf",&v.x);
				else if (GSF[st].informat[f] == 6) fscanf (infile, "%lf",&v.y);
				else if (GSF[st].informat[f] == 7) fscanf (infile, "%lf",&v.z);
				else if (GSF[st].informat[f] == 8) fscanf (infile, "%lf",&v.w);
				else if (GSF[st].informat[f] == 9) fscanf (infile, "%lf",&rho[st]);
				else if (GSF[st].informat[f] == 10) fscanf (infile, "%lf",&spin.x);
				else if (GSF[st].informat[f] == 11) fscanf (infile, "%lf",&spin.y);
				else if (GSF[st].informat[f] == 12) fscanf (infile, "%lf",&spin.z);
				else if (GSF[st].informat[f] == 13) fscanf (infile, "%d",&index);
				else if (GSF[st].informat[f] == 14) fscanf (infile, "%lf",&skip);
				else if (GSF[st].informat[f] == 15) fscanf (infile, "%f",&aelimits.x);
				else if (GSF[st].informat[f] == 16) fscanf (infile, "%f",&aelimits.y);
				else if (GSF[st].informat[f] == 17) fscanf (infile, "%f",&aelimits.z);
				else if (GSF[st].informat[f] == 18) fscanf (infile, "%f",&aelimits.w);
				else if (GSF[st].informat[f] == 19){
					if(P.ict == 0) fscanf (infile, "%lf",&P.ict);
					else fscanf (infile, "%lf",&skip);
					}
			}

			if(v.w == 0){
				v.w = cbrt((x.w * 0.75 ) / (M_PI * rho[st] * AU * AU * AU / Solarmass));
			}
			if(x.w > 0.0 || P.UseTestParticles == 0){
				x4_h[ii + NBS] = x;
				v4_h[ii + NBS] = v;
				spin_h[ii + NBS] = spin;
				if(Nst == 1) index_h[ii + NBS] = index;
				else index_h[ii + NBS] = index % 100 + 100*st;
				aelimits_h[ii + NBS] = aelimits;
				++ii;
			}
			else{
				x4small_h[iismall + NsmallS] = x;
				v4small_h[iismall + NsmallS] = v;
				spinsmall_h[iismall + NsmallS] = spin;
				if(Nst == 1) indexsmall_h[iismall + NsmallS] = index;
				else indexsmall_h[iismall + NsmallS] = index % 100 + 100*st;
				aelimitssmall_h[iismall + NsmallS] = aelimits;
				++iismall;
			}
		}
	}
	else{
	//read from restart time step
		char Ets[160]; //exact time at restart time step, must be the same format as the coordinate output
		sprintf(Ets, "%.16g", (P.tRestart * P.idt) / 365.25 + P.ict);
		double Et = atof(Ets);
		double time = 0.0;
		double aecount = 0.0;

		if(FormatP == 1){
			//skip previous time steps
			if(FormatT == 0) fscanf (infile, "%lf",&time);
			if(FormatT == 1){
				fscanf (infile, "%lf",&time);
				while(time < Et){
					if(time == Et) break;
					for(int j = 0; j < 20; ++j){
						fscanf (infile, "%lf",&skip);
					}
					fscanf (infile, "%lf",&time);
				}
			}

			//skip previous simulation data
			if(FormatS == 1){
				for(int i = 0; i < NBS * 21; ++i){
					fscanf (infile, "%lf",&skip);
				}
			}

			for(int i = 0; i < N; ++i){
				if(i > 0) fscanf (infile, "%lf",&time);
				fscanf (infile, "%d",&index_h[i + NBS]);
				fscanf (infile, "%lf",&x4_h[i + NBS].w);
				fscanf (infile, "%lf",&v4_h[i + NBS].w);
				fscanf (infile, "%lf",&x4_h[i + NBS].x);
				fscanf (infile, "%lf",&x4_h[i + NBS].y);
				fscanf (infile, "%lf",&x4_h[i + NBS].z);
				fscanf (infile, "%lf",&v4_h[i + NBS].x);
				fscanf (infile, "%lf",&v4_h[i + NBS].y);
				fscanf (infile, "%lf",&v4_h[i + NBS].z);
				fscanf (infile, "%lf",&spin_h[i + NBS].x);
				fscanf (infile, "%lf",&spin_h[i + NBS].y);
				fscanf (infile, "%lf",&spin_h[i + NBS].z);
				fscanf (infile, "%f",&aelimits_h[i + NBS].x);
				fscanf (infile, "%f",&aelimits_h[i + NBS].y);
				fscanf (infile, "%f",&aelimits_h[i + NBS].z);
				fscanf (infile, "%f",&aelimits_h[i + NBS].w);
				fscanf (infile, "%lf",&skip);
				fscanf (infile, "%lf",&aecount);
				fscanf (infile, "%lld",&enccountT_h[i + NBS]);
				fscanf (infile, "%lf",&ttest);

				if(FormatS == 0) index_h[i + NBS] += 100*st;
				aecountT_h[i + NBS] = (long long)(aecount * P.tRestart);

				++ii;
			}
			for(int i = 0; i < Nsmall; ++i){
				fscanf (infile, "%lf",&skip);
				fscanf (infile, "%d",&indexsmall_h[i + NsmallS]);
				fscanf (infile, "%lf",&x4small_h[i + NsmallS].w);
				fscanf (infile, "%lf",&v4small_h[i + NsmallS].w);
				fscanf (infile, "%lf",&x4small_h[i + NsmallS].x);
				fscanf (infile, "%lf",&x4small_h[i + NsmallS].y);
				fscanf (infile, "%lf",&x4small_h[i + NsmallS].z);
				fscanf (infile, "%lf",&v4small_h[i + NsmallS].x);
				fscanf (infile, "%lf",&v4small_h[i + NsmallS].y);
				fscanf (infile, "%lf",&v4small_h[i + NsmallS].z);
				fscanf (infile, "%lf",&spinsmall_h[i + NsmallS].x);
				fscanf (infile, "%lf",&spinsmall_h[i + NsmallS].y);
				fscanf (infile, "%lf",&spinsmall_h[i + NsmallS].z);
				fscanf (infile, "%f",&aelimitssmall_h[i + NsmallS].x);
				fscanf (infile, "%f",&aelimitssmall_h[i + NsmallS].y);
				fscanf (infile, "%f",&aelimitssmall_h[i + NsmallS].z);
				fscanf (infile, "%f",&aelimitssmall_h[i + NsmallS].w);
				fscanf (infile, "%lf",&skip);
				fscanf (infile, "%lf",&aecount);
				fscanf (infile, "%lld",&enccountsmallT_h[i + NsmallS]);
				fscanf (infile, "%lf",&ttest);

				indexsmall_h[i + NsmallS] += 100*st;
				aecountsmallT_h[i + NsmallS] = (long long)(aecount * P.tRestart);

				++ii;
			}
		}
		if(FormatP == 0){
			ii = 0;
			int iismall = 0;
		
			OrigInfile = fopen(OrigInfilename, "r");
			for(int k = 0; k < 1000000000; ++k){
				int i;
				double skip = 0.0;
				int eri = 1;
				for(int f = 0; f < 22; ++f){
					if(GSF[st].informat[f] == 13){
						eri = fscanf (OrigInfile, "%d",&i);
					}
					else if(GSF[st].informat[f] > 0){
						eri = fscanf (OrigInfile, "%lf",&skip);
					}
				}
				if(eri < 0) break;
		
				int er = 0;
				char infilename[160];
				sprintf(infilename, "%sOut%s_p%.6d.dat", GSF[st].path, GSF[st].X, i);
				infile = fopen(infilename, "r");
				if(infile == NULL) continue;
	
				//skip previous time steps
				fscanf (infile, "%lf",&time);
				while(time < Et){
					if(time == Et) break;
					for(int j = 0; j < 20; ++j){
						fscanf (infile, "%lf",&skip);
					}
					er = fscanf (infile, "%lf",&time);
					if(er <= 0) break;
				}
				if(er <= 0) continue;

				int index;
				double m;
				fscanf (infile, "%d",&index);
				fscanf (infile, "%lf",&m);
				if(m > 0 || P.UseTestParticles == 0){
					index_h[ii + NBS] = index;	
					x4_h[ii + NBS].w = m;
					fscanf (infile, "%lf",&v4_h[ii + NBS].w);
					fscanf (infile, "%lf",&x4_h[ii + NBS].x);
					fscanf (infile, "%lf",&x4_h[ii + NBS].y);
					fscanf (infile, "%lf",&x4_h[ii + NBS].z);
					fscanf (infile, "%lf",&v4_h[ii + NBS].x);
					fscanf (infile, "%lf",&v4_h[ii + NBS].y);
					fscanf (infile, "%lf",&v4_h[ii + NBS].z);
					fscanf (infile, "%lf",&spin_h[ii + NBS].x);
					fscanf (infile, "%lf",&spin_h[ii + NBS].y);
					fscanf (infile, "%lf",&spin_h[ii + NBS].z);
					fscanf (infile, "%f",&aelimits_h[ii + NBS].x);
					fscanf (infile, "%f",&aelimits_h[ii + NBS].y);
					fscanf (infile, "%f",&aelimits_h[ii + NBS].z);
					fscanf (infile, "%f",&aelimits_h[ii + NBS].w);
					fscanf (infile, "%lf",&skip);
					fscanf (infile, "%lf",&aecount);
					fscanf (infile, "%lld",&enccountT_h[ii + NBS]);
					fscanf (infile, "%lf",&ttest);

					if(FormatS == 0) index_h[ii + NBS] += 100*st;
					aecountT_h[ii + NBS] = (long long)(aecount * P.tRestart);

					++ii;
				}
				else{
					indexsmall_h[iismall + NsmallS] = index;
					x4small_h[iismall + NsmallS].w = m;
					fscanf (infile, "%lf",&v4small_h[iismall + NsmallS].w);
					fscanf (infile, "%lf",&x4small_h[iismall + NsmallS].x);
					fscanf (infile, "%lf",&x4small_h[iismall + NsmallS].y);
					fscanf (infile, "%lf",&x4small_h[iismall + NsmallS].z);
					fscanf (infile, "%lf",&v4small_h[iismall + NsmallS].x);
					fscanf (infile, "%lf",&v4small_h[iismall + NsmallS].y);
					fscanf (infile, "%lf",&v4small_h[iismall + NsmallS].z);
					fscanf (infile, "%lf",&spinsmall_h[iismall + NsmallS].x);
					fscanf (infile, "%lf",&spinsmall_h[iismall + NsmallS].y);
					fscanf (infile, "%lf",&spinsmall_h[iismall + NsmallS].z);
					fscanf (infile, "%f",&aelimitssmall_h[iismall + NsmallS].x);
					fscanf (infile, "%f",&aelimitssmall_h[iismall + NsmallS].y);
					fscanf (infile, "%f",&aelimitssmall_h[iismall + NsmallS].z);
					fscanf (infile, "%f",&aelimitssmall_h[iismall + NsmallS].w);
					fscanf (infile, "%lf",&skip);
					fscanf (infile, "%lf",&aecount);
					fscanf (infile, "%lld",&enccountsmallT_h[iismall + NsmallS]);
					fscanf (infile, "%lf",&ttest);

					indexsmall_h[iismall + NsmallS] += 100*st;
					aecountsmallT_h[iismall + NsmallS] = (long long)(aecount * P.tRestart);

					++iismall;
				}
				fclose(infile);
				if(ii + iismall == N + Nsmall) break;
			}
			fclose(OrigInfile);
		}
	}
	if(FormatP == 1 || P.tRestart == 0) fclose(infile);
	return ii + iismall;
} 


// **************************************
//This function converts heliocentric coordinares to democratic coordinates.
__host__ void Data::HelioToDemo(double4 *x4_h, double4 *v4_h, double Msun, int N, double4 *x4small_h, double4 *v4small_h, int Nsmall){

	double mtot = 0.0;
	double3 vcom;
	vcom.x = 0.0;
	vcom.y = 0.0;
	vcom.z = 0.0;
	
	for(int i = 0; i < N; ++i){
		if(x4_h[i].w > 0.0){
			mtot += x4_h[i].w;
			vcom.x += x4_h[i].w * v4_h[i].x;
			vcom.y += x4_h[i].w * v4_h[i].y;
			vcom.z += x4_h[i].w * v4_h[i].z;
		}
	}
	for(int i = 0; i < Nsmall; ++i){
		if(x4small_h[i].w > 0.0){
			mtot += x4small_h[i].w;
			vcom.x += x4small_h[i].w * v4small_h[i].x;
			vcom.y += x4small_h[i].w * v4small_h[i].y;
			vcom.z += x4small_h[i].w * v4small_h[i].z;
		}
	}
	mtot += Msun;
	vcom.x /= mtot;
	vcom.y /= mtot;
	vcom.z /= mtot;

	for(int i = 0; i < N; ++i){
		v4_h[i].x -= vcom.x;
		v4_h[i].y -= vcom.y;
		v4_h[i].z -= vcom.z;
	}
	for(int i = 0; i < Nsmall; ++i){
		v4small_h[i].x -= vcom.x;
		v4small_h[i].y -= vcom.y;
		v4small_h[i].z -= vcom.z;
	}

}
// **************************************
//This function converts democratic coordinares to heliocentric coordinates.
__host__ void Data::DemoToHelio(double4 *x4_h, double4 *v4_h, double Msun, int N, double4 *x4small_h, double4 *v4small_h, int Nsmall){

	double3 vcom;
	vcom.x = 0.0;
	vcom.y = 0.0;
	vcom.z = 0.0;

	for(int i = 0; i < N; ++i){
		if(x4_h[i].w > 0.0){
			vcom.x += x4_h[i].w * v4_h[i].x;
			vcom.y += x4_h[i].w * v4_h[i].y;
			vcom.z += x4_h[i].w * v4_h[i].z;
		}
	}
	for(int i = 0; i < Nsmall; ++i){
		if(x4small_h[i].w > 0.0){
			vcom.x += x4small_h[i].w * v4small_h[i].x;
			vcom.y += x4small_h[i].w * v4small_h[i].y;
			vcom.z += x4small_h[i].w * v4small_h[i].z;
		}
	}
	vcom.x /= Msun;
	vcom.y /= Msun;
	vcom.z /= Msun;

	for(int i = 0; i < N; ++i){
		v4_h[i].x += vcom.x;
		v4_h[i].y += vcom.y;
		v4_h[i].z += vcom.z;
	}
	for(int i = 0; i < Nsmall; ++i){
		v4small_h[i].x += vcom.x;
		v4small_h[i].y += vcom.y;
		v4small_h[i].z += vcom.z;
	}

}


// **************************************
//This kernel removes ghost-masses and decreases the number of bodies.
//It also removes bodies wich a semi major axis bigger than Rcut.
//It runs with only one thread ond the GPU, to avoid unnecesary data copies
//Authors: Simon Grimm, Joachim Stadel
//March 2014
// ***************************************
__global__ void remove_kernel(double4 *x4_d, double4 *v4_d, double3 *a_d, int *N_d, int *index_d, double3 *spin_d, double *Energy_d, double *test_d, double *rcrit_d, double *rcritv_d, int NBS, int st, float4 *aelimits_d, int *aecount_d, int *enccount_d, long long *aecountT_d, long long *enccountT_d, double *K_d, double *Kold_d, int *BddSign_d, double4 *StopTime_d, int NB){

	int NOld;
	int N = N_d[st];
	int f = 1;
	int fc = 0;

	while(f == 1 || fc < 100){
		NOld = N;
		f = 0;
		++fc;
		for(int j = 0; j < N; ++j){
			//remove ghost bodies and rearrange arrays
			if(x4_d[j + NBS].w < 0){
				x4_d[j + NBS] = x4_d[N-1 + NBS];
				v4_d[j + NBS] = v4_d[N-1 + NBS];

				x4_d[N-1 + NBS].x = Rcut;
				x4_d[N-1 + NBS].y = Rcut;
				x4_d[N-1 + NBS].z = Rcut;
				x4_d[N-1 + NBS].w = -1.0e-12;
	
				v4_d[N-1 + NBS].x = 0.0;
				v4_d[N-1 + NBS].y = 0.0;
				v4_d[N-1 + NBS].z = 0.0;
				v4_d[N-1 + NBS].w = 0.0;

				a_d[j + NBS] = a_d[N-1 + NBS];
				a_d[N-1 + NBS].x = 0.0;
				a_d[N-1 + NBS].y = 0.0;
				a_d[N-1 + NBS].z = 0.0;

				index_d[j + NBS] = index_d[N-1 + NBS];
				index_d[N-1 + NBS] = -1;

				spin_d[j + NBS] = spin_d[N-1 + NBS];
				spin_d[N-1 + NBS].x = 0.0;
				spin_d[N-1 + NBS].y = 0.0;
				spin_d[N-1 + NBS].z = 0.0;

				rcrit_d[j + NBS] = rcrit_d[N-1 + NBS];
				rcritv_d[j + NBS] = rcritv_d[N-1 + NBS];
				rcrit_d[N-1 + NBS] = 0.0;
				rcritv_d[N-1 + NBS] = 0.0;

				aelimits_d[j + NBS] = aelimits_d[N-1 + NBS];
				aelimits_d[N-1 + NBS].x = 0.0f;
				aelimits_d[N-1 + NBS].y = 0.0f;
				aelimits_d[N-1 + NBS].z = 0.0f;	
				aelimits_d[N-1 + NBS].w = 0.0f;

				aecount_d[j + NBS] = aecount_d[N-1 + NBS];
				aecount_d[N-1 + NBS] = 0;
				enccount_d[j + NBS] = enccount_d[N-1 + NBS];
				enccount_d[N-1 + NBS] = 0;
				aecountT_d[j + NBS] = aecountT_d[N-1 + NBS];
				aecountT_d[N-1 + NBS] = 0;
				enccountT_d[j + NBS] = enccountT_d[N-1 + NBS];
				enccountT_d[N-1 + NBS] = 0;

				Energy_d[j + NBS] += Energy_d[N-1 + NBS];
				Energy_d[N-1 + NBS] = 0.0;

				test_d[j + NBS] = test_d[N-1 + NBS];
				test_d[N-1 + NBS] = -1.0;
#if G3 == 1
				for(int i = 0; i < N; ++i){
					K_d[(j + NBS) * NB + i] = K_d[(N-1 + NBS) * NB + i];
					K_d[i * NB + j + NBS] = K_d[i * NB + (N-1 + NBS)];
					K_d[(N-1 + NBS) * NB + i] = 1.0;
					K_d[i * NB + (N-1 + NBS)] = 1.0;
					Kold_d[(j + NBS) * NB + i] = Kold_d[(N-1 + NBS) * NB + i];
					Kold_d[i * NB + j + NBS] = Kold_d[i * NB + (N-1 + NBS)];
					Kold_d[(N-1 + NBS) * NB + i] = 1.0;
					Kold_d[i * NB + (N-1 + NBS)] = 1.0;
					BddSign_d[(j + NBS) * NB + i] = BddSign_d[(N-1 + NBS) * NB + i];
					BddSign_d[i * NB + j + NBS] = BddSign_d[i * NB + (N-1 + NBS)];
					BddSign_d[(N-1 + NBS) * NB + i] = 1.0;
					BddSign_d[i * NB + (N-1 + NBS)] = 1.0;
					StopTime_d[(j + NBS) * NB + i] = StopTime_d[(N-1 + NBS) * NB + i];
					StopTime_d[i * NB + j + NBS] = StopTime_d[i * NB + (N-1 + NBS)];
					StopTime_d[(N-1 + NBS) * NB + i].x = -1.0;
					StopTime_d[i * NB + (N-1 + NBS)].x = -1.0;
					StopTime_d[(N-1 + NBS) * NB + i].y = -1.0;
					StopTime_d[i * NB + (N-1 + NBS)].y = -1.0;
					StopTime_d[(N-1 + NBS) * NB + i].z = -1.0;
					StopTime_d[i * NB + (N-1 + NBS)].z = -1.0;
					StopTime_d[(N-1 + NBS) * NB + i].w = -1.0;
					StopTime_d[i * NB + (N-1 + NBS)].w = -1.0;
				}
#endif

				N -= 1;
			}
		}
		if(NOld != N) f = 1;
	}
	N_d[st] = N;
}


// **************************************
//This kernel removes ghost-masses from the test particles and decreases the number of bodies.
//It also removes bodies wich a semi major axis bigger than Rcut.
//It runs with only one thread ond the GPU, to avoid unnecesary data copies
// Authors: Simon Grimm, Joachim Stadel
//March 2014
// ****************************************3
__global__ void removesmall_kernel(double4 *x4small_d, double4 *v4small_d, double3 *asmall_d, int *Nsmall_d, int *indexsmall_d, double3 *spinsmall_d, int NsmallS, int st, float4 *aelimitssmall_d, int *aecountsmall_d, int *enccountsmall_d, long long *aecountsmallT_d, long long *enccountsmallT_d){

	int NOldsmall;
	volatile int Nsmall= Nsmall_d[st];
	int f = 1;
	int fc = 0;

	while(f == 1 || fc < 100){
		NOldsmall = Nsmall;
		f = 0;
		++fc;
		for(int j = 0; j < Nsmall; ++j){
			//remove ghost bodies and rearrange arrays
			if(x4small_d[j + NsmallS].w < 0){
				x4small_d[j + NsmallS] = x4small_d[Nsmall-1 + NsmallS];
				v4small_d[j + NsmallS] = v4small_d[Nsmall-1 + NsmallS];

				x4small_d[Nsmall-1 + NsmallS].x = Rcut;
				x4small_d[Nsmall-1 + NsmallS].y = Rcut;
				x4small_d[Nsmall-1 + NsmallS].z = Rcut;
				x4small_d[Nsmall-1 + NsmallS].w = -1.0e-12;
	
				v4small_d[Nsmall-1 + NsmallS].x = 0.0;
				v4small_d[Nsmall-1 + NsmallS].y = 0.0;
				v4small_d[Nsmall-1 + NsmallS].z = 0.0;
				v4small_d[Nsmall-1 + NsmallS].w = 0.0;

				asmall_d[j + NsmallS] = asmall_d[Nsmall-1 + NsmallS];
	
				asmall_d[Nsmall-1 + NsmallS].x = 0.0;
				asmall_d[Nsmall-1 + NsmallS].y = 0.0;
				asmall_d[Nsmall-1 + NsmallS].z = 0.0;
				
				indexsmall_d[j + NsmallS] = indexsmall_d[Nsmall-1 + NsmallS];
				indexsmall_d[Nsmall-1 + NsmallS] = -1;

				spinsmall_d[j + NsmallS] = spinsmall_d[Nsmall-1 + NsmallS];
				spinsmall_d[Nsmall-1 + NsmallS].x = 0.0;
				spinsmall_d[Nsmall-1 + NsmallS].y = 0.0;
				spinsmall_d[Nsmall-1 + NsmallS].z = 0.0;

				aelimitssmall_d[j + NsmallS] = aelimitssmall_d[Nsmall-1 + NsmallS];
				aelimitssmall_d[Nsmall-1 + NsmallS].x = 0.0f;
				aelimitssmall_d[Nsmall-1 + NsmallS].y = 0.0f;
				aelimitssmall_d[Nsmall-1 + NsmallS].z = 0.0f;
				aelimitssmall_d[Nsmall-1 + NsmallS].w = 0.0f;

				aecountsmall_d[j + NsmallS] = aecountsmall_d[Nsmall-1 + NsmallS];
				aecountsmall_d[Nsmall-1 + NsmallS] = 0;
				enccountsmall_d[j + NsmallS] = enccountsmall_d[Nsmall-1 + NsmallS];
				enccountsmall_d[Nsmall-1 + NsmallS] = 0;
				aecountsmallT_d[j + NsmallS] = aecountsmallT_d[Nsmall-1 + NsmallS];
				aecountsmallT_d[Nsmall-1 + NsmallS] = 0;
				enccountsmallT_d[j + NsmallS] = enccountsmallT_d[Nsmall-1 + NsmallS];
				enccountsmallT_d[Nsmall-1 + NsmallS] = 0;

				Nsmall -= 1;
			}
		}
		if(NOldsmall != Nsmall) f = 1;
	}
	Nsmall_d[st] = Nsmall;
}



// **************************************
//This function prints out data of ejected bodies
//It sets the masses of ejecte bodies to zero, this are then later removed
//It Updates the lost Energy term U
//
//Authors: Simon Grimm, Joachim Stadel
//March 2014
//****************************************
__host__ void Data::Ejection(double time){

	FILE *ejectfile;
	FILE *logfile;

	if(Nst == 1) EjectionFlag_m[1] = 1;

	for(int st = 0; st < Nst; ++st){
		if(EjectionFlag_m[st + 1] > 0){
			int NBS = NBS_h[st];
			int NsmallS = NsmallS_h[st];

			ejectfile = fopen(GSF[st].ejectfilename, "a");
			logfile = fopen(GSF[st].logfilename, "a");

			cudaMemcpy(x4_h + NBS, x4_d + NBS, sizeof(double4)*N_h[st], cudaMemcpyDeviceToHost);
			cudaMemcpy(v4_h + NBS, v4_d + NBS, sizeof(double4)*N_h[st], cudaMemcpyDeviceToHost);
			cudaMemcpy(index_h + NBS, index_d + NBS, sizeof(int)*N_h[st], cudaMemcpyDeviceToHost);
			cudaMemcpy(spin_h + NBS, spin_d + NBS, sizeof(double3)*N_h[st], cudaMemcpyDeviceToHost);

			int c = 0;
			for(int i = 0; i < N_h[st]; ++i){
				vcomsmall_h[st].x = 0.0;
				vcomsmall_h[st].y = 0.0;
				vcomsmall_h[st].z = 0.0;
				cudaMemcpy(vcomsmall_d + st, vcomsmall_h + st, sizeof(double3), cudaMemcpyHostToDevice);
				c = 0;
				double rsq = x4_h[i + NBS].x*x4_h[i + NBS].x + x4_h[i + NBS].y*x4_h[i + NBS].y + x4_h[i + NBS].z*x4_h[i + NBS].z;
				if(rsq > Rcut * Rcut && x4_h[i + NBS].w >= 0){
					c = -3;
					if(Nst == 1){
						printf("Body %d ejected\n", index_h[i + NBS]);
						fprintf(logfile, "Body %d ejected\n", index_h[i + NBS]);
					}
					else{
						printf("In Simulation %s: Body %d ejected \n", GSF[st].path, index_h[i + NBS] % 100);
						fprintf(logfile, "Body %d ejected\n", index_h[i + NBS] % 100);
					}
				}
				if( rsq < RcutSun * RcutSun && x4_h[i + NBS].w >= 0){
					c = -2;
					if(Nst == 1){
						printf("Body %d too close to central mass -> removed\n", index_h[i + NBS]);
						fprintf(logfile, "Body %d too close to central mass -> removed\n", index_h[i + NBS]);
					}
					else{
						printf("In Simulation %s: Body %d too close to central mass -> removed\n", GSF[st].path, index_h[i + NBS] % 100);
						fprintf(logfile, "Body %d too close to central mass -> removed\n", index_h[i + NBS] % 100);
					}
				}
				if(c < 0){
					if(Nst == 1) fprintf(ejectfile, "%g %d %g %g %g %g %g %g %g %g %g %g %g %d\n", time/365.25, index_h[i + NBS], x4_h[i + NBS].w, v4_h[i + NBS].w, x4_h[i + NBS].x, x4_h[i + NBS].y, x4_h[i + NBS].z, v4_h[i + NBS].x, v4_h[i + NBS].y, v4_h[i + NBS].z, spin_h[i + NBS].x, spin_h[i + NBS].y, spin_h[i + NBS].z, c);
					else fprintf(ejectfile, "%g %d %g %g %g %g %g %g %g %g %g %g %g %d\n", time/365.25, index_h[i + NBS] % 100, x4_h[i + NBS].w, v4_h[i + NBS].w, x4_h[i + NBS].x, x4_h[i + NBS].y, x4_h[i + NBS].z, v4_h[i + NBS].x, v4_h[i + NBS].y, v4_h[i + NBS].z, spin_h[i + NBS].x, spin_h[i + NBS].y, spin_h[i + NBS].z, c);
					
					EjectionEnergyCall(NB[st], x4_d + NBS , v4_d + NBS, spin_d + NBS, Msun_h[st], i, U_d + st, LI_d + st, vcomsmall_d + st, N_h[st]);
					
					if(Nsmall_h[st] > 0) EjectionEnergysmallCall(v4small_d + NsmallS, Nsmall_h[st], vcomsmall_d + st);
				}
			}
			fclose(ejectfile);
			fclose(logfile);
		}
	}
}

__host__ void Data::Ejectionsmall(double time){

	FILE *ejectfile;
	FILE *logfile;

	if(Nst == 1) EjectionFlag_m[1] = 1;

	for(int st = 0; st < Nst; ++st){
		if(EjectionFlag_m[st + 1] > 0){
			int NsmallS = NsmallS_h[st];
			cudaMemcpy(x4small_h + NsmallS, x4small_d + NsmallS, sizeof(double4)*Nsmall_h[st], cudaMemcpyDeviceToHost);
			cudaMemcpy(v4small_h + NsmallS, v4small_d + NsmallS, sizeof(double4)*Nsmall_h[st], cudaMemcpyDeviceToHost);
			cudaMemcpy(indexsmall_h + NsmallS, indexsmall_d + NsmallS, sizeof(int)*Nsmall_h[st], cudaMemcpyDeviceToHost);
			cudaMemcpy(spinsmall_h + NsmallS, spinsmall_d + NsmallS, sizeof(double3)*Nsmall_h[st], cudaMemcpyDeviceToHost);

			ejectfile = fopen(GSF[st].ejectfilename, "a");
			logfile = fopen(GSF[st].logfilename, "a");
			int c = 0;
			for(int i = 0; i < Nsmall_h[st]; ++i){
				c = 0;
				double rsq = x4small_h[i + NsmallS].x*x4small_h[i + NsmallS].x + x4small_h[i + NsmallS].y*x4small_h[i + NsmallS].y + x4small_h[i + NsmallS].z*x4small_h[i + NsmallS].z;
				if(rsq > Rcut * Rcut && x4small_h[i + NsmallS].w >= 0){
					c = -3;
					if(Nst == 1){
						printf("Test particle %d ejected\n", indexsmall_h[i + NsmallS]);
						fprintf(logfile, "Test particle %d ejected\n", indexsmall_h[i + NsmallS]);
					}
					else{
						printf("In Simulation %s: Test particle %d ejected\n", GSF[st].path, indexsmall_h[i + NsmallS] % 100);
						fprintf(logfile, "Test particle %d ejected\n", indexsmall_h[i + NsmallS] % 100);
					}
				}
				if( rsq < RcutSun * RcutSun && x4small_h[i + NsmallS].w >= 0){
					c = -2;
					if(Nst == 1){
						printf("Test particle %d too close to central mass -> removed\n", indexsmall_h[i + NsmallS]);
						fprintf(logfile, "Test particle %d too close to central mass -> removed\n", indexsmall_h[i + NsmallS]);
					}
					else{
						printf("In Simulation %s: Test particle %d too close to central mass -> removed\n", GSF[st].path, indexsmall_h[i + NsmallS] % 100);
						fprintf(logfile, "In Simulation %s: Test particle %d too close to central mass -> removed\n", GSF[st].path, indexsmall_h[i + NsmallS] % 100);
					}
				}
				if(c < 0){
					if( Nst == 1) fprintf(ejectfile, "%g %d %g %g %g %g %g %g %g %g %g %g %g %d\n", time/365.25, indexsmall_h[i + NsmallS], x4small_h[i + NsmallS].w, v4small_h[i + NsmallS].w, x4small_h[i + NsmallS].x, x4small_h[i + NsmallS].y, x4small_h[i + NsmallS].z, v4small_h[i + NsmallS].x, v4small_h[i + NsmallS].y, v4small_h[i + NsmallS].z, spinsmall_h[i + NsmallS].x, spinsmall_h[i + NsmallS].y, spinsmall_h[i + NsmallS].z, c);
					else fprintf(ejectfile, "%g %d %g %g %g %g %g %g %g %g %g %g %g %d\n", time/365.25, indexsmall_h[i + NsmallS] % 100, x4small_h[i + NsmallS].w, v4small_h[i + NsmallS].w, x4small_h[i + NsmallS].x, x4small_h[i + NsmallS].y, x4small_h[i + NsmallS].z, v4small_h[i + NsmallS].x, v4small_h[i + NsmallS].y, v4small_h[i + NsmallS].z, spinsmall_h[i + NsmallS].x, spinsmall_h[i + NsmallS].y, spinsmall_h[i + NsmallS].z, c);
					EjectionEnergysmall2Call(x4small_d + NsmallS, i);
				}

			}
			fclose(ejectfile);
			fclose(logfile);
			EjectionFlag_m[st + 1] = 0;
		}
	}
}


//This function removes ghost particles and reorders the arrays
//It returns 1 if a simulations has less than the minimal number of bodies, otherwise zero
__host__ int Data::remove(){

	int NminFlag = 0;
	for(int st = 0; st < Nst; ++st){

		remove_kernel <<<1, 1>>> (x4_d, v4_d, a_d, N_d, index_d, spin_d, Energy_d, test_d, rcrit_d, rcritv_d, NBS_h[st], st, aelimits_d, aecount_d, enccount_d, aecountT_d, enccountT_d, K_d, Kold_d, BddSign_d, StopTime_d, NB[st]);
		removesmall_kernel <<< 1, 1>>> (x4small_d, v4small_d, asmall_d, Nsmall_d, indexsmall_d, spinsmall_d, NsmallS_h[st], st, aelimitssmall_d, aecountsmall_d, enccountsmall_d, aecountsmallT_d, enccountsmallT_d);
		cudaMemcpy(N_h + st, N_d + st, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(Nsmall_h + st, Nsmall_d + st, sizeof(int), cudaMemcpyDeviceToHost);
		resize(N_h[st], NB[st], N4[st], N2[st]);

		if(Nst == 1 && N_h[0] < Nmin[0]){
			printf("Number of bodies smaller than Nmin, simulation stopped\n");
			fprintf(masterfile,"Number of bodies smaller than Nmin, simulation stopped\n");
			GSF[0].logfile = fopen(GSF[0].logfilename, "a");
			fprintf(GSF[0].logfile,"Number of bodies smaller than Nmin, simulation stopped\n");
			fclose(GSF[0].logfile);
			NminFlag = 1;
			N_h[0] = 0;
		}
		if(Nst > 1 && N_h[st] < Nmin[st]){
			NminFlag = 1;
			N_h[st] = 0;
		}

	}
	
	cudaMemcpy(x4_h, x4_d, sizeof(double)*4*NT, cudaMemcpyDeviceToHost);
	cudaMemcpy(v4_h, v4_d, sizeof(double)*4*NT, cudaMemcpyDeviceToHost);
	cudaMemcpy(index_h, index_d, sizeof(int)*NT, cudaMemcpyDeviceToHost);
	cudaMemcpy(spin_h, spin_d, sizeof(double)*3*NT, cudaMemcpyDeviceToHost);
	return NminFlag;
}



// **************************************
//This function recomputes the value of NB, which is the next bigger 
//number to N which is a power of two.
__host__ void Data::resize(int &N, int &NB, int &N4, int &N2){

	NB = 16;
	if( N > 16) NB = 32;
	if( N > 32) NB = 64;
	if( N > 64) NB = 128;
	if( N > 128) NB = 256;
	if( N > 256) NB = 512;
	if( N > 512) NB = 1024;
	if( N > 1024) NB = 2048;
	if( N > 2048) NB = 4096;
	if( N > 4096) NB = 8192;

	N4 = N;
	if(N4 %4 == 3) N4 +=1;
	if(N4 %4 == 2) N4 +=2;
	if(N4 %4 == 1) N4 +=3;
	N4 /= 4;

	N2 = N;
	if(N2 %2 == 1) N2 +=1;
	N2 /=2;
}


//This function rearranges the memory if a simulations is stopped
//It runs with only one thread ond the GPU, to avoid unnecesary data copies
__global__ void removeM_kernel(double4 *x4_d, double4 *v4_d, double3 *spin_d, double *test_d, int *index_d, double *rcrit_d, double *rcritv_d, double4 *x4small_d, double4 *v4small_d, double3 *spinsmall_d, int *indexsmall_d, int st, int NBS, int NsmallS, int *N_d, int *Nsmall_d, int NT, int NsmallT, float4 *aelimits_d, float4 *aelimitssmall_d, int *aecount_d, int *aecountsmall_d, int *enccount_d, int *enccountsmall_d, long long *aecountT_d, long long *aecountsmallT_d, long long *enccountT_d, long long *enccountsmallT_d){

	for(int j = 0; j < N_d[st]; ++j){
		x4_d[j + NT] = x4_d[j + NBS];
		v4_d[j + NT] = v4_d[j + NBS];
		spin_d[j + NT] = spin_d[j + NBS];
		test_d[j + NT] = test_d[j + NBS];
		index_d[j + NT] = index_d[j + NBS];
		rcrit_d[j + NT] = rcrit_d[j + NBS];
		rcritv_d[j + NT] = rcritv_d[j + NBS];
		aelimits_d[j + NT] = aelimits_d[j + NBS];
		enccount_d[j + NT] = enccount_d[j + NBS];
		aecount_d[j + NT] = aecount_d[j + NBS];
		aecountT_d[j + NT] = aecountT_d[j + NBS];
		enccountT_d[j + NT] = enccountT_d[j + NBS];
	}

	for(int j = 0; j < Nsmall_d[st]; ++j){
		x4small_d[j + NsmallT] = x4small_d[j + NsmallS];
		v4small_d[j + NsmallT] = v4small_d[j + NsmallS];
		spinsmall_d[j + NsmallT] = spinsmall_d[j + NsmallS];
		indexsmall_d[j + NsmallT] = indexsmall_d[j + NsmallS];
		aelimitssmall_d[j + NsmallT] = aelimitssmall_d[j + NsmallS];
		aecountsmall_d[j + NsmallT] = aecountsmall_d[j + NsmallS];
		enccountsmall_d[j + NsmallT] = enccountsmall_d[j + NsmallS];
		aecountsmallT_d[j + NsmallT] = aecountsmallT_d[j + NsmallS];
		enccountsmallT_d[j + NsmallT] = enccountsmallT_d[j + NsmallS];
	}
}


//this kernel rearranges the simulations index
__global__ void remove3M_kernel(int *index_d, int *N_d, int *NBS_d){

	int idy = threadIdx.x;
	int st = blockIdx.x;

	int N = N_d[st];
	int NBS = NBS_d[st];

	if(idy < N){

		int index = index_d[idy + NBS] % 100;
		index_d[idy + NBS] = index + st * 100;
	}
}



//This function stopps simulations with less than the minimal number of bodies, and rearanges the memory
__host__ void Data::stopSimulations(){
	NT = 0;
	NsmallT = 0;
	NB2T = 0;
	Nsmall2T = 0;
	NEnergyT = 0;

	for(int st = 0; st < Nst; ++st){
		//rearange arrays//
		removeM_kernel <<< 1, 1>>>(x4_d, v4_d, spin_d, test_d, index_d, rcrit_d, rcritv_d, x4small_d, v4small_d, spinsmall_d, indexsmall_d, st, NBS_h[st], NsmallS_h[st], N_d, Nsmall_d, NT, NsmallT, aelimits_d, aelimitssmall_d, aecount_d, aecountsmall_d, enccount_d, enccountsmall_d, aecountT_d, aecountsmallT_d, enccountT_d, enccountsmallT_d);

		NBS_h[st] = NT;
		NsmallS_h[st] = NsmallT;
		NB2S[st] = NB2T;
		Nsmall2S[st] = Nsmall2T;
		NEnergy[st] = NEnergyT;
		NT += N_h[st];
		NsmallT += Nsmall_h[st];
		NB2T += NB[st] * NmaxTestParticles;
		Nsmall2T += Nsmall_h[st] * NmaxTestParticles;
		NEnergyT += max(N_h[st], 8);
	}

	cudaMemcpy(U_h, U_d, Nst*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(LI_h, LI_d, Nst*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(Energy0_h, Energy0_d, Nst*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(LI0_h, LI0_d, Nst*sizeof(double), cudaMemcpyDeviceToHost);

	for(int st = 0; st < Nst; ++st){
		if(N_h[st] < Nmin[st]){
			printf("In Simulation %s: Number of bodies smaller than Nmin, simulation stopped\n", GSF[st].path);
			fprintf(masterfile,"In Simulation %s: Number of bodies smaller than Nmin, simulation stopped\n", GSF[st].path);
			GSF[st].logfile = fopen(GSF[st].logfilename, "a");
			fprintf(GSF[st].logfile,"Number of bodies smaller than Nmin, simulation stopped\n");
			fclose(GSF[st].logfile);

			for(int sst = st; sst < Nst - 1; ++sst){
				GSF[sst] = GSF[sst + 1];

				Nsmall_h[sst] = Nsmall_h[sst + 1];
				N_h[sst] = N_h[sst + 1];
				NB[sst] = NB[sst + 1];
				N4[sst] = N4[sst + 1];
				N2[sst] = N2[sst + 1];
				Msun_h[sst] = Msun_h[sst + 1];
				n1_h[sst] = n1_h[sst + 1];
				n2_h[sst] = n2_h[sst + 1];
				rho[sst] = rho[sst + 1];
				Nmin[sst] = Nmin[sst + 1];
				dtiMsun_h[sst] = dtiMsun_h[sst + 1];
				Nconst[sst] = Nconst[sst + 1];

				U_h[sst] = U_h[sst + 1];
				LI_h[sst] = LI_h[sst + 1];
				Energy0_h[sst] = Energy0_h[sst + 1];
				LI0_h[sst] = LI0_h[sst + 1];

				NBS_h[sst] = NBS_h[sst + 1];
				NsmallS_h[sst] = NsmallS_h[sst + 1];
				NB2S[sst] = NB2S[sst + 1];
				Nsmall2S[sst] = Nsmall2S[sst + 1];
				NEnergy[sst] = NEnergy[sst + 1];
			}
			st -= 1;
			Nst -= 1;

		}
	}

	cudaMemcpy(n1_d, n1_h, Nst*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(n2_d, n2_h, Nst*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(N_d, N_h, Nst*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Nsmall_d, Nsmall_h, Nst*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Msun_d, Msun_h, Nst*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dtiMsun_d, dtiMsun_h, Nst*sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy(U_d, U_h, Nst*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(LI_d, LI_h, Nst*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Energy0_d, Energy0_h, Nst*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(LI0_d, LI0_h, Nst*sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy(NBS_d, NBS_h, Nst*sizeof(int), cudaMemcpyHostToDevice);

	remove3M_kernel <<< Nst, 16 >>> (index_d, N_d, NBS_d);

}


__host__ int Data::freeOrbit(){
	
	cudaError_t error;
	
	free(x4_h);
	free(v4_h);
	free(xold_h);
	free(vold_h);
	free(index_h);
	free(spin_h);
	free(rcrit_h);
	cudaFreeHost(Nenc_m);
	free(aelimits_h);
	free(aecount_h);
	free(enccount_h);
	free(aecountT_h);
	free(enccountT_h);

	free(x4small_h);
	free(v4small_h);
	free(xoldsmall_h);
	free(voldsmall_h);
	free(indexsmall_h);
	free(spinsmall_h);
	free(rcritvsmall_h);
	cudaFreeHost(Nencpairssmall_h);
	cudaFreeHost(Nencpairssmall2_h);
	cudaFreeHost(Nencsmall_m);
	free(vcomsmall_h);
	free(aelimitssmall_h);
	free(aecountsmall_h);
	free(enccountsmall_h);
	free(aecountsmallT_h);
	free(enccountsmallT_h);

	free(U_h);
	free(LI_h);
	free(Energy_h);
	free(Energy0_h);
	free(LI0_h);
	cudaFreeHost(Ncoll_m);
	cudaFreeHost(EjectionFlag_m);
	free(Coll_h);
	cudaFreeHost(test_h);
#if poincareFlag == 1
	free(PFlag_h);
#endif	
	cudaFree(x4_d);
	cudaFree(v4_d);
	cudaFree(xold_d);
	cudaFree(vold_d);
	cudaFree(index_d);
	cudaFree(spin_d);
	cudaFree(a_d);
	cudaFree(rcrit_d);
	cudaFree(rcritv_d);
	cudaFree(Nencpairs_d);
	cudaFree(Nencpairs2_d);
	cudaFree(Encpairs_d);
	cudaFree(Encpairs2_d);

	cudaFree(aelimits_d);
	cudaFree(aecount_d);
	cudaFree(enccount_d);
	cudaFree(aecountT_d);
	cudaFree(enccountT_d);
	
	cudaFree(x4small_d);
	cudaFree(v4small_d);
	cudaFree(xoldsmall_d);
	cudaFree(voldsmall_d);
	cudaFree(indexsmall_d);
	cudaFree(spinsmall_d);
	cudaFree(asmall_d);
	cudaFree(rcritvsmall_d);
	cudaFree(Nencpairssmall_d);
	cudaFree(Nencpairssmall2_d);
	cudaFree(Encpairssmall_d);
	cudaFree(Encpairssmall2_d);

	cudaFree(vcomsmall_d);
	cudaFree(aelimitssmall_d);
	cudaFree(aecountsmall_d);
	cudaFree(enccountsmall_d);
	cudaFree(aecountsmallT_d);
	cudaFree(enccountsmallT_d);
	
	cudaFree(U_d);
	cudaFree(LI_d);
	cudaFree(Energy_d);
	cudaFree(Energy0_d);
	cudaFree(LI0_d);

#if poincareFlag == 1
	cudaFree(PFlag_d);
#endif
#if G3 == 1
	cudaFree(K_d);
	cudaFree(Kold_d);
	cudaFree(BddSign_d);
	cudaFree(groupIndex_d);
	cudaFree(groupIndexOld_d);
	cudaFree(groupIndexsmall_d);
	cudaFree(groupIndexsmallOld_d);
	cudaFree(StopTime_d);
#endif

	cudaFree(Coll_d);
	cudaFree(test_d);
	
	error = cudaGetLastError();
	if(error != 0){
		printf("Cuda Orbit free error = %d = %s\n",error, cudaGetErrorString(error));
		fprintf(masterfile, "Cuda Orbit free error = %d = %s\n",error, cudaGetErrorString(error));
		return 0;
	}
	return 1;
}


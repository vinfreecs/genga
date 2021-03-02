
//**************************************
//This kernel performs a scan operation, used for  stream compactions
//
//It works for the case of multiple blocks
//must be followed by Scan32d2 and Scan32d3
//
//Uses shuffle instructions
//Authors: Simon Grimm
//March 2020
//  *****************************************
__global__ void Scan32d1_kernel(int *Encpairs3_d, int *Nencpairs3_d, const int N, const int NencMax){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	int t1 = 0;
	int t2 = 0;

	__shared__ int t_s[32];
	int lane = threadIdx.x % warpSize;
	int warp = threadIdx.x / warpSize;

	if(warp == 0){
		t_s[threadIdx.x] = 0;
	}

	if(id < N){
		t1 = Encpairs3_d[id * NencMax + 3];
	}
	__syncthreads();
//if(id < 1024) printf("Scan a %d %d %d\n", id, idy, t1);

	for(int i = 1; i < 32; i*=2){
#if OldShuffle == 0
		t2 = __shfl_up_sync(0xffffffff, t1, i, 32);
#else
		t2 = __shfl_up(t1, i);
#endif
		if(idy % 32 >= i) t1 += t2;
	}		
	__syncthreads();

	int t0 = t1;

	if(blockDim.x > warpSize){
		//reduce across warps

		if(lane == 31){
			t_s[warp] = t1;
		}

		__syncthreads();
		//reduce previous warp results in the first warp
		if(warp == 0){
			t1 = t_s[threadIdx.x];
			for(int i = 1; i < 32; i*=2){
#if OldShuffle == 0
				t2 = __shfl_up_sync(0xffffffff, t1, i, 32);
#else
				t2 = __shfl_up(t1, i);
#endif
				if(lane >= i) t1 += t2;
			}
		}
		if(idy < blockDim.x / 32){
			t_s[idy] = t1;
		}

		__syncthreads();

		if(idy >= 32){
			t0 += t_s[warp - 1];
		}
	}
	__syncthreads();
//if(id < 1024) printf("Scan C %d %d %d\n", id, idy, t0);

	Encpairs3_d[id * NencMax + 3] = t0;

	if(idy == blockDim.x - 1){
		Encpairs3_d[blockIdx.x * NencMax + 1] = t0;
//printf("ScanD %d %d\n", blockIdx.x, t0);
	}

}



//**************************************
//This kernel reads the result from the multiple thread block kernel Scan32d1
//and performs the last summation step in
// --a single thread block --
//
//must be followed by Scan32d3
//
//Uses shuffle instructions
//Authors: Simon Grimm
//March 2020
//  *****************************************
__global__ void Scan32d2_kernel(int *Encpairs3_d, int *EncpairsScan_d, int *Nencpairs3_d, const int N, const int NencMax){

	int idy = threadIdx.x;

	int t1 = 0;
	int t2 = 0;

	__shared__ int t_s[32];
	int lane = threadIdx.x % warpSize;
	int warp = threadIdx.x / warpSize;

	if(warp == 0){
		t_s[threadIdx.x] = 0;
	}

	t1 = Encpairs3_d[idy * NencMax + 1];
	if(t1 < 0) t1 = 0;

	__syncthreads();
//if(idy < 32) printf("Scan a %d %d\n", idy, t1);

	for(int i = 1; i < 32; i*=2){
#if OldShuffle == 0
		t2 = __shfl_up_sync(0xffffffff, t1, i, 32);
#else
		t2 = __shfl_up(t1, i);
#endif
		if(idy % 32 >= i) t1 += t2;
	}		
	__syncthreads();
//if(idy < 32) printf("Scan b %d %d\n", idy, t1);

	int t0 = t1;

	if(blockDim.x > warpSize){
		//reduce across warps

		if(lane == 31){
			t_s[warp] = t1;
		}

		__syncthreads();
		//reduce previous warp results in the first warp
		if(warp == 0){
			t1 = t_s[threadIdx.x];
			for(int i = 1; i < 32; i*=2){
#if OldShuffle == 0
				t2 = __shfl_up_sync(0xffffffff, t1, i, 32);
#else
				t2 = __shfl_up(t1, i);
#endif
				if(lane >= i) t1 += t2;
			}
		}
		if(idy < blockDim.x / 32){
			t_s[idy] = t1;
		}

		__syncthreads();

		if(idy >= 32){
			t0 += t_s[warp - 1];
		}
	}
	__syncthreads();
//printf("Scan CC %d %d\n", idy, t0);
	if(idy < (N + 1023) / 1024){
//printf("Scan CC1 %d %d\n", idy, t0);
		EncpairsScan_d[idy] = t0;
	}
}

__global__ void Scan32d3_kernel(int *Encpairs3_d, int *EncpairsScan_d, int *Nencpairs3_d, const int N, const int NencMax){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	if(id < N){
		int ii = id / 1024;
		int t = Encpairs3_d[id * NencMax + 3];
		if(id >= 1024){
			t += EncpairsScan_d[ii - 1];
		}
		Encpairs3_d[id * NencMax + 3] = t;
//if(id % 100 == 0) printf("Scan E %d %d %d\n", id, ii, t, EncpairsScan_d[ii]);

//printf("Scan b %d %d %d\n", idy, t1, Encpairs3_d[idy * NencMax + 0]);
		if(Encpairs3_d[id * NencMax + 0] > 0){
			Encpairs3_d[(t - 1) * NencMax + 1] = id;
		}
	
		if(id == N - 1){
			Nencpairs3_d[0] = t;
//printf("Scan F %d\n",  t);
		}
	}


}

//**************************************
//This kernel performs a scan operation, used for  stream compactions
//
//It works for the case of multiple warps, but only 1 thread block
//
//Uses shuffle instructions
//Authors: Simon Grimm
//March 2020
//  *****************************************
__global__ void Scan32a_kernel(int *Encpairs3_d, int *Nencpairs3_d, const int N, const int NencMax){

	int idy = threadIdx.x;

	int t1 = 0;
	int t2 = 0;

	if(idy < N){
		t1 = Encpairs3_d[idy * NencMax + 3];
	}
	__syncthreads();
//printf("Scan a %d %d %d\n", 0, idy, t1);

	for(int i = 1; i < 32; i*=2){
#if OldShuffle == 0
		t2 = __shfl_up_sync(0xffffffff, t1, i, 32);
#else
		t2 = __shfl_up(t1, i);
#endif
		if(idy % 32 >= i) t1 += t2;
//printf("Scan a %d %d %d\n", i, idy, t1);
	}		
//printf("Scan A %d %d %d\n", 0, idy, t1);

	__syncthreads();

	int t0 = t1;

	if(blockDim.x > warpSize){
		//reduce across warps
		__shared__ int t_s[32];
		int lane = threadIdx.x % warpSize;
		int warp = threadIdx.x / warpSize;
		if(warp == 0){
			t_s[threadIdx.x] = 0;
		}
		__syncthreads();

		if(lane == 31){
			t_s[warp] = t1;
		}

		__syncthreads();
		//reduce previous warp results in the first warp
		if(warp == 0){
			t1 = t_s[threadIdx.x];
			for(int i = 1; i < 32; i*=2){
#if OldShuffle == 0
				t2 = __shfl_up_sync(0xffffffff, t1, i, 32);
#else
				t2 = __shfl_up(t1, i);
#endif
				if(lane >= i) t1 += t2;
//printf("Scan b %d %d %d\n", i, idy, t1);
			}
		}
		if(idy < blockDim.x / 32){
			t_s[idy] = t1;
		}

		__syncthreads();

		if(idy >= 32){
			t0 += t_s[warp - 1];
		}
	}
	__syncthreads();
//printf("Scan C %d %d %d\n", 0, idy, t0);



	if(idy < N){
//printf("Scan c %d %d %d\n", idy, t0, Encpairs3_d[idy * NencMax + 0]);
		Encpairs3_d[idy * NencMax + 3] = t0;
		if(Encpairs3_d[idy * NencMax + 0] > 0){
//printf("Scan d %d %d\n", t0 - 1, idy);
			Encpairs3_d[(t0 - 1) * NencMax + 1] = idy;
		}
	}
	if(idy == blockDim.x - 1){
		Nencpairs3_d[0] = t0;
	}

}

//**************************************
//This kernel performs a scan operation, used for  stream compactions
//
//It works for the case of only 1 single warp
//
//Uses shuffle instructions
//Authors: Simon Grimm
//March 2020
//  *****************************************
__global__ void Scan32c_kernel(int *Encpairs3_d, int *Nencpairs3_d, const int N, const int NencMax){

	int idy = threadIdx.x;

	int t1 = 0;
	int t2 = 0;

	if(idy < N){
		t1 = Encpairs3_d[idy * NencMax + 3];
	}
	__syncthreads();
//printf("Scan a %d %d %d\n", 0, idy, t1);

	for(int i = 1; i < 32; i*=2){
#if OldShuffle == 0
		t2 = __shfl_up_sync(0xffffffff, t1, i, 32);
#else
		t2 = __shfl_up(t1, i);
#endif
		if(idy % 32 >= i) t1 += t2;
//printf("Scan a %d %d %d\n", i, idy, t1);
	}		

	__syncthreads();

	if(idy< N){
//printf("Scan b %d %d %d\n", idy, t1, Encpairs3_d[idy * NencMax + 0]);
		Encpairs3_d[idy * NencMax + 3] = t1;
		if(Encpairs3_d[idy * NencMax + 0] > 0){
			Encpairs3_d[(t1 - 1) * NencMax + 1] = idy;
		}
	}
	if(idy == 31){
		Nencpairs3_d[0] = t1;
	}
}



//**************************************
//This kernel performs a scan operation, used for stream compaction
//
//It works for the case of multiple blocks
//must be followed by Scan32d2 and Scan32d3
//
//Uses shuffle instructions
//Authors: Simon Grimm
//March 2020
//  *****************************************
__global__ void Scan32d1_kernel(int2 *scan_d, const int N){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	int t1 = 0;
	int t2 = 0;

	extern __shared__ int Scand1_s[];
	int *t_s = Scand1_s;

	int lane = threadIdx.x % warpSize;
	int warp = threadIdx.x / warpSize;

	if(warp == 0){
		t_s[threadIdx.x] = 0;
	}

	if(id < N){
		t1 = scan_d[id].x;
	}
	__syncthreads();
//if(id < 1024) printf("Scan a %d %d %d\n", id, idy, t1);

	for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
		t2 = __shfl_up_sync(0xffffffff, t1, i, warpSize);
#else
		t2 = __shfl_up(t1, i);
#endif
		if(idy % warpSize >= i) t1 += t2;
	}		
	__syncthreads();

	int t0 = t1;

	if(blockDim.x > warpSize){
		//reduce across warps

		if(lane == warpSize - 1){
			t_s[warp] = t1;
		}

		__syncthreads();
		//reduce previous warp results in the first warp
		if(warp == 0){
			t1 = t_s[threadIdx.x];
			for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
				t2 = __shfl_up_sync(0xffffffff, t1, i, warpSize);
#else
				t2 = __shfl_up(t1, i);
#endif
				if(lane >= i) t1 += t2;
			}
		}
		if(idy < blockDim.x / warpSize){
			t_s[idy] = t1;
		}

		__syncthreads();

		if(idy >= warpSize){
			t0 += t_s[warp - 1];
		}
	}
	__syncthreads();
//if(id < 1024) printf("Scan C %d %d %d\n", id, idy, t0);

	if(id < N){
		scan_d[id].x = t0;
	}

	if(idy == blockDim.x - 1){
		scan_d[blockIdx.x].y = t0;
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
__global__ void Scan32d2_kernel(int2 *scan_d, const int N){

	int idy = threadIdx.x;

	int t1 = 0;
	int t2 = 0;

	extern __shared__ int Scand2_s[];
	int *t_s = Scand2_s;

	int lane = threadIdx.x % warpSize;
	int warp = threadIdx.x / warpSize;

	if(warp == 0){
		t_s[threadIdx.x] = 0;
	}

	t1 = scan_d[idy].y;
	if(t1 < 0) t1 = 0;

	__syncthreads();
//if(idy < 32) printf("Scan a %d %d\n", idy, t1);

	for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
		t2 = __shfl_up_sync(0xffffffff, t1, i, warpSize);
#else
		t2 = __shfl_up(t1, i);
#endif
		if(idy % warpSize >= i) t1 += t2;
	}		
	__syncthreads();
//if(idy < 32) printf("Scan b %d %d\n", idy, t1);

	int t0 = t1;

	if(blockDim.x > warpSize){
		//reduce across warps

		if(lane == warpSize - 1){
			t_s[warp] = t1;
		}

		__syncthreads();
		//reduce previous warp results in the first warp
		if(warp == 0){
			t1 = t_s[threadIdx.x];
			for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
				t2 = __shfl_up_sync(0xffffffff, t1, i, warpSize);
#else
				t2 = __shfl_up(t1, i);
#endif
				if(lane >= i) t1 += t2;
			}
		}
		if(idy < blockDim.x / warpSize){
			t_s[idy] = t1;
		}

		__syncthreads();

		if(idy >= warpSize){
			t0 += t_s[warp - 1];
		}
	}
	__syncthreads();
//printf("Scan CC %d %d\n", idy, t0);
	if(idy < (N + 1023) / 1024){
//printf("Scan CC1 %d %d\n", idy, t0);
		scan_d[idy].y = t0;
	}
}

__global__ void Scan32d3_kernel(int *Encpairs3_d, int2 *scan_d, int *Nencpairs3_d, const int N, const int NencMax){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockDim.x + idy;

	if(id < N){
		int ii = id / 1024;
		int t = scan_d[id].x;
		if(id >= 1024){
			t += scan_d[ii - 1].y;
		}
		scan_d[id].x = t;
//if(id % 100 == 0) printf("Scan E %d %d %d\n", id, ii, t, scan_d[ii].y);

//printf("Scan b %d %d\n", idy, Encpairs3_d[idy * NencMax + 0]);
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
//This kernel performs a scan operation, used for stream compaction
//
//It works for the case of multiple warps, but only 1 thread block
//
//Uses shuffle instructions
//Authors: Simon Grimm
//March 2020
//  *****************************************
__global__ void Scan32a_kernel(int2 *scan_d, int *Encpairs3_d, int *Nencpairs3_d, const int N, const int NencMax){

	int idy = threadIdx.x;

	int t1 = 0;
	int t2 = 0;

	if(idy < N){
		t1 = scan_d[idy].x;
	}
	__syncthreads();
//printf("Scan a %d %d %d\n", 0, idy, t1);

	for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
		t2 = __shfl_up_sync(0xffffffff, t1, i, warpSize);
#else
		t2 = __shfl_up(t1, i);
#endif
		if(idy % warpSize >= i) t1 += t2;
//printf("Scan a %d %d %d\n", i, idy, t1);
	}		
//printf("Scan A %d %d %d\n", 0, idy, t1);

	__syncthreads();

	int t0 = t1;

	if(blockDim.x > warpSize){
		//reduce across warps
		extern __shared__ int Scana_s[];
		int *t_s = Scana_s;

		int lane = threadIdx.x % warpSize;
		int warp = threadIdx.x / warpSize;
		if(warp == 0){
			t_s[threadIdx.x] = 0;
		}
		__syncthreads();

		if(lane == warpSize - 1){
			t_s[warp] = t1;
		}

		__syncthreads();
		//reduce previous warp results in the first warp
		if(warp == 0){
			t1 = t_s[threadIdx.x];
			for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
				t2 = __shfl_up_sync(0xffffffff, t1, i, warpSize);
#else
				t2 = __shfl_up(t1, i);
#endif
				if(lane >= i) t1 += t2;
//printf("Scan b %d %d %d\n", i, idy, t1);
			}
		}
		if(idy < blockDim.x / warpSize){
			t_s[idy] = t1;
		}

		__syncthreads();

		if(idy >= warpSize){
			t0 += t_s[warp - 1];
		}
	}
	__syncthreads();
//printf("Scan C %d %d %d\n", 0, idy, t0);



	if(idy < N){
//printf("Scan c %d %d %d\n", idy, t0, Encpairs3_d[idy * NencMax + 0]);
		scan_d[idy].x = t0;
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
//This kernel performs a scan operation, used for stream compaction
//
//It works for the case of only 1 single warp
//
//Uses shuffle instructions
//Authors: Simon Grimm
//March 2020
//  *****************************************
__global__ void Scan32c_kernel(int2 *scan_d, int *Encpairs3_d, int *Nencpairs3_d, const int N, const int NencMax){

	int idy = threadIdx.x;

	int t1 = 0;
	int t2 = 0;

	if(idy < N){
		t1 = scan_d[idy].x;
	}
	__syncthreads();
//printf("Scan a %d %d %d\n", 0, idy, t1);

	for(int i = 1; i < warpSize; i*=2){
#if def_OldShuffle == 0
		t2 = __shfl_up_sync(0xffffffff, t1, i, warpSize);
#else
		t2 = __shfl_up(t1, i);
#endif
		if(idy % warpSize >= i) t1 += t2;
//printf("Scan b %d %d %d\n", i, idy, t1);
	}		

	__syncthreads();

	if(idy < N){
//printf("Scan c %d %d %d\n", idy, t1, Encpairs3_d[idy * NencMax + 0]);
		scan_d[idy].x = t1;
		if(Encpairs3_d[idy * NencMax + 0] > 0){
			Encpairs3_d[(t1 - 1) * NencMax + 1] = idy;
		}
	}
	if(idy == warpSize - 1){
		Nencpairs3_d[0] = t1;
	}
}

#if def_CPU == 1
void Scan_cpu(int2 *scan_h, int *Encpairs3_h, int *Nencpairs3_h, const int N, const int NencMax){

	for(int id = 0; id < N; ++id){
		if(id > 0){
			scan_h[id].x += scan_h[id - 1].x;
		}

		if(Encpairs3_h[id * NencMax + 0] > 0){
			int t1 = scan_h[id].x;
			Encpairs3_h[(t1 - 1) * NencMax + 1] = id;
		}
	}

	Nencpairs3_h[0] = scan_h[N - 1].x;
}
#endif


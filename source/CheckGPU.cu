#include <stdlib.h>
#include <stdio.h>

// **********************************
// This code prints the number of NVIDIA GPUs and the compute capabilities
// compile with: nvcc -o CheckGPU CheckGPU.cu 

// Date: April 2022
// Author: Simon Grimm
// **********************************

int main(){

	int devCount = 0;
	int runtimeVersion = 0;
	int driverVersion = 0;

	cudaError_t error;
	error = cudaGetDeviceCount(&devCount);
	if(error > 0){
		printf("device error = %d = %s\n",error, cudaGetErrorString(error));
		return 0;
	}
	if(devCount == 0){
		printf("Error: No valid cuda device!\n");
		return 0;
	}

	cudaDeviceProp devProp;

	cudaRuntimeGetVersion(&runtimeVersion);
	cudaDriverGetVersion(&driverVersion);

	printf("There are %d CUDA devices.\n", devCount);
	printf("Runtime Version: %d\n", runtimeVersion);
	printf("Driver Version: %d\n", driverVersion);

	for(int i = 0; i < devCount; ++i){
		cudaGetDeviceProperties(&devProp, i);

		int computeCapability = devProp.major * 10 + devProp.minor;

		printf("Name:%s, Major:%d, Minor:%d, Compute Capability: %d\n",
		devProp.name, devProp.major, devProp.minor, computeCapability);
	}

	return 0;
}


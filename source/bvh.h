typedef Node* NodePtr;

// **************************************************
// This kernel performs a direct N^2 collision check
// It is used to find encounters between test particles, which
// are not tested in the gravity pre-checker. 
// For large N, a tree code should be used instead
//
// Date: December 2022
// Author: Simon Grimm
// **************************************************
__global__ void collisioncheck_kernel(double4 *x4_d, double *rcritv_d, int *index_d, int *Nencpairs2_d, int2 *Encpairs2_d, const int N1, const int N, const int iy){

	int idx = blockIdx.x * blockDim.x + threadIdx.x + N1;
	int idy = (blockIdx.y + iy) * blockDim.y + threadIdx.y + N1;

	if(idx < N){
		 if(idy < N){
			double4 x4i = x4_d[idx];
			double4 x4j = x4_d[idy];
			double rcritvi = def_pc * rcritv_d[idx];
			double rcritvj = def_pc * rcritv_d[idy];

			bool overlap = true;
			if(x4i.x - rcritvi > x4j.x + rcritvj || x4i.x + rcritvi < x4j.x - rcritvj){
				overlap = false;
			}
			if(x4i.y - rcritvi > x4j.y + rcritvj || x4i.y + rcritvi < x4j.y - rcritvj){
				overlap = false;
			}
			if(x4i.z - rcritvi > x4j.z + rcritvj || x4i.z + rcritvi < x4j.z - rcritvj){
				overlap = false;
			}
			if(idx >= idy){
				overlap = false;
			}
			//ingnore encounters within the same particle cloud
			if(index_d[idx] / WriteEncountersCloudSize_c[0] == index_d[idy] / WriteEncountersCloudSize_c[0]){
				overlap = false;
			}

			if(overlap){
#if def_CPU == 0
				int ne = atomicAdd(&Nencpairs2_d[0], 1);
#else
				int ne;
				#pragma omp atomic capture
				ne = Nencpairs2_d[0]++;
#endif
				Encpairs2_d[ne].x = idx;
				Encpairs2_d[ne].y = idy;
//printf("collisionB %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", idx, idy, x4i.x, x4i.y, x4i.z, x4j.x, x4j.y, x4j.z);
//printf("collisionB %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", idx, idy, x4i.x - rcriti, x4i.y - rcriti, x4i.z, x4j.x - rcritj, x4j.y - rcritj, x4j.z);
//printf("collisionB %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", idx, idy, x4i.x + rcriti, x4i.y - rcriti, x4i.z, x4j.x + rcritj, x4j.y - rcritj, x4j.z);
//printf("collisionB %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", idx, idy, x4i.x - rcriti, x4i.y + rcriti, x4i.z, x4j.x - rcritj, x4j.y + rcritj, x4j.z);
//printf("collisionB %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", idx, idy, x4i.x + rcriti, x4i.y + rcriti, x4i.z, x4j.x + rcritj, x4j.y + rcritj, x4j.z);
			}
		}
	}
}

// *****************************************************
// This function prints the binary representation of a variable
// Usefull for testing
// *****************************************************
__device__ void CheckBinary(unsigned int n){

	printf("%015u ", n);
	for(unsigned int i = 1u << 31; i > 0u; i = i >> 1){
		if((n & i)) printf("1");
		else printf("0");
	}
	printf("\n");
}


// **************************************************
// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
// See: https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
// **************************************************
__device__ unsigned int expandBits(unsigned int v){

	v = (v * 0x00010001u) & 0xFF0000FFu;    //65537  0000000000000010000000000000001 & 11111111000000000000000011111111
	v = (v * 0x00000101u) & 0x0F00F00Fu;    //257    0000000000000000000000100000001 & 00001111000000001111000000001111
	v = (v * 0x00000011u) & 0xC30C30C3u;    //17     0000000000000000000000000010001 & 11000011000011000011000011000011
	v = (v * 0x00000005u) & 0x49249249u;    //5      1001001001001001001001001001001 & 01001001001001001001001001001001
	return v;
}

// **************************************************
// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
// See: https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
// **************************************************
__device__ unsigned int morton3D(double4 x4i){
	//Normalize the coordinates to the range 0 to 1
	float xf = (float)((x4i.x + RcutSun_c[0]) / (2.0 * RcutSun_c[0]));
	float yf = (float)((x4i.y + RcutSun_c[0]) / (2.0 * RcutSun_c[0]));
	float zf = (float)((x4i.z + RcutSun_c[0]) / (2.0 * RcutSun_c[0]));

	xf = min(max(xf * 1024.0f, 0.0f), 1023.0f);
	yf = min(max(yf * 1024.0f, 0.0f), 1023.0f);
	zf = min(max(zf * 1024.0f, 0.0f), 1023.0f);

	unsigned int xx = expandBits((unsigned int)xf);
	unsigned int yy = expandBits((unsigned int)yf);
	unsigned int zz = expandBits((unsigned int)zf);

	return xx * 4 + yy * 2 + zz;
}



// *******************************************************
// This funtion reads the last 4 bits of the array morton_d and sorts them with a counting sort routine
// Each threads has to perform a prefix scan on 16 buckets
// This kernel could still be parallelized further
// 256 * 16 is the maximum to be stored in shared memory
//
// Date: October 2022
// Author: Simon Grimm
// *******************************************************
template <int BL >
__global__ void sort_kernel(double4 *x4_d, unsigned int *morton_d, int2 *sortIndex_d, unsigned int *sortCount_d, unsigned int *sortRank_d, const unsigned int bit, const int N){

	int itx = threadIdx.x;
	int id = blockIdx.x * blockDim.x + itx;

	__shared__ unsigned int c_s[BL * 16];


	for(int i = 0; i < 16; ++i){
		c_s[i * BL + itx] = 0u;
	}

	__syncthreads();

	unsigned int a = 0xFFFFFFFFu;
	if(id < N){
		int iid;
		if(bit == 0u){
			sortIndex_d[id].x = id;
			iid = id;
			a = morton3D(x4_d[id]);
			morton_d[id] = a;
		}
		else{
			iid = sortIndex_d[id].x;
			a = morton_d[iid];
		}
		sortIndex_d[id].y = iid;
	}
//if(id < 20)   printf("%d %u\n", id, a);
//if(id == 0)   printf("%d %u\n", id, a);
//if(id == 0)   CheckBinary(a);

	//mask all bits except the last 4
	unsigned int b = (a >> bit) & 15;	//15 = 1111, 4 bits
//if(id == 0)   CheckBinary(b);

	c_s[b * BL + itx] = 1u;

	__syncthreads();

	//if(id < 20)   printf("%d %u %d\n", id, b, c_s[13 * BL + itx]);

	//Do parallel prefix sum
	unsigned int t;
	for(int j = 0; j < 16; ++j){
		for(int i = 1; i < blockDim.x; i *= 2){
			if(itx >= i){
				t = c_s[j * BL + itx - i] + c_s[j * BL + itx];
			}
			__syncthreads();
			if(itx >= i){
				c_s[j * BL + itx] = t;
			}
			__syncthreads();
		}
	}
//if(id < 20)   printf("%d %u %d\n", id, b, c_s[13 * BL + itx]);

	if(itx < 16){
		unsigned int count = c_s[itx * BL + (BL - 1)];
		sortCount_d[blockIdx.x * 16 + itx] = count;
//if(blockIdx.x < 2) printf("count %d %d %d\n", blockIdx.x, itx, count);
	}

	unsigned int rank = c_s[b * BL + itx] - 1u;
	if(id < N){
		sortRank_d[id] = rank;
//if(blockIdx.x < 2 && itx < 20) printf("rank %d %d %u %d\n", id, itx, b, rank);
	}
}


// ***********************************************
// This kernel combines the buckets from the previous thread blocks.
//
// Date: October 2022
// Author: Simon Grimm
// ************************************************
template <int BL >
__global__ void sortmerge_kernel(unsigned int *sortCount_d, const int NN){

	int itx = threadIdx.x;

	__shared__ unsigned int c_s[BL * 16];
	__shared__ unsigned int sum_s[16];	//needed when NN > 256


	if(itx < 16){
		sum_s[itx] = 0u;
	}
	__syncthreads();
	unsigned int t;

	for(int ii = 0; ii < NN; ii += BL){
		if(itx + ii < NN){
			for(int i = 0; i < 16; ++i){
				c_s[i * BL + itx] = sortCount_d[(itx + ii) * 16 + i];
			}
		}
		else{
			for(int i = 0; i < 16; ++i){
				c_s[i * BL + itx] = 0u;
			}
		}
		__syncthreads();
		if(itx < 16){
			c_s[itx * BL + 0] += sum_s[itx];
		}
		__syncthreads();

//if(itx < 20) printf("%d %u\n", itx, c_s[1 * BL + itx]);
		for(int j = 0; j < 16; ++j){
			for(int i = 1; i < blockDim.x; i *= 2){
				if(itx >= i){
					t = c_s[j * BL + itx - i] + c_s[j * BL + itx];
				}
				__syncthreads();
				if(itx >= i){
					c_s[j * BL + itx] = t;
				}
				__syncthreads();
			}
		}
//if(itx < 20) printf("%d %u\n", itx, c_s[1 * BL + itx]);


		if(itx + ii < NN){
			for(int i = 0; i < 16; ++i){
				sortCount_d[(itx + ii) * 16 + i] = c_s[i * BL + itx];

//if(i == 0) printf("count %d %d %d\n", i, itx + ii, sortCount_d[(itx + ii) * 16 + i]);
//if(itx + ii == NN-1) printf("countT %d %d\n", i, sortCount_d[(itx + ii) * 16 + i]);
			}
		}
		if(itx < 16){
			sum_s[itx] = c_s[itx * BL + (BL - 1)];
		}


		__syncthreads();
	}

	//now scan the total numbers 
	for(int i = 1; i < 16; i *= 2){
		if(itx >= i && itx < 16){
			t = sum_s[itx - i] + sum_s[itx];
//printf("T %d %d %u %u\n", i, itx, sum_s[itx - i], sum_s[itx]);
		}
		__syncthreads();
		if(itx >= i && itx < 16){
			sum_s[itx] = t;
		}
		__syncthreads();
	}
	if(itx < 16){
		sortCount_d[NN * 16 + itx] = sum_s[itx];
//printf("TT %d %u\n", itx, sum_s[itx]);
	}
}

// ***********************************************
// Last part of the radix sort
// This functions calculates the new address in the sorted array 
// The sortIndex_d array contains now the sorted indices of the particle arrays
//
// Date: October 2022
// Author: Simon Grimm
// ************************************************
__global__ void sortscatter_kernel(unsigned int *morton_d, int2 *sortIndex_d, unsigned int *sortCount_d, unsigned int *sortRank_d, const unsigned int bit, const int N, const int NN){

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < N){

		int iid = sortIndex_d[id].y;
		unsigned int a = morton_d[iid];
		unsigned int b = (a >> bit) & 15;	//15 = 1111, 4 bits

		unsigned int rank = sortRank_d[id];
		unsigned int count = 0;
		unsigned int countT = 0;
		if(blockIdx.x > 0){
			count = sortCount_d[(blockIdx.x - 1) * 16 + b];
		}
		if(b > 0){
			countT = sortCount_d[NN * 16 + b -1];
		}

		unsigned int d = rank + count + countT;
//if(threadIdx.x < 20 && blockIdx.x < 2) printf("%d %d %u | %u %u %u\n", blockIdx.x, threadIdx.x, b, rank, count, countT);

		if(d < N){
			sortIndex_d[d].x = sortIndex_d[id].y;
//printf("%d %u %u %d %d\n", id, a, b, d, sortIndex_d[d].x);
		}

		if(bit == 28u){
			//make a copy of morton_d to sort it afterwards
			//the array sortRankd is nod needed here anymore
			sortRank_d[id] = morton_d[id];
		}
	}
}

// ***********************************************
// The array sortRank contains a copy of morton_d.
// morton_d gets sorted now
//
// Date: October 2022
// Author: Simon Grimm
// ************************************************
__global__ void sortscatter2_kernel(unsigned int *morton_d, unsigned int *sortRank_d, int2* sortIndex_d, const int N){

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < N){
		int d = sortIndex_d[id].x;
		morton_d[id] = sortRank_d[d];
	}
}


// *********************************************
// partition part for the quick sort algorithm
// Sortrank can be useda as a temporary copy of morton_d
// https://beginnersbook.com/2015/02/quicksort-program-in-c/
// **********************************************
__host__ void quickSort(unsigned int *morton_d, int2 *sortIndex_d, int first, int last){

	int i, j, pivot;
	unsigned int temp;
	int temp2;

	if(first < last){
		pivot = first;
		i = first;
		j = last;

		while(i < j){
			while(morton_d[i] <= morton_d[pivot] && i < last)
				i++;
			while(morton_d[j] > morton_d[pivot])
				j--;
			if(i < j){
				temp = morton_d[i];
				temp2 = sortIndex_d[i].x;
				morton_d[i] = morton_d[j];
				morton_d[j] = temp;
				sortIndex_d[i].x = sortIndex_d[j].x;
				sortIndex_d[j].x = temp2;
			}

		}


		temp = morton_d[pivot];
		temp2 = sortIndex_d[pivot].x;
		morton_d[pivot] = morton_d[j];
		morton_d[j] = temp;
		sortIndex_d[pivot].x = sortIndex_d[j].x;
		sortIndex_d[j].x = temp2;


		quickSort(morton_d, sortIndex_d, first, j-1);
		quickSort(morton_d, sortIndex_d, j+1, last);
	}

}


__global__ void sortCheck_kernel(unsigned int *morton_d, int2 *sortIndex_d, const int N){

	for(int i = 0; i < N; ++i){
		int iid = sortIndex_d[i].x;
		unsigned int a = morton_d[iid];
		printf("%02d ",i);
		CheckBinary(a);
	}
}

__global__ void sortCheck2_kernel(unsigned int *morton_d, const int N){

	for(int i = 0; i < N; ++i){
		unsigned int a = morton_d[i];
		printf("%02d ",i);
		CheckBinary(a);
	}
}



__global__ void setLeafNode_kernel(Node *leafNodes_d, int2 *sortIndex_d, double4 *x4_d, double *rcritv_d, const int N){

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < N){
		int ii = sortIndex_d[id].x;
		leafNodes_d[id].isLeaf = true;
		leafNodes_d[id].nodeID = ii;
		leafNodes_d[id].childL = nullptr;
		leafNodes_d[id].childR = nullptr;
		leafNodes_d[id].parent = nullptr;

		leafNodes_d[id].rangeL = id;
		leafNodes_d[id].rangeR = id;

		leafNodes_d[id].counter = 1;	//all threads compute ranges

		double4 x4 = x4_d[ii];
		double rcritv = rcritv_d[ii];

		leafNodes_d[id].xmin = float(x4.x - rcritv);
		leafNodes_d[id].xmax = float(x4.x + rcritv);
		leafNodes_d[id].ymin = float(x4.y - rcritv);
		leafNodes_d[id].ymax = float(x4.y + rcritv);
		leafNodes_d[id].zmin = float(x4.z - rcritv);
		leafNodes_d[id].zmax = float(x4.z + rcritv);


//if(ii == 127 || ii == 493){
//printf("LeafNode %d %u %g %g\n", id, ii, x4.x, x4.y);
//printf("LeafNode %d %u %g %g\n", id, ii, x4.x - rcritv, x4.y - rcritv);
//printf("LeafNode %d %u %g %g\n", id, ii, x4.x - rcritv, x4.y + rcritv);
//printf("LeafNode %d %u %g %g\n", id, ii, x4.x + rcritv, x4.y - rcritv);
//printf("LeafNode %d %u %g %g\n", id, ii, x4.x + rcritv, x4.y + rcritv);
//}
	}
}

__global__ void setInternalNode_kernel(Node *internalNodes_d, const int N){

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < N){
		internalNodes_d[id].isLeaf = false;
		internalNodes_d[id].nodeID = id;
		internalNodes_d[id].childL = nullptr;
		internalNodes_d[id].childR = nullptr;
		internalNodes_d[id].parent = nullptr;

		internalNodes_d[id].rangeL = -1;
		internalNodes_d[id].rangeR = -1;

		internalNodes_d[id].counter = 0;	//only 1 thread per node computes parent node

		double r = 10.0 * RcutSun_c[0];

		internalNodes_d[id].xmin =  r;
		internalNodes_d[id].xmax = -r;
		internalNodes_d[id].ymin =  r;
		internalNodes_d[id].ymax = -r;
		internalNodes_d[id].zmin =  r;
		internalNodes_d[id].zmax = -r;
	}
}

// ***********************************************************
// This function computes the highest differing bit as described in 
// Apetri 2014 "Fast and Simple Agglomerative LBVH Construction"
// When the two morton codes are identical, then use the index instead
// see Karras 2012 "Maximizing Parallelism in the Construction of BVHs,
// Octrees, and k-d Tree"
// ***********************************************************
__device__ int highestBit(unsigned int *morton_d, int i){

	unsigned int mi = morton_d[i];
	unsigned int mj = morton_d[i + 1];

	if(mi != mj){
		return (mi ^ mj);
	}
	else{
		return (i ^ (i+1)) + 32;
	}
}

// ***********************************************************
// This kernel build a BVH tree
// It uses a bottom up approach as described in Apetri 2014 "Fast and Simple Agglomerative LBVH Construction"
//
// (Apetri 2014: Optional kernel to store delta(i,j) beforehand)
// Date: October 2022
// Author: Simon Grimm
// ***********************************************************
__global__ void buildBVH_kernel(unsigned int *morton_d, Node *leafNodes_d, Node *internalNodes_d, const int N){

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < N){
		int p = 0;
		float xmin;
		float xmax;
		float ymin;
		float ymax;
		float zmin;
		float zmax;

		Node *node = &leafNodes_d[id];

		Node *childL;
		Node *childR;
		Node *parent;

		int rangeL, rangeR;

		for(int i = 0; i < N; ++i){
			if(p >= 0){
#if def_CPU == 0
				if(atomicAdd(&(node->counter), 1) == 1){
#else
				//not parallel yet
				if(node->counter++ == 1){
#endif
					rangeL = node->rangeL;
					rangeR = node->rangeR;

					if(rangeL == 0 || (rangeR != (N - 1) && highestBit(morton_d, rangeR) < highestBit(morton_d, rangeL - 1))){
						parent = &internalNodes_d[rangeR];
						parent->childL = node;
						parent->rangeL = rangeL;
						node->parent = parent;
					}
					else{
						parent = &internalNodes_d[rangeL - 1];
						parent->childR = node;
						parent->rangeR = rangeR;
						node->parent = parent;
					}
//printf("node %d | %d | %u %d %d | %u %d %u\n", i, id, node->nodeID, rangeL, rangeR, parent->nodeID, parent->isLeaf, node->parent->nodeID);

					if(!(node->isLeaf)){
						childL = node->childL;
						childR = node->childR;

						xmin = fminf(childL->xmin, childR->xmin);
						xmax = fmaxf(childL->xmax, childR->xmax);
						ymin = fminf(childL->ymin, childR->ymin);
						ymax = fmaxf(childL->ymax, childR->ymax);
						zmin = fminf(childL->zmin, childR->zmin);
						zmax = fmaxf(childL->zmax, childR->zmax);

						node->xmin = xmin;
						node->xmax = xmax;
						node->ymin = ymin;
						node->ymax = ymax;
						node->zmin = zmin;
						node->zmax = zmax;

//printf("volume %d %d %d %d\n", i, node->nodeID, childL->nodeID, childR->nodeID);
//printf("LeafNode %d %d %g %g\n", i, node->nodeID, xmin, ymin);
//printf("LeafNode %d %d %g %g\n", i, node->nodeID, xmin, ymax);
//printf("LeafNode %d %d %g %g\n", i, node->nodeID, xmax, ymin);
//printf("LeafNode %d %d %g %g\n", i, node->nodeID, xmax, ymax);
					}
					//node is root node
					if(rangeL == 0 && rangeR == N -1){
//printf("root %d %d\n", id, node->nodeID);
						internalNodes_d[N - 1].childR = node; // escape node
						break;
					}

					node = parent;
					//make sure that global memory updates is visible to other threads
					__threadfence();
					}
					else{
						p = -1;
					break;
				}
			}
		}
	}
}

// ***********************************************************
// This functions checks if two bounding boxes overlap
// ***********************************************************
__device__ bool checkOverlap(Node *nodeA, Node *nodeB){

	if(nodeA->xmin > nodeB->xmax || nodeA->xmax < nodeB->xmin){
		return false;
	}
	if(nodeA->ymin > nodeB->ymax || nodeA->ymax < nodeB->ymin){
		return false;
	}
	if(nodeA->zmin > nodeB->zmax || nodeA->zmax < nodeB->zmin){
		return false;
	}
	return true;
}



// ***********************************************************
// This kernel walks down the BVH tree and compares the nodes to a leaf node
// Ever leaf node is processed in parallel
// The kernel uses a local stack to store the traversal path
// See https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu/
// ***********************************************************
__global__ void traverseBVH_kernel(Node *leafNodes_d, Node *internalNodes_d, int *index_d, int *Nencpairs2_d, int2 *Encpairs2_d, unsigned int N1, const int N){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < N){
		NodePtr stack[64];
		NodePtr *stackPtr = stack;
		*stackPtr++ = nullptr;

		Node *node = internalNodes_d[N - 1].childL;	//root
		Node *leaf = &leafNodes_d[id];

		Node *childL;
		Node *childR;

		for(int i = 0; i < N; ++i){
			//if(id == 520) printf("%d %d\n", i, node->nodeID);
			childL = node->childL;
			childR = node->childR;

			bool overlapL = checkOverlap(leaf, childL);
			bool overlapR = checkOverlap(leaf, childR);

			//remove collisions with j >= i
			if(childL->rangeR <= leaf->rangeR){
				overlapL = false;
			}
			if(childR->rangeR <= leaf->rangeR){
				overlapR = false;
			}
//printf("traverse %d leaf: %d, node: %d L: %d R: %d | %d %d\n", id, leaf->nodeID, node->nodeID, childL->nodeID, childR->nodeID, overlapL, overlapR);

			if(overlapL && childL->isLeaf){
				if(leaf->nodeID >= N1 && childL->nodeID >= N1){

					//ingnore encounters within the same particle cloud
					if(index_d[leaf->nodeID] / WriteEncountersCloudSize_c[0] != index_d[childL->nodeID] / WriteEncountersCloudSize_c[0]){
#if def_CPU == 0
						int ne = atomicAdd(&Nencpairs2_d[0], 1);
#else
						int ne;
						#pragma omp atomic capture
						ne = Nencpairs2_d[0]++;

#endif
						Encpairs2_d[ne].x = leaf->nodeID;
						Encpairs2_d[ne].y = childL->nodeID;
					}
//if(leaf->nodeID == 127 || leaf->nodeID == 493) printf("encounter %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", leaf->nodeID, childL->nodeID, leaf->xmin, leaf->ymin, leaf->zmin, childL->xmin, childL->ymin, childL->zmin);
//printf("encounter %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", leaf->nodeID, childL->nodeID, leaf->xmin, leaf->ymax, leaf->zmin, childL->xmin, childL->ymax, childL->zmin);
//printf("encounter %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", leaf->nodeID, childL->nodeID, leaf->xmax, leaf->ymin, leaf->zmin, childL->xmax, childL->ymin, childL->zmin);
//printf("encounter %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", leaf->nodeID, childL->nodeID, leaf->xmax, leaf->ymax, leaf->zmin, childL->xmax, childL->ymax, childL->zmin);
				}
			}
			if(overlapR && childR->isLeaf){
				if(leaf->nodeID >= N1 && childR->nodeID >= N1){
					//ingnore encounters within the same particle cloud
					if(index_d[leaf->nodeID] / WriteEncountersCloudSize_c[0] != index_d[childR->nodeID] / WriteEncountersCloudSize_c[0]){
#if def_CPU == 0
						int ne = atomicAdd(&Nencpairs2_d[0], 1);
#else
						int ne;
						#pragma omp atomic capture
						ne = Nencpairs2_d[0]++;
#endif
						Encpairs2_d[ne].x = leaf->nodeID;
						Encpairs2_d[ne].y = childR->nodeID;
					}
//if(leaf->nodeID == 127 || leaf->nodeID == 493) printf("encounter %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", leaf->nodeID, childR->nodeID, leaf->xmin, leaf->ymin, leaf->zmin, childR->xmin, childR->ymin, childR->zmin);
//printf("encounter %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", leaf->nodeID, childR->nodeID, leaf->xmin, leaf->ymax, leaf->zmin, childR->xmin, childR->ymax, childR->zmin);
//printf("encounter %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", leaf->nodeID, childR->nodeID, leaf->xmax, leaf->ymin, leaf->zmin, childR->xmax, childR->ymin, childR->zmin);
//printf("encounter %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", leaf->nodeID, childR->nodeID, leaf->xmax, leaf->ymax, leaf->zmin, childR->xmax, childR->ymax, childR->zmin);
				}
			}

			//overlap with internal node
			bool traverseL = overlapL && !(childL->isLeaf);
			bool traverseR = overlapR && !(childR->isLeaf);

			if(!traverseL && !traverseR){
				node = *--stackPtr;
			}
			else{
				node = (traverseL) ? childL : childR;
			if(traverseL && traverseR){
				*stackPtr++ = childR;
			}

			}
			if(node == nullptr){
//printf("%d id %d %d reached the end\n", i, id, leaf->nodeID);
			break;
			}
			__threadfence();
		}
	}
}



__global__ void checkNodes_kernel(Node *internalNodes_d, const int N){

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < N - 1){
		//printf("internal node %d L %d %u R %d %u | P %d\n", id, internalNodes_d[id].childL->isLeaf, internalNodes_d[id].childL->nodeID, internalNodes_d[id].childR->isLeaf, internalNodes_d[id].childR->nodeID, 0);
		printf("internal node %d L %d %u R %d %u | P %u\n", id, internalNodes_d[id].childL->isLeaf, internalNodes_d[id].childL->nodeID, internalNodes_d[id].childR->isLeaf, internalNodes_d[id].childR->nodeID, internalNodes_d[id].parent->nodeID);

	}
}


__host__ void Data::BVHCall1(){
	int N = N_h[0] + Nsmall_h[0];

#if def_CPU == 0
	for(int i = 0; i < N; i += 32768){
		int Ny = min(N - i, 32768);
		if(Ny > 0){
			collisioncheck_kernel <<< dim3((N + 255) / 256, Ny, 1), dim3(256, 1, 1)>>> (x4_d, rcritv_d, index_d, Nencpairs2_d, Encpairs2_d, N_h[0], N, i);
		}
	}
#else
	collisioncheck_kernel <<< dim3((N + 255) / 256, Ny, 1), dim3(256, 1, 1)>>> (x4_d, rcritv_d, index_d, Nencpairs2_d, Encpairs2_d, N_h[0], N, 0);

#endif
}

__host__ void Data::BVHCall2(){
	int N = N_h[0] + Nsmall_h[0];

#if def_CPU == 0
	for(unsigned int b = 0; b < 32; b += 4){
		//printf("******** %u *********\n", b);
		sort_kernel <256> <<< (N + 255) / 256, 256 >>> (x4_d, morton_d, sortIndex_d, sortCount_d, sortRank_d, b, N);
		int NN = (N + 255) / 256;
		//printf("NN %d\n", NN); 

		sortmerge_kernel <256> <<< 1, 256 >>> (sortCount_d, NN);
		sortscatter_kernel <<< (N + 255) / 256, 256 >>> (morton_d, sortIndex_d, sortCount_d, sortRank_d, b, N, NN);

//		sortCheck_kernel(morton_h, sortIndex_h, min(N,100));
	}
	sortscatter2_kernel <<< (N + 255) / 256, 256 >>> (morton_d, sortRank_d, sortIndex_d, N);
#else
	for(int i = 0; i < N; ++i){
		sortIndex_d[i].x = i;
		morton_d[i] = morton3D(x4_d[i]);
	}
	//sortCheck2_kernel <<< 1, 1 >>> (morton_d, N);

	quickSort(morton_d, sortIndex_d, 0, N-1);

#endif

	//sortCheck2_kernel <<< 1, 1 >>> (morton_d, N);

	setLeafNode_kernel <<< (N + 255) / 256, 256 >>> (leafNodes_d, sortIndex_d, x4_d, rcritv_d, N);
	setInternalNode_kernel <<< (N + 255) / 256, 256 >>> (internalNodes_d, N);

	buildBVH_kernel <<< (N + 255) / 256, 256 >>> (morton_d, leafNodes_d, internalNodes_d, N);
	//checkNodes_kernel <<< (N + 255) / 256, 256 >>> (internalNodes_d, N);

	traverseBVH_kernel <<< (N + 255) / 256, 256 >>> (leafNodes_d, internalNodes_d, index_d, Nencpairs2_d, Encpairs2_d, N_h[0], N);
}


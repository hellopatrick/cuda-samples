#include <cuda.h>
#include <cuda_runtime_api.h>

#include "dimensions.h"
// no caching version.
__global__ void naive_game_of_life(int *current, int *future) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	
	if(x < DIM_X && y < DIM_Y) { // MUST MAKE SURE X, Y ARE GOOD! OTHERWISE HAVOC!
		int neighbor_count = 0;
		
		neighbor_count += current[y * DIM_X + ((x - 1 + DIM_X) % DIM_X)];
		neighbor_count += current[y * DIM_X + ((x + 1) % DIM_X)];
		neighbor_count += current[((y - 1 + DIM_Y) % DIM_Y) * DIM_X + x];
		neighbor_count += current[((y + 1) % DIM_Y) * DIM_X + x];
		neighbor_count += current[((y - 1 + DIM_Y) % DIM_Y) * DIM_X + ((x - 1 + DIM_X) % DIM_X)];
		neighbor_count += current[((y - 1 + DIM_Y) % DIM_Y) * DIM_X + ((x + 1) % DIM_X)];
		neighbor_count += current[((y + 1) % DIM_Y) * DIM_X + ((x - 1 + DIM_X) % DIM_X)];
		neighbor_count += current[((y + 1) % DIM_Y) * DIM_X + ((x + 1) % DIM_X)];

		if(neighbor_count == 3) {
			future[y * DIM_X + x] = 1;
		} else if(neighbor_count == 2 && current[y * DIM_X + x] == 1) {
			future[y * DIM_X + x] = 1;
		} else {
			future[y * DIM_X + x] = 0;
		}
	}
}

// the wrapper around the kernel call for main program to call.
extern "C" void naive_game_of_life_wrapper(int *current, int *future) {
	dim3 threads(16,16);
	dim3 blocks(DIM_X/16 + 1, DIM_Y/16 + 1);
	
	naive_game_of_life<<<blocks, threads>>>(current, future);
}
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "dimensions.h"
// no caching version.
__global__ void cached_game_of_life(int *current, int *future) {
	__shared__ int cache[18][18];
	
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	
	int local_x = threadIdx.x;
	int cache_x = local_x + 1;
	int local_y = threadIdx.y;
	int cache_y = local_y + 1;
	
	if(x < DIM_X && y < DIM_Y) { // MUST MAKE SURE X, Y ARE GOOD! OTHERWISE HAVOC!
		cache[cache_x - 1][cache_y - 1] = current[((y - 1 + DIM_Y) % DIM_Y) * DIM_X + ((x - 1 + DIM_X) % DIM_X)];
		cache[cache_x - 1][cache_y + 1] = current[((y + 1) % DIM_Y) * DIM_X + ((x - 1 + DIM_X) % DIM_X)];
		cache[cache_x + 1][cache_y - 1] = current[((y - 1 + DIM_Y) % DIM_Y) * DIM_X + ((x + 1) % DIM_X)];
		cache[cache_x + 1][cache_y + 1] = current[((y + 1) % DIM_Y) * DIM_X + ((x + 1) % DIM_X)];
		
		__syncthreads();
		
		int neighbor_count = 0;
		
		neighbor_count += cache[cache_x - 1][cache_y];
		neighbor_count += cache[cache_x + 1][cache_y];
		
		neighbor_count += cache[cache_x][cache_y - 1];
		neighbor_count += cache[cache_x][cache_y + 1];
		
		neighbor_count += cache[cache_x - 1][cache_y - 1];
		neighbor_count += cache[cache_x - 1][cache_y + 1];
		
		neighbor_count += cache[cache_x + 1][cache_y - 1];
		neighbor_count += cache[cache_x + 1][cache_y + 1];

		if(neighbor_count == 3) {
			future[y * DIM_X + x] = 1;
		} else if(neighbor_count == 2 && cache[cache_x][cache_y] == 1) {
			future[y * DIM_X + x] = 1;
		} else {
			future[y * DIM_X + x] = 0;
		}
	}
}

// the wrapper around the kernel call for main program to call.
void cached_game_of_life_wrapper(int *current, int *future) {
	dim3 threads(16,16);
	dim3 blocks(DIM_X/16 + 1, DIM_Y/16 + 1);
	
	cached_game_of_life<<<blocks, threads>>>(current, future);
}
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "dimensions.h"

/*
	Any live cell with fewer than two live neighbours dies, as if caused by under-population.
	Any live cell with more than three live neighbours dies, as if by overcrowding.
	Any live cell with two or three live neighbours lives on to the next generation.
	Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction
*/

#define CACHE_LEFT_OKAY (local_x > 0)
#define CACHE_RIGHT_OKAY (local_x < 15)
#define CACHE_ABOVE_OKAY (local_y > 0)
#define CACHE_BELOW_OKAY (local_y < 15)

#define LEFT_OKAY (x > 0)
#define RIGHT_OKAY (x < (DIM_X - 1))
#define ABOVE_OKAY (y > 0)
#define BELOW_OKAY (y < (DIM_Y - 1))

__global__ void cached_game_of_life(int *current, int *future) {
	__shared__ int cache[16][16];
	
	int local_x = threadIdx.x;
	int local_y = threadIdx.y;
	
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
		
	if(x < DIM_X && y < DIM_Y) {
		cache[local_y][local_x] = current[y * DIM_X + x];
		__syncthreads();
		
		int neighbor_count = 0;
		
		if(LEFT_OKAY) { neighbor_count += current[y * DIM_X + (x - 1)]; }
		if(RIGHT_OKAY) { neighbor_count += current[y * DIM_X + (x + 1)]; }
		if(ABOVE_OKAY) { neighbor_count += current[(y - 1) * DIM_X + x]; }
		if(BELOW_OKAY) { neighbor_count += current[(y + 1) * DIM_X + x]; }
		if(LEFT_OKAY && ABOVE_OKAY) { neighbor_count += current[(y - 1) * DIM_X + (x - 1)]; }
		if(LEFT_OKAY && BELOW_OKAY) { neighbor_count += current[(y + 1) * DIM_X + (x - 1)]; }
		if(RIGHT_OKAY && ABOVE_OKAY) { neighbor_count += current[(y - 1) * DIM_X + (x + 1)]; }
		if(RIGHT_OKAY && BELOW_OKAY) { neighbor_count += current[(y + 1) * DIM_X + (x + 1)]; }
	}
}

__global__ void simple_game_of_life(int *current, int *future) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	
	if(x < DIM_X && y < DIM_Y) { // MUST MAKE SURE X, Y ARE GOOD! OTHERWISE HAVOC!
		int neighbor_count = 0;
		
		if(LEFT_OKAY) { neighbor_count += current[y * DIM_X + (x - 1)]; }
		if(RIGHT_OKAY) { neighbor_count += current[y * DIM_X + (x + 1)]; }
		if(ABOVE_OKAY) { neighbor_count += current[(y - 1) * DIM_X + x]; }
		if(BELOW_OKAY) { neighbor_count += current[(y + 1) * DIM_X + x]; }
		if(LEFT_OKAY && ABOVE_OKAY) { neighbor_count += current[(y - 1) * DIM_X + (x - 1)]; }
		if(LEFT_OKAY && BELOW_OKAY) { neighbor_count += current[(y + 1) * DIM_X + (x - 1)]; }
		if(RIGHT_OKAY && ABOVE_OKAY) { neighbor_count += current[(y - 1) * DIM_X + (x + 1)]; }
		if(RIGHT_OKAY && BELOW_OKAY) { neighbor_count += current[(y + 1) * DIM_X + (x + 1)]; }

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
void simple_game_of_life_wrapper(int *current, int *future) {
	dim3 threads(16,16);
	dim3 blocks(DIM_X/16 + 1, DIM_Y/16 + 1);
	
	simple_game_of_life<<<blocks, threads>>>(current, future);
}
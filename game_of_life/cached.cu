#include "kernel.cuh"

// cache reads locally for each block.
__global__ void local_cache_game_of_life(int *current, int *future) {
	__shared__ int cache[16][16];
	
	int local_x = threadIdx.x;
	int local_y = threadIdx.y;
	
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	
	if(x < DIM_X && y < DIM_Y) {
		int neighbor_count = 0;
		cache[local_x][local_y] = current[y * DIM_X + x];
		__syncthreads();
		
		if(LEFT_OKAY) {
			if(CACHE_LEFT_OKAY) {
				neighbor_count += cache[local_x - 1][local_y];
			} else {
				neighbor_count += current[y * DIM_X + (x - 1)]; 
			}
		}
		
		if(RIGHT_OKAY) { 
			if(CACHE_RIGHT_OKAY) {
				neighbor_count += cache[local_x + 1][local_y];
			} else {
				neighbor_count += current[y * DIM_X + (x + 1)]; 
			} 
		}
		
		if(ABOVE_OKAY) { 
			if(CACHE_ABOVE_OKAY) {
				neighbor_count += cache[local_x][local_y - 1];
			} else {
				neighbor_count += current[(y - 1) * DIM_X + x]; 
			} 
		}
		
		if(BELOW_OKAY) { 
			if(CACHE_BELOW_OKAY) {
				neighbor_count += cache[local_x][local_y + 1];
			} else {
				neighbor_count += current[(y + 1) * DIM_X + x]; 
			} 
		}
		
		if(LEFT_OKAY && ABOVE_OKAY) { 
			if(CACHE_LEFT_OKAY && CACHE_ABOVE_OKAY) {
				neighbor_count += cache[local_x - 1][local_y - 1];
			} else {
				neighbor_count += current[(y - 1) * DIM_X + (x - 1)]; 
			} 
		}
		
		if(LEFT_OKAY && BELOW_OKAY) { 
			if(CACHE_LEFT_OKAY && CACHE_BELOW_OKAY) {
				neighbor_count += cache[local_x - 1][local_y + 1];
			} else {
				neighbor_count += current[(y + 1) * DIM_X + (x - 1)]; 
			} 
		}
		
		if(RIGHT_OKAY && ABOVE_OKAY) { 
			if(CACHE_RIGHT_OKAY && CACHE_ABOVE_OKAY) {
				neighbor_count += cache[local_x + 1][local_y - 1];
			} else {
				neighbor_count += current[(y - 1) * DIM_X + (x + 1)]; 
			} 
		}
		
		if(RIGHT_OKAY && BELOW_OKAY) { 
			if(CACHE_RIGHT_OKAY && CACHE_BELOW_OKAY) {
				neighbor_count += cache[local_x + 1][local_y + 1];
			} else {
				neighbor_count += current[(y + 1) * DIM_X + (x + 1)]; 
			}
		}
		
		if(neighbor_count == 3) {
			future[y * DIM_X + x] = 1;
		} else if(neighbor_count == 2 && current[y * DIM_X + x] == 1) {
			future[y * DIM_X + x] = 1;
		} else {
			future[y * DIM_X + x] = 0;
		}
	}
}

void cached_game_of_life_wrapper(int *current, int *future) {
	dim3 threads(16,16);
	dim3 blocks(DIM_X/16 + 1, DIM_Y/16 + 1);
	
	local_cache_game_of_life<<<blocks, threads>>>(current, future);
}
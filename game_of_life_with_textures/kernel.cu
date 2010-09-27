#include <cuda.h>
#include <cuda_runtime_api.h>

#include "dimensions.h"

/*
	Any live cell with fewer than two live neighbours dies, as if caused by under-population.
	Any live cell with more than three live neighbours dies, as if by overcrowding.
	Any live cell with two or three live neighbours lives on to the next generation.
	Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction
*/

#define LEFT_OKAY (x > 0)
#define RIGHT_OKAY (x < (DIM_X - 1))
#define ABOVE_OKAY (y > 0)
#define BELOW_OKAY (y < (DIM_Y - 1))

texture<int> a, b;

__global__ void kernel(int current_is_a, int *future) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int point = x + y * DIM_X;
	
	if(x < DIM_X && y < DIM_Y) { // MUST MAKE SURE X, Y ARE GOOD! OTHERWISE HAVOC!
		int neighbor_count = 0;
		int alive = 0;
		int left = point - 1;
		int right = point + 1;
		
		int top = point - DIM_X;
		int top_left = top - 1;
		int top_right = top + 1;
		int bottom = point + DIM_X;
		int bottom_left = bottom - 1;
		int bottom_right = bottom + 1;
		
		if(current_is_a) {
			alive = tex1Dfetch(a, point);
			if(LEFT_OKAY) { neighbor_count += tex1Dfetch(a,left); }
			if(RIGHT_OKAY) { neighbor_count += tex1Dfetch(a,right); }
			if(ABOVE_OKAY) { neighbor_count += tex1Dfetch(a,top); }
			if(BELOW_OKAY) { neighbor_count += tex1Dfetch(a,bottom); }
			if(LEFT_OKAY && ABOVE_OKAY) { neighbor_count += tex1Dfetch(a,top_left); }
			if(LEFT_OKAY && BELOW_OKAY) { neighbor_count += tex1Dfetch(a,bottom_left); }
			if(RIGHT_OKAY && ABOVE_OKAY) { neighbor_count += tex1Dfetch(a,top_right); }
			if(RIGHT_OKAY && BELOW_OKAY) { neighbor_count += tex1Dfetch(a,bottom_right); }
		} else {
			alive = tex1Dfetch(b, point);
			if(LEFT_OKAY) { neighbor_count += tex1Dfetch(b,left); }
			if(RIGHT_OKAY) { neighbor_count += tex1Dfetch(b,right); }
			if(ABOVE_OKAY) { neighbor_count += tex1Dfetch(b,top); }
			if(BELOW_OKAY) { neighbor_count += tex1Dfetch(b,bottom); }
			if(LEFT_OKAY && ABOVE_OKAY) { neighbor_count += tex1Dfetch(b,top_left); }
			if(LEFT_OKAY && BELOW_OKAY) { neighbor_count += tex1Dfetch(b,bottom_left); }
			if(RIGHT_OKAY && ABOVE_OKAY) { neighbor_count += tex1Dfetch(b,top_right); }
			if(RIGHT_OKAY && BELOW_OKAY) { neighbor_count += tex1Dfetch(b,bottom_right); }
		}
				
		if(neighbor_count == 3) {
					future[y * DIM_X + x] = 1;
				} else if(neighbor_count == 2 && alive == 1) {
					future[y * DIM_X + x] = 1;
				} else {
					future[y * DIM_X + x] = 0;
				}
	}
}

extern "C" void setup_textures(int *first, int *second) {
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
	cudaBindTexture(NULL, a, first, desc, DIM_X * DIM_Y * sizeof(int));
	cudaBindTexture(NULL, b, second, desc, DIM_X * DIM_Y * sizeof(int));
}

extern "C" void free_textures() {
	cudaUnbindTexture(a);
	cudaUnbindTexture(b);
}

// the wrapper around the kernel call for main program to call.
extern "C" void kernel_wrapper(int current_is_a, int *out) {
	dim3 threads(16,16);
	dim3 blocks((DIM_X + DIM_X - 1)/16 + 1, (DIM_Y + DIM_Y - 1)/16);
	
	kernel<<<blocks, threads>>>(current_is_a, out);
}
#include <cuda.h>
#include <cuda_runtime_api.h>

// the actual instructions to be performed by each thread on the gpu
__global__ void square(int vector_length, float *g_data, float *g_result) {
	// calculate the a global thread index
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	
	// if that index is in g_data, compute & store square.
	if(tid < vector_length) { 
		g_result[tid] = g_data[tid] * g_data[tid];
	}
}

// the wrapper around the kernel call for C program to call.
void kernel_wrapper(int vector_length, float *g_data, float *g_result) {
	// number of threads in block. it should at least be a power of 2... and preferably a multiple of 32?
	int threads_per_block = 256;
	
	// number of blocks in the grid.
	int blocks_per_grid = 4096;
		
	// the total number of threads executed = threads_per_block * blocks_per_grid
	// call the kernel using "triple chevron" notation.
	square<<<blocks_per_grid, threads_per_block >>>(vector_length, g_data, g_result);
}
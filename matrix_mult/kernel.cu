#include <cuda.h>
#include <cuda_runtime_api.h>

// the actual instructions to be performed by each thread on the gpu
__global__ void matrix_mult(int dimension, int *matrix_a, int *matrix_b, int *matrix_c) {
	// calculate the a global thread index
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	if(col < dimension && row < dimension) {
		int sum = 0;
		for(int j = 0; j < dimension; j++) {
			sum += matrix_a[row*dimension + j] * matrix_b[j*dimension + col];
		}
		matrix_c[row*dimension + col] = sum;
	}
}

// the wrapper around the kernel call for main program to call.
void kernel_wrapper(int dimension, int *matrix_a, int *matrix_b, int *matrix_c) {
	dim3 threads(16,16);	
	dim3 blocks(256, 256);
	
	// the total number of threads executed = threads_per_block * blocks_per_grid
	// call the kernel using "triple chevron" notation.
	matrix_mult<<<blocks, threads>>>(dimension, matrix_a, matrix_b, matrix_c);
}
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#define VECTOR_LENGTH 256*2048
#define SMALL 0.00001f

// function stub that will be linked in from kernel.
void kernel_wrapper(int vector_length, float *g_data, float *g_result);

int main(int argc, char const *argv[]) {
	// arrays for data		
	float *host_data, *gpu_data;
	float *gpu_results, *host_results, *host_copy_of_gpu_results;
	
	// allocating memory for the host
	// we have two different versions of malloc.
	// use cudaMallocHost only for data that will be copied to GPU.
	cudaMallocHost((void**) &host_data, VECTOR_LENGTH * sizeof(float));
	host_results = (float *) malloc(VECTOR_LENGTH * sizeof(float));
	host_copy_of_gpu_results = (float *) malloc(VECTOR_LENGTH * sizeof(float));
	
	// allocating memory for the device
	cudaMalloc((void**) &gpu_data, VECTOR_LENGTH * sizeof(float));
	cudaMalloc((void**) &gpu_results, VECTOR_LENGTH * sizeof(float));
	
	srand(time(NULL));
	for(int n = 0; n < VECTOR_LENGTH; n++) {
		host_data[n] = ((float) rand())/rand(); // populate host_data with random floats.
		host_results[n] = host_data[n] * host_data[n];	// calculate squares.
	}
	
	// copy host_data to gpu.
	cudaMemcpy(gpu_data, host_data, VECTOR_LENGTH * sizeof(float), cudaMemcpyHostToDevice);

	// run kernel
	kernel_wrapper(VECTOR_LENGTH, gpu_data, gpu_results);
	
	// copy gpu_results back to host_data.
	cudaMemcpy(host_copy_of_gpu_results, gpu_results, VECTOR_LENGTH * sizeof(float), cudaMemcpyDeviceToHost);
	
	// check answers
	float diff;
	for(int n = 0; n < VECTOR_LENGTH; n++){
		diff = host_copy_of_gpu_results[n] - host_results[n];
		assert(-SMALL < diff && diff < SMALL);
	}
	
	printf("PASS!\n");
	
	free(host_results);
	free(host_copy_of_gpu_results);
	
	cudaFree(host_data);
	cudaFree(gpu_data);
	cudaFree(gpu_results);
	
	return 0;
}
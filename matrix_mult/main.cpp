#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#define DIM 768

void kernel_wrapper(int dimension, int *matrix_a, int *matrix_b, int *matrix_c);

void random_matrices(int *matrix_a, int *matrix_b);
void print_matrix(int *matrix);
void matrix_product(int *matrix_a, int *matrix_b, int *matrix_c);
bool check_product(int *gpu, int *cpu);

int main(int argc, char const *argv[]) {
	// arrays for data		
	int *host_a, *host_b, *host_c, *host_gold_c;
	int *gpu_a, *gpu_b, *gpu_c;
	
	cudaMallocHost((void**) &host_a, DIM * DIM * sizeof(int));
	cudaMallocHost((void**) &host_b, DIM * DIM * sizeof(int));
	cudaMallocHost((void**) &host_c, DIM * DIM * sizeof(int));
	host_gold_c =  (int *) malloc(DIM * DIM * sizeof(int));

	cudaMalloc((void**) &gpu_a, DIM * DIM * sizeof(int));
	cudaMalloc((void**) &gpu_b, DIM * DIM * sizeof(int));
	cudaMalloc((void**) &gpu_c, DIM * DIM * sizeof(int));
	
	srand(time(NULL));
	random_matrices(host_a, host_b);
//	print_matrix(host_a);
//	print_matrix(host_b);
	
	// copy host_data to gpu.
	cudaMemcpy(gpu_a, host_a, DIM * DIM * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_b, host_b, DIM * DIM * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_c, host_c, DIM * DIM * sizeof(int), cudaMemcpyHostToDevice);

	// run kernel
	printf("computing matrix product on gpu...\n");
	
	cudaEvent_t	start_gpu, stop_gpu; 
	cudaEventCreate(&start_gpu);
	cudaEventCreate(&stop_gpu);
	cudaEventRecord(start_gpu, 0);
	
	kernel_wrapper(DIM, gpu_a, gpu_b, gpu_c);
	cudaMemcpy(host_c, gpu_c, DIM * DIM * sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaEventRecord(stop_gpu, 0);
	cudaEventSynchronize(stop_gpu);
	
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start_gpu, stop_gpu); 
	printf( "Time for GPU: %3.1f s\n", elapsedTime/1000.0f );
	cudaEventDestroy(start_gpu);
	cudaEventDestroy(stop_gpu);
	
	clock_t start_cpu, stop_cpu;
	printf("computing matrix product on cpu...\n");
	start_cpu = clock();
	matrix_product(host_a, host_b, host_gold_c);
	stop_cpu = clock();
	
	printf("Time for GPU: %3.1f s\n", (float)(stop_cpu - start_cpu)/CLOCKS_PER_SEC);
	// copy gpu_results back to host_data.
//	print_matrix(host_c);
	
	if (check_product(host_c, host_gold_c)) {
		printf("Worked!\n");
	}
	
	free(host_gold_c);
	cudaFree(host_a);
	cudaFree(host_b);
	cudaFree(host_c);
	cudaFree(gpu_a);
	cudaFree(gpu_b);
	cudaFree(gpu_c);
	
	return 0;
}

void random_matrices(int *matrix_a, int *matrix_b) {
	for(int row = 0; row < DIM; row++) {
		for(int column = 0; column < DIM; column++) {
			matrix_a[row*DIM + column] = rand() % 100;
			matrix_b[row*DIM + column] = rand() % 100;
		}
	}
}

void print_matrix(int *matrix) {
	for(int row = 0; row < DIM; row++) {
		for(int column = 0; column < DIM; column++) {
			printf("%d ", matrix[row*DIM + column]);
		}
		printf("\n");
	}
}

void matrix_product(int *matrix_a, int *matrix_b, int *matrix_c) {
	for(int row = 0; row < DIM; row++) {
		for(int col = 0; col < DIM; col++) {
			int sum = 0;
			for(int j = 0; j < DIM; j++) {
				sum += matrix_a[row*DIM + j] * matrix_b[j*DIM + col];
			}
			matrix_c[row*DIM + col] = sum;
		}
	}
}

bool check_product(int *gpu, int *cpu) {
	for(int row = 0; row < DIM; row++) {
		for(int column = 0; column < DIM; column++) {
			assert(gpu[row*DIM + column] == cpu[row*DIM + column]);
		}
	}
	return true;
}
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "dimensions.h"

#define LEFT_OKAY (x > 0)
#define RIGHT_OKAY (x < (DIM_X - 1))
#define ABOVE_OKAY (y > 0)
#define BELOW_OKAY (y < (DIM_Y - 1))

void setup_textures(int *first, int *second);
void free_textures();
void kernel_wrapper(bool current_is_a, int *out);

void fill_board(int *board, int percent);
void print_board(int *board);
void print_neighbors(int *board);
void update_board(int *current, int *future);
void check_boards(int *one, int *two);

cudaError_t error;

int main(int argc, char const *argv[]) {
	printf("Computing Game Of Life On %d x %d Board.\n", DIM_X, DIM_Y);
	
	int *host_current, *host_future, *host_gpu_results;
	int *gpu_current, *gpu_future;
	
	cudaMallocHost((void**) &host_current, DIM_X * DIM_Y * sizeof(int));
	cudaMallocHost((void**) &host_future, DIM_X * DIM_Y * sizeof(int));	
	cudaMallocHost((void**) &host_gpu_results, DIM_X * DIM_Y * sizeof(int));
	cudaMalloc((void**) &gpu_current, DIM_X * DIM_Y * sizeof(int));
	cudaMalloc((void**) &gpu_future, DIM_X * DIM_Y * sizeof(int));
	assert(cudaGetLastError() == cudaSuccess);
	
	fill_board(host_current, 40);
	
	cudaMemcpy(gpu_current, host_current, DIM_X * DIM_Y * sizeof(int), cudaMemcpyHostToDevice);
	setup_textures(gpu_current, gpu_future);
	
	assert(cudaGetLastError() == cudaSuccess);
	
	clock_t start, stop;
	bool current_is_a = true;
	printf("START!\n");
	for(int i = 1; i < 10; i++) {
		start = clock();
		
		int *output = current_is_a ? gpu_future : gpu_current;
		kernel_wrapper(current_is_a, output);
		current_is_a = !current_is_a;
		
		cudaMemcpy(host_gpu_results, output, DIM_X * DIM_Y * sizeof(int), cudaMemcpyDeviceToHost);
		assert(cudaGetLastError() == cudaSuccess);
		
		stop = clock();
		printf("Time for Textured GPU To Compute Next Phase: %.5f s\n", (float)(stop - start)/CLOCKS_PER_SEC);
				
		start = clock();
		update_board(host_current, host_future);
		stop = clock();
		printf("Time for CPU To Compute Next Phase: %.5f s\n", (float)(stop - start)/CLOCKS_PER_SEC);
		
		printf("======\n");
		check_boards(host_gpu_results, host_future);
				
		int *temp;
		temp = host_current;
		host_current = host_future;
		host_future = temp;
	}
	
	cudaFree(host_future);
	cudaFree(host_current);
	cudaFree(host_gpu_results);
	cudaFree(gpu_current);
	cudaFree(gpu_future);
	
	return 0;
}

void add_glider(int *board) {
	for(int y = 0; y < DIM_Y; y++) {
		for(int x = 0; x < DIM_X; x++) {
			board[y * DIM_X + x] = 0;
		}
	}
	
	board[2 * DIM_X + 1] = 1;
	board[3 * DIM_X + 2] = 1;
	board[1 * DIM_X + 3] = 1;
	board[2 * DIM_X + 3] = 1;
	board[3 * DIM_X + 3] = 1;
}

void print_neighbors(int *board) {
	for(int y = 0; y < DIM_Y; y++) {
		for(int x = 0; x < DIM_X; x++) {
			printf("%d", board[y * DIM_X + x]);
		}
		printf("\n");
	}
}

void print_board(int *board) {
	for(int y = 0; y < DIM_Y; y++) {
		for(int x = 0; x < DIM_X; x++) {
			if(board[y * DIM_X + x] == 1) { printf("*"); }
			else { printf(" ");}
		}
		printf("\n");
	}
}

void update_board(int *current, int *future) {
	for(int y = 0; y < DIM_Y; y++) {
		for(int x = 0; x < DIM_X; x++) {
			int neighbor_count = 0;
			//printf("(%d, %d) >>", x, y);
			if(LEFT_OKAY) { neighbor_count += current[y * DIM_X + (x - 1)]; }
			if(RIGHT_OKAY) { neighbor_count += current[y * DIM_X + (x + 1)]; }
			if(ABOVE_OKAY) { neighbor_count += current[(y - 1) * DIM_X + x]; }
			if(BELOW_OKAY) { neighbor_count += current[(y + 1) * DIM_X + x]; }
			if(LEFT_OKAY && ABOVE_OKAY) { neighbor_count += current[(y - 1) * DIM_X + (x - 1)]; }
			if(LEFT_OKAY && BELOW_OKAY) { neighbor_count += current[(y + 1) * DIM_X + (x - 1)]; }
			if(RIGHT_OKAY && ABOVE_OKAY) { neighbor_count += current[(y - 1) * DIM_X + (x + 1)]; }
			if(RIGHT_OKAY && BELOW_OKAY) { neighbor_count += current[(y + 1) * DIM_X + (x + 1)]; }

			//printf("\n");
			if(neighbor_count == 3) {
				future[y * DIM_X + x] = 1;
			} else if(neighbor_count == 2 && current[y * DIM_X + x] == 1) {
				future[y * DIM_X + x] = 1;
			} else {
				future[y * DIM_X + x] = 0;
			}
		}
	}
}

void check_boards(int *one, int *two) {
	for(int y = 0; y < DIM_Y; y++) {
		for(int x = 0; x < DIM_X; x++) {
			assert(one[y * DIM_X + x] == two[y * DIM_X + x]);
		}
	}
}

void fill_board(int *board, int percent) {
	int used_dots = 0;
	srand(time(NULL));
	for(int y = 0; y < DIM_Y; y++) {
		for(int x = 0; x < DIM_X; x++) {
			if( rand() % 100 < percent) { board[y*DIM_X + x] = 1; }
			else { board[y*DIM_X + x] = 0; }
		}
	}
}
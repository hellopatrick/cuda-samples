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

void simple_game_of_life_wrapper(int *current, int *future);
void add_glider(int *board);
void print_board(int *board);
void update_board(int *current, int *future);
void check_boards(int *one, int *two);

int main(int argc, char const *argv[]) {
	printf("Computing Game Of Life On %d x %d Board.\n", DIM_X, DIM_Y);
	
	int *host_current, *host_future;
	int *gpu_current, *gpu_future;
	
	cudaMallocHost((void**) &host_current, DIM_X * DIM_Y * sizeof(int));
	host_future = (int *) malloc(DIM_X * DIM_Y * sizeof(int));
		
	cudaMalloc((void**) &gpu_current, DIM_X * DIM_Y * sizeof(int));
	cudaMalloc((void**) &gpu_future, DIM_X * DIM_Y * sizeof(int));
	
	add_glider(host_current); print_board(host_current);
		
	for(int i = 1; i < 50; i++) {
		cudaMemcpy(gpu_current, host_current, DIM_X * DIM_Y * sizeof(int), cudaMemcpyHostToDevice);
		simple_game_of_life_wrapper(gpu_current, gpu_future);
		update_board(host_current, host_future);
		cudaMemcpy(host_current, gpu_future, DIM_X * DIM_Y * sizeof(int), cudaMemcpyDeviceToHost);
		
		// host current now is the gpu_future
		printf("=========\n");
		print_board(host_current);
				
		check_boards(host_current, host_future);
	}
	
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

void print_board(int *board) {
	for(int y = 0; y < DIM_Y; y++) {
		for(int x = 0; x < DIM_X; x++) {
			printf("%d", board[y * DIM_X + x]);
		}
		printf("\n");
	}
}

void update_board(int *current, int *future) {
	for(int y = 0; y < DIM_Y; y++) {
		for(int x = 0; x < DIM_X; x++) {
			int neighbor_count = 0;
			//printf("(%d, %d) >>", x, y);
			if(LEFT_OKAY) { neighbor_count += current[y * DIM_X + (x - 1)]; /*printf("(%d, %d)[%d] ", x-1, y, neighbor_count);*/}
			if(RIGHT_OKAY) { neighbor_count += current[y * DIM_X + (x + 1)]; /*printf("(%d, %d)[%d] ", x+1, y, neighbor_count);*/}
			if(ABOVE_OKAY) { neighbor_count += current[(y - 1) * DIM_X + x]; /*printf("(%d, %d)[%d] ", x, y-1, neighbor_ count);*/}
			if(BELOW_OKAY) { neighbor_count += current[(y + 1) * DIM_X + x]; /*printf("(%d, %d)[%d] ", x, y+1, neighbor_count);*/}
			if(LEFT_OKAY && ABOVE_OKAY) { neighbor_count += current[(y - 1) * DIM_X + (x - 1)]; /*printf("(%d, %d)[%d] ", x-1, y-1, neighbor_count);*/}
			if(LEFT_OKAY && BELOW_OKAY) { neighbor_count += current[(y + 1) * DIM_X + (x - 1)]; /*printf("(%d, %d)[%d] ", x-1, y+1, neighbor_count);*/}
			if(RIGHT_OKAY && ABOVE_OKAY) { neighbor_count += current[(y - 1) * DIM_X + (x + 1)]; /*printf("(%d, %d)[%d] ", x+1, y-1, neighbor_count);*/}
			if(RIGHT_OKAY && BELOW_OKAY) { neighbor_count += current[(y + 1) * DIM_X + (x + 1)]; /*printf("(%d, %d)[%d] ", x+1, y+1, neighbor_count);*/}

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
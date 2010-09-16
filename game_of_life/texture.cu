#include "kernel.cuh"

texture<int,2> a;
texture<int,2> b;
bool which = true;

__global__ void textured_game_of_life(int *future) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	
	if(x < DIM_X && y < DIM_Y) { // MUST MAKE SURE X, Y ARE GOOD! OTHERWISE HAVOC!
		int neighbor_count = 0;
		int alive;
		bool current = true;
		if(current) {
			alive = tex2D(a, x, y);
			if(LEFT_OKAY) { neighbor_count += tex2D(a, x-1, y); }
			if(RIGHT_OKAY) { neighbor_count += tex2D(a, x+1, y); }
			if(ABOVE_OKAY) { neighbor_count += tex2D(a, x, y-1); }
			if(BELOW_OKAY) { neighbor_count += tex2D(a, x, y+1); }
			if(LEFT_OKAY && ABOVE_OKAY) { neighbor_count += tex2D(a, x-1, y-1); }
			if(LEFT_OKAY && BELOW_OKAY) { neighbor_count += tex2D(a, x-1, y+1); }
			if(RIGHT_OKAY && ABOVE_OKAY) { neighbor_count += tex2D(a, x+1, y-1); }
			if(RIGHT_OKAY && BELOW_OKAY) { neighbor_count += tex2D(a, x+1, y+1); }
		} else {
			alive = tex2D(b, x, y);
			if(LEFT_OKAY) { neighbor_count += tex2D(b, x-1, y); }
			if(RIGHT_OKAY) { neighbor_count += tex2D(b, x+1, y); }
			if(ABOVE_OKAY) { neighbor_count += tex2D(b, x, y-1); }
			if(BELOW_OKAY) { neighbor_count += tex2D(b, x, y+1); }
			if(LEFT_OKAY && ABOVE_OKAY) { neighbor_count += tex2D(b, x-1, y-1); }
			if(LEFT_OKAY && BELOW_OKAY) { neighbor_count += tex2D(b, x-1, y+1); }
			if(RIGHT_OKAY && ABOVE_OKAY) { neighbor_count += tex2D(b, x+1, y-1); }
			if(RIGHT_OKAY && BELOW_OKAY) { neighbor_count += tex2D(b, x+1, y+1); }
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

void setup_textures(int *first, int *second) {
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
	
	cudaBindTexture2D(NULL, a, first, desc, DIM_X, DIM_Y, sizeof(int) * DIM_X);
	cudaBindTexture2D(NULL, b, second, desc, DIM_X, DIM_Y, sizeof(int) * DIM_X);
}

void textured_game_of_life_wrapper(int *out) {
	dim3 threads(16,16);
	dim3 blocks(DIM_X/16 + 1, DIM_Y/16 + 1);
	
	textured_game_of_life<<<blocks, threads>>>(out);
	which = !which;
}
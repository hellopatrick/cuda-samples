#include <cuda.h>
#include <cuda_runtime_api.h>
#include "png_helper.h"

__global__ void naive_transpose(uchar4 *in, uchar4 *out, int width, int height) {
//	__shared__ uchar4 cache[16][17];
	
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if(x < width && y < height) {
		int in_idx = y * width + x;
		int out_idx = x * height + y;
		out[out_idx] = in[in_idx];
	}
}

// the wrapper around the kernel call for main program to call.
extern "C" void naive_transpose_wrapper(uchar4 *in, uchar4 *out, png_t *info) {
	dim3 threads(16,16);
	dim3 blocks((info->width)/16 + 1, (info->height)/16 + 1);
	
	naive_transpose<<<blocks, threads>>>(in, out, info->width, info->height);
}

extern "C" void cpu_transpose(uchar4 *in, uchar4 *out, png_t *info) {
	int x, y;
	for(y = 0; y < info->height; y++) {
		for(x = 0; x < info->width; x++) {
			out[x*(info->height) + y] = in[y*(info->width) + x];
		}
	}
}
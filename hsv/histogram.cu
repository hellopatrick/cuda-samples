#include <cuda.h>
#include <cuda_runtime_api.h>
#include "hsv_convert.h"

__global__ void compute_histogram(float4 *hsv, int *histogram, int width, int height) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	
	__shared__ unsigned int cached_histogram[360];
	
	if(x < width && y < height) {
		int __float2uint_rn()
	}
}

extern "C" void compute_histogram_image(float4 *hsv, int *histogram, uchar4 *image, int width, int height) {
	dim3 threads(16,16);
	dim3 blocks((width + 15)/16, (height + 15)/16);
	
	compute_histogram<<<blocks, threads>>>(hsv, histogram, width, height);
	draw_image<<<blocks, threads>>>(histogram, image);
}
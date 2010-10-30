#include <cuda.h>
#include <cuda_runtime_api.h>
#include "hsv_convert.cuh"

__global__ void convert_to_hsv(uchar4 *rgb, float4 *hsv, int width, int height) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	
	if(x < width && y < height) {
		uchar4 rgb_pixel = rgb[x + width*y];
		float4 hsv_pixel = convert_one_pixel_to_hsv(rgb_pixel);
		hsv[x + width*y] = hsv_pixel;
	}
}

extern "C" void convert_to_hsv_wrapper(uchar4 *rgb, float4 *hsv, int width, int height) {
	dim3 threads(16,16);
	dim3 blocks((width + 15)/16, (height + 15)/16);
	
	convert_to_hsv<<<blocks, threads>>>(rgb, hsv, width, height);
}

__global__ void convert_to_rgb(float4 *hsv, uchar4 *rgb, int width, int height) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	
	if(x < width && y < height) {
		float4 hsv_pixel = hsv[x + width*y];
		uchar4 rgb_pixel = convert_one_pixel_to_rgb(hsv_pixel);
		rgb[x + width*y] = rgb_pixel;
	}
}

extern "C" void convert_to_rgb_wrapper(float4 *hsv, uchar4 *rgb, int width, int height) {
	dim3 threads(16,16);
	dim3 blocks((width + 15)/16, (height + 15)/16);
	
	convert_to_rgb<<<blocks, threads>>>(hsv, rgb, width, height);
}
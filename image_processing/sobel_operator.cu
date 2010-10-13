#include <cuda.h>
#include <cuda_runtime_api.h>
#include "png_helper.h"

texture <uchar4, 2, cudaReadModeElementType> tex;

typedef unsigned char uchar;

__device__ unsigned char mean(uchar x0, uchar x1, uchar x2, uchar x3, uchar x4, uchar x5, uchar x6, uchar x7, uchar x8) {
	int value = 8*x4 - x0 - x1 - x2 - x3 - x5 - x6 - x7 - x8;
	return (unsigned char) (value / 9);
}

__global__ void sobel_kernel(uchar4 *out, int width, int height) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	
	if(x < width && y < height) {
		uchar4 x0, x1, x2, x3, x4, x5, x6, x7, x8;
		x0 = tex2D(tex, x-1, y-1);
		x1 = tex2D(tex, x, y-1);
		x2 = tex2D(tex, x+1, y);
		x3 = tex2D(tex, x-1, y);
		x4 = tex2D(tex, x, y);
		x5 = tex2D(tex, x+1, y);
		x6 = tex2D(tex, x-1, y+1);
		x7 = tex2D(tex, x, y+1);
		x8 = tex2D(tex, x+1, y+1);
		
		uchar4 new_pixel;
		new_pixel.x = mean(x0.x, x1.x, x2.x, x3.x, x4.x, x5.x, x6.x, x7.x, x8.x);
		new_pixel.y = mean(x0.y, x1.y, x2.y, x3.y, x4.y, x5.y, x6.y, x7.y, x8.y);
		new_pixel.z = mean(x0.z, x1.z, x2.z, x3.z, x4.z, x5.z, x6.z, x7.z, x8.z);
		new_pixel.w = 255;
		
		out[x + y * width] = new_pixel;
	}
}

extern "C" void sobel_wrapper(uchar4 *out, png_t *info) {
	dim3 threads(16,16);
	dim3 blocks((info->width)/16 + 1, (info->height)/16 + 1);
	
	sobel_kernel<<<blocks, threads>>>(out, info->width, info->height);
}

//why must these be in here?
extern "C" struct cudaArray* setup_textures(uchar4 *in, int width, int height) {
	cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
	
	struct cudaArray *array;
	cudaMallocArray(&array, &channel_desc, width, height);
	cudaMemcpyToArray(array, 0, 0, in, width*height*sizeof(uchar4), cudaMemcpyHostToDevice);
	
	tex.addressMode[0] = cudaAddressModeClamp;
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.filterMode = cudaFilterModePoint;
	tex.normalized = false;
	
	cudaBindTextureToArray(tex, array, channel_desc);
	return array;
}

extern "C" void free_textures() {
	cudaUnbindTexture(tex);
}
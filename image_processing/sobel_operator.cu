#include <cuda.h>
#include <cuda_runtime_api.h>
#include "png_helper.h"

texture <uchar4, 2, cudaReadModeElementType> tex;

#define THRESHOLD 200
#define SECONDARY_THRESHOLD 75

__device__ float sobel(int a, int b, int c, int d, int e, int f) {
	return (float) ((a + 2*b + c) - (d + 2*e + f));
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
		
		float dfdy_r = sobel(x6.x, x7.x, x8.x, x0.x, x1.x, x2.x);
		float dfdx_r = sobel(x2.x, x5.x, x8.x, x0.x, x3.x, x6.x);
		
		float dfdy_g = sobel(x6.y, x7.y, x8.y, x0.y, x1.y, x2.y);
		float dfdx_g = sobel(x2.y, x5.y, x8.y, x0.y, x3.y, x6.y);
		
		float dfdy_b = sobel(x6.z, x7.z, x8.z, x0.z, x1.z, x2.z);
		float dfdx_b = sobel(x2.z, x5.z, x8.z, x0.z, x3.z, x6.z);
		
		float gradient_r = abs(dfdy_r) + abs(dfdy_r);
		float gradient_g = abs(dfdy_g) + abs(dfdy_g);
		float gradient_b = abs(dfdy_b) + abs(dfdy_b);
		
		float dir_r = atanf(dfdy_r/dfdx_r);
		float dir_g = atanf(dfdy_g/dfdx_g);
		float dir_b = atanf(dfdy_b/dfdx_b);
		
		uchar4 new_pixel = (uchar4) {0,0,0,255};
		if(gradient_r > THRESHOLD || gradient_g > THRESHOLD || gradient_b > THRESHOLD) {
			new_pixel.x = 255;
			new_pixel.y = 255;
			new_pixel.z = 255;
		} else if(gradient_r > SECONDARY_THRESHOLD || gradient_g > SECONDARY_THRESHOLD || gradient_b > SECONDARY_THRESHOLD) {
			new_pixel.x = 128;
			new_pixel.y = 128;
			new_pixel.z = 128;
		}
				
		out[x + y * width] = new_pixel;
	}
}

extern "C" void sobel_wrapper(struct cudaArray *in, uchar4 *out, png_t *info) {
	dim3 threads(16,16);
	dim3 blocks((info->width)/16 + 1, (info->height)/16 + 1);
	
	cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
	cudaBindTextureToArray(tex, in, channel_desc);
	
	tex.addressMode[0] = cudaAddressModeClamp;
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.filterMode = cudaFilterModePoint;
	tex.normalized = false;
	
	sobel_kernel<<<blocks, threads>>>(out, info->width, info->height);
	cudaUnbindTexture(tex);
}
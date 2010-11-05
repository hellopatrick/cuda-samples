#include <cuda.h>
#include <cuda_runtime_api.h>

__device__ uchar4 convert_one_pixel_to_rgb(float4 pixel) {
	float r, g, b;
	float h, s, v;
	
	h = pixel.x;
	s = pixel.y;
	v = pixel.z;
	
	float f = h/60.0f;
	float hi = floorf(f);
	f = f - hi;
	float p = v * (1 - s);
	float q = v * (1 - s * f);
	float t = v * (1 - s * (1 - f));
	
	if(hi == 0.0f || hi == 6.0f) {
		r = v;
		g = t;
		b = p;
	} else if(hi == 1.0f) {
		r = q;
		g = v;
		b = p;
	} else if(hi == 2.0f) {
		r = p;
		g = v;
		b = t;
	} else if(hi == 3.0f) {
		r = p;
		g = q;
		b = v;
	} else if(hi == 4.0f) {
		r = t;
		g = p;
		b = v;
	} else {
		r = v;
		g = p;
		b = q;
	}
	
	unsigned char red = (unsigned char) __float2uint_rn(255.0f * r);
	unsigned char green = (unsigned char) __float2uint_rn(255.0f * g);
	unsigned char blue = (unsigned char) __float2uint_rn(255.0f * b);
	unsigned char alpha = (unsigned char) __float2uint_rn(pixel.w);
	return (uchar4) {red, green, blue, alpha};
}

__device__ float4 convert_one_pixel_to_hsv(uchar4 pixel) {
	float r, g, b, a;
	float h, s, v;
	
	r = pixel.x / 255.0f;
	g = pixel.y / 255.0f;
	b = pixel.z / 255.0f;
	a = pixel.w;
	
	float max = fmax(r, fmax(g, b));
	float min = fmin(r, fmin(g, b));
	float diff = max - min;
	
	v = max;
	
	if(v == 0.0f) { // black
		h = s = 0.0f;
	} else {
		s = diff / v;
		if(diff < 0.001f) { // grey
			h = 0.0f;
		} else { // color
			if(max == r) {
				h = 60.0f * (g - b)/diff;
				if(h < 0.0f) { h += 360.0f; }
			} else if(max == g) {
				h = 60.0f * (2 + (b - r)/diff);
			} else {
				h = 60.0f * (4 + (r - g)/diff);
			}
		}		
	}
	
	return (float4) {h, s, v, a};
}

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


__global__ void compute_hue_histogram(float4 *hsv, unsigned int *histogram, int width, int height) {
	__shared__ unsigned int cached_histogram[360];
	
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	
	cached_histogram[threadIdx.x] = 0;
	
	if(x < width && y < height) {
		float h = hsv[x + y*width].x;
		int hue = max(0, min(359, (int) h));
		atomicAdd(&(cached_histogram[hue]), 1);
	}
	__syncthreads();
	
	atomicAdd(&(histogram[threadIdx.x]), cached_histogram[threadIdx.x]);
}

__global__ void draw_histogram(unsigned int *histogram, uchar4 *image) {
	__shared__ unsigned int max;

	int x = threadIdx.x;
	
	unsigned int count = histogram[x];
	if(x == 0) { max = count;}
	
	atomicMax(&max, count);
	__syncthreads();
	unsigned int scaled_size = (250 * count) / max;
	uchar4 rgb_pixel = convert_one_pixel_to_rgb((float4) {(float) x, 1.0f, 1.0f, 255.0f});
	uchar4 black = (uchar4) {0,0,0,255};
	
	uchar4 *paintbrush = &rgb_pixel;
	
	int y;
	for(y = 0; y < 256; y++) {
		image[x + (255-y)*360] = *paintbrush;
		if(y > scaled_size) { paintbrush = &black; }
	}
}

extern "C" void compute_histogram_image(float4 *hsv, unsigned int *histogram, uchar4 *image, int width, int height) {
	dim3 threads(360,1);
	dim3 blocks((width + 359)/360, height);
	
	compute_hue_histogram<<<blocks, threads>>>(hsv, histogram, width, height);
	draw_histogram<<<1, threads>>>(histogram, image);
}
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "png_helper.h"

extern void convert_to_hsv_wrapper(uchar4 *rgb, float4 *hsv, int width, int height);
extern void convert_to_rgb_wrapper(float4 *hsv, uchar4 *rgb, int width, int height);
extern void compute_histogram_image_wrapper(float4 *hsv, unsigned int *histogram, uchar4 *image, int width, int height);

int main (int argc, char* argv[]) {
	// host
	uchar4 *host_image, *host_out;
	png_t *info;
	
	// device
	uchar4 *image, *histogram_image;
	float4 *hsv;
	unsigned int *histogram;
	
	// timing
	cudaEvent_t	start, stop;
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	if(argc < 2) {
		printf("Must include file name to process. `%s <file_name>`\n", argv[0]);
		return -1;
	}
	
	if(read_png(argv[1], &info, &host_image) == PNG_FAILURE) {
		printf("Error reading file (%s).\n", argv[1]);
		return -1;
	}
	
	size_t number_of_bytes_rgb = sizeof(uchar4) * info->width * info->height;
	size_t number_of_bytes_hsv = sizeof(float4) * info->width * info->height;
	size_t number_of_bytes_hist_img = sizeof(uchar4) * 360 * 256;
	size_t number_of_bytes_histogram = sizeof(unsigned int) * 360;
	
	host_out = malloc(number_of_bytes_hist_img);
	
	cudaMalloc((void **) &image, number_of_bytes_rgb);	
	cudaMalloc((void **) &hsv, number_of_bytes_hsv);	
	cudaMalloc((void **) &histogram, number_of_bytes_histogram);
	cudaMalloc((void **) &histogram_image, number_of_bytes_hist_img);
	assert(cudaGetLastError() == cudaSuccess);
	
	cudaEventRecord(start, 0);
	cudaMemset(histogram, 0, number_of_bytes_histogram);
	cudaMemcpy(image, host_image, number_of_bytes_rgb, cudaMemcpyHostToDevice);
	assert(cudaGetLastError() == cudaSuccess);
	
	convert_to_hsv_wrapper(image, hsv, info->width, info->height);	
	compute_histogram_image_wrapper(hsv, histogram, histogram_image, info->width, info->height);
	cudaEventRecord(stop, 0);
	
	cudaMemcpy(host_out, histogram_image, number_of_bytes_hist_img, cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);
	
	float elapsed_time;
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("Time to compute histogram & make image with GPU: %3.1f ms\n", elapsed_time);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	char *output_file;	
	if(argc > 3) { output_file = argv[2]; } 
	else { output_file = "data/out.png"; }
	info->width = 360;
	info->height = 256;
	if(write_png(output_file, info, host_out) == PNG_FAILURE) {
		printf("Error writing to file (%s)\n", output_file);
	}
	
	cudaFree(hsv);
	cudaFree(image);
	cudaFree(histogram_image);
	cudaFree(histogram);
	
	free(host_image);
	free(host_out);
	return 0;
}
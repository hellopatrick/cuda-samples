#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "png_helper.h"

extern void convert_to_hsv_wrapper(uchar4 *rgb, float4 *hsv, int width, int height);
extern void convert_to_rgb_wrapper(float4 *hsv, uchar4 *rgb, int width, int height);
extern void compute_histogram_image(float4 *hsv, unsigned int *histogram, uchar4 *image, int width, int height);
int main (int argc, char* argv[]) {
	// host variables
	uchar4 *host_in, *host_out;
	unsigned int *host_histogram;
	
	// device variables
	uchar4 *image, *histogram_image;
	float4 *hsv;
	unsigned int *histogram;
	
	// png variables
	png_t *info;
	
	if(argc < 2) {
		printf("Must include file name to process. `%s <file_name>`\n", argv[0]);
		return -1;
	}
	
	if(read_png(argv[1], &info, &host_in) == PNG_FAILURE) {
		printf("Error reading file (%s).\n", argv[1]);
		return -1;
	}
	
	size_t number_of_bytes_rgb = sizeof(uchar4) * info->width * info->height;
	size_t number_of_bytes_hsv = sizeof(float4) * info->width * info->height;
	size_t number_of_bytes_hist_img = sizeof(uchar4) * 360 * 256;
	size_t number_of_bytes_histogram = sizeof(unsigned int) * 360;
	
	host_out = malloc(number_of_bytes_hist_img);
	host_histogram = malloc(number_of_bytes_histogram);
	
	cudaMalloc((void **) &image, number_of_bytes_rgb);	
	cudaMalloc((void **) &hsv, number_of_bytes_hsv);	
	cudaMalloc((void **) &histogram, number_of_bytes_histogram);
	assert(cudaGetLastError() == cudaSuccess);
	
	cudaMemset(histogram, 0, number_of_bytes_histogram);
	cudaMemcpy(image, host_in, number_of_bytes_rgb, cudaMemcpyHostToDevice);
	assert(cudaGetLastError() == cudaSuccess);
	
	convert_to_hsv_wrapper(image, hsv, info->width, info->height);	
	cudaFree(image); // free up this space.
	
	cudaMalloc((void **) &histogram_image, number_of_bytes_hist_img);
	
	compute_histogram_image(hsv, histogram, histogram_image, info->width, info->height);
	cudaMemcpy(host_out, histogram_image, number_of_bytes_hist_img, cudaMemcpyDeviceToHost);
	
	cudaFree(hsv);
	cudaFree(histogram_image);
	cudaFree(histogram);
	
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
	cudaFree(histogram);
	free(host_in);
	free(host_out);
	free(host_histogram);

	return 0;
}
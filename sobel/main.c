#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "png_helper.h"

extern struct cudaArray* setup_cuda_array(uchar4 *in, int width, int height);
extern void sobel_wrapper(struct cudaArray *in, uchar4 *out, png_t *info);

int main (int argc, char* argv[]) {
	uchar4 *host_in, *host_out, *host_golden;
	
	struct cudaArray *dev_in;
	uchar4 *dev_out;
	
	png_t *info;
	
	if(argc < 2) {
		printf("Must include file name to process. `%s <file_name>`\n", argv[0]);
		return -1;
	}
	
	if(read_png(argv[1], &info, &host_in) == PNG_FAILURE) {
		printf("Error reading file (%s).\n", argv[1]);
		return -1;
	}
		
	size_t number_of_bytes = sizeof(uchar4) * info->width * info->height;
	host_out = malloc(number_of_bytes);
	host_golden = malloc(number_of_bytes);
	
	cudaMalloc((void **) &dev_out, number_of_bytes);	
	dev_in = setup_cuda_array(host_in, info->width, info->height);
	assert(cudaGetLastError() == cudaSuccess);
	
	sobel_wrapper(dev_in, dev_out, info);
	cudaMemcpy(host_out, dev_out, number_of_bytes, cudaMemcpyDeviceToHost);
	
	char *output_file;
	
	if(argc > 3) {
		output_file = argv[2];
	} else {
		output_file = "data/out.png";
	}
	
	if(write_png(output_file, info, host_out) == PNG_FAILURE) {
		printf("Error writing to file (%s)\n", output_file);
	}
	
	cudaFree(dev_in);
	cudaFree(dev_out);
	return 0;
}
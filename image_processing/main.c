#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#ifdef __APPLE__
	#include <OpenGL/gl.h>
	#include <GLUT/glut.h>
#else
	#include <GL/gl.h>
	#include <GL/glut.h>
#endif

#include "png_helper.h"

extern void naive_transpose_wrapper(uchar4 *in, uchar4 *out, png_t *info);
extern void cpu_transpose(uchar4 *in, uchar4 *out, png_t *info);

extern struct cudaArray* setup_textures(uchar4 *in, int width, int height);
void free_textures();

extern void sobel_wrapper(uchar4 *out, png_t *info);

int main (int argc, char const *argv[]) {
	uchar4 *host_in, *host_out, *host_golden;
	
	struct cudaArray *dev_in;
	uchar4 *dev_out;
	
	png_t *info;
	
	read_png("data/in.png", &info, &host_in);
		
	size_t number_of_bytes = sizeof(uchar4) * info->width * info->height;
	host_out = malloc(number_of_bytes);
	host_golden = malloc(number_of_bytes);
	
	cudaMalloc((void **) &dev_out, number_of_bytes);
	assert(cudaGetLastError() == cudaSuccess);
	
	dev_in = setup_textures(host_in, info->width, info->height);
	assert(cudaGetLastError() == cudaSuccess);
	
	sobel_wrapper(dev_out, info);
	cudaMemcpy(host_out, dev_out, number_of_bytes, cudaMemcpyDeviceToHost);
	
	write_png("data/gpu.png", info, host_out);
	
	cudaFree(dev_in);
	cudaFree(dev_out);
	return 0;
}
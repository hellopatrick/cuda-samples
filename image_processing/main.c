#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

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

#define L(x,y) x + y * info->width

int main (int argc, char const *argv[]) {
	uchar4 *pixels;
	png_t *info;
	
	read_png("test.png", &info, &pixels);
	
	int x, y;
	
	printf("size = %d x %d\n", info->width, info->height);
	printf("color_type = %d\n", info->color_type);
	printf("bit_depth = %d\n", info->bit_depth);
	
	for(x = 0; x < info->width; x++) {
		for(y = 0; y < info->height; y++) {
			pixels[L(x,y)].x = 255;
		}
	}
	write_png("test_write.png", info, pixels);
	return 0;
}
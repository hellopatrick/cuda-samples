#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <string.h>

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

#include "dimensions.h"

int *host_current, *host_future;
int *gpu_current, *gpu_future;
uchar4 *pixels;

GLuint bufferObj;
struct cudaGraphicsResource *resource;

void cached_game_of_life_wrapper(int *current, int *future, uchar4 *pixels);
void fill_board(int *board, int percent);

void display() {
	int x, y;
	
	glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels(DIM_X, DIM_Y, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glutSwapBuffers();
}

void idle() {
	size_t size;
	cudaGraphicsMapResources(1, &resource, NULL);
	cudaGraphicsResourceGetMappedPointer((void **) &pixels, &size, resource);
	
	cached_game_of_life_wrapper(gpu_current, gpu_future, pixels);
	
	int *temp = gpu_current;
	gpu_current = gpu_future;
	gpu_future = temp;
	
	cudaGraphicsUnmapResources(1, &resource, NULL);
	glutPostRedisplay();
}

int main(int argc, char **argv) {
	struct cudaDeviceProp prop;
	int dev;
	
	memset(&prop, 0, sizeof(struct cudaDeviceProp));
	prop.major = 1; 
	prop.minor = 0; 
	cudaChooseDevice(&dev, &prop);
	cudaGLSetGLDevice(dev);
	assert(cudaGetLastError() == cudaSuccess);
	
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(DIM_X,DIM_Y);
  glutCreateWindow("cuda");

	glGenBuffers(1, &bufferObj);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, DIM_X * DIM_Y * 4, NULL, GL_DYNAMIC_DRAW_ARB);
	cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone);
	assert(cudaGetLastError() == cudaSuccess);
	
  glutDisplayFunc(display);
	glutIdleFunc(idle);

	cudaMallocHost((void**) &host_current, DIM_X * DIM_Y * sizeof(int));
	cudaMallocHost((void**) &host_future, DIM_X * DIM_Y * sizeof(int));	
	assert(cudaGetLastError() == cudaSuccess);
	
	cudaMalloc((void**) &gpu_current, DIM_X * DIM_Y * sizeof(int));
	cudaMalloc((void**) &gpu_future, DIM_X * DIM_Y * sizeof(int));
	printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	assert(cudaGetLastError() == cudaSuccess);
	
	fill_board(host_current, 40);	
	cudaMemcpy(gpu_current, host_current, DIM_X * DIM_Y * sizeof(int), cudaMemcpyHostToDevice);
	
	glutMainLoop();
	
	printf("Freeing memory...");

	cudaFree(host_future);
	cudaFree(host_current);
	cudaFree(gpu_current);
	cudaFree(gpu_future);
	
	return 0;
}

void fill_board(int *board, int percent) {
	int used_dots = 0;
	srand(time(NULL));
	for(int y = 0; y < DIM_Y; y++) {
		for(int x = 0; x < DIM_X; x++) {
			if( rand() % 100 < percent) { board[y*DIM_X + x] = 1; }
			else { board[y*DIM_X + x] = 0; }
		}
	}
}
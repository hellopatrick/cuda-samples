#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#ifdef __APPLE__
    #include <OpenGL/gl.h>
    #include <GLUT/glut.h>
#else
	#include <GL/gl.h>
    #include <GL/glut.h>
#endif

#include "dimensions.h"

#define STEPS 100

int *host_current, *host_future, *host_future_naive, *host_future_cached;
int *gpu_current, *gpu_future;

uchar4 *pixels;
clock_t start, stop;

void cached_game_of_life_wrapper(int *current, int *future);
void fill_board(int *board, int percent);

void add_glider(int *board) {
	for(int y = 0; y < DIM_Y; y++) {
		for(int x = 0; x < DIM_X; x++) {
			board[y * DIM_X + x] = 0;
		}
	}
	
	board[2 * DIM_X + 1] = 1;
	board[3 * DIM_X + 2] = 1;
	board[1 * DIM_X + 3] = 1;
	board[2 * DIM_X + 3] = 1;
	board[3 * DIM_X + 3] = 1;
}

cudaError_t error;

void display() {
	int x, y;
	
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glBegin(GL_POINTS);
	glColor3f(1.0,0.0,0.0);
	
	for(x = 0; x < DIM_X; x++){
		for(y = 0; y < DIM_Y; y++) {
			if(host_future[y*DIM_X + x] == 1) { 
				glVertex2i(x, y);
			}
		}
	}
	glEnd();
	glFlush();
}

void idle() {
	cached_game_of_life_wrapper(gpu_current, gpu_future);
	cudaMemcpy(host_future, gpu_future, DIM_X * DIM_Y * sizeof(int), cudaMemcpyDeviceToHost);
//	update_board(host_current, host_future);
	
	int *temp = gpu_current;
	gpu_current = gpu_future;
	gpu_future = temp;
	glutPostRedisplay();
}

int main(int argc, char **argv) {	
	glutInit(&argc, argv);
	glutInitWindowSize(DIM_X,DIM_Y);
    glutCreateWindow(argv[0]);
    glutDisplayFunc(display);
	glutIdleFunc(idle);
	
	gluOrtho2D(0, DIM_X, DIM_Y, 0);
//	glEnable(GL_BLEND);
//	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	
	cudaMallocHost((void**) &host_current, DIM_X * DIM_Y * sizeof(int));
	cudaMallocHost((void**) &host_future, DIM_X * DIM_Y * sizeof(int));	
	cudaMallocHost((void**) &host_future_naive, DIM_X * DIM_Y * sizeof(int));
	cudaMallocHost((void**) &host_future_cached, DIM_X * DIM_Y * sizeof(int));
	assert(cudaGetLastError() == cudaSuccess);
	
	cudaMalloc((void**) &gpu_current, DIM_X * DIM_Y * sizeof(int));
	cudaMalloc((void**) &gpu_future, DIM_X * DIM_Y * sizeof(int));
	printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	assert(cudaGetLastError() == cudaSuccess);
	
//	add_glider(host_current);
	fill_board(host_current, 40);	
	cudaMemcpy(gpu_current, host_current, DIM_X * DIM_Y * sizeof(int), cudaMemcpyHostToDevice);
	
	glutMainLoop();
	
	printf("Freeing memory...");

	cudaFree(host_future);
	cudaFree(host_future_naive);
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
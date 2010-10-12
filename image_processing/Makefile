PROJECT_NAME = image_processing

# NVCC is path to nvcc. Here it is assumed /usr/local/cuda is on one's PATH.
# CC is the compiler for C++ host code.

NVCC = nvcc
CC = clang

CUDAPATH = /usr/local/cuda

BUILD_DIR = build
# note that nvcc defaults to 32-bit architecture. thus, force C/LFLAGS to comply.
# you could also force nvcc to compile 64-bit with -m64 flag. (and remove -m32 instances)

CFLAGS = -c -m32 -I$(CUDAPATH)/include -I/usr/X11/include
NVCCFLAGS = -c -I$(CUDAPATH)/include

# this is mac only. only linux system, remove -Xlinker and add in -lGL -lGLUT?
LFLAGS = -m32 -L$(CUDAPATH)/lib -L/usr/X11/lib -lcuda -lcudart -lm -lpng -lz -Xlinker -framework,OpenGL,-framework,GLUT

all: build clean

build: build_dir gpu cpu
	$(NVCC) $(LFLAGS) -o $(BUILD_DIR)/$(PROJECT_NAME) *.o

build_dir:
	mkdir -p $(BUILD_DIR)

gpu:
	$(NVCC) $(NVCCFLAGS) *.cu

cpu:			
	$(CC) $(CFLAGS) *.c

clean:
	rm *.o

run:
	./$(BUILD_DIR)/$(PROJECT_NAME)
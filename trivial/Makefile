PROJECT_NAME = trivial

# NVCC is path to nvcc. Here it is assumed /usr/local/cuda is on one's PATH.
# CC is the compiler for C++ host code.

NVCC = nvcc
CC = g++

CUDAPATH = /usr/local/cuda

BUILD_DIR = build
# note that nvcc defaults to 32-bit architecture. thus, force C/LFLAGS to comply.
# you could also force nvcc to compile 64-bit with -m64 flag. (and remove -m32 instances)

CFLAGS = -c -m32 -I$(CUDAPATH)/include
NVCCFLAGS = -c -I$(CUDAPATH)/include
LFLAGS = -m32 -L$(CUDAPATH)/lib -lcuda -lcudart -lm

all: build clean

build: build_dir gpu cpu
	$(CC) $(LFLAGS) -o $(BUILD_DIR)/$(PROJECT_NAME) *.o

build_dir:
	mkdir -p $(BUILD_DIR)

gpu:
	$(NVCC) $(NVCCFLAGS) *.cu

cpu:			
	$(CC) $(CFLAGS) *.cpp

clean:
	rm *.o

run:
	./$(BUILD_DIR)/$(PROJECT_NAME)
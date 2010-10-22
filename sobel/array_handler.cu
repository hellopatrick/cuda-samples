#include <cuda.h>
#include <cuda_runtime_api.h>

extern "C" struct cudaArray* setup_cuda_array(uchar4 *in, int width, int height) {
	cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
	
	struct cudaArray *array;
	cudaMallocArray(&array, &channel_desc, width, height);
	cudaMemcpyToArray(array, 0, 0, in, width*height*sizeof(uchar4), cudaMemcpyHostToDevice);

	return array;
}
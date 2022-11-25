#include <cuda_runtime.h>

struct EncodeKernelData {
	double* dev_factors;
	char* dev_hash;
	int width;
	int height;
	double maximumValue;
};

char* runEncodeKernel(double* dev_factors, int width, int height, int compX, int compY, double maximumValue, int hashSize);
__global__ void encodeKernel(EncodeKernelData data);
__host__ __device__ int sign(double n);
__host__ __device__ double signPow(double val, double exp);

char* runEncodeKernel(double* dev_factors, int width, int height, int compX, int compY, double maximumValue, int hashSize) {
	char* dev_hash = 0;
	cudaError_t cudaStatus;

	int encodeThreads = compX * compY - 1;

	cudaStatus = cudaMalloc((void**)&dev_hash, hashSize * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	EncodeKernelData encodeData;
	encodeData.dev_factors = dev_factors;
	encodeData.dev_hash = dev_hash;
	encodeData.height = height;
	encodeData.width = width;
	encodeData.maximumValue = maximumValue;


	// run encode kernel
	encodeKernel << <1, encodeThreads >> > (encodeData);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	return dev_hash;

Error:
	cudaFree(dev_hash);

	return nullptr;
}

__global__ void encodeKernel(EncodeKernelData data) {
	int threadIndex = threadIdx.x;
	int valueIndex = (threadIndex + 1) * 3;

	// scale factors
	double scale = 1.0 / (data.width * data.height);
	data.dev_factors[valueIndex] *= scale;
	data.dev_factors[valueIndex + 1] *= scale;
	data.dev_factors[valueIndex + 2] *= scale;

	int quantR = max(0, min(18, (int)floor(signPow(data.dev_factors[valueIndex] / data.maximumValue, 0.5) * 9 + 9.5)));
	int quantG = max(0, min(18, (int)floor(signPow(data.dev_factors[valueIndex + 1] / data.maximumValue, 0.5) * 9 + 9.5)));
	int quantB = max(0, min(18, (int)floor(signPow(data.dev_factors[valueIndex + 2] / data.maximumValue, 0.5) * 9 + 9.5)));

	int quant = quantR * 19 * 19 + quantG * 19 + quantB;
	encode83(data.dev_hash, 6 + (threadIndex) * 2, quant, 2);
}

__host__ __device__ int sign(double n) {
	return n < 0 ? -1 : 1;
}

__host__ __device__ double signPow(double val, double exp) {
	return sign(val) * pow(abs(val), exp);
}

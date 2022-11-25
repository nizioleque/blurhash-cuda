#include <cuda_runtime.h>
#include <chrono>

#define PI 3.14159265358979323846
const int PIXELS_PER_BLOCK = 512;

using namespace std::chrono;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
static __inline__ __device__ double atomicAdd(double *address, double val)
{
	unsigned long long int *address_as_ull = (unsigned long long int *)address;
	unsigned long long int old = *address_as_ull, assumed;
	if (val == 0.0)
		return __longlong_as_double(old);
	do
	{
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}
#endif

struct FactorKernelData
{
	unsigned char *dev_img;
	double *dev_factors;
	int imagePixelsCount;
	int width;
	int height;
	int compX;
	int compY;
};

double *runFactorKernel(unsigned char *img, int width, int height, int compX, int compY);
__global__ void factorKernel(FactorKernelData data);
__device__ void multiplyBasisFunction(int x, int y, int threadX, int threadY, FactorKernelData *data, double basis);
__device__ double sRGBToLinear(unsigned char value);

double *runFactorKernel(unsigned char *img, int width, int height, int compX, int compY)
{
	unsigned char *dev_img = 0;
	double *dev_factors = 0;
	cudaError_t cudaStatus;

	int imagePixelsCount = width * height;
	int imageArraySize = imagePixelsCount * 3;
	int blocks = (imagePixelsCount - 1) / PIXELS_PER_BLOCK + 1;

	// Allocate memory for pixel data
	cudaStatus = cudaMalloc((void **)&dev_img, imageArraySize * sizeof(char));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy pixel data to GPU memory
	// auto copyStart = high_resolution_clock::now();
	cudaStatus = cudaMemcpy(dev_img, img, imageArraySize * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	// auto copyEnd = high_resolution_clock::now();
	// std::cout << "Image CPU->GPU copy time: " << duration_cast<milliseconds>(copyEnd - copyStart).count() << " ms \n";

	// Allocate memory for factors
	cudaStatus = cudaMalloc((void **)&dev_factors, compX * compY * 3 * sizeof(double));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc (dev_factors) failed!");
		goto Error;
	}

	// Fill with zeros
	cudaStatus = cudaMemset((void *)dev_factors, 0, compX * compY * 3 * sizeof(double));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemset (dev_factors) failed!");
		goto Error;
	}

	FactorKernelData data;
	data.dev_img = dev_img;
	data.dev_factors = dev_factors;
	data.imagePixelsCount = imagePixelsCount;
	data.height = height;
	data.width = width;
	data.compX = compX;
	data.compY = compY;

	// run kernel
	// auto kernelStart = high_resolution_clock::now();
	factorKernel<<<blocks, PIXELS_PER_BLOCK>>>(data);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// auto kernelEnd = high_resolution_clock::now();
	// std::cout << "Factor calculation (kernel) time: " << duration_cast<milliseconds>(kernelEnd - kernelStart).count() << " ms \n";

	cudaFree(dev_img);
	return dev_factors;

Error:
	cudaFree(dev_factors);
	cudaFree(dev_img);

	return nullptr;
}

__global__ void factorKernel(FactorKernelData data)
{
	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadIndex >= data.imagePixelsCount)
		return;

	int threadPixelX = threadIndex % data.width;
	int threadPixelY = threadIndex / data.width;

	// calculate factors
	for (int currentCompY = 0; currentCompY < data.compY; currentCompY++)
	{
		for (int currentCompX = 0; currentCompX < data.compX; currentCompX++)
		{
			int normalisation = currentCompX == 0 && currentCompY == 0 ? 1 : 2;
			double basis = normalisation *
						   cos((PI * currentCompX * threadPixelX) / (double)data.width) *
						   cos((PI * currentCompY * threadPixelY) / (double)data.height);
			multiplyBasisFunction(
				currentCompX,
				currentCompY,
				threadPixelX,
				threadPixelY,
				&data,
				basis);
		}
	}
}

__device__ void multiplyBasisFunction(int currentCompX, int currentCompY, int threadX, int threadY, FactorKernelData *data, double basis)
{
	int factorsArrayIndex = (currentCompY * data->compX + currentCompX) * 3;
	int basePixelIndex = (threadY * data->width + threadX) * 3;

	atomicAdd(data->dev_factors + factorsArrayIndex, basis * sRGBToLinear(data->dev_img[basePixelIndex]));
	atomicAdd(data->dev_factors + factorsArrayIndex + 1, basis * sRGBToLinear(data->dev_img[basePixelIndex + 1]));
	atomicAdd(data->dev_factors + factorsArrayIndex + 2, basis * sRGBToLinear(data->dev_img[basePixelIndex + 2]));
}

__device__ double sRGBToLinear(unsigned char value)
{
	double v = value / 255.0;
	if (v <= 0.04045)
	{
		return v / 12.92;
	}
	else
	{
		return pow((v + 0.055) / 1.055, 2.4);
	}
}
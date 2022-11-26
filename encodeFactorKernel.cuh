#include <cuda_runtime.h>
#include <chrono>

#define PI 3.14159265358979323846
const int PIXELS_PER_BLOCK = 1024;

using namespace std::chrono;

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
	_V2::system_clock::time_point copyStart, copyEnd, kernelStart, kernelEnd;

	int imagePixelsCount = width * height;
	int componentCount = compX * compY;
	int imageArraySize = imagePixelsCount * 3;
	int factorArraySize = imagePixelsCount * 3 * componentCount;
	int threadCount = imagePixelsCount * componentCount;
	int blocks = (threadCount - 1) / PIXELS_PER_BLOCK + 1;

	// Allocate memory for pixel data
	cudaStatus = cudaMalloc((void **)&dev_img, imageArraySize * sizeof(char));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy pixel data to GPU memory
	copyStart = high_resolution_clock::now();
	cudaStatus = cudaMemcpy(dev_img, img, imageArraySize * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	copyEnd = high_resolution_clock::now();
	std::cout << "Image CPU->GPU copy time: " << duration_cast<milliseconds>(copyEnd - copyStart).count() << " ms \n";

	// Allocate memory for factors
	cudaStatus = cudaMalloc((void **)&dev_factors, factorArraySize * sizeof(double));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc (dev_factors) failed!");
		goto Error;
	}

	// Fill with zeros
	cudaStatus = cudaMemset((void *)dev_factors, 0, factorArraySize * sizeof(double));
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
	kernelStart = high_resolution_clock::now();
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

	kernelEnd = high_resolution_clock::now();
	std::cout << "Factor calculation (kernel) time: " << duration_cast<milliseconds>(kernelEnd - kernelStart).count() << " ms \n";

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

	if (threadIndex >= data.imagePixelsCount * data.compX * data.compY)
		return;

	int imgPixelCount = data.width * data.height;
	int componentIndex = threadIndex / imgPixelCount;
	int pixelIndex = threadIndex % imgPixelCount;

	int componentY = componentIndex / data.compX;
	int componentX = componentIndex % data.compX;

	int pixelY = pixelIndex / data.width;
	int pixelX = pixelIndex % data.width;

	int normalisation = componentX == 0 && componentY == 0 ? 1 : 2;
	double basis = normalisation *
				   cos((PI * componentX * pixelX) / (double)data.width) *
				   cos((PI * componentY * pixelY) / (double)data.height);
	multiplyBasisFunction(componentX, componentY, pixelX, pixelY, &data, basis);
}

__device__ void multiplyBasisFunction(int componentX, int componentY, int pixelX, int pixelY, FactorKernelData *data, double basis)
{
	int componentIndex = (componentY * data->compX + componentX) * data->width * data->height * 3;
	int pixelR = pixelY * data->width + pixelX;
	int pixelG = pixelR + data->width * data->height;
	int pixelB = pixelG + data->width * data->height;
	int basePixelIndex = (pixelY * data->width + pixelX) * 3;

	data->dev_factors[componentIndex + pixelR] = basis * sRGBToLinear(data->dev_img[basePixelIndex]);
	data->dev_factors[componentIndex + pixelG] = basis * sRGBToLinear(data->dev_img[basePixelIndex + 1]);
	data->dev_factors[componentIndex + pixelB] = basis * sRGBToLinear(data->dev_img[basePixelIndex + 2]);
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
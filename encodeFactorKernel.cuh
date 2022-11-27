#include <cuda_runtime.h>
#include <chrono>

#define PI 3.14159265358979323846

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
	int blocks = componentCount * height;

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
	factorKernel<<<blocks, width>>>(data);

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
	int pixelX = threadIdx.x;
	int rowIndex = blockIdx.x;
	int componentCount = data.compX * data.compY;
	int componentIndex = rowIndex / data.height;
	int pixelY = rowIndex % data.height;
	int colorOffset = componentCount * data.height * data.width;

	if (componentIndex >= componentCount || pixelY >= data.height)
		return;

	int componentY = componentIndex / data.compX;
	int componentX = componentIndex % data.compX;

	int normalisation = componentX == 0 && componentY == 0 ? 1 : 2;
	double basis = normalisation *
				   cos((PI * componentX * pixelX) / (double)data.width) *
				   cos((PI * componentY * pixelY) / (double)data.height);
	multiplyBasisFunction(componentX, componentY, pixelX, pixelY, &data, basis);

	// sum values in each image row
	for (int dist = 1; dist < data.width; dist *= 2)
	{
		if (pixelX % dist != 0)
			return;
		if (pixelX + dist >= data.width)
			return;

		int baseIndex = rowIndex * data.width + pixelX;

		__syncthreads();

		data.dev_factors[baseIndex + 0 * colorOffset] += data.dev_factors[baseIndex + 0 * colorOffset + dist];
		data.dev_factors[baseIndex + 1 * colorOffset] += data.dev_factors[baseIndex + 1 * colorOffset + dist];
		data.dev_factors[baseIndex + 2 * colorOffset] += data.dev_factors[baseIndex + 2 * colorOffset + dist];

		__syncthreads();
	}
}

__device__ void multiplyBasisFunction(int componentX, int componentY, int pixelX, int pixelY, FactorKernelData *data, double basis)
{
	int componentStart = (componentY * data->compX + componentX) * data->width * data->height;
	int colorOffset = data->compX * data->compY * data->height * data->width;
	int basePixelIndex = (pixelY * data->width + pixelX);

	data->dev_factors[componentStart + basePixelIndex + 0 * colorOffset] = basis * sRGBToLinear(data->dev_img[basePixelIndex * 3 + 0]);
	data->dev_factors[componentStart + basePixelIndex + 1 * colorOffset] = basis * sRGBToLinear(data->dev_img[basePixelIndex * 3 + 1]);
	data->dev_factors[componentStart + basePixelIndex + 2 * colorOffset] = basis * sRGBToLinear(data->dev_img[basePixelIndex * 3 + 2]);
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
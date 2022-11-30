#include <cuda_runtime.h>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std::chrono;

struct DevFactors
{
	double *r;
	double *g;
	double *b;
};

struct FactorKernelData
{
	unsigned char *dev_img;
	DevFactors dev_factors;
	int imagePixelsCount;
	int width;
	int height;
	int compX;
	int compY;
};

DevFactors runFactorKernel(unsigned char *img, int width, int height, int compX, int compY);
__global__ void factorKernel(FactorKernelData data);
__device__ void multiplyBasisFunction(int componentX, int componentY, int pixelX, int pixelY, FactorKernelData *data, double basis, double *dev_factors_current, int colorIndex);
__device__ double sRGBToLinear(unsigned char value);

DevFactors runFactorKernel(unsigned char *img, int width, int height, int compX, int compY)
{
	unsigned char *dev_img = 0;
	double *dev_factors_r = 0, *dev_factors_g = 0, *dev_factors_b = 0;
	cudaError_t cudaStatus;
	_V2::system_clock::time_point copyStart, copyEnd, kernelStart, kernelEnd;

	int imagePixelsCount = width * height;
	int componentCount = compX * compY;
	int imageArraySize = imagePixelsCount * 3;
	int factorArraySize = imagePixelsCount * componentCount;
	int blocks = componentCount * height * 3;
	DevFactors dev_factors;

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
	cudaStatus = cudaMalloc((void **)&dev_factors_r, factorArraySize * sizeof(double));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc (dev_factors) failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void **)&dev_factors_g, factorArraySize * sizeof(double));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc (dev_factors) failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void **)&dev_factors_b, factorArraySize * sizeof(double));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc (dev_factors) failed!");
		goto Error;
	}

	dev_factors.r = dev_factors_r;
	dev_factors.g = dev_factors_g;
	dev_factors.b = dev_factors_b;

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

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	kernelEnd = high_resolution_clock::now();
	std::cout << "Factor calculation time: " << duration_cast<milliseconds>(kernelEnd - kernelStart).count() << " ms \n";

	cudaFree(dev_img);
	return dev_factors;

Error:
	cudaFree(dev_factors.r);
	cudaFree(dev_factors.g);
	cudaFree(dev_factors.b);
	cudaFree(dev_img);

	return {};
}

__global__ void factorKernel(FactorKernelData data)
{
	int pixelX = threadIdx.x;
	int rowIndex = blockIdx.x / 3;
	int colorIndex = blockIdx.x % 3;
	int componentCount = data.compX * data.compY;
	int componentIndex = rowIndex / data.height;
	int pixelY = rowIndex % data.height;

	bool working = !((componentIndex >= componentCount || pixelY >= data.height));

	double *dev_factors_current = 0;
	switch (colorIndex)
	{
	case 0:
		dev_factors_current = data.dev_factors.r;
		break;
	case 1:
		dev_factors_current = data.dev_factors.g;
		break;
	case 2:
		dev_factors_current = data.dev_factors.b;
		break;
	}

	if (working)
	{
		int componentY = componentIndex / data.compX;
		int componentX = componentIndex % data.compX;

		int normalisation = componentX == 0 && componentY == 0 ? 1 : 2;
		double basis = normalisation *
					   cos((M_PI * componentX * pixelX) / (double)data.width) *
					   cos((M_PI * componentY * pixelY) / (double)data.height);
		multiplyBasisFunction(componentX, componentY, pixelX, pixelY, &data, basis, dev_factors_current, colorIndex);
	}

	// sum values in each image row
	for (int dist = 1; dist < data.width; dist *= 2)
	{
		__syncthreads();

		if (working)
		{
			if (pixelX % (dist * 2) != 0 || pixelX + dist >= data.width)
				continue;

			int baseIndex = rowIndex * data.width + pixelX;

			dev_factors_current[baseIndex] += dev_factors_current[baseIndex + dist];
		}
	}
}

__device__ void multiplyBasisFunction(int componentX, int componentY, int pixelX, int pixelY, FactorKernelData *data, double basis, double *dev_factors_current, int colorIndex)
{
	int componentStart = (componentY * data->compX + componentX) * data->width * data->height;
	int basePixelIndex = (pixelY * data->width + pixelX);

	dev_factors_current[componentStart + basePixelIndex] = basis * sRGBToLinear(data->dev_img[basePixelIndex * 3 + colorIndex]);
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
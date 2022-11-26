#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include "base83.cuh"
#include "encodeFactorKernel.cuh"
#include "encodeKernel.cuh"
#include "encodeHashStart.cuh"
#include "encodeFactorSumKernel.cuh"

using namespace std::chrono;

int encode(const char *filename, int compX, int compY);
double *getFactorsFromDevice(double *dev_factors, int compX, int compY);
void calculateMaximumValue(double &maximumValue, int &quantisedMaximumValue, double *factors, int width, int height, int compX, int compY, double scale);

int encode(const char *filename, int compX, int compY)
{
	// read image
	int width, height;
	auto readStart = high_resolution_clock::now();
	unsigned char *img = stbi_load(filename, &width, &height, nullptr, 3);
	auto readEnd = high_resolution_clock::now();
	if (img == NULL)
	{
		printf("Error in loading the image\n");
		return 1;
	}

	std::cout << "Image read time: " << duration_cast<milliseconds>(readEnd - readStart).count() << " ms \n";
	printf("Loaded image with a width of %dpx, a height of %dpx\n", width, height);

	if (compX < 1 || compX > 9 || compY < 1 || compY > 9)
	{
		printf("Blurhash must have between 1 and 9 components\n");
		return 1;
	}

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return 1;
	}

	int hashSize = 6 + 2 * (compX * compY - 1);
	double scale = 1.0 / (width * height);

	double *dev_factors = runFactorKernel(img, width, height, compX, compY);
	double *dev_factors_sum = runFactorSumKernel(dev_factors, width, height, compX, compY);
	double *factors = getFactorsFromDevice(dev_factors_sum, compX, compY);

	double maximumValue;
	int quantisedMaximumValue;
	calculateMaximumValue(maximumValue, quantisedMaximumValue, factors, width, height, compX, compY, scale);

	auto encodeStart = high_resolution_clock::now();
	char *dev_hash = runEncodeKernel(dev_factors_sum, width, height, compX, compY, maximumValue, hashSize);
	char *hash = encodeHashStart(hashSize, dev_hash, compX, compY, quantisedMaximumValue, factors, scale);
	auto encodeEnd = high_resolution_clock::now();
	std::cout << "String encoding time: " << duration_cast<milliseconds>(readEnd - readStart).count() << " ms \n";

	std::cout << "Result: " << hash << '\n';

	// Free memory
	stbi_image_free(img);
	cudaFree(dev_factors);
	cudaFree(dev_factors_sum);
	free(factors);
	cudaFree(dev_hash);
	free(hash);

	return 0;
}

double *getFactorsFromDevice(double *dev_factors, int compX, int compY)
{
	cudaError_t cudaStatus;

	// Copy output vector from GPU buffer to host memory.
	double *factors = (double *)malloc(compX * compY * 3 * sizeof(double));
	cudaStatus = cudaMemcpy(factors, dev_factors, compX * compY * 3 * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	return factors;

Error:
	cudaFree(dev_factors);
	free(factors);

	return nullptr;
}

void calculateMaximumValue(double &maximumValue, int &quantisedMaximumValue, double *factors, int width, int height, int compX, int compY, double scale)
{
	// calculate maximum value
	maximumValue = factors[3];
	for (int i = 4; i < compX * compY * 3; i++)
	{
		if (factors[i] > maximumValue)
			maximumValue = factors[i];
	}
	maximumValue *= scale;
	quantisedMaximumValue = floor(std::max(0, std::min(82, (int)floor(maximumValue * 166 - 0.5))));
	maximumValue = (quantisedMaximumValue + 1) / 166.0;
}

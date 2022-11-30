#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include "utils.cuh"
#include "factorCalculation.cuh"
#include "writeAc.cuh"
#include "writeFlagsDc.cuh"
#include "factorSum.cuh"

using namespace std::chrono;

int encode(const char *filename, int compX, int compY);

int encode(const char *filename, int compX, int compY)
{
	if (compX < 1 || compX > 9 || compY < 1 || compY > 9)
	{
		printf("Blurhash must have between 1 and 9 components\n");
		return 1;
	}

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

	if (width > 1024 || height > 1024)
	{
		printf("Image width and height must not exceed 1024 pixels\n");
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

	DevFactors dev_factors = runFactorKernel(img, width, height, compX, compY);
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

	std::cout << "Blurhash:\n"
			  << hash << '\n';

	// Free memory
	stbi_image_free(img);
	cudaFree(dev_factors.r);
	cudaFree(dev_factors.g);
	cudaFree(dev_factors.b);
	cudaFree(dev_factors_sum);
	free(factors);
	cudaFree(dev_hash);
	free(hash);

	return 0;
}

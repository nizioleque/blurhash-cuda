#include <cuda_runtime.h>
#include <math.h>
#include <chrono>

using namespace std::chrono;

__host__ __device__ void encode83(char *array, int startIndex, double value, int length);
double *getFactorsFromDevice(double *dev_factors_sum, int compX, int compY);
void calculateMaximumValue(double &maximumValue, int &quantisedMaximumValue, double *factors, int width, int height, int compX, int compY, double scale);

__host__ __device__ void encode83(char *array, int startIndex, double value, int length)
{
	const char *digits83 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz#$%*+,-.:;=?@[]^_{|}~";
	for (int i = 1; i <= length; i++)
	{
		int digit = (int)(floor(value) / pow(83, length - i)) % 83;
		array[startIndex + i - 1] = digits83[digit];
	}
}

double *getFactorsFromDevice(double *dev_factors_sum, int compX, int compY)
{
	cudaError_t cudaStatus;
	_V2::system_clock::time_point copyStart, copyEnd;

	// Copy output vector from GPU buffer to host memory.
	copyStart = high_resolution_clock::now();
	double *factors = (double *)malloc(compX * compY * 3 * sizeof(double));
	cudaStatus = cudaMemcpy(factors, dev_factors_sum, compX * compY * 3 * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	copyEnd = high_resolution_clock::now();
	std::cout << "Factors GPU -> CPU copy time: " << duration_cast<milliseconds>(copyEnd - copyStart).count() << " ms \n";

	return factors;

Error:
	cudaFree(dev_factors_sum);
	free(factors);

	return nullptr;
}

void calculateMaximumValue(double &maximumValue, int &quantisedMaximumValue, double *factors, int width, int height, int compX, int compY, double scale)
{
	if (compX * compY == 1)
	{
		maximumValue = 1;
		quantisedMaximumValue = 0;
		return;
	}
	// calculate maximum value
	maximumValue = abs(factors[3]);
	for (int i = 4; i < compX * compY * 3; i++)
	{
		if (abs(factors[i]) > maximumValue)
			maximumValue = abs(factors[i]);
	}
	maximumValue *= scale;
	quantisedMaximumValue = std::max(0, std::min(82, (int)floor(maximumValue * 166 - 0.5)));
	maximumValue = (quantisedMaximumValue + 1) / 166.0;
}

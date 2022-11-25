#include <cuda_runtime.h>

char* encodeHashStart(int hashSize, char* dev_hash, int compX, int compY, int quantisedMaximumValue, double* factors, double scale);
unsigned char linearTosRGB(double value);

char* encodeHashStart(int hashSize, char* dev_hash, int compX, int compY, int quantisedMaximumValue, double* factors, double scale) {
	unsigned char roundedR, roundedG, roundedB;
	int dc, sizeFlag;
	cudaError_t cudaStatus;
	
	
	// Copy output vector from GPU buffer to host memory.
	char* hash = (char*)malloc((hashSize + 1) * sizeof(char));
	cudaStatus = cudaMemcpy(hash, dev_hash, hashSize * sizeof(char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	hash[hashSize] = 0;

	// encode size, maxval
	sizeFlag = compX - 1 + (compY - 1) * 9;
	encode83(hash, 0, sizeFlag, 1);
	encode83(hash, 1, quantisedMaximumValue, 1);

	// encode DC component
	roundedR = linearTosRGB(factors[0] * scale);
	roundedG = linearTosRGB(factors[1] * scale);
	roundedB = linearTosRGB(factors[2] * scale);
	dc = (roundedR << 16) + (roundedG << 8) + roundedB;
	encode83(hash, 2, dc, 4);

	return hash;

Error:
	free(hash);
	cudaFree(dev_hash);
	free(factors);

	return nullptr;
}

unsigned char linearTosRGB(double value) {
	double v = std::max(0.0, std::min(1.0, value));
	if (v <= 0.0031308) {
		return trunc(v * 12.92 * 255.0 + 0.5);
	}
	else {
		return trunc((1.055 * pow(v, 1 / 2.4) - 0.055) * 255.0 + 0.5);
	}
}

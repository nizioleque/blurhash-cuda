#include <cuda_runtime.h>
#include <math.h>

__host__ __device__ void encode83(char* array, int startIndex, double value, int length);

__host__ __device__ void encode83(char* array, int startIndex, double value, int length) {
	const char* digits83 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz#$%*+,-.:;=?@[]^_{|}~";
	for (int i = 1; i <= length; i++) {
		int digit = (int)(floor(value) / pow(83, length - i)) % 83;
		array[startIndex + i - 1] = digits83[digit];
	}
}
#include "encode.h"
#include "common.h"

#include <string.h>
#include <time.h>
#include <stdio.h>

static double *multiplyBasisFunction(int xComponent, int yComponent, int width, int height, uint8_t *rgb, size_t bytesPerRow);
static char *encode_int(int value, int length, char *destination);

static int encodeDC(double r, double g, double b);
static int encodeAC(double r, double g, double b, double maximumValue);

const char *blurHashForPixels(int xComponents, int yComponents, int width, int height, uint8_t *rgb, size_t bytesPerRow)
{
	clock_t startFactor, endFactor, startEncode, endEncode;

	static char buffer[2 + 4 + (9 * 9 - 1) * 2 + 1];

	if (xComponents < 1 || xComponents > 9)
		return NULL;
	if (yComponents < 1 || yComponents > 9)
		return NULL;

	double factors[yComponents][xComponents][3];
	memset(factors, 0, sizeof(factors));

	startFactor = clock();
	for (int y = 0; y < yComponents; y++)
	{
		for (int x = 0; x < xComponents; x++)
		{
			double *factor = multiplyBasisFunction(x, y, width, height, rgb, bytesPerRow);
			factors[y][x][0] = factor[0];
			factors[y][x][1] = factor[1];
			factors[y][x][2] = factor[2];
		}
	}
	endFactor = clock();
	printf("Factor calculation time: %d ms\n", (int)(((double)(endFactor - startFactor)) / CLOCKS_PER_SEC * 1000));

	double *dc = factors[0][0];
	double *ac = dc + 3;
	int acCount = xComponents * yComponents - 1;
	char *ptr = buffer;

	startEncode = clock();

	int sizeFlag = (xComponents - 1) + (yComponents - 1) * 9;
	ptr = encode_int(sizeFlag, 1, ptr);

	double maximumValue;
	if (acCount > 0)
	{
		double actualMaximumValue = 0;
		for (int i = 0; i < acCount * 3; i++)
		{
			actualMaximumValue = fmax(fabs(ac[i]), actualMaximumValue);
		}

		int quantisedMaximumValue = fmax(0, fmin(82, floor(actualMaximumValue * 166 - 0.5)));
		maximumValue = ((double)quantisedMaximumValue + 1) / 166;
		ptr = encode_int(quantisedMaximumValue, 1, ptr);
	}
	else
	{
		maximumValue = 1;
		ptr = encode_int(0, 1, ptr);
	}

	ptr = encode_int(encodeDC(dc[0], dc[1], dc[2]), 4, ptr);

	for (int i = 0; i < acCount; i++)
	{
		ptr = encode_int(encodeAC(ac[i * 3 + 0], ac[i * 3 + 1], ac[i * 3 + 2], maximumValue), 2, ptr);
	}

	*ptr = 0;

	endEncode = clock();
	printf("Image read time: %d ms\n", (int)(((double)(endEncode - startEncode)) / CLOCKS_PER_SEC * 1000));

	return buffer;
}

static double *multiplyBasisFunction(int xComponent, int yComponent, int width, int height, uint8_t *rgb, size_t bytesPerRow)
{
	double r = 0, g = 0, b = 0;
	double normalisation = (xComponent == 0 && yComponent == 0) ? 1 : 2;

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			double basis = cos(M_PI * xComponent * x / width) * cos(M_PI * yComponent * y / height);
			r += basis * sRGBToLinear(rgb[3 * x + 0 + y * bytesPerRow]);
			g += basis * sRGBToLinear(rgb[3 * x + 1 + y * bytesPerRow]);
			b += basis * sRGBToLinear(rgb[3 * x + 2 + y * bytesPerRow]);
		}
	}

	double scale = normalisation / (width * height);

	static double result[3];
	result[0] = r * scale;
	result[1] = g * scale;
	result[2] = b * scale;

	return result;
}

static int encodeDC(double r, double g, double b)
{
	int roundedR = linearTosRGB(r);
	int roundedG = linearTosRGB(g);
	int roundedB = linearTosRGB(b);
	return (roundedR << 16) + (roundedG << 8) + roundedB;
}

static int encodeAC(double r, double g, double b, double maximumValue)
{
	int quantR = fmax(0, fmin(18, floor(signPow(r / maximumValue, 0.5) * 9 + 9.5)));
	int quantG = fmax(0, fmin(18, floor(signPow(g / maximumValue, 0.5) * 9 + 9.5)));
	int quantB = fmax(0, fmin(18, floor(signPow(b / maximumValue, 0.5) * 9 + 9.5)));

	return quantR * 19 * 19 + quantG * 19 + quantB;
}

static char characters[83] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz#$%*+,-.:;=?@[]^_{|}~";

static char *encode_int(int value, int length, char *destination)
{
	int divisor = 1;
	for (int i = 0; i < length - 1; i++)
		divisor *= 83;

	for (int i = 0; i < length; i++)
	{
		int digit = (value / divisor) % 83;
		divisor /= 83;
		*destination++ = characters[digit];
	}
	return destination;
}

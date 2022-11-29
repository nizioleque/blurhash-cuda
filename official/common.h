#ifndef __BLURHASH_COMMON_H__
#define __BLURHASH_COMMON_H__

#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline int linearTosRGB(double value)
{
	double v = fmax(0, fmin(1, value));
	if (v <= 0.0031308)
		return v * 12.92 * 255 + 0.5;
	else
		return (1.055 * pow(v, 1 / 2.4) - 0.055) * 255 + 0.5;
}

static inline double sRGBToLinear(int value)
{
	double v = (double)value / 255;
	if (v <= 0.04045)
		return v / 12.92;
	else
		return pow((v + 0.055) / 1.055, 2.4);
}

static inline double signPow(double value, double exp)
{
	return copysign(pow(fabs(value), exp), value);
}

#endif

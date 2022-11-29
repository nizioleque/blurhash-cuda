

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <sstream>
#include <iostream>

#include "encodeMain.cuh"

int main(int argc, const char **argv)
{

	if (argc != 4)
	{
		printf("Incorrect argument(s)\n");
		printf("The arguments should be: <components x> <components y> <filename>\n");
		return 1;
	}

	std::stringstream compXStr;
	compXStr << argv[1];
	int compX;
	compXStr >> compX;

	std::stringstream compYStr;
	compYStr << argv[2];
	int compY;
	compYStr >> compY;

	return encode(argv[3], compX, compY);
}

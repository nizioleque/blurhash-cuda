


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <sstream>
#include <iostream>

#include "encode.cuh"
#include "decode.cuh"

int main(int argc, const char** argv)
{
	// read arguments
	if (argc < 2) {
		printf("Incorrect argument(s)\n");
		printf("Argument 1 should be \"encode\" or \"decode\"\n");
		return 1;
	}

	// run encoding
	if (strcmp(argv[1], "encode") == 0) {
		if (argc != 5) {
			printf("Incorrect argument(s)\n");
			printf("For \"encode\" the arguments should be: <filename> <components x> <components y>\n");
			return 1;
		}

		std::stringstream compXStr;
		compXStr << argv[3];
		int compX;
		compXStr >> compX;

		std::stringstream compYStr;
		compYStr << argv[4];
		int compY;
		compYStr >> compY;

		return encode(argv[2], compX, compY);
	}

	// run decoding
	if (strcmp(argv[2], "decode") == 0) {
		if (argc != 5) {
			printf("Incorrect argument(s)\n");
			printf("For \"encode\" the arguments should be: <filename> <components x> <components y>\n");
			return 1;
		}

		return decode(argv[2]);
	}

	// display error
	printf("Incorrect argument \"%s\"\n", argv[1]);
	printf("Argument 1 should be \"encode\" or \"decode\"\n");

	return 1;
}



encode: main.cu encodeMain.cuh factorCalculation.cuh factorSum.cuh utils.cuh writeAc.cuh writeFlagsDc.cuh
	# nvcc main.cu -o encode
	nvcc main.cu -o encode -O3

.PHONY: clean
clean:
	rm -f encode
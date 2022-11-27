kernel: base83.cuh decode.cuh encode.cuh encodeFactorKernel.cuh encodeFactorSumKernel.cuh encodeHashStart.cuh encodeKernel.cuh kernel.cu
	# nvcc kernel.cu -o kernel
	nvcc kernel.cu -o kernel -O3

.PHONY: clean
clean:
	rm -f kernel
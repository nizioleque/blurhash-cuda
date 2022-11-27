double *runFactorSumKernel(double *dev_factors, int width, int height, int compX, int compY);
__global__ void factorSumKernel(double *dev_factors, double *dev_factors_sum, int width, int height, int compX, int compY);

double *runFactorSumKernel(double *dev_factors, int width, int height, int compX, int compY)
{
    double *dev_factors_sum = 0;
    cudaError_t cudaStatus;
    _V2::system_clock::time_point sumStart, sumEnd;

    int threads = height;
    int blocks = compX * compY;

    // Allocate memory for factors
    cudaStatus = cudaMalloc((void **)&dev_factors_sum, compX * compY * 3 * sizeof(double));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc (dev_factors_sum) failed!");
        goto Error;
    }

    // Fill with zeros
    cudaStatus = cudaMemset((void *)dev_factors_sum, 0, compX * compY * 3 * sizeof(double));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemset (dev_factors_sum) failed!");
        goto Error;
    }

    sumStart = high_resolution_clock::now();

    factorSumKernel<<<blocks, threads>>>(dev_factors, dev_factors_sum, width, height, compX, compY);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }
    sumEnd = high_resolution_clock::now();
    std::cout << "Factor reduction (kernel): " << duration_cast<milliseconds>(sumEnd - sumStart).count() << " ms \n";

    cudaFree(dev_factors);
    return dev_factors_sum;

Error:
    cudaFree(dev_factors);
    cudaFree(dev_factors_sum);

    return nullptr;
}

__global__ void factorSumKernel(double *dev_factors, double *dev_factors_sum, int width, int height, int compX, int compY)
{
    int blockIndex = blockIdx.x;
    int imagePixelCount = height * width;
    int componentStart = blockIndex * imagePixelCount;

    int lineIndex = threadIdx.x;
    int baseIndex = componentStart + lineIndex * width;

    int colorOffset = compX * compY * height * width;

    // sum values in the 1st column of each image
    for (int dist = 1; dist < height; dist *= 2)
    {
        if (lineIndex % dist != 0)
            return;
        if (lineIndex + dist >= height)
            return;

        __syncthreads();

        dev_factors[baseIndex + 0 * colorOffset] += dev_factors[baseIndex + 0 * colorOffset + dist * width];
        dev_factors[baseIndex + 1 * colorOffset] += dev_factors[baseIndex + 1 * colorOffset + dist * width];
        dev_factors[baseIndex + 2 * colorOffset] += dev_factors[baseIndex + 2 * colorOffset + dist * width];
        __syncthreads();
    }

    __syncthreads();

    dev_factors_sum[blockIndex * 3 + 0] = dev_factors[componentStart + 0 * colorOffset];
    dev_factors_sum[blockIndex * 3 + 1] = dev_factors[componentStart + 1 * colorOffset];
    dev_factors_sum[blockIndex * 3 + 2] = dev_factors[componentStart + 2 * colorOffset];
}
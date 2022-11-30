double *runFactorSumKernel(DevFactors dev_factors, int width, int height, int compX, int compY);
__global__ void factorSumKernel(DevFactors dev_factors, double *dev_factors_sum, int width, int height, int compX, int compY);

double *runFactorSumKernel(DevFactors dev_factors, int width, int height, int compX, int compY)
{
    double *dev_factors_sum = 0;
    cudaError_t cudaStatus;
    _V2::system_clock::time_point sumStart, sumEnd;

    int threads = height;
    int blocks = compX * compY * 3;

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

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching factorSumKernel!\n", cudaStatus);
        goto Error;
    }
    sumEnd = high_resolution_clock::now();
    std::cout << "Factor reduction time: " << duration_cast<milliseconds>(sumEnd - sumStart).count() << " ms \n";

    cudaFree(dev_factors.r);
    cudaFree(dev_factors.g);
    cudaFree(dev_factors.b);
    return dev_factors_sum;

Error:
    cudaFree(dev_factors.r);
    cudaFree(dev_factors.g);
    cudaFree(dev_factors.b);
    cudaFree(dev_factors_sum);

    return nullptr;
}

__global__ void factorSumKernel(DevFactors dev_factors, double *dev_factors_sum, int width, int height, int compX, int compY)
{
    int blockIndex = blockIdx.x / 3;
    int colorIndex = blockIdx.x % 3;
    int imagePixelCount = height * width;
    int componentStart = blockIndex * imagePixelCount;

    int lineIndex = threadIdx.x;
    int baseIndex = componentStart + lineIndex * width;

    double *dev_factors_current = 0;
    switch (colorIndex)
    {
    case 0:
        dev_factors_current = dev_factors.r;
        break;
    case 1:
        dev_factors_current = dev_factors.g;
        break;
    case 2:
        dev_factors_current = dev_factors.b;
        break;
    }

    // sum values in the 1st column of each image
    for (int dist = 1; dist < height; dist *= 2)
    {
        __syncthreads();

        if (lineIndex % (dist * 2) != 0 || lineIndex + dist >= height)
            continue;

        dev_factors_current[baseIndex] += dev_factors_current[baseIndex + dist * width];
    }

    __syncthreads();

    if (lineIndex == 0)
    {
        dev_factors_sum[blockIndex * 3 + colorIndex] = dev_factors_current[componentStart];
    }
}
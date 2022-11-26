double *runFactorSumKernel(double *dev_factors, int width, int height, int compX, int compY);
__global__ void factorSumKernel(double *dev_factors, double *dev_factors_sum, int width, int height, int compX, int compY);

double *runFactorSumKernel(double *dev_factors, int width, int height, int compX, int compY)
{
    double *dev_factors_sum = 0;
    cudaError_t cudaStatus;
    _V2::system_clock::time_point sumStart, sumEnd;

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

    factorSumKernel<<<1, compX * compY>>>(dev_factors, dev_factors_sum, width, height, compX, compY);

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
    int componentIndex = threadIdx.x;
    int imgPixelCount = width * height;
    int factorsStartIndex = componentIndex * imgPixelCount * 3;

    // a[i] = size;
    for (int i = 1; i < imgPixelCount; i++)
    {
        // R
        dev_factors_sum[componentIndex * 3] += dev_factors[factorsStartIndex + i];
        // G
        dev_factors_sum[componentIndex * 3 + 1] += dev_factors[factorsStartIndex + imgPixelCount + i];
        // B
        dev_factors_sum[componentIndex * 3 + 2] += dev_factors[factorsStartIndex + 2 * imgPixelCount + i];
    }
}
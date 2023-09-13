#include "orb_gpu/cuda.cuh"

__device__ inline uint8_t getPixelValue(Image& aImg, int x, int y)
{
    return (x < 0 || x >= aImg.width || y < 0 || y >= aImg.height) ? 125 : aImg.getValue(x, y);
}

__global__ void gaussianBlur(Image aInput, Image aOutput)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx > aInput.width || idy > aInput.height) {
        return;
    }

    uint32_t result = 0;
#pragma unroll
    for (int i = -2; i <= 2; ++i) {
#pragma unroll
        for (int j = -2; j <= 2; ++j) {
            int kernelVal = KERNEL_GAUSS_5[i + 2 + (j + 2) * 5];
            int pixelVal = getPixelValue(aInput, idx + i, idy + j);
            result += (pixelVal * kernelVal);
        }
    }
    result /= GAUSS_M;
    aOutput.setValue(idx, idy, result);
}

void callGaussianBlur(const Image& aImg, Image& aOutputImage, cudaStream_t aStream)
{
    dim3 gridDim(ceil((double)aImg.width / 32), ceil((double)aImg.height / 8));
    dim3 blockDim(32, 8);

    gaussianBlur<<<gridDim, blockDim, 0, aStream>>>(aImg, aOutputImage);
}

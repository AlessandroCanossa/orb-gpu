#include "orb_gpu/cuda.cuh"

__global__ void gaussianBlur(Image aInput, Image aOutput)
{

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idy = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ uint32_t kernel[5*5];

    if (idx < 3 || idx > aInput.width - 3 || idy < 3 || idy > aInput.height - 3) {
        return;
    }
}

cudaError_t callGaussianBlur(const Image& aImg, Image& aOutputImage, cudaStream_t aStream)
{
    dim3 gridDim{ static_cast<unsigned int>((aImg.width - 1) / 32 + 1),
                  static_cast<unsigned int>((aImg.height - 1) / 32 + 1) };
    dim3 blockDim{ 32, 32 };

    gaussianBlur<<<gridDim, blockDim, 0, aStream>>>(aImg, aOutputImage);
    return cudaStreamSynchronize(aStream);
}

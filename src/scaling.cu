#include "orb_gpu/cuda.cuh"

__global__ void scaling(const Image aInput,
                        Image aOutput,
                        float aXScalingRatio,
                        float aYScalingRatio)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx > aOutput.width || idy > aOutput.height) {
        return;
    }

    int sx = idx * aXScalingRatio;
    int sy = idy * aYScalingRatio;

    aOutput.setValue(idx, idy, aInput.getValue(sx, sy));
}

void callImageScaling(const Image& aInput, Image& aOutput, cudaStream_t aStream)
{

    dim3 gridDim{ static_cast<unsigned int>((aOutput.width - 1) / 32 + 1),
                  static_cast<unsigned int>((aOutput.height - 1) / 32 + 1) };
    dim3 blockDim{ 32, 32 };

    float aXScalingRatio = float(aInput.width) / float(aOutput.width);
    float aYScalingRatio = float(aInput.height) / float(aOutput.height);

    scaling<<<gridDim, blockDim, 0, aStream>>>(aInput, aOutput, aXScalingRatio, aYScalingRatio);
}

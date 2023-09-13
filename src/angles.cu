#include "math_constants.h"
#include "orb_gpu/cuda.cuh"

__global__ void computeOrientation(Image aImg, Keypoint* aKpts, int aSize)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id > aSize) {
        return;
    }

    int step = aImg.width;
    int m_01 = 0;
    int m_10 = 0;
    auto& kp = aKpts[id];

    uint8_t* center = aImg.img + (kp.corner.x + step * kp.corner.y);

#pragma unroll
    for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u) {
        m_10 += u * center[u];
    }

    // Go line by line in the circular patch
    for (int v = 1; v <= HALF_PATCH_SIZE; ++v) {
        // Proceed over the two lines
        int v_sum = 0;
        int d = U_MAX[v];
        for (int u = -d; u <= d; ++u) {
            int val_plus = center[u + v * step];
            int val_minus = center[u - v * step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }

    int a = (int)atan2f((float)m_01, (float)m_10) * (180.0F / CUDART_PI_F);

    if (a < 0) {
        a += 360;
    }

    kp.angle = a;
}

void callAngles(Image& aImg, Keypoint* aKpts, int aSize, cudaStream_t aStream)
{
    computeOrientation<<<ceil((double)aSize / 32), 32, 0, aStream>>>(aImg, aKpts, aSize);
}
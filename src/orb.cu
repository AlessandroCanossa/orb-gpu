#include "orb_gpu/cuda.cuh"
#include <math_constants.h>

struct SamplePoint
{
    uint8_t x, y;
};

__device__ inline uint8_t getValue(const SamplePoint* aPattern,
                                   int aIdx,
                                   const Image& aImg,
                                   Keypoint& aKp,
                                   float a,
                                   float b)
{
    int x = (int)roundf(aPattern[aIdx].x * a - aPattern[aIdx].y * b);
    int y = (int)roundf(aPattern[aIdx].x * b + aPattern[aIdx].y * a);

    if (x > aImg.width || y > aImg.height) {
        return 0;
    }

    return aImg.img[((aKp.corner.y + y) * aImg.width) + (aKp.corner.x + x)];
}

__global__ void orb(Image aImg, Keypoint* aKpts, int aSize, int aLayer, float aScale)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id > aSize) {
        return;
    }

    auto& kp = aKpts[id];
    kp.octave = aLayer;

    float angle = kp.angle * (CUDART_PI_F / 180.0F);
    float a = cosf(angle);
    float b = sinf(angle);

    const auto* pattern = (SamplePoint*)SAMPLE_PATTERN;

    for (int i = 0; i < 32; ++i) {
        uint8_t t0 = 0;
        uint8_t t1 = 0;
        uint8_t val = 0;
        t0 = getValue(pattern, 0, aImg, kp, a, b);
        t1 = getValue(pattern, 1, aImg, kp, a, b);
        val = t0 < t1;
        t0 = getValue(pattern, 2, aImg, kp, a, b);
        t1 = getValue(pattern, 3, aImg, kp, a, b);
        val |= (t0 < t1) << 1;
        t0 = getValue(pattern, 4, aImg, kp, a, b);
        t1 = getValue(pattern, 5, aImg, kp, a, b);
        val |= (t0 < t1) << 2;
        t0 = getValue(pattern, 6, aImg, kp, a, b);
        t1 = getValue(pattern, 7, aImg, kp, a, b);
        val |= (t0 < t1) << 3;
        t0 = getValue(pattern, 8, aImg, kp, a, b);
        t1 = getValue(pattern, 9, aImg, kp, a, b);
        val |= (t0 < t1) << 4;
        t0 = getValue(pattern, 10, aImg, kp, a, b);
        t1 = getValue(pattern, 11, aImg, kp, a, b);
        val |= (t0 < t1) << 5;
        t0 = getValue(pattern, 12, aImg, kp, a, b);
        t1 = getValue(pattern, 13, aImg, kp, a, b);
        val |= (t0 < t1) << 6;
        t0 = getValue(pattern, 14, aImg, kp, a, b);
        t1 = getValue(pattern, 15, aImg, kp, a, b);
        val |= (t0 < t1) << 7;

        kp.desc[i] = val;
    }

    kp.corner.x = (int)floorf(kp.corner.x * aScale);
    kp.corner.y = (int)floorf(kp.corner.y * aScale);
}

void callDescriptors(Image& aImg,
                     Keypoint* aKpts,
                     int aSize,
                     int aLayer,
                     float aScale,
                     cudaStream_t aStream)
{
    orb<<<ceil((double)aSize / 32), 32, 0, aStream>>>(aImg, aKpts, aSize, aLayer, aScale);
}

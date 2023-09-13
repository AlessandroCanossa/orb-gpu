#include "orb_gpu/cuda.cuh"

__device__ inline void computeCirclePixels(uint8_t (&aCircle)[BRESENHAM_CIRCUMFERENCE],
                                           Image& aImg,
                                           int aX,
                                           int aY)
{
    aCircle[0] = aImg.getValue(aX + 0, aY + 3);    // 0 + aStride * 3;
    aCircle[1] = aImg.getValue(aX + 1, aY + 3);    // 1 + aStride * 3;
    aCircle[2] = aImg.getValue(aX + 2, aY + 2);    // 2 + aStride * 2;
    aCircle[3] = aImg.getValue(aX + 3, aY + 1);    // 3 + aStride * 1;
    aCircle[4] = aImg.getValue(aX + 3, aY + 0);    // 3 + aStride * 0;
    aCircle[5] = aImg.getValue(aX + 3, aY + -1);   // 3 + aStride * -1;
    aCircle[6] = aImg.getValue(aX + 2, aY + -2);   // 2 + aStride * -2;
    aCircle[7] = aImg.getValue(aX + 1, aY + -3);   // 1 + aStride * -3;
    aCircle[8] = aImg.getValue(aX + 0, aY + -3);   // 0 + aStride * -3;
    aCircle[9] = aImg.getValue(aX + -1, aY + -3);  //-1 + aStride * -3;
    aCircle[10] = aImg.getValue(aX + -2, aY + -2); //-2 + aStride * -2;
    aCircle[11] = aImg.getValue(aX + -3, aY + -1); //-3 + aStride * -1;
    aCircle[12] = aImg.getValue(aX + -3, aY + 0);  //-3 + aStride * 0;
    aCircle[13] = aImg.getValue(aX + -3, aY + 1);  //-3 + aStride * 1;
    aCircle[14] = aImg.getValue(aX + -2, aY + 2);  //-2 + aStride * 2;
    aCircle[15] = aImg.getValue(aX + -1, aY + 3);  //-1 + aStride * 3;
}

__device__ void nonMaxSuppression(Image& aImg, Kp& aKp, Kp* aKpts)
{
    const uint8_t& pixelScore = aKp.score;
    const uint16_t& x = aKp.x;
    const uint16_t& y = aKp.y;

    bool shouldDelete = false;

    shouldDelete |= (aKpts[x + -1 + (y + -1) * aImg.width].score > pixelScore);
    shouldDelete |= (aKpts[x + -1 + (y + 0) * aImg.width].score > pixelScore);
    shouldDelete |= (aKpts[x + -1 + (y + 1) * aImg.width].score > pixelScore);
    shouldDelete |= (aKpts[x + 0 + (y + -1) * aImg.width].score > pixelScore);
    shouldDelete |= (aKpts[x + 0 + (y + 0) * aImg.width].score > pixelScore);
    shouldDelete |= (aKpts[x + 0 + (y + 1) * aImg.width].score > pixelScore);
    shouldDelete |= (aKpts[x + 1 + (y + -1) * aImg.width].score > pixelScore);
    shouldDelete |= (aKpts[x + 1 + (y + 0) * aImg.width].score > pixelScore);
    shouldDelete |= (aKpts[x + 1 + (y + 1) * aImg.width].score > pixelScore);

    __syncthreads();

    if (shouldDelete) {
        aKp.isCorner = false;
        aKp.score = 0;
    }
}

__device__ void cornerDetection(Image& aImg, uint8_t aThreshold, Kp& aKp)
{
    uint8_t circlePixels[BRESENHAM_CIRCUMFERENCE];
    computeCirclePixels(circlePixels, aImg, aKp.x, aKp.y);

    const size_t pixelPos = aKp.x + aImg.width * aKp.y;

    const uint8_t centerPixel = aImg.img[pixelPos];

    const uint8_t pixelPlusTh = (255 - aThreshold) < centerPixel ? 255 : centerPixel + aThreshold;
    const uint8_t pixelMinusTh = (aThreshold > centerPixel) ? 0 : centerPixel - aThreshold;

    bool isCorner = false;
    for (int i = 0; i < BRESENHAM_CIRCUMFERENCE; ++i) {
        bool allLess = true;
        bool allGreater = true;
#pragma unroll
        for (int j = 0; j < 9; ++j) {
            // since BC is multiple of 2 , % can be done this way
            const int wrappedVal = (i + j) & (BRESENHAM_CIRCUMFERENCE - 1);
            const uint8_t& testPixel = circlePixels[wrappedVal];

            allLess &= testPixel < pixelMinusTh;
            allGreater &= testPixel > pixelPlusTh;
        }

        isCorner |= (allLess || allGreater);
    }

    aKp.isCorner = isCorner;

    uint8_t score = 0;

#pragma unroll
    for (int i = 0; i < BRESENHAM_CIRCUMFERENCE; ++i) {
        score += abs(centerPixel - circlePixels[i]);
    }

    aKp.score = isCorner ? score : 0;
}

__global__ void fast9(Image aImg, uint8_t aThreshold, Kp* aKpts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < 3 || idx > aImg.width - 3 || idy < 3 || idy > aImg.height - 3) {
        return;
    }

    Kp& kp = aKpts[idx + idy * aImg.width];
    kp.x = idx;
    kp.y = idy;

    cornerDetection(aImg, aThreshold, kp);

    __syncthreads();

    nonMaxSuppression(aImg, kp, aKpts);
}

// -------------------------------------------------------------------------

void callFast(Image& aImg, uint8_t aThreshold, Kp* aKpts, cudaStream_t aStream)
{
    dim3 gridDim{ static_cast<unsigned int>((aImg.width - 1) / 32 + 1),
                  static_cast<unsigned int>((aImg.height - 1) / 32 + 1) };
    dim3 blockDim{ 32, 32 };

    fast9<<<gridDim, blockDim, 0, aStream>>>(aImg, aThreshold, aKpts);
}

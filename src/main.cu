#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <locale>
#include <memory>
#include <string>
#include <sys/types.h>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "orb_gpu/stb_image.h"

__constant__ const int BRESENHAM_CIRCUMFERENCE = 16;

struct Image
{
    uint8_t* img{};
    size_t width{};
    size_t heigth{};
    bool isGpu;

    __host__ Image() = default;

    __host__ Image(size_t width, size_t heigth, bool gpu)
      : width(width)
      , heigth(heigth)
      , isGpu(gpu)
    {
        if (!gpu) {
            img = new uint8_t[width * heigth];
        } else {
            cudaMalloc(&img, sizeof(uint8_t) * heigth * width);
        }
    }

    __host__ void free() const
    {
        if (!isGpu) {
            delete[] img;
        } else {
            cudaFree(img);
        }
    }

    __host__ cudaError_t upload(Image& aImg, cudaStream_t aStream = 0) const
    {
        assert(isGpu);
        return cudaMemcpyAsync(
          this->img, aImg.img, sizeof(uint8_t) * width * heigth, cudaMemcpyHostToDevice, aStream);
    }

    __host__ __device__ inline uint8_t getValue(size_t x, size_t y) const
    {
        return img[y * this->width + x];
    }

    __host__ __device__ inline void setValue(size_t x, size_t y, uint8_t value) const
    {
        img[y * this->width + x] = value;
    }
};

__global__ void clearCorners(bool* aCorners, uint8_t* aScores, int aWidth)
{

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idy = blockIdx.y * blockDim.y + threadIdx.y;

    aCorners[idx + aWidth * idy] = false;
    aScores[idx + aWidth * idy] = 0;
}

__device__ inline int wrapValue(int aValue, int aMin, int aMax)
{
    int range = aMax - aMin + 1;

    if (aValue < aMin) {
        aValue += range * ((aMin - aValue) / range + 1);
    }

    return aMin + (aValue - aMin) % range;
}

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

__device__ void nonMaxSuppression(Image& aImg,
                                  size_t aX,
                                  size_t aY,
                                  bool* aCorners,
                                  uint8_t* aScores)
{
    uint8_t pixelScore = aScores[aX + aImg.width * aY];
    if (pixelScore == 0) {
        return;
    }

    constexpr int MASK_SIZE = 3;
    bool shouldDelete = false;

#pragma unroll
    for (int i = -MASK_SIZE / 2; i <= MASK_SIZE / 2; ++i) {
#pragma unroll
        for (int j = -MASK_SIZE / 2; j <= MASK_SIZE / 2; ++j) {
            shouldDelete |= (aScores[aX + i + (aY + j) * aImg.width] > pixelScore);
        }
    }

    __syncthreads();

    if (shouldDelete) {
        aCorners[aX + aY * aImg.width] = false;
        aScores[aX + aY * aImg.width] = 0;
    }
}

__device__ void cornerDetection(Image& aImg,
                                size_t aX,
                                size_t aY,
                                uint8_t aThreshold,
                                bool* aCorners,
                                uint8_t* aScores)
{
    uint8_t circlePixels[BRESENHAM_CIRCUMFERENCE];
    computeCirclePixels(circlePixels, aImg, aX, aY);

    const size_t pixelPos = aX + aImg.width * aY;

    const uint8_t centerPixel = aImg.img[pixelPos];

    const uint8_t pixelPlusTh = (255 - aThreshold) < centerPixel ? 255 : centerPixel + aThreshold;
    const uint8_t pixelMinusTh = (aThreshold > centerPixel) ? 0 : centerPixel - aThreshold;

    // uint16_t isCornerBits = 0;
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

        // isCornerBits = isCornerBits | (static_cast<int>(allLess || allGreater) << i);
        isCorner |= (allLess || allGreater);
    }

    if (!isCorner) {
        return;
    }
    aCorners[pixelPos] = true;

    uint8_t score = 0;

#pragma unroll
    for (int i = 0; i < BRESENHAM_CIRCUMFERENCE; ++i) {
        score += abs(centerPixel - circlePixels[i]);
    }

    aScores[pixelPos] = score;
}

__global__ void fast9(Image aImg, uint8_t aThreshold, bool* aCorners, uint8_t* aScores)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < 3 || idx > aImg.width - 3 || idy < 3 || idy > aImg.heigth - 3) {
        return;
    }

    cornerDetection(aImg, idx, idy, aThreshold, aCorners, aScores);

    __syncthreads();

    nonMaxSuppression(aImg, idx, idy, aCorners, aScores);
}

cudaError_t callFast(Image& aImg,
                     uint8_t aThreshold,
                     bool* aCorners,
                     uint8_t* aScores,
                     cudaStream_t aStream = 0)
{
    dim3 gridDim{ static_cast<unsigned int>((aImg.width - 1) / 32 + 1),
                  static_cast<unsigned int>((aImg.heigth - 1) / 32 + 1) };
    dim3 blockDim{ 32, 32 };

    cudaError_t err;

    err = cudaStreamSynchronize(aStream);
    if (err != 0) {
        return err;
    }
    clearCorners<<<gridDim, blockDim, 0, aStream>>>(aCorners, aScores, aImg.width);
    cudaStreamSynchronize(aStream);
    if (err != 0) {
        return err;
    }
    fast9<<<gridDim, blockDim, 0, aStream>>>(aImg, aThreshold, aCorners, aScores);
    return cudaStreamSynchronize(aStream);
}

int main()
{
    std::string imagePath{ "../images/lena.png" };
    int w = 512;
    int h = 512;
    int c = 1;

    Image image;
    image.width = w;
    image.heigth = h;

    image.img = stbi_load(imagePath.c_str(), &w, &h, &c, 1);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    Image imgGpu(w, h, true);
    if (imgGpu.upload(image, stream) != 0) {
        std::cout << "Failed to upload image to Gpu\n";
        imgGpu.free();
        image.free();
        return -1;
    }

    bool* corners = nullptr;
    uint8_t* scores = nullptr;
    cudaMalloc(&corners, sizeof(bool) * w * h);
    cudaMalloc(&scores, sizeof(uint8_t) * w * h);

    auto tic = std::chrono::steady_clock::now();
    cudaError_t err = callFast(imgGpu, 50, corners, scores, stream);
    auto toc = std::chrono::steady_clock::now();
    if (err != 0) {
        std::cout << "Err during kernel: " << cudaGetErrorString(err) << '\n';
        return -1;
    }

    std::cerr << "Fast elapsed time: "
              << std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic).count() << "ns\n";

    bool* cornersCpu = new bool[w * h];
    std::vector<uint8_t> scoresCpu(w * h);
    err =
      cudaMemcpyAsync(cornersCpu, corners, sizeof(bool) * w * h, cudaMemcpyDeviceToHost, stream);
    if (err != 0) {
        std::cout << "Err during memcpy: " << cudaGetErrorString(err) << '\n';
        return -1;
    }
    err = cudaMemcpyAsync(
      scoresCpu.data(), scores, sizeof(uint8_t) * w * h, cudaMemcpyDeviceToHost, stream);
    if (err != 0) {
        std::cout << "Err during memcpy: " << cudaGetErrorString(err) << '\n';
        return -1;
    }
    cudaStreamSynchronize(stream);

    struct Keypoint
    {
        size_t x, y;
        uint8_t score;

        Keypoint(size_t x, size_t y, uint8_t score)
          : x(x)
          , y(y)
          , score(score)
        {}

        Keypoint()
          : Keypoint(0, 0, 0)
        {}
    };

    std::vector<Keypoint> kpts;
    kpts.reserve(10000);

    int counter = 0;
    for (int i = 0; i < w; ++i) {
        for (int j = 0; j < h; ++j) {
            if (cornersCpu[i + j * w]) {
                kpts.emplace_back(i, j, scoresCpu[i + j * w]);
                ++counter;
            }
        }
    }

    std::sort(
      kpts.begin(), kpts.end(), [](auto& kpt1, auto& kpt2) { return kpt1.score > kpt2.score; });

    size_t maxSize = std::min(kpts.size(), 1000UL);
    kpts.resize(maxSize);

    for (const auto& kpt : kpts) {
        printf("%zu,%zu,%d\n", kpt.x, kpt.y, kpt.score);
    }

    delete[] cornersCpu;
    cudaStreamDestroy(stream);
    cudaFree(corners);
    cudaFree(scores);
    imgGpu.free();
    image.free();
    return 0;
}

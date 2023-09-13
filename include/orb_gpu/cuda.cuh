#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

__constant__ const int BRESENHAM_CIRCUMFERENCE = 16;

static void inline tkCudaHandleError(cudaError_t aErr, const char* aFile, int aLine)
{
    if (aErr != cudaSuccess) {
        std::cout << "\033[0;31m" << '[' << __func__ << ':' << __LINE__ << "] "
                  << "Error: " << aErr << "\t " << cudaGetErrorString(aErr) << " - " << aFile << ":"
                  << aLine << "\n"
                  << "\033[0;0m"
                  << "\n";

        if (aErr == cudaErrorUnknown) {
            std::cout << "\033[0;31m" << '[' << __func__ << ':' << __LINE__ << "] "
                      << "Maybe compiled with wrong sm architecture"
                      << "\033[0;0m"
                      << "\n";
        }

        throw std::runtime_error("cudaError");
    }
}
#define tkCUDA(aErr) (tkCudaHandleError(aErr, __FILE__, __LINE__))

struct Image
{
    uint8_t* img{};
    size_t width{};
    size_t height{};
    bool isGpu{ false };

    __host__ Image() = default;

    __host__ Image(size_t width, size_t heigth, bool gpu)
      : width(width)
      , height(heigth)
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
        cudaMemcpyAsync(
          this->img, aImg.img, sizeof(uint8_t) * width * height, cudaMemcpyHostToDevice, aStream);
        return cudaStreamSynchronize(aStream);
    }

    __host__ cudaError_t upload(const uint8_t* aImg, cudaStream_t aStream = 0) const
    {
        assert(isGpu);
        cudaMemcpyAsync(
          this->img, aImg, sizeof(uint8_t) * width * height, cudaMemcpyHostToDevice, aStream);
        return cudaStreamSynchronize(aStream);
    }

    __host__ __device__ inline uint8_t getValue(size_t x, size_t y) const
    {
        return img[y * this->width + x];
    }

    __host__ __device__ inline void setValue(size_t x, size_t y, uint8_t value) const
    {
        img[y * this->width + x] = value;
    }

    __host__ __device__ inline size_t size() const { return width * height; }
};
struct Kp
{
    uint16_t x, y;
    uint8_t score;
    bool isCorner;
};

void callFast(Image& aImg, uint8_t aThreshold, Kp* aKpts, cudaStream_t aStream = 0);

cudaError_t callGaussianBlur(const Image& aImg, Image& aOutputImage, cudaStream_t aStream = 0);

void callImageScaling(const Image& aInput, Image& aOutput, cudaStream_t aStream = 0);

void callAccumKpts(Kp* aInput, int aSize, Kp* aOutput, int* aCounter, cudaStream_t aStream = 0);

void collect(Kp* kpts, int size, Kp* out, cudaStream_t aStream = 0);

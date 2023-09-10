#include <cstdint>
#include <cstdio>

struct Image
{
    uint8_t* img{};
    size_t width{};
    size_t heigth{};

    __host__ Image(size_t width, size_t heigth, bool gpu)
      : width(width)
      , heigth(heigth)
    {
        if (!gpu) {
            img = new uint8_t[width * heigth];
        } else {
            cudaMalloc(&img, sizeof(uint8_t) * heigth * width);
        }
    }

    __host__ __device__ inline uint8_t getValue(size_t x, size_t y) const
    {
        return img[y * this->width + x];
    }

    __host__ __device__ inline void setValue(size_t x, size_t y, uint8_t value)
    {
        img[y * this->width + x] = value;
    }
};

__device__ inline int wrapValue(int aValue, int aMin, int aMax)
{
    int range = aMax - aMin + 1;

    if (aValue < aMin) {
        aValue += range * ((aMin - aValue) / range + 1);
    }

    return aMin + (aValue - aMin) % range;
}

__device__ void cornerDetection() {}

__global__ void fast9() {}

int main()
{
    // for (int i = 0; i < 16; ++i) {
    //     for (int j = -4; j <= 4; ++j) {
    //         printf("%d ", wrapValueCpu(i + j, 0, 15));
    //         break;
    //     }
    //     break;
    //     printf("\n");
    // }

    return 0;
}

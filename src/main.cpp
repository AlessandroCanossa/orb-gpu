#include <chrono>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "orb_gpu/cuda.cuh"

constexpr float SCALING = 1.7;
constexpr float SCALING2 = SCALING * SCALING;

int main(int argc, char** argv)
{
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::imshow("Display image", img);
    cv::waitKey(0);

    std::array<cudaStream_t, 3> streams{ nullptr };
    for (auto& stream : streams) {
        cudaStreamCreate(&stream);
    }

    std::array<Image, 3> gpuImgs{
        Image(img.cols, img.rows, true),
        Image(img.cols / SCALING, img.rows / SCALING, true),
        Image(img.cols / SCALING2, img.rows / SCALING2, true),
    };

    if (gpuImgs[0].upload(img.data, streams[0]) != 0) {
        std::cout << "Failed to upload image to Gpu\n";
        for (int i = 0; i < 3; ++i) {
            cudaStreamDestroy(streams[i]);
            gpuImgs[i].free();
        }
        return -1;
    }

    std::array<Image, 3> gpuImgsBlur{
        Image(img.cols, img.rows, true),
        Image(img.cols / SCALING, img.rows / SCALING, true),
        Image(img.cols / SCALING2, img.rows / SCALING2, true),
    };

    std::array<Kp*, 3> kpts{ nullptr };
    for (int i = 0; i < 3; ++i) {
        cudaMalloc(&kpts[i], sizeof(Kp) * gpuImgs[i].size());
    }
    std::array<Kp*, 3> accumKpts{ nullptr };
    for (int i = 0; i < 3; ++i) {
        cudaMalloc(&accumKpts[i], sizeof(Kp) * 5000);
    }

    // scaling
    {
        auto tic = std::chrono::steady_clock::now();
        callImageScaling(gpuImgs[0], gpuImgs[1], streams[1]);
        callImageScaling(gpuImgs[0], gpuImgs[2], streams[2]);

        tkCUDA(cudaStreamSynchronize(streams[1]));
        tkCUDA(cudaStreamSynchronize(streams[2]));
        auto toc = std::chrono::steady_clock::now();
        std::cerr << "Resize elapsed time: "
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic).count()
                  << "ns\n";
    }

    // fast
    {
        auto tic = std::chrono::steady_clock::now();
        for (int i = 0; i < 3; ++i) {
            callFast(gpuImgs[i], 50, kpts[i], streams[i]);
        }

        for (auto& stream : streams) {
            tkCUDA(cudaStreamSynchronize(stream));
        }
        auto toc = std::chrono::steady_clock::now();

        std::cerr << "Fast elapsed time: "
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic).count()
                  << "ns\n";
    }

    // accum
    {
        auto tic = std::chrono::steady_clock::now();
        for (int i = 0; i < 3; ++i) {
            collect(kpts[i], gpuImgs[i].size(), accumKpts[i], streams[i]);
        }
        for (auto& stream : streams) {
            tkCUDA(cudaStreamSynchronize(stream));
        }
        auto toc = std::chrono::steady_clock::now();
        std::cerr << "accum kpts gpu "
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic).count()
                  << "ns\n";
    }
    for (int i = 0; i < 3; ++i) {
        cudaStreamDestroy(streams[i]);
        cudaFree(kpts[i]);
        gpuImgs[i].free();
        gpuImgsBlur[i].free();
        cudaFree(accumKpts[i]);
    }
    return 0;
}

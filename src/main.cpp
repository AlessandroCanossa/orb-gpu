#include <chrono>
#include <cmath>
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
constexpr int LAYERS = 3;
constexpr size_t MAX_KPTS_LAYER = 10000;

int main(int argc, char** argv)
{
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::imshow("Display image", img);
    cv::waitKey(0);

    std::array<cudaStream_t, LAYERS> streams{ nullptr };
    for (auto& stream : streams) {
        cudaStreamCreate(&stream);
    }

    std::array<Image, LAYERS> gpuImgs{
        Image(img.cols, img.rows, true),
        Image(img.cols / SCALING, img.rows / SCALING, true),
        Image(img.cols / SCALING * SCALING, img.rows / SCALING * SCALING, true),
    };

    if (gpuImgs[0].upload(img.data, streams[0]) != 0) {
        std::cout << "Failed to upload image to Gpu\n";
        for (int i = 0; i < LAYERS; ++i) {
            cudaStreamDestroy(streams[i]);
            gpuImgs[i].free();
        }
        return -1;
    }

    std::array<Image, LAYERS> gpuImgsBlur{
        Image(img.cols, img.rows, true),
        Image(img.cols / SCALING, img.rows / SCALING, true),
        Image(img.cols / SCALING * SCALING, img.rows / SCALING * SCALING, true),
    };

    std::array<Corner*, LAYERS> corners{ nullptr };
    for (int i = 0; i < LAYERS; ++i) {
        cudaMalloc(&corners[i], sizeof(Corner) * gpuImgs[i].size());
    }
    std::array<Corner*, LAYERS> accumCorners{ nullptr };
    for (int i = 0; i < LAYERS; ++i) {
        cudaMalloc(&accumCorners[i], sizeof(Corner) * MAX_KPTS_LAYER);
    }
    std::array<Keypoint*, LAYERS> keypoints{ nullptr };
    for (int i = 0; i < LAYERS; ++i) {
        cudaMalloc(&keypoints[i], sizeof(Keypoint) * MAX_KPTS_LAYER);
    }

    long timeTotal = 0;
    // scaling
    {
        auto tic = std::chrono::steady_clock::now();
        for (int i = 1; i < LAYERS; ++i) {
            callImageScaling(gpuImgs[0], gpuImgs[i], streams[i]);
        }
        for (int i = 1; i < LAYERS; ++i) {
            tkCUDA(cudaStreamSynchronize(streams[i]));
        }
        auto toc = std::chrono::steady_clock::now();
        auto delta = std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic).count();
        std::cerr << "Resize elapsed time: " << delta << "ns\n";
        timeTotal += delta;
    }

    // execute image blur
    {
        auto tic = std::chrono::steady_clock::now();
        for (int i = 0; i < LAYERS; ++i) {
            callGaussianBlur(gpuImgs[i], gpuImgsBlur[i], streams[i]);
        }
        // for (auto& stream : streams) {
        //     tkCUDA(cudaStreamSynchronize(stream));
        // }
        // auto toc = std::chrono::steady_clock::now();
        // auto delta = std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic).count();
        // std::cerr << "gauss " << delta << "ns\n";
        // timeTotal += delta;
        // }
        // fast
        // {
        // auto tic = std::chrono::steady_clock::now();
        for (int i = 0; i < LAYERS; ++i) {
            callFast(gpuImgs[i], 50, corners[i], streams[i]);
        }

        for (auto& stream : streams) {
            tkCUDA(cudaStreamSynchronize(stream));
        }
        auto toc = std::chrono::steady_clock::now();

        auto delta = std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic).count();
        std::cerr << "Fast elapsed time: " << delta << "ns\n";
        timeTotal += delta;
    }

    // accum
    {
        auto tic = std::chrono::steady_clock::now();
        for (int i = 0; i < LAYERS; ++i) {
            collect(corners[i], gpuImgs[i].size(), accumCorners[i], streams[i]);
        }
        for (auto& stream : streams) {
            tkCUDA(cudaStreamSynchronize(stream));
        }
        auto toc = std::chrono::steady_clock::now();
        auto delta = std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic).count();
        std::cerr << "accum kpts gpu " << delta << "ns\n";
        timeTotal += delta;
    }

    // convert to kpts
    {
        auto tic = std::chrono::steady_clock::now();
        for (int i = 0; i < LAYERS; ++i) {
            toKpts(accumCorners[i], MAX_KPTS_LAYER, keypoints[i], streams[i]);
        }
        for (auto& stream : streams) {
            tkCUDA(cudaStreamSynchronize(stream));
        }
        auto toc = std::chrono::steady_clock::now();
        auto delta = std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic).count();
        std::cerr << "conversion to kpts " << delta << "ns\n";
        timeTotal += delta;
    }
    // calc corners angles
    {
        auto tic = std::chrono::steady_clock::now();
        for (int i = 0; i < LAYERS; ++i) {
            callAngles(gpuImgs[i], keypoints[i], MAX_KPTS_LAYER, streams[i]);
        }
        for (auto& stream : streams) {
            tkCUDA(cudaStreamSynchronize(stream));
        }
        auto toc = std::chrono::steady_clock::now();
        auto delta = std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic).count();
        std::cerr << "computing angles " << delta << "ns\n";
        timeTotal += delta;
    }
    // calc descriptors
    {
        auto tic = std::chrono::steady_clock::now();
        for (int i = 0; i < LAYERS; ++i) {
            callDescriptors(
              gpuImgsBlur[i], keypoints[i], MAX_KPTS_LAYER, i, std::pow(SCALING, i), streams[i]);
        }
        for (auto& stream : streams) {
            tkCUDA(cudaStreamSynchronize(stream));
        }
        auto toc = std::chrono::steady_clock::now();
        auto delta = std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic).count();
        std::cerr << "descriptors " << delta << "ns\n";
        timeTotal += delta;
    }
    // cv::Mat blurImg(img.rows, img.cols, CV_8UC1);

    // cudaMemcpyAsync(blurImg.data,
    //                 gpuImgsBlur[0].img,
    //                 sizeof(uint8_t) * img.total(),
    //                 cudaMemcpyDeviceToHost,
    //                 streams[0]);
    // tkCUDA(cudaStreamSynchronize(streams[0]));

    // cv::imshow("blurred img", blurImg);
    // cv::waitKey();

    auto tic = std::chrono::steady_clock::now();
    std::vector<Keypoint> kpts(MAX_KPTS_LAYER * LAYERS);
    for (int i = 0; i < LAYERS; ++i) {
        cudaMemcpyAsync(kpts.data() + (MAX_KPTS_LAYER * i),
                        keypoints[i],
                        MAX_KPTS_LAYER,
                        cudaMemcpyDeviceToHost,
                        streams[i]);
    }
    for (auto& stream : streams) {
        tkCUDA(cudaStreamSynchronize(stream));
    }
    auto toc = std::chrono::steady_clock::now();
    auto delta = std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic).count();
    std::cerr << "kpts collection " << delta << "ns\n";

    std::sort(kpts.begin(), kpts.end(), [](Keypoint& kp1, Keypoint& kp2) {
        return kp1.corner.score > kp2.corner.score;
    });

    kpts.resize(15000);

    std::vector<cv::KeyPoint> cvKpts(15000);
    // std::vector<cv::Desc
    for (int i = 0; i < 15000; ++i) {
        cv::KeyPoint kp(kpts[i].corner.x, kpts[i].corner.y, kpts[i].corner.score, kpts[i].angle);
        cvKpts[i] = kp;
    }

    cv::Mat output(img.rows, img.cols, CV_8UC1);
    cv::drawKeypoints(img, cvKpts, output);
    cv::imshow("result", output);
    cv::waitKey();
    std::cerr << "\nTotal time elapsed: " << timeTotal << "ns\n";

    for (int i = 0; i < LAYERS; ++i) {
        cudaStreamDestroy(streams[i]);
        cudaFree(corners[i]);
        gpuImgs[i].free();
        gpuImgsBlur[i].free();
        cudaFree(accumCorners[i]);
        cudaFree(keypoints[i]);
    }
    return 0;
}

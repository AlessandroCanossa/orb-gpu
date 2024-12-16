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

class App
{
  public:
    App()
    {
        for (int i = 0; i < LAYERS; ++i) {
            cudaStreamCreate(&mStreams[i]);
        }
    }

    App(const App&) = default;
    App(App&&) = delete;
    App& operator=(const App&) = default;
    App& operator=(App&&) = delete;
    void init(const cv::Mat& aInput)
    {
        if (!mInit) {
            for (int i = 0; i < LAYERS; ++i) {
                size_t cols = aInput.cols / std::pow(SCALING, i);
                size_t rows = aInput.rows / std::pow(SCALING, i);
                mGpuImgs[i] = Image(cols, rows, true);
                mGpuImgsBlur[i] = Image(cols, rows, true);
                cudaMalloc(&mCorners[i], sizeof(Corner) * cols * rows);
                cudaMalloc(&mAccumCorners[i], sizeof(Corner) * MAX_KPTS_LAYER);
                cudaMalloc(&mKeypoints[i], sizeof(Keypoint) * MAX_KPTS_LAYER);
            }

            mInit = true;
        }

        tkCUDA(mGpuImgs[0].upload(aInput.data, mStreams[0]));
    }

    void run()
    {
        size_t totalTime = 0;
        auto tic = std::chrono::steady_clock::now();
        scaling();
        auto toc = std::chrono::steady_clock::now();
        auto delta = std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic).count();
        std::cerr << "Scaling time: " << delta << "ns\n";
        totalTime += delta;

        tic = std::chrono::steady_clock::now();
        blur();
        // toc = std::chrono::steady_clock::now();

        // tic = std::chrono::steady_clock::now();
        fast();
        toc = std::chrono::steady_clock::now();
        delta = std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic).count();
        std::cerr << "Blur/fast time: " << delta << "ns\n";
        totalTime += delta;

        tic = std::chrono::steady_clock::now();
        collect();
        toc = std::chrono::steady_clock::now();
        delta = std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic).count();
        std::cerr << "Collect time: " << delta << "ns\n";
        totalTime += delta;

        tic = std::chrono::steady_clock::now();
        angles();
        toc = std::chrono::steady_clock::now();
        delta = std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic).count();
        std::cerr << "Orientation time: " << delta << "ns\n";
        totalTime += delta;

        tic = std::chrono::steady_clock::now();
        descriptors();
        toc = std::chrono::steady_clock::now();
        delta = std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic).count();
        std::cerr << "Descriptors time: " << delta << "ns\n";
        totalTime += delta;
        std::cerr << "\nTotal time elapsed: " << totalTime << "ns\n";
    }

    void output(std::vector<cv::KeyPoint>& aKpts, cv::Mat& aDescriptors)
    {

        std::vector<Keypoint> kpts(MAX_KPTS_LAYER * LAYERS);
        for (int i = 0; i < LAYERS; ++i) {
            cudaMemcpyAsync(kpts.data() + (MAX_KPTS_LAYER * i),
                            mKeypoints[i],
                            MAX_KPTS_LAYER,
                            cudaMemcpyDeviceToHost,
                            mStreams[i]);
        }
        sync();

        std::sort(kpts.begin(), kpts.end(), [](Keypoint& kp1, Keypoint& kp2) {
            return kp1.corner.score > kp2.corner.score;
        });

        kpts.resize(10000);

        aDescriptors = cv::Mat(kpts.size(), 32, CV_8U);
        aKpts.resize(10000);
        for (int i = 0; i < 10000; ++i) {
            cv::KeyPoint kp(
              kpts[i].corner.x, kpts[i].corner.y, kpts[i].corner.score, kpts[i].angle);
            aKpts[i] = kp;

            for (size_t j = 0; j < 32; ++j) {
                aDescriptors.at<uint8_t>(i, j) = kpts[i].desc[j];
            }
        }
    }

    ~App()
    {
        for (int i = 0; i < LAYERS; ++i) {
            cudaStreamDestroy(mStreams[i]);
            cudaFree(mCorners[i]);
            mGpuImgs[i].free();
            mGpuImgsBlur[i].free();
            cudaFree(mAccumCorners[i]);
            cudaFree(mKeypoints[i]);
        }
    }

  private:
    bool mInit = false;
    std::array<cudaStream_t, LAYERS> mStreams{ nullptr };

    std::array<Image, LAYERS> mGpuImgsBlur;
    std::array<Image, LAYERS> mGpuImgs;
    std::array<Corner*, LAYERS> mCorners{ nullptr };
    std::array<Corner*, LAYERS> mAccumCorners{ nullptr };
    std::array<Keypoint*, LAYERS> mKeypoints{ nullptr };

    inline void scaling()
    {
        for (int i = 1; i < LAYERS; ++i) {
            callImageScaling(mGpuImgs[0], mGpuImgs[i], mStreams[i]);
        }
        for (int i = 1; i < LAYERS; ++i) {
            tkCUDA(cudaStreamSynchronize(mStreams[i]));
        }
    }

    inline void blur()
    {
        for (int i = 0; i < LAYERS; ++i) {
            callGaussianBlur(mGpuImgs[i], mGpuImgsBlur[i], mStreams[i]);
        }
    }

    inline void fast()
    {
        for (int i = 0; i < LAYERS; ++i) {
            callFast(mGpuImgs[i], 50, mCorners[i], mStreams[i]);
        }
        sync();
    }

    inline void sync()
    {
        for (auto& stream : mStreams) {
            tkCUDA(cudaStreamSynchronize(stream));
        }
    }

    inline void collect()
    {
        for (int i = 0; i < LAYERS; ++i) {
            ::collect(mCorners[i], mGpuImgs[i].size(), mAccumCorners[i], mStreams[i]);
        }

        sync();
        for (int i = 0; i < LAYERS; ++i) {
            toKpts(mAccumCorners[i], MAX_KPTS_LAYER, mKeypoints[i], mStreams[i]);
        }

        sync();
    }

    inline void angles()
    {
        for (int i = 0; i < LAYERS; ++i) {
            callAngles(mGpuImgs[i], mKeypoints[i], MAX_KPTS_LAYER, mStreams[i]);
        }
        sync();
    }

    inline void descriptors()
    {
        for (int i = 0; i < LAYERS; ++i) {
            callDescriptors(
              mGpuImgsBlur[i], mKeypoints[i], MAX_KPTS_LAYER, i, std::pow(SCALING, i), mStreams[i]);
        }
        sync();
    }
};

int main(int argc, char** argv)
{
    App app;
    cv::Mat img1 = cv::imread("/workspaces/orb_gpu/images/00000.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread("/workspaces/orb_gpu/images/00030.jpg", cv::IMREAD_GRAYSCALE);
    cv::imshow("image1", img1);
    cv::imshow("image2", img2);
    cv::waitKey(0);

    app.init(img1);

    app.run();

    std::vector<cv::KeyPoint> cvKpts;
    cv::Mat desc;
    app.output(cvKpts, desc);

    app.init(img2);

    app.run();

    std::vector<cv::KeyPoint> cvKpts2;
    cv::Mat desc2;
    app.output(cvKpts2, desc2);

    cv::Ptr<cv::DescriptorMatcher> matcher =
      cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
    std::vector<cv::DMatch> matches;
    matcher->match(desc, desc2, matches);

    std::cout << "Num matches: " << matches.size() << std::endl;
    cv::Mat imgMatches;
    cv::drawMatches(img1, cvKpts, img2, cvKpts2, matches, imgMatches);

    cv::imshow("matches", imgMatches);

    cv::Mat output1(img1.rows, img1.cols, CV_8UC1);
    cv::drawKeypoints(img1, cvKpts, output1);
    cv::imshow("result1", output1);
    cv::Mat output2(img2.rows, img2.cols, CV_8UC1);
    cv::drawKeypoints(img2, cvKpts2, output2);
    cv::imshow("result2", output2);
    cv::waitKey();

    return 0;
}

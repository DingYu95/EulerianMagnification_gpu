#ifndef UTIL_H
#define UTIL_H
#include <iostream>
#include <chrono>

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

void bgr2ntsc(cv::Mat &bgrImg, cv::Mat &ntscImg);

void bgr2ntsc(cv::gpu::GpuMat &bgrImg, cv::gpu::GpuMat &ntscImg,
              cv::gpu::Stream &stream=cv::gpu::Stream::Null());

void ntsc2bgr(cv::Mat &ntscImg, cv::Mat &bgrImg);

void ntsc2bgr(cv::gpu::GpuMat &ntscImg, cv::gpu::GpuMat &bgrImg,
              cv::gpu::Stream &stream=cv::gpu::Stream::Null());

void colorAttenuate(cv::Mat &YIQImg, const double chromaFactor);

void colorAttenuate(cv::gpu::GpuMat &YIQImg, const double chromaFactor,
                    cv::gpu::Stream &stream=cv::gpu::Stream::Null());

class simpleTimer {
 private:
    typedef std::chrono::high_resolution_clock clock_type;
    clock_type::time_point begin;
    clock_type::time_point end;
    bool startFlag;

 public:
    simpleTimer() {}
    void set() {
        begin = clock_type::now();
        startFlag = true;
    }
    void count(){
        if (!startFlag) {
            std::cout << "No starting point" << std::endl;
        }
        end = clock_type::now();
        std::cout << "Time spent " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
        << " milliseconds" <<std::endl;
        startFlag = false;
        std::cout<< "Timer stopped, call set" << std::endl;
    }
};

#endif

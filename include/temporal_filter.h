#ifndef TEMPORAL_FILTER_H
#define TEMPORAL_FILTER_H
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

extern int curFrameIdx;

void iirPyr(std::vector<cv::Mat> &lowpassPyr, const std::vector<cv::Mat> &srcPyr, double ratio);

void pyrSubtract(const std::vector<cv::Mat> &lowpass1, const std::vector<cv::Mat> &lowpass2, std::vector<cv::Mat> &filteredPyr);

void iirPyr(std::vector<cv::gpu::GpuMat> &lowpassPyr, const std::vector<cv::gpu::GpuMat> &srcPyr, double ratio,
            cv::gpu::Stream &stream=cv::gpu::Stream::Null());

void pyrSubtract(const std::vector<cv::gpu::GpuMat> &lowpass1, const std::vector<cv::gpu::GpuMat> &lowpass2,
                 std::vector<cv::gpu::GpuMat> &filteredPyr, cv::gpu::Stream &stream=cv::gpu::Stream::Null());

void idealFilter(cv::Mat &srcImg, cv::Mat &dstImg, int cutoffLow, int cutoffHigh);

void idealFilter(cv::gpu::GpuMat &srcImg, cv::gpu::GpuMat &dstImg, int cutoffLow, int cutoffHigh,
                 cv::gpu::Stream &stream=cv::gpu::Stream::Null());

#endif

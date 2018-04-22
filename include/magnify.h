#ifndef MAGNIFY_H
#define MAGNIFY_H
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "../include/spatial_pyr.h"
#include "../include/temporal_filter.h"
#include "../include/util.h"

extern int sampleRate;
extern int idealSize;
extern int curFrameIdx;
extern bool flag;
extern int pyrLevel;
extern double cutoffHigh;
extern double cutoffLow;
extern double lambda_c;
extern double alphaMax;
extern double chromaAttenuation;
extern double exaFactor;

extern cv::Mat idealStack;
extern cv::gpu::GpuMat gpu_idealStack;

extern std::vector<cv::Mat> spatialPyr;
extern std::vector<cv::Mat> lowpass1;
extern std::vector<cv::Mat> lowpass2;
extern std::vector<cv::Mat> filtered;

extern std::vector<cv::gpu::GpuMat> gpu_spatialPyr;
extern std::vector<cv::gpu::GpuMat> gpu_lowpass1;
extern std::vector<cv::gpu::GpuMat> gpu_lowpass2;
extern std::vector<cv::gpu::GpuMat> gpu_filtered;

void laplacianMagnify(cv::Mat& srcImg, double lambda, double delta, cv::Mat& dstImg);

void gaussianMagnify(cv::Mat& srcImg, cv::Mat& colImg, double alpha, cv::Mat& dstImg);

void laplacianMagnify(cv::gpu::GpuMat& srcImg, double lambda, double delta, cv::gpu::GpuMat& dstImg);

void gaussianMagnify(cv::gpu::GpuMat& srcImg, cv::gpu::GpuMat& colImg, double alpha, cv::gpu::GpuMat& dstImg);

void _transposeGpuMat(cv::gpu::GpuMat& srcImg, cv::gpu::GpuMat& dstImg);

void _transposeGpuMat(cv::gpu::GpuMat& srcImg, cv::gpu::GpuMat& dstImg,
                      cv::gpu::GpuMat (&srcChannels)[3], cv::gpu::GpuMat (&dstChannels)[3]);

#endif

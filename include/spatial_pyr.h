#ifndef SPATIAL_PYR_H
#define SPATIAL_PYR_H
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

void initBufferPyr(std::vector<cv::Mat> &bufferPyr, cv::Size initSize, const int matType, const int level);

void initBufferPyr(std::vector<cv::gpu::GpuMat> &bufferPyr, cv::Size initSize, const int matType, const int level);

void copyPyr(std::vector<cv::Mat> &srcPyr, std::vector<cv::Mat> &dstPyr);

void copyPyr(std::vector<cv::gpu::GpuMat> &srcPyr, std::vector<cv::gpu::GpuMat> &dstPyr);

void gaussianPyr(const cv::Mat &srcImg, std::vector<cv::Mat> &gaussPyr);

void gaussianPyr(const cv::gpu::GpuMat &srcImg, std::vector<cv::gpu::GpuMat> &dstPyr,
                 cv::gpu::Stream &stream=cv::gpu::Stream::Null());

void gaussianRecon(std::vector<cv::Mat> &gaussPyr, cv::Mat &dstImg);

void gaussianRecon(std::vector<cv::Mat> &srcPyr, std::vector<cv::Mat> &bufferPyr, cv::Mat &dstImg);

void gaussianRecon(std::vector<cv::gpu::GpuMat> &srcPyr, cv::gpu::GpuMat &dstImg,
                   cv::gpu::Stream &stream=cv::gpu::Stream::Null());

void laplacianPyr(const cv::Mat &srcImg, std::vector<cv::Mat> &lapPyr);

void laplacianPyr(const cv::Mat &srcImg, std::vector<cv::Mat> &bufferPyr, std::vector<cv::Mat> &dstPyr);

void laplacianPyr(const cv::gpu::GpuMat &srcImg, std::vector<cv::gpu::GpuMat> &dstPyr,
                  cv::gpu::Stream &stream=cv::gpu::Stream::Null());

void laplacianRecon(std::vector<cv::Mat> &lapPyr, cv::Mat &dstImg);

void laplacianRecon(std::vector<cv::Mat> &srcPyr, std::vector<cv::Mat> &bufferPyr, cv::Mat &dstImg);

void laplacianRecon(std::vector<cv::gpu::GpuMat> &srcPyr, cv::gpu::GpuMat &dstImg,
                    cv::gpu::Stream &stream=cv::gpu::Stream::Null());

#endif
#include "../include/temporal_filter.h"
using namespace std;
using namespace cv;
using namespace gpu;

void iirPyr(vector<Mat> &lowpassPyr, const vector<Mat> &srcPyr, double ratio) {
    /*
        y[n] = ratio*x[n] + (1-ratio)*y[n-1]
    */
    assert(lowpassPyr.size() == srcPyr.size());
    //size of image decrease as level increase
    int level = srcPyr.size();
    for (int i=0; i<level; i++) {
        assert(lowpassPyr[i].size() == srcPyr[i].size());
        // Saturation is not applied when the output array has the depth CV_32S.
        // May even get result of an incorrect sign in the case of overflow.
        cv::addWeighted(lowpassPyr[i], 1-ratio, srcPyr[i], ratio, 0.0, lowpassPyr[i]);
    }
}

void iirPyr(vector<GpuMat> &lowpassPyr, const vector<GpuMat> &srcPyr, double ratio, Stream &stream) {
    /*
        y[n] = ratio*x[n] + (1-ratio)*y[n-1]
    */
    assert(lowpassPyr.size() == srcPyr.size());
    // size of image decrease as level increase
    int level = srcPyr.size();
    for (int i = 0; i < level; i++) {
        assert(lowpassPyr[i].size() == srcPyr[i].size());
        // Saturation is not applied when the output array has the depth CV_32S.
        // May even get result of an incorrect sign in the case of overflow.
        gpu::addWeighted(lowpassPyr[i], 1.0-ratio, srcPyr[i], ratio, 0.0, lowpassPyr[i], -1, stream);
    }
}

void pyrSubtract(const vector<Mat> &lowpass1, const vector<Mat> &lowpass2, vector<Mat> &filteredPyr) {
    assert(lowpass1.size() == lowpass2.size());
    assert(lowpass1.size() == filteredPyr.size());
    // size of image decrease as level increase
    int level = lowpass1.size();
    for (int i = 0; i < level; i++) {
        assert(lowpass1[i].size() == lowpass2[i].size());
        cv::subtract(lowpass1[i], lowpass2[i], filteredPyr[i]);
    }
}

void pyrSubtract(const vector<GpuMat> &lowpass1, const vector<GpuMat> &lowpass2, vector<GpuMat> &filteredPyr, Stream &stream) {
    assert(lowpass1.size() == lowpass2.size());
    assert(lowpass1.size() == filteredPyr.size());
    // size of image decrease as level increase
    int level = lowpass1.size();
    for (int i = 0; i < level; i++) {
        assert(lowpass1[i].size() == lowpass2[i].size());
        gpu::subtract(lowpass1[i], lowpass2[i], filteredPyr[i], GpuMat(), -1, stream);
    }
}

void idealFilter(cv::Mat &srcImg, cv::Mat &dstImg, int cutoffLow, int cutoffHigh) {
    Mat srcChannels[3];
    cv:split(srcImg, srcChannels);

    for (int i = 0; i < srcImg.channels(); i++) {
        cv::dft(srcChannels[i], srcChannels[i], cv::DFT_ROWS|cv::DFT_SCALE);
        srcChannels[i].colRange(0, cutoffLow).setTo(0);
        srcChannels[i].colRange(cutoffHigh, srcImg.cols).setTo(0);
        cv::idft(srcChannels[i], srcChannels[i], cv::DFT_ROWS);
    }
    cv::merge(srcChannels, srcImg.channels(), dstImg);
    cv::normalize(dstImg, dstImg, 0, 1, NORM_MINMAX);
}

void idealFilter(gpu::GpuMat &srcImg, gpu::GpuMat &dstImg, int cutoffLow, int cutoffHigh, gpu::Stream &stream) {
    gpu::GpuMat srcChannels[3];
    gpu::split(srcImg, srcChannels, stream);
    gpu::GpuMat dftBuffer(cv::Size(srcImg.rows, srcImg.cols/2 + 1), CV_32FC2, 0);
    for (int i = 0; i < srcImg.channels(); i++) {
        bool a = (srcChannels[i].type() == CV_32FC1);
        gpu::dft(srcChannels[i], dftBuffer, srcChannels[i].size(),
                 DFT_ROWS, stream);
        dftBuffer.colRange(0, cutoffLow/2).setTo(0);
        dftBuffer.colRange(cutoffHigh/2, dftBuffer.cols).setTo(0);
        gpu::dft(dftBuffer, srcChannels[i], srcChannels[i].size(),
                 DFT_REAL_OUTPUT|DFT_INVERSE|DFT_ROWS|DFT_SCALE, stream);
        gpu::normalize(srcChannels[i], srcChannels[i], 0, 1, NORM_MINMAX, -1);
    }
    gpu::merge(srcChannels, srcImg.channels(), dstImg);
}

void dftCircularShift(cv::Mat &srcImg, cv::Mat &shiftBase, cv::Mat &dstImg, int shiftLen) {
    shiftBase *= shiftLen;
    Mat shiftedMat;
    cv::mulSpectrums(srcImg, shiftBase, shiftedMat, cv::DFT_ROWS);
    cv::polarToCart(cv::Mat(), shiftedMat, dstImg, cv::noArray());
    shiftBase /= shiftLen;
}

void initShiftMat(int width, int height, cv::Mat &dstImg) {
    cv::Mat cosRow(1, width, CV_32FC1);
    uchar* cosRowPtr = cosRow.data;
    for (int i = 0; i < width; i++) {
        cosRowPtr[i] = 2 * 3.14 * i / width;
    }
    cv::repeat(cosRow, height, 1, dstImg);
}

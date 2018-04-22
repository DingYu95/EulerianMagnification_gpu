#include "../include/magnify.h"

using namespace std;
using namespace cv;

void laplacianMagnify(cv::Mat& srcImg, double lambda, double delta, cv::Mat& dstImg) {
    laplacianPyr(srcImg, spatialPyr);
    iirPyr(lowpass1, spatialPyr, cutoffHigh);
    iirPyr(lowpass2, spatialPyr, cutoffLow);
    pyrSubtract(lowpass1, lowpass2, filtered);

    // Amplification in each level
    double curAlpha = lambda/delta/8 - 1;
    // ignore lowest and highest level
    filtered[0].setTo(0);
    filtered[pyrLevel-1].setTo(0);
    for (int i=1; i < pyrLevel-1; i++) {
        curAlpha = lambda/delta/8 - 1;
        curAlpha *= exaFactor;
        // a tmp mat here?
        filtered[i].convertTo(filtered[i], -1, std::min(alphaMax, curAlpha));
        lambda /= 2.0;
    }

    // Reconstruction from pyramid
    laplacianRecon(filtered, dstImg);
    colorAttenuate(dstImg, chromaAttenuation);
    cv::add(dstImg, srcImg, dstImg);
}

void laplacianMagnify(gpu::GpuMat& srcImg, double lambda, double delta, gpu::GpuMat& dstImg) {
    laplacianPyr(srcImg, gpu_spatialPyr);
    iirPyr(gpu_lowpass1, gpu_spatialPyr, cutoffHigh);
    iirPyr(gpu_lowpass2, gpu_spatialPyr, cutoffLow);
    pyrSubtract(gpu_lowpass1, gpu_lowpass2, gpu_filtered);

    // Amplification in each level
    double curAlpha = lambda/delta/8 - 1;
    // ignore lowest and highest level
    gpu_filtered[0].setTo(0);
    gpu_filtered[pyrLevel-1].setTo(0);
    for (int i=1; i < pyrLevel-1; i++) {
        curAlpha = lambda/delta/8 - 1;
        curAlpha *= exaFactor;
        // a tmp mat here?
        // gpu::multiply(gpu_filtered[i], std::min(alphaMax, curAlpha), gpu_filtered[i]);
        gpu_filtered[i].convertTo(gpu_filtered[i], -1, std::min(alphaMax, curAlpha));
        lambda /= 2.0;
    }

    // Reconstruction from pyramid
    laplacianRecon(gpu_filtered, dstImg);
    colorAttenuate(dstImg, chromaAttenuation);
    gpu::add(dstImg, srcImg, dstImg);
}


void gaussianMagnify(cv::Mat& srcImg, cv::Mat& colImg, double alpha, cv::Mat& dstImg) {
    // curFrameIdx++;
    // curFrameIdx %= idealSize;
    int lCut = cutoffLow * idealSize;
    int hCut = cutoffHigh * idealSize;
    int bluredImgHeight = spatialPyr.back().rows;

    Mat idftBuffer = cv::Mat::zeros(colImg.size(), colImg.type());
    gaussianPyr(srcImg, spatialPyr);

    Mat bluredImg = spatialPyr.back();  // The smallest image in pyramid
    bluredImg = bluredImg.reshape(0, bluredImg.cols * bluredImg.rows);
    if (curFrameIdx < (idealSize - 1)) {
        bluredImg.copyTo(colImg.col(curFrameIdx));  // Use assignto here?
        curFrameIdx++;
    } else {
        colImg.colRange(1, idealSize-1).copyTo(colImg.colRange(0, idealSize-2));  // Use assignto here?
        bluredImg.copyTo(colImg.col(idealSize - 1));
    }

    idealFilter(colImg, idftBuffer, lCut, hCut);

    Mat filteredImg = idftBuffer.col(curFrameIdx).clone().reshape(0, bluredImgHeight);
    filteredImg *= alpha;

    // colorAttenuate(filteredImg, chromaAttenuation);
    // Gaussian Recon?
    cv::resize(filteredImg, dstImg, srcImg.size());

    cv::add(dstImg, srcImg, dstImg);
}

void gaussianMagnify(gpu::GpuMat& srcImg, gpu::GpuMat& rowImg, double alpha, gpu::GpuMat& dstImg) {
    curFrameIdx++;
    curFrameIdx %= idealSize;
    int lCut = cutoffLow * idealSize;
    int hCut = cutoffHigh * idealSize;
    int bluredImgHeight = gpu_spatialPyr.back().rows;

    gpu::GpuMat colImg;
    gpu::GpuMat idftBufferT;
    gpu::GpuMat rowImgChannels[3];
    gpu::GpuMat colImgChannels[3];
    gpu::GpuMat filteredImg;

    gaussianPyr(srcImg, gpu_spatialPyr);

    gpu::GpuMat idftBuffer(rowImg.cols, rowImg.rows, rowImg.type(), 0);
    gpu::createContinuous(idftBuffer.size(), idftBuffer.type(), idftBuffer);

    gpu::GpuMat tmpImg = gpu_spatialPyr.back().reshape(0, 1);
    gpu::GpuMat curRow = rowImg.row(curFrameIdx);
    tmpImg.copyTo(curRow);

    // curRow = tmpImg;  //gpu_spatialPyr.back().reshape(0, 1).clone();

    // if (curFrameIdx < (idealSize - 1)) {
    //     rowImg.col(curFrameIdx) = gpu_spatialPyr.back().reshape(0, 1);
    //     curFrameIdx++;
    // } else {
    //     rowImg.rowRange(0, idealSize-2) = rowImg.rowRange(1, idealSize - 1);
    //     rowImg.row(idealSize - 1) = gpu_spatialPyr.back().reshape(0, 1);
    // }

    _transposeGpuMat(rowImg, colImg, rowImgChannels, colImgChannels);

    idealFilter(colImg, idftBuffer, lCut, hCut);

    _transposeGpuMat(idftBuffer, idftBufferT, colImgChannels, rowImgChannels);

    // row for GpuMat is always continuous
    filteredImg = idftBufferT.row(curFrameIdx).clone().reshape(0, bluredImgHeight);
    gpu::multiply(filteredImg, alpha, filteredImg);

    // colorAttenuate(filteredImg, chromaAttenuation);
    // Gaussian Recon?
    gpu::resize(filteredImg, dstImg, srcImg.size());

    gpu::add(dstImg, srcImg, dstImg);
}


/*
    Helper function for transpose GpuMat
    Since gpu::transpose(...) only supports 1-, 4-, 8-byte element sizes
*/
void _transposeGpuMat(gpu::GpuMat& srcImg, gpu::GpuMat& dstImg) {
    gpu::GpuMat srcChannels[3];
    gpu::GpuMat dstChannels[3];
    gpu::split(srcImg, srcChannels);
    for (int i=0; i < srcImg.channels(); i++) {
        gpu::transpose(srcChannels[i], dstChannels[i]);
    }
    gpu::merge(dstChannels, srcImg.channels(), dstImg);
}


/*
    Overload with buffer that can be preallocate
*/
void _transposeGpuMat(gpu::GpuMat &srcImg, gpu::GpuMat &dstImg,
                      gpu::GpuMat (&srcChannels)[3], gpu::GpuMat (&dstChannels)[3]) {
    gpu::split(srcImg, srcChannels);
    for (int i=0; i < srcImg.channels(); i++) {
        gpu::transpose(srcChannels[i], dstChannels[i]);
    }
    gpu::merge(dstChannels, srcImg.channels(), dstImg);
}

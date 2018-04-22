#include <iostream>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "../include/magnify_gui.h"
#include "../include/util.h"
#include "../include/spatial_pyr.h"
#include "../include/temporal_filter.h"
#include "../include/magnify.h"

#define CPU_MODE 0
#define GPU_MODE 1

using namespace std;
using namespace cv;

// Magnification configurations.
double lambda_c = 16;
double alphaMax = 10;  // Maximum magnification factor
double chromaAttenuation = 0.5;
double exaFactor = 2.0;  // exaggeration_factor for better visualize
int pyrLevel = 5;
double cutoffHigh = 0.7;
double cutoffLow = 0.1;

int imageType = CV_32FC3;

Mat bgrImg;
Mat bgrResult;

// Images
Mat yiqImg;
Mat magnified;

// Pyramids
vector<Mat> spatialPyr;
vector<Mat> lowpass1;
vector<Mat> lowpass2;
vector<Mat> filtered;

Mat idealStack;

gpu::GpuMat gpu_bgrImg;
gpu::GpuMat gpu_yiqImg;
gpu::GpuMat gpu_magnified;
gpu::GpuMat gpu_bgrResult;

vector<gpu::GpuMat> gpu_spatialPyr;
vector<gpu::GpuMat> gpu_lowpass1;
vector<gpu::GpuMat> gpu_lowpass2;
vector<gpu::GpuMat> gpu_filtered;

gpu::GpuMat gpu_idealStack;

int idealSize = 128;  // power of two or 2^p * 3^q * 5^r, or call getOptimalDFTSize()
int curFrameIdx = 0;
int sampleRate = 30;
bool flag = true;

int main() {
    double delta = lambda_c/8/(1+alphaMax);
    double lambda;

    int imgWidth, imgHeight;
    double lambdaMax;

    // Create gui change configuration in realtime
    namedWindow("Magnify Configuration", WINDOW_AUTOSIZE);
    createTrackbar("alphaMax", "Magnify Configuration", &alphaMaxSilder, alphaMaxSliderMax, alphaMaxTrackbar);
    createTrackbar("Ratio_1", "Magnify Configuration", &cutoffHighSlider, ratioSliderMax, cutoffHighTrackbar);
    createTrackbar("Ratio_2", "Magnify Configuration", &cutoffLowSlider, ratioSliderMax, cutoffLowTrackbar);
    createTrackbar("chromaAttenuate", "Magnify Configuration", &chromaSlider, chromaSliderMax, chromaTrackbar);

    cv::VideoCapture cap(0);  // "./face.mp4"
    if (cap.isOpened() == false) {
        cout << "Cannot open the video camera" << endl;
        cin.get();
        return -1;
    }

    imgWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    imgHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    lambdaMax = sqrt(imgWidth * imgWidth + imgHeight * imgHeight) / 3.0;

#if CPU_MODE
    // Init pyramids
    initBufferPyr(spatialPyr, cv::Size(imgWidth, imgHeight), imageType, pyrLevel);
    initBufferPyr(lowpass1, cv::Size(imgWidth, imgHeight), imageType, pyrLevel);
    initBufferPyr(lowpass2, cv::Size(imgWidth, imgHeight), imageType, pyrLevel);
    initBufferPyr(filtered, cv::Size(imgWidth, imgHeight), imageType, pyrLevel);
#endif

#if GPU_MODE
    initBufferPyr(gpu_spatialPyr, cv::Size(imgWidth, imgHeight), imageType, pyrLevel);
    initBufferPyr(gpu_lowpass1, cv::Size(imgWidth, imgHeight), imageType, pyrLevel);
    initBufferPyr(gpu_lowpass2, cv::Size(imgWidth, imgHeight), imageType, pyrLevel);
    initBufferPyr(gpu_filtered, cv::Size(imgWidth, imgHeight), imageType, pyrLevel);

    // Init image stack for ideal filter
    gpu::GpuMat gpu_idealStack = gpu::GpuMat(idealSize, gpu_spatialPyr.back().rows*gpu_spatialPyr.back().cols,
                                 imageType, 0);
    gpu::createContinuous(gpu_idealStack.size(), gpu_idealStack.type(), gpu_idealStack);

    gpu::createContinuous(gpu_spatialPyr.back().size(), gpu_spatialPyr.back().type(),
                          gpu_spatialPyr.back());
#endif

    while (cap.isOpened()) {
        cap >> bgrImg;

        bgrImg.convertTo(bgrImg, imageType);

#if GPU_MODE
        /********* GPU Version **************/
        gpu_bgrImg.upload(bgrImg);
        bgr2ntsc(gpu_bgrImg, gpu_yiqImg);
        // laplacianMagnify(gpu_yiqImg, lambdaMax, delta, gpu_magnified);
        gaussianMagnify(gpu_yiqImg, gpu_idealStack, alphaMax, gpu_magnified);
        ntsc2bgr(gpu_magnified, gpu_bgrResult);
        gpu_bgrResult.download(bgrResult);
        /********* GPU Version **************/
#endif

#if CPU_MODE
        /********** CPU Version *************/
        bgr2ntsc(bgrImg, yiqImg);
        // laplacianMagnify(yiqImg, lambdaMax, delta, magnified);
        gaussianMagnify(yiqImg, idealStack, alphaMax, magnified);
        ntsc2bgr(magnified, bgrResult);
        /********** CPU Version *************/
#endif

        bgrResult.convertTo(bgrResult, CV_8UC3);
        imshow("magnified", bgrResult);

        if (27 == waitKey(10)%256) break;
    }

    return 0;
}


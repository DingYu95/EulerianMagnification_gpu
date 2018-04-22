#include "../include/util.h"
using namespace std;
using namespace cv;
using namespace cv::gpu;

void bgr2ntsc(Mat &bgrImg, Mat &ntscImg) {
    /* convert bgr image to ntsc(YIQ) image
        Y     0.299  0.587  0.114   R
        I  =  0.596 -0.274 -0.322   G
        Q     0.211 -0.523  0.312   B

        src and dst needs to be signed (float) value, otherwise cause serious color distortion.
    */
    // TODO: convert to float or double outside? Replace with functional call?
    Mat bgrChannels[3]; // Pre allocation use vector?
    Mat yiqChannels[3]; // Vector bgrChannels(Mat(bgrImg.height, bgrImg.width, bgrImg.type()), 3);
    cv::split(bgrImg, bgrChannels);
    yiqChannels[0] = 0.299 * bgrChannels[2] + 0.588 * bgrChannels[1] + 0.114 * bgrChannels[0];
    yiqChannels[1] = 0.596 * bgrChannels[2] - 0.274 * bgrChannels[1] - 0.322 * bgrChannels[0];
    yiqChannels[2] = 0.211 * bgrChannels[2] - 0.523 * bgrChannels[1] + 0.312 * bgrChannels[0];

    cv::merge(yiqChannels, 3, ntscImg);

}


void bgr2ntsc(GpuMat &bgrImg, GpuMat &ntscImg, Stream &stream) {
    /*
     GPU version of bgr2ntsc.
    */
    //Possible improvement: 1. pre allocation.
    vector<gpu::GpuMat> bgrChannels(3), yiqChannels(3);
    // GpuMat bgrChannels[3];
    // GpuMat yiqChannels[3];
    gpu::split(bgrImg, bgrChannels, stream);

    gpu::addWeighted(bgrChannels[2], 0.299, bgrChannels[1], 0.588, 0, yiqChannels[0], -1, stream);
    gpu::scaleAdd(bgrChannels[0], 0.114, yiqChannels[0], yiqChannels[0], stream);
    gpu::addWeighted(bgrChannels[2], 0.596, bgrChannels[1], -0.274, 0, yiqChannels[1], -1, stream);
    gpu::scaleAdd(bgrChannels[0], -0.322, yiqChannels[1], yiqChannels[1], stream);
    gpu::addWeighted(bgrChannels[2], 0.211, bgrChannels[1], -0.523, 0, yiqChannels[2], -1, stream);
    gpu::scaleAdd(bgrChannels[0], 0.312, yiqChannels[2], yiqChannels[2], stream);

    gpu::merge(yiqChannels, ntscImg, stream);
}


void ntsc2bgr(Mat &ntscImg, Mat &bgrImg) {
    /* convert ntsc(YIQ) image to bgr image
     R   1.000  0.956    0.621     Y
     G = 1.000  -0.272   -0.647 *  I
     B   1.000  -1.105   1.702     Q

     src and dst needs to be signed(float) value, otherwise cause serious color distortion.
    */
    Mat bgrChannels[3];
    Mat yiqChannels[3];
    cv::split(ntscImg, yiqChannels);
    bgrChannels[2] = yiqChannels[0] + 0.956 * yiqChannels[1] + 0.621 * yiqChannels[2];
    bgrChannels[1] = yiqChannels[0] - 0.272 * yiqChannels[1] - 0.647 * yiqChannels[2];
    bgrChannels[0] = yiqChannels[0] - 1.105 * yiqChannels[1] + 1.702 * yiqChannels[2];

    cv::merge(bgrChannels, 3, bgrImg);

}


void ntsc2bgr(GpuMat &ntscImg, GpuMat &bgrImg, Stream &stream) {
    /*
     GPU version of ntsc2bgr
     convert bgr image to ntsc(YIQ) image
     R   1.000  0.956    0.621     Y
     G = 1.000  -0.272   -0.647 *  I
     B   1.000  -1.105   1.702     Q
    */
    //Possible improvement: 1. pre allocation. 2. add more temporal GpuMat, instead "inplace"
    GpuMat yiqChannels[3];
    GpuMat bgrChannels[3];
    gpu::split(ntscImg, yiqChannels, stream);
    gpu::scaleAdd(yiqChannels[1],  0.956, yiqChannels[0], bgrChannels[2], stream);
    gpu::scaleAdd(yiqChannels[2],  0.621, bgrChannels[2], bgrChannels[2], stream);
    gpu::scaleAdd(yiqChannels[1], -0.272, yiqChannels[0], bgrChannels[1], stream);
    gpu::scaleAdd(yiqChannels[2], -0.647, bgrChannels[1], bgrChannels[1], stream);
    gpu::scaleAdd(yiqChannels[1], -1.105, yiqChannels[0], bgrChannels[0], stream);
    gpu::scaleAdd(yiqChannels[2],  1.702, bgrChannels[0], bgrChannels[0], stream);

    gpu::merge(bgrChannels, 3, bgrImg, stream);
}

/*
  Color attenuation.
  Attenuate by multiply I and Q channel with a constant factor
*/
void colorAttenuate(Mat &YIQImg, const double chromaFactor) {
    Mat YIQChannels[3];
    cv::split(YIQImg, YIQChannels);

    YIQChannels[1].convertTo(YIQChannels[1], -1, chromaFactor);
    YIQChannels[2].convertTo(YIQChannels[2], -1, chromaFactor);
    cv::merge(YIQChannels, 3, YIQImg);
}

/*
  Color attenuation with GpuMat
  Attenuate by multiply blue and green channel with a constant factor
*/
void colorAttenuate(GpuMat &YIQImg, const double chromaFactor, Stream &stream) {
    GpuMat YIQChannels[3];
    gpu::split(YIQImg, YIQChannels, stream);
    gpu::multiply(YIQChannels[1], chromaFactor, YIQChannels[1], 1.0, -1, stream);
    gpu::multiply(YIQChannels[2], chromaFactor, YIQChannels[2], 1.0, -1, stream);
    // YIQChannels[0].convertTo(YIQChannels[0], -1, chromaFactor);
    // YIQChannels[1].convertTo(YIQChannels[1], -1, chromaFactor);
    gpu::merge(YIQChannels, 3, YIQImg, stream);
}

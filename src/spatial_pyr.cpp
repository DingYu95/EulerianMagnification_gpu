#include "../include/spatial_pyr.h"

using namespace std;
using namespace cv;
using namespace cv::gpu;


/*
    Build buffer pyramid, where every level is pre allocated
*/
void initBufferPyr(vector<Mat> &bufferPyr, cv::Size initSize, const int matType, const int level) {
    // size of image decrease as level increase
    for (int i=0; i<level; i++) {
        bufferPyr.push_back(Mat::zeros(initSize, matType));
        initSize.width /= 2;
        initSize.height /= 2;
    } // Perhaps a struct better?
}

/*
    Create buffer for GpuMat, no zero() function. Use ensureSizeIsEnough()
*/
void initBufferPyr(vector<GpuMat> &bufferPyr, cv::Size initSize, const int matType, const int level) {
    // size of image decrease as level increase
    for (int i=0; i<level; i++) {
        bufferPyr.push_back(GpuMat(initSize, matType, 0.0));  // Must init with value!!!
        ensureSizeIsEnough(initSize, matType, bufferPyr[i]);
        initSize.width /= 2;
        initSize.height /= 2;
    } // Perhaps a struct better?
}

/*
    Deep copy of vector of Mat. Break the reference.
    If use vector<Mat> a = b, inside a[i] is reference of b[i].
*/
void copyPyr(vector<Mat> &srcPyr, vector<Mat> &dstPyr) {
    for (int i=0; i<srcPyr.size(); i++) {
        dstPyr[i] = srcPyr[i].clone();
    }
}

/*
    Deep copy of vector of GpuMat. Break the reference.
    If use vector<GpuMat> a = b, inside a[i] is reference of b[i].
*/
void copyPyr(vector<GpuMat> &srcPyr, vector<GpuMat> &dstPyr) {
    for (int i=0; i<srcPyr.size(); i++) {
        dstPyr[i] = srcPyr[i].clone();
    }
}

/*
    Build Gaussian Pyramid. dstPyr should defined with level number.
*/
void gaussianPyr(const Mat &srcImg, vector<Mat> &dstPyr) {
    // size of image decrease as level increase, first level is same size as srcImg
    // dstPyr should be pre allocated
    int level = dstPyr.size();
    dstPyr[0] = srcImg.clone();
    for (int i=0; i<level-1; i++) {
        cv::pyrDown(dstPyr[i], dstPyr[i+1]);
    }
}


/*
    overLoad of gaussianPyr use gpumat.
*/
void gaussianPyr(const GpuMat &srcImg, vector<GpuMat> &dstPyr, Stream &stream) {
    // size of image decrease as level increase, first level is same size as srcImg
    // dstPyr should be pre allocated
    int level = dstPyr.size();
    dstPyr[0] = srcImg; //use reference directly?
    for (int i=0; i<level-1; i++) {
        gpu::pyrDown(dstPyr[i], dstPyr[i+1], stream);
    }
}


/*
    Reconstruction from Gaussian Pyramid. Only use last level of srcPyr.
*/
void gaussianRecon(vector<Mat> &srcPyr, Mat &dstImg) {
    int level = srcPyr.size();
    // size of image decrease as level increase, fist level is same size as srcImg
    Mat curLevel = srcPyr[level-1];
    for (int i=level-2; i>=0; i--) {
        Mat upLevel;
        cv::pyrUp(curLevel, upLevel);
        curLevel = upLevel;
    }
    //Ensure size
    cv::resize(curLevel, curLevel, dstImg.size());
    dstImg = curLevel; // Use copy here?
}

/*
    overLoad of gaussianRecon use pre allocated bufferPyr, with size specified
    in every level.
*/
void gaussianRecon(vector<Mat> &srcPyr, vector<Mat> &bufferPyr, Mat &dstImg) {
    assert(srcPyr.size() == bufferPyr.size());
    int level = srcPyr.size();
    bufferPyr[level-1] = srcPyr[level-1]; // Clone here?
    for (int i=level-1; i>=1; i--) {
        cv::pyrUp(bufferPyr[i], bufferPyr[i-1], bufferPyr[i-1].size());
    }

    dstImg = bufferPyr[level-1];
}

/*
    Reconstruction from Gaussian Pyramid. Only use last level of srcPyr.
*/
void gaussianRecon(vector<GpuMat> &srcPyr, GpuMat &dstImg, Stream &stream) {
    int level = srcPyr.size();
    // size of image decrease as level increase, fist level is same size as srcImg
    GpuMat curLevel = srcPyr[level-1]; //Pass by reference
    for (int i=level-2; i>=0; i--) {
        GpuMat upLevel;
        gpu::pyrUp(curLevel, upLevel, stream);
        curLevel = upLevel;
    }
    //Ensure size
    //cv::gpu::resize(curLevel, curLevel, dstImg.size());
    dstImg = curLevel; // Use copy here?
}


/*
    Build Laplacian Pyramid. dstPyr should defined with level number.
    Last level is not strict Laplacian, used to reconstruction.
*/
void laplacianPyr(const Mat &srcImg, vector<Mat> &dstPyr) {
    //size of image decrease as level increase, first level is same size as srcImg
    int level = dstPyr.size();
    Mat curLevel = srcImg; // Use clone?
    for (int i=0; i<level-1; i++) {
        Mat downLevel; //size changes, how to optimize?
        cv::pyrDown(curLevel, downLevel);
        cv::pyrUp(downLevel, dstPyr[i], curLevel.size()); // Use inplace
        cv::subtract(curLevel, dstPyr[i], dstPyr[i]); // Replace uplevel with dstPyr[i]
        curLevel = downLevel;
    }
    // Last level of dstPyr is direct copy of down pyramid to former level
    dstPyr[level-1] = curLevel;
}

/*
    An overLoad use pre allocated buffer pyramid(call initPyr).
    buffer pyramids only stores gaussian pyrdown images.
*/
void laplacianPyr(const Mat &srcImg, vector<Mat> &bufferPyr, vector<Mat> &dstPyr) {
    // Need at least n + 1 level buffer for n level Laplacian pyramid
    assert(bufferPyr.size() - dstPyr.size() == 1);
    // size of image decrease as level increase
    int level = dstPyr.size();
    srcImg.copyTo(bufferPyr[0]);
    for (int i=0; i<level-1; i++) {
        cv::pyrDown(bufferPyr[i], bufferPyr[i+1]);
        cv::pyrUp(bufferPyr[i+1], dstPyr[i], bufferPyr[i].size());
        cv::subtract(bufferPyr[i], dstPyr[i], dstPyr[i]); // a tmp mat here?
    }
    // Last level of dstPyr is direct copy of down pyramid to former level
    dstPyr[level-1] = bufferPyr[level];
}

/*
    Build Laplacian Pyramid. dstPyr should defined with level number.
    Last level is not strict Laplacian, used to reconstruction.
*/
void laplacianPyr(const GpuMat &srcImg, vector<GpuMat> &dstPyr, Stream &stream) {
    //size of image decrease as level increase, first level is same size as srcImg
    int level = dstPyr.size();
    GpuMat curLevel = srcImg; // Use clone?
    for (int i=0; i<level-1; i++) {
        GpuMat downLevel; //size changes, how to optimize?
        gpu::pyrDown(curLevel, downLevel, stream);
        gpu::pyrUp(downLevel, dstPyr[i], stream);
        gpu::subtract(curLevel, dstPyr[i], dstPyr[i], GpuMat(), -1, stream); // Replace uplevel with dstPyr[i]
        curLevel = downLevel;
    }
    // Last level of dstPyr is direct copy of down pyramid to former level
    dstPyr[level-1] = curLevel;
}

/*
    Reconstruct image from laplacian Pyramid.
    Start from smallest level.
*/
void laplacianRecon(vector<Mat> &srcPyr, Mat &dstImg) {
    int level = srcPyr.size();
    // reconstruction from smallest level
    for (int i=level-2; i>=0; i--) {
        Mat upLevel;
        cv::pyrUp(srcPyr[i+1], upLevel, srcPyr[i].size());
        cv::add(srcPyr[i], upLevel, srcPyr[i]);
    }
    //Ensure size
    //resize(curLevel, curLevel, dstImg.size);
    dstImg = srcPyr[0]; // clone?
}

/*
    overLoad laplacianRecon use buffer.
*/
void laplacianRecon(vector<Mat> &srcPyr, vector<Mat> &bufferPyr, Mat &dstImg) {
    int level = srcPyr.size();
    // reconstruction from smallest level
    for (int i=level-2; i>=0; i--) {
        cv::pyrUp(srcPyr[i+1], bufferPyr[i], srcPyr[i].size());
        cv::add(srcPyr[i], bufferPyr[i], srcPyr[i]);
    }
    //Ensure size
    //resize(curLevel, curLevel, dstImg.size);
    dstImg = srcPyr[0]; // clone?
}

void laplacianRecon(vector<GpuMat> &srcPyr, GpuMat &dstImg, Stream &stream) {
    int level = srcPyr.size();
    // reconstruction from smallest level
    for (int i=level-2; i>=0; i--) {
        GpuMat upLevel;
        gpu::pyrUp(srcPyr[i+1], upLevel, stream); // No overLoad for gpu::pyrUp
        gpu::add(srcPyr[i], upLevel, srcPyr[i], GpuMat(), -1, stream);
    }
    //Ensure size
    //resize(curLevel, curLevel, dstImg.size);
    dstImg = srcPyr[0]; // Use copy here?
}

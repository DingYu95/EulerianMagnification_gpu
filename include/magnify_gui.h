#ifndef MAGNIFY_GUI_H
#define MAGNIFY_GUI_H

#include <opencv2/opencv.hpp>

extern double lambda_c;
extern double alphaMax;
extern double chromaAttenuation;
extern double exaFactor;
extern int pyrLevel;
extern double cutoffHigh;
extern double cutoffLow;

extern int alphaMaxSliderMax;
extern int alphaMaxSilder;

extern int ratioSliderMax;
extern int cutoffHighSlider;
extern int cutoffLowSlider;

extern int chromaSliderMax;
extern int chromaSlider;

void alphaMaxTrackbar(int, void*);

void cutoffHighTrackbar(int, void*);

void cutoffLowTrackbar(int, void*);

void chromaTrackbar(int, void*);

#endif
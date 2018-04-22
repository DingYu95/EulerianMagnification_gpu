#include "../include/magnify_gui.h"

int alphaMaxSliderMax = 100;
int alphaMaxSilder = 10;
int ratioSliderMax = 100;
int cutoffHighSlider = 70;
int cutoffLowSlider = 10;
int chromaSliderMax = 10;
int chromaSlider = 5;

void alphaMaxTrackbar(int, void*) {
    alphaMax = (double) alphaMaxSilder;
}

void cutoffHighTrackbar(int, void*) {
    cutoffHigh = (double) cutoffHighSlider * 1.0 / ratioSliderMax;
}

void cutoffLowTrackbar(int, void*) {
    cutoffLow = (double) cutoffLowSlider * 1.0 / ratioSliderMax;
}

void chromaTrackbar(int, void*) {
    chromaAttenuation = (double) chromaSlider * 1.0 / chromaSliderMax;
}

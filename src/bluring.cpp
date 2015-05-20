#include "bluring.h"

void gaussianBlurGrayscaleImage(GrayscaleImage& src, GrayscaleImage& dst, int radius, float sigma) {
    linearFilterGrayscaleImage(src, dst, gaussKernel(radius, sigma), radius);
}

void boxBlurGrayscaleImage(GrayscaleImage& src, GrayscaleImage& dst, int radius) {
    linearFilterGrayscaleImage(src, dst, boxKernel(radius), radius);
}

void gaussianBlurColorImage(RGBImage src, RGBImage dst, int radius, float sigma) {
    linearFilterColorImage(src, dst, gaussKernel(radius, sigma), radius);
}

void boxBlurColorImage(RGBImage src, RGBImage dst, int radius) {
    linearFilterColorImage(src, dst, boxKernel(radius), radius);
}
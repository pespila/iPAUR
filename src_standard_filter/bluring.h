#include "image.h"
#include "grayscale.h"
#include "rgb.h"
#include "create_filter.h"
#include "linear_filter.h"

#ifndef __BLURING_H__
#define __BLURING_H__

void gaussianBlurGrayscaleImage(GrayscaleImage&, GrayscaleImage&, int, float);
void boxBlurGrayscaleImage(GrayscaleImage&, GrayscaleImage&, int);
void gaussianBlurColorImage(RGBImage, RGBImage, int, float);
void boxBlurColorImage(RGBImage, RGBImage, int);

#endif //__BLURING_H__
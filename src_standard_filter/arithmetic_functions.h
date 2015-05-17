#include "image.h"
#include "grayscale.h"
#include "rgb.h"

#ifndef __ARTIHMETICFUNCTIONS_H__
#define __ARTIHMETICFUNCTIONS_H__

void markRed(GrayscaleImage&, RGBImage&, int);
void addGrayscaleImages(GrayscaleImage&, GrayscaleImage&, GrayscaleImage&);
void multiplyGrayscaleImages(GrayscaleImage&, GrayscaleImage&, GrayscaleImage&);
void addColorImages(RGBImage&, RGBImage&, RGBImage&);
void inRangeGrayscaleImage(GrayscaleImage&, GrayscaleImage&, int, int);
void inRangeColorImage(RGBImage&, RGBImage&, int, int, int, int, int, int);

#endif //__ARTIHMETICFUNCTIONS_H__
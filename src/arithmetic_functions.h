#include "image.h"
#include "grayscale.h"
#include "rgb.h"

#ifndef __ARTIHMETIC_FUNCTIONS_H__
#define __ARTIHMETIC_FUNCTIONS_H__

void mark_red(GrayscaleImage&, RGBImage&, int);
void add_images(GrayscaleImage&, GrayscaleImage&, GrayscaleImage&);
// void multiplyGrayscaleImages(GrayscaleImage&, GrayscaleImage&, GrayscaleImage&);
// void addColorImages(RGBImage&, RGBImage&, RGBImage&);
// void inRangeGrayscaleImage(GrayscaleImage&, GrayscaleImage&, int, int);
// void inRangeColorImage(RGBImage&, RGBImage&, int, int, int, int, int, int);

#endif //__ARTIHMETIC_FUNCTIONS_H__
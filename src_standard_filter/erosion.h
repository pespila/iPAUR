#include "image.h"
#include "grayscale.h"
#include "rgb.h"

#ifndef __EROSION_H__
#define __EROSION_H__

void erosionGrayscaleImage(GrayscaleImage&, GrayscaleImage&, int*, int);
void erosionColorImage(RGBImage&, RGBImage&, int*, int);

#endif //__EROSION_H__
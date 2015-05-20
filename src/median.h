#include "image.h"
#include "grayscale.h"
#include "rgb.h"

#ifndef __MEDIAN_H__
#define __MEDIAN_H__

unsigned char getMedian(unsigned char*);
void medianFilterGrayscaleImage(GrayscaleImage&, GrayscaleImage&);
void medianFilterColorImage(RGBImage&, RGBImage&);

#endif //__MEDIAN_H__
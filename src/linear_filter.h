#include "image.h"
#include "grayscale.h"
#include "rgb.h"

#ifndef __LINEARFILTER_H__
#define __LINEARFILTER_H__

void linearFilterGrayscaleImage(GrayscaleImage&, GrayscaleImage&, float*, int);
void linearFilterColorImage(RGBImage&, RGBImage&, float*, int);

#endif //__LINEARFILTER_H__
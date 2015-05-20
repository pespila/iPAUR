#include "image.h"
#include "grayscale.h"
#include "rgb.h"

#ifndef __DILATATION_H__
#define __DILATATION_H__

void dilatationGrayscaleImage(GrayscaleImage&, GrayscaleImage&, int*, int);
void dilatationColorImage(RGBImage&, RGBImage&, int*, int);

#endif //__DILATATION_H__
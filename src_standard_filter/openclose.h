#include "image.h"
#include "rgb.h"
#include "grayscale.h"
#include "dilatation.h"
#include "erosion.h"

#ifndef __OPENCLOSE_H__
#define __OPENCLOSE_H__

void openGrayscaleImage(GrayscaleImage&, GrayscaleImage&, int*, int);
void closeGrayscaleImage(GrayscaleImage&, GrayscaleImage&, int*, int);
void openColorImage(RGBImage&, RGBImage&, int*, int);
void closeColorImage(RGBImage&, RGBImage&, int*, int);

#endif //__OPENCLOSE_H__
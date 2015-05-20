#include "image.h"
#include "rgb.h"
#include "grayscale.h"
#include "openclose.h"

#ifndef __TOPHATS_H__
#define __TOPHATS_H__

void whiteTopHatGrayscaleImage(GrayscaleImage&, GrayscaleImage&, int*, int);
void blackTopHatGrayscaleImage(GrayscaleImage&, GrayscaleImage&, int*, int);
void whiteTopHatColorImage(RGBImage&, RGBImage&, int*, int);
void blackTopHatColorImage(RGBImage&, RGBImage&, int*, int);

#endif //__TOPHATS_H__
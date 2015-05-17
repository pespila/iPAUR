#include <cmath>
#include "image.h"
#include "rgb.h"
#include "grayscale.h"
#include "hsi.h"
#include "ycrcb.h"

#ifndef __TYPECONVERSION_H__
#define __TYPECONVERSION_H__

void rgb2gray(RGBImage&, GrayscaleImage&);
void rgb2ycrcb(RGBImage&, YCrCbImage&);
void rgb2hsi(RGBImage&, HSIImage&);

#endif //__TYPECONVERSION_H__
#include "image.h"
#include "rgb.h"
#include "grayscale.h"

#ifndef __INVERSE_H__
#define __INVERSE_H__

void inverseGrayscaleImage(GrayscaleImage&, GrayscaleImage&);
void inverseColorImage(RGBImage&, RGBImage&);

#endif //__INVERSE_H__
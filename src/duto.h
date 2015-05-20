#include "image.h"
#include "linear_filter.h"

#ifndef __DUTO_H__
#define __DUTO_H__

void dutoGrayscaleImage(GrayscaleImage&, GrayscaleImage&, float*, int, float);
void dutoColorImage(RGBImage&, RGBImage&, float*, int, float);

#endif //__DUTO_H__
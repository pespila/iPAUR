#include "image.h"
#include "grayscale.h"
#include "rgb.h"

#ifndef __ARTIHMETIC_FUNCTIONS_H__
#define __ARTIHMETIC_FUNCTIONS_H__

void mark_red(GrayscaleImage&, RGBImage&, int);
void add_images(GrayscaleImage&, GrayscaleImage&, GrayscaleImage&);

#endif //__ARTIHMETIC_FUNCTIONS_H__
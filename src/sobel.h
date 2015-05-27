#include "image.h"
#include "grayscale.h"

#ifndef __SOBEL_H__
#define __SOBEL_H__

void sobel(GrayscaleImage&, GrayscaleImage&);
void sobel_operator(GrayscaleImage&, GrayscaleImage&, GrayscaleImage&);

#endif //__SOBEL_H__
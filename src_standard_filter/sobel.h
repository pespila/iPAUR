#include "image.h"
#include "grayscale.h"

#ifndef __SOBEL_H__
#define __SOBEL_H__

void sobel(GrayscaleImage&, GrayscaleImage&);
void sobelForCanny(GrayscaleImage&, GrayscaleImage&, GrayscaleImage&);

#endif //__SOBEL_H__
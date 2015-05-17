#include "image.h"
#include "grayscale.h"
#include "linear_filter.h"
#include "laplace.h"

#ifndef __LAPLACIANOFGAUSSIAN_H__
#define __LAPLACIANOFGAUSSIAN_H__

void laplacianOfGaussian(GrayscaleImage&, GrayscaleImage&, float**, int);

#endif //__LAPLACIANOFGAUSSIAN_H__
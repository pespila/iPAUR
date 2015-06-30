#include "image.h"
#include "create_filter.h"
#include "linear_filter.h"

#ifndef __BLURING_H__
#define __BLURING_H__

void gaussian_blur(Image&, WriteableImage&, int, float);
void box_blur(Image&, WriteableImage&, int);

#endif //__BLURING_H__
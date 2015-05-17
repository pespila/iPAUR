#include "grayscale.h"
#include <iostream>

#ifndef __MSMINIMIZER_H__
#define __MSMINIMIZER_H__

struct dual
{
    double y1;
    double y2;
};

void init(float*, float*, struct dual*, GrayscaleImage&);
float project_x(float, float);
float project_y(float);
void primal_descent(float*, struct dual*, float, int, int);
void dual_ascent(float*, float*, struct dual*, float, int, int);
void primal_dual_algorithm(GrayscaleImage& src, GrayscaleImage& dst);
// void primal_dual_algorithm(GrayscaleImage& src, GrayscaleImage& dst, int);

#endif //__MSMINIMIZER_H__
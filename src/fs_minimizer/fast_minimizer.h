#include "image.h"

#ifndef __FAST_MINIMIZER_H__
#define __FAST_MINIMIZER_H__

void nabla(double*, double*, double*, int, int, int);
void vector_of_inner_product(double*, double*, double*, int, int, int);
void prox_r_star(double*, double*, double*, double*, double, double, double, int, int, int, int);
void nabla_transpose(double*, double*, double*, int, int, int);
void prox_d(double*, double*, double*, double, int);
void fast_minimizer(Image&, WriteableImage&, int, int);

#endif //__FAST_MINIMIZER_H__
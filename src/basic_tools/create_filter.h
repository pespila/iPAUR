#include <stdlib.h>
#include <math.h>

#ifndef __CREATE_FILTER_H__
#define __CREATE_FILTER_H__

#define PI 3.14159265359

float* gaussian_kernel(int, float);
float* box_kernel(int);
float* binomial_filter(int);
float* second_derivative();
int* morphological_standard_filter(int);
float** laplace_of_gauss_filter(int, float);

#endif //__CREATE_FILTER_H__
#include <cstdlib>
#include <cmath>

using namespace std;

#ifndef __CREATEFILTER_H__
#define __CREATEFILTER_H__

#define PI 3.14159265359

float* gaussKernel(int, float);
float* binomialFilter(int);
float* boxKernel(int);
float* secondDerivative();
int* morphologicalStandardFilter(int);
float** laplacianOfGaussianFilter(int, float);

#endif //__CREATEFILTER_H__
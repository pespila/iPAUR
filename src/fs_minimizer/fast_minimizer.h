#include "image.h"

#ifndef __FAST_MINIMIZER_H__
#define __FAST_MINIMIZER_H__

class Parameter
{
public:
	Parameter():alpha(20.0), lambda(0.1), tau(0.25), sigma(0.5), theta(1.0), cartoon(0) {}
	Parameter(float, float, float, float, float, int);
	~Parameter();

	float alpha;
	float lambda;
	float tau;
	float sigma;
	float theta;
	int cartoon;
};

void nabla(float*, float*, float*, int, int, int);
void vector_of_inner_product(float*, float*, float*, int, int, int);
void prox_r_star(float*, float*, float*, float*, float, float, float, int, int, int, int);
void nabla_transpose(float*, float*, float*, int, int, int);
void prox_d(float*, float*, float*, float, int);
void fast_minimizer(Image&, WriteableImage&, Parameter&, int);

#endif //__FAST_MINIMIZER_H__
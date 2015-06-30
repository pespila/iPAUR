#include "image.h"
#include "parameter.h"

#ifndef __TVL1_MODEL_H__
#define __TVL1_MODEL_H__

class TVL1_Minimizer
{
public:
	TVL1_Minimizer():steps(0), height(0), width(0), channel(0), size(0), f(NULL), u(NULL), u_n(NULL), u_bar(NULL), gradient_x(NULL), gradient_y(NULL), gradient_transpose(NULL), p_x(NULL), p_y(NULL) {}
	TVL1_Minimizer(Image&, int);
	~TVL1_Minimizer();

	void initialize(Image&);
	void set_solution(WriteableImage&);
	void nabla(float*, float*, float*);
	void vector_of_inner_product(float*, float*, float*);
	void prox_r_star(float*, float*, float*, float*, float, float);
	void nabla_transpose(float*, float*, float*);
	void prox_d(float*, float*, float*, float, float);
	void tvl1_model(Image&, WriteableImage&, Parameter&);

private:
	int steps;
	int height;
	int width;
	int channel;
	int size;
	float* f;
	float* u;
	float* u_n;
	float* u_bar;
	float* gradient_x;
	float* gradient_y;
	float* gradient_transpose;
	float* p_x;
	float* p_y;
};

#endif //__TVL1_MODEL_H__
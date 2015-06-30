#include "image.h"
#include "parameter.h"

#ifndef __HUBER_ROF_MODEL_H__
#define __HUBER_ROF_MODEL_H__

class Huber_ROF_Minimizer
{
public:
	Huber_ROF_Minimizer():steps(0), height(0), width(0), channel(0), size(0), f(NULL), u(NULL), u_n(NULL), u_bar(NULL), gradient_x(NULL), gradient_y(NULL), gradient_transpose(NULL), p_x(NULL), p_y(NULL) {}
	Huber_ROF_Minimizer(Image&, int);
	~Huber_ROF_Minimizer();

	void initialize(Image&);
	void set_solution(WriteableImage&);
	void nabla(float*, float*, float*);
	void prox_r_star(float*, float*, float*, float*, float, float);
	void nabla_transpose(float*, float*, float*);
	void prox_d(float*, float*, float*, float, float);
	void huber_rof_model(Image&, WriteableImage&, Parameter&);

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

#endif //__HUBER_ROF_MODEL_H__
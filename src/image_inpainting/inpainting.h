#include "image.h"
#include "parameter.h"

#ifndef __INPAINTING_H__
#define __INPAINTING_H__

class Image_Inpainting
{
public:
	Image_Inpainting():steps(0), height(0), width(0), channel(0), size(0), hash_table(NULL), f(NULL), u(NULL), u_n(NULL), u_bar(NULL), gradient_x(NULL), gradient_y(NULL), gradient_transpose(NULL), p_x(NULL), p_y(NULL) {}
	Image_Inpainting(Image&, int);
	~Image_Inpainting();

	void initialize(Image&);
	void set_solution(WriteableImage&);
	void nabla(float*, float*, float*);
	void prox_r_star(float*, float*, float*, float*, float, float);
	void nabla_transpose(float*, float*, float*);
	void prox_d(float*, float*, float*, unsigned char*, float, float);
	void inpainting(Image&, WriteableImage&, Parameter&);

private:
	int steps;
	int height;
	int width;
	int channel;
	int size;
	unsigned char* hash_table;
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

#endif //__INPAINTING_H__
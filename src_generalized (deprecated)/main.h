// #include "analytical_operators.h"

#ifndef __MAIN_H__
#define __MAIN_H__

struct grayscaled_image
{
	unsigned char* image_data;
	unsigned char* approximation;
	double* iterative_data;
	int image_height;
	int image_width;
	char image_type;
};

typedef struct grayscaled_image gray_img;

unsigned char* alloc_image_data(int, int);
double* alloc_double_array(int, int);
void run(int, const char**);
// void set_parameter(struct parameter);

#endif //__MAIN_H__
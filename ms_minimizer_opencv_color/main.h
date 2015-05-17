// #include "analytical_operators.h"

#ifndef __MAIN_H__
#define __MAIN_H__

struct color_image
{
	unsigned char* red_data;
	unsigned char* green_data;
	unsigned char* blue_data;
	unsigned char* red_approximation;
	unsigned char* green_approximation;
	unsigned char* blue_approximation;
	double* red_iterative_data;
	double* green_iterative_data;
	double* blue_iterative_data;
	int image_height;
	int image_width;
	char image_type;
};

typedef struct color_image color_img;

unsigned char** alloc_image_data(int, int);
double** alloc_double_array(int, int);
void run(int, const char**);
// void set_parameter(struct parameter);

#endif //__MAIN_H__
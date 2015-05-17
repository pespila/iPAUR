#include <stdlib.h>
#include <png.h>

#ifndef __MS_MINIMIZER_H__
#define __MS_MINIMIZER_H__

struct grayscaled_image
{
	unsigned char* approximation;
	float* iterative_data;
	int image_height;
	int image_width;
	png_byte bit_depth;
	int color_type;
};

typedef struct grayscaled_image gray_img;

struct dual_vector_2d {
	float* x1;
	float* x2;
};

struct parameter
{
	float sigma;
	float tau;
	float lambda;
	float theta;
	float alpha;
	float gamma;
	float mu;
	float delta;
	float L2;
};

void add_vectors(float*, float*, float*, float, float, int, int);
void gradient_of_image_value(struct dual_vector_2d*, float, float*, int, int, int);
void divergence_of_dual_vector(float*, float, struct dual_vector_2d*, int, int, int);
void huber_rof_proximation_f_star(struct dual_vector_2d* proximated, struct dual_vector_2d*, struct parameter*, int, int);
void proximation_g(float*, float*, gray_img*, struct parameter*);
void proximation_tv_l1_g(float*, float*, gray_img*, struct parameter*);

struct parameter* set_input_parameter(gray_img*, float, float, float, float, float, float, int);
void init_vectors(float*, float*, struct dual_vector_2d*, gray_img*, int, int);
void update_input_parameters(struct parameter*);
void update_fast_ms_parameter(struct parameter*);
void free_memory_of_vectors(float*, float*, float*, float*, struct dual_vector_2d*, struct dual_vector_2d*);
void set_approximation(gray_img*, float*);
void primal_dual_algorithm(gray_img*, void (*prox_f)(struct dual_vector_2d*, struct dual_vector_2d*, struct parameter*, int, int), void (*prox_g)(float*, float*, gray_img*, struct parameter*), struct parameter*, int, int, int);

#endif //__MS_MINIMIZER_H__
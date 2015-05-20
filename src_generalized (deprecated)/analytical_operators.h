#include "main.h"

#ifndef __ANALYTICAL_OPERATORS_H__
#define __ANALYTICAL_OPERATORS_H__

struct dual_vector_2d {
	double* x1;
	double* x2;
};

struct parameter
{
	double sigma;
	double tau;
	double lambda;
	double theta;
	double alpha;
	double gamma;
	double mu;
	double delta;
	double L2;
};

void add_vectors(double*, double*, double*, double, double, int, int);
void gradient_of_image_value(struct dual_vector_2d, double, double*, int, int, int);
void divergence_of_dual_vector(double*, double, struct dual_vector_2d, int, int, int);
void huber_rof_proximation_f_star(struct dual_vector_2d proximated, struct dual_vector_2d, struct parameter*, int, int);
void proximation_fast_ms_minimizer_f_star(struct dual_vector_2d proximated, struct dual_vector_2d, struct parameter*, int, int);
void proximation_g(double*, double*, gray_img*, struct parameter*);
void proximation_fast_ms_minimizer_g(double*, double*, gray_img*, struct parameter*);
void proximation_tv_l1_g(double*, double*, gray_img*, struct parameter*);

#endif //__ANALYTICAL_OPERATORS_H__
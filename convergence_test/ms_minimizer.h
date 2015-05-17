#include <stdlib.h>
#include "main.h"

#ifndef __MS_MINIMIZER_H__
#define __MS_MINIMIZER_H__

struct parameter* set_input_parameter(gray_img*, double, double, double, double, double, double, int);
void init_vectors(double*, double*, struct dual_vector_2d, struct dual_vector_2d, gray_img*, int, int);
void update_input_parameters(struct parameter*);
void update_fast_ms_parameter(struct parameter*);
void free_memory_of_vectors(double*, double*, double*, double*, double*, struct dual_vector_2d, struct dual_vector_2d, struct dual_vector_2d);
void set_approximation(gray_img*, double*);
void primal_dual_algorithm(gray_img*, void (*prox_f)(struct dual_vector_2d, struct dual_vector_2d, struct parameter*, int, int), void (*prox_g)(double*, double*, gray_img*, struct parameter*), struct parameter*, int, int, int, const char*);

#endif //__MS_MINIMIZER_H__
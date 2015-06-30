#include "image.h"

#ifndef __PRIMAL_DUAL_H__
#define __PRIMAL_DUAL_H__

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

struct vec_3d
{
	double x1;
	double x2;
	double x3;
};

struct matrix_2_times_2
{
	double a11;
	double a12;
	double a21;
	double a22;
};

typedef struct vec_3d vec;
typedef struct matrix_2_times_2 Matrix;

void gradient_operator(vec*, float*, int, int, int);
void divergence_operator(float*, vec*, int, int, int);
double f(double, double, double, double, double, int);
const inline double f_prime(double);
const inline double f_prime_prime();
const inline double schwarz();
inline double vector_norm(vec*, int);
void soft_shrinkage_operator(vec*, vec*, float, int);
void truncation_operator(float*, float*, int);
void scale_vector(vec*, float, float, vec*, int);
void add_vector(vec*, float, float, vec*, vec*, int);
void add_array(float*, float, float, float*, float*, int);
void copy_array(float*, float*, int);
void extrapolation(float*, float, float*, float*, int);
void newton_parabola(vec*, vec*, double, double, double* image, int, int, int);
void dykstra_projection(vec*, vec*, double*, param*, int, int, int, int);
inline void init_data_structures(float* image, float*, float*, float*, vec*, vec*, int, int, int, gray_img*);
inline void free_data_structures(float* image, float*, float*, float*, float*, vec*, vec*);
void primal_dual(gray_img*, param*, int, int);

#endif //__PRIMAL_DUAL_H__
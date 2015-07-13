#include "grayscale.h"
#include "parameter.h"

#ifndef __PRIMAL_DUAL_ALGORITHM_H__
#define __PRIMAL_DUAL_ALGORITHM_H__

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

class Primal_Dual
{
public:
	Primal_Dual():steps(0), height(0), width(0), size(0), level(0), f(NULL), u(NULL), u_n(NULL), u_bar(NULL), gradient_x(NULL), gradient_y(NULL), gradient_z(NULL), gradient_transpose(NULL), p_x(NULL), p_y(NULL), p_z(NULL), x1(NULL), x2(NULL), x3(NULL), y1(NULL), y2(NULL), y3(NULL), p1(NULL), p2(NULL), p3(NULL), q1(NULL), q2(NULL), q3(NULL), z1(NULL), z2(NULL), g1(NULL), g2(NULL), delta1(NULL), delta2(NULL) {}
	Primal_Dual(GrayscaleImage&, int, int);
	~Primal_Dual();

	void Initialize(GrayscaleImage&);
	void SetSolution(GrayscaleImage&);
	void Nabla(float*, float*, float*, float*);
	void soft_shrinkage_operator(float*, float*, float*, float);
	void newton_projection(float*, float*, float*, float*, float, float);
	float constraint(float, float, float, float, float, int);
	// void VectorOfInnerProduct(float*, float*, float*);
	void dykstra_algorithm(float*, float*, float*, float*, float*, float*, float*, float, float, float);
	void NablaTranspose(float*, float*, float*, float*);
	void truncation_operator(float*, float*);
	void primal_dual_algorithm(GrayscaleImage&, GrayscaleImage&, Parameter&);

private:
	int steps;
	int height;
	int width;
	int size;
	int level;
	float* f;
	float* u;
	float* u_n;
	float* u_bar;
	float* gradient_x;
	float* gradient_y;
	float* gradient_z;
	float* gradient_transpose;
	float* p_x;
	float* p_y;
	float* p_z;
	float* x1;
	float* x2;
	float* x3;
	float* y1;
	float* y2;
	float* y3;
	float* p1;
	float* p2;
	float* p3;
	float* q1;
	float* q2;
	float* q3;
	float* z1;
	float* z2;
	float* g1;
	float* g2;
	float* delta1;
	float* delta2;
};

#endif //__PRIMAL_DUAL_ALGORITHM_H__
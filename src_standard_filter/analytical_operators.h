#include <string>
#include "image.h"

#ifndef __ANALYTICAL_OPERATORS_H__
#define __ANALYTICAL_OPERATORS_H__

class Analytical_Operators
{
public:
	Analytical_Operators():height(0), width(0), dual_x1(NULL), dual_x2(NULL), dual_proximated1(NULL), dual_proximated2(NULL), sigma(0.0), tau(0.0), lambda(0.0), theta(0.0), alpha(0.0), L2(0.0) {}
	Analytical_Operators(int, int, double, double, double, double);
	~Analytical_Operators();

	// virtual void proximation_g() = 0;
	// virtual void proximation_f_star() = 0;

	void add_vectors(double, double, double*, double*, double*);
	void add_dual_variables(double, double);
	void update_theta_tau_sigma(void);
	void gradient_of_image_value(double, double*);
	void divergence_of_dual_vector(double, double*);

	int height;
	int width;

	double* dual_x1;
	double* dual_x2;
	double* dual_proximated1;
	double* dual_proximated2;
	double sigma;
	double tau;
	double lambda;
	double theta;
	double alpha;
	double gamma;
	double L2;

};

#endif //__ANALYTICAL_OPERATORS_H__
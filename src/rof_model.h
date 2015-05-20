#include "grayscale.h"
#include "analytical_operators.h"

#ifndef __HUBER_ROF_MODEL_H__
#define __HUBER_ROF_MODEL_H__

class Huber_ROF_Model : public Analytical_Operators
{
public:
	Huber_ROF_Model() : Analytical_Operators() {x = NULL, x_bar = NULL, x_current = NULL, divergence = NULL;}
	Huber_ROF_Model(int, int, double, double, double, double);
	// Huber_ROF_Model(int height, int width, double tau, double lambda, double theta, double alpha) : Analytical_Operators(height, width, tau, lambda, theta, alpha) {};
	~Huber_ROF_Model();

	void init_vectors(GrayscaleImage&);
	void proximation_g(GrayscaleImage&, double*, double*);
	void proximation_f_star();
	void set_approximation(GrayscaleImage&);
	void primal_dual_algorithm(GrayscaleImage&, GrayscaleImage&);

	double* x;
	double* x_bar;
	double* x_current;
	double* divergence;
};

#endif //__HUBER_ROF_MODEL_H__
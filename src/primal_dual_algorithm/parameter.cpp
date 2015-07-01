#include "parameter.h"

Parameter::Parameter(float nu, float lambda, float tau, float sigma, float theta, float L) {
	this->nu = nu;
	this->lambda = lambda;
	this->tau = tau;
	this->sigma = sigma;
	this->theta = theta;
	this->L = L;
}

Parameter::~Parameter(){}
#include "parameter.h"

Parameter::Parameter(float alpha, float lambda, float tau, float theta, int size) {
	this->alpha = alpha;
	this->lambda = lambda;
	this->tau = tau;
	this->theta = theta;
	this->sigma = 1.0 / (tau * 8.0 * size);
}

Parameter::~Parameter(){}
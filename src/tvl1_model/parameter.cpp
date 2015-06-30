#include "parameter.h"

Parameter::Parameter(float alpha, float lambda, float tau, float sigma, float theta) {
	this->alpha = alpha;
	this->lambda = lambda;
	this->tau = tau;
	this->sigma = sigma;
	this->theta = theta;
}

Parameter::~Parameter(){}
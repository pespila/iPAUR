#include <cmath>

#ifndef __PARAMETER_H__
#define __PARAMETER_H__

class Parameter
{
public:
	Parameter():nu(5.0), lambda(0.1), tau(1.0 / sqrt(12)), sigma(1.0 / sqrt(12)), theta(1.0), L(sqrt(12)) {}
	Parameter(float, float, float, float, float, float);
	~Parameter();

	float nu;
	float lambda;
	float tau;
	float sigma;
	float theta;
	float L;
};

#endif //__PARAMETER_H__
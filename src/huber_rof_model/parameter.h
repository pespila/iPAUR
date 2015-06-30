#ifndef __PARAMETER_H__
#define __PARAMETER_H__

class Parameter
{
public:
	Parameter():alpha(0.05), lambda(32.0), tau(0.01), sigma(0.0), theta(1.0) {}
	Parameter(float, float, float, float, int);
	~Parameter();

	float alpha;
	float lambda;
	float tau;
	float sigma;
	float theta;
};

#endif //__PARAMETER_H__
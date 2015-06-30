#ifndef __PARAMETER_H__
#define __PARAMETER_H__

class Parameter
{
public:
	Parameter():alpha(0.05), lambda(0.7), tau(0.35), sigma(1.0 / (0.35 * 8.0)), theta(1.0) {}
	Parameter(float, float, float, float, float);
	~Parameter();

	float alpha;
	float lambda;
	float tau;
	float sigma;
	float theta;
};

#endif //__PARAMETER_H__
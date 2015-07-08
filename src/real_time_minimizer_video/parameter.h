#ifndef __PARAMETER_H__
#define __PARAMETER_H__

class Parameter
{
public:
	Parameter():alpha(20.0), lambda(0.1), tau(0.25), sigma(0.5), theta(1.0), cartoon(0) {}
	Parameter(float, float, float, float, float, int);
	~Parameter();

	float alpha;
	float lambda;
	float tau;
	float sigma;
	float theta;
	int cartoon;
};

#endif //__PARAMETER_H__
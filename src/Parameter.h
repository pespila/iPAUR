/* FIT THE (GOOD DEFAULT) PARAMETER TO THE ALGORTIHMS
	- Huber-ROF-Model: alpha = 0.05, lambda = 32.0, tau = 0.01, sigma = 0.0, theta = 1.0, cartoon = -1
	- Image Inpainting: alpha = 0.05, lambda = 32.0, tau = 0.01, sigma = 0.0, theta = 1.0, cartoon = -1
	- TVL1-Model: alpha = 0.05, lambda = 0.7, tau = 0.35, sigma = 1.0 / (0.35 * 8.0f), theta = 1.0, cartoon = -1
	- Real-Time-Minimizer: alpha = 20.0, lambda = 0.1, tau = 0.25, sigma = 0.5, theta = 1.0, cartoon = 1/0
*/

#include <cmath>

#ifndef __PARAMETER_H__
#define __PARAMETER_H__

template<typename aType>
class Parameter
{
public:
	Parameter():alpha(20.0f), lambda(0.01f), tau(0.25f), sigma(0.5f), theta(1.0f), L(sqrt(12.f)), nu(5.f), cartoon(1) {}
	Parameter(aType, aType, aType, aType, aType, aType, aType, int);
	~Parameter();

	aType alpha;
	aType lambda;
	aType tau;
	aType sigma;
	aType theta;
	aType L;
	aType nu;
	int cartoon;
};

#include "Parameter.tpp"

#endif //__PARAMETER_H__
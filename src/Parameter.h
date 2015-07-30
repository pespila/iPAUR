/* FIT THE (GOOD DEFAULT) PARAMETER TO THE ALGORTIHMS
	- Huber-ROF-Model: alpha = 0.05, lambda = 32.0, tau = 0.01, sigma = 0.0, theta = 1.0, cartoon = -1
	- Image Inpainting: alpha = 0.05, lambda = 32.0, tau = 0.01, sigma = 0.0, theta = 1.0, cartoon = -1
	- TVL1-Model: alpha = 0.05, lambda = 0.7, tau = 0.35, sigma = 1.0 / (0.35 * 8.0), theta = 1.0, cartoon = -1
	- Real-Time-Minimizer: alpha = 20.0, lambda = 0.1, tau = 0.25, sigma = 0.5, theta = 1.0, cartoon = 1/0
*/

#include <cmath>

#ifndef __PARAMETER_H__
#define __PARAMETER_H__

class Parameter
{
public:
	Parameter():alpha(20.0), lambda(0.01), tau(0.25), sigma(0.5), theta(1.0), L(sqrt(12)), nu(5), cartoon(1) {}
	Parameter(float, float, float, float, float, float, float, int);
	~Parameter();

	float alpha;
	float lambda;
	float tau;
	float sigma;
	float theta;
	float L;
	float nu;
	int cartoon;
};


#endif //__PARAMETER_H__
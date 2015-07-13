#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include "../Image/Image.h"
#include "../Parameter/Parameter.h"

#ifndef __HUBERROFMODEL_H__
#define __HUBERROFMODEL_H__

class HuberROFModel
{
private:
	int steps;
	int height;
	int width;
	int channel;
	int size;
	float* f;
	float* u;
	float* u_n;
	float* u_bar;
	float* gradient_x;
	float* gradient_y;
	float* gradient_transpose;
	float* p_x;
	float* p_y;
	
	void Initialize(Image&);
	void SetSolution(WriteableImage&);
	void Nabla(float*, float*, float*);
	void ProxRstar(float*, float*, float*, float*, float, float);
	void NablaTranspose(float*, float*, float*);
	void ProxD(float*, float*, float*, float, float);

public:
	HuberROFModel():steps(0), height(0), width(0), channel(0), size(0), f(NULL), u(NULL), u_n(NULL), u_bar(NULL), gradient_x(NULL), gradient_y(NULL), gradient_transpose(NULL), p_x(NULL), p_y(NULL) {}
	HuberROFModel(Image&, int);
	~HuberROFModel();

	void HuberROF(Image&, WriteableImage&, Parameter&);
};

#endif //__HUBERROFMODEL_H__
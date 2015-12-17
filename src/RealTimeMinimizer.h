#include <math.h>
#include "Image.h"
#include "Parameter.h"

#ifndef __REALTIMEMINIMIZER_H__
#define __REALTIMEMINIMIZER_H__

template<typename aType>
class RealTimeMinimizer
{
private:
	int steps;
	int height;
	int width;
	int channel;
	int size;
	aType* f;
	aType* u;
	aType* u_n;
	aType* u_bar;
	aType* gradient_x;
	aType* gradient_y;
	aType* gradient_transpose;
	aType* p_x;
	aType* p_y;
	
	void Initialize(Image<aType>&);
	void SetSolution(WriteableImage<aType>&);
	void Nabla(aType*, aType*, aType*, aType*, aType*, aType);
	void VectorOfInnerProduct(aType*, aType*, aType*);
	void ProxRstar(aType*, aType*, aType*, aType*, aType, aType, aType, int);
	void NablaTranspose(aType*, aType*, aType*, aType*, aType);
	void ProxD(aType*, aType*, aType*, aType);
	void Extrapolation(aType*, aType*, aType*, aType);

public:
	RealTimeMinimizer():steps(0), height(0), width(0), channel(0), size(0), f(NULL), u(NULL), u_n(NULL), u_bar(NULL), gradient_x(NULL), gradient_y(NULL), gradient_transpose(NULL), p_x(NULL), p_y(NULL) {}
	RealTimeMinimizer(Image<aType>&, int);
	~RealTimeMinimizer();

	void RTMinimizer(Image<aType>&, WriteableImage<aType>&, Parameter<aType>&);
};

#include "RealTimeMinimizer.tpp"

#endif //__REALTIMEMINIMIZER_H__
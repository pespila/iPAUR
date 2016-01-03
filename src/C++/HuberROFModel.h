#include <cmath>
#include <iostream>
#include "Image.h"

using namespace std;

#ifndef __HUBERROFMODEL_H__
#define __HUBERROFMODEL_H__

template<typename aType>
class HuberROFModel
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
	void SetSolution(Image<aType>&);
	void Nabla(aType*, aType*, aType*, aType*, aType*, aType);
	void ProxRstar(aType*, aType*, aType*, aType*, aType, aType);
	void NablaTranspose(aType*, aType*, aType*, aType*, aType);
	void ProxD(aType*, aType*, aType*, aType, aType);
	void Extrapolation(aType*, aType*, aType*, aType);

public:
	HuberROFModel():steps(0), height(0), width(0), channel(0), size(0), f(NULL), u(NULL), u_n(NULL), u_bar(NULL), gradient_x(NULL), gradient_y(NULL), gradient_transpose(NULL), p_x(NULL), p_y(NULL) {}
	HuberROFModel(Image<aType>&, int);
	~HuberROFModel();

	void HuberROF(Image<aType>&, Image<aType>&, aType, aType, aType);
};

#include "HuberROFModel.tpp"

#endif //__HUBERROFMODEL_H__
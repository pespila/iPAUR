#include <cmath>
#include <iostream>
#include "Image.h"

using namespace std;

#ifndef __ROFMODEL_H__
#define __ROFMODEL_H__

template<typename aType>
class ROFModel
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
	void ProxRstar(aType*, aType*, aType*, aType*);
	void NablaTranspose(aType*, aType*, aType*, aType*, aType);
	void ProxD(aType*, aType*, aType*, aType, aType);
	void Extrapolation(aType*, aType*, aType*, aType);
	aType Energy(aType*, aType*, aType*, aType*, aType);
	aType PrimalEnergy(aType*, aType*, aType);
	aType DualEnergy(aType*, aType*, aType*, aType);
	
public:
	ROFModel():steps(0), height(0), width(0), channel(0), size(0), f(NULL), u(NULL), u_n(NULL), u_bar(NULL), gradient_x(NULL), gradient_y(NULL), gradient_transpose(NULL), p_x(NULL), p_y(NULL) {}
	ROFModel(Image<aType>&, int);
	~ROFModel();

	void ROF(Image<aType>&, Image<aType>&, aType, aType);
};

#include "ROFModel.tpp"

#endif //__ROFMODEL_H__
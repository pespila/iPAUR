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
	aType* u_bar;
	aType* p_x;
	aType* p_y;
	
	void Initialize(Image<aType>&);
	void DualAscent(aType*, aType*, aType*, aType, aType);
	aType PrimalDescent(aType*, aType*, aType*, aType*, aType, aType, aType);
	void SetSolution(Image<aType>&);
	
public:
	HuberROFModel():steps(0), height(0), width(0), channel(0), size(0), f(NULL), u(NULL), u_bar(NULL), p_x(NULL), p_y(NULL) {}
	HuberROFModel(Image<aType>&, int);
	~HuberROFModel();

	void HuberROF(Image<aType>&, Image<aType>&, aType, aType, aType);
};

#include "HuberROFModel.tpp"

#endif //__HUBERROFMODEL_H__
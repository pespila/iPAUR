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
	aType* u_bar;
	aType* p_x;
	aType* p_y;
	
	void Initialize(Image<aType>&);
	void DualAscent(aType*, aType*, aType*, aType);
	aType PrimalDescent(aType*, aType*, aType*, aType*, aType, aType, aType);
	void SetSolution(Image<aType>&);
	
public:
	ROFModel():steps(0), height(0), width(0), channel(0), size(0), f(NULL), u(NULL), u_bar(NULL), p_x(NULL), p_y(NULL) {}
	ROFModel(Image<aType>&, int);
	~ROFModel();

	void ROF(Image<aType>&, Image<aType>&, aType, aType);
};

#include "ROFModel.tpp"

#endif //__ROFMODEL_H__
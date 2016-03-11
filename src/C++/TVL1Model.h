#include <cmath>
#include <iostream>
#include "Image.h"

using namespace std;

#ifndef __TVL1MODEL_H__
#define __TVL1MODEL_H__

template<typename aType>
class TVL1Model
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
	TVL1Model():steps(0), height(0), width(0), channel(0), size(0), f(NULL), u(NULL), u_bar(NULL), p_x(NULL), p_y(NULL) {}
	TVL1Model(Image<aType>&, int);
	~TVL1Model();

	void TVL1(Image<aType>&, Image<aType>&, aType, aType);
};

#include "TVL1Model.tpp"

#endif //__TVL1MODEL_H__
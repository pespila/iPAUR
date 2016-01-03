#include <cmath>
#include <iostream>
#include "Image.h"

using namespace std;

#ifndef __TVL1Model_H__
#define __TVL1Model_H__

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
	
public:
	TVL1Model():steps(0), height(0), width(0), channel(0), size(0), f(NULL), u(NULL), u_n(NULL), u_bar(NULL), gradient_x(NULL), gradient_y(NULL), gradient_transpose(NULL), p_x(NULL), p_y(NULL) {}
	TVL1Model(Image<aType>&, int);
	~TVL1Model();

	void TVL1(Image<aType>&, Image<aType>&, aType, aType);
};

#include "TVL1Model.tpp"

#endif //__TVL1Model_H__
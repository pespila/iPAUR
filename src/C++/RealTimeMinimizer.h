#include <cmath>
#include <iostream>
#include "Image.h"

using namespace std;

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
	aType* u_bar;
	aType* p_x;
	aType* p_y;
	
	void Initialize(Image<aType>&);
	void DualAscent(aType*, aType*, aType*, aType, aType, aType);
	aType PrimalDescent(aType*, aType*, aType*, aType*, aType*, aType, aType);
	void EdgeHighlighting(aType*, aType, aType);
	void SetSolution(Image<aType>&);
	
public:
	RealTimeMinimizer():steps(0), height(0), width(0), channel(0), size(0), f(NULL), u(NULL), u_bar(NULL), p_x(NULL), p_y(NULL) {}
	RealTimeMinimizer(Image<aType>&, int);
	~RealTimeMinimizer();

	void RTMinimizer(Image<aType>&, Image<aType>&, aType, aType, bool);
};

#include "RealTimeMinimizer.tpp"

#endif //__REALTIMEMINIMIZER_H__
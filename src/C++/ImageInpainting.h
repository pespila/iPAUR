#include <cmath>
#include <iostream>
#include "Image.h"

using namespace std;

#ifndef __IMAGEINPAINTING_H__
#define __IMAGEINPAINTING_H__

template<typename aType>
class ImageInpainting
{
private:
	int steps;
	int height;
	int width;
	int channel;
	int size;
	int* ht;
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
	ImageInpainting():steps(0), height(0), width(0), channel(0), size(0), f(NULL), u(NULL), u_bar(NULL), p_x(NULL), p_y(NULL), ht(NULL) {}
	ImageInpainting(Image<aType>&, int);
	~ImageInpainting();

	void Inpaint(Image<aType>&, Image<aType>&, aType, aType);
};

#include "ImageInpainting.tpp"

#endif //__IMAGEINPAINTING_H__
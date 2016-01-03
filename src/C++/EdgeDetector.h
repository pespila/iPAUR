#include "Image.h"

#ifndef __EDGEDETECTOR_H__
#define __EDGEDETECTOR_H__

#define PI 3.14159265359

template<typename aType>
class EdgeDetector
{
private:
	int height;
	int width;
	char type;
	aType* gx;
	aType* gy;

	void NablaX(aType*, Image<aType>&, int);
	void NablaY(aType*, Image<aType>&, int);
	void EvalGradient(aType*, aType*, aType*);
	void NonMaximumSupression(aType*, aType*, aType*);
	void Hysteresis(Image<aType>&, aType*, const int, const int);
	void SetEdges(Image<aType>&, aType*, aType*);
public:
	EdgeDetector():height(0), width(0), type(0), gx(NULL), gy(NULL) {}
	EdgeDetector(Image<aType>&);
	~EdgeDetector();

	void Sobel(Image<aType>&, Image<aType>&);
	void Prewitt(Image<aType>&, Image<aType>&);
	void RobertsCross(Image<aType>&, Image<aType>&);
	void Laplace(Image<aType>&, Image<aType>&);
	void Canny(Image<aType>&, Image<aType>&, const int, const int);
};

#include "EdgeDetector.tpp"

#endif //__EDGEDETECTOR_H__
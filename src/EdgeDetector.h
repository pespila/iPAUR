#include "Image.h"
#include "Util.h"

#ifndef __EDGEDETECTOR_H__
#define __EDGEDETECTOR_H__

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
	void Hysteresis(WriteableImage<aType>&, aType*, const int, const int);
	void SetEdges(WriteableImage<aType>&, aType*, aType*);
public:
	EdgeDetector():height(0), width(0), type(0), gx(NULL), gy(NULL) {}
	EdgeDetector(int, int, char);
	EdgeDetector(Image<aType>&);
	~EdgeDetector();

	void Sobel(Image<aType>&, WriteableImage<aType>&);
	void Prewitt(Image<aType>&, WriteableImage<aType>&);
	void RobertsCross(Image<aType>&, WriteableImage<aType>&);
	void Laplace(Image<aType>&, WriteableImage<aType>&);
	void Canny(Image<aType>&, WriteableImage<aType>&, const int, const int);
};

#include "EdgeDetector.tpp"

#endif //__EDGEDETECTOR_H__
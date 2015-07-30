#include "Image.h"
#include "Util.h"

#ifndef __EDGEDETECTOR_H__
#define __EDGEDETECTOR_H__

class EdgeDetector
{
private:
	int height;
	int width;
	char type;
	short* gx;
	short* gy;

	void NablaX(short*, Image&, unsigned char);
	void NablaY(short*, Image&, unsigned char);
	void EvalGradient(short*, short*, short*);
	void NonMaximumSupression(short*, short*, short*);
	void Hysteresis(WriteableImage&, short*, const int, const int);
	void SetEdges(WriteableImage&, short*, short*);
public:
	EdgeDetector():height(0), width(0), type(0), gx(NULL), gy(NULL) {}
	EdgeDetector(int, int, char);
	EdgeDetector(Image&);
	~EdgeDetector();

	void Sobel(Image&, WriteableImage&);
	void Prewitt(Image&, WriteableImage&);
	void RobertsCross(Image&, WriteableImage&);
	void Laplace(Image&, WriteableImage&);
	void Canny(Image&, WriteableImage&, const int, const int);
};

#endif //__EDGEDETECTOR_H__
#include "image.h"

#ifndef __LINEARFILTER_H__
#define __LINEARFILTER_H__

#define PI 3.14159265359

class LinearFilter
{
private:
	int height;
	int width;
	int channel;
	char type;
	unsigned char* filtered;

	void CreateGaussFilter(float*, float, int, int);
	void CreateBinomialFilter(float*, int, int);
	void CreateBoxFilter(float*, int, int);
	void FilterDx(Image&, float*, int);
	void FilterDy(WriteableImage&, float*, int);

public:
	LinearFilter():height(0), width(0), channel(0), type(0) {}
	LinearFilter(int, int, int, char);
	LinearFilter(Image&);
	~LinearFilter();

	void Gauss(Image&, WriteableImage&, int, float);
	void Binomial(Image&, WriteableImage&, int);
	void Box(Image&, WriteableImage&, int);
};

#endif //__LINEARFILTER_H__
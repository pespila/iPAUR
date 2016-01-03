#include "Image.h"

#ifndef __LINEARFILTER_H__
#define __LINEARFILTER_H__

#define PI 3.14159265359

template<typename aType>
class LinearFilter
{
private:
	int height;
	int width;
	int channel;
	char type;
	aType* filtered;

	void CreateGaussFilter(aType*, aType, int, int);
	void CreateBinomialFilter(aType*, int, int);
	void CreateBoxFilter(aType*, int, int);
	void FilterDx(Image<aType>&, aType*, int);
	void FilterDy(Image<aType>&, aType*, int);

public:
	LinearFilter():height(0), width(0), channel(0), type(0) {}
	LinearFilter(Image<aType>&);
	~LinearFilter();

	void Gauss(Image<aType>&, Image<aType>&, int, aType);
	void Binomial(Image<aType>&, Image<aType>&, int);
	void Box(Image<aType>&, Image<aType>&, int);
	void Duto(Image<aType>&, Image<aType>&, int, aType, aType);
};

#include "LinearFilter.tpp"

#endif //__LINEARFILTER_H__
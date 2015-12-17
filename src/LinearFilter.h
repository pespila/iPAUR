#include "Image.h"
#include "Util.h"

#ifndef __LINEARFILTER_H__
#define __LINEARFILTER_H__

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
	void FilterDy(WriteableImage<aType>&, aType*, int);

public:
	LinearFilter():height(0), width(0), channel(0), type(0) {}
	LinearFilter(int, int, int, char);
	LinearFilter(Image<aType>&);
	~LinearFilter();

	void Gauss(Image<aType>&, WriteableImage<aType>&, int, aType);
	void Binomial(Image<aType>&, WriteableImage<aType>&, int);
	void Box(Image<aType>&, WriteableImage<aType>&, int);
	void Duto(Image<aType>&, WriteableImage<aType>&, int, aType);
};

#include "LinearFilter.tpp"

#endif //__LINEARFILTER_H__
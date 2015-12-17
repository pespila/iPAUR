#include "Image.h"
#include "Util.h"

#ifndef __MORPHOLOGICALFILTER_H__
#define __MORPHOLOGICALFILTER_H__

template<typename aType>
class MorphologicalFilter
{
private:
	int height;
	int width;
	int channel;
	char type;
	aType* filtered;

	void MedianOfArray(aType*, int);
	void CreateOnes(aType*, int);
	void FilterDx(Image<aType>&, aType*, int, char);
	void FilterDy(WriteableImage<aType>&, aType*, int, char);

public:
	MorphologicalFilter():height(0), width(0), channel(0), type(0) {}
	MorphologicalFilter(int, int, int, char);
	MorphologicalFilter(Image<aType>&);
	~MorphologicalFilter();

	void Erosion(Image<aType>&, WriteableImage<aType>&, int);
	void Dilatation(Image<aType>&, WriteableImage<aType>&, int);
	void Median(Image<aType>&, WriteableImage<aType>&, int);
	void Open(Image<aType>&, WriteableImage<aType>&, int);
	void Close(Image<aType>&, WriteableImage<aType>&, int);
	void WhiteTopHat(Image<aType>&, WriteableImage<aType>&, int);
	void BlackTopHat(Image<aType>&, WriteableImage<aType>&, int);
};

#include "MorphologicalFilter.tpp"

#endif //__MORPHOLOGICALFILTER_H__
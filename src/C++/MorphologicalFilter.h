#include "Image.h"

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
	void FilterDy(Image<aType>&, aType*, int, char);

public:
	MorphologicalFilter():height(0), width(0), channel(0), type(0) {}
	MorphologicalFilter(Image<aType>&);
	~MorphologicalFilter();

	void Inverse(Image<aType>&, Image<aType>&);
	void Erosion(Image<aType>&, Image<aType>&, int);
	void Dilatation(Image<aType>&, Image<aType>&, int);
	void Median(Image<aType>&, Image<aType>&, int);
	void Open(Image<aType>&, Image<aType>&, int);
	void Close(Image<aType>&, Image<aType>&, int);
};

#include "MorphologicalFilter.tpp"

#endif //__MORPHOLOGICALFILTER_H__
#include "../Image/Image.h"
#include "../Util/Util.h"

#ifndef __MORPHOLOGICALFILTER_H__
#define __MORPHOLOGICALFILTER_H__

class MorphologicalFilter
{
private:
	int height;
	int width;
	int channel;
	char type;
	unsigned char* filtered;

	void MedianOfArray(unsigned char*, int);
	void CreateOnes(float*, int);
	void FilterDx(Image&, float*, int, char);
	void FilterDy(WriteableImage&, float*, int, char);

public:
	MorphologicalFilter():height(0), width(0), channel(0), type(0) {}
	MorphologicalFilter(int, int, int, char);
	MorphologicalFilter(Image&);
	~MorphologicalFilter();

	void Erosion(Image&, WriteableImage&, int);
	void Dilatation(Image&, WriteableImage&, int);
	void Median(Image&, WriteableImage&, int);
	void Open(Image&, WriteableImage&, int);
	void Close(Image&, WriteableImage&, int);
	void WhiteTopHat(Image&, WriteableImage&, int);
	void BlackTopHat(Image&, WriteableImage&, int);
};

#endif //__MORPHOLOGICALFILTER_H__
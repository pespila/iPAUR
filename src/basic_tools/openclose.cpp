#include "openclose.h"

void open(Image& src, WriteableImage& dst, int* filter, int radius) {
	dilatation(src, dst, filter, radius);
	erosion(dst, dst, filter, radius);
}

void close(Image& src, WriteableImage& dst, int* filter, int radius) {
	erosion(src, dst, filter, radius);
	dilatation(dst, dst, filter, radius);
}
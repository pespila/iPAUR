#include "Image.h"
#include "GrayscaleImage.h"
#include "RGBImage.h"
#include "RGBAImage.h"
#include "HSIImage.h"
#include "YCrCbImage.h"

#ifndef __UTIL_H__
#define __UTIL_H__

#define PI 3.14159265359

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

template<typename aType>
class Util
{
public:
	Util();
	~Util();

	void MarkRed(RGBImage<aType>&, GrayscaleImage<aType>&, RGBImage<aType>&);
	void AddImages(Image<aType>&, Image<aType>&, WriteableImage<aType>&);
	void InverseImage(Image<aType>&, WriteableImage<aType>&);
};

#include "Util.tpp"

#endif //__UTIL_H__
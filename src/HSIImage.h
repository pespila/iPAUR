#include "RGBImage.h"

#ifndef __HSIIMAGE_H__
#define __HSIIMAGE_H__

template<typename aType>
class HSIImage : public RGBImage<aType>
{
public:
	HSIImage() : RGBImage<aType>() {}
	~HSIImage();
};

#include "HSIImage.tpp"

#endif //__HSIIMAGE_H__
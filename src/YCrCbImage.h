#include "RGBImage.h"

#ifndef __YCrCbIMAGE_H__
#define __YCrCbIMAGE_H__

template<typename aType>
class YCrCbImage : public RGBImage<aType>
{
public:
	YCrCbImage() : RGBImage<aType>() {}
	~YCrCbImage();
};

#include "YCrCbImage.tpp"

#endif //__YCrCbIMAGE_H__
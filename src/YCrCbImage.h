#include "RGBImage.h"

#ifndef __YCrCbIMAGE_H__
#define __YCrCbIMAGE_H__

class YCrCbImage : public RGBImage
{
public:
	YCrCbImage() : RGBImage() {}
	~YCrCbImage();
};

#endif //__YCrCbIMAGE_H__
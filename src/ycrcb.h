#include "rgb.h"

#ifndef __YCrCb_H__
#define __YCrCb_H__

class YCrCbImage : public RGBImage
{
public:
	YCrCbImage() : RGBImage() {}
	~YCrCbImage();
};

#endif //__YCrCb_H__
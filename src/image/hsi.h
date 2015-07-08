#include "rgb.h"

#ifndef __HSI_H__
#define __HSI_H__

class HSIImage : public RGBImage
{
public:
	HSIImage() : RGBImage() {}
	~HSIImage();
};

#endif //__HSI_H__
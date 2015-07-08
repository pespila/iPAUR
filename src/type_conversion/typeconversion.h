#include "image.h"
#include "rgb.h"
#include "grayscale.h"
#include "hsi.h"
#include "ycrcb.h"

#ifndef __TYPECONVERSION_H__
#define __TYPECONVERSION_H__

class TypeConversion
{
private:
	int height;
	int width;
	int channel;
	char type;
public:
	TypeConversion():height(0), width(0), channel(0), type(0) {}
	TypeConversion(int, int, int, char);
	TypeConversion(Image&);
	~TypeConversion();

	void rgb2gray(RGBImage&, GrayscaleImage&);
	void gray2rgb(GrayscaleImage&, RGBImage&);
	void rgb2ycrcb(RGBImage&, YCrCbImage&);
	void rgb2hsi(RGBImage&, HSIImage&);
};

#endif //__TYPECONVERSION_H__
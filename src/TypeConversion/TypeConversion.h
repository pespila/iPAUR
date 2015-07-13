#include "../Image/Image.h"
#include "../Image/RGBImage.h"
#include "../Image/RGBAImage.h"
#include "../Image/GrayscaleImage.h"
#include "../Image/HSIImage.h"
#include "../Image/YCrCbImage.h"

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

	void RGB2Gray(RGBImage&, GrayscaleImage&);
	void Gray2RGB(GrayscaleImage&, RGBImage&);
	void RGB2YCrCb(RGBImage&, YCrCbImage&);
	void RGB2HSI(RGBImage&, HSIImage&);
};

#endif //__TYPECONVERSION_H__
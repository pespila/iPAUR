#include "../Image/Image.h"
#include "../Image/GrayscaleImage.h"
#include "../Image/RGBImage.h"
#include "../Image/RGBAImage.h"
#include "../Image/HSIImage.h"
#include "../Image/YCrCbImage.h"

#ifndef __UTIL_H__
#define __UTIL_H__

#define PI 3.14159265359

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

class Util
{
private:
	int height;
	int width;
	int channel;
	char type;

public:
	Util():height(0), width(0), channel(0), type(0) {}
	Util(int, int, int, char);
	Util(Image&);
	~Util();

	void MarkRed(RGBImage&, GrayscaleImage&, RGBImage&);
	void AddImages(Image&, Image&, WriteableImage&);
	void InverseImage(Image&, WriteableImage&);
};

#endif //__UTIL_H__
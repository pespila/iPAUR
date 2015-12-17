#include "Image.h"
#include "RGBImage.h"
#include "RGBAImage.h"
#include "GrayscaleImage.h"
#include "HSIImage.h"
#include "YCrCbImage.h"

#ifndef __TYPECONVERSION_H__
#define __TYPECONVERSION_H__

template<typename aType>
class TypeConversion
{
public:
	TypeConversion();
	~TypeConversion();

	void RGB2Gray(RGBImage<aType>&, GrayscaleImage<aType>&);
	void Gray2RGB(GrayscaleImage<aType>&, RGBImage<aType>&);
	void RGB2YCrCb(RGBImage<aType>&, YCrCbImage<aType>&);
	void RGB2HSI(RGBImage<aType>&, HSIImage<aType>&);
};

#include "TypeConversion.tpp"

#endif //__TYPECONVERSION_H__
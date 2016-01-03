#include "Image.h"

#ifndef __TYPECONVERSION_H__
#define __TYPECONVERSION_H__

template<typename aType>
class TypeConversion
{
public:
	TypeConversion() {}
	~TypeConversion() {}

	void RGB2Gray(Image<aType>&, Image<aType>&);
	void Gray2RGB(Image<aType>&, Image<aType>&);
	void RGB2YCrCb(Image<aType>&, Image<aType>&);
	void RGB2HSI(Image<aType>&, Image<aType>&);
};

#include "TypeConversion.tpp"

#endif //__TYPECONVERSION_H__
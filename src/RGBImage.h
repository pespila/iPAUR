#include "Image.h"

#ifndef __RGBIMAGE_H__
#define __RGBIMAGE_H__

template<typename aType>
class RGBImage : public WriteableImage<aType>
{
public:
	RGBImage():image(NULL), height(0), width(0), channels(0), type(0) {}
	~RGBImage();

	int GetChannels() {return this->channels;}
	int GetHeight() {return this->height;}
    int GetWidth() {return this->width;}
	int GetType() {return this->type;}
	aType Get(int i, int j, int k) {return this->image[j + i * this->width + k * this->height * this->width];}

	void Set(int i, int j, int k, aType value) {this->image[j + i * this->width + k * this->height * this->width] = value;}

	void Read(const string);
	void Write(const string);
	void Reset(int, int, char);

protected:
	aType* image;
	int height;
	int width;
	int channels;
	char type;
};

#include "RGBImage.tpp"

#endif //__RGBIMAGE_H__
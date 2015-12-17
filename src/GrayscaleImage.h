#include <string>
#include "Image.h"

#ifndef __GRAYSCALEIMAGE_H__
#define __GRAYSCALEIMAGE_H__

template<typename aType>
class GrayscaleImage : public WriteableImage<aType>
{
public:
	GrayscaleImage():image(NULL), height(0), width(0), channels(0), type(0) {}
	~GrayscaleImage();

    int GetChannels() {return this->channels;}
    int GetHeight() {return this->height;}
    int GetWidth() {return this->width;}
    int GetType() {return this->type;}
    aType Get(int i, int j, int k = 0) {return this->image[j + i * this->width + k * this->height * this->width];}

    void Set(int i, int j, int k, aType value) {this->image[j + i * this->width + k * this->height * this->width] = value;}

	void Read(const string);
	void Write(const string);
    void Reset(int, int, char);

private:
	aType* image;
	int height;
	int width;
	int channels;
	char type;
};

#include "GrayscaleImage.tpp"

#endif //__GRAYSCALEIMAGE_H__
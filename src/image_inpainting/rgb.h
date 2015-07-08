#include "image.h"

#ifndef __RGB_H__
#define __RGB_H__

class RGBImage : public WriteableImage
{
public:
	RGBImage():image(NULL), height(0), width(0), channels(0), type(0) {}
	~RGBImage();

	unsigned short GetChannels() {return this->channels;}
	unsigned short GetHeight() {return this->height;}
    unsigned short GetWidth() {return this->width;}
	unsigned short GetType() {return this->type;}
	unsigned char Get(int i, int j, int k) {return this->image[j + i * this->width + k * this->height * this->width];}

	void Set(int i, int j, int k, unsigned char value) {this->image[j + i * this->width + k * this->height * this->width] = value;}

	void Read(const string);
	void Write(const string);
	void Reset(unsigned short, unsigned short, char);

protected:
	unsigned char* image;
	unsigned short height;
	unsigned short width;
	int channels;
	char type;
};

#endif //__RGB_H__
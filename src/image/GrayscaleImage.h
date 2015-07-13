#include <string>
#include "Image.h"

#ifndef __GRAYSCALEIMAGE_H__
#define __GRAYSCALEIMAGE_H__

class GrayscaleImage : public WriteableImage
{
public:
	GrayscaleImage():image(NULL), height(0), width(0), channels(0), type(0) {}
	~GrayscaleImage();

    unsigned short GetChannels() {return this->channels;}
    unsigned short GetHeight() {return this->height;}
    unsigned short GetWidth() {return this->width;}
    unsigned short GetType() {return this->type;}
    unsigned char Get(int i, int j, int k = 0) {return this->image[j + i * this->width + k * this->height * this->width];}

    void Set(int i, int j, int k, unsigned char value) {this->image[j + i * this->width + k * this->height * this->width] = value;}

	void Read(const string);
	void Write(const string);
    void Reset(unsigned short, unsigned short, char);

private:
	unsigned char* image;
	unsigned short height;
	unsigned short width;
	int channels;
	char type;
};

#endif //__GRAYSCALEIMAGE_H__
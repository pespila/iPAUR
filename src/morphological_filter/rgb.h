#include "image.h"

#ifndef __RGB_H__
#define __RGB_H__

class RGBImage : public WriteableImage
{
public:
	RGBImage():image(NULL), height(0), width(0), channels(0), type(0) {}
	~RGBImage();

	unsigned short get_channels() {return this->channels;}
	unsigned short get_height() {return this->height;}
    unsigned short get_width() {return this->width;}
	unsigned short get_type() {return this->type;}
	unsigned char get_pixel(int i, int j, int k) {return this->image[j + i * this->width + k * this->height * this->width];}

	void set_pixel(int i, int j, int k, unsigned char value) {this->image[j + i * this->width + k * this->height * this->width] = value;}

	void read_image(const string);
	void write_image(const string);
	void reset_image(unsigned short, unsigned short, char);

protected:
	unsigned char* image;
	unsigned short height;
	unsigned short width;
	int channels;
	char type;
};

#endif //__RGB_H__
#include "image.h"

#ifndef __RGB_H__
#define __RGB_H__

class RGBImage : public WriteableImage
{
public:
	RGBImage():image(NULL), height(0), width(0), type(0) {}
	~RGBImage();


	unsigned short get_height() {return this->height;}
    unsigned short get_width() {return this->width;}
	unsigned short get_type() {return this->type;}
	unsigned char get_color_pixel_at_position(int i, int j, int k) {return this->image[j + i * this->width][k];}

	void set_color_pixel_at_position(int i, int j, int k, unsigned char value) {this->image[j + i * this->width][k] = value;}

	void read_image(const string);
	void write_image(const string);
	void reset_image(unsigned short, unsigned short, char);

protected:
	unsigned char** image;
	unsigned short height;
	unsigned short width;
	char type;
};

#endif //__RGB_H__
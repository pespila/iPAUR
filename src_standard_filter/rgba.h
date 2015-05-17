#include "image.h"
#include <vector>

#ifndef __RGBA_H__
#define __RGBA_H__

class RGBAImage : public WriteableImage
{
public:
	RGBAImage():image(NULL), height(0), width(0), type(0) {}
	~RGBAImage();


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

#endif //__RGBA_H__
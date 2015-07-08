#include <string>
#include "image.h"

#ifndef __GRAYSCALE_H__
#define __GRAYSCALE_H__

class GrayscaleImage : public WriteableImage
{
public:
	GrayscaleImage():image(NULL), height(0), width(0), channels(0), type(0) {}
	~GrayscaleImage();

    unsigned short get_channels() {return this->channels;}
    unsigned short get_height() {return this->height;}
    unsigned short get_width() {return this->width;}
    unsigned short get_type() {return this->type;}
    unsigned char get_pixel(int i, int j, int k = 0) {return this->image[j + i * this->width + k * this->height * this->width];}

    void set_pixel(int i, int j, int k, unsigned char value) {this->image[j + i * this->width + k * this->height * this->width] = value;}

	void read_image(const string);
	void write_image(const string);
    void reset_image(unsigned short, unsigned short, char);

private:
	unsigned char* image;
	unsigned short height;
	unsigned short width;
	int channels;
	char type;
};

#endif //__GRAYSCALE_H__
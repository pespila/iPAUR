#include <png.h>

#ifndef __IMAGE_H__
#define __IMAGE_H__

struct grayscaled_image
{
	float* approximation;
	int image_height;
	int image_width;
	png_byte bit_depth;
	int color_type;
};

typedef struct grayscaled_image gray_img;

gray_img *read_image_data(const char*);
gray_img *initalize_raw_image(int, int, char);
void write_image_data(gray_img*, const char*);
void destroy_image(gray_img*);

#endif //__IMAGE_H__
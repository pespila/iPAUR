#include "main.h"

#ifndef __IMAGE_H__
#define __IMAGE_H__

color_img *read_image_data(const char*);
color_img *initalize_raw_image(int, int, char);
void write_image_data(color_img*, const char*);
void destroy_image(color_img*);

#endif //__IMAGE_H__
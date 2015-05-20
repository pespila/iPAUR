#include "image.h"
#include "util.h"

#ifndef __INPAINTING_H__
#define __INPAINTING_H__

void laplace_operator(float*, float*, float*, int, int);
void identity(float*, float*, float*, int, int);
void proximation_g_inpainting(float*, float*, float*, float, float, int);
void image_inpainting(gray_img*, param*, const char*, int);
void image_inpainting_color(color_img*, param*, const char*, int);

#endif //__INPAINTING_H__
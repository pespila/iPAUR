#include "image.h"
#include "util.h"

#ifndef __INPAINTING_COLOR_H__
#define __INPAINTING_COLOR_H__

void laplace_operator_color(float**, float**, float**, int, int);
void identity_color(float**, float**, float**, int, int);
void proximation_g_inpainting_color(float**, float**, float**, float, float, int);
void image_inpainting_color(color_img*, param*, const char*, int);

#endif //__INPAINTING_COLOR_H__
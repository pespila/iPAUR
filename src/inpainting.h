#include "image.h"
#include "util.h"

#ifndef __INPAINTING_H__
#define __INPAINTING_H__

void proximation_g_inpainting(float*, float*, float*, float, float, int);
void image_inpainting(gray_img*, param*, const char*, int);

#endif //__INPAINTING_H__
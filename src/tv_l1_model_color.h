#include "image.h"
#include "util.h"

#ifndef __TV_L1_MODEL_COLOR_H__
#define __TV_L1_MODEL_COLOR_H__

void gradient_of_image_value_without_scaling_color(float**, float**, float**, int, int);
void divergence_of_dual_vector_without_scaling_color(float**, float**, float**, int, int);
void proximation_tv_l1_g_color(float**, float**, float**, float, float, int);
float computed_energy_of_tv_l1_functional_color(float**, float**, float**, float*, int);
void tv_l1_model_color(color_img*, param*, const char*, int);

#endif //__TV_L1_MODEL_COLOR_H__
#include "image.h"
#include "util.h"

#ifndef __HUBER_ROF_MODEL_COLOR_H__
#define __HUBER_ROF_MODEL_COLOR_H__

void gradient_of_image_value_color(float**, float**, float**, int, int);
void divergence_of_dual_vector_color(float**, float**, float**, int, int);
void proximation_f_star_huber_rof_color(float**, float**, float**, float**, float, float, int);
void proximation_g_color(float**, float**, float**, float, float, int);
float computed_energy_of_huber_rof_functional_color(float**, float**, float**, float**, int);
void huber_rof_model_color(color_img*, param*, const char*, int);

#endif //__HUBER_ROF_MODEL_COLOR_H__
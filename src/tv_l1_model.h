#include "image.h"
#include "util.h"

#ifndef __TV_L1_MODEL_H__
#define __TV_L1_MODEL_H__

void gradient_of_image_value_without_scaling(float*, float*, float*, int, int);
void divergence_of_dual_vector_without_scaling(float*, float*, float*, int, int);
void proximation_tv_l1_g(float*, float*, float*, float, float, int);
float computed_energy_of_tv_l1_functional(float*, float*, float*, float*, int);
void tv_l1_model(gray_img*, param*, const char*, int);
void tv_l1_model_color(color_img*, param*, const char*, int);

#endif //__TV_L1_MODEL_H__
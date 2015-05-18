#include "image.h"
#include "util.h"

#ifndef __HUBER_ROF_MODEL_H__
#define __HUBER_ROF_MODEL_H__

void gradient_of_image_value(float*, float*, float*, int, int);
void divergence_of_dual_vector(float*, float*, float*, int, int);
void proximation_f_star_huber_rof(float*, float*, float*, float*, float, float, int);
void proximation_g(float*, float*, float*, float, float, int);
float computed_energy_of_huber_rof_functional(float*, float*, float*, float*, int);
void huber_rof_model(gray_img*, param*, int);

#endif //__HUBER_ROF_MODEL_H__
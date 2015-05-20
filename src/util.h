#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#ifndef __UTIL_H__
#define __UTIL_H__

struct parameter
{
	float sigma;
	float tau;
	float lambda;
	float theta;
	float alpha;
	float L2;
};

typedef struct parameter param;

param* set_input_parameter(float, float, float, float, int);
float isotropic_total_variation_norm(float*, float*, int);
float isotropic_total_variation_norm_one_component(float*, float*, int);
float standard_squared_l2_norm(float*, float*, int);
void energy_to_file(float*, int, const char*);
cv::Mat convert_into_opencv_image(float*, int, int, int);

#endif //__UTIL_H__
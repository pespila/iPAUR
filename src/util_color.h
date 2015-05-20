#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#ifndef __UTIL_COLOR_H__
#define __UTIL_COLOR_H__

float isotropic_total_variation_norm_color(float**, float**, int);
float isotropic_total_variation_norm_one_component_color(float**, float**, int);
float standard_squared_l2_norm_color(float**, float**, int);
cv::Mat convert_into_opencv_color_image(float**, int, int, int);

#endif //__UTIL_COLOR_H__
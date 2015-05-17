#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "ms_minimizer.h"
#include "main.h"

using namespace cv;

#ifndef __MS_MINIMIZER_VIDEO_H__
#define __MS_MINIMIZER_VIDEO_H__

Mat write_image_data(color_img*);
void primal_dual_algorithm_video(color_img*, void (*prox_f)(struct dual_vector_2d, struct dual_vector_2d, struct parameter*, int, int), void (*prox_g)(double**, double**, color_img*, struct parameter*), struct parameter*, int, int, int);

#endif //__MS_MINIMIZER_VIDEO_H__
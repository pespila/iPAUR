#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "util_color.h"

float isotropic_total_variation_norm_color(float** y1, float** y2, int size) {
	float norm = 0.0;
	for (int i = 0; i < size; i++) {
		for (int k = 0; k < 3; k++) {
			norm += sqrt(pow(y1[i][k], 2) + pow(y2[i][k], 2));
		}
	}
	return norm;
}

float isotropic_total_variation_norm_one_component_color(float** u, float** image, int size) {
	float norm = 0.0;
	for (int i = 0; i < size; i++) {
		for (int k = 0; k < 3; k++) {
			norm += sqrt(pow(u[i][k] - image[i][k], 2));
		}
	}
	return norm;
}

float standard_squared_l2_norm_color(float** u, float** image, int size) {
	float norm = 0.0;
	for (int i = 0; i < size; i++) {
		for (int k = 0; k < 3; k++) {
			norm += (pow(u[i][k] - image[i][k], 2));
		}
	}
	return norm;
}

cv::Mat convert_into_opencv_color_image(float** image, int height, int width, int type) {
    cv::Mat img(height, width, type);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            img.at<cv::Vec3b>(i, j)[2] = image[j + i * width][2];
            img.at<cv::Vec3b>(i, j)[1] = image[j + i * width][1];
            img.at<cv::Vec3b>(i, j)[0] = image[j + i * width][0];
        }
    }
    return img;
}
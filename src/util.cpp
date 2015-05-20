#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "util.h"

param* set_input_parameter(float tau, float lambda, float theta, float alpha, int size) {
	param* output = (param*)malloc(sizeof(param));

	output->L2 = 8.0 * size;
	output->lambda = lambda;
	output->alpha = alpha;
	output->tau = tau;
	output->sigma = 1.0 / (tau * output->L2);
	output->theta = theta;
	return output;
}

float isotropic_total_variation_norm(float* y1, float* y2, int size) {
	float norm = 0.0;
	for (int i = 0; i < size; i++)
		norm += sqrt(pow(y1[i], 2) + pow(y2[i], 2));
	return norm;
}

float isotropic_total_variation_norm_one_component(float* u, float* image, int size) {
	float norm = 0.0;
	for (int i = 0; i < size; i++)
		norm += sqrt(pow(u[i] - image[i], 2));
	return norm;
}

float standard_squared_l2_norm(float* u, float* image, int size) {
	float norm = 0.0;
	for (int i = 0; i < size; i++)
		norm += (pow(u[i] - image[i], 2));
	return norm;
}

void energy_to_file(float* energy, int steps, const char* filename) {
    FILE *file;
    file = fopen(filename,"w");
    if(file == NULL) {
        printf("ERROR: File not found!");
    } else {
    	for (int i = 0; i < steps; i++) {
    		fprintf(file, "%g %g\n", float(i), energy[i]);
    	}
    }
    fclose (file);
}

cv::Mat convert_into_opencv_image(float* image, int height, int width, int type) {
    cv::Mat img(height, width, type);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            img.at<uchar>(i, j) = image[j + i * width];
        }
    }
    return img;
}
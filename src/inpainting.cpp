#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "huber_rof_model.h"
#include "inpainting.h"
#include "util.h"
#include "image.h"

void proximation_g_inpainting(float* x_out, float* x_in, float* image, unsigned char* hash_table, float tau, float lambda, int size) {
	float tau_mult_lambda = tau * lambda;
	for (int i = 0; i < size; i++)
		x_out[i] = !hash_table[i] ? x_in[i] : (x_in[i] + tau_mult_lambda * image[i]) / (1.0 + tau_mult_lambda);
}

void image_inpainting(gray_img* src, param* parameter, const char* filename, int steps) {
	const int M = src->image_height;
	const int N = src->image_width;
	const int size = M*N;
	double stopping_criterion = 1e-5;
	int i, k;

	unsigned char* hash_table = (unsigned char*)malloc(size*sizeof(unsigned char));
	float* image = (float*)malloc(size*sizeof(float));
	float* y1 = (float*)malloc(size*sizeof(float));
	float* y2 = (float*)malloc(size*sizeof(float));
	float* grad_u1 = (float*)malloc(size*sizeof(float));
	float* grad_u2 = (float*)malloc(size*sizeof(float));
	float* proximated1 = (float*)malloc(size*sizeof(float));
	float* proximated2 = (float*)malloc(size*sizeof(float));
	float* x = (float*)malloc(size*sizeof(float));
	float* x_bar = (float*)malloc(size*sizeof(float));
	float* x_current = (float*)malloc(size*sizeof(float));
	float* divergence = (float*)malloc(size*sizeof(float));
	float* energy = (float*)malloc((steps+1)*sizeof(float));

	for (i = 0; i < size; i++) {
		x[i] = src->approximation[i];
		x_bar[i] = src->approximation[i];
		image[i] = src->approximation[i];
		proximated1[i] = 0.0;
		proximated2[i] = 0.0;
		hash_table[i] = x[i] == 0 ? 0 : 1;
	}

	cv::Mat gray_image, writing_image;
	cv::VideoWriter output_cap;

	if (filename != NULL) {
		output_cap = cv::VideoWriter(filename, CV_FOURCC('m', 'p', '4', 'v'), 50, cv::Size(src->image_width, src->image_height), false);
		if (!output_cap.isOpened()) {
			printf("ERROR by opening!\n");
	   	}
	}

	gradient_of_image_value(grad_u1, grad_u2, x_bar, M, N);
	float normalization_value = computed_energy_of_huber_rof_functional(image, grad_u1, grad_u2, x_bar, size);
	energy[0] = 1.0;

	for (k = 1; k <= steps; k++) {
		for (i = 0; i < size; i++) {x_current[i] = x[i];}
		gradient_of_image_value(y1, y2, x_bar, M, N);
		for (i = 0; i < size; i++) {y1[i] = parameter->sigma * y1[i] + proximated1[i];}
		for (i = 0; i < size; i++) {y2[i] = parameter->sigma * y2[i] + proximated2[i];}
		proximation_f_star_huber_rof(proximated1, proximated2, y1, y2, parameter->sigma, parameter->alpha, size);
		divergence_of_dual_vector(divergence, proximated1, proximated2, M, N);
		for (i = 0; i < size; i++) {divergence[i] = parameter->tau * divergence[i] + x_current[i];}
		proximation_g_inpainting(x, divergence, image, hash_table, parameter->tau, parameter->lambda/2.0, size);
		for (i = 0; i < size; i++) {x_bar[i] = x[i] + parameter->theta * (x[i] - x_current[i]);}

		gradient_of_image_value(grad_u1, grad_u2, x_bar, M, N);
		energy[k] = computed_energy_of_huber_rof_functional(image, grad_u1, grad_u2, x_bar, size) / normalization_value;
		if (k > 200) {
			if (fabs(energy[k - 1] - energy[k]) < stopping_criterion) {
				for (i = 0; i < size; i++) {src->approximation[i] = (unsigned char)x_bar[i];}
				printf("Steps: %d\n", k);
				break;
			}
		}
		if (filename != NULL) {
			gray_image = convert_into_opencv_image(x_bar, src->image_height, src->image_width, src->image_type);
			cvtColor(gray_image, writing_image, CV_GRAY2BGR);
			output_cap.write(writing_image);
		}
		if (k == steps) {
			for (i = 0; i < size; i++) {src->approximation[i] = (unsigned char)x_bar[i];}
			printf("Steps: %d\n", k);
			break;
		}
	}
	energy_to_file(energy, steps+1, "./plot.dat");

	free(y1);
	free(y2);
	free(proximated1);
	free(proximated2);
	free(x);
	free(x_bar);
	free(x_current);
	free(divergence);
}
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "tv_l1_model_color.h"
#include "huber_rof_model_color.h"
#include "util_color.h"
#include "util.h"
#include "image.h"

void gradient_of_image_value_without_scaling_color(float** y1, float** y2, float** image, int M, int N) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < 3; k++) {
				y1[j + i * N][k] = i + 1 < M ? (image[j + (i+1) * N][k] - image[j + i * N][k]) : 0.0;
				y2[j + i * N][k] = j + 1 < N ? (image[j + 1 + i * N][k] - image[j + i * N][k]) : 0.0;
			}
		}
	}
}

void divergence_of_dual_vector_without_scaling_color(float** image, float** y1, float** y2, int M, int N) {
	float computed_argument_x1 = 0.0;
	float computed_argument_x2 = 0.0;
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < 3; k++) {
	            if (i == 0) computed_argument_x1 = y1[j + i * N][k];
	            else if (i == M-1) computed_argument_x1 = -y1[j + (i-1) * N][k];
	            else computed_argument_x1 = y1[j + i * N][k] - y1[j + (i-1) * N][k];

	            if (j == 0) computed_argument_x2 = y2[j + i * N][k];
	            else if (j == N-1) computed_argument_x2 = -y2[(j-1) + i * N][k];
	            else computed_argument_x2 = y2[j + i * N][k] - y2[(j-1) + i * N][k];

	            image[j + i * N][k] = computed_argument_x1 + computed_argument_x2;
	        }
		}
	}
}

void proximation_tv_l1_g_color(float** x_out, float** x_in, float** image, float tau, float lambda, int size) {
	float tau_mult_lambda = tau * lambda;
	float u_tilde_minus_original_image = 0.0;
	for (int i = 0; i < size; i++) {
		for (int k = 0; k < 3; k++) {
			u_tilde_minus_original_image = x_in[i][k] - image[i][k];
			if (u_tilde_minus_original_image > tau_mult_lambda) 			x_out[i][k] = x_in[i][k] - tau_mult_lambda;
			if (u_tilde_minus_original_image < -tau_mult_lambda) 			x_out[i][k] = x_in[i][k] + tau_mult_lambda;
			if (fabs(u_tilde_minus_original_image) <= tau_mult_lambda) 		x_out[i][k] = image[i][k];
		}
	}
}

float computed_energy_of_tv_l1_functional_color(float** image, float** y1, float** y2, float** x_bar, int size) {
	return (isotropic_total_variation_norm_color(y1, y2, size) + isotropic_total_variation_norm_one_component_color(x_bar, image, size)) / (size);
}

void tv_l1_model_color(color_img* src, param* parameter, const char* filename, int steps) {
	const int M = src->image_height;
	const int N = src->image_width;
	const int size = M*N;
	double stopping_criterion = 0.5e-2;
	int i, j, k;

	float** image = (float**)malloc(size*sizeof(float*));
	float** y1 = (float**)malloc(size*sizeof(float*));
	float** y2 = (float**)malloc(size*sizeof(float*));
	float** grad_u1 = (float**)malloc(size*sizeof(float*));
	float** grad_u2 = (float**)malloc(size*sizeof(float*));
	float** proximated1 = (float**)malloc(size*sizeof(float*));
	float** proximated2 = (float**)malloc(size*sizeof(float*));
	float** x = (float**)malloc(size*sizeof(float*));
	float** x_bar = (float**)malloc(size*sizeof(float*));
	float** x_current = (float**)malloc(size*sizeof(float*));
	float** divergence = (float**)malloc(size*sizeof(float*));
	float* energy = (float*)malloc((steps+1)*sizeof(float));

	for (i = 0; i < size; i++) {
		image[i] = (float*)malloc(3*sizeof(float));
		y1[i] = (float*)malloc(3*sizeof(float));
		y2[i] = (float*)malloc(3*sizeof(float));
		grad_u1[i] = (float*)malloc(3*sizeof(float));
		grad_u2[i] = (float*)malloc(3*sizeof(float));
		proximated1[i] = (float*)malloc(3*sizeof(float));
		proximated2[i] = (float*)malloc(3*sizeof(float));
		x[i] = (float*)malloc(3*sizeof(float));
		x_bar[i] = (float*)malloc(3*sizeof(float));
		x_current[i] = (float*)malloc(3*sizeof(float));
		divergence[i] = (float*)malloc(3*sizeof(float));
	}

	for (i = 0; i < size; i++) {
		x[i][0] = src->red[i];
		x[i][1] = src->green[i];
		x[i][2] = src->blue[i];
		x_bar[i][0] = src->red[i];
		x_bar[i][1] = src->green[i];
		x_bar[i][2] = src->blue[i];
		image[i][0] = src->red[i];
		image[i][1] = src->green[i];
		image[i][2] = src->blue[i];
		proximated1[i][0] = 0.0;
		proximated1[i][1] = 0.0;
		proximated1[i][2] = 0.0;
		proximated2[i][0] = 0.0;
		proximated2[i][1] = 0.0;
		proximated2[i][2] = 0.0;
	}

	cv::Mat writing_image;
	cv::VideoWriter output_cap;

	if (filename != NULL) {
		output_cap = cv::VideoWriter(filename, CV_FOURCC('m', 'p', '4', 'v'), 50, cv::Size(src->image_width, src->image_height), false);
		if (!output_cap.isOpened()) {
			printf("ERROR by opening!\n");
	   	}
	}

	gradient_of_image_value_without_scaling_color(grad_u1, grad_u2, x_bar, M, N);
	float normalization_value = computed_energy_of_huber_rof_functional_color(image, grad_u1, grad_u2, x_bar, size);
	energy[0] = 1.0;

	for (k = 1; k <= steps; k++) {
		for (i = 0; i < size; i++) {
			for (j = 0; j < 3; j++) {
				x_current[i][j] = x[i][j];
			}
		}
		gradient_of_image_value_without_scaling_color(y1, y2, x_bar, M, N);
		for (i = 0; i < size; i++) {
			for (j = 0; j < 3; j++) {
				y1[i][j] = parameter->sigma * y1[i][j] + proximated1[i][j];
			}
		}
		for (i = 0; i < size; i++) {
			for (j = 0; j < 3; j++) {
				y2[i][j] = parameter->sigma * y2[i][j] + proximated2[i][j];
			}
		}
		proximation_f_star_huber_rof_color(proximated1, proximated2, y1, y2, parameter->sigma, parameter->alpha, size);
		divergence_of_dual_vector_without_scaling_color(divergence, proximated1, proximated2, M, N);
		for (i = 0; i < size; i++) {
			for (j = 0; j < 3; j++) {
				divergence[i][j] = parameter->tau * divergence[i][j] + x_current[i][j];
			}
		}
		proximation_tv_l1_g_color(x, divergence, image, parameter->tau, parameter->lambda/2.0, size);
		for (i = 0; i < size; i++) {
			for (j = 0; j < 3; j++) {
				x_bar[i][j] = x[i][j] + parameter->theta * (x[i][j] - x_current[i][j]);
			}
		}

		gradient_of_image_value_without_scaling_color(grad_u1, grad_u2, x_bar, M, N);
		energy[k] = computed_energy_of_huber_rof_functional_color(image, grad_u1, grad_u2, x_bar, size) / normalization_value;
		if (k > 200) {
			if (fabs(energy[k - 1] - energy[k]) < stopping_criterion) {
				for (i = 0; i < size; i++) {
					src->red[i] = (unsigned char)x_bar[i][0];
					src->green[i] = (unsigned char)x_bar[i][1];
					src->blue[i] = (unsigned char)x_bar[i][2];
				}
				printf("Steps: %d\n", k);
				break;
			}
		}
		if (filename != NULL) {
			writing_image = convert_into_opencv_color_image(x_bar, src->image_height, src->image_width, src->image_type);
			output_cap.write(writing_image);
		}
		if (k == steps) {
			for (i = 0; i < size; i++) {
				src->red[i] = (unsigned char)x_bar[i][0];
				src->green[i] = (unsigned char)x_bar[i][1];
				src->blue[i] = (unsigned char)x_bar[i][2];
			}
			printf("Steps: %d\n", k);
			break;
		}
	}
	energy_to_file(energy, steps+1, "./plot.dat");

	for (int i = 0; i < size; i++)
	{
		free(image[i]);
		free(y1[i]);
		free(y2[i]);
		free(grad_u1[i]);
		free(grad_u2[i]);
		free(proximated1[i]);
		free(proximated2[i]);
		free(x[i]);
		free(x_bar[i]);
		free(x_current[i]);
		free(divergence[i]);
	}

	free(image);
	free(y1);
	free(y2);
	free(grad_u1);
	free(grad_u2);
	free(proximated1);
	free(proximated2);
	free(x);
	free(x_bar);
	free(x_current);
	free(divergence);
}
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "huber_rof_model_color.h"
#include "inpainting_color.h"
#include "util_color.h"
#include "util.h"
#include "image.h"

void proximation_g_inpainting_color(float** x_out, float** x_in, float** image, unsigned char* hash_table, float tau, float lambda, int size) {
	float tau_mult_lambda = tau * lambda;
	for (int i = 0; i < size; i++) {
		for (int k = 0; k < 3; k++) {
			x_out[i][k] = !hash_table[i] ? x_in[i][k] : (x_in[i][k] + tau_mult_lambda * image[i][k]) / (1.0 + tau_mult_lambda);
		}
	}
}

void laplace_operator_color(float** y1, float** y2, float** image, int M, int N) {
	float h_M = 1.0/M;
	float h_N = 1.0/N;
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < 3; k++) {
				y1[j + i * N][k] = i + 1 < M && i - 1 >= 0 ? (image[j + (i+1) * N][k] - 2.0 * image[j + i * N][k] + image[j + (i-1) * N][k]) / h_M : 0.0;
				y2[j + i * N][k] = j + 1 < N && j - 1 >= 0 ? (image[j + 1 + i * N][k] - 2.0 * image[j + i * N][k] - image[j - 1 + i * N][k]) / h_N : 0.0;
			}
		}
	}
}

void identity_color(float** y1, float** y2, float** image, int M, int N) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < 3; k++) {
				y1[j + i * N][k] = image[j + i * N][k];
				y2[j + i * N][k] = image[j + i * N][k];
			}
		}
	}
}

void image_inpainting_color(color_img* src, param* parameter, const char* filename, int steps) {
	const int M = src->image_height;
	const int N = src->image_width;
	const int size = M*N;
	double stopping_criterion = 1e-5;
	int i, j, k;

	unsigned char* hash_table = (unsigned char*)malloc(size*sizeof(unsigned char));
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
		hash_table[i] = x[i][0] == 0 && x[i][1] == 0 && x[i][2] == 0 ? 0 : 1;
	}

	cv::Mat writing_image;
	cv::VideoWriter output_cap;

	if (filename != NULL) {
		output_cap = cv::VideoWriter(filename, CV_FOURCC('m', 'p', '4', 'v'), 50, cv::Size(src->image_width, src->image_height), false);
		if (!output_cap.isOpened()) {
			printf("ERROR by opening!\n");
	   	}
	}

	gradient_of_image_value_color(grad_u1, grad_u2, x_bar, M, N);
	float normalization_value = computed_energy_of_huber_rof_functional_color(image, grad_u1, grad_u2, x_bar, size);
	energy[0] = 1.0;

	for (k = 1; k <= steps; k++) {
		for (i = 0; i < size; i++) {
			for (j = 0; j < 3; j++) {
				x_current[i][j] = x[i][j];
			}
		}
		gradient_of_image_value_color(y1, y2, x_bar, M, N);
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
		divergence_of_dual_vector_color(divergence, proximated1, proximated2, M, N);
		for (i = 0; i < size; i++) {
			for (j = 0; j < 3; j++) {
				divergence[i][j] = parameter->tau * divergence[i][j] + x_current[i][j];
			}
		}
		proximation_g_inpainting_color(x, divergence, image, hash_table, parameter->tau, parameter->lambda/2.0, size);
		for (i = 0; i < size; i++) {
			for (j = 0; j < 3; j++) {
				x_bar[i][j] = x[i][j] + parameter->theta * (x[i][j] - x_current[i][j]);
			}
		}

		gradient_of_image_value_color(grad_u1, grad_u2, x_bar, M, N);
		energy[k] = computed_energy_of_huber_rof_functional_color(image, grad_u1, grad_u2, x_bar, size) / normalization_value;
		if (k > 300) {
			if (energy[k - 1] - energy[k] < stopping_criterion) {
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

	free(hash_table);
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
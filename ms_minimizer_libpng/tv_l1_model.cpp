#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "tv_l1_model.h"
#include "huber_rof_model.h"
#include "util.h"
#include "image.h"

void gradient_of_image_value_without_scaling(float* y1, float* y2, float* image, int M, int N) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			y1[j + i * N] = i + 1 < M ? (image[j + (i+1) * N] - image[j + i * N]) : 0.0;
			y2[j + i * N] = j + 1 < N ? (image[j + 1 + i * N] - image[j + i * N]) : 0.0;
		}
	}
}

void divergence_of_dual_vector_without_scaling(float* image, float* y1, float* y2, int M, int N) {
	float computed_argument_x1 = 0.0;
	float computed_argument_x2 = 0.0;
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
            if (i == 0) computed_argument_x1 = y1[j + i * N];
            else if (i == M-1) computed_argument_x1 = -y1[j + (i-1) * N];
            else computed_argument_x1 = y1[j + i * N] - y1[j + (i-1) * N];

            if (j == 0) computed_argument_x2 = y2[j + i * N];
            else if (j == N-1) computed_argument_x2 = -y2[(j-1) + i * N];
            else computed_argument_x2 = y2[j + i * N] - y2[(j-1) + i * N];

            image[j + i * N] = computed_argument_x1 + computed_argument_x2;
		}
	}
}

void proximation_tv_l1_g(float* x_out, float* x_in, float* image, float tau, float lambda, int size) {
	float tau_mult_lambda = tau * lambda;
	float u_tilde_minus_original_image = 0.0;
	for (int i = 0; i < size; i++) {
		u_tilde_minus_original_image = x_in[i] - image[i];
		if (u_tilde_minus_original_image > tau_mult_lambda) 			x_out[i] = x_in[i] - tau_mult_lambda;
		if (u_tilde_minus_original_image < -tau_mult_lambda) 			x_out[i] = x_in[i] + tau_mult_lambda;
		if (fabs(u_tilde_minus_original_image) <= tau_mult_lambda) 		x_out[i] = image[i];
	}
}

float computed_energy_of_tv_l1_functional(float* image, float* y1, float* y2, float* x_bar, int size) {
	return (isotropic_total_variation_norm(y1, y2, size) + isotropic_total_variation_norm_one_component(x_bar, image, size)) / (size);
}

void tv_l1_model(gray_img* src, param* parameter, int steps) {
	const int M = src->image_height;
	const int N = src->image_width;
	const int size = M*N;
	float stopping_criterion = 1e-5;
	int i, k;

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
	}

	gradient_of_image_value_without_scaling(grad_u1, grad_u2, x_bar, M, N);
	float normalization_value = computed_energy_of_tv_l1_functional(x_bar, grad_u1, grad_u2, x_bar, size);
	energy[0] = 1.0;

	for (k = 1; k <= steps; k++) {
		for (i = 0; i < size; i++) {x_current[i] = x[i];}
		gradient_of_image_value_without_scaling(y1, y2, x_bar, M, N);
		for (i = 0; i < size; i++) {y1[i] = parameter->sigma * y1[i] + proximated1[i];}
		for (i = 0; i < size; i++) {y2[i] = parameter->sigma * y2[i] + proximated2[i];}
		proximation_f_star_huber_rof(proximated1, proximated2, y1, y2, parameter->sigma, parameter->alpha, size);
		divergence_of_dual_vector_without_scaling(divergence, proximated1, proximated2, M, N);
		for (i = 0; i < size; i++) {divergence[i] = parameter->tau * divergence[i] + x_current[i];}
		proximation_tv_l1_g(x, divergence, image, parameter->tau, parameter->lambda, size);
		for (i = 0; i < size; i++) {x_bar[i] = x[i] + parameter->theta * (x[i] - x_current[i]);}

		gradient_of_image_value_without_scaling(grad_u1, grad_u2, x_bar, M, N);
		energy[k] = computed_energy_of_tv_l1_functional(x_bar, grad_u1, grad_u2, x_bar, size) / normalization_value;
		if (k > 200) {
			if (energy[k - 1] - energy[k] < stopping_criterion) {
				for (i = 0; i < size; i++) {src->approximation[i] = (unsigned char)x_bar[i];}
				printf("Steps: %d\n", k);
				break;
			}
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
	free(image);
	free(x_bar);
	free(x_current);
	free(divergence);
}
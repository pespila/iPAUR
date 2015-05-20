#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "analytical_operators.h"
#include "ms_minimizer.h"
#include "main.h"
#include "image.h"

struct parameter* set_input_parameter(gray_img* src_image, double tau, double lambda, double theta, double alpha, double gamma, double delta, int spacing) {
	struct parameter* function_output = (struct parameter*)malloc(sizeof(struct parameter));

	function_output->delta = delta;
	function_output->gamma = gamma;
	function_output->L2 = spacing ? 8.0 * (src_image->image_height * src_image->image_width) : 8.0;
	function_output->lambda = lambda;
	function_output->alpha = alpha;
	function_output->mu = 2.0 * sqrt((gamma * delta) / function_output->L2);
	if (delta != 0) {
		function_output->tau = function_output->mu / (2.0 * gamma);
		function_output->sigma = function_output->mu / (2.0 * delta);
		function_output->theta = 1.0/(1.0 + function_output->mu);
	} else {
		function_output->tau = tau;
		function_output->sigma = 1.0/(tau * function_output->L2);
		function_output->theta = theta;
	}

	return function_output;
}

void update_input_parameters(struct parameter* input_parameter) {
	input_parameter->theta = 1.0/(sqrt(1.0 + 2.0 * input_parameter->gamma * input_parameter->tau));
	input_parameter->tau = input_parameter->theta * input_parameter->tau;
	input_parameter->sigma = input_parameter->sigma/input_parameter->theta;
}

void set_approximation(gray_img* src_image, double* x) {
	const int M = src_image->image_height;
	const int N = src_image->image_width;
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			src_image->approximation[j + i * N] = (unsigned char)x[j + i * N];
		}
	}
}

void init_vectors(double* x, double* x_bar, struct dual_vector_2d proximated, gray_img* src_image, int M, int N) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			x[j + i * N] = src_image->iterative_data[j + i * N];
			x_bar[j + i * N] = src_image->iterative_data[j + i * N];
			proximated.x1[j + i * N] = 0.0;
			proximated.x2[j + i * N] = 0.0;
		}
	}
}

void free_memory_of_vectors(double* x, double* x_bar, double* x_current, double* divergence, struct dual_vector_2d y, struct dual_vector_2d proximated) {
	free(x);
	free(x_bar);
	free(x_current);
	free(divergence);
	free(y.x1);
	free(y.x2);
	free(proximated.x1);
	free(proximated.x2);
}

void primal_dual_algorithm(gray_img* src, void (*prox_f)(struct dual_vector_2d, struct dual_vector_2d, struct parameter*, int, int), void (*prox_g)(double*, double*, gray_img*, struct parameter*), struct parameter* input_parameter, int spacing, int update, int steps) {
	const int M = src->image_height;
	const int N = src->image_width;

	struct dual_vector_2d y = {alloc_double_array(M, N), alloc_double_array(M, N)};
	struct dual_vector_2d proximated = {alloc_double_array(M, N), alloc_double_array(M, N)};
	double* x = alloc_double_array(M, N);
	double* x_bar = alloc_double_array(M, N);
	double* x_current = alloc_double_array(M, N);
	double* divergence = alloc_double_array(M, N);

	init_vectors(x, x_bar, proximated, src, M, N);

	for (int k = 1; k <= steps; k++) {
		add_vectors(x_current, x, x, 1.0, 0.0, M, N);
		gradient_of_image_value(y, input_parameter->sigma, x_bar, M, N, spacing);
		add_vectors(y.x1, proximated.x1, y.x1, 1.0, 1.0, M, N);
		add_vectors(y.x2, proximated.x2, y.x2, 1.0, 1.0, M, N);
		prox_f(proximated, y, input_parameter, M, N);
		divergence_of_dual_vector(divergence, input_parameter->tau, proximated, M, N, spacing);
		add_vectors(divergence, x_current, divergence, 1.0, 1.0, M, N);
		prox_g(x, divergence, src, input_parameter);
		add_vectors(x_bar, x_current, x, -input_parameter->theta, (1 + input_parameter->theta), M, N);
		if (update) {
			update_input_parameters(input_parameter);
		}
		if (k == steps) {
			set_approximation(src, x_bar);
		}
	}

	free_memory_of_vectors(x, x_bar, x_current, divergence, y, proximated);
}
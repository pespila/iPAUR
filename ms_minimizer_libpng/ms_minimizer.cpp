#include <math.h>
#include <omp.h>
#include "ms_minimizer.h"
#include "image.h"

void add_vectors(float* x_out, float* x_in1, float* x_in2, float parameter_x_in1, float parameter_x_in2, int M, int N) {
	float calculation = 0.0;
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			calculation = parameter_x_in1 * x_in1[j + i * N] + parameter_x_in2 * x_in2[j + i * N];
			x_out[j + i * N] = calculation;
		}
	}
}

// void gradient_of_image_value(struct dual_vector_2d* y, float gradient_scalar, float* x, int M, int N, int spacing) {
// 	float h_M = spacing ? 1.0/M : 1.0;
// 	float h_N = spacing ? 1.0/N : 1.0;
// 	for (int i = 0; i < M; i++) {
// 		for (int j = 0; j < N; j++) {
// 			y->x1[j + i * N] = i + 1 < M ? gradient_scalar * (x[j + (i+1) * N] - x[j + i * N]) / h_M : 0.0;
// 			y->x2[j + i * N] = j + 1 < N ? gradient_scalar * (x[j + 1 + i * N] - x[j + i * N]) / h_N : 0.0;
// 		}
// 	}
// }

void gradient_of_image_value(struct dual_vector_2d* y, float gradient_scalar, float* x, int M, int N, int spacing) {
	float h_M = spacing ? 1.0/M : 1.0;
	float h_N = spacing ? 1.0/N : 1.0;
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			y->x1[j + i * N] = i + 1 < M ? gradient_scalar * (x[j + (i+1) * N] - x[j + i * N]) / h_M : 0.0;
			y->x2[j + i * N] = j + 1 < N ? gradient_scalar * (x[j + 1 + i * N] - x[j + i * N]) / h_N : 0.0;
		}
	}
}

void divergence_of_dual_vector(float* x, float divergence_scalar, struct dual_vector_2d* y, int M, int N, int spacing) {
	float computed_argument_x1 = 0.0;
	float computed_argument_x2 = 0.0;
	float h_M = spacing ? 1.0/M : 1.0;
	float h_N = spacing ? 1.0/N : 1.0;
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
            if (i == 0) computed_argument_x1 = y->x1[j + i * N];
            else if (i == M-1) computed_argument_x1 = -y->x1[j + (i-1) * N];
            else computed_argument_x1 = y->x1[j + i * N] - y->x1[j + (i-1) * N];

            if (j == 0) computed_argument_x2 = y->x2[j + i * N];
            else if (j == N-1) computed_argument_x2 = -y->x2[(j-1) + i * N];
            else computed_argument_x2 = y->x2[j + i * N] - y->x2[(j-1) + i * N];

            x[j + i * N] = divergence_scalar * (computed_argument_x1 / h_M + computed_argument_x2 / h_N);
		}
	}
}

void huber_rof_proximation_f_star(struct dual_vector_2d* proximated, struct dual_vector_2d* y, struct parameter* input_parameter, int M, int N) {
	float vector_norm = 0.0;
	float sigma_mult_alpha = input_parameter->sigma * input_parameter->alpha;
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			vector_norm = sqrt(y->x1[j + i * N] * y->x1[j + i * N] + y->x2[j + i * N] * y->x2[j + i * N]);
			proximated->x1[j + i * N] = (y->x1[j + i * N]/(1.0 + sigma_mult_alpha))/fmax(1.0, vector_norm/(1.0 + sigma_mult_alpha));
			proximated->x2[j + i * N] = (y->x2[j + i * N]/(1.0 + sigma_mult_alpha))/fmax(1.0, vector_norm/(1.0 + sigma_mult_alpha));
		}
	}
}

void proximation_g(float* x_out, float* x_in, gray_img* src_image, struct parameter* input_parameter) {
	const int M = src_image->image_height;
	const int N = src_image->image_width;
	float tau_mult_lambda = input_parameter->tau * input_parameter->lambda;
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			x_out[j + i * N]  = (x_in[j + i * N] + tau_mult_lambda * src_image->iterative_data[j + i * N])/(1.0 + tau_mult_lambda);
		}
	}
}

void proximation_tv_l1_g(float* x_out, float* x_in, gray_img* src_image, struct parameter* input_parameter) {
	const int M = src_image->image_height;
	const int N = src_image->image_width;
	float tau_mult_lambda = input_parameter->tau * input_parameter->lambda;
	float u_tilde_minus_original_image = 0.0;
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			u_tilde_minus_original_image = x_in[j + i * N] - src_image->iterative_data[j + i * N];
			if (u_tilde_minus_original_image > tau_mult_lambda) 			x_out[j + i * N] = x_in[j + i * N] - tau_mult_lambda;
			if (u_tilde_minus_original_image < -tau_mult_lambda) 			x_out[j + i * N] = x_in[j + i * N] + tau_mult_lambda;
			if (fabs(u_tilde_minus_original_image) <= tau_mult_lambda) 	x_out[j + i * N] = src_image->iterative_data[j + i * N];
		}
	}
}

struct parameter* set_input_parameter(gray_img* src_image, float tau, float lambda, float theta, float alpha, float gamma, float delta, int spacing) {
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

void set_approximation(gray_img* src_image, float* x) {
	const int M = src_image->image_height;
	const int N = src_image->image_width;
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			src_image->approximation[j + i * N] = (unsigned char)x[j + i * N];
		}
	}
}

void init_vectors(float* x, float* x_bar, struct dual_vector_2d* proximated, gray_img* src_image, int M, int N) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			x[j + i * N] = src_image->iterative_data[j + i * N];
			x_bar[j + i * N] = src_image->iterative_data[j + i * N];
			proximated->x1[j + i * N] = 0.0;
			proximated->x2[j + i * N] = 0.0;
		}
	}
}

void free_memory_of_vectors(float* x, float* x_bar, float* x_current, float* divergence, struct dual_vector_2d* y, struct dual_vector_2d* proximated) {
	free(x);
	free(x_bar);
	free(x_current);
	free(divergence);
	free(y->x1);
	free(y->x2);
	free(proximated->x1);
	free(proximated->x2);
}

void primal_dual_algorithm(gray_img* src, void (*prox_f)(struct dual_vector_2d*, struct dual_vector_2d*, struct parameter*, int, int), void (*prox_g)(float*, float*, gray_img*, struct parameter*), struct parameter* input_parameter, int spacing, int update, int steps) {
	const int M = src->image_height;
	const int N = src->image_width;
	int i, k;

	// struct dual_vector_2d* y = {(float*)malloc(M*N*sizeof(float)), (float*)malloc(M*N*sizeof(float))};
	// struct dual_vector_2d* proximated = {(float*)malloc(M*N*sizeof(float)), (float*)malloc(M*N*sizeof(float))};
	struct dual_vector_2d* y;
	y->x1 = (float*)malloc(M*N*sizeof(float));
	printf("here..\n");
	y->x2 = (float*)malloc(M*N*sizeof(float));
	struct dual_vector_2d* proximated;
	proximated->x1 = (float*)malloc(M*N*sizeof(float));
	proximated->x2 = (float*)malloc(M*N*sizeof(float));

	float* x = (float*)malloc(M*N*sizeof(float));
	float* x_bar = (float*)malloc(M*N*sizeof(float));
	float* x_current = (float*)malloc(M*N*sizeof(float));
	float* divergence = (float*)malloc(M*N*sizeof(float));

	for (i = 0; i < M*N; i++) {
		x[i] = src->iterative_data[i];
		x_bar[i] = src->iterative_data[i];
		proximated->x1[i] = 0.0;
		proximated->x2[i] = 0.0;
	}
	// init_vectors(x, x_bar, proximated, src, M, N);

	for (k = 1; k <= steps; k++) {

		for (i = 0; i < M*N; i++) {x_current[i] = x[i];}
		// add_vectors(x_current, x, x, 1.0, 0.0, M, N);

		gradient_of_image_value(y, input_parameter->sigma, x_bar, M, N, spacing);
		
		for (i = 0; i < M*N; i++) {y->x1[i] += proximated->x1[i];}
		for (i = 0; i < M*N; i++) {y->x2[i] += proximated->x2[i];}
		// add_vectors(y->x1, proximated->x1, y->x1, 1.0, 1.0, M, N);
		// add_vectors(y->x2, proximated->x2, y->x2, 1.0, 1.0, M, N);

		prox_f(proximated, y, input_parameter, M, N);
		divergence_of_dual_vector(divergence, input_parameter->tau, proximated, M, N, spacing);
		
		for (i = 0; i < M*N; i++) {divergence[i] += x_current[i];}
		// add_vectors(divergence, x_current, divergence, 1.0, 1.0, M, N);

		prox_g(x, divergence, src, input_parameter);

		add_vectors(x_bar, x_current, x, -input_parameter->theta, (1 + input_parameter->theta), M, N);
		// if (update) {
		// 	update_input_parameters(input_parameter);
		// }
		if (k == steps) {
			for (i = 0; i < M*N; i++) {src->approximation[i] = (unsigned char)x[i];}
			// set_approximation(src, x_bar);
		}
	}

	free(x);
	free(x_bar);
	free(x_current);
	free(divergence);
	free(y->x1);
	free(y->x2);
	free(proximated->x1);
	free(proximated->x2);

	// free_memory_of_vectors(x, x_bar, x_current, divergence, y, proximated);
}
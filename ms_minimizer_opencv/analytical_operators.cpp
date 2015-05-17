#include <math.h>
// #include "ms_minimizer.h"
#include "analytical_operators.h"

void add_vectors(double* x_out, double* x_in1, double* x_in2, double parameter_x_in1, double parameter_x_in2, int M, int N) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			x_out[j + i * N] = parameter_x_in1 * x_in1[j + i * N] + parameter_x_in2 * x_in2[j + i * N];
		}
	}
}

void gradient_of_image_value(struct dual_vector_2d y, double gradient_scalar, double* x, int M, int N, int spacing) {
	double h_M = spacing ? 1.0/M : 1.0;
	double h_N = spacing ? 1.0/N : 1.0;
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			y.x1[j + i * N] = i + 1 < M ? gradient_scalar * (x[j + (i+1) * N] - x[j + i * N]) / h_M : 0.0;
			y.x2[j + i * N] = j + 1 < N ? gradient_scalar * (x[j + 1 + i * N] - x[j + i * N]) / h_N : 0.0;
		}
	}
}

void divergence_of_dual_vector(double* x, double divergence_scalar, struct dual_vector_2d y, int M, int N, int spacing) {
	double computed_argument_x1 = 0.0;
	double computed_argument_x2 = 0.0;
	double h_M = spacing ? 1.0/M : 1.0;
	double h_N = spacing ? 1.0/N : 1.0;
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
            if (i == 0) computed_argument_x1 = y.x1[j + i * N];
            else if (i == M-1) computed_argument_x1 = -y.x1[j + (i-1) * N];
            else computed_argument_x1 = y.x1[j + i * N] - y.x1[j + (i-1) * N];

            if (j == 0) computed_argument_x2 = y.x2[j + i * N];
            else if (j == N-1) computed_argument_x2 = -y.x2[(j-1) + i * N];
            else computed_argument_x2 = y.x2[j + i * N] - y.x2[(j-1) + i * N];

            x[j + i * N] = divergence_scalar * (computed_argument_x1 / h_M + computed_argument_x2 / h_N);
		}
	}
}

void huber_rof_proximation_f_star(struct dual_vector_2d proximated, struct dual_vector_2d y, struct parameter* input_parameter, int M, int N) {
	double vector_norm = 0.0;
	double sigma_mult_alpha = input_parameter->sigma * input_parameter->alpha;
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			vector_norm = sqrt(y.x1[j + i * N] * y.x1[j + i * N] + y.x2[j + i * N] * y.x2[j + i * N]);
			proximated.x1[j + i * N] = (y.x1[j + i * N]/(1.0 + sigma_mult_alpha))/fmax(1.0, vector_norm/(1.0 + sigma_mult_alpha));
			proximated.x2[j + i * N] = (y.x2[j + i * N]/(1.0 + sigma_mult_alpha))/fmax(1.0, vector_norm/(1.0 + sigma_mult_alpha));
		}
	}
}

void proximation_fast_ms_minimizer_f_star(struct dual_vector_2d proximated, struct dual_vector_2d y, struct parameter* input_parameter, int M, int N) {
	double vector_norm = 0.0;
	double comparison_criterion = sqrt(input_parameter->lambda/input_parameter->alpha * input_parameter->sigma * (input_parameter->sigma + 2.0 * input_parameter->alpha));
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			vector_norm = sqrt(y.x1[j + i * N] * y.x1[j + i * N] + y.x2[j + i * N] * y.x2[j + i * N]);
			proximated.x1[j + i * N] = vector_norm <= comparison_criterion ? (2.0*input_parameter->alpha)/(input_parameter->sigma+2.0*input_parameter->alpha) * y.x1[j + i * N] : 0.0;
			proximated.x2[j + i * N] = vector_norm <= comparison_criterion ? (2.0*input_parameter->alpha)/(input_parameter->sigma+2.0*input_parameter->alpha) * y.x2[j + i * N] : 0.0;
		}
	}
}

void proximation_fast_ms_minimizer_g(double* x_out, double* x_in, gray_img* src_image, struct parameter* input_parameter) {
	const int M = src_image->image_height;
	const int N = src_image->image_width;
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			x_out[j + i * N] = (x_in[j + i * N] + 2.0 * input_parameter->tau * src_image->iterative_data[j + i * N])/(1.0 + 2.0 * input_parameter->tau);
		}
	}
}

void proximation_g(double* x_out, double* x_in, gray_img* src_image, struct parameter* input_parameter) {
	const int M = src_image->image_height;
	const int N = src_image->image_width;
	double tau_mult_lambda = input_parameter->tau * input_parameter->lambda;
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			x_out[j + i * N]  = (x_in[j + i * N] + tau_mult_lambda * src_image->iterative_data[j + i * N])/(1.0 + tau_mult_lambda);
		}
	}
}

void proximation_tv_l1_g(double* x_out, double* x_in, gray_img* src_image, struct parameter* input_parameter) {
	const int M = src_image->image_height;
	const int N = src_image->image_width;
	double tau_mult_lambda = input_parameter->tau * input_parameter->lambda;
	double u_tilde_minus_original_image = 0.0;
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			u_tilde_minus_original_image = x_in[j + i * N] - src_image->iterative_data[j + i * N];
			if (u_tilde_minus_original_image > tau_mult_lambda) 			x_out[j + i * N] = x_in[j + i * N] - tau_mult_lambda;
			if (u_tilde_minus_original_image < -tau_mult_lambda) 			x_out[j + i * N] = x_in[j + i * N] + tau_mult_lambda;
			if (fabs(u_tilde_minus_original_image) <= tau_mult_lambda) 	x_out[j + i * N] = src_image->iterative_data[j + i * N];
		}
	}
}
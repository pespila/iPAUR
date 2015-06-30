#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "image.h"
#include "primal_dual.h"

void gradient_operator(vec* y, float* x, int M, int N, int K) {
	for (int k = 0; k < K; k++) {
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				y[j + i * N + k * N * M].x1 = i + 1 < M ? (x[j + (i+1) * N + k * N * M] - x[j + i * N + k * N * M]) * M : 0.0;
				y[j + i * N + k * N * M].x2 = j + 1 < N ? (x[j + 1 + i * N + k * N * M] - x[j + i * N + k * N * M]) * N : 0.0;
				y[j + i * N + k * N * M].x3 = k + 1 < K ? (x[j + i * N + (k+1) * N * M] - x[j + i * N + k * N * M]) * K : 0.0;
			}
		}
	}
}

void divergence_operator(float* x, vec* y, int M, int N, int K) {
	float x1 = 0.0, x2 = 0.0, x3 = 0.0;
	for (int k = 0; k < K; k++) {
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				if (i == 0) x1 = y[j + i * N + k * N * M].x1;
	            else if (i == M-1) x1 = -y[j + (i-1) * N + k * N * M].x1;
	            else x1 = y[j + i * N + k * N * M].x1 - y[j + (i-1) * N + k * N * M].x1;

	            if (j == 0) x2 = y[j + i * N + k * N * M].x2;
	            else if (j == N-1) x2 = -y[j - 1 + i * N + k * N * M].x2;
	            else x2 = y[j + i * N + k * N * M].x2 - y[j - 1 + i * N + k * N * M].x2;

	            if (k == 0) x3 = y[j + i * N + k * N * M].x3;
	            else if (k == K-1) x3 = -y[j + i * N + (k-1) * N * M].x3;
	            else x3 = y[j + i * N + k * N * M].x3 - y[j + i * N + (k-1) * N * M].x3;

	            x[j + i * N + k * N * M] = x1 * M + x2 * N + x3 * K;
			}
		}
	}
}

double f(double x1, double x2, double lambda, double L, double image, int k) {
	return 0.25 * (x1*x1 + x2*x2) - lambda * ((double)k / L - image);
}

const inline double f_prime(double x) {
	return 0.5 * x;
}

const inline double f_prime_prime() {
	return 0.5;
}

const inline double schwarz() {
	return 0.0;
}

void soft_shrinkage_operator(vec* out, vec* in, float nu, int size) {
	for (int i = 0; i < size; i++) {
		// if (i == 0) printf("%f\n", fmax(abs(in[i].x1 - nu), 0.0) * sgn(in[i].x1));
		out[i].x1 = fmax(abs(in[i].x1 - nu), 0.0) * sgn(in[i].x1);
		out[i].x2 = fmax(abs(in[i].x2 - nu), 0.0) * sgn(in[i].x2);
		out[i].x3 = fmax(abs(in[i].x3 - nu), 0.0) * sgn(in[i].x3);
	}
}

void truncation_operator(float* out, float* in, int size) {
	for (int i = 0; i < size; i++)
		out[i] = fmin(1.0, fmax(0.0, in[i]));
}

void scale_vector(vec* out, float scale1, float scale2, vec* in, int size) {
	for (int i = 0; i < size; i++) {
		out[i].x1 = (scale1 + scale2) * in[i].x1;
		out[i].x2 = (scale1 + scale2) * in[i].x2;
		out[i].x3 = (scale1 + scale2) * in[i].x3;
	}
}

void add_vector(vec* out, float factor1, float factor2, vec* in1, vec* in2, int size) {
	for (int i = 0; i < size; i++) {
		out[i].x1 = factor1 * in1[i].x1 + factor2 * in2[i].x1;
		out[i].x2 = factor1 * in1[i].x2 + factor2 * in2[i].x2;
		out[i].x3 = factor1 * in1[i].x3 + factor2 * in2[i].x3;
	}
}

void add_array(float* out, float factor1, float factor2, float* in1, float* in2, int size) {
	for (int i = 0; i < size; i++)
		out[i] = factor1 * in1[i] + factor2 * in2[i];
}

void copy_array(float* out, float* in, int size) {
	for (int i = 0; i < size; i++)
		out[i] = in[i];
}

void extrapolation(float* out, float theta, float* in1, float* in2, int size) {
	for (int i = 0; i < size; i++)
		out[i] = in1[i] + theta * (in1[i] - in2[i]);
}

void newton_parabola(vec* xk, vec* point, double lambda, double L, float* image, int M, int N, int K) {
	M = 1;
	N = 1;
	K = 1;
	int size = K*N*M;
	vec* g_x = (vec*)malloc(size*sizeof(vec));
	vec* delta_x = (vec*)malloc(size*sizeof(vec));
	Matrix jacobi = {0.0, 0.0, 0.0, 0.0};
	double x1 = 0.0;
	double x2 = 0.0;
	double img = 0.0;
	// double norm = 1.0;
	int l = 0;
	int steps = 0;

// norm > 1E-6 && 
	for (int k = 0; k < K; k++) {
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				l = j + i * N + k * N * M;
				// if (i == 0 && j == 0 && k == 0) printf("l: %d: (%f, %f, %f)\n", l, point[l].x1, point[l].x2, point[l].x3);
				// printf("l: %d: (%f, %f, %f)\n", l, point[l].x1, point[l].x2, point[l].x3);
				// helper variables
				x1 = xk[l].x1;
				x2 = xk[l].x2;

				img = image[j + i * N];
				steps = 0;
				while (steps < 10) {
					
					// generate g(x)
					g_x[l].x1 = x1 - point[l].x1 + f_prime(x1) * (f(x1, x2, lambda, L, img, k) - point[l].x3);
					g_x[l].x2 = x2 - point[l].x2 + f_prime(x2) * (f(x1, x2, lambda, L, img, k) - point[l].x3);

					//generate J(f)
					jacobi.a11 = 1.0 + f_prime_prime() * (f(x1, x2, lambda, L, img, k) - point[l].x3) + (f_prime(x1) * f_prime(x1));
					jacobi.a22 = 1.0 + f_prime_prime() * (f(x1, x2, lambda, L, img, k) - point[l].x3) + (f_prime(x2) * f_prime(x2));
					// jacobi.a12 = schwarz() * (f(x1, x2, lambda, L, img, k) - point[l].x3) + (f_prime(x1) * f_prime(x2));
					jacobi.a12 = f_prime(x1) * f_prime(x2); // schwarz() returns 0.0!!!
					jacobi.a21 = jacobi.a12;
					// if (i == 0 && j == 0 && k == 0) printf("%f %f\n%f %f\n", jacobi.a11, jacobi.a12, jacobi.a21, jacobi.a22);

					// make upper triangluar matrix via gauss
					jacobi.a11 *= jacobi.a21; jacobi.a12 *= jacobi.a21; g_x[l].x1 *= jacobi.a21;
					jacobi.a21 *= jacobi.a11; jacobi.a22 *= jacobi.a11; g_x[l].x2 *= jacobi.a11;
					jacobi.a21 -= jacobi.a11;
					jacobi.a22 -= jacobi.a12;

					// set delta
					delta_x[l].x2 = jacobi.a22 != 0 ? -g_x[l].x2 / jacobi.a22 : 0.0;
					delta_x[l].x1 = jacobi.a11 != 0 ? (-g_x[l].x1 - jacobi.a12 * delta_x[l].x2) / jacobi.a11 : 0.0;

					// update x^k
					xk[l].x1 += delta_x[l].x1;
					xk[l].x2 += delta_x[l].x2;
					xk[l].x3 = f(xk[l].x1, xk[l].x2, lambda, L, img, k);
					steps++;
					// if (steps == 4 && i == 0 && j == 0 && k == 0) printf("img: %f: (%f, %f, %f)\n", img, xk[l].x1, xk[l].x2, xk[l].x3);
				}
			}
		}
		// norm = sqrtf(pow(xk[l].x1 - x1, 2) + pow(xk[l].x2 - x2, 2));
	}

	free(g_x);
	free(delta_x);
}

inline double vector_norm(vec* x, int size) {
	double norm = 0.0;
	for (int i = 0; i < size; i++)
		norm += sqrt(x[i].x1*x[i].x1 + x[i].x2*x[i].x2 + x[i].x3*x[i].x3);

	return norm;
}

void dykstra_projection(vec* y, vec* x, float* image, param* parameter, int max_iter, int M, int N, int K) {
	int size = M*N*K;
	vec* u = (vec*)malloc(size*sizeof(vec));
	for (int i = 0; i < size; i++) {u[i].x1 = 1.0; u[i].x2 = 1.0; u[i].x3 = 1.0;}
	vec* u_tmp = (vec*)malloc(size*sizeof(vec));
	vec* p = (vec*)malloc(size*sizeof(vec));
	vec* q = (vec*)malloc(size*sizeof(vec));
	vec* current = (vec*)malloc(size*sizeof(vec));
	double norm = 0.0;
	for (int i = 0; i < max_iter; i++) {
		scale_vector(current, 1.0, 0.0, x, size);
		add_vector(u_tmp, 1.0, 1.0, x, p, size);
		newton_parabola(u, u_tmp, parameter->lambda, parameter->L, image, M, N, K);
		// if (i == 0) printf("%f\n", u[0].x1);
		add_vector(p, 1.0, 1.0, x, p, size);
		add_vector(p, 1.0, -1.0, p, u, size);
		add_vector(u_tmp, 1.0, 1.0, u, q, size);
		soft_shrinkage_operator(y, u_tmp, parameter->nu, size);
		add_vector(q, 1.0, 1.0, u, q, size);
		add_vector(q, 1.0, -1.0, q, y, size);
		add_vector(current, 1.0, -1.0, current, y, size);
		norm = vector_norm(current, size);
		// printf("%f\n", norm);
		if (norm < 1E-3) break;
	}
}

inline void init_data_structures(float* image, float* x, float* x_bar, float* divergence, vec* y, vec* y_old, int M, int N, int K, gray_img* src) {
	for (int i = 0; i < M*N; i++) {
		image[i] = src->approximation[i] / 255.0;
		for (int k = 0; k < K; k++) {
			x[i + M*N*k] = src->approximation[i] / 255.0 * (double)(K-k) / (double)K;
			x_bar[i + M*N*k] = src->approximation[i] / 255.0 * (double)(K-k) / (double)K;
			divergence[i + M*N*k] = 0.0;
			y[i + M*N*k].x1 = 0.0;
			y[i + M*N*k].x2 = 0.0;
			y[i + M*N*k].x3 = 0.0;
			y_old[i + M*N*k].x1 = 0.0;
			y_old[i + M*N*k].x2 = 0.0;
			y_old[i + M*N*k].x3 = 0.0;
		}
	}
}

inline void free_data_structures(float* image, float* x, float* x_bar, float* x_current, float* divergence, vec* y, vec* y_old) {
	free(image);
	free(x);
	free(x_bar);
	free(x_current);
	free(divergence);
	free(y);
	free(y_old);
}

void primal_dual(gray_img* src, param* parameter, int levels, int steps) {
	const int M = src->image_height;
	const int N = src->image_width;
	const int K = levels;
	const int size = M*N*K;
	int i, k;

	float* image = (float*)malloc(M*N*sizeof(float));
	float* x = (float*)malloc(size*sizeof(float));
	float* x_bar = (float*)malloc(size*sizeof(float));
	float* x_current = (float*)malloc(size*sizeof(float));
	float* divergence = (float*)malloc(size*sizeof(float));
	vec* y = (vec*)malloc(size*sizeof(vec));
	vec* y_old = (vec*)malloc(size*sizeof(vec));

	init_data_structures(image, x, x_bar, divergence, y, y_old, M, N, K, src);

	for (k = 1; k <= steps; k++) {
		copy_array(x_current, x, size);
		gradient_operator(y_old, x_bar, M, N, K);
		scale_vector(y_old, parameter->sigma, 1.0, y_old, size);
		dykstra_projection(y, y_old, image, parameter, 50, M, N, K);
		// if (k == 1) printf("%f\n", y[0].x1);
		divergence_operator(divergence, y, M, N, K);
		add_array(divergence, parameter->tau, 1.0, divergence, x_current, size);
		truncation_operator(x, divergence, size);
		extrapolation(x_bar, parameter->theta, x, x_current, size);
	}
	// for (i = 0; i < M*N; i++) {src->approximation[i] = (unsigned char)x_bar[i];}
	for (i = 0; i < M*N; i++) {src->approximation[i] = (unsigned char)(x_bar[i] * 255.0);}

	free_data_structures(image, x, x_bar, x_current, divergence, y, y_old);
}
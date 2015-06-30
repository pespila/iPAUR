#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include "image.h"
#include "fast_minimizer.h"

void nabla(double* gradient_x, double* gradient_y, double* u_bar, int height, int width, int channel) {
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				gradient_x[j + i * width + k * height * width] = i + 1 < height ? (u_bar[j + (i+1) * width + k * height * width] - u_bar[j + i * width + k * height * width]) : 0.0;
				gradient_y[j + i * width + k * height * width] = j + 1 < width ? (u_bar[j + 1 + i * width + k * height * width] - u_bar[j + i * width + k * height * width]) : 0.0;
			}
		}
	}
}

void vector_of_inner_product(double* vector_norm_squared, double* p_tilde_x, double* p_tilde_y, int height, int width, int channel) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < channel; k++) {
				vector_norm_squared[j + i * width] += pow(p_tilde_x[j + i * width + k * height * width], 2) + pow(p_tilde_y[j + i * width + k * height * width], 2);
			}
		}
	}
}

void prox_r_star(double* p_x, double* p_y, double* p_tilde_x, double* p_tilde_y, double alpha, double lambda, double sigma, int height, int width, int channel, int cartoon) {
	double* vector_norm_squared = (double*)malloc(height*width*sizeof(double));
	double factor = cartoon ? 1.0 : (2.0 * alpha) / (sigma + 2.0 * alpha);
	double bound = cartoon ? 2.0 * lambda * sigma : (lambda / alpha) * sigma * (sigma + 2.0 * alpha);
	vector_of_inner_product(vector_norm_squared, p_tilde_x, p_tilde_y, height, width, channel);
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				p_x[j + i * width + k * height * width] = vector_norm_squared[j + i * width] * factor <= bound ? factor * p_tilde_x[j + i * width + k * height * width] : 0.0;
				p_y[j + i * width + k * height * width] = vector_norm_squared[j + i * width] * factor <= bound ? factor * p_tilde_y[j + i * width + k * height * width] : 0.0;
			}
		}
	}
}

void nabla_transpose(double* gradient_transpose, double* p_x, double* p_y, int height, int width, int channel) {
	double x = 0.0;
	double x_minus_one = 0.0;
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				x = i + 1 < height ? p_x[j + i * width + k * height * width] : 0.0;
				x_minus_one = i > 0 ? p_x[j + (i-1) * width + k * height * width] : 0.0;
				gradient_transpose[j + i * width + k * height * width] = x_minus_one - x;
				x = j + 1 < width ? p_y[j + i * width + k * height * width] : 0.0;
				x_minus_one = j > 0 ? p_y[j - 1 + i * width + k * height * width] : 0.0;
				gradient_transpose[j + i * width + k * height * width] += (x_minus_one - x);
			}
		}
	}
}

void prox_d(double* u, double* u_tilde, double* f, double tau, int size) {
	for (int i = 0; i < size; i++)
		u[i] = f[i] + (u_tilde[i] - f[i]) / (1.0 + 2.0 * tau);
		// u[i] = (u_tilde[i] + 2.0 * tau * f[i]) / (1.0 + 2.0 * tau);
}

void fast_minimizer(Image& src, WriteableImage& dst, int steps, int cartoon) {
	const int channel = src.get_channels();
	const int height = src.get_height();
	const int width = src.get_width();
	const int size = height * width * channel;
	int i;

	double alpha = 20.0;
	double lambda = 0.1;
	double tau = 1.0 / 4.0;
	double sigma = 1.0 / 2.0;
	double theta = 1.0;

	dst.reset_image(height, width, src.get_type());

	double* f = (double*)malloc(size*sizeof(double));
	double* u = (double*)malloc(size*sizeof(double));
	double* u_n = (double*)malloc(size*sizeof(double));
	double* u_bar = (double*)malloc(size*sizeof(double));
	double* gradient_x = (double*)malloc(size*sizeof(double));
	double* gradient_y = (double*)malloc(size*sizeof(double));
	double* gradient_transpose = (double*)malloc(size*sizeof(double));
	double* p_x = (double*)malloc(size*sizeof(double));
	double* p_y = (double*)malloc(size*sizeof(double));

	for (int k = 0; k < channel; k++)
	{
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				f[j + i * width + k * height * width] = (double)src.get_pixel(i, j, k) / 255.0;
				u[j + i * width + k * height * width] = (double)src.get_pixel(i, j, k) / 255.0;
				u_bar[j + i * width + k * height * width] = (double)src.get_pixel(i, j, k) / 255.0;
				p_x[j + i * width + k * height * width] = 0.0;
				p_y[j + i * width + k * height * width] = 0.0;
			}
		}
	}


	for (int k = 0; k < steps; k++)
	{
		for (i = 0; i < size; i++) {u_n[i] = u[i];}
		nabla(gradient_x, gradient_y, u_bar, height, width, channel);
		for (i = 0; i < size; i++) {gradient_x[i] = sigma * gradient_x[i] + p_x[i];}
		for (i = 0; i < size; i++) {gradient_y[i] = sigma * gradient_y[i] + p_y[i];}
		prox_r_star(p_x, p_y, gradient_x, gradient_y, alpha, lambda, sigma, height, width, channel, cartoon);
		nabla_transpose(gradient_transpose, p_x, p_y, height, width, channel);
		for (i = 0; i < size; i++) {gradient_transpose[i] = u_n[i] - tau * gradient_transpose[i];}
		prox_d(u, gradient_transpose, f, tau, size);
		theta = 1.0 / sqrt(1.0 + 4.0 * tau);
		tau *= theta;
		sigma /= theta;
		for (i = 0; i < size; i++) {u_bar[i] = u[i] + theta * (u[i] - u_n[i]);}
	}

	for (int k = 0; k < channel; k++)
	{
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				dst.set_pixel(i, j, k, (unsigned char)abs(u_bar[j + i * width + k * height * width] * 255.0));
			}
		}
	}

	free(f);
	free(u);
	free(u_n);
	free(u_bar);
	free(gradient_x);
	free(gradient_y);
	free(gradient_transpose);
	free(p_x);
	free(p_y);
}
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include "image.h"
#include "parameter.h"
#include "tvl1_model.h"

TVL1_Minimizer::TVL1_Minimizer(Image& src, int steps) {
	this->steps = steps;
	this->channel = src.get_channels();
	this->height = src.get_height();
	this->width = src.get_width();
	this->size = height * width * channel;
	this->f = (float*)malloc(size*sizeof(float));
	this->u = (float*)malloc(size*sizeof(float));
	this->u_n = (float*)malloc(size*sizeof(float));
	this->u_bar = (float*)malloc(size*sizeof(float));
	this->gradient_x = (float*)malloc(size*sizeof(float));
	this->gradient_y = (float*)malloc(size*sizeof(float));
	this->gradient_transpose = (float*)malloc(size*sizeof(float));
	this->p_x = (float*)malloc(size*sizeof(float));
	this->p_y = (float*)malloc(size*sizeof(float));
}

TVL1_Minimizer::~TVL1_Minimizer() {
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

void TVL1_Minimizer::initialize(Image& src) {
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				f[j + i * width + k * height * width] = (float)src.get_pixel(i, j, k);
				u[j + i * width + k * height * width] = (float)src.get_pixel(i, j, k);
				u_bar[j + i * width + k * height * width] = (float)src.get_pixel(i, j, k);
				p_x[j + i * width + k * height * width] = 0.0;
				p_y[j + i * width + k * height * width] = 0.0;
			}
		}
	}
}

void TVL1_Minimizer::set_solution(WriteableImage& dst) {
	for (int k = 0; k < channel; k++)
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				dst.set_pixel(i, j, k, (unsigned char)u_bar[j + i * width + k * height * width]);
}

void TVL1_Minimizer::nabla(float* gradient_x, float* gradient_y, float* u_bar) {
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				gradient_x[j + i * width + k * height * width] = i + 1 < height ? (u_bar[j + (i+1) * width + k * height * width] - u_bar[j + i * width + k * height * width]) : 0.0;
				gradient_y[j + i * width + k * height * width] = j + 1 < width ? (u_bar[j + 1 + i * width + k * height * width] - u_bar[j + i * width + k * height * width]) : 0.0;
			}
		}
	}
}

void TVL1_Minimizer::prox_r_star(float* p_x, float* p_y, float* p_tilde_x, float* p_tilde_y, float alpha, float sigma) {
	float vector_norm = 0.0;
	float factor = 1.0 + sigma * alpha;
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				vector_norm = sqrt(pow(p_tilde_x[j + i * width + k * height * width], 2) + pow(p_tilde_y[j + i * width + k * height * width], 2));
				p_x[j + i * width + k * height * width] = (p_tilde_x[j + i * width + k * height * width] / factor) / fmax(1.0, vector_norm / factor);
				p_y[j + i * width + k * height * width] = (p_tilde_y[j + i * width + k * height * width] / factor) / fmax(1.0, vector_norm / factor);
			}
		}
	}
}

void TVL1_Minimizer::nabla_transpose(float* gradient_transpose, float* p_x, float* p_y) {
	float x = 0.0;
	float x_minus_one = 0.0;
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

void TVL1_Minimizer::prox_d(float* u, float* u_tilde, float* f, float tau, float lambda) {
	float tau_mult_lambda = tau * lambda;
	float u_tilde_minus_original_image = 0.0;
	for (int i = 0; i < size; i++) {
		u_tilde_minus_original_image = u_tilde[i] - f[i];
		if (u_tilde_minus_original_image > tau_mult_lambda) 			u[i] = u_tilde[i] - tau_mult_lambda;
		if (u_tilde_minus_original_image < -tau_mult_lambda) 			u[i] = u_tilde[i] + tau_mult_lambda;
		if (fabs(u_tilde_minus_original_image) <= tau_mult_lambda) 		u[i] = f[i];
	}
}

void TVL1_Minimizer::tvl1_model(Image& src, WriteableImage& dst, Parameter& par) {
	int i;
	dst.reset_image(height, width, src.get_type());
	initialize(src);
	for (int k = 0; k < steps; k++)
	{
		for (i = 0; i < size; i++) {u_n[i] = u[i];}
		nabla(gradient_x, gradient_y, u_bar);
		for (i = 0; i < size; i++) {gradient_x[i] = par.sigma * gradient_x[i] + p_x[i];}
		for (i = 0; i < size; i++) {gradient_y[i] = par.sigma * gradient_y[i] + p_y[i];}
		prox_r_star(p_x, p_y, gradient_x, gradient_y, par.alpha, par.sigma);
		nabla_transpose(gradient_transpose, p_x, p_y);
		for (i = 0; i < size; i++) {gradient_transpose[i] = u_n[i] - par.tau * gradient_transpose[i];}
		prox_d(u, gradient_transpose, f, par.tau, par.lambda);
		for (i = 0; i < size; i++) {u_bar[i] = u[i] + par.theta * (u[i] - u_n[i]);}
	}
	set_solution(dst);
}
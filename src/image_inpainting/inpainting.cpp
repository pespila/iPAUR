#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include "image.h"
#include "parameter.h"
#include "inpainting.h"

Image_Inpainting::Image_Inpainting(Image& src, int steps) {
	this->steps = steps;
	this->channel = src.GetChannels();
	this->height = src.GetHeight();
	this->width = src.GetWidth();
	this->size = height * width * channel;
	this->hash_table = (unsigned char*)malloc(this->height*this->width*sizeof(unsigned char));
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

Image_Inpainting::~Image_Inpainting() {
	free(hash_table);
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

void Image_Inpainting::initialize(Image& src) {
	int small_size = height*width;
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				f[j + i * width + k * height * width] = (float)src.Get(i, j, k);
				u[j + i * width + k * height * width] = (float)src.Get(i, j, k);
				u_bar[j + i * width + k * height * width] = (float)src.Get(i, j, k);
				p_x[j + i * width + k * height * width] = 0.0;
				p_y[j + i * width + k * height * width] = 0.0;
			}
		}
	}

	for (int i = 0; i < small_size; i++)
		hash_table[i] = f[i + 0 * small_size] == 0 && f[i + 1 * small_size] == 0 && f[i + 2 * small_size] == 0 ? 0 : 1;
}

void Image_Inpainting::set_solution(WriteableImage& dst) {
	for (int k = 0; k < channel; k++)
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				dst.Set(i, j, k, (unsigned char)u_bar[j + i * width + k * height * width]);
}

void Image_Inpainting::nabla(float* gradient_x, float* gradient_y, float* u_bar) {
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				gradient_x[j + i * width + k * height * width] = i + 1 < height ? (u_bar[j + (i+1) * width + k * height * width] - u_bar[j + i * width + k * height * width]) * height : 0.0;
				gradient_y[j + i * width + k * height * width] = j + 1 < width ? (u_bar[j + 1 + i * width + k * height * width] - u_bar[j + i * width + k * height * width]) * width : 0.0;
			}
		}
	}
}

void Image_Inpainting::prox_r_star(float* p_x, float* p_y, float* p_tilde_x, float* p_tilde_y, float alpha, float sigma) {
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

void Image_Inpainting::nabla_transpose(float* gradient_transpose, float* p_x, float* p_y) {
	float x = 0.0;
	float x_minus_one = 0.0;
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				x = i + 1 < height ? p_x[j + i * width + k * height * width] : 0.0;
				x_minus_one = i > 0 ? p_x[j + (i-1) * width + k * height * width] : 0.0;
				gradient_transpose[j + i * width + k * height * width] = (x_minus_one - x) * height;
				x = j + 1 < width ? p_y[j + i * width + k * height * width] : 0.0;
				x_minus_one = j > 0 ? p_y[j - 1 + i * width + k * height * width] : 0.0;
				gradient_transpose[j + i * width + k * height * width] += ((x_minus_one - x) * width);
			}
		}
	}
}

void Image_Inpainting::prox_d(float* u, float* u_tilde, float* f, unsigned char* hash_table, float tau, float lambda) {
	int small_size = height*width;
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < small_size; i++) {
			u[i + k * small_size] = !hash_table[i] ? u_tilde[i + k * small_size] : (u_tilde[i + k * small_size] + tau * lambda * f[i + k * small_size]) / (1.0 + tau * lambda);
		}
	}
}

void Image_Inpainting::inpainting(Image& src, WriteableImage& dst, Parameter& par) {
	int i;
	dst.Reset(height, width, src.GetType());
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
		prox_d(u, gradient_transpose, f, hash_table, par.tau, par.lambda);
		for (i = 0; i < size; i++) {u_bar[i] = u[i] + par.theta * (u[i] - u_n[i]);}
	}
	set_solution(dst);
}
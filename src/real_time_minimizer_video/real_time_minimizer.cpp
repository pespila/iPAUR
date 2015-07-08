#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include "image.h"
#include "parameter.h"
#include "real_time_minimizer.h"

using namespace cv;

MS_Minimizer::MS_Minimizer(Image& src, int steps) {
	this->steps = steps;
	this->channel = src.GetChannels();
	this->height = src.GetHeight();
	this->width = src.GetWidth();
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

MS_Minimizer::~MS_Minimizer() {
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

void MS_Minimizer::initialize(Image& src) {
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				f[j + i * width + k * height * width] = (float)src.Get(i, j, k) / 255.0;
				u[j + i * width + k * height * width] = (float)src.Get(i, j, k) / 255.0;
				u_bar[j + i * width + k * height * width] = (float)src.Get(i, j, k) / 255.0;
				p_x[j + i * width + k * height * width] = 0.0;
				p_y[j + i * width + k * height * width] = 0.0;
			}
		}
	}
}

void MS_Minimizer::set_solution(WriteableImage& dst) {
	for (int k = 0; k < channel; k++)
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				dst.Set(i, j, k, (unsigned char)abs(u_bar[j + i * width + k * height * width] * 255.0));
}

void MS_Minimizer::nabla(float* gradient_x, float* gradient_y, float* u_bar) {
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				gradient_x[j + i * width + k * height * width] = i + 1 < height ? (u_bar[j + (i+1) * width + k * height * width] - u_bar[j + i * width + k * height * width]) : 0.0;
				gradient_y[j + i * width + k * height * width] = j + 1 < width ? (u_bar[j + 1 + i * width + k * height * width] - u_bar[j + i * width + k * height * width]) : 0.0;
			}
		}
	}
}

void MS_Minimizer::vector_of_inner_product(float* vector_norm_squared, float* p_tilde_x, float* p_tilde_y) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < channel; k++) {
				vector_norm_squared[j + i * width] += pow(p_tilde_x[j + i * width + k * height * width], 2) + pow(p_tilde_y[j + i * width + k * height * width], 2);
			}
		}
	}
}

void MS_Minimizer::prox_r_star(float* p_x, float* p_y, float* p_tilde_x, float* p_tilde_y, float alpha, float lambda, float sigma, int cartoon) {
	float* vector_norm_squared = (float*)malloc(height*width*sizeof(float));
	float factor = cartoon ? 1.0 : (2.0 * alpha) / (sigma + 2.0 * alpha);
	float bound = cartoon ? 2.0 * lambda * sigma : (lambda / alpha) * sigma * (sigma + 2.0 * alpha);
	vector_of_inner_product(vector_norm_squared, p_tilde_x, p_tilde_y);
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				p_x[j + i * width + k * height * width] = vector_norm_squared[j + i * width] * factor <= bound ? factor * p_tilde_x[j + i * width + k * height * width] : 0.0;
				p_y[j + i * width + k * height * width] = vector_norm_squared[j + i * width] * factor <= bound ? factor * p_tilde_y[j + i * width + k * height * width] : 0.0;
			}
		}
	}
}

void MS_Minimizer::nabla_transpose(float* gradient_transpose, float* p_x, float* p_y) {
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

void MS_Minimizer::prox_d(float* u, float* u_tilde, float* f, float tau) {
	for (int i = 0; i < size; i++)
		u[i] = (u_tilde[i] + 2.0 * tau * f[i]) / (1.0 + 2.0 * tau);
}

void MS_Minimizer::real_time_minimizer(Image& src, WriteableImage& dst, Parameter& par) {
	int i;
	dst.Reset(height, width, src.GetType());
	initialize(src);
	for (int k = 0; k < steps; k++)
	{
		for (i = 0; i < size; i++) {u_n[i] = u[i];}
		nabla(gradient_x, gradient_y, u_bar);
		for (i = 0; i < size; i++) {gradient_x[i] = par.sigma * gradient_x[i] + p_x[i];}
		for (i = 0; i < size; i++) {gradient_y[i] = par.sigma * gradient_y[i] + p_y[i];}
		prox_r_star(p_x, p_y, gradient_x, gradient_y, par.alpha, par.lambda, par.sigma, par.cartoon);
		nabla_transpose(gradient_transpose, p_x, p_y);
		for (i = 0; i < size; i++) {gradient_transpose[i] = u_n[i] - par.tau * gradient_transpose[i];}
		prox_d(u, gradient_transpose, f, par.tau);
		par.theta = 1.0 / sqrt(1.0 + 4.0 * par.tau);
		par.tau *= par.theta;
		par.sigma /= par.theta;
		for (i = 0; i < size; i++) {u_bar[i] = u[i] + par.theta * (u[i] - u_n[i]);}
	}
	set_solution(dst);
}

void MS_Minimizer::video(WriteableImage& src, WriteableImage& dst, Parameter& par) {
	VideoCapture cap("./test.mp4");
	Mat edges;
	namedWindow("frame",1);
	while(1) {
		Mat frame;
		cap >> frame;
		for (int k = 0; k < channel; k++)
		{
			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					src.Set(i, j, k, frame.at<Vec3b>(i, j)[k]);
				}
			}
		}
		real_time_minimizer(src, dst, par);
		for (int k = 0; k < channel; k++)
		{
			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					frame.at<Vec3b>(i, j)[k] = dst.Get(i, j, k);
				}
			}
		}
		// cvtColor(frame, edges, COLOR_BGR2GRAY);
        // GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
        // Canny(edges, edges, 0, 30, 3);
        imshow("frame", frame);
        if(waitKey(30) >= 0) break;
	}
}
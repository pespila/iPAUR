#include "TVL1Model.h"

TVL1Model::TVL1Model(Image& src, int steps) {
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

TVL1Model::~TVL1Model() {
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

void TVL1Model::Initialize(Image& src) {
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
}

void TVL1Model::SetSolution(WriteableImage& dst) {
	for (int k = 0; k < channel; k++)
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				dst.Set(i, j, k, (unsigned char)u_bar[j + i * width + k * height * width]);
}

void TVL1Model::Nabla(float* gradient_x, float* gradient_y, float* u_bar) {
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				gradient_x[j + i * width + k * height * width] = i + 1 < height ? (u_bar[j + (i+1) * width + k * height * width] - u_bar[j + i * width + k * height * width]) : 0.0;
				gradient_y[j + i * width + k * height * width] = j + 1 < width ? (u_bar[j + 1 + i * width + k * height * width] - u_bar[j + i * width + k * height * width]) : 0.0;
			}
		}
	}
}

void TVL1Model::ProxRstar(float* p_x, float* p_y, float* p_tilde_x, float* p_tilde_y, float alpha, float sigma) {
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

void TVL1Model::NablaTranspose(float* gradient_transpose, float* p_x, float* p_y) {
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

void TVL1Model::ProxD(float* u, float* u_tilde, float* f, float tau, float lambda) {
	float tau_mult_lambda = tau * lambda;
	float u_tilde_minus_original_image = 0.0;
	for (int i = 0; i < size; i++) {
		u_tilde_minus_original_image = u_tilde[i] - f[i];
		if (u_tilde_minus_original_image > tau_mult_lambda) 			u[i] = u_tilde[i] - tau_mult_lambda;
		if (u_tilde_minus_original_image < -tau_mult_lambda) 			u[i] = u_tilde[i] + tau_mult_lambda;
		if (fabs(u_tilde_minus_original_image) <= tau_mult_lambda) 		u[i] = f[i];
	}
}

void TVL1Model::TVL1(Image& src, WriteableImage& dst, Parameter& par) {
	int i;
	dst.Reset(height, width, src.GetType());
	Initialize(src);
	for (int k = 0; k < steps; k++)
	{
		for (i = 0; i < size; i++) {u_n[i] = u[i];}
		Nabla(gradient_x, gradient_y, u_bar);
		for (i = 0; i < size; i++) {gradient_x[i] = par.sigma * gradient_x[i] + p_x[i];}
		for (i = 0; i < size; i++) {gradient_y[i] = par.sigma * gradient_y[i] + p_y[i];}
		ProxRstar(p_x, p_y, gradient_x, gradient_y, par.alpha, par.sigma);
		NablaTranspose(gradient_transpose, p_x, p_y);
		for (i = 0; i < size; i++) {gradient_transpose[i] = u_n[i] - par.tau * gradient_transpose[i];}
		ProxD(u, gradient_transpose, f, par.tau, par.lambda);
		for (i = 0; i < size; i++) {u_bar[i] = u[i] + par.theta * (u[i] - u_n[i]);}
	}
	SetSolution(dst);
}
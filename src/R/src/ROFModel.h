#include <math.h>
#include <Rcpp.h>

using namespace Rcpp;
using namespace std;

#ifndef __ROFMODEL_H__
#define __ROFMODEL_H__

template<typename aType>
class ROFModel
{
private:
	int steps;
	int height;
	int width;
	int channel;
	int size;
	aType* f;
	aType* u;
	aType* u_n;
	aType* u_bar;
	aType* gradient_x;
	aType* gradient_y;
	aType* gradient_transpose;
	aType* p_x;
	aType* p_y;
	
	void Initialize(const NumericMatrix&);
  	void SetSolution(NumericMatrix&);
	void Nabla(aType*, aType*, aType*, aType*, aType*, aType);
	void ProxRstar(aType*, aType*, aType*, aType*, aType);
	void NablaTranspose(aType*, aType*, aType*, aType*, aType);
	void ProxD(aType*, aType*, aType*, aType, aType);
	void Extrapolation(aType*, aType*, aType*, aType);
	
public:
	ROFModel():steps(0), height(0), width(0), channel(0), size(0), f(NULL), u(NULL), u_n(NULL), u_bar(NULL), gradient_x(NULL), gradient_y(NULL), gradient_transpose(NULL), p_x(NULL), p_y(NULL) {}
	ROFModel(const NumericMatrix&, int, int, int, int);
	~ROFModel();

	void ROF(const NumericMatrix&, NumericMatrix&, aType, aType);
};

template<typename aType>
ROFModel<aType>::ROFModel(const NumericMatrix& src, int steps, int height, int width, int channel) {
  	this->steps = steps;
	this->channel = channel;
	this->height = height;
	this->width = width;
	this->size = height * width * channel;
	this->f = (aType*)malloc(size*sizeof(aType));
	this->u = (aType*)malloc(size*sizeof(aType));
	this->u_n = (aType*)malloc(size*sizeof(aType));
	this->u_bar = (aType*)malloc(size*sizeof(aType));
	this->gradient_x = (aType*)malloc(size*sizeof(aType));
	this->gradient_y = (aType*)malloc(size*sizeof(aType));
	this->gradient_transpose = (aType*)malloc(size*sizeof(aType));
	this->p_x = (aType*)malloc(size*sizeof(aType));
	this->p_y = (aType*)malloc(size*sizeof(aType));
}

template<typename aType>
ROFModel<aType>::~ROFModel() {
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

template<typename aType>
void ROFModel<aType>::Initialize(const NumericMatrix& src) {
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				f[j + i * width + k * height * width] = (aType)src(i, j);
				u[j + i * width + k * height * width] = (aType)src(i, j);
				u_n[j + i * width + k * height * width] = (aType)src(i, j);
				u_bar[j + i * width + k * height * width] = (aType)src(i, j);
				p_x[j + i * width + k * height * width] = 0.0;
				p_y[j + i * width + k * height * width] = 0.0;
			}
		}
	}
}

template<typename aType>
void ROFModel<aType>::SetSolution(NumericMatrix& dst) {
	for (int k = 0; k < channel; k++)
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
        		dst(i, j) = (aType)u[j + i * width + k * height * width];
}

template<typename aType>
void ROFModel<aType>::Nabla(aType* gradient_x, aType* gradient_y, aType* u_bar, aType* p_x, aType* p_y, aType sigma) {
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				gradient_x[j + i * width + k * height * width] = i + 1 < height ? u_bar[j + (i+1) * width + k * height * width] - u_bar[j + i * width + k * height * width] : 0.0;
				gradient_y[j + i * width + k * height * width] = j + 1 < width ? u_bar[j + 1 + i * width + k * height * width] - u_bar[j + i * width + k * height * width] : 0.0;
				gradient_x[j + i * width + k * height * width] = sigma * gradient_x[j + i * width + k * height * width] + p_x[j + i * width + k * height * width];
				gradient_y[j + i * width + k * height * width] = sigma * gradient_y[j + i * width + k * height * width] + p_y[j + i * width + k * height * width];
			}
		}
	}
}

template<typename aType>
void ROFModel<aType>::ProxRstar(aType* p_x, aType* p_y, aType* p_tilde_x, aType* p_tilde_y, aType sigma) {
	aType vector_norm = 0.0;
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				vector_norm = sqrt(pow(p_tilde_x[j + i * width + k * height * width], 2) + pow(p_tilde_y[j + i * width + k * height * width], 2));
				p_x[j + i * width + k * height * width] = p_tilde_x[j + i * width + k * height * width] / fmax(1.0, vector_norm);
				p_y[j + i * width + k * height * width] = p_tilde_y[j + i * width + k * height * width] / fmax(1.0, vector_norm);
			}
		}
	}
}

template<typename aType>
void ROFModel<aType>::NablaTranspose(aType* gradient_transpose, aType* p_x, aType* p_y, aType* u_n, aType tau) {
	aType x = 0.0;
	aType x_minus_one = 0.0;
	for (int k = 0; k < channel; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				x = i + 1 < height ? p_x[j + i * width + k * height * width] : 0.0;
				x_minus_one = i > 0 ? p_x[j + (i-1) * width + k * height * width] : 0.0;
				gradient_transpose[j + i * width + k * height * width] = x_minus_one - x;
				x = j + 1 < width ? p_y[j + i * width + k * height * width] : 0.0;
				x_minus_one = j > 0 ? p_y[j - 1 + i * width + k * height * width] : 0.0;
				gradient_transpose[j + i * width + k * height * width] += x_minus_one - x;
				gradient_transpose[j + i * width + k * height * width] = u_n[j + i * width + k * height * width] - tau * gradient_transpose[j + i * width + k * height * width];
			}
		}
	}
}

template<typename aType>
void ROFModel<aType>::ProxD(aType* u, aType* u_tilde, aType* f, aType tau, aType lambda) {
	for (int i = 0; i < size; i++)
		u[i] = (u_tilde[i] + tau * lambda * f[i]) / (1.0 + tau * lambda);
}

template<typename aType>
void ROFModel<aType>::Extrapolation(aType* u_bar, aType* u, aType* u_n, aType theta) {
	for (int i = 0; i < size; i++)
	{
		u_bar[i] = u[i] + theta * (u[i] - u_n[i]);
		u_n[i] = u[i];
	}
}

template<typename aType>
void ROFModel<aType>::ROF(const NumericMatrix& src, NumericMatrix& dst, aType lambda, aType tau) {
	Initialize(src);
	aType sigma = 1.f / (tau * 8.f);
	aType theta = 1.f;
	for (int k = 0; k < steps; k++)
	{
		Nabla(gradient_x, gradient_y, u_bar, p_x, p_y, sigma);
		ProxRstar(p_x, p_y, gradient_x, gradient_y, sigma);
		NablaTranspose(gradient_transpose, p_x, p_y, u_n, tau);
		ProxD(u, gradient_transpose, f, tau, lambda);
		Extrapolation(u_bar, u, u_n, theta);
	}
	SetSolution(dst);
}

#endif //__ROFMODEL_H__
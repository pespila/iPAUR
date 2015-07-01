#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include "grayscale.h"
#include "parameter.h"
#include "primal_dual_algorithm.h"

Primal_Dual::Primal_Dual(GrayscaleImage& src, int steps, int level) {
	this->steps = steps;
	this->level = level;
	this->height = src.get_height();
	this->width = src.get_width();
	this->size = height * width * level;
	this->f = (float*)malloc(this->height*this->width*sizeof(float));
	this->u = (float*)malloc(size*sizeof(float));
	this->u_n = (float*)malloc(size*sizeof(float));
	this->u_bar = (float*)malloc(size*sizeof(float));
	this->gradient_x = (float*)malloc(size*sizeof(float));
	this->gradient_y = (float*)malloc(size*sizeof(float));
	this->gradient_z = (float*)malloc(size*sizeof(float));
	this->gradient_transpose = (float*)malloc(size*sizeof(float));
	this->p_x = (float*)malloc(size*sizeof(float));
	this->p_y = (float*)malloc(size*sizeof(float));
	this->p_z = (float*)malloc(size*sizeof(float));
	this->x1 = (float*)malloc(size*sizeof(float));
	this->x2 = (float*)malloc(size*sizeof(float));
	this->x3 = (float*)malloc(size*sizeof(float));
	this->y1 = (float*)malloc(size*sizeof(float));
	this->y2 = (float*)malloc(size*sizeof(float));
	this->y3 = (float*)malloc(size*sizeof(float));
	this->p1 = (float*)malloc(size*sizeof(float));
	this->p2 = (float*)malloc(size*sizeof(float));
	this->p3 = (float*)malloc(size*sizeof(float));
	this->q1 = (float*)malloc(size*sizeof(float));
	this->q2 = (float*)malloc(size*sizeof(float));
	this->q3 = (float*)malloc(size*sizeof(float));
	this->z1 = (float*)malloc(size*sizeof(float));
	this->z2 = (float*)malloc(size*sizeof(float));
	this->g1 = (float*)malloc(size*sizeof(float));
	this->g2 = (float*)malloc(size*sizeof(float));
	this->delta1 = (float*)malloc(size*sizeof(float));
	this->delta2 = (float*)malloc(size*sizeof(float));
}

Primal_Dual::~Primal_Dual() {
	free(f);
	free(u);
	free(u_n);
	free(u_bar);
	free(gradient_x);
	free(gradient_y);
	free(gradient_z);
	free(gradient_transpose);
	free(p_x);
	free(p_y);
	free(p_z);
	free(x1);
	free(x2);
	free(x3);
	free(y1);
	free(y2);
	free(y3);
	free(p1);
	free(p2);
	free(p3);
	free(q1);
	free(q2);
	free(q3);
	free(z1);
	free(z2);
	free(delta1);
	free(delta2);
	free(g1);
	free(g2);
}

void Primal_Dual::initialize(GrayscaleImage& src) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
				f[j + i * width] = (float)src.get_pixel(i, j, 0) / 255.0;
			for (int k = 0; k < level; k++) {
				u[j + i * width + k * height * width] = ((float)src.get_pixel(i, j, 0) / 255.0) * ((float)(level-k) / (float)level);
				u_bar[j + i * width + k * height * width] = ((float)src.get_pixel(i, j, 0) / 255.0) * ((float)(level-k) / (float)level);
				p_x[j + i * width + k * height * width] = 0.0;
				p_y[j + i * width + k * height * width] = 0.0;
				p_z[j + i * width + k * height * width] = 0.0;
			}
		}
	}
}

void Primal_Dual::set_solution(GrayscaleImage& dst) {
	string filename[16] = {"01.png", "02.png", "03.png", "04.png", "05.png", "06.png", "07.png", "08.png", "09.png", "10.png", "11.png", "12.png", "13.png", "14.png", "15.png", "16.png"};
	for (int k = 0; k < level; k++)
	{
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				dst.set_pixel(i, j, 0, (unsigned char)abs(u_bar[j + i * width + k * height * width] * 255.0));
			}
		}
		dst.write_image(filename[k]);
	}
}

void Primal_Dual::nabla(float* gradient_x, float* gradient_y, float* gradient_z, float* u_bar) {
	for (int k = 0; k < level; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				gradient_x[j + i * width + k * height * width] = i + 1 < height ? (u_bar[j + (i+1) * width + k * height * width] - u_bar[j + i * width + k * height * width]) : 0.0;
				gradient_y[j + i * width + k * height * width] = j + 1 < width ? (u_bar[j + 1 + i * width + k * height * width] - u_bar[j + i * width + k * height * width]) : 0.0;
				gradient_z[j + i * width + k * height * width] = k + 1 < level ? (u_bar[j + i * width + (k+1) * height * width] - u_bar[j + i * width + k * height * width]) : 0.0;
			}
		}
	}
}

void Primal_Dual::soft_shrinkage_operator(float* x1, float* x2, float* x3, float nu) {
	for (int i = 0; i < size; i++) {
		x1[i] = fmax(abs(x1[i] - nu), 0.0) * sgn(x1[i]);
		x2[i] = fmax(abs(x2[i] - nu), 0.0) * sgn(x2[i]);
		x3[i] = fmax(abs(x3[i] - nu), 0.0) * sgn(x3[i]);
	}
}

float Primal_Dual::constraint(float x, float y, float lambda, float f, float L, int k) {
	return (0.25 * (x * x + y * y) - lambda * pow((float)k / L - f, 2));
}

void Primal_Dual::newton_projection(float* y1, float* y2, float* y3, float* f, float lambda, float L) {
	float z1_k_minus_one = 0.0;
	float z2_k_minus_one = 0.0;
	float dist = 0.0;
	float a11 = 0.0, a12 = 0.0, a21 = 0.0, a22 = 0.0;
	float schwarz = 0.0;
	float tmp = 0.0;
	int small_size = height*width;
	for (int k = 0; k < 5; k++)
	{
		for (int m = 0; m < level; m++) {
			for (int i = 0; i < small_size; i++) {
				if (y3[i + m * small_size] < constraint(y1[i + m * small_size], y2[i + m * small_size], lambda, f[i], L, m)) {
					z1_k_minus_one = z1[i + m * small_size];
					z2_k_minus_one = z2[i + m * small_size];

					dist = constraint(z1[i + m * small_size], z2[i + m * small_size], lambda, f[i + m * small_size], L, k) - y3[i + m * small_size];
					g1[i + m * small_size] = z1[i + m * small_size] - y1[i + m * small_size] + 0.5 * z1[i + m * small_size] * dist;
					g2[i + m * small_size] = z2[i + m * small_size] - y2[i + m * small_size] + 0.5 * z2[i + m * small_size] * dist;

					schwarz = 0.25 * z1[i + m * small_size]*z2[i + m * small_size];
					a11 = 1.0 + 0.25 * (dist + (z1[i + m * small_size]*z1[i + m * small_size]));
					a22 = 1.0 + 0.25 * (dist + (z2[i + m * small_size]*z2[i + m * small_size]));
					a12 = schwarz;
					a21 = schwarz;
					if ((a11*a22 - a12*a21) <= 0){
						z1[i + m * small_size] += delta1[i + m * small_size];
						z2[i + m * small_size] += delta2[i + m * small_size];
						continue;
					}
					tmp = a11;
					a11 *= a21; a12 *= a21; g1[i + m * small_size] *= a21;
					a21 *= tmp; a22 *= tmp; g2[i + m * small_size] *= tmp;
					a21 -= a11;
					a22 -= a12;

					tmp = -g2[i + m * small_size] / a22;
					delta2[i + m * small_size] = tmp;
					delta1[i + m * small_size] = (-g1[i + m * small_size] - a12 * tmp) / a11;

					z1[i + m * small_size] += delta1[i + m * small_size];
					z2[i + m * small_size] += delta2[i + m * small_size];
				}
			}
		}
	}
	for (int i = 0; i < size; i++) y1[i] = z1[i];
	for (int i = 0; i < size; i++) y2[i] = z2[i];
	for (int k = 0; k < level; k++) {
		for (int i = 0; i < small_size; i++)
		{
			y3[i + k * small_size] = constraint(y1[i + k * small_size], y2[i + k * small_size], lambda, f[i], L, k);
		}
	}
}

void Primal_Dual::dykstra_algorithm(float* p_x, float* p_y, float* p_z, float* p_tilde_x, float* p_tilde_y, float* p_tilde_z, float* f, float L, float lambda, float nu) {
	int i;
	for (i = 0; i < size; i++) x1[i] = p_tilde_x[i];
	for (i = 0; i < size; i++) x2[i] = p_tilde_y[i];
	for (i = 0; i < size; i++) x3[i] = p_tilde_z[i];
	for (int k = 0; k < 20; k++)
	{
		for (i = 0; i < size; i++) y1[i] = x1[i] + p1[i];
		for (i = 0; i < size; i++) y2[i] = x2[i] + p2[i];
		for (i = 0; i < size; i++) y3[i] = x3[i] + p3[i];
		newton_projection(y1, y2, y3, f, lambda, L);
		for (i = 0; i < size; i++) p1[i] = x1[i] + p1[i] - y1[i];
		for (i = 0; i < size; i++) p2[i] = x2[i] + p2[i] - y2[i];
		for (i = 0; i < size; i++) p3[i] = x3[i] + p3[i] - y3[i];
		for (i = 0; i < size; i++) x1[i] = y1[i] + q1[i];
		for (i = 0; i < size; i++) x2[i] = y2[i] + q2[i];
		for (i = 0; i < size; i++) x3[i] = y3[i] + q3[i];
		soft_shrinkage_operator(x1, x2, x3, nu);
		for (i = 0; i < size; i++) q1[i] = y1[i] + q1[i] - x1[i];
		for (i = 0; i < size; i++) q2[i] = y2[i] + q2[i] - x2[i];
		for (i = 0; i < size; i++) q3[i] = y3[i] + q3[i] - x3[i];
	}
	for (i = 0; i < size; i++) p_x[i] = x1[i];
	for (i = 0; i < size; i++) p_y[i] = x2[i];
	for (i = 0; i < size; i++) p_z[i] = x3[i];
}

void Primal_Dual::nabla_transpose(float* gradient_transpose, float* p_x, float* p_y, float* p_z) {
	float x = 0.0;
	float x_minus_one = 0.0;
	for (int k = 0; k < level; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				x = i + 1 < height ? p_x[j + i * width + k * height * width] : 0.0;
				x_minus_one = i > 0 ? p_x[j + (i-1) * width + k * height * width] : 0.0;
				gradient_transpose[j + i * width + k * height * width] = x_minus_one - x;
				x = j + 1 < width ? p_y[j + i * width + k * height * width] : 0.0;
				x_minus_one = j > 0 ? p_y[j - 1 + i * width + k * height * width] : 0.0;
				gradient_transpose[j + i * width + k * height * width] += (x_minus_one - x);
				x = k + 1 < level ? p_z[j + i * width + k * height * width] : 0.0;
				x_minus_one = k > 0 ? p_z[j + i * width + (k-1) * height * width] : 0.0;
				gradient_transpose[j + i * width + k * height * width] += (x_minus_one - x);
			}
		}
	}
}

void Primal_Dual::truncation_operator(float* u, float* u_tilde) {
	for (int k = 0; k < level; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				u[j + i * width + k * height * width] = fmin(1.0, fmax(0.0, u_tilde[j + i * width + k * height * width]));
			}
		}
	}
}

void Primal_Dual::primal_dual_algorithm(GrayscaleImage& src, GrayscaleImage& dst, Parameter& par) {
	int i;
	dst.reset_image(height, width, src.get_type());
	initialize(src);
	for (int k = 0; k < steps; k++)
	{
		for (i = 0; i < size; i++) {u_n[i] = u[i];}
		nabla(gradient_x, gradient_y, gradient_z, u_bar);
		for (i = 0; i < size; i++) {gradient_x[i] = par.sigma * gradient_x[i] + p_x[i];}
		for (i = 0; i < size; i++) {gradient_y[i] = par.sigma * gradient_y[i] + p_y[i];}
		for (i = 0; i < size; i++) {gradient_z[i] = par.sigma * gradient_z[i] + p_z[i];}
		dykstra_algorithm(p_x, p_y, p_z, gradient_x, gradient_y, gradient_z, f, par.L, par.lambda, par.nu);
		nabla_transpose(gradient_transpose, p_x, p_y, p_z);
		for (i = 0; i < size; i++) {gradient_transpose[i] = u_n[i] - par.tau * gradient_transpose[i];}
		truncation_operator(u, gradient_transpose);
		for (i = 0; i < size; i++) {u_bar[i] = u[i] + par.theta * (u[i] - u_n[i]);}
	}
	set_solution(dst);
}
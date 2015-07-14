#include "PrimalDualAlgorithm.h"

PrimalDualAlgorithm::PrimalDualAlgorithm(Image& src, int level, int steps) {
	this->level = level;
	this->steps = steps;
	this->height = src.GetHeight();
	this->width = src.GetWidth();
	this->size = height * width * level;
	this->solution = (float*)malloc(height*width*sizeof(float));
	this->f = (float*)malloc(height*width*sizeof(float));
	this->u = (float*)malloc(size*sizeof(float));
	this->u_n = (float*)malloc(size*sizeof(float));
	this->u_bar = (float*)malloc(size*sizeof(float));
	this->gradient.x1 = (float*)malloc(size*sizeof(float));
	this->gradient.x2 = (float*)malloc(size*sizeof(float));
	this->gradient.x3 = (float*)malloc(size*sizeof(float));
	this->gradient_transpose = (float*)malloc(size*sizeof(float));
	this->p_dual.x1 = (float*)malloc(size*sizeof(float));
	this->p_dual.x2 = (float*)malloc(size*sizeof(float));
	this->p_dual.x3 = (float*)malloc(size*sizeof(float));
	this->x.x1 = (float*)malloc(size*sizeof(float));
	this->x.x2 = (float*)malloc(size*sizeof(float));
	this->x.x3 = (float*)malloc(size*sizeof(float));
	this->y.x1 = (float*)malloc(size*sizeof(float));
	this->y.x2 = (float*)malloc(size*sizeof(float));
	this->y.x3 = (float*)malloc(size*sizeof(float));
	this->p.x1 = (float*)malloc(size*sizeof(float));
	this->p.x2 = (float*)malloc(size*sizeof(float));
	this->p.x3 = (float*)malloc(size*sizeof(float));
	this->q.x1 = (float*)malloc(size*sizeof(float));
	this->q.x2 = (float*)malloc(size*sizeof(float));
	this->q.x3 = (float*)malloc(size*sizeof(float));
}

PrimalDualAlgorithm::~PrimalDualAlgorithm() {
	free(solution);
	free(f);
	free(u);
	free(u_n);
	free(u_bar);
	free(gradient.x1);
	free(gradient.x2);
	free(gradient.x3);
	free(gradient_transpose);
	free(p_dual.x1);
	free(p_dual.x2);
	free(p_dual.x3);
	free(x.x1);
	free(x.x2);
	free(x.x3);
	free(y.x1);
	free(y.x2);
	free(y.x3);
	free(p.x1);
	free(p.x2);
	free(p.x3);
	free(q.x1);
	free(q.x2);
	free(q.x3);
}

void PrimalDualAlgorithm::ScaleArray(float* out, float factor, float* in) {
	for (int i = 0; i < size; i++)
		out[i] = factor * in[i];
}

void PrimalDualAlgorithm::AddArray(float* out, float factor1, float* in1, float factor2, float* in2) {
	for (int i = 0; i < size; i++)
		out[i] = factor1 * in1[i] - factor2 * in2[i];
}

void PrimalDualAlgorithm::ScaleVector3D(struct Vector3D out, float factor, struct Vector3D in) {
	for (int i = 0; i < size; i++) {
		out.x1[i] = factor * in.x1[i];
		out.x2[i] = factor * in.x2[i];
		out.x3[i] = factor * in.x3[i];
	}
}

void PrimalDualAlgorithm::AddVector3D(struct Vector3D out, float factor1, struct Vector3D in1, float factor2, struct Vector3D in2) {
	for (int i = 0; i < size; i++) {
		out.x1[i] = factor1 * in1.x1[i] + factor2 * in2.x1[i];
		out.x2[i] = factor1 * in1.x2[i] + factor2 * in2.x2[i];
		out.x3[i] = factor1 * in1.x3[i] + factor2 * in2.x3[i];
	}
}

void PrimalDualAlgorithm::Nabla(struct Vector3D gradient, float* u_bar) {
	int index;
	for (int k = 0; k < level; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				index = j + i * width + k * height * width;
				gradient.x1[index] = i + 1 < height ? (u_bar[j + (i+1) * width + k * height * width] - u_bar[index]) : 0.0;
				gradient.x2[index] = j + 1 < width ? (u_bar[j + 1 + i * width + k * height * width] - u_bar[index]) : 0.0;
				gradient.x3[index] = k + 1 < level ? (u_bar[j + i * width + (k+1) * height * width] - u_bar[index]) : 0.0;
				// gradient.x3[index] = k + 1 < level ? (u_bar[j + i * width + (k+1) * height * width] - u_bar[index]) : - u_bar[index];
			}
		}
	}
}

void PrimalDualAlgorithm::ProjectionOntoParabola(struct Vector3D y, struct Vector3D p) {
	float l2norm, v, a, b, c, d, alpha = 0.25;
	for (int i = 0; i < size; i++) {
		l2norm = sqrt(pow(p.x1[i], 2) + pow(p.x2[i], 2));
		if (alpha * pow(l2norm, 2) <= p.x3[i]) continue;
		a = 2.0 * alpha * l2norm;
		b = 2.0 / 3.0 * (1.0 - 2.0 * alpha * p.x3[i]);
		d = b < 0 ? (a - pow(sqrt(-b), 3)) * (a + pow(sqrt(-b), 3)) : a * a + b * b * b;
		c = pow((a + sqrt(d)), 1.0 / 3.0);
		if (d >= 0) {
			v = c != 0.0 ? c - b / c : 0.0;
		} else {
			v = 2.0 * sqrt(-b) * cos((1.0 / 3.0) * acos(a / (pow(sqrt(-b), 3))));
		}
		y.x1[i] = p.x1[i] != 0.0 ? (v / (2.0 * alpha)) * (p.x1[i] / l2norm) : 0.0;
		y.x2[i] = p.x2[i] != 0.0 ? (v / (2.0 * alpha)) * (p.x2[i] / l2norm) : 0.0;
		y.x3[i] = alpha * pow(l2norm, 2);
	}
}

void PrimalDualAlgorithm::ProjectionOntoConvexCone(struct Vector3D x, struct Vector3D q, float nu) {
	float l2norm;
	for (int i = 0; i < size; i++) {
		l2norm = sqrt(pow(q.x1[i], 2) + pow(q.x2[i], 2));
		x.x1[i] = l2norm <= nu ? p.x1[i] : nu * p.x1[i] / l2norm;
		x.x2[i] = l2norm <= nu ? p.x2[i] : nu * p.x2[i] / l2norm;
	}
}

void PrimalDualAlgorithm::DykstraAlgorithm(struct Vector3D x_bar, struct Vector3D r, float* f, float L, float lambda, float nu, int max_iter) {
	ScaleVector3D(x, 1.0, r);
	for (int i = 0; i < max_iter; i++) {
		AddVector3D(p, 1.0, x, 1.0, p);
		ProjectionOntoParabola(y, p);
		AddVector3D(p, 1.0, p, -1.0, y);
		AddVector3D(q, 1.0, y, 1.0, q);
		// ProjectionOntoConvexCone(x, q, nu);
		AddVector3D(q, 1.0, q, -1.0, x);
	}
	ScaleVector3D(x_bar, 1.0, x);
}

void PrimalDualAlgorithm::NablaTranspose(float* gradient_transpose, struct Vector3D p_dual) {
	float x , x_minus_one;
	int index;
	for (int k = 0; k < level; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				index = j + i * width + k * height * width;
				x = i + 1 < height ? p_dual.x1[index] : 0.0;
				x_minus_one = i > 0 ? p_dual.x1[j + (i-1) * width + k * height * width] : 0.0;
				gradient_transpose[index] = x_minus_one - x;
				x = j + 1 < width ? p_dual.x2[index] : 0.0;
				x_minus_one = j > 0 ? p_dual.x2[j - 1 + i * width + k * height * width] : 0.0;
				gradient_transpose[index] += (x_minus_one - x);
				x = k + 1 < level ? p_dual.x3[index] : 0.0;
				x_minus_one = k > 0 ? p_dual.x3[j + i * width + (k-1) * height * width] : 0.0;
				gradient_transpose[index] += (x_minus_one - x);
				// x = k + 1 < level ? p_dual.x3[index] : 0.0;
				// x_minus_one = k > 0 ? p_dual.x3[j + i * width + (k-1) * height * width] : - p_dual.x3[index];
				// gradient_transpose[index] += (x_minus_one - x);
			}
		}
	}
}

void PrimalDualAlgorithm::TruncationOperation(float* u, float* u_tilde) {
	for (int i = 0; i < size; i++)
		u[i] = fmin(1.0, fmax(0.0, u_tilde[i]));
}

void PrimalDualAlgorithm::ComputeIsosurface(float* u) {
	float uk0, uk1;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < level-1; k++) {
				uk0 = u[j + i * width + k * height * width];
				uk1 = u[j + i * width + (k+1) * height * width];
				if (uk0 > 0.5 && uk1 <= 0.5) {
					solution[j + i * width] = (0.5 - uk0) / (uk1 - uk0) + (k + 1) / (level);
					break;
				}
			}
		}
	}
}

void PrimalDualAlgorithm::SetSolution(WriteableImage& dst) {
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			dst.Set(i, j, 0, (unsigned char)(solution[j + i * width] * 255.0));
}


void PrimalDualAlgorithm::Initialize(Image& src) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			f[j + i * width] = (float)src.Get(i, j, 0) / 255.0;
			for (int k = 0; k < level; k++) {
				if (((k + 1) / level) <= src.Get(i, j, 0)) {
					u[j + i * width + k * height * width] = (float)src.Get(i, j, 0) / 255.0;
				}
				if (k == 0) {
					u[j + i * width + k * height * width] = 0.0;
				}
				u_bar[j + i * width + k * height + width] = u[j + i * width + k * height * width];
				p_dual.x1[j + i * width + k * height * width] = 0.0;
				p_dual.x2[j + i * width + k * height * width] = 0.0;
				p_dual.x3[j + i * width + k * height * width] = 0.0;
			}
		}
	}
}

void PrimalDualAlgorithm::PrimalDual(Image& src, WriteableImage& dst, Parameter& par, int dykstra_max_iter) {
	dst.Reset(height, width, src.GetType());
	Initialize(src);
	for (int k = 0; k < steps; k++) {
		ScaleArray(u_n, 1.0, u);
		Nabla(gradient, u_bar);
		AddVector3D(gradient, par.sigma, gradient, 1.0, p_dual);
		DykstraAlgorithm(p_dual, gradient, f, par.L, par.lambda, par.nu, dykstra_max_iter);
		NablaTranspose(gradient_transpose, p_dual);
		AddArray(gradient_transpose, 1.0, u_n, -par.tau, gradient_transpose);
		TruncationOperation(u, gradient_transpose);
		AddArray(u_bar, (1.0 + par.theta), u, par.theta, u_n);
	}
	ComputeIsosurface(u_bar);
	SetSolution(dst);
}
#include "PrimalDualAlgorithm.h"

PrimalDualAlgorithm::PrimalDualAlgorithm(Image& src, int level, int steps) {
	this->level = level;
	this->steps = steps;
	this->height = src.GetHeight();
	this->width = src.GetWidth();
	this->size = height * width * level;
	this->solution = new float[height*width];
	this->f = new float[height*width];
	this->u = new float[size];
	this->u_n = new float[size];
	this->u_bar = new float[size];
	this->gradient_transpose = new float[size];
	this->gradient = Vector3D(height, width, level);
	this->p_dual = Vector3D(height, width, level);
	this->x = Vector3D(height, width, level);
	this->y = Vector3D(height, width, level);
	this->p = Vector3D(height, width, level);
	this->q = Vector3D(height, width, level);
}

PrimalDualAlgorithm::~PrimalDualAlgorithm() {
	delete[] solution;
	delete[] f;
	delete[] u;
	delete[] u_n;
	delete[] u_bar;
	delete[] gradient_transpose;
	gradient.Free();
	p_dual.Free();
	x.Free();
	y.Free();
	p.Free();
	q.Free();
}

void PrimalDualAlgorithm::ScaleArray(float* out, float factor, float* in) {
	for (int i = 0; i < size; i++)
		out[i] = factor * in[i];
}

void PrimalDualAlgorithm::AddArray(float* out, float factor1, float* in1, float factor2, float* in2) {
	for (int i = 0; i < size; i++)
		out[i] = factor1 * in1[i] + factor2 * in2[i];
}

void PrimalDualAlgorithm::ScaleVector3D(Vector3D& out, float factor, Vector3D& in) {
	for (int c = 0; c < 3; c++)
		for (int k = 0; k < level; k++)
			for (int i = 0; i < height; i++)
				for (int j = 0; j < width; j++)
					out.Set(i, j, k, c, factor * in.Get(i, j, k, c));
}

void PrimalDualAlgorithm::AddVector3D(Vector3D& out, float factor1, Vector3D& in1, float factor2, Vector3D& in2) {
	for (int c = 0; c < 3; c++)
		for (int k = 0; k < level; k++)
			for (int i = 0; i < height; i++)
				for (int j = 0; j < width; j++)
					out.Set(i, j, k, c, factor1 * in1.Get(i, j, k, c) + factor2 * in2.Get(i, j, k, c));
}

void PrimalDualAlgorithm::Nabla(Vector3D& gradient, float* u_bar) {
	float nabla_x = 0.0;
	float nabla_y = 0.0;
	float nabla_z = 0.0;
	for (int k = 0; k < level; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				nabla_x = i + 1 < height ? (u_bar[j + (i+1) * width + k * height * width] - u_bar[j + i * width + k * height * width]) : 0.0;
				nabla_y = j + 1 < width ? (u_bar[j + 1 + i * width + k * height * width] - u_bar[j + i * width + k * height * width]) : 0.0;
				nabla_z = k + 1 < level ? (u_bar[j + i * width + (k+1) * height * width] - u_bar[j + i * width + k * height * width]) : 0.0;
				gradient.Set(i, j, k, 0, nabla_x);
				gradient.Set(i, j, k, 1, nabla_y);
				gradient.Set(i, j, k, 2, nabla_z);
			}
		}
	}
}

void PrimalDualAlgorithm::ProjectionOntoParabola(Vector3D& y, Vector3D& p) {
	float l2norm, v, a, b, c, d, value, alpha = 0.25;
	for (int k = 0; k < level; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				l2norm = sqrt(pow(p.Get(i, j, k, 0), 2) + pow(p.Get(i, j, k, 1), 2));
				if (alpha * pow(l2norm, 2) > p.Get(i, j, k, 2)) {
					a = 2.0 * alpha * l2norm;
					b = 2.0 / 3.0 * (1.0 - 2.0 * alpha * p.Get(i, j, k, 2));
					d = b < 0 ? (a - pow(sqrt(-b), 3)) * (a + pow(sqrt(-b), 3)) : a * a + b * b * b;
					c = pow((a + sqrt(d)), 1.0 / 3.0);
					if (d >= 0) {
						v = c != 0.0 ? c - b / c : 0.0;
					} else {
						v = 2.0 * sqrt(-b) * cos((1.0 / 3.0) * acos(a / (pow(sqrt(-b), 3))));
					}
					for (int c = 0; c < 2; c++) {
						value = p.Get(i, j, k, c) != 0.0 ? (v / (2.0 * alpha)) * (p.Get(i, j, k, c) / l2norm) : 0.0;
						y.Set(i, j, k, c, value);
					}
					y.Set(i, j, k, 2, alpha * pow(l2norm, 2));
				}
			}
		}
	}
}

// void PrimalDualAlgorithm::ProjectionOntoConvexCones(Vector3D& x, Vector3D& q, float nu, int k1, int k2) {
// 	float K = (float)(k2 - k1) + 1.0;
// 	float s = 0.0;
// 	float s0 = 0.0;
// 	int i, j, k;
// 	for (i = 0; i < height; i++)
// 	{
// 		for (j = 0; j < width; j++)
// 		{
// 			s0 = 0.0;
// 			for (k = k1; k < k2; k++) {
// 				s0 += q[j + i * width + k * height * width];
// 			}
// 			for (k = 0; k < level; k++)
// 			{
// 				if (k < k1) {
// 					x[j + i * width + k * height * width] = q[j + i * width + k * height * width];
// 				} else if (k > k2) {
// 					x[j + i * width + k * height * width] = q[j + i * width + k * height * width];
// 				} else {
// 					x[j + i * width + k * height * width] = q[j + i * width + k * height * width] + (s - s0) / K;
// 				}
// 			}
// 		}
// 	}
// 	float l2norm;
// 	for (int i = 0; i < size; i++) {
// 		l2norm = sqrt(pow(q.x1[i], 2) + pow(q.x2[i], 2));
// 		x.x1[i] = l2norm <= nu ? p.x1[i] : nu * p.x1[i] / l2norm;
// 		x.x2[i] = l2norm <= nu ? p.x2[i] : nu * p.x2[i] / l2norm;
// 	}
// }

void PrimalDualAlgorithm::DykstraAlgorithm(Vector3D& x_bar, Vector3D& r, float* f, float L, float lambda, float nu, int max_iter) {
	ScaleVector3D(x, 1.0, r);
	for (int i = 0; i < max_iter; i++) {
		AddVector3D(p, 1.0, x, 1.0, p);
		ProjectionOntoParabola(y, p);
		AddVector3D(p, 1.0, p, -1.0, y);
		AddVector3D(q, 1.0, y, 1.0, q);
		// ProjectionOntoConvexCones(x, q, nu);
		AddVector3D(q, 1.0, q, -1.0, x);
	}
	ScaleVector3D(x_bar, 1.0, x);
}

// void PrimalDualAlgorithm::DykstraAlgorithm(Vector3D& x_bar, Vector3D& r, float* f, float L, float lambda, float nu, int max_iter) {
// 	int k1 = 0, k2 = level;
// 	ScaleVector3D(x, 1.0, r);
// 	for (int k = 0; k < max_iter; k++) {
// 		AddVector3D(p, 1.0, x, 1.0, p);
// 		ProjectionOntoParabola(y, p);
// 		AddVector3D(p, 1.0, p, -1.0, y);
// 		for (int i = 0; i < k1; i++)
// 		{
// 			for (int j = 0; j < k2; j++)
// 			{
// 				AddVector3D(q, 1.0, y, 1.0, q);
// 				ProjectionOntoConvexCones(x, q, nu, i, j);
// 				AddVector3D(q, 1.0, q, -1.0, x);
// 			}
// 		}
// 	}
// 	ScaleVector3D(x_bar, 1.0, x);
// }

void PrimalDualAlgorithm::NablaTranspose(float* gradient_transpose, Vector3D& p_dual) {
	float x , x_minus_one, nabla_transpose;
	for (int k = 0; k < level; k++) {
		for (int i = 0; i < height; i++) {
			nabla_transpose = 0.0;
			for (int j = 0; j < width; j++) {
				x = i + 1 < height ? p_dual.Get(i, j, k, 0) : 0.0;
				x_minus_one = i > 0 ? p_dual.Get(i-1, j, k, 0) : 0.0;
				nabla_transpose += (x_minus_one - x);
				x = j + 1 < width ? p_dual.Get(i, j, k, 1) : 0.0;
				x_minus_one = j > 0 ? p_dual.Get(i, j-1, k, 1) : 0.0;
				nabla_transpose += (x_minus_one - x);
				x = k + 1 < level ? p_dual.Get(i, j, k, 2) : 0.0;
				x_minus_one = k > 0 ? p_dual.Get(i, j, k-1, 2) : 0.0;
				nabla_transpose += (x_minus_one - x);
				gradient_transpose[j + i * width + k * height * width] = nabla_transpose;
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
				u[j + i * width + k * height * width] = k <= ((float)src.Get(i, j, 0) / 255.0) * level ? 1.0 : 0.0;
				u_bar[j + i * width + k * height + width] = u[j + i * width + k * height * width];
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
		AddArray(u_bar, 2.0, u, -1.0, u_n);
	}
	ComputeIsosurface(u_bar);
	SetSolution(dst);
}
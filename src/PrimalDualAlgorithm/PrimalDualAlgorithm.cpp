#include "PrimalDualAlgorithm.h"

PrimalDualAlgorithm::PrimalDualAlgorithm(Image& src, int level) {
	this->level = level;
	this->height = src.GetHeight();
	this->width = src.GetWidth();
	this->u = Vectors(height, width, level, 1);
	this->u_n = Vectors(height, width, level, 1);
	this->u_bar = Vectors(height, width, level, 1);
	this->gradient_transpose = Vectors(height, width, level, 1);
	this->gradient = Vectors(height, width, level, 3);
	this->p_dual = Vectors(height, width, level, 3);
	this->x = Vectors(height, width, level, 3);
	this->y = Vectors(height, width, level, 3);
	this->p = Vectors(height, width, level, 3);
	this->q = Vectors(height, width, level, 3);
}

void PrimalDualAlgorithm::ScaleVectors(Vectors& out, float factor, Vectors& in) {
	for (int c = 0; c < in.Dimension(); c++)
		for (int k = 0; k < level; k++)
			for (int i = 0; i < height; i++)
				for (int j = 0; j < width; j++)
					out.Set(i, j, k, c, factor * in.Get(i, j, k, c));
}

void PrimalDualAlgorithm::AddVectors(Vectors& out, float factor1, Vectors& in1, float factor2, Vectors& in2) {
	for (int c = 0; c < in1.Dimension(); c++)
		for (int k = 0; k < level; k++)
			for (int i = 0; i < height; i++)
				for (int j = 0; j < width; j++)
					out.Set(i, j, k, c, factor1 * in1.Get(i, j, k, c) + factor2 * in2.Get(i, j, k, c));
}

void PrimalDualAlgorithm::Nabla(Vectors& gradient, Vectors& u_bar) {
	float current;
	for (int k = 0; k < level; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				current = u_bar.Get(i, j, k, 0);
				gradient.Set(i, j, k, 0, i + 1 < height ? u_bar.Get(i+1, j, k, 0) - current : 0.0);
				gradient.Set(i, j, k, 1, j + 1 < width ? u_bar.Get(i, j+1, k, 0) - current : 0.0);
				gradient.Set(i, j, k, 2, k + 1 < level ? u_bar.Get(i, j, k+1, 0) - current : 0.0);
			}
		}
	}
}

void PrimalDualAlgorithm::NablaTranspose(Vectors& gradient_transpose, Vectors& p_dual) {
	float nabla_transpose;
	for (int k = 0; k < level; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				nabla_transpose = 0.0;
				nabla_transpose += i > 0 ? p_dual.Get(i-1, j, k, 0) : 0.0 - i + 1 < height ? p_dual.Get(i, j, k, 0) : 0.0;
				nabla_transpose += j > 0 ? p_dual.Get(i, j-1, k, 1) : 0.0 - j + 1 < width ? p_dual.Get(i, j, k, 1) : 0.0;
				nabla_transpose += k > 0 ? p_dual.Get(i, j, k-1, 2) : 0.0 - k + 1 < level ? p_dual.Get(i, j, k, 2) : 0.0;
				gradient_transpose.Set(i, j, k, 0, nabla_transpose);
			}
		}
	}
}

void PrimalDualAlgorithm::TruncationOperation(Vectors& u, Vectors& u_tilde) {
	for (int c = 0; c < u_tilde.Dimension(); c++)
		for (int k = 0; k < level; k++)
			for (int i = 0; i < height; i++)
				for (int j = 0; j < width; j++)
					u.Set(i, j, k, c, fmin(1.0, fmin(0.0, u_tilde.Get(i, j, k, c))));
}

float PrimalDualAlgorithm::L2Norm(Vectors& x, int size) {
	float norm = 0.0;
	for (int i = 0; i < size; i++) {
		norm += pow(x.Get(0, i, 0, 0), 2);
	}
	return sqrt(norm);
}

void PrimalDualAlgorithm::L2Projection(Vectors& out, Vectors& in, float radius) {
	float norm = L2Norm(in, in.Dimension());
	for (int i = 0; i < in.Dimension(); i++) {
		if (norm <= radius) {
			out.Set(0, i, 0, 0, in.Get(0, i, 0, 0));
		} else {
			out.Set(0, i, 0, 0, radius * (in.Get(0, i, 0, 0) / norm));
		}
	}
}

void PrimalDualAlgorithm::SoftShrinkageScheme(Vectors& out, Vectors& in, int i, int j, int k1, int k2, float nu) {
	float K = (float)(k2 - k1 + 1);
	Vectors s(1, 1, 1, 2), s0(1, 1, 1, 2);
	for (int k = k1; k < k2; k++) {
		for (int c = 0; c < 2; c++) {
			s0.Set(0, k, 0, 0, s0.Get(0, k, 0, 0) + in.Get(i, j, k, c));
		}
	}
	L2Projection(s, s0, nu);
	for (int k = 0; k < level; k++) {
		for (int c = 0; c < 2; c++) {
			if (k >= k1 && k < k2) {
				out[i].Set(j, in[i].Get(j) + (s.Get(j) - s0.Get(j)) / K);
			} else {
				out.Set(i, j, k, c, in.Get(i, j, k, c));
			}
		}
	}
}

void PrimalDualAlgorithm::ProjectionOntoParabola(Vectors& y, Vectors& p, Image& f, float lambda, float L, int i, int j, int k) {
	float l2norm, v, a, b, c, d, value, bound, alpha = 0.25;
	bound = p.Get(i, j, k, 2) + lambda * pow(float(k) / L - f.Get(i, j, 0), 2);
	l2norm = sqrt(pow(p.Get(i, j, k, 0), 2) + pow(p.Get(i, j, k, 1), 2));
	if (alpha * pow(l2norm, 2) > bound) {
		a = 2.0 * alpha * l2norm;
		b = 2.0 / 3.0 * (1.0 - 2.0 * alpha * bound);
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
	} else {
		for (int c = 0; c < 2; c++) {
			y.Set(i, j, k, c, p.Get(i, j, k, c));
		}
	}
}

void PrimalDualAlgorithm::Add(Vectors& out, float factor1, Vectors& in1, float factor2, Vectors& in2, int i, int j, int k) {
	for (int c = 0; c < in1.Dimension(); c++)
		out.Set(i, j, k, c, factor1 * in1.Get(i, j, k, c) + factor2 * in2.Get(i, j, k, c));
}

void PrimalDualAlgorithm::DykstraAlgorithm(Vectors& out, Vectors& in, Image& f, float lambda, float L, float nu, int dykstra_max_iter) {
	int projections = level * level + level;
	Vectors* corrections = new Vectors[projections];
	Vectors* variables = new Vectors[projections];

	for (int i = 0; i < projections; i++) {
		corrections[i] = Vectors(3);
		variables[i] = Vectors(3);
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			for (int m = 0; m < dykstra_max_iter; m++) {
				for (int x = 0; x < projections; x++) {
					for (int k1 = 0; k1 < level; k1++) {
						if (x < level) {
							if (m == 0) {
								Add(corrections[x], 1.0, in, 1.0, corrections[x], i, j, k1);
							} else {
								Add(corrections[x], 1.0, variables[level * level + level-1], 1.0, corrections[x], i, j, k1);
							}
							ProjectionOntoParabola(variables[x], corrections[x], f, lambda, L, i, j, k1);
							Add(corrections[x], 1.0, corrections[x], -1.0, variables[x], i, j, k1);
						} else {
							for (int k2 = 0; k2 < level; k2++) {
								Add(corrections[x], 1.0, variables[x-1], 1.0, corrections[x], i, j, k1);
								SoftShrinkageScheme(variables[x], corrections[x], i, j, k1, k2, nu);
								Add(corrections[x], 1.0, corrections[x], -1.0, variables[x], i, j, k1);
							}
						}
					}
				}
			}
		}
	}
	
	delete[] corrections;
	delete[] variables;
}

void PrimalDualAlgorithm::ComputeIsosurface(WriteableImage& dst, Vectors& u) {
	float uk0, uk1;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < level-1; k++) {
				uk0 = u.Get(i, j, k, 0);
				uk1 = u.Get(i, j, k+1, 0);
				if (uk0 > 0.5 && uk1 <= 0.5) {
					dst.Set(i, j, 0, (0.5 - uk0) / (uk1 - uk0) + (k + 1) / (level));
					break;
				}
			}
		}
	}
}

void PrimalDualAlgorithm::PrimalDual(Image& src, WriteableImage& dst, Parameter& par, int dykstra_max_iter, int steps) {
	dst.Reset(height, width, src.GetType());
	for (int k = 0; k < steps; k++) {
		ScaleVectors(u_n, 1.0, u);
		Nabla(gradient, u_bar);
		AddVectors(gradient, par.sigma, gradient, 1.0, p_dual);
		DykstraAlgorithm(p_dual, gradient, src, par.lambda, par.L, par.nu, dykstra_max_iter);
		NablaTranspose(gradient_transpose, p_dual);
		AddVectors(gradient_transpose, 1.0, u_n, par.tau, gradient_transpose);
		TruncationOperation(u, gradient_transpose);
		AddVectors(u_bar, 2.0, u, -1.0, u_n);
	}
	ComputeIsosurface(dst, u_bar);
}
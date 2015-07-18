#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
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

float PrimalDualAlgorithm::EuclideanDistance(float* u, float* u_n) {
	float l2norm = 0.0;
	for (int i = 0; i < size; i++)
		l2norm += pow(u[i] - u_n[i], 2);
	return sqrt(l2norm);
}

float PrimalDualAlgorithm::EuclideanDistance3D(Vector3D& u, Vector3D& u_n) {
	float l2norm = 0.0;
	for (int c = 0; c < 3; c++)
		for (int k = 0; k < level; k++)
			for (int i = 0; i < height; i++)
				for (int j = 0; j < width; j++)
					l2norm += pow(u.Get(i, j, k, c) - u_n.Get(i, j, k, c), 2);

	return sqrt(l2norm);
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

void PrimalDualAlgorithm::ProjectionOntoParabola(Vector3D& y, Vector3D& p, float* f, float lambda, float L) {
	float l2norm, v, a, b, c, d, value, bound, alpha = 0.25;
	for (int k = 0; k < level; k++) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				bound = p.Get(i, j, k, 2) + lambda * pow(float(k) / L - f[j + i * width], 2);
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
				}
			}
		}
	}
}

void PrimalDualAlgorithm::ProjectionOntoConvexCones(Vector3D& x, Vector3D& q, float nu, int k1, int k2) {
	float K = (float)(k2 - k1) + 1.0;
	float s1, s2, s01, s02, l2norm;
	int i, j, k;
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			s01 = 0.0;
			s02 = 0.0;
			for (k = k1; k < k2; k++)
			{
				s01 += q.Get(i, j, k, 0);
				s02 += q.Get(i, j, k, 1);
			}
			l2norm = sqrt(s01*s01 + s02*s02);
			if (l2norm <= nu) {
				s1 = s01;
				s2 = s02;
			} else {
				s1 = s01 / l2norm * nu;
				s2 = s02 / l2norm * nu;
			}
			for (k = 0; k < level; k++)
			{
				if (k < k1 || k > k2) {
					x.Set(i, j, k, 0, q.Get(i, j, k, 0));
					x.Set(i, j, k, 1, q.Get(i, j, k, 1));
				} else {
					x.Set(i, j, k, 0, q.Get(i, j, k, 0) + (s1 - s01) / K);
					x.Set(i, j, k, 1, q.Get(i, j, k, 1) + (s2 - s02) / K);
				}
			}
		}
	}
}

// void Dykstra(Point& r) {
// 	Point y, p, q, z, x(r.x, r.y);
// 	int max = 2;
// 	Point* var = new Point[max];
// 	Point* cor = new Point[max];
// 	int i, k;
// 	for (i = 0; i < 100; i++) {
// 		for (k = 0; k < max; k++) {
// 			if (k == 0) {
// 				Add(cor[k], 1.0, x, 1.0, cor[k]);
// 			} else {
// 				Add(cor[k], 1.0, var[k-1], 1.0, cor[k]);
// 			}
// 			L2Projection(var[k], cor[k], k, 0.0, 1.0);
// 			Add(cor[k], 1.0, cor[k], -1.0, var[k]);
// 		}
// 		r.x = var[max-1].x;
// 		r.y = var[max-1].y;
// 	}
// 	delete[] var;
// 	delete[] cor;
// }

void PrimalDualAlgorithm::DykstraAlgorithm(Vector3D& x_bar, Vector3D& r, float* f, float lambda, float L, float nu, int max_iter) {
	int i, j, k, m;
	// ScaleVector3D(x, 1.0, r);
	// int k1 = 0, k2 = level;
	Vector3D* variables = new Vector3D[level * level + 1];
	Vector3D* corrections = new Vector3D[level * level + 1];
	for (k = 0; k < max_iter; k++) {
		for (i = 0; i < level; i++) {
			for (j = 0; j < level; j++) {
				m = i + j;
				if (m == 0) {
					AddVector3D(corrections[m], 1.0, r, 1.0, corrections[m]);
				} else {
					AddVector3D(corrections[m], 1.0, variables[m-1], 1.0, corrections[m]);
				}
				if (i == level - 1 && j == level - 1) {
					ProjectionOntoParabola(variables[m], corrections[m], f, lambda, L);
				} else {
					ProjectionOntoConvexCones(variables[m], corrections[m], nu, i, j);
					AddVector3D(corrections[m], 1.0, corrections[m], -1.0, variables[m]);
				}
			}
		}
	}
	ScaleVector3D(x_bar, 1.0, corrections[level * level]);
	// for (k = 0; k < max_iter; k++) {
	// 	AddVector3D(p, 1.0, x, 1.0, p);
	// 	ProjectionOntoParabola(y, p, f, lambda, L);
	// 	AddVector3D(p, 1.0, p, -1.0, y);
	// 	AddVector3D(x, 1.0, y, 0.0, y);
	// 	for (int i = 0; i < k1; i++)
	// 	{
	// 		for (int j = 0; j < k2; j++)
	// 		{
	// 			AddVector3D(q, 1.0, x, 1.0, q);
	// 			ProjectionOntoConvexCones(x, q, nu, i, j);
	// 			AddVector3D(q, 1.0, q, -1.0, x);
	// 		}
	// 	}
	// 	ScaleVector3D(x_bar, 1.0, x);
	// }
	delete[] variables;
	delete[] corrections;
}

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
			// for (int k = 0; k < level; k++) {
			// 	// u[j + i * width + k * height * width] = f[j + i * width];
			// 	u[j + i * width + k * height * width] = k <= ((float)src.Get(i, j, 0) / 255.0) * level ? 1.0 : 0.0;
			// 	u_bar[j + i * width + k * height + width] = u[j + i * width + k * height * width];
			// 	for (int c = 0; c < 3; c++)
			// 	{
			// 		p_dual.Set(i, j, k, c, 0.0);
			// 	}
			// }
		}
	}
}

Mat PrimalDualAlgorithm::Write(Image& src) {
	Mat img(height, width, src.GetType());
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img.at<uchar>(i, j) = src.Get(i, j, 0);
	return img;
}

void PrimalDualAlgorithm::PrimalDual(Image& src, WriteableImage& dst, Parameter& par, int dykstra_max_iter) {
	dst.Reset(height, width, src.GetType());
	Initialize(src);
	// Mat gray, color;
	// VideoWriter output_cap("/Users/michael/Documents/Programming/image-processing/img/eye.mp4",
 //                                  CV_FOURCC('m', 'p', '4', 'v'),
 //                                  30,
 //                                  cv::Size(width, height),
 //                                  false);
	// if (!output_cap.isOpened()) {
	// 	printf("ERROR by opening!\n");
 //   	}
	for (int k = 0; k < steps; k++) {
		ScaleArray(u_n, 1.0, u);
		Nabla(gradient, u_bar);
		AddVector3D(gradient, par.sigma, gradient, 1.0, p_dual);
		// DykstraAlgorithm(p_dual, gradient, f, par.lambda, par.L, par.nu, dykstra_max_iter);
		NablaTranspose(gradient_transpose, p_dual);
		AddArray(gradient_transpose, 1.0, u_n, -par.tau, gradient_transpose);
		TruncationOperation(u, gradient_transpose);
		AddArray(u_bar, 2.0, u, -1.0, u_n);
		// ComputeIsosurface(u_bar);
		// SetSolution(dst);
		// gray = Write(dst);
		// cvtColor(gray, color, CV_GRAY2BGR);
		// output_cap.write(color);
	}
	ComputeIsosurface(u_bar);
	SetSolution(dst);
}
#include "Algebra.h"
#include "Vector.h"
#include <cmath>

#ifndef __PROJECTION_H__
#define __PROJECTION_H__

template<class F>
class Projection
{
public:
	Projection() {}
	~Projection() {}

	void L2(Vector<F>& dst, Vector<F>& src, F bound) {
		if (!(dst.EqualProperties(src))) {
			cout << "ERROR 06 (L2): Height, Width, Level and/or Dimension do not match for used vectors" << endl;
		} else {
			F norm = src.EuclideanNorm();
			for (int i = 0; i < src.Size(); i++) {
				if (norm <= bound) {
					dst.Set(0, i, 0, 0, src.Get(0, i, 0, 0));
				} else {
					dst.Set(0, i, 0, 0, bound * (src.Get(0, i, 0, 0) / norm));
				}
			}
		}
	}

	void LMax(Vector<F>& dst, Vector<F>& src, F bound) {
		if (!(dst.EqualProperties(src))) {
			cout << "ERROR 07 (LMax): Height, Width, Level and/or Dimension do not match for used vectors" << endl;
		} else {
			for (int i = 0; i < src.Size(); i++) {
				if (fabs(src.Get(0, i, 0, 0)) <= bound) {
					dst.Set(0, i, 0, 0, src.Get(0, i, 0, 0));
				} else {
					dst.Set(0, i, 0, 0, bound);
				}
			}
		}
	}

	void TruncationOperation(Vector<F>& dst, Vector<F>& src) {
		if (!(dst.EqualProperties(src))) {
			cout << "ERROR 08 (TruncationOperation): Height, Width, Level and/or Dimension do not match for used vectors" << endl;
		} else {
			for (int i = 0; i < src.Size(); i++)
				dst.Set(0, i, 0, 0, fmin(1.0, fmax(0.0, src.Get(0, i, 0, 0))));
		}
	}

	void OnParabola(Vector<F>& dst, Vector<F>& src, F alpha) {
		if (!(dst.EqualProperties(src))) {
			cout << "ERROR 09 (OnParabola): Height, Width, Level and/or Dimension do not match for used vectors" << endl;
		} else {
			F norm, v, a, b, c, d, bound, value;
			int last_element = src.Size()-1;
			bound = src.Get(0, last_element, 0, 0);
			norm = src.EuclideanNorm(last_element);
			if (alpha * pow(norm, 2) > bound) {
				a = 2.0 * alpha * norm;
				b = 2.0 / 3.0 * (1.0 - 2.0 * alpha * bound);
				d = b < 0 ? (a - pow(sqrt(-b), 3)) * (a + pow(sqrt(-b), 3)) : a * a + b * b * b;
				c = pow((a + sqrt(d)), 1.0 / 3.0);
				if (d >= 0) {
					v = c != 0.0 ? c - b / c : 0.0;
				} else {
					v = 2.0 * sqrt(-b) * cos((1.0 / 3.0) * acos(a / (pow(sqrt(-b), 3))));
				}
				for (int i = 0; i < last_element; i++) {
					value = src.Get(0, i, 0, 0) != 0.0 ? (v / (2.0 * alpha)) * (src.Get(0, i, 0, 0) / norm) : 0.0;
					dst.Set(0, i, 0, 0, value);
				}
				norm = dst.EuclideanNorm(last_element);
				dst.Set(0, last_element, 0, 0, alpha * pow(norm, 2));
			} else {
				for (int i = 0; i < src.Size(); i++) {
					dst.Set(0, i, 0, 0, src.Get(0, i, 0, 0));
				}
			}
		}
	}

	void SoftShrinkage(Vector<F>& dst, Vector<F>& src, int i, int j, int k1, int k2, F nu) {
		int elements = 2;
		if (!(dst.EqualProperties(src)) || src.Dimension() != elements+1) {
			cout << "ERROR 10 (SoftShrinkage): Height, Width, Level and/or Dimension do not match for used vectors" << endl;
		} else {
			F K = (F)(k2 - k1 + 1);
			Vector<F> s(1, 1, 1, elements, 0.0);
			Vector<F> s0(1, 1, 1, elements, 0.0);
			for (int k = k1; k < k2; k++) {
				for (int c = 0; c < elements; c++) {
					s0.Set(0, 0, 0, c, s0.Get(0, 0, 0, c) + src.Get(i, j, k, c));
				}
			}
			L2(s, s0, nu);
			for (int k = 0; k < src.Level(); k++) {
				for (int c = 0; c < elements; c++) {
					if (k >= k1 && k < k2) {
						dst.Set(i, j, k, c, s.Get(0, 0, 0, c) / K);
						// dst.Set(i, j, k, c, src.Get(i, j, k, c) + (s.Get(0, 0, 0, c) - s0.Get(0, 0, 0, c)) / K);
					} else {
						dst.Set(i, j, k, c, src.Get(i, j, k, c));
					}
				}
			}
		}
	}

	void InitProjectionVector(Vector<F>* dst1, Vector<F>* dst2, int size, int level) {
		for (int i = 0; i < size; i++) {
			if (i < level) {
				dst1[i] = Vector<F>(1, 1, 1, 3, 0.0);
				dst2[i] = Vector<F>(1, 1, 1, 3, 0.0);
			} else {
				dst1[i] = Vector<F>(1, 1, level, 3, 0.0);
				dst2[i] = Vector<F>(1, 1, level, 3, 0.0);
			}
		}
	}

	void SetProjectionVector(Vector<F>* dst, Vector<F>& src, int size, int level, int i, int j) {
		for (int k = 0; k < level; k++) {
			for (int c = 0; c < 3; c++) {
				dst[k].Set(0, 0, 0, c, src.Get(i, j, k, c));
			}
		}
		for (int x = level; x < size; x++) {
			for (int k = 0; k < level; k++) {
				for (int c = 0; c < 3; c++) {
					dst[x].Set(0, 0, k, c, src.Get(i, j, k, c));
				}
			}
		}
	}

	void WriteProjectionVectors(Vector<F>& dst, Vector<F>* src, int size, int level, int i, int j) {
		for (int k = 0; k < level; k++) {
			for (int c = 0; c < 3; c++) {
				dst.Set(i, j, k, c, src[k].Get(0, 0, 0, c));
			}
		}
		for (int x = level; x < size; x++) {
			for (int k = 0; k < level; k++) {
				for (int c = 0; c < 3; c++) {
					dst.Set(i, j, k, c, src[x].Get(0, 0, k, c));
				}
			}
		}
	}

	void Dykstra(Vector<F>& dst, Vector<F>& src, Algebra<F>& linalg, F alpha, F nu, int i, int j, int dykstra_max_iter) {
		int level = src.Level();
		int projections = level * (level + 1) / 2 + level;
		Vector<F>* corrections = new Vector<F>[projections];
		Vector<F>* variables = new Vector<F>[projections];

		InitProjectionVector(variables, corrections, projections, level);
		SetProjectionVector(variables, src, projections, level, i, j);

		for (int m = 0; m < dykstra_max_iter; m++) {
			for (int k = 0; k < level; k++) {
				if (m == 0) {
					linalg.AddVector(corrections[k], variables[k], corrections[k], 1.0, 1.0);
				} else {
					linalg.AddVector(corrections[k], variables[level - 1], corrections[k], 1.0, 1.0);
				}
				OnParabola(variables[k], corrections[k], 1.0);
				linalg.AddVector(corrections[k], corrections[k], variables[k], 1.0, -1.0);
			}
			int l = 0;
			for (int k1 = 0; k1 < level; k1++) {
				for (int k2 = k1; k2 < level; k2++) {
					linalg.AddVector(corrections[level + l], variables[projections - 1], corrections[level + l], 1.0, 1.0);
					SoftShrinkage(variables[level + l], corrections[level + l], i, j, k1, k2, nu);
					linalg.AddVector(corrections[level + l], corrections[level + l], variables[level + l], 1.0, -1.0);
					l++;
				}
			}
		}

		WriteProjectionVectors(dst, variables, projections, level, i, j);
		
		delete[] corrections;
		delete[] variables;
	}
};

#endif //__PROJECTION_H__
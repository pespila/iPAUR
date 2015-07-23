#include "Algebra.h"
#include "Vector.h"
#include <cmath>

#ifndef __PROJECTION_H__
#define __PROJECTION_H__

template<class F>
class Projection
{
private:
	void InitProjectionVector(Vector<F>* dst1, Vector<F>* dst2, int level, int depth) {
		for (int i = 0; i < level; i++) {
			dst1[i] = Vector<F>(1, 1, depth, 3, 0.0);
			dst2[i] = Vector<F>(1, 1, depth, 3, 0.0);
		}
	}
	void SetProjectionVector(Vector<F>* dst, Vector<F>& src, int level, int i, int j) {
		for (int k = 0; k < level; k++) {
			for (int c = 0; c < 3; c++) {
				dst[k].Set(0, 0, 0, c, src.Get(i, j, k, c));
			}
		}
	}
	void WriteProjectionVectors(Vector<F>& dst, Vector<F>* src, int level, int i, int j) {
		for (int k = 0; k < level; k++) {
			for (int c = 0; c < 3; c++) {
				dst.Set(i, j, k, c, src[k].Get(0, 0, 0, c));
			}
		}
	}
	void ParabolaConstraints(Vector<F>* variables, Vector<F>* corrections, Algebra<F>& linalg, F img, F alpha, F L, F lambda, int level, int m) {
		for (int k = 0; k < level; k++) {
			if (k == 0) {
				linalg.AddVector(corrections[k], variables[k], corrections[k], 1.0, 1.0);
			} else {
				linalg.AddVector(corrections[k], variables[k - 1], corrections[k], 1.0, 1.0);
			}
			OnParabola(variables[k], corrections[k], img, alpha, L, lambda, k);
			linalg.AddVector(corrections[k], corrections[k], variables[k], 1.0, -1.0);
		}
	}
	void UpdateConstraintsFirstTurn(Vector<F>* dst, Vector<F>* src, int semilevel, int level) {
		for (int k = 0; k < level; k++) {
			for (int c = 0; c < 3; c++) {
				dst[semilevel - 1].Set(0, 0, k, c, src[k].Get(0, 0, 0, c));
			}
		}
	}
	void SumConstraints(Vector<F>* variables, Vector<F>* corrections, Algebra<F>& linalg, F nu, int semilevel, int level, int i, int j) {
		int l = 0;
		for (int k1 = 0; k1 < level; k1++) {
			for (int k2 = k1; k2 < level; k2++) {
				if (l == 0) {
					linalg.AddVector(corrections[l], variables[semilevel - 1], corrections[l], 1.0, 1.0);
				} else {
					linalg.AddVector(corrections[l], variables[l - 1], corrections[l], 1.0, 1.0);
				}
				SoftShrinkage(variables[l], corrections[l], i, j, k1, k2, nu);
				linalg.AddVector(corrections[l], corrections[l], variables[l], 1.0, -1.0);
				l++;
			}
		}
	}
	void UpdateConstraintsSecondTurn(Vector<F>* dst, Vector<F>* src, int semilevel, int level) {
		for (int k = 0; k < level; k++) {
			for (int c = 0; c < 3; c++) {
				dst[k].Set(0, 0, 0, c, src[semilevel - 1].Get(0, 0, k, c));
			}
		}
	}
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
	void OnParabola(Vector<F>& dst, Vector<F>& src, F img, F alpha, F L, F lambda, int k) {
		if (!(dst.EqualProperties(src))) {
			cout << "ERROR 09 (OnParabola): Height, Width, Level and/or Dimension do not match for used vectors" << endl;
		} else {
			F norm, v, a, b, c, d, bound, value;
			int last_element = src.Size()-1;
			// bound = p.Get(i, j, k, 2) + lambda * pow(float(k) / L - f.Get(i, j, 0), 2);
			bound = src.Get(0, 0, 0, last_element) + lambda * pow((F)(k+1) / (F)(L) - img, 2);
			// cout << src.Get(0, 0, 0, last_element) << "   " << bound << endl;
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
			for (int k = k1; k <= k2; k++) {
				for (int c = 0; c < elements; c++) {
					s0.Set(0, 0, 0, c, s0.Get(0, 0, 0, c) + src.Get(i, j, k, c));
				}
			}
			L2(s, s0, nu);
			for (int k = 0; k < src.Level(); k++) {
				for (int c = 0; c < elements; c++) {
					if (k >= k1 && k <= k2) {
						// dst.Set(i, j, k, c, s.Get(0, 0, 0, c) / K);
						dst.Set(i, j, k, c, src.Get(i, j, k, c) + (s.Get(0, 0, 0, c) - s0.Get(0, 0, 0, c)) / K);
					} else {
						dst.Set(i, j, k, c, src.Get(i, j, k, c));
					}
				}
			}
		}
	}
	void Dykstra(Vector<F>& dst, Vector<F>& src, Algebra<F>& linalg, F img, F alpha, F nu, F L, F lambda, int i, int j, int dykstra_max_iter) {
		int level = src.Level();
		int semilevel = level * (level + 1) / 2;
		Vector<F>* parabola_var = new Vector<F>[level];
		Vector<F>* parabola_corr = new Vector<F>[level];
		Vector<F>* sumconstraint_var = new Vector<F>[semilevel];
		Vector<F>* sumconstraint_corr = new Vector<F>[semilevel];

		InitProjectionVector(parabola_var, parabola_corr, level, 1);
		InitProjectionVector(sumconstraint_var, sumconstraint_corr, semilevel, level);
		SetProjectionVector(parabola_var, src, level, i, j);

		for (int m = 0; m < dykstra_max_iter; m++) {
			ParabolaConstraints(parabola_var, parabola_corr, linalg, img, alpha, L, lambda, level, m);
			UpdateConstraintsFirstTurn(sumconstraint_var, parabola_var, semilevel, level);
			SumConstraints(sumconstraint_var, sumconstraint_corr, linalg, nu, semilevel, level, i, j);
			UpdateConstraintsSecondTurn(parabola_var, sumconstraint_var, semilevel, level);
		}

		WriteProjectionVectors(dst, parabola_var, level, i, j);
		
		delete[] parabola_var;
		delete[] parabola_corr;
		delete[] sumconstraint_var;
		delete[] sumconstraint_corr;
	}
};

#endif //__PROJECTION_H__
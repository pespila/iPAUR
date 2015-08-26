#include "Algebra.h"
#include "Vector.h"
#include <cmath>

#ifndef __PROJECTION_H__
#define __PROJECTION_H__

template<class F>
class Projection
{
private:
	int level;
	int semilevel;

	primaldual::Vector<F>* parabola_var;
	primaldual::Vector<F>* parabola_corr;
	primaldual::Vector<F>* sumconstraint_var;
	primaldual::Vector<F>* sumconstraint_corr;

	void Reset() {
		for (int k = 0; k < level; k++) {
			for (int c = 0; c < 3; c++) {
				parabola_var[k].Set(0, c, 0.0);
				parabola_corr[k].Set(0, c, 0.0);
			}
		}
		for (int k = 0; k < semilevel; k++) {
			for (int c = 0; c < 3; c++) {
				sumconstraint_var[k].Set(0, c, 0.0);
				sumconstraint_corr[k].Set(0, c, 0.0);
			}
		}
	}

	void InitProjectionVector(primaldual::Vector<F>* dst1, primaldual::Vector<F>* dst2, int level, int depth) {
		for (int i = 0; i < level; i++) {
			dst1[i] = primaldual::Vector<F>(depth, 3, 0.0);
			dst2[i] = primaldual::Vector<F>(depth, 3, 0.0);
		}
	}
	void SetProjectionVector(primaldual::Vector<F>* dst, primaldual::Vector<F>& src, int level, int i, int j) {
		if (src.Dimension() == dst[0].Width()) {
			for (int k = 0; k < level; k++) {
				for (int c = 0; c < 3; c++) {
					dst[k].Set(0, c, src.Get(i, j, k, c));
				}
			}
		} else {
			cout << "ERROR 11 (SetProjectionVector): Dimension of src and width of dst do not match." << endl;
		}
	}
	void WriteProjectionVectors(primaldual::Vector<F>& dst, primaldual::Vector<F>* src, int level, int i, int j) {
		if (dst.Dimension() == src[0].Width()) {
			for (int k = 0; k < level; k++) {
				for (int c = 0; c < 3; c++) {
					dst.Set(i, j, k, c, src[k].Get(0, c));
				}
			}
		} else {
			cout << "ERROR 12 (WriteProjectionVectors): Dimension of dst and width of src do not match." << endl;
		}
	}
	void ParabolaConstraints(primaldual::Vector<F>* variables, primaldual::Vector<F>* corrections, Algebra<F>& linalg, F img, F alpha, F L, F lambda, int level, int m) {
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
	void UpdateConstraintsFirstTurn(primaldual::Vector<F>* dst, primaldual::Vector<F>* src, int level) {
		if (dst[0].Width() == src[0].Width() && dst[0].Height() == level * src[0].Height()) {
			for (int k = 0; k < semilevel; k++) {
				for (int i = 0; i < level; i++) {
					for (int c = 0; c < 3; c++) {
						dst[k].Set(i, c, src[i].Get(0, c));
					}
				}
			}
		} else {
			cout << "ERROR 13 (UpdateConstraintsFirstTurn): Dimensions and/or Level do not match." << endl;
		}
	}
	void SumConstraints(primaldual::Vector<F>* variables, primaldual::Vector<F>* corrections, Algebra<F>& linalg, F nu, int level) {
		int l = 0;
		for (int k1 = 0; k1 < level; k1++) {
			for (int k2 = k1; k2 < level; k2++) {
				if (l == 0) {
					linalg.AddVector(corrections[l], variables[l], corrections[l], 1.0, 1.0);
				} else {
					linalg.AddVector(corrections[l], variables[l - 1], corrections[l], 1.0, 1.0);
				}
				SoftShrinkage(variables[l], corrections[l], k1, k2, nu, level);
				linalg.AddVector(corrections[l], corrections[l], variables[l], 1.0, -1.0);
				l++;
			}
		}
	}
	void UpdateConstraintsSecondTurn(primaldual::Vector<F>* dst, primaldual::Vector<F>* src, int semilevel, int level) {
		if (dst[0].Width() == src[0].Width() && src[0].Height() == level * dst[0].Height()) {
			for (int k = 0; k < semilevel; k++) {
				for (int i = 0; i < level; i++) {
					for (int c = 0; c < 3; c++) {
						dst[i].Set(0, c, src[k].Get(i, c));
					}
				}
			}
		} else {
			cout << "ERROR 14 (UpdateConstraintsSecondTurn): Dimensions and/or Level do not match." << endl;
		}
	}
public:
	Projection() {}
	Projection(int level) {
		this->level = level;
		this->semilevel = level * (level + 1) / 2;
		this->parabola_var = new primaldual::Vector<F>[level];
		this->parabola_corr = new primaldual::Vector<F>[level];
		this->sumconstraint_var = new primaldual::Vector<F>[semilevel];
		this->sumconstraint_corr = new primaldual::Vector<F>[semilevel];
		InitProjectionVector(parabola_var, parabola_corr, level, 1);
		InitProjectionVector(sumconstraint_var, sumconstraint_corr, semilevel, level);
	}
	~Projection() {
		delete[] this->parabola_var;
		delete[] this->parabola_corr;
		delete[] this->sumconstraint_var;
		delete[] this->sumconstraint_corr;
	}

	void L2(primaldual::Vector<F>& dst, primaldual::Vector<F>& src, F bound) {
		if (!(dst.EqualProperties(src))) {
			cout << "ERROR 06 (L2): Height, Width, Level and/or Dimension do not match for used vectors." << endl;
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
	void LMax(primaldual::Vector<F>& dst, primaldual::Vector<F>& src, F bound) {
		if (!(dst.EqualProperties(src))) {
			cout << "ERROR 07 (LMax): Height, Width, Level and/or Dimension do not match for used vectors." << endl;
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
	void TruncationOperation(primaldual::Vector<F>& dst, primaldual::Vector<F>& src, int i, int j, int k) {
		if (!(dst.EqualProperties(src))) {
			cout << "ERROR 08 (TruncationOperation): Height, Width, Level and/or Dimension do not match for used vectors." << endl;
		} else {
			if (k == 0) {
				dst.Set(i, j, k, 0, 1.0);
			} else if (k == level - 1) {
				dst.Set(i, j, k, 0, 0.0);
			} else {
				dst.Set(i, j, k, 0, fmin(1.0, fmax(0.0, src.Get(i, j, k, 0))));
			}
		}
	}
	void OnParabola(primaldual::Vector<F>& dst, primaldual::Vector<F>& src, F img, F alpha, F L, F lambda, int k) {
		if (!(dst.EqualProperties(src))) {
			cout << "ERROR 09 (OnParabola): Height, Width, Level and/or Dimension do not match for used vectors" << endl;
		} else {
			F norm, v, a, b, c, d, bound, value;
			int last_element = src.Size()-1;
			bound = src.Get(0, last_element) + lambda * pow((F)(k) / (F)(L) - img, 2);
			norm = src.EuclideanNorm(last_element);
			// cout << bound << " " << lambda * pow((F)(k) / (F)(L) - img, 2) << endl;
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
					value = src.Get(0, i) != 0.0 ? (v / (2.0 * alpha)) * (src.Get(0, i) / norm) : 0.0;
					dst.Set(0, i, value);
				}
				norm = dst.EuclideanNorm(last_element);
				dst.Set(0, last_element, alpha * pow(norm, 2) - lambda * pow((F)(k) / (F)(L) - img, 2));
			} else {
				for (int i = 0; i < src.Size(); i++) {
					dst.Set(0, i, src.Get(0, i));
				}
			}
		}
	}
	void SoftShrinkage(primaldual::Vector<F>& dst, primaldual::Vector<F>& src, int k1, int k2, F nu, int level) {
		int elements = 2;
		if (!(dst.EqualProperties(src)) || src.Width() != elements+1) {
			cout << "ERROR 10 (SoftShrinkage): Height, Width, Level and/or Dimension do not match for used vectors/" << endl;
		} else {
			F K = (F)(k2 - k1 + 1);
			primaldual::Vector<F> s(1, elements, 0.0);
			primaldual::Vector<F> s0(1, elements, 0.0);
			for (int k = k1; k <= k2; k++) {
				for (int c = 0; c < elements; c++) {
					s0.Set(0, c, s0.Get(0, c) + src.Get(k, c));
				}
			}
			L2(s, s0, nu);
			for (int k = 0; k < level; k++) {
				for (int c = 0; c < elements; c++) {
					if (k >= k1 && k <= k2) {
						dst.Set(k, c, src.Get(k, c) + (s.Get(0, c) - s0.Get(0, c)) / K);
					} else {
						dst.Set(k, c, src.Get(k, c));
					}
				}
			}
		}
	}
	int IsConverged(primaldual::Vector<F>* xk1, primaldual::Vector<F>* xk0) {
		F value;
		int count = 0;
		for (int k = 0; k < level; k++) {
			value = 0.0;
			for (int i = 0; i < 3; i++) {
				value += pow(xk1[k].Get(0, i) - xk0[k].Get(0, i), 2);
			}
			value = sqrt(value);
			if (value < pow(10, -6)) {
				count++;
			} else {
				break;
			}
		}
		if (count == level) {
			return 1;
		} else {
			return 0;
		}
	}
	void Dykstra(primaldual::Vector<F>& dst, primaldual::Vector<F>& src, Algebra<F>& linalg, F img, F alpha, F nu, F L, F lambda, int i, int j, int level, int dykstra_max_iter) {
		Reset();
		int conv = 0;
		primaldual::Vector<F>* current = new primaldual::Vector<F>[level];
		for (int k = 0; k < level; k++) {
			current[k] = primaldual::Vector<F>(1, 3, 0.0);
		}
		SetProjectionVector(parabola_var, src, level, i, j);

		for (int m = 0; m < dykstra_max_iter; m++) {
			for (int k = 0; k < level; k++) {
				for (int l = 0; l < 3; l++) {
					current[k].Set(0, l, parabola_var[k].Get(0, l));
				}
			}
			ParabolaConstraints(parabola_var, parabola_corr, linalg, img, alpha, L, lambda, level, m);
			UpdateConstraintsFirstTurn(sumconstraint_var, parabola_var, level);
			SumConstraints(sumconstraint_var, sumconstraint_corr, linalg, nu, level);
			UpdateConstraintsSecondTurn(parabola_var, sumconstraint_var, semilevel, level);
			conv = IsConverged(parabola_var, current);
			if (conv) {
				// cout << m << endl;
				break;
			} else {
				continue;
			}
		}

		WriteProjectionVectors(dst, parabola_var, level, i, j);
	}
};

#endif //__PROJECTION_H__
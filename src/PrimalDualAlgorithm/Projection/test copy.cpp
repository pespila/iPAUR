#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>

using namespace std;

class Point
{
public:
	double x;
	double y;

	void Set(double, double);
	void Print() {
		printf("Your Point is: (%f, %f).\n", this->x, this->y);
	}

	Point():x(0), y(0) {}
	Point(double x, double y) {
		this->x = x;
		this->y = y;
	}
	~Point() {}
};

template<class T>
class Vector
{
private:
	vector<T> v;

public:
	Vector() {v.resize(3, (T)0);}
	Vector(int size) {v.resize(size, (T)0);}
	~Vector() {v.clear();}

	int Size() {return v.size();}
	T Get(int i) {return v[i];}
	void Set(int i, T value) {v[i] = value;}
	void Print() {
		for (int i = 0; i < v.size(); i++)
			cout << v[i] << " ";
		cout << endl;
	}
};

void Add(Point& out, double fac1, Point& in1, double fac2, Point& in2) {
	out.x = fac1 * in1.x + fac2 * in2.x;
	out.y = fac1 * in1.y + fac2 * in2.y;
}

void L2Projection(Point& out, Point& in, double shiftx, double shifty, double radius) {
	double norm = sqrt(in.x * in.x + in.y * in.y);
	if (norm <= radius) {
		out.x = in.x + shiftx;
		out.y = in.y + shifty;
	} else {
		out.x = radius * in.x / norm + shiftx;
		out.y = radius * in.y / norm + shifty;
	}
}

void LInfProjection(Point& out, Point& in, double shiftx, double shifty, double radius) {
	if (fabs(in.x) <= radius) {
		out.x = in.x + shiftx;
	} else {
		out.x = radius + shiftx;
	}
	if (fabs(in.y) <= radius) {
		out.y = in.y + shifty;
	} else {
		out.y = radius + shifty;
	}
}

double EuclideanDistance(Point& x, Point& y) {
	return sqrt(pow(x.x-y.x, 2) + pow(x.y-y.y, 2));
}

void ProjectionOntoParabola(Point& out, Point& in, double alpha, double shifty) {
	double bound = in.y - shifty;
	double a, b, c, d, v, l2norm;
	l2norm = sqrt(pow(in.x, 2));
	if (bound <  alpha * pow(l2norm, 2)) {
		a = 2.0 * alpha * l2norm;
		b = 2.0 / 3.0 * (1.0 - 2.0 * alpha * bound);
		d = b < 0 ? (a - pow(sqrt(-b), 3)) * (a + pow(sqrt(-b), 3)) : a * a + b * b * b;
		c = pow((a + sqrt(d)), 1.0 / 3.0);
		if (d >= 0) {
			v = c != 0.0 ? c - b / c : 0.0;
		} else {
			v = 2.0 * sqrt(-b) * cos((1.0 / 3.0) * acos(a / (pow(sqrt(-b), 3))));
		}
		in.x = in.x != 0.0 ? (v / (2.0 * alpha)) * (in.x / l2norm) : 0.0;
		in.y = alpha * pow(in.x, 2) + shifty;
	}
	out.x = in.x;
	out.y = in.y;
}

void Dykstra(Point& r) {
	Point y, p, q, z, x(r.x, r.y);
	int max = 2;
	Point* var = new Point[max];
	Point* cor = new Point[max];
	int i, k;
	for (i = 0; i < 100; i++) {
		for (k = 0; k < max; k++) {
			if (k == 0) {
				Add(cor[k], 1.0, x, 1.0, cor[k]);
			} else {
				Add(cor[k], 1.0, var[k-1], 1.0, cor[k]);
			}
			L2Projection(var[k], cor[k], k, 0.0, 1.0);
			Add(cor[k], 1.0, cor[k], -1.0, var[k]);
		}
		r.x = var[max-1].x;
		r.y = var[max-1].y;
	}
	delete[] var;
	delete[] cor;
}

// void Dykstra(Point& r) {
// 	Point y, p, q, z, x(r.x, r.y);
// 	int i, k;
// 	for (i = 0; i < 100; i++) {
// 		Add(p, 1.0, x, 1.0, p);
// 		L2Projection(y, p, 0.0, 0.0, 1.0);
// 		Add(p, 1.0, p, -1.0, y);
// 		Add(q, 1.0, y, 1.0, q);
// 		L2Projection(x, q, 1.0, 0.0, 1.0);
// 		Add(q, 1.0, q, -1.0, x);
// 		r.x = x.x;
// 		r.y = x.y;
// 	}
// }

template<class F>
F L2Norm(Vector<F>& x, int size) {
	F norm = 0.0;
	for (int i = 0; i < size; i++)
		norm += pow(x.Get(i), 2);
	return sqrt(norm);
}

template<class F>
void ParabolaProjection(Vector<F>& out, Vector<F>& in, F alpha, F f, F lambda, F L, F k) {
	F bound = in.Get(2) + lambda * pow((k / L - f), 2);
	F a, b, c, d, v, norm = L2Norm(in, in.Size()-1);
	if (bound < alpha * pow(norm, 2)) {
		a = 2.0 * alpha * norm;
		b = 2.0 / 3.0 * (1.0 - 2.0 * alpha * bound);
		d = b < 0 ? (a - pow(sqrt(-b), 3)) * (a + pow(sqrt(-b), 3)) : a * a + b * b * b;
		c = pow((a + sqrt(d)), 1.0 / 3.0);
		if (d >= 0) {
			v = c != 0.0 ? c - b / c : 0.0;
		} else {
			v = 2.0 * sqrt(-b) * cos((1.0 / 3.0) * acos(a / (pow(sqrt(-b), 3))));
		}
		for (int i = 0; i < in.Size() - 1; i++) {
			in.Set(i, in.Get(i) != 0 ? (v / (2.0 * alpha)) * (in.Get(i) / norm) : 0.0);
		}
		norm = L2Norm(in, in.Size() - 1);
		in.Set(in.Size() - 1, alpha * pow(norm, 2) - lambda * pow((k / L - f), 2));
	}
	for (int i = 0; i < in.Size(); i++)
		out.Set(i, in.Get(i));
}

template<class F>
void L2Projection(Vector<F>& out, Vector<F>& in, F radius) {
	F norm = L2Norm(in, in.Size());
	for (int i = 0; i < in.Size(); i++) {
		if (norm <= radius) {
			out.Set(i, in.Get(i));
		} else {
			out.Set(i, radius * (in.Get(i) / norm));
		}
	}
}

template<class F>
void SoftShrinkageScheme(Vector<F>* out, Vector<F>* in, F nu, int k1, int k2, int size) {
	F K = (F)(k2 - k1 + 1);
	Vector<F> s(2), s0(2);
	for (int i = k1; i < k2; i++) {
		for (int j = 0; j < 2; j++) {
			s0.Set(j, s0.Get(j) + in[i].Get(j));
		}
	}
	L2Projection(s, s0, nu);
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < 2; j++) {
			if (i >= k1 && i < k2) {
				out[i].Set(j, s.Get(j) / K);
				// out[i].Set(j, in[i].Get(j) + (s.Get(j) - s0.Get(j)) / K);
			} else {
				out[i].Set(j, in[i].Get(j));
			}
		}
	}
}

template<class F>
void Add(Vector& out, F fac1, Vector& in1, F fac2, Vector& in2) {
	for (int i = 0; i < in1.Size(); i++)
		out.Set(i, fac1 * in1.Get(i) + fac2 * in2.Get(i));
}

template<class F>
void Dykstra(Vector<F>* out, Vector<F>* in, int size, int max_iter) {
	Vector<F>* variable = new Vector<F>[in.Size()];
	Vector<F>* correction = new Vector<F>[in.Size()];
	for (int l = 0; l < max_iter; l++)
	{
		for (int m = 0; m < in.Size(); m++)
		{
			Add(correction[m], 1.0, in[m], 1.0, correction[m]);
			ProjectionOntoParabola(variable[m], correction[m], 0.25, 1.0, 0.01, sqrt(12), m);
			Add(correction[m], 1.0, correction[m], -1.0, variable[m]);

		}
		Add(correction[])
	}
	Vector<F>* variable = new Vector<F>[size * size + 1];
	Vector<F>* correction = new Vector<F>[size * size + 1];
	int m;
	for (int k = 0; k < max_iter; k++)
	{
		for (int i = 0; i < size; i++)
		{
			for (int j = 0; j < size; j++)
			{
				m = i + j;
				if (m == 0) {
					Add(correction[m], 1.0, in[m], 1.0, correction[m]);
				}
				
				SoftShrinkageScheme(variable[m], correction[m])
			}
		}
	}
	Point y, p, q, z, x(r.x, r.y);
	int max = 2;
	Point* var = new Point[max];
	Point* cor = new Point[max];
	int i, k;
	for (i = 0; i < 100; i++) {
		for (k = 0; k < max; k++) {
			if (k == 0) {
				Add(cor[k], 1.0, x, 1.0, cor[k]);
			} else {
				Add(cor[k], 1.0, var[k-1], 1.0, cor[k]);
			}
			L2Projection(var[k], cor[k], k, 0.0, 1.0);
			Add(cor[k], 1.0, cor[k], -1.0, var[k]);
		}
		r.x = var[max-1].x;
		r.y = var[max-1].y;
	}
	delete[] var;
	delete[] cor;
}

int main(void) {
	int size = 2;
	Vector<double>* Tvec = new Vector<double>[size];

	for (int i = 0; i < size; i++)
		for (int j = 0; j < Tvec[i].Size(); j++)
			Tvec[i].Set(j, rand()%10 / 10.0);

	for (int i = 0; i < size; i++)
		Tvec[i].Print();

	cout << endl;
	

	SoftShrinkageScheme(Tvec, Tvec, 5.0, 1000, size, size);

	for (int i = 0; i < size; i++)
		Prototype(Tvec[i], Tvec[i], 0.25, 0.5, 0.01, sqrt(12), 4.0);

	for (int i = 0; i < size; i++)
		Tvec[i].Print();

	delete[] Tvec;
	return 0;
}
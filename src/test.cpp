#include <iostream>
#include <cmath>

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

class Array
{
public:
	Array():size(0), p(NULL) {}
	Array(int size) {
		this->size = size;
		this->p = new Point[size];
		for (int i = 0; i < size; i++) {
			p[i].x = 24;
			p[i].y = 42;
		}
	}
	~Array() {}

	void Free() {
		delete[] p;
	}

	int size;
	Point* p;
};

class Printer
{
public:
	Printer() {}
	Printer(int size) {
		this->A = Array(size);
	}
	~Printer() {A.Free();}

	Array A;

	void DoIt() {
		for (int i = 0; i < A.size; i++)
			printf("(%f, %f)\n", A.p->x, A.p->y);
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

int main(void) {
	// Point r(5.0, 4.0), x;
	// r.Print();
	// // ProjectionOntoParabola(x, r, -1.0, 0.0);
	// Dykstra(r);
	// // x.Print();
	// r.Print();
	Printer iPrint(10);
	iPrint.DoIt();
	return 0;
}
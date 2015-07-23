#include <iostream>
#include <vector>
#include "Image.h"
#include "Vector.h"
#include "LinearOperator.h"
#include "Algebra.h"
#include "Projection.h"

using namespace std;

int main(int argc, char* argv[]) {
	float L = sqrt(12);
	float sigma = 1.0 / L;
	float tau = sigma;
	float alpha = 1.0;
	float lambda = 0.01;
	float nu = 5.0;

	LinearOperator<float> apply;
	Algebra<float> linalg;
	Projection<float> proj;

	// Image<float> img(5, 5, 0.0);
	// img.Set(1, 2, 1.0);
	// img.Set(2, 2, 1.0);
	// img.Set(3, 2, 1.0);
	// img.Set(2, 1, 1.0);
	// img.Set(2, 3, 1.0);

	Image<float> img(3, 3, 0.0);
	img.Set(1, 1, 1.0);
	img.Print();
	cout << endl;

	// Vector<float> x(5, 5, 8, 1, 1.0);
	// Vector<float> xcur(5, 5, 8, 1, 1.0);
	// Vector<float> xbar(5, 5, 8, 1, 1.0);
	// Vector<float> gradient(5, 5, 8, 3, 1.0);
	// Vector<float> y(5, 5, 8, 3, 1.0);

	Vector<float> x(3, 3, 16, 1, 1.0);
	Vector<float> grad_T(3, 3, 16, 1, 1.0);
	Vector<float> xcur(3, 3, 16, 1, 1.0);
	Vector<float> xbar(3, 3, 16, 1, 1.0);
	Vector<float> gradient(3, 3, 16, 3, 1.0);
	Vector<float> y(3, 3, 16, 3, 1.0);

	linalg.Image2Vector(x, img);
	linalg.ScaleVector(xbar, x, 1.0);

	for (int X = 0; X < 100; X++) {
		cout << "Step: " << X << endl;
		linalg.ScaleVector(xcur, x, 1.0);

		for (int k = 0; k < gradient.Level(); k++) {
			for (int i = 0; i < gradient.Height(); i++) {
				for (int j = 0; j < gradient.Width(); j++) {
					apply.Nabla(gradient, xbar, i, j, k);
				}
			}
		}

		linalg.AddVector(gradient, gradient, y, sigma, 1.0);

		for (int i = 0; i < y.Height(); i++) {
			for (int j = 0; j < y.Width(); j++) {
				proj.Dykstra(y, gradient, linalg, img.Get(i, j), alpha, nu, L, lambda, i, j, 100);
			}
		}

		for (int k = 0; k < gradient.Level(); k++) {
			for (int i = 0; i < gradient.Height(); i++) {
				for (int j = 0; j < gradient.Width(); j++) {
					apply.NablaTranspose(grad_T, y, i, j, k);
				}
			}
		}

		linalg.AddVector(grad_T, xcur, grad_T, 1.0, -tau);

		proj.TruncationOperation(x, grad_T);

		linalg.AddVector(xbar, x, xcur, 2.0, -1.0);

	}

	apply.Isosurface(img, xbar);

	img.Print();

	// src.Print();
	// proj.Dykstra(dst, src, linalg, 1.0, 5.0, 1);
	// y.Print();

	return 0;
}
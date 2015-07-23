#include <iostream>
#include <vector>
#include "Image.h"
#include "Vector.h"
#include "LinearOperator.h"
#include "Algebra.h"
#include "Projection.h"

using namespace std;

int main(int argc, char* argv[]) {
	float sigma = 1.0 / sqrt(12);
	Image<float> img(5, 5, 0.0);
	img.Set(1, 2, 1.0);
	img.Set(2, 2, 1.0);
	img.Set(3, 2, 1.0);
	img.Set(2, 1, 1.0);
	img.Set(2, 3, 1.0);

	Vector<float> x(5, 5, 3, 1, 0.0);
	Vector<float> xbar(5, 5, 3, 1, 0.0);
	Vector<float> ybar(5, 5, 3, 3, 0.0);
	Vector<float> y(5, 5, 3, 3, 0.0);

	LinearOperator<float> apply;
	Algebra<float> linalg;
	Projection<float> proj;

	linalg.Image2Vector(x, img);
	linalg.ScaleVector(xbar, x, 1.0);

	for (int k = 0; k < ybar.Level(); k++) {
		for (int i = 0; i < ybar.Height(); i++) {
			for (int j = 0; j < ybar.Width(); j++) {
				apply.Nabla(ybar, xbar, i, j, k);
			}
		}
	}
	linalg.ScaleVector(ybar, ybar, sigma);
	linalg.AddVector(y, y, ybar, 1.0, 1.0);

	y.Print();

	// for (int i = 0; i < y.Height(); i++) {
	// 	for (int j = 0; j < y.Width(); j++) {
			proj.Dykstra(y, y, linalg, 1.0, 5.0, 2, 2, 1);
	// 	}
	// }

	y.Print();
	// src.Print();
	// proj.Dykstra(dst, src, linalg, 1.0, 5.0, 1);
	// y.Print();

	return 0;
}
#include <iostream>
#include <vector>
#include "Image.h"
#include "Vector.h"
#include "LinearOperator.h"
#include "Algebra.h"
#include "Projection.h"

using namespace std;

int main(int argc, char* argv[]) {
	Image<float> img(3, 3, 0.0);
	img.Set(1, 1, 1.0);
	Vector<float> src(3, 3, 4, 3, 5.0);
	Vector<float> dst(2, 2, 4, 3, 0.0);
	LinearOperator<float> apply;
	Algebra<float> linalg;
	Projection<float> proj;

	// src.Print();
	// proj.Dykstra(dst, src, linalg, 1.0, 5.0, 1);
	// dst.Print();

	img.Print();

	return 0;
}
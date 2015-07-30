#include <iostream>
#include <vector>
#include "Image.h"
#include "Vector.h"
#include "LinearOperator.h"
#include "Algebra.h"
#include "Projection.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
	float L = sqrt(12);
	float sigma = 1.0 / L;
	float tau = sigma;
	float alpha = 0.25;
	float lambda = 0.01;
	float nu = 5.0;

	// Image<float> img(3, 3, 0.0);
	// img.Set(1, 1, 1.0);
	// img.Print();
	// cout << endl;

	// int height = img.Height();
	// int width = img.Width();
	// int level = 32;

	LinearOperator<float> apply;
	Algebra<float> linalg;
	Projection<float> proj;
	// Projection<float> proj(level);

	Image<float> img(5, 5, 0.0);
	img.Set(1, 2, 255.0);
	img.Set(2, 2, 255.0);
	img.Set(3, 2, 255.0);
	img.Set(2, 1, 255.0);
	img.Set(2, 3, 255.0);
	img.Print();
	cout << endl;

	img.Write("./imageold.jpg");

	// Image<float> img;
	// img.Read(argv[1]);

	int height = img.Height();
	int width = img.Width();
	int level = 32;

	for (int i = 0; i < img.Height(); i++)
	{
		for (int j = 0; j < img.Width(); j++)
		{
			img.Set(i, j, img.Get(i, j) / 255.0);
		}
	}

	primaldual::Vector<float> x(height, width, level, 1, 0.0);
	primaldual::Vector<float> grad_T(height, width, level, 1, 0.0);
	primaldual::Vector<float> xcur(height, width, level, 1, 0.0);
	primaldual::Vector<float> xbar(height, width, level, 1, 0.0);
	primaldual::Vector<float> gradient(height, width, level, 3, 0.0);
	primaldual::Vector<float> y(height, width, level, 3, 0.0);

	// primaldual::Vector<float> x(3, 3, 64, 1, 0.0);
	// primaldual::Vector<float> grad_T(3, 3, 64, 1, 0.0);
	// primaldual::Vector<float> xcur(3, 3, 64, 1, 0.0);
	// primaldual::Vector<float> xbar(3, 3, 64, 1, 0.0);
	// primaldual::Vector<float> gradient(3, 3, 64, 3, 0.0);
	// primaldual::Vector<float> y(3, 3, 64, 3, 0.0);

	linalg.Image2Vector(x, img);
	linalg.ScaleVector(xbar, x, 1.0);

	for (int X = 0; X < 100; X++) {
		// if (X%10 == 0) {
		// 	cout << "Step: " << X << endl;
		// }
		// cout << "Step: " << X << endl;
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
				proj.Dykstra(y, gradient, linalg, img.Get(i, j), alpha, nu, L, lambda, i, j, 5);
			}
		}

		for (int k = 0; k < gradient.Level(); k++) {
			for (int i = 0; i < gradient.Height(); i++) {
				for (int j = 0; j < gradient.Width(); j++) {
					apply.NablaTranspose(grad_T, y, i, j, k);
				}
			}
		}

		linalg.AddVector(grad_T, xcur, grad_T, 1.0, tau);

		proj.TruncationOperation(x, grad_T);

		linalg.AddVector(xbar, x, xcur, 2.0, -1.0);

	}

	apply.Isosurface(img, xbar);

	for (int i = 0; i < img.Height(); i++)
	{
		for (int j = 0; j < img.Width(); j++)
		{
			img.Set(i, j, img.Get(i, j) * 255.0);
		}
	}

	img.Write("./imagenew.jpg");
	img.Print();

	return 0;
}
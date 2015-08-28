#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include "Image.h"
#include "Vector.h"
#include "LinearOperator.h"
#include "Algebra.h"
#include "Projection.h"

using namespace std;
using namespace cv;

template<class F>
void PrimalDual(Image<F>& dst, Image<F>& img, int dykstra_iter, int max_iter, int level) {
	int height = img.Height();
	int width = img.Width();
	
	F L = sqrt(12);
	F tau = 1 / L;
	F sigma = 1 / L;
	F lambda = 0.1;
	F nu = 5.0;
	F dist = 0.0;

	LinearOperator<F> apply;
	Algebra<F> linalg;
	Projection<F> proj(level);

	for (int i = 0; i < img.Height(); i++) {
		for (int j = 0; j < img.Width(); j++) {
			dst.Set(i, j, img.Get(i, j) / 255.0);
		}
	}


	primaldual::Vector<F> x(height, width, level, 1, 0.0);
	primaldual::Vector<F> grad_T(height, width, level, 1, 0.0);
	primaldual::Vector<F> xcur(height, width, level, 1, 0.0);
	primaldual::Vector<F> xbar(height, width, level, 1, 0.0);
	primaldual::Vector<F> gradient(height, width, level, 3, 0.0);
	primaldual::Vector<F> y(height, width, level, 3, 0.0);

	linalg.Image2Vector(x, dst);
	linalg.ScaleVector(xbar, x, 1.0);


	int X;
	for (X = 0; X < max_iter; X++) {
		apply.Isosurface(dst, xbar);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				dst.Set(i, j, dst.Get(i, j) * 255.0);
			}
		}

		string title = "frame";
		string file = ".png";
		string result;
		stringstream sstm;
		if (X < 10) {
			sstm << "./steps/" << "0" << X << title << file;
		} else {
			sstm << "./steps/" << X << title << file;
		}
		result = sstm.str();
		dst.Write(result);

		cout << "PD-Step: " << X+1 << endl;

		linalg.ScaleVector(xcur, x, 1.0);

		for (int k = 0; k < gradient.Level(); k++) {
			for (int i = 0; i < gradient.Height(); i++) {
				for (int j = 0; j < gradient.Width(); j++) {
					apply.Nabla(gradient, xbar, i, j, k);
				}
			}
		}

		linalg.AddVector(gradient, y, gradient, 1.0, sigma);

		for (int k = 0; k < y.Level(); k++) {
			for (int i = 0; i < y.Height(); i++) {
				for (int j = 0; j < y.Width(); j++) {
					proj.Dykstra(y, gradient, img, linalg, nu, L, lambda, i, j, k, dykstra_iter);
				}
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
		
		for (int k = 0; k < gradient.Level(); k++) {
			for (int i = 0; i < gradient.Height(); i++) {
				for (int j = 0; j < gradient.Width(); j++) {
					proj.TruncationOperation(x, grad_T, i, j, k);
				}
			}
		}

		linalg.AddVector(xbar, x, xcur, 2.0, -1.0);

		linalg.AddVector(xcur, xcur, xbar, 1.0, -1.0);
		dist = xcur.EuclideanNorm();
		cout << "Stop criterion: " << dist << endl;
		if (dist < pow(10, -6) && X >= 50) {
			break;
		}

	}

	cout << "Steps: " << X << endl;

	apply.Isosurface(dst, x);

	for (int i = 0; i < img.Height(); i++) {
		for (int j = 0; j < img.Width(); j++) {
			dst.Set(i, j, (int)(dst.Get(i, j) * 255.0));
		}
	}
}

int main(int argc, char* argv[]) {
	Image<double> img;
	// img.Read("./ball.png");
	img.Read("./frau.png");
	// img.Print();
	
	PrimalDual(img, img, 10, 1, 8);
	// PrimalDual(img, 10, 100, 32);
	
	// img.Write("./ball_new.png");
	img.Write("./frau_new.png");
	// img.Print();
	
	return 0;
}
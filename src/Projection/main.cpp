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
void PrimalDual(Image<F>& img, int dykstra_iter, int max_iter, int level) {
	int height = img.Height();
	int width = img.Width();

	Image<F> dst(height, width, 0.0);
	
	double L = sqrt(12);
	double tau = 1000;
	// double tau = sigma;
	double sigma = 1.0 / (L*L*tau);
	double alpha = 0.25;
	double lambda = 0.1;
	double nu = 5.0;

	LinearOperator<double> apply;
	Algebra<double> linalg;
	Projection<double> proj(level);

	for (int i = 0; i < img.Height(); i++) {
		for (int j = 0; j < img.Width(); j++) {
			img.Set(i, j, img.Get(i, j) / 255.0);
		}
	}


	primaldual::Vector<double> x(height, width, level, 1, 0.0);
	primaldual::Vector<double> grad_T(height, width, level, 1, 0.0);
	primaldual::Vector<double> xcur(height, width, level, 1, 0.0);
	primaldual::Vector<double> xbar(height, width, level, 1, 0.0);
	primaldual::Vector<double> gradient(height, width, level, 3, 0.0);
	primaldual::Vector<double> y(height, width, level, 3, 0.0);

	linalg.Image2Vector(x, img);
	linalg.ScaleVector(xbar, x, 1.0);

	for (int k = 0; k < gradient.Level(); k++) {
		for (int i = 0; i < gradient.Height(); i++) {
			for (int j = 0; j < gradient.Width(); j++) {
				proj.TruncationOperation(x, x, i, j, k);
			}
		}
	}

	for (int X = 0; X < max_iter; X++) {

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

		cout << "PD-Step: " << X << endl;

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
				proj.Dykstra(y, gradient, linalg, img.Get(i, j), alpha, nu, L, lambda, i, j, level, dykstra_iter);
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
		
		for (int k = 0; k < gradient.Level(); k++) {
			for (int i = 0; i < gradient.Height(); i++) {
				for (int j = 0; j < gradient.Width(); j++) {
					proj.TruncationOperation(x, grad_T, i, j, k);
				}
			}
		}

		linalg.AddVector(xbar, x, xcur, 2.0, -1.0);

	}

	apply.Isosurface(img, xbar);

	for (int i = 0; i < img.Height(); i++) {
		for (int j = 0; j < img.Width(); j++) {
			img.Set(i, j, img.Get(i, j) * 255.0);
		}
	}
}

int main(int argc, char* argv[]) {
	Image<double> img;
	// img.Read("./ball_test.png");
	img.Read("./synth_noise.png");
	
	PrimalDual(img, 10, 100, 32);
	
	img.Write("./synth_new.png");
	
	return 0;
}
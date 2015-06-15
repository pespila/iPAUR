#include <stdlib.h>
#include <stdio.h>
#include <math.h>

double v(double);
double _f(double, double(*f)(double));
double __f(double, double(*f)(double));
double newton(double, double(*h)(double));
double newton_parabola(double, double, double, double(*f)(double));

int main(int argc, const char* argv[]) {
	double solved = newton_parabola(5, 4, 1, (&v));
	printf("(%f, %f)\n", solved, v(solved));
	return 0;
}

double v(double x) {return 0.5*x*x;}

double _f(double x, double(*f)(double)) {
	double h = 1e-9;
	return (f(x+h)-f(x))/h;
}

double __f(double x, double(*f)(double)) {
	double h = 1e-3;
	return (f(x+h) - 2.0*f(x) + f(x-h)) / (h*h);
}

double newton_parabola(double x0, double y1, double y2, double(*f)(double)) {
	double xk = x0;
	double num = 0.0;
	double denom = 1.0;
	double norm = 1.0;
	double xk_minus_eins = 0.0;
	int steps = 0;
	while (norm > 1E-10) {
	// for (int i = 0; i < 100; i++) {
		xk_minus_eins = xk;
		num = xk - y1 + _f(xk, f) * (f(xk) - y2);
		denom = 1.0 + _f(xk, f) * _f(xk, f) + __f(xk, f) * (f(xk) - y2);
		xk -= denom != 0 ? num/denom : 0.0;
		norm = fabs(xk - xk_minus_eins);
		steps++;
	}
	printf("%d\n", steps);
	printf("%g\n", norm);
	return xk;
}

double newton(double x0, double(*f)(double)) {
	double xk = x0;
	for (int i = 0; i < 20; i++)
		xk -= f(xk)/_f(xk, f);
	return xk;
}
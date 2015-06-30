#include <stdlib.h>
#include <stdio.h>
#include <math.h>

struct vector_3d
{
	double x1;
	double x2;
	double x3;
};

struct matrix_2_times_2
{
	double a11;
	double a12;
	double a21;
	double a22;
};

typedef struct vector_3d Vector;
typedef struct matrix_2_times_2 Matrix;

Vector* init_vector_3d(double, double, double);

double v(double, double);
double partial_x1_f(double, double, double(*f)(double, double));
double partial_x2_f(double, double, double(*f)(double, double));
double partial_x1_x1_f(double, double, double(*f)(double, double));
double partial_x2_x2_f(double, double, double(*f)(double, double));
double partial_x1_x2_f(double, double, double(*f)(double, double));

void newton_parabola(Vector*, Vector*, double(*f)(double, double));

int main(int argc, const char* argv[]) {
	Vector* xk = init_vector_3d(0.0, 0.0, v(0.0, 0.0));
	Vector* point = init_vector_3d(9.0, 7.0, 14.0);
	
	newton_parabola(xk, point, (&v));
	printf("(%f, %f, %f)\n", xk->x1, xk->x2, xk->x3);

	free(xk);
	free(point);
	return 0;
}

Vector* init_vector_3d(double x1, double x2, double x3) {
	Vector* out = (Vector*)malloc(sizeof(Vector));
	out->x1 = x1;
	out->x2 = x2;
	out->x3 = x3;
	return out;
}

double v(double x1, double x2) {return x1*x1+x2*x2;}

double partial_x1_f(double x1, double x2, double(*f)(double, double)) {
	double h = 1e-9;
	return (f(x1+h, x2) - f(x1, x2)) / h;
}

double partial_x2_f(double x1, double x2, double(*f)(double, double)) {
	double h = 1e-9;
	return (f(x1, x2+h) - f(x1, x2)) / h;
}

double partial_x1_x1_f(double x1, double x2, double(*f)(double, double)) {
	double h = 1e-6;
	return (f(x1+h, x2) - 2.0 * f(x1, x2) + f(x1-h, x2)) / (h*h);
}

double partial_x2_x2_f(double x1, double x2, double(*f)(double, double)) {
	double h = 1e-6;
	return (f(x1, x2+h) - 2.0 * f(x1, x2) + f(x1, x2-h)) / (h*h);
}

double partial_x1_x2_f(double x1, double x2, double(*f)(double, double)) {
	double h = 1e-6;
	return (f(x1+h, x2+h) + f(x1, x2) - f(x1+h, x2) - f(x1, x2+h)) / (h*h);
}

void newton_parabola(Vector* xk, Vector* point, double(*f)(double, double)) {
	Vector* g_x = init_vector_3d(0.0, 0.0, 0.0);
	Vector* delta_x = init_vector_3d(0.0, 0.0, 0.0);
	Matrix jacobi = {0.0, 0.0, 0.0, 0.0};
	double schwarz = 0.0;
	double first_line = 0.0;
	double second_line = 0.0;
	double norm = 1.0;
	double x_k1_minus_eins = 0.0;
	double x_k2_minus_eins = 0.0;
	int steps = 0;


	while (norm > 1E-6) {
	// for (int i = 0; i < 10; i++) {
		x_k1_minus_eins = xk->x1;
		x_k2_minus_eins = xk->x2;
		// generate g(x)
		g_x->x1 = xk->x1 - point->x1 + partial_x1_f(xk->x1, xk->x2, f) * (f(xk->x1, xk->x2) - point->x3);
		g_x->x2 = xk->x2 - point->x2 + partial_x2_f(xk->x1, xk->x2, f) * (f(xk->x1, xk->x2) - point->x3);
		
		// generate J(f)
		jacobi.a11 = 1.0 + partial_x1_x1_f(xk->x1, xk->x2, f) * (f(xk->x1, xk->x2) - point->x3) + (partial_x1_f(xk->x1, xk->x2, f) * partial_x1_f(xk->x1, xk->x2, f));
		jacobi.a22 = 1.0 + partial_x2_x2_f(xk->x1, xk->x2, f) * (f(xk->x1, xk->x2) - point->x3) + (partial_x2_f(xk->x1, xk->x2, f) * partial_x2_f(xk->x1, xk->x2, f));
		schwarz = partial_x1_x2_f(xk->x1, xk->x2, f) * (f(xk->x1, xk->x2) - point->x3) + partial_x1_f(xk->x1, xk->x2, f) * partial_x2_f(xk->x1, xk->x2, f);
		jacobi.a12 = schwarz;
		jacobi.a21 = schwarz;
		
		// solve linear equation system with gauss
		first_line = jacobi.a11 == 0 ? 1.0 : jacobi.a11;
		second_line = jacobi.a21 == 0 ? 1.0 : jacobi.a21;
		jacobi.a11 *= second_line; jacobi.a12 *= second_line; g_x->x1 *= second_line;
		jacobi.a21 *= first_line; jacobi.a22 *= first_line; g_x->x2 *= first_line;
		jacobi.a21 -= jacobi.a11;
		jacobi.a22 -= jacobi.a12;

		// check: dividing with zero is forbidden (except you are a physicist) :)
		delta_x->x2 = jacobi.a22 != 0 ? -g_x->x2 / jacobi.a22 : 0.0;
		delta_x->x1 = jacobi.a11 != 0 ? (-g_x->x1 - jacobi.a12 * delta_x->x2) / jacobi.a11 : 0.0;

		// update x^k
		xk->x1 += delta_x->x1;
		xk->x2 += delta_x->x2;
		norm = sqrtf(pow(xk->x1 - x_k1_minus_eins, 2) + pow(xk->x2 - x_k2_minus_eins, 2));
		steps++;
	}
	printf("%d\n", steps);
	printf("%f\n", norm);
	
	free(g_x);
	free(delta_x);
}
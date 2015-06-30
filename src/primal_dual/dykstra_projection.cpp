#include <stdlib.h>
#include <stdio.h>
#include <math.h>

struct point
{
	double x1;
	double x2;
};

typedef struct point Pt;

int main(int argc, const char* argv[]) {
	Pt xk = {8.0, 8.0};
	Pt pk = {0.0, 0.0};
	Pt qk = {0.0, 0.0};
	Pt yk = {0.0, 0.0};

	double radius1 = 4.0;
	double radius2 = 2.0;
	Pt m1 = {5.0, 0.0};
	Pt m2 = {1.0, 0.0};

	double norm = 0.0;
	double inner_value_x1 = 0.0;
	double inner_value_x2 = 0.0;

	double tmp = 0.0;

	for (int i = 0; i < 1000; i++)
	{
		inner_value_x1 = (-1.0 - xk.x1) + pk.x1;
		inner_value_x2 = xk.x2 + pk.x2;
		norm = sqrtf(inner_value_x1*inner_value_x1 + inner_value_x2*inner_value_x2);

		// projection 1
		tmp = 1.0 / radius1 * inner_value_x1 / norm;
		yk.x1 = tmp + m1.x1;
		yk.x2 = inner_value_x2 < 0 ? -radius1 * sin(acos(tmp)) + m1.x2 : radius1 * sin(acos(tmp)) + m1.x2;
		// yk.x1 = radius1 * (inner_value_x1 + 1.0) / norm;
		// yk.x2 = radius1 * inner_value_x2 / norm;
		// yk.x2 = radius2 * inner_value_x2 / norm - m1.x2;

		// printf("(%f, %f)\n", yk.x1, yk.x2);

		// norm = sqrtf(inner_value_x1*inner_value_x1 + inner_value_x2*inner_value_x2);
		// yk.x1 = inner_value_x1 / fmax(1.0, norm);
		// yk.x2 = inner_value_x2 / fmax(1.0, norm);

		pk.x1 = xk.x1 + pk.x1 - yk.x1;
		pk.x2 = xk.x2 + pk.x2 - yk.x2;

		inner_value_x1 = (1.0 - yk.x1) + qk.x1;
		inner_value_x2 = yk.x2 + qk.x2;
		norm = sqrtf(inner_value_x1*inner_value_x1 + inner_value_x2*inner_value_x2);

		// projection 2
		tmp = 1.0 / radius2 * inner_value_x1 / norm;
		xk.x1 = tmp + m2.x1;
		xk.x2 = inner_value_x2 < 0 ? -radius2 * sin(acos(tmp)) + m2.x2 : radius2 * sin(acos(tmp)) + m2.x2;
		// xk.x1 = radius2 * inner_value_x1 / norm;
		// xk.x2 = radius2 * inner_value_x2 / norm;

		// printf("(%f, %f)\n", xk.x1, xk.x2);

		// norm = sqrtf(inner_value_x1*inner_value_x1 + inner_value_x2*inner_value_x2);
		// xk.x1 = inner_value_x1 / fmax(1.0, norm);
		// xk.x2 = inner_value_x2 / fmax(1.0, norm);

		qk.x1 = yk.x1 + qk.x1 - xk.x1;
		qk.x2 = yk.x2 + qk.x2 - xk.x2;
	}

	printf("(%f, %f)\n", xk.x1, xk.x2);

	return 0;
}
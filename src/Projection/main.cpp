#include <iostream>
#include <vector>
#include "Image.h"
#include "Projection.h"
#include "LinearOperator.h"
#include "Algebra.h"

using namespace std;

int main(int argc, char* argv[]) {
	Image<float> img(4, 4, 1.0);
	img.Print();
	return 0;
}
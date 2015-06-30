#include "bluring.h"

void gaussian_blur(Image& src, WriteableImage& dst, int radius, float sigma) {
	linear_filter(src, dst, gaussian_kernel(radius, sigma), radius);
}

void box_blur(Image& src, WriteableImage& dst, int radius) {
	linear_filter(src, dst, box_kernel(radius), radius);
}
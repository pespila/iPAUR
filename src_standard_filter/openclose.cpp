#include "openclose.h"

void openGrayscaleImage(GrayscaleImage& src, GrayscaleImage& dst, int* filter, int radius) {
	GrayscaleImage bypassImage;
	dilatationGrayscaleImage(src, bypassImage, filter, radius);
	erosionGrayscaleImage(bypassImage, dst, filter, radius);
}

void closeGrayscaleImage(GrayscaleImage& src, GrayscaleImage& dst, int* filter, int radius) {
	GrayscaleImage bypassImage;
	erosionGrayscaleImage(src, bypassImage, filter, radius);
	dilatationGrayscaleImage(bypassImage, dst, filter, radius);
}

void openColorImage(RGBImage& src, RGBImage& dst, int* filter, int radius) {
	RGBImage bypassImage;
	dilatationColorImage(src, bypassImage, filter, radius);
	erosionColorImage(bypassImage, dst, filter, radius);
}

void closeColorImage(RGBImage& src, RGBImage& dst, int* filter, int radius) {
	RGBImage bypassImage;
	erosionColorImage(src, bypassImage, filter, radius);
	dilatationColorImage(bypassImage, dst, filter, radius);
}
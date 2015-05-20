#include "laplace.h"

void laplace(GrayscaleImage& src, GrayscaleImage& dst) {
	unsigned short height = src.get_height(), width = src.get_width();
    dst.reset_image(height, width, src.get_type());
    int a = 0, b = 0, c = 0, d = 0, sum = 0;
    for (int i = 0; i < height; i++) {
        a = i + 1;
        b = i - 1;
        a = a >= height ? height-1 : a;
        b = b < 0 ? 0 : b;
        for (int j = 0; j < width; j++) {
            c = j + 1;
            d = j - 1;
            c = c >= width ? width-1 : c;
            d = d < 0 ? 0 : d;
            sum = src.get_gray_pixel_at_position(b, j) + src.get_gray_pixel_at_position(a, j) + src.get_gray_pixel_at_position(i, c)
            	+ src.get_gray_pixel_at_position(i, d) - 4*src.get_gray_pixel_at_position(i, j);
            sum = sum > 255 ? 255 : sum;
            sum = sum < 0 ? 0 : sum;
            sum = sum > 10 ? 255 : sum;
            dst.set_gray_pixel_at_position(i, j, sum);
        }
    }
}
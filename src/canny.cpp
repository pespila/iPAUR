#include "canny.h"

void non_maximum_supression(GrayscaleImage& src, GrayscaleImage& dst, GrayscaleImage& ang) {
	unsigned short height = src.get_height();
	unsigned short width = src.get_width();
	unsigned short angle = 0;
    dst.reset_image(height, width, src.get_type());
    int a = 0, b = 0, c = 0, d = 0;
    int i, j;

    for (i = 0; i < height; i++) {
    	for (j = 0; j < width; j++) {
    		angle = ang.get_pixel(i, j);
			a = i + 1; b = i - 1; c = j + 1; d = j - 1;
			a = a >= height ? height-2 : a;
			b = b < 0 ? 1 : b;
			c = c >= width ? width-2 : c;
			d = d < 0 ? 1 : d;
			if (angle == 0) {
				if (src.get_pixel(i, j) >= src.get_pixel(i, d) && src.get_pixel(i, j) >= src.get_pixel(i, c)) {
					dst.set_pixel(i, j, src.get_pixel(i, j));
				} else {
					dst.set_pixel(i, j, 0);
				}
			} else if (angle == 45) {
				if (src.get_pixel(i, j) >= src.get_pixel(b, c) && src.get_pixel(i, j) >= src.get_pixel(a, d)) {
					dst.set_pixel(i, j, src.get_pixel(i, j));
				} else {
					dst.set_pixel(i, j, 0);
				}
			} else if (angle == 90) {
				if (src.get_pixel(i, j) >= src.get_pixel(a, j) && src.get_pixel(i, j) >= src.get_pixel(b, j)) {
					dst.set_pixel(i, j, src.get_pixel(i, j));
				} else {
					dst.set_pixel(i, j, 0);
				}
			} else if (angle == 135) {
				if (src.get_pixel(i, j) >= src.get_pixel(a, c) && src.get_pixel(i, j) >= src.get_pixel(b, d)) {
					dst.set_pixel(i, j, src.get_pixel(i, j));
				} else {
					dst.set_pixel(i, j, 0);
				}
			} else {
				dst.set_pixel(i, j, src.get_pixel(i, j));
			}
    	}
    }
}

void hystersis(GrayscaleImage& src, GrayscaleImage& dst, int T_1, int T_2) {
	unsigned short height = src.get_height();
	unsigned short width = src.get_width();
    dst.reset_image(height, width, src.get_type());
    int x = 0, y = 0;
    int i, j, k, l;

	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			if (src.get_pixel(i, j) > T_1) {
				dst.set_pixel(i, j, 255);
				continue;
			}
			for (k = -1; k <= 1; k++) {
				for (l = -1; l <= 1; l++) {
					x = i + k; y = j + l;
					x = x >= height ? height-1 : x;
	    			x = x < 0 ? 0 : x;
	    			y = y >= width ? width-1 : y;
	    			y = y < 0 ? 0 : y;
	    			if (src.get_pixel(x, y) > T_2) {
	    				dst.set_pixel(x, y, 255);
	    			} else {
	    				dst.set_pixel(x, y, 0);
	    			}
				}

			}
		}
	}
}

void canny(GrayscaleImage& src, GrayscaleImage& dst, int T_1, int T_2) {
    unsigned short height = src.get_height();
    unsigned short width = src.get_width();
    dst.reset_image(height, width, src.get_type());

	GrayscaleImage bypass, tmp, sobel_derivative, angles;
	linear_filter(src, bypass, binomial_filter(1), 1);
	sobel_operator(bypass, sobel_derivative, angles);
	non_maximum_supression(sobel_derivative, tmp, angles);
	hystersis(tmp, dst, T_1, T_2);
}
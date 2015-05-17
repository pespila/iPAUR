#include "canny.h"

void nonMaximumSupression(GrayscaleImage& src, GrayscaleImage& dst, GrayscaleImage& ang) {
	unsigned short height = src.get_height(), width = src.get_width(), angle = 0;
    dst.reset_image(height, width, src.get_type());
    int a = 0, b = 0, c = 0, d = 0;

    for (int i = 0; i < height; i++) {
    	for (int j = 0; j < width; j++) {
    		angle = ang.get_gray_pixel_at_position(i, j);
			a = i + 1;
			b = i - 1;
			c = j + 1;
			d = j - 1;
			a = a >= height ? height-2 : a;
			b = b < 0 ? 1 : b;
			c = c >= width ? width-2 : c;
			d = d < 0 ? 1 : d;
			if (angle == 0) {
				if (src.get_gray_pixel_at_position(i, j) >= src.get_gray_pixel_at_position(i, d) && src.get_gray_pixel_at_position(i, j) >= src.get_gray_pixel_at_position(i, c)) {
					dst.set_gray_pixel_at_position(i, j, src.get_gray_pixel_at_position(i, j));
				} else {
					dst.set_gray_pixel_at_position(i, j, 0);
				}
			} else if (angle == 45) {
				if (src.get_gray_pixel_at_position(i, j) >= src.get_gray_pixel_at_position(b, c) && src.get_gray_pixel_at_position(i, j) >= src.get_gray_pixel_at_position(a, d)) {
					dst.set_gray_pixel_at_position(i, j, src.get_gray_pixel_at_position(i, j));
				} else {
					dst.set_gray_pixel_at_position(i, j, 0);
				}
			} else if (angle == 90) {
				if (src.get_gray_pixel_at_position(i, j) >= src.get_gray_pixel_at_position(a, j) && src.get_gray_pixel_at_position(i, j) >= src.get_gray_pixel_at_position(b, j)) {
					dst.set_gray_pixel_at_position(i, j, src.get_gray_pixel_at_position(i, j));
				} else {
					dst.set_gray_pixel_at_position(i, j, 0);
				}
			} else if (angle == 135) {
				if (src.get_gray_pixel_at_position(i, j) >= src.get_gray_pixel_at_position(a, c) && src.get_gray_pixel_at_position(i, j) >= src.get_gray_pixel_at_position(b, d)) {
					dst.set_gray_pixel_at_position(i, j, src.get_gray_pixel_at_position(i, j));
				} else {
					dst.set_gray_pixel_at_position(i, j, 0);
				}
			} else {
				dst.set_gray_pixel_at_position(i, j, src.get_gray_pixel_at_position(i, j));
			}
    	}
    }
}

void hystersis(GrayscaleImage& src, GrayscaleImage& dst, int T_1, int T_2) {
	unsigned short height = src.get_height(), width = src.get_width();
    dst.reset_image(height, width, src.get_type());
    int x = 0, y = 0;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (src.get_gray_pixel_at_position(i, j) > T_1) {
				dst.set_gray_pixel_at_position(i, j, 255);
				continue;
			}
			for (int k = -1; k <= 1; k++) {
				for (int l = -1; l <= 1; l++) {
					x = i + k;
					y = j + l;
					x = x >= height ? height-1 : x;
	    			x = x < 0 ? 0 : x;
	    			y = y >= width ? width-1 : y;
	    			y = y < 0 ? 0 : y;
	    			if (src.get_gray_pixel_at_position(x, y) > T_2) {
	    				dst.set_gray_pixel_at_position(x, y, 255);
	    			} else {
	    				dst.set_gray_pixel_at_position(x, y, 0);
	    			}
				}

			}
		}
	}
}

void canny(GrayscaleImage& src, GrayscaleImage& dst, int T_1, int T_2) {
    unsigned short height = src.get_height(), width = src.get_width();
    dst.reset_image(height, width, src.get_type());

	GrayscaleImage bypass, tmp, sobelDerivative, angles;
	linearFilterGrayscaleImage(src, bypass, binomialFilter(1), 1);
	sobelForCanny(bypass, sobelDerivative, angles);
	nonMaximumSupression(sobelDerivative, tmp, angles);
	hystersis(tmp, dst, T_1, T_2);
}
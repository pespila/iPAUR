#include "robertscross.h"

void roberts_cross(GrayscaleImage& src, GrayscaleImage& dst) {
    unsigned short height = src.get_height();
    unsigned short width = src.get_width();
    dst.reset_image(height, width, src.get_type());
    int x_sum = 0, y_sum = 0;
    int x = 0, y = 0;
    int i, j;
    for (i = 0; i < height; i++) {
    	y = i + 1;
    	y = y >= height ? height-1 : y;
        for (j = 0; j < width; j++) {
        	x = j + 1;
        	x = x >= width ? width - 1 : x;
            x_sum = abs(src.get_pixel(i, j) - src.get_pixel(y, x));
            y_sum = abs(src.get_pixel(y, j) - src.get_pixel(i, x));
            x_sum = x_sum > 255 ? 255 : x_sum;
            y_sum = y_sum > 255 ? 255 : y_sum;
            x_sum = x_sum < 0 ? 0 : x_sum;
            y_sum = y_sum < 0 ? 0 : y_sum;
            dst.set_pixel(i, j, x_sum + y_sum);
        }
    }
}
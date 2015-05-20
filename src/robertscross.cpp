#include "robertscross.h"

void robertsCross(GrayscaleImage& src, GrayscaleImage& dst) {
    unsigned short height = src.get_height(), width = src.get_width();
    dst.reset_image(height, width, src.get_type());
    int x = 0, y = 0, x_sum = 0, y_sum = 0;
    for (int i = 0; i < height; i++) {
    	y = i + 1;
    	y = y >= height ? height-1 : y;
        for (int j = 0; j < width; j++) {
        	x = j + 1;
        	x = x >= width ? width-1 : x;
            x_sum = abs(src.get_gray_pixel_at_position(i, j) - src.get_gray_pixel_at_position(y, x));
            y_sum = abs(src.get_gray_pixel_at_position(y, j) - src.get_gray_pixel_at_position(i, x));
            x_sum = x_sum > 255 ? 255 : x_sum;
            y_sum = y_sum > 255 ? 255 : y_sum;
            x_sum = x_sum < 0 ? 0 : x_sum;
            y_sum = y_sum < 0 ? 0 : y_sum;
            // x_sum + y_sum > 25 ? dst.set_gray_pixel_at_position(i, j, 255) : dst.set_gray_pixel_at_position(i, j, 0);;
            // sum = x_sum + y_sum;
            // sum = sum > 10 ? 255 : 0;
            dst.set_gray_pixel_at_position(i, j, x_sum + y_sum);
        }
    }
}
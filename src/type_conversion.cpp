#include "type_conversion.h"

void rgb2gray(RGBImage& src, GrayscaleImage& dst) {
	int height = src.get_height(), width = src.get_width();
	dst.reset_image(height, width, CV_8UC1);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            dst.set_gray_pixel_at_position(i, j, src.get_color_pixel_at_position(i, j, 2) * 0.299
            								   + src.get_color_pixel_at_position(i, j, 1) * 0.587
            								   + src.get_color_pixel_at_position(i, j, 0) * 0.114);
        }
    }
}

void rgb2ycrcb(RGBImage& src, YCrCbImage& dst) {
	int height = src.get_height(), width = src.get_width();
	dst.reset_image(height, width, src.get_type());
	unsigned char delta = 128;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            dst.set_color_pixel_at_position(i, j, 0, src.get_color_pixel_at_position(i, j, 2) * 0.299
            								   + src.get_color_pixel_at_position(i, j, 1) * 0.587
            								   + src.get_color_pixel_at_position(i, j, 0) * 0.114);
            dst.set_color_pixel_at_position(i, j, 1, (src.get_color_pixel_at_position(i, j, 2) - dst.get_color_pixel_at_position(i, j, 0)) * 0.713 + delta);
            dst.set_color_pixel_at_position(i, j, 2, (src.get_color_pixel_at_position(i, j, 0) - dst.get_color_pixel_at_position(i, j, 0)) * 0.564 + delta);
        }
    }
}

void rgb2hsi(RGBImage& src, HSIImage& dst) {
	int height = src.get_height(), width = src.get_width();
	dst.reset_image(height, width, src.get_type());
	int r_value = 0, g_value = 0, b_value = 0, max_rgb = 0, min_rgb = 0, difference_of_max_and_min = 0, set_H_value = 0, set_S_value = 0;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
        	r_value = src.get_color_pixel_at_position(i, j, 2);
        	g_value = src.get_color_pixel_at_position(i, j, 1);
        	b_value = src.get_color_pixel_at_position(i, j, 0);
        	max_rgb = max(max(r_value, g_value), b_value);
    		min_rgb = min(min(r_value, g_value), b_value);
    		difference_of_max_and_min = max_rgb - min_rgb;
        	dst.set_color_pixel_at_position(i, j, 2, max_rgb);
        	if (max_rgb != 0) {
                set_S_value = 255 * (max_rgb - min_rgb) / max_rgb;
        		dst.set_color_pixel_at_position(i, j, 1, set_S_value);
        	} else {
        		dst.set_color_pixel_at_position(i, j, 1, 0);
        	}

            if (max_rgb != min_rgb) {
            	if (max_rgb == r_value) {
            		set_H_value = 60 * (g_value - b_value) / difference_of_max_and_min;
            		set_H_value = set_H_value < 0 ? set_H_value + 360 : set_H_value;
            	} else if (max_rgb == g_value) {
            		set_H_value = 120 + 60 * (b_value - r_value) / difference_of_max_and_min;
            		set_H_value = set_H_value < 0 ? set_H_value + 360 : set_H_value;
            	} else if (max_rgb == b_value) {
            		set_H_value = 240 + 60 * (r_value - g_value) / difference_of_max_and_min;
            		set_H_value = set_H_value < 0 ? set_H_value + 360 : set_H_value;
                }
                dst.set_color_pixel_at_position(i, j, 0, set_H_value / 2);
            } else {
        		dst.set_color_pixel_at_position(i, j, 0, 0);
            }
        }
    }
}
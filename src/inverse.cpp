#include "inverse.h"

void inverseGrayscaleImage(GrayscaleImage& src, GrayscaleImage& dst) {
    unsigned short height = src.get_height(), width = src.get_width();
    dst.reset_image(height, width, src.get_type());
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            dst.set_gray_pixel_at_position(i, j, 255 - src.get_gray_pixel_at_position(i, j));
        }
    }
}

void inverseColorImage(RGBImage& src, RGBImage& dst) {
    unsigned short height = src.get_height(), width = src.get_width();
    dst.reset_image(height, width, src.get_type());
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < 3; k++) {
                dst.set_color_pixel_at_position(i, j, k, 255 - src.get_color_pixel_at_position(i, j, k));
            }
        }
    }
}
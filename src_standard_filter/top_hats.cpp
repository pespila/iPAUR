#include "top_hats.h"

void whiteTopHatGrayscaleImage(GrayscaleImage& src, GrayscaleImage& dst, int* filter, int radius) {
    unsigned short height = src.get_height(), width = src.get_width(), value = 0;

    openGrayscaleImage(src, dst, filter, radius);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            value = src.get_gray_pixel_at_position(i, j) - dst.get_gray_pixel_at_position(i, j);
            value = value > 255 ? 255 : value;
            value = value < 0 ? 0 : value;
            dst.set_gray_pixel_at_position(i, j, value);
        }
    }
}

void blackTopHatGrayscaleImage(GrayscaleImage& src, GrayscaleImage& dst, int* filter, int radius) {
    unsigned short height = src.get_height(), width = src.get_width(), value = 0;

    closeGrayscaleImage(src, dst, filter, radius);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            value = dst.get_gray_pixel_at_position(i, j) - src.get_gray_pixel_at_position(i, j);
            value = value > 255 ? 255 : value;
            value = value < 0 ? 0 : value;
            dst.set_gray_pixel_at_position(i, j, value);
        }
    }
}

void whiteTopHatColorImage(RGBImage& src, RGBImage& dst, int* filter, int radius) {
    unsigned short height = src.get_height(), width = src.get_width(), value = 0;

    openColorImage(src, dst, filter, radius);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < 3; k++) {
                value = src.get_color_pixel_at_position(i, j, k) - dst.get_color_pixel_at_position(i, j, k);
                value = value > 255 ? 255 : value;
                value = value < 0 ? 0 : value;
                dst.set_color_pixel_at_position(i, j, k, value);
            }
        }
    }
}

void blackTopHatColorImage(RGBImage& src, RGBImage& dst, int* filter, int radius) {
    unsigned short height = src.get_height(), width = src.get_width(), value = 0;

    closeColorImage(src, dst, filter, radius);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < 3; k++) {
                value = dst.get_color_pixel_at_position(i, j, k) - src.get_color_pixel_at_position(i, j, k);
                value = value > 255 ? 255 : value;
                value = value < 0 ? 0 : value;
                dst.set_color_pixel_at_position(i, j, k, value);
            }
        }
    }
}
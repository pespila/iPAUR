#include "top_hats.h"

void white_top_hat(Image& src, WriteableImage& dst, int* filter, int radius) {
    unsigned short height = src.get_height();
    unsigned short width = src.get_width();
    int channel = src.get_channels();
    int value = 0;
    int i, j, k;

    open(src, dst, filter, radius);

    for (k = 0; k < channel; k++) {
        dst.set_c_channel(k);
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                value = src.get_pixel(i, j, k) - dst.get_pixel(i, j, k);
                value = value > 255 ? 255 : value;
                value = value < 0 ? 0 : value;
                dst.set_pixel(i, j, value);
            }
        }
    }
}

void black_top_hat(Image& src, WriteableImage& dst, int* filter, int radius) {
    unsigned short height = src.get_height();
    unsigned short width = src.get_width();
    int channel = src.get_channels();
    int value = 0;
    int i, j, k;

    close(src, dst, filter, radius);

    for (k = 0; k < channel; k++) {
        dst.set_c_channel(k);
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                value = dst.get_pixel(i, j, k) - src.get_pixel(i, j, k);
                value = value > 255 ? 255 : value;
                value = value < 0 ? 0 : value;
                dst.set_pixel(i, j, value);
            }
        }
    }
}
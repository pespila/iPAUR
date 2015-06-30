#include "arithmetic_functions.h"

void mark_red(GrayscaleImage& src, RGBImage& dst, int type) {
    unsigned short height = src.get_height();
    unsigned short width = src.get_width();
    int i, j;
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            if (src.get_pixel(i, j) > 50) {
                dst.set_c_channel(0); dst.set_pixel(i, j, 255);
                dst.set_c_channel(1); dst.set_pixel(i, j, 0);
                dst.set_c_channel(2); dst.set_pixel(i, j, 0);
            }
        }
    }
}

void add_images(Image& src1, Image& src2, WriteableImage& dst) {
    unsigned short height = src1.get_height();
    unsigned short width = src1.get_width();
    dst.reset_image(height, width, src1.get_type());
    int channels = src1.get_channels();
    int sum = 0;
    int i, j, k;
    for (k = 0; k < channels; k++) {
        dst.set_c_channel(k);
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                sum = src1.get_pixel(i, j, k) + src2.get_pixel(i, j, k);
                sum = sum > 255 ? sum - 255 : sum;
                dst.set_pixel(i, j, sum);
            }
        }
    }
}
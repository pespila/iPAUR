#include "inverse.h"

void inverse(Image& src, WriteableImage& dst) {
    unsigned short height = src.get_height();
    unsigned short width = src.get_width();
    dst.reset_image(height, width, src.get_type());
    int i, j, k;
    for (k = 0; k < 3; k++) {
        dst.set_c_channel(k);
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                dst.set_pixel(i, j, 255 - src.get_pixel(i, j, k));
            }
        }
    }
}
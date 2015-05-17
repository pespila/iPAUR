#include "dilatation.h"

void dilatationGrayscaleImage(GrayscaleImage& src, GrayscaleImage& dst, int* filter, int radius) {
    int x = 0, y = 0;
    unsigned short height = src.get_height(), width = src.get_width(), sup = 0;
    dst.reset_image(height, width, src.get_type());
    short* filtered = (short*)malloc(height*width*sizeof(short));

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            sup = 0;
            y = width - 1;
            for (int k = (-1)*radius; k <= radius; k++) {
                x = j + k;
                if (x < 0) {
                    if (filter[k+radius] == 1 && src.get_gray_pixel_at_position(i, 0) > sup) {
                        sup = src.get_gray_pixel_at_position(i, 0);
                    }
                } else if (x >= width) {
                    if (filter[k+radius] == 1 && src.get_gray_pixel_at_position(i, y) > sup) {
                        sup = src.get_gray_pixel_at_position(i, y);
                    }
                } else {
                    if (filter[k+radius] == 1 && src.get_gray_pixel_at_position(i, x) > sup) {
                        sup = src.get_gray_pixel_at_position(i, x);
                    }
                }
            }
            filtered[j+i*width] = sup;
        }
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            sup = filtered[j+i*width];
            y = height - 1;
            for (int k = (-1)*radius; k <= radius; k++) {
                x = i + k;
                if (x < 0) {
                    if (filter[k+radius] == 1 && filtered[j] > sup) {
                        sup = filtered[j];
                    }
                } else if (x >= height) {
                    if (filter[k+radius] == 1 && filtered[j+y*width] > sup) {
                        sup = filtered[j+y*width];
                    }
                } else {
                    if (filter[k+radius] == 1 && filtered[j+x*width] > sup) {
                        sup = filtered[j+x*width];
                    }
                }
            }
            dst.set_gray_pixel_at_position(i, j, sup);
        }
    }

    free(filtered);
}

void dilatationColorImage(RGBImage& src, RGBImage& dst, int* filter, int radius) {
    int x = 0, y = 0;
    unsigned short height = src.get_height(), width = src.get_width(), sup = 0;
    dst.reset_image(height, width, src.get_type());
    short** filtered = (short**)malloc(height*width*sizeof(short*));
    for (int i = 0; i < height*width; i++) {
        filtered[i] = (short*)malloc(3*sizeof(short));
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int l = 0; l < 3; l++) {
                sup = 0;
                y = width - 1;
                for (int k = (-1)*radius; k <= radius; k++) {
                    x = j + k;
                    if (x < 0) {
                        if (filter[k+radius] == 1 && src.get_color_pixel_at_position(i, 0, l) > sup) {
                            sup = src.get_color_pixel_at_position(i, 0, l);
                        }
                    } else if (x >= width) {
                        if (filter[k+radius] == 1 && src.get_color_pixel_at_position(i, y, l) > sup) {
                            sup = src.get_color_pixel_at_position(i, y, l);
                        }
                    } else {
                        if (filter[k+radius] == 1 && src.get_color_pixel_at_position(i, x, l) > sup) {
                            sup = src.get_color_pixel_at_position(i, x, l);
                        }
                    }
                }
                filtered[j+i*width][l] = sup;
            }
        }
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int l = 0; l < 3; l++) {
                sup = filtered[j+i*width][l];
                y = height - 1;
                for (int k = (-1)*radius; k <= radius; k++) {
                    x = i + k;
                    if (x < 0) {
                        if (filter[k+radius] == 1 && filtered[j][l] > sup) {
                            sup = filtered[j][l];
                        }
                    } else if (x >= height) {
                        if (filter[k+radius] == 1 && filtered[j+y*width][l] > sup) {
                            sup = filtered[j+y*width][l];
                        }
                    } else {
                        if (filter[k+radius] == 1 && filtered[j+x*width][l] > sup) {
                            sup = filtered[j+x*width][l];
                        }
                    }
                }
                dst.set_color_pixel_at_position(i, j, l, sup);
            }
        }
    }

    for (int i = 0; i < height*width; i++) {
        free(filtered[i]);
    }
    free(filtered);
}
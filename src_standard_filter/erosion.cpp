#include "erosion.h"

void erosionGrayscaleImage(GrayscaleImage& src, GrayscaleImage& dst, int* filter, int radius) {
    int x = 0, y = 0;
    unsigned short height = src.get_height(), width = src.get_width(), inf = 255;
    dst.reset_image(height, width, src.get_type());
    short* filtered = (short*)malloc(height*width*sizeof(short));

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            inf = 255;
            y = width - 1;
            for (int k = (-1)*radius; k <= radius; k++) {
                x = j + k;
                if (x < 0) {
                    if (filter[k+radius] == 1 && src.get_gray_pixel_at_position(i, 0) < inf) {
                        inf = src.get_gray_pixel_at_position(i, 0);
                    }
                } else if (x >= width) {
                    if (filter[k+radius] == 1 && src.get_gray_pixel_at_position(i, y) < inf) {
                        inf = src.get_gray_pixel_at_position(i, y);
                    }
                } else {
                    if (filter[k+radius] == 1 && src.get_gray_pixel_at_position(i, x) < inf) {
                        inf = src.get_gray_pixel_at_position(i, x);
                    }
                }
            }
            filtered[j+i*width] = inf;
        }
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            inf = filtered[j+i*width];
            y = height - 1;
            for (int k = (-1)*radius; k <= radius; k++) {
                x = i + k;
                if (x < 0) {
                    if (filter[k+radius] == 1 && filtered[j] < inf) {
                        inf = filtered[j];
                    }
                } else if (x >= height) {
                    if (filter[k+radius] == 1 && filtered[j+y*width] < inf) {
                        inf = filtered[j+y*width];
                    }
                } else {
                    if (filter[k+radius] == 1 && filtered[j+x*width] < inf) {
                        inf = filtered[j+x*width];
                    }
                }
            }
            dst.set_gray_pixel_at_position(i, j, inf);
        }
    }

    free(filtered);
}

void erosionColorImage(RGBImage& src, RGBImage& dst, int* filter, int radius) {
    int x = 0, y = 0;
    unsigned short height = src.get_height(), width = src.get_width(), inf = 255;
    dst.reset_image(height, width, src.get_type());
    short** filtered = (short**)malloc(height*width*sizeof(short*));
    for (int i = 0; i < height*width; i++) {
        filtered[i] = (short*)malloc(3*sizeof(short));
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int l = 0; l < 3; l++) {
                inf = 255;
                y = width - 1;
                for (int k = (-1)*radius; k <= radius; k++) {
                    x = j + k;
                    if (x < 0) {
                        if (filter[k+radius] == 1 && src.get_color_pixel_at_position(i, 0, l) < inf) {
                            inf = src.get_color_pixel_at_position(i, 0, l);
                        }
                    } else if (x >= width) {
                        if (filter[k+radius] == 1 && src.get_color_pixel_at_position(i, y, l) < inf) {
                            inf = src.get_color_pixel_at_position(i, y, l);
                        }
                    } else {
                        if (filter[k+radius] == 1 && src.get_color_pixel_at_position(i, x, l) < inf) {
                            inf = src.get_color_pixel_at_position(i, x, l);
                        }
                    }
                }
                filtered[j+i*width][l] = inf;
            }
        }
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int l = 0; l < 3; l++) {
                inf = filtered[j+i*width][l];
                y = height - 1;
                for (int k = (-1)*radius; k <= radius; k++) {
                    x = i + k;
                    if (x < 0) {
                        if (filter[k+radius] == 1 && filtered[j][l] < inf) {
                            inf = filtered[j][l];
                        }
                    } else if (x >= height) {
                        if (filter[k+radius] == 1 && filtered[j+y*width][l] < inf) {
                            inf = filtered[j+y*width][l];
                        }
                    } else {
                        if (filter[k+radius] == 1 && filtered[j+x*width][l] < inf) {
                            inf = filtered[j+x*width][l];
                        }
                    }
                }
                dst.set_color_pixel_at_position(i, j, l, inf);
            }
        }
    }

    for (int i = 0; i < height*width; i++) {
        free(filtered[i]);
    }
    free(filtered);
}
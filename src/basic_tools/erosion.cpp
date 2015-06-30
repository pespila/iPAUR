#include "erosion.h"

void erosion(Image& src, WriteableImage& dst, int* filter, int radius) {
    unsigned short height = src.get_height();
    unsigned short width = src.get_width();
    dst.reset_image(height, width, src.get_type());
    unsigned short inf = 255;
    int size = height * width;
    int channel = src.get_channels();
    int x = 0, y = 0;
    int i, j, k, l;
    
    short** filtered = (short**)malloc(size*sizeof(short*));
    for (i = 0; i < size; i++) filtered[i] = (short*)malloc(channel*sizeof(short));

    for (l = 0; l < channel; l++) {
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                inf = 255;
                y = width - 1;
                for (k = (-1) * radius; k <= radius; k++) {
                    x = j + k;
                    if (x < 0) {
                        if (filter[k + radius] == 1 && src.get_pixel(i, 0, l) < inf) {
                            inf = src.get_pixel(i, 0, l);
                        }
                    } else if (x >= width) {
                        if (filter[k + radius] == 1 && src.get_pixel(i, y, l) < inf) {
                            inf = src.get_pixel(i, y, l);
                        }
                    } else {
                        if (filter[k + radius] == 1 && src.get_pixel(i, x, l) < inf) {
                            inf = src.get_pixel(i, x, l);
                        }
                    }
                }
                filtered[j + i * width][l] = inf;
            }
        }
    }

    for (l = 0; l < channel; l++) {
        dst.set_c_channel(l);
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                inf = filtered[j + i * width][l];
                y = height - 1;
                for (k = (-1) * radius; k <= radius; k++) {
                    x = i + k;
                    if (x < 0) {
                        if (filter[k + radius] == 1 && filtered[j][l] < inf) {
                            inf = filtered[j][l];
                        }
                    } else if (x >= height) {
                        if (filter[k + radius] == 1 && filtered[j + y * width][l] < inf) {
                            inf = filtered[j + y * width][l];
                        }
                    } else {
                        if (filter[k + radius] == 1 && filtered[j + x * width][l] < inf) {
                            inf = filtered[j + x * width][l];
                        }
                    }
                }
                dst.set_pixel(i, j, inf);
            }
        }
    }

    for (i = 0; i < size; i++) free(filtered[i]);
    free(filtered);
}
#include "linear_filter.h"

void linear_filter(Image& src, WriteableImage& dst, float* filter, int radius) {
    unsigned short height = src.get_height();
    unsigned short width = src.get_width();
    dst.reset_image(height, width, src.get_type());
    int size = height * width;
    int channel = src.get_channels();
    int i, j, k, l;
    int sum = 0;
    unsigned char** filtered = (unsigned char**)malloc(size*sizeof(unsigned char*));
    for (int i = 0; i < size; i++) {
        filtered[i] = (unsigned char*)malloc(channel*sizeof(unsigned char));
    }
    
    for (l = 0; l < channel; l++) {
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                sum = 0;
                for (k = (-1) * radius; k <= radius; k++) {
                    if (j + k < 0) {
                        sum += filter[k + radius] * src.get_pixel(i, 0, l);
                    } else if (j + k >= width) {
                        sum += filter[k + radius] * src.get_pixel(i, width - 1, l);
                    } else {
                        sum += filter[k + radius] * src.get_pixel(i, j + k, l);
                    }
                }
                sum = sum < 0 ? 0 : sum;
                sum = sum > 255 ? 255 : sum;
                filtered[j + i * width][l] = sum;
            }
        }
    }
    for (l = 0; l < channel; l++) {
        dst.set_c_channel(l);
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                sum = 0.0;
                for (k = (-1) * radius; k <= radius; k++) {
                    if (i + k < 0) {
                        sum += filter[k + radius] * filtered[j][l];
                    } else if (i + k >= height) {
                        sum += filter[k + radius] * filtered[j + (height - 1) * width][l];
                    } else {
                        sum += filter[k + radius] * filtered[j + (i + k) * width][l];
                    }
                }
                sum = sum < 0 ? 0 : sum;
                sum = sum > 255 ? 255 : sum;
                dst.set_pixel(i, j, sum);
            }
        }
    }
    for (i = 0; i < size; i++) {
        free(filtered[i]);
    }
    free(filtered);
}
#include "linear_filter.h"

void linearFilterGrayscaleImage(GrayscaleImage& src, GrayscaleImage& dst, float* filter, int radius) {
    unsigned short height = src.get_height(), width = src.get_width(), sum = 0;
    dst.reset_image(height, width, src.get_type());
    
    short* filtered = (short*)malloc(height*width*sizeof(short));
    
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            sum = 0;
            for (int k = (-1)*radius; k <= radius; k++) {
                if (j+k < 0) {
                    sum += filter[k+radius] * src.get_gray_pixel_at_position(i, 0);
                } else if (j+k >= width) {
                    sum += filter[k+radius] * src.get_gray_pixel_at_position(i, width-1);
                } else {
                    sum += filter[k+radius] * src.get_gray_pixel_at_position(i, j+k);
                }
            }
            sum = sum < 0 ? 0 : sum;
            sum = sum > 255 ? 255 : sum;
            filtered[j + i * width] = sum;
        }
    }
    
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            sum = 0.0;
            for (int k = (-1)*radius; k <= radius; k++) {
                if (i+k < 0) {
                    sum += filter[k+radius] * filtered[j];
                } else if (i+k >= height) {
                    sum += filter[k+radius] * filtered[j+(height-1)*width];
                } else {
                    sum += filter[k+radius] * filtered[j+(i+k)*width];
                }
            }
            sum = sum < 0 ? 0 : sum;
            sum = sum > 255 ? 255 : sum;
            dst.set_gray_pixel_at_position(i, j, sum);
        }
    }
    
    free(filtered);
}

void linearFilterColorImage(RGBImage& src, RGBImage& dst, float* filter, int radius) {
    unsigned short height = src.get_height(), width = src.get_width(), sum = 0;
    dst.reset_image(height, width, src.get_type());
    
    short** filtered = (short**)malloc(height*width*sizeof(short*));
    for (int i = 0; i < width*height; i++) {
        filtered[i] = (short*)malloc(3*sizeof(short));
    }
    
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int l = 0; l < 3; l++) {
                sum = 0;
                for (int k = (-1)*radius; k <= radius; k++) {
                    if (j+k < 0) {
                        sum += filter[k+radius] * src.get_color_pixel_at_position(i, 0, l);
                    } else if (j+k >= width) {
                        sum += filter[k+radius] * src.get_color_pixel_at_position(i, width-1, l);
                    } else {
                        sum += filter[k+radius] * src.get_color_pixel_at_position(i, j+k, l);
                    }
                }
                sum = sum < 0 ? 0 : sum;
                sum = sum > 255 ? 255 : sum;
                filtered[j + i * width][l] = sum;
            }
        }
    }
    
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int l = 0; l < 3; l++) {
                sum = 0.0;
                for (int k = (-1)*radius; k <= radius; k++) {
                    if (i+k < 0) {
                        sum += filter[k+radius] * filtered[j][l];
                    } else if (i+k >= height) {
                        sum += filter[k+radius] * filtered[j+(height-1)*width][l];
                    } else {
                        sum += filter[k+radius] * filtered[j+(i+k)*width][l];
                    }
                }
                sum = sum < 0 ? 0 : sum;
                sum = sum > 255 ? 255 : sum;
                dst.set_color_pixel_at_position(i, j, l, sum);
            }
        }
    }
    
    for (int i = 0; i < width*height; i++) {
        free(filtered[i]);
    }
    
    free(filtered);
}
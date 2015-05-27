#include "sobel.h"

void sobel(GrayscaleImage& src, GrayscaleImage& dst) {
    unsigned short height = src.get_height();
    unsigned short width = src.get_width();
    dst.reset_image(height, width, src.get_type());
    int size = height * width;
    short* x_filtered = (short*)malloc(size*sizeof(short));
    short* y_filtered = (short*)malloc(size*sizeof(short));
    int x = 0, y = 0;
    int sum = 0;
    int i, j;

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            x = j + 1;
            x = x >= width ? width - 1 : x;
            y = j - 1;
            y = y < 0 ? 0 : y;
            x_filtered[j + i * width] = (src.get_pixel(i, y) - src.get_pixel(i, x)) / 2;
        }
    }
    for (i = 0; i < height; i++) {
        x = i + 1;
        x = x >= height ? height - 1 : x;
        y = i - 1;
        y = y < 0 ? 0 : y;
        for (j = 0; j < width; j++) {
            x_filtered[j + i * width] += (x_filtered[j + y * width] + x_filtered[j + i * width] + x_filtered[j + x * width]) / 4;
        }
    }

    for (i = 0; i < height; i++) {
        x = i + 1;
        x = x >= height ? height - 1 : x;
        y = i - 1;
        y = y < 0 ? 0 : y;
        for (j = 0; j < width; j++) {
            y_filtered[j + i * width] = (src.get_pixel(y, j) - src.get_pixel(x, j)) / 2;
        }
    }
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            x = j + 1;
            x = x >= width ? width - 1 : x;
            y = j - 1;
            y = y < 0 ? 0 : y;
            y_filtered[j + i * width] += (y_filtered[y + i * width] + y_filtered[j + i * width] + y_filtered[x + i * width]) / 4;
        }
    }

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            sum = abs(x_filtered[j + i * width]) + abs(y_filtered[j + i * width]);
            sum = sum > 255 ? 255 : sum;
            sum = sum < 0 ? 0 : sum;
            sum = sum > 10 ? 255 : 0;
            dst.set_pixel(i, j, sum);
        }
    }
    
    free(x_filtered);
    free(y_filtered);
}

void sobel_operator(GrayscaleImage& src, GrayscaleImage& dst, GrayscaleImage& ang) {
    unsigned short height = src.get_height();
    unsigned short width = src.get_width();
    dst.reset_image(height, width, src.get_type());
    ang.reset_image(height, width, src.get_type());
    int magnitude = 0;
    int angle = 0;
    int size = height * width;
    short* x_filtered = (short*)malloc(size*sizeof(short));
    short* y_filtered = (short*)malloc(size*sizeof(short));
    int x = 0, y = 0;
    int sum = 0;
    int i, j;

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            x = j + 1;
            x = x >= width ? width - 1 : x;
            y = j - 1;
            y = y < 0 ? 0 : y;
            x_filtered[j + i * width] = (src.get_pixel(i, y) - src.get_pixel(i, x)) / 2;
        }
    }
    for (i = 0; i < height; i++) {
        x = i + 1;
        x = x >= height ? height - 1 : x;
        y = i - 1;
        y = y < 0 ? 0 : y;
        for (j = 0; j < width; j++) {
            x_filtered[j + i * width] += (x_filtered[j + y * width] + x_filtered[j + i * width] + x_filtered[j + x * width]) / 4;
        }
    }

    for (i = 0; i < height; i++) {
        x = i + 1;
        x = x >= height ? height - 1 : x;
        y = i - 1;
        y = y < 0 ? 0 : y;
        for (j = 0; j < width; j++) {
            y_filtered[j + i * width] = (src.get_pixel(y, j) - src.get_pixel(x, j)) / 2;
        }
    }
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            x = j + 1;
            x = x >= width ? width - 1 : x;
            y = j - 1;
            y = y < 0 ? 0 : y;
            y_filtered[j + i * width] += (y_filtered[y + i * width] + y_filtered[j + i * width] + y_filtered[x + i * width]) / 4;
        }
    }

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            sum = abs(x_filtered[j + i * width]) + abs(y_filtered[j + i * width]);
            sum = sum > 255 ? 255 : sum;
            sum = sum < 0 ? 0 : sum;
            sum = sum > 10 ? 255 : 0;
            dst.set_pixel(i, j, sum);
        }
    }

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            magnitude = abs(x_filtered[j + i * width]) + abs(y_filtered[j + i * width]);
            magnitude = magnitude > 255 ? 255 : magnitude;
            magnitude = magnitude < 0 ? 0 : magnitude;
            dst.set_pixel(i, j, magnitude);
            if (x_filtered[j + i * width] == 0) {
                if (y_filtered[j + i * width] == 0) {
                    angle = 0;
                } else {
                    angle = 90;
                }
            } else {
                angle = atan(y_filtered[j + i * width] / x_filtered[j + i * width]);
            }
            angle = angle > 255 ? 360 - angle : angle;
            if ( ( angle >= 0 && angle < 23 ) || ( angle >= 158 && angle <= 180 ) ) {
                angle = 0;
            } else if ( angle >= 23 && angle < 68 ) {
                angle = 45;
            } else if ( angle >= 68 && angle < 113 ) {
                angle = 90;
            } else if ( angle >= 113 && angle < 158 ) {
                angle = 135;
            } else {
                angle = 0;
            }
            ang.set_pixel(i, j, angle);
        }
    }
    
    free(x_filtered);
    free(y_filtered);
}
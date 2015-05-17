#include "sobel.h"

void sobel(GrayscaleImage& src, GrayscaleImage& dst) {
    int x = 0, y = 0;
    unsigned short height = src.get_height(), width = src.get_width(), sum = 0;
    dst.reset_image(height, width, src.get_type());
    short* x_filtered = (short*)malloc(height*width*sizeof(short));
    short* y_filtered = (short*)malloc(height*width*sizeof(short));

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            x = j + 1;
            x = x >= width ? width-1 : x;
            y = j - 1;
            y = y < 0 ? 0 : y;
            x_filtered[j + i * width] = (src.get_gray_pixel_at_position(i, y) - src.get_gray_pixel_at_position(i, x)) / 2;
        }
    }
    for (int i = 0; i < height; i++) {
        x = i + 1;
        x = x >= height ? height-1 : x;
        y = i - 1;
        y = y < 0 ? 0 : y;
        for (int j = 0; j < width; j++) {
            x_filtered[j + i * width] += (x_filtered[j + y * width] + x_filtered[j + i * width] + x_filtered[j + x * width]) / 4;
        }
    }

    for (int i = 0; i < height; i++) {
        x = i + 1;
        x = x >= height ? height-1 : x;
        y = i - 1;
        y = y < 0 ? 0 : y;
        for (int j = 0; j < width; j++) {
            y_filtered[j + i * width] = (src.get_gray_pixel_at_position(y, j) - src.get_gray_pixel_at_position(x, j)) / 2;
        }
    }
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            x = j + 1;
            x = x >= width ? width-1 : x;
            y = j - 1;
            y = y < 0 ? 0 : y;
            y_filtered[j + i * width] += (y_filtered[y + i * width] + y_filtered[j + i * width] + y_filtered[x + i * width]) / 4;
        }
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            sum = abs(x_filtered[j+i*width]) + abs(y_filtered[j+i*width]);
            sum = sum > 255 ? 255 : sum;
            sum = sum < 0 ? 0 : sum;
            sum = sum > 10 ? 255 : 0;
            dst.set_gray_pixel_at_position(i, j, sum);
        }
    }
    
    free(x_filtered);
    free(y_filtered);
}

void sobelForCanny(GrayscaleImage& src, GrayscaleImage& dst, GrayscaleImage& ang) {
    int x = 0, y = 0;
    unsigned short height = src.get_height(), width = src.get_width(), magnitude = 0, angle = 0;
    dst.reset_image(height, width, src.get_type());
    ang.reset_image(height, width, src.get_type());
    short* x_filtered = (short*)malloc(height*width*sizeof(short));
    short* y_filtered = (short*)malloc(height*width*sizeof(short));

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            x = j + 1;
            x = x >= width ? width-1 : x;
            y = j - 1;
            y = y < 0 ? 0 : y;
            x_filtered[j + i * width] = (src.get_gray_pixel_at_position(i, y) - src.get_gray_pixel_at_position(i, x)) / 2;
        }
    }
    for (int i = 0; i < height; i++) {
        x = i + 1;
        x = x >= height ? height-1 : x;
        y = i - 1;
        y = y < 0 ? 0 : y;
        for (int j = 0; j < width; j++) {
            x_filtered[j + i * width] += (x_filtered[j + y * width] + x_filtered[j + i * width] + x_filtered[j + x * width]) / 4;
        }
    }

    for (int i = 0; i < height; i++) {
        x = i + 1;
        x = x >= height ? height-1 : x;
        y = i - 1;
        y = y < 0 ? 0 : y;
        for (int j = 0; j < width; j++) {
            y_filtered[j + i * width] = (src.get_gray_pixel_at_position(y, j) - src.get_gray_pixel_at_position(x, j)) / 2;
        }
    }
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            x = j + 1;
            x = x >= width ? width-1 : x;
            y = j - 1;
            y = y < 0 ? 0 : y;
            y_filtered[j + i * width] += (y_filtered[y + i * width] + y_filtered[j + i * width] + y_filtered[x + i * width]) / 4;
        }
    }


    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            magnitude = abs(x_filtered[j+i*width]) + abs(y_filtered[j+i*width]);
            magnitude = magnitude > 255 ? 255 : magnitude;
            magnitude = magnitude < 0 ? 0 : magnitude;
            dst.set_gray_pixel_at_position(i, j, magnitude);
            if (x_filtered[j + i * width] == 0) {
                if (y_filtered[j + i * width] == 0) {
                    angle = 0;
                } else {
                    angle = 90;
                }
            } else {
                angle = atan(y_filtered[j + i * width] / x_filtered[j + i * width]);
            }
            angle = angle > 255 ? 360-angle : angle;
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
            ang.set_gray_pixel_at_position(i, j, angle);
        }
    }
    
    free(x_filtered);
    free(y_filtered);
}
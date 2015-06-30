#include "prewitt.h"

void prewitt(GrayscaleImage& src, GrayscaleImage& dst) {
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
        	x_filtered[j + i * width] = (x_filtered[j + y * width] + x_filtered[j + i * width] + x_filtered[j + x * width]) / 3;
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
        	y_filtered[j + i * width] = (y_filtered[y + i * width] + y_filtered[j + i * width] + y_filtered[x + i * width]) / 3;
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
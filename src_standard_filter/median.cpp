#include "median.h"

unsigned char getMedian(unsigned char* array) {
    for (int i = 8; i > 0; --i) {
        for (int j = 0; j < i; ++j) {
            if (array[j] > array[j+1]) {
                unsigned char tmp = array[j];
                array[j] = array[j+1];
                array[j+1] = tmp;
            }
        }
    }
    return array[4];
}

void medianFilterGrayscaleImage(GrayscaleImage& src, GrayscaleImage& dst) {
	unsigned short height = src.get_height(), width = src.get_width();
    dst.reset_image(height, width, src.get_type());
    int x = 0, y = 0;

    unsigned char* arrayOfCurrentValues = (unsigned char*)malloc(9*sizeof(unsigned char));

    for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = -1; k <= 1; k++) {
				for (int l = -1; l <= 1; l++) {
					x = i + k;
					y = j + l;
					x = x >= height ? height-1 : x;
	    			x = x < 0 ? 0 : x;
	    			y = y >= width ? width-1 : y;
	    			y = y < 0 ? 0 : y;
	    			arrayOfCurrentValues[(l+1) + (k+1) * 3] = src.get_gray_pixel_at_position(x, y);
				}
			}
			dst.set_gray_pixel_at_position(i, j, getMedian(arrayOfCurrentValues));
		}
	}
	free(arrayOfCurrentValues);
}

void medianFilterColorImage(RGBImage& src, RGBImage& dst) {
	unsigned short height = src.get_height(), width = src.get_width();
    dst.reset_image(height, width, src.get_type());
    int x = 0, y = 0;

    unsigned char* arrayOfCurrentValues = (unsigned char*)malloc(9*sizeof(unsigned char));

    for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			for (int m = 0; m < 3; m++) {
				for (int k = -1; k <= 1; k++) {
					for (int l = -1; l <= 1; l++) {
						x = i + k;
						y = j + l;
						x = x >= height ? height-1 : x;
		    			x = x < 0 ? 0 : x;
		    			y = y >= width ? width-1 : y;
		    			y = y < 0 ? 0 : y;
		    			arrayOfCurrentValues[(l+1) + (k+1) * 3] = src.get_color_pixel_at_position(x, y, m);
					}
				}
    			dst.set_color_pixel_at_position(i, j, m, getMedian(arrayOfCurrentValues));
			}
		}
	}
	free(arrayOfCurrentValues);
}
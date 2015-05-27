#include "median.h"

unsigned char median_of_array(unsigned char* array) {
	unsigned char tmp = 0;
	int i, j;
    for (i = 8; i > 0; --i) {
        for (j = 0; j < i; ++j) {
            if (array[j] > array[j + 1]) {
                tmp = array[j];
                array[j] = array[j + 1];
                array[j + 1] = tmp;
            }
        }
    }
    return array[4];
}

void median(Image& src, WriteableImage& dst) {
	unsigned short height = src.get_height();
	unsigned short width = src.get_width();
    dst.reset_image(height, width, src.get_type());
    int channel = src.get_channels();
    int x = 0, y = 0;
    int i, j, k, l, m;

    unsigned char* array_with_values_to_check = (unsigned char*)malloc(9*sizeof(unsigned char));

	for (m = 0; m < channel; m++) {
		dst.set_c_channel(m);
	    for (i = 0; i < height; i++) {
			for (j = 0; j < width; j++) {
				for (k = -1; k <= 1; k++) {
					for (l = -1; l <= 1; l++) {
						x = i + k; y = j + l;
						x = x >= height ? height - 1 : x;
		    			x = x < 0 ? 0 : x;
		    			y = y >= width ? width - 1 : y;
		    			y = y < 0 ? 0 : y;
		    			array_with_values_to_check[(l + 1) + (k + 1) * channel] = src.get_pixel(x, y, m);
					}
				}
    			dst.set_pixel(i, j, median_of_array(array_with_values_to_check));
			}
		}
	}
	free(array_with_values_to_check);
}
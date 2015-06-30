#include "duto.h"

void duto(Image& src, WriteableImage& dst, float* filter, int radius, float eta) {
	unsigned short height = src.get_height();
    unsigned short width = src.get_width();
    dst.reset_image(height, width, src.get_type());
    int size = height * width;
    int channel = src.get_channels();
    int i, j, k;

    unsigned char** filtered = (unsigned char**)malloc(size*sizeof(unsigned char*));
    for (i = 0; i < size; i++) {
    	filtered[i] = (unsigned char*)malloc(channel*sizeof(unsigned char));
    }

	for (k = 0; k < channel; k++) {
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
	    		filtered[j + i * width][k] = eta * src.get_pixel(i, j, k);
    		}
    	}
    }
    linear_filter(src, dst, filter, radius);
	for (k = 0; k < channel; k++) {
        dst.set_c_channel(k);
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
	    		dst.set_pixel(i, j, eta * dst.get_pixel(i, j, k) + filtered[j + i * width][k]);
    		}
    	}
    }

    for (i = 0; i < size; i++) {
    	free(filtered[i]);
    }
    free(filtered);
}
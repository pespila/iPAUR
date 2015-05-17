#include "duto.h"

void dutoGrayscaleImage(GrayscaleImage& src, GrayscaleImage& dst, float* filter, int radius, float eta) {
	unsigned short height = src.get_height(), width = src.get_width();
    dst.reset_image(height, width, src.get_type());

    unsigned char* filtered = (unsigned char*)malloc(height*width*sizeof(unsigned char));

    for (int i = 0; i < height; i++) {
    	for (int j = 0; j < width; j++) {
    		filtered[j + i * width] = (1 - eta) * src.get_gray_pixel_at_position(i, j);
    	}
    }
    linearFilterGrayscaleImage(src, dst, filter, radius);
    for (int i = 0; i < height; i++) {
    	for (int j = 0; j < width; j++) {
    		dst.set_gray_pixel_at_position(i, j, eta * dst.get_gray_pixel_at_position(i, j) + filtered[j + i * width]);
    	}
    }

    free(filtered);
}

void dutoColorImage(RGBImage& src, RGBImage& dst, float* filter, int radius, float eta) {
	unsigned short height = src.get_height(), width = src.get_width();
    dst.reset_image(height, width, src.get_type());

    unsigned char** filtered = (unsigned char**)malloc(height*width*sizeof(unsigned char*));
    for (int i = 0; i < height*width; i++) {
    	filtered[i] = (unsigned char*)malloc(3*sizeof(unsigned char));
    }

    for (int i = 0; i < height; i++) {
    	for (int j = 0; j < width; j++) {
    		for (int k = 0; k < 3; k++)
    		{
	    		filtered[j + i * width][k] = eta * src.get_color_pixel_at_position(i, j, k);
    		}
    	}
    }
    linearFilterColorImage(src, dst, filter, radius);
    for (int i = 0; i < height; i++) {
    	for (int j = 0; j < width; j++) {
    		for (int k = 0; k < 3; k++)
    		{
	    		dst.set_color_pixel_at_position(i, j, k, eta * dst.get_color_pixel_at_position(i, j, k) + filtered[j + i * width][k]);
    		}
    	}
    }

    for (int i = 0; i < height*width; i++) {
    	free(filtered[i]);
    }
    free(filtered);
}
#include "image.h"
#include "typeconversion.h"

TypeConversion::TypeConversion(int height, int width, int channel, char type) {
    this->height = height;
    this->width = width;
    this->channel = channel;
    this->type = type;
}
TypeConversion::TypeConversion(Image& src) {
    this->height = src.get_height();
    this->width = src.get_width();
    this->channel = src.get_channels();
    this->type = src.get_type();
}
TypeConversion::~TypeConversion() {}

void TypeConversion::rgb2gray(RGBImage& src, GrayscaleImage& dst) {
    dst.reset_image(height, width, CV_8UC1);
    for (int i = 0; i < height*width; i++)
        dst.set_pixel(0, i, 0, src.get_pixel(0, i, 2) * 0.299 + src.get_pixel(0, i, 1) * 0.587 + src.get_pixel(0, i, 0) * 0.114);
}

void TypeConversion::gray2rgb(GrayscaleImage& src, RGBImage& dst) {
    dst.reset_image(height, width, CV_8UC3);
    for (int k = 0; k < channel; k++) {
        for (int i = 0; i < height*width; i++) {
            dst.set_pixel(0, i, k, src.get_pixel(0, i, 0));
        }
    }
}

void TypeConversion::rgb2ycrcb(RGBImage& src, YCrCbImage& dst) {
    dst.reset_image(height, width, CV_8UC3);
    for (int i = 0; i < height*width; i++) {
        dst.set_pixel(0, i, 0, src.get_pixel(0, i, 2) * 0.299 + src.get_pixel(0, i, 1) * 0.587 + src.get_pixel(0, i, 0) * 0.114);
        dst.set_pixel(0, i, 1, (src.get_pixel(0, i, 2) - dst.get_pixel(0, i, 0)) * 0.713 + 128); // delta = 128
        dst.set_pixel(0, i, 2, (src.get_pixel(0, i, 0) - dst.get_pixel(0, i, 0)) * 0.564 + 128); // delta = 128
    }
}

void TypeConversion::rgb2hsi(RGBImage& src, HSIImage& dst) {
    dst.reset_image(height, width, CV_8UC3);
    int max, min, max_minus_min, s_value, h_value;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            max = 0;
            min = 255;
            for (int k = 0; k < channel; k++) {
                max = src.get_pixel(i, j, k) <= max ? max : src.get_pixel(i, j, k);
                min = src.get_pixel(i, j, k) >= min ? min : src.get_pixel(i, j, k);
            }
            max_minus_min = max - min;
            dst.set_pixel(i, j, 2, max);
            s_value = max != 0 ? 255 * (max_minus_min) / max : 0;
            dst.set_pixel(i, j, 1, s_value);
            h_value = max == min ? 0 : -1;
            if (h_value == -1) {
                if (max == src.get_pixel(i, j, 2)) {
                    h_value = 60 * (src.get_pixel(i, j, 1) - src.get_pixel(i, j, 0)) / max_minus_min;
                    h_value = h_value < 0 ? h_value + 360 : h_value;
                } else if (max == src.get_pixel(i, j, 1)) {
                    h_value = 120 + 60 * (src.get_pixel(i, j, 0) - src.get_pixel(i, j, 2)) / max_minus_min;
                    h_value = h_value < 0 ? h_value + 360 : h_value;
                } else if (max == src.get_pixel(i, j, 0)) {
                    h_value = 240 + 60 * (src.get_pixel(i, j, 2) - src.get_pixel(i, j, 1)) / max_minus_min;
                    h_value = h_value < 0 ? h_value + 360 : h_value;
                }
            }
            dst.set_pixel(i, j, 0, h_value / 2);
        }
    }
}
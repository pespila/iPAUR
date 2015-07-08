#include "image.h"
#include "typeconversion.h"

TypeConversion::TypeConversion(int height, int width, int channel, char type) {
    this->height = height;
    this->width = width;
    this->channel = channel;
    this->type = type;
}
TypeConversion::TypeConversion(Image& src) {
    this->height = src.GetHeight();
    this->width = src.GetWidth();
    this->channel = src.GetChannels();
    this->type = src.GetType();
}
TypeConversion::~TypeConversion() {}

void TypeConversion::rgb2gray(RGBImage& src, GrayscaleImage& dst) {
    dst.Reset(height, width, CV_8UC1);
    for (int i = 0; i < height*width; i++)
        dst.Set(0, i, 0, src.Get(0, i, 2) * 0.299 + src.Get(0, i, 1) * 0.587 + src.Get(0, i, 0) * 0.114);
}

void TypeConversion::gray2rgb(GrayscaleImage& src, RGBImage& dst) {
    dst.Reset(height, width, CV_8UC3);
    for (int k = 0; k < channel; k++) {
        for (int i = 0; i < height*width; i++) {
            dst.Set(0, i, k, src.Get(0, i, 0));
        }
    }
}

void TypeConversion::rgb2ycrcb(RGBImage& src, YCrCbImage& dst) {
    dst.Reset(height, width, CV_8UC3);
    for (int i = 0; i < height*width; i++) {
        dst.Set(0, i, 0, src.Get(0, i, 2) * 0.299 + src.Get(0, i, 1) * 0.587 + src.Get(0, i, 0) * 0.114);
        dst.Set(0, i, 1, (src.Get(0, i, 2) - dst.Get(0, i, 0)) * 0.713 + 128); // delta = 128
        dst.Set(0, i, 2, (src.Get(0, i, 0) - dst.Get(0, i, 0)) * 0.564 + 128); // delta = 128
    }
}

void TypeConversion::rgb2hsi(RGBImage& src, HSIImage& dst) {
    dst.Reset(height, width, CV_8UC3);
    int max, min, max_minus_min, s_value, h_value;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            max = 0;
            min = 255;
            for (int k = 0; k < channel; k++) {
                max = src.Get(i, j, k) <= max ? max : src.Get(i, j, k);
                min = src.Get(i, j, k) >= min ? min : src.Get(i, j, k);
            }
            max_minus_min = max - min;
            dst.Set(i, j, 2, max);
            s_value = max != 0 ? 255 * (max_minus_min) / max : 0;
            dst.Set(i, j, 1, s_value);
            h_value = max == min ? 0 : -1;
            if (h_value == -1) {
                if (max == src.Get(i, j, 2)) {
                    h_value = 60 * (src.Get(i, j, 1) - src.Get(i, j, 0)) / max_minus_min;
                    h_value = h_value < 0 ? h_value + 360 : h_value;
                } else if (max == src.Get(i, j, 1)) {
                    h_value = 120 + 60 * (src.Get(i, j, 0) - src.Get(i, j, 2)) / max_minus_min;
                    h_value = h_value < 0 ? h_value + 360 : h_value;
                } else if (max == src.Get(i, j, 0)) {
                    h_value = 240 + 60 * (src.Get(i, j, 2) - src.Get(i, j, 1)) / max_minus_min;
                    h_value = h_value < 0 ? h_value + 360 : h_value;
                }
            }
            dst.Set(i, j, 0, h_value / 2);
        }
    }
}
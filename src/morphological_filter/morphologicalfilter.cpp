#include "morphologicalfilter.h"

MorphologicalFilter::MorphologicalFilter(int height, int width, int channel, char type) {
    this->height = height;
    this->width = width;
    this->channel = channel;
    this->type = type;
    this->filtered = new unsigned char[this->height*this->width*this->channel];
}

MorphologicalFilter::MorphologicalFilter(Image& src) {
    this->height = src.get_height();
    this->width = src.get_width();
    this->channel = src.get_channels();
    this->type = src.get_type();
    this->filtered = new unsigned char[this->height*this->width*this->channel];
}

MorphologicalFilter::~MorphologicalFilter() {
    delete[] filtered;
}

void MorphologicalFilter::MedianOfArray(unsigned char* kernel, int diameter) {
    int tmp;
    for (int i = diameter - 1; i > 0; --i) {
        for (int j = 0; j < i; ++j) {
            if (kernel[j] > kernel[j + 1]) {
                tmp = kernel[j + 1];
                kernel[j] = kernel[j + 1];
                kernel[j + 1] = tmp;
            }
        }
    }
}

void MorphologicalFilter::CreateOnes(float* kernel, int diameter) {
    for (int i = 0; i < diameter; i++)
        kernel[i] = 1.0;
}

void MorphologicalFilter::FilterDx(Image& src, float* kernel, int radius, char factor) {
    int i, j, k, l, inf;
    for (k = 0; k < channel; k++) {
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                inf = 255;
                for (l = -radius; l <= radius; l++) {
                    if (j + l < 0) {
                        inf = (kernel[l + radius] == 1 && factor * src.get_pixel(i, 0, k) < inf) ? factor * src.get_pixel(i, 0, k) : inf;
                    } else if (j + l >= width) {
                        inf = (kernel[l + radius] == 1 && factor * src.get_pixel(i, width - 1, k) < inf) ? factor * src.get_pixel(i, width - 1, k) : inf;
                    } else {
                        inf = (kernel[l + radius] == 1 && factor * src.get_pixel(i, j + l, k) < inf) ? factor * src.get_pixel(i, j + l, k) : inf;
                    }
                }
                filtered[j + i * width + k * height * width] = factor * inf;
            }
        }
    }
}

void MorphologicalFilter::FilterDy(WriteableImage& dst, float* kernel, int radius, char factor) {
    int i, j, k, l, inf;
    for (k = 0; k < channel; k++) {
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                inf = filtered[j + i * width + k * height * width];
                for (l = -radius; l <= radius; l++) {
                    if (i + l < 0) {
                        inf = (kernel[l + radius] == 1 && factor * filtered[j + k * height * width] < inf) ? factor * filtered[j + k * height * width] : inf;
                    } else if (i + l >= height) {
                        inf = (kernel[l + radius] == 1 && factor * filtered[j + (height - 1) * width + k * height * width] < inf) ? factor * filtered[j + (height - 1) * width + k * height * width] : inf;
                    } else {
                        inf = (kernel[l + radius] == 1 && factor * filtered[j + (i + l) * width + k * height * width] < inf) ? factor * filtered[j + (i + l) * width + k * height * width] : inf;
                    }
                }
                dst.set_pixel(i, j, k, factor * inf);
            }
        }
    }
}

void MorphologicalFilter::Erosion(Image& src, WriteableImage& dst, int radius) {
    dst.reset_image(height, width, type);
    int diameter = 2 * radius + 1;
    float* kernel = new float[diameter];
    CreateOnes(kernel, diameter);
    FilterDx(src, kernel, radius, 1);
    FilterDy(dst, kernel, radius, 1);
    delete[] kernel;
}

void MorphologicalFilter::Dilatation(Image& src, WriteableImage& dst, int radius) {
    dst.reset_image(height, width, type);
    int diameter = 2 * radius + 1;
    float* kernel = new float[diameter];
    CreateOnes(kernel, diameter);
    FilterDx(src, kernel, radius, -1);
    FilterDy(dst, kernel, radius, -1);
    delete[] kernel;
}

void MorphologicalFilter::Median(Image& src, WriteableImage& dst, int radius) {
    dst.reset_image(height, width, type);
    int i, j, k, l, m, x, y;
    int diameter = pow(2 * radius + 1, 2);
    unsigned char* kernel = new unsigned char[diameter];
    for (k = 0; k < channel; k++) {
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                for (l = -radius; l <= radius; l++) {
                    for (m = -radius; m <= radius; m++) {
                        x = i + l >= height ? height - 1 : (i + l < 0 ? 0 : i + l);
                        y = j + m >= width ? width - 1 : (j + m < 0 ? 0 : j + m);
                        kernel[(m + radius) + (l + radius) * (2 * radius + 1)] = src.get_pixel(x, y, k);
                    }
                }
                MedianOfArray(kernel, diameter);
                dst.set_pixel(i, j, k, kernel[(diameter - 1) / 2]);
            }
        }
    }
    delete[] kernel;
}

void MorphologicalFilter::Open(Image& src, WriteableImage& dst, int radius) {
    Dilatation(src, dst, radius);
    Erosion(dst, dst, radius);
}

void MorphologicalFilter::Close(Image& src, WriteableImage& dst, int radius) {
    Erosion(src, dst, radius);
    Dilatation(dst, dst, radius);
}

void MorphologicalFilter::WhiteTopHat(Image& src, WriteableImage& dst, int radius) {
    Open(src, dst, radius);
    int value;
    for (int k = 0; k < channel; k++) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                value = src.get_pixel(i, j, k) - dst.get_pixel(i, j, k) < 0 ? 0 : src.get_pixel(i, j, k) - dst.get_pixel(i, j, k);
                dst.set_pixel(i, j, k, value);
            }
        }
    }
}

void MorphologicalFilter::BlackTopHat(Image& src, WriteableImage& dst, int radius) {
    Close(src, dst, radius);
    int value;
    for (int k = 0; k < channel; k++) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                value = dst.get_pixel(i, j, k) - src.get_pixel(i, j, k) < 0 ? 0 : dst.get_pixel(i, j, k) - src.get_pixel(i, j, k);
                dst.set_pixel(i, j, k, value);
            }
        }
    }
}
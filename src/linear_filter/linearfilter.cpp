#include "linearfilter.h"

LinearFilter::LinearFilter(int height, int width, int channel, char type) {
    this->height = height;
    this->width = width;
    this->channel = channel;
    this->type = type;
    this->filtered = new unsigned char[this->height*this->width*this->channel];
}

LinearFilter::LinearFilter(Image& src) {
    this->height = src.get_height();
    this->width = src.get_width();
    this->channel = src.get_channels();
    this->type = src.get_type();
    this->filtered = new unsigned char[this->height*this->width*this->channel];
}

LinearFilter::~LinearFilter() {
    delete[] filtered;
}

void LinearFilter::CreateGaussFilter(float* kernel, float sigma, int radius, int diameter) {
    int i;
    float sum = 0.0;
    for (i = -radius; i <= radius; i++) {
        kernel[i + radius] = 1.0 / (sqrt(2 * PI) * sigma) * exp(pow(i, 2) * (-1.0) / (2.0 * pow(sigma, 2)));
        sum += kernel[i + radius];
    }
    for (i = 0; i < diameter; i++) {
        kernel[i] /= sum;
    }
}

void LinearFilter::CreateBinomialFilter(float* kernel, int radius, int diameter) {
    int i, j, sum = 0;
    kernel[0] = 1.0;
    for (i = 1; i < diameter; i++){
        kernel[i] = 0.0;
    }
    for (i = 1; i < diameter; i++) {
        for (j = i; j > 0; j--) {
            kernel[j] += kernel[j - 1];
        }
    }
    for (i = 0; i < diameter; i++) {
        sum += kernel[i];
    }
    for (i = 0; i < diameter; i++) {
        kernel[i] /= sum;
    }
}

void LinearFilter::CreateBoxFilter(float* kernel, int radius, int diameter) {
    for (int i = -radius; i <= radius; i++)
        kernel[i + radius] = 1.0 / diameter;
}

void LinearFilter::FilterDx(Image& src, float* kernel, int radius) {
    int i, j, k, l, sum;
    for (k = 0; k < channel; k++) {
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                sum = 0;
                for (l = -radius; l <= radius; l++) {
                    if (j + l < 0) {
                        sum += kernel[l + radius] * src.get_pixel(i, 0, k);
                    } else if (j + l >= width) {
                        sum += kernel[l + radius] * src.get_pixel(i, width - 1, k);
                    } else {
                        sum += kernel[l + radius] * src.get_pixel(i, j + l, k);
                    }
                }
                filtered[j + i * width + k * height * width] = sum;
            }
        }
    }
}

void LinearFilter::FilterDy(WriteableImage& dst, float* kernel, int radius) {
    int i, j, k, l, sum;
    for (k = 0; k < channel; k++) {
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                sum = 0;
                for (l = -radius; l <= radius; l++) {
                    if (i + l < 0) {
                        sum += kernel[l + radius] * filtered[j + k * height * width];
                    } else if (i + l >= height) {
                        sum += kernel[l + radius] * filtered[j + (height - 1) * width + k * height * width];
                    } else {
                        sum += kernel[l + radius] * filtered[j + (i + l) * width + k * height * width];
                    }
                }
                dst.set_pixel(i, j, k, sum);
            }
        }
    }
}

void LinearFilter::Gauss(Image& src, WriteableImage& dst, int radius, float sigma) {
    dst.reset_image(height, width, type);
    int diameter = 2 * radius + 1;
    float* kernel = new float[diameter];
    CreateGaussFilter(kernel, sigma, radius, diameter);
    FilterDx(src, kernel, radius);
    FilterDy(dst, kernel, radius);
    delete[] kernel;
}

void LinearFilter::Binomial(Image& src, WriteableImage& dst, int radius) {
    dst.reset_image(height, width, type);
    int diameter = 2 * radius + 1;
    float* kernel = new float[diameter];
    CreateBinomialFilter(kernel, radius, diameter);
    FilterDx(src, kernel, radius);
    FilterDy(dst, kernel, radius);
    delete[] kernel;
}

void LinearFilter::Box(Image& src, WriteableImage& dst, int radius) {
    dst.reset_image(height, width, type);
    int diameter = 2 * radius + 1;
    float* kernel = new float[diameter];
    CreateBoxFilter(kernel, radius, diameter);
    FilterDx(src, kernel, radius);
    FilterDy(dst, kernel, radius);
    delete[] kernel;
}
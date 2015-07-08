#include "grayscale.h"

using namespace cv;

GrayscaleImage::~GrayscaleImage() {
    free(image);
    image = NULL;
    this->type = 0;
    this->height = 0;
    this->width = 0;
    this->channels = 0;
}

void GrayscaleImage::read_image(const string filename) {
    Mat img = imread(filename, 0); // force gray scale
    this->channels = 1;
    this->width = img.cols;
    this->height = img.rows;
    this->type = img.type();
    this->image = (unsigned char*)malloc(this->width * this->height*sizeof(unsigned char));
    for (int i = 0; i < this->height; i++) {
        for (int j = 0; j < this->width; j++) {
            this->image[j + i * this->width] = img.at<uchar>(i, j);
        }
    }
}

void GrayscaleImage::write_image(const string filename) {
    Mat img(this->height, this->width, this->type);
    for (int i = 0; i < this->height; i++) {
        for (int j = 0; j < this->width; j++) {
            img.at<uchar>(i, j) = this->image[j + i * this->width];
        }
    }
    imwrite(filename, img);
}

void GrayscaleImage::reset_image(unsigned short height, unsigned short width, char type) {
    free(image);
    this->image = NULL;
    this->channels = 1;
    this->height = height;
    this->width = width;
    this->type = type;
    this->image = (unsigned char*)malloc(this->width * this->height*sizeof(unsigned char));
}
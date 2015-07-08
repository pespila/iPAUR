#include "rgba.h"

using namespace cv;

RGBAImage::~RGBAImage() {
    free(image);
    image = NULL;
    this->type = 0;
    this->height = 0;
    this->width = 0;
    this->channels = 0;
}

void RGBAImage::Read(const string filename) {
    Mat img = imread(filename);
    this->channels = 4;
    this->width = img.cols;
    this->height = img.rows;
    this->type = img.type();
    this->image = (unsigned char*)malloc(this->width * this->height * this->channels*sizeof(unsigned char));
    for (int k = 0; k < this->channels; k++) {
        for (int i = 0; i < this->height; i++) {
            for (int j = 0; j < this->width; j++) {
                this->image[j + i * this->width + k * this->height * this->width] = img.at<Vec4b>(i, j)[k];
            }
        }
    }
}

void RGBAImage::Write(const string filename) {
    Mat img(this->height, this->width, this->type);
    for (int k = 0; k < this->channels; k++) {
        for (int i = 0; i < this->height; i++) {
            for (int j = 0; j < this->width; j++) {
                img.at<Vec4b>(i, j)[k] = this->image[j + i * this->width + k * this->height * this->width];
            }
        }
    }
    imwrite(filename, img);
}

void RGBAImage::Reset(unsigned short height, unsigned short width, char type) {
    free(image);
    image = NULL;
    this->channels = 4;
    this->height = height;
    this->width = width;
    this->type = type;
    this->image = (unsigned char*)malloc(height * width * this->channels*sizeof(unsigned char));
}
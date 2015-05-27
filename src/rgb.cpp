#include "rgb.h"

RGBImage::~RGBImage() {
    for (int i = 0; i < this->width * this->height; i++) free(image[i]);
    free(image);

    image = NULL;
    this->type = 0;
    this->height = 0;
    this->width = 0;
}

void RGBImage::read_image(const string filename) {
    Mat img = imread(filename);
    this->channels = 3;
    this->width = img.cols;
    this->height = img.rows;
    this->type = img.type();
    int i, j, k;
    
    this->image = (unsigned char**)malloc(this->width * this->height*sizeof(unsigned char*));
    for (i = 0; i < this->width * this->height; i++) this->image[i] = (unsigned char*)malloc(this->channels*sizeof(unsigned char));

    for (i = 0; i < this->height; i++) {
        for (j = 0; j < this->width; j++) {
            for (k = 0; k < this->channels; k++) {
                this->image[j + i * this->width][k] = img.at<Vec3b>(i, j)[k];
            }
        }
    }
}

void RGBImage::write_image(const string filename) {
    Mat img(this->height, this->width, this->type);
    int i, j, k;
    for (i = 0; i < this->height; i++) {
        for (j = 0; j < this->width; j++) {
            for (k = 0; k < this->channels; k++) {
                img.at<Vec3b>(i, j)[k] = this->image[j + i * this->width][k];
            }
        }
    }
    imwrite(filename, img);
}

void RGBImage::reset_image(unsigned short height, unsigned short width, char type) {
    for (int i = 0; i < this->width * this->height; i++) free(image[i]);
    free(image);
    image = NULL;

    this->channels = 3;
    this->height = height;
    this->width = width;
    this->type = type;

    this->image = (unsigned char**)malloc(height * width*sizeof(unsigned char*));
    for (int i = 0; i < height * width; i++) this->image[i] = (unsigned char*)malloc(this->channels*sizeof(unsigned char));
}
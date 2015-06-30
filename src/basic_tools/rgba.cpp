#include "rgba.h"

RGBAImage::~RGBAImage() {
    for (int i = 0; i < this->width * this->height; i++) free(image[i]);
    free(image);
    image = NULL;

    this->type = 0;
    this->height = 0;
    this->width = 0;
}

void RGBAImage::read_image(const string filename) {
    Mat img = imread(filename, -1);
    this->width = img.cols;
    this->height = img.rows;
    this->type = img.type();
    int i, j, k;

    this->image = (unsigned char**)malloc(this->width * this->height*sizeof(unsigned char*));
    for (i = 0; i < this->width*this->height; i++) this->image[i] = (unsigned char*)malloc(4*sizeof(unsigned char));

    for (i = 0; i < this->height; i++) {
        for (j = 0; j < this->width; j++) {
            for (k = 0; k < 4; k++) {
                this->image[j + i * this->width][k] = img.at<Vec4b>(i, j)[k];
            }
        }
    }
}

void RGBAImage::write_image(const string filename) {
    Mat img(this->height, this->width, this->type);
    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);
    int i, j;
    for (i = 0; i < this->height; i++) {
        for (j = 0; j < this->width; j++) {
            Vec4b& rgba = img.at<Vec4b>(i, j);
            rgba[0] = this->image[j + i * this->width][0];
            rgba[1] = this->image[j + i * this->width][1];
            rgba[2] = this->image[j + i * this->width][2];
            rgba[3] = this->image[j + i * this->width][3];
        }
    }
    imwrite(filename, img, compression_params);
}

void RGBAImage::reset_image(unsigned short height, unsigned short width, char type) {
    for (int i = 0; i < this->width*this->height; i++) free(image[i]);
    free(image);
    image = NULL;

    this->height = height;
    this->width = width;
    this->type = type;
    this->image = (unsigned char**)malloc(height*width*sizeof(unsigned char*));
    for (int i = 0; i < height * width; i++) this->image[i] = (unsigned char*)malloc(4*sizeof(unsigned char));
}
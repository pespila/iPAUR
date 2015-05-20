#include "ycrcb.h"

YCrCbImage::~YCrCbImage() {
    for (int i = 0; i < this->width*this->height; i++) {
        free(image[i]);
    }
    free(image);
    image = NULL;
    this->type = 0;
    this->height = 0;
    this->width = 0;
}
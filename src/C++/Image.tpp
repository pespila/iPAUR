template<typename aType>
Image<aType>::~Image() {
    free(this->image);
}

template<typename aType>
Image<aType>::Image(const string filename, bool gray) {
    Mat img = imread(filename, (gray ? 0 : -1));
    this->channels = img.channels();
    this->width = img.cols;
    this->height = img.rows;
    this->type = img.type();
    this->image = (aType*)malloc(this->width * this->height * this->channels*sizeof(aType));
    if (this->channels == 1) {
        for (int i = 0; i < this->height; i++) {
            for (int j = 0; j < this->width; j++) {
                this->image[j + i * this->width] = img.at<uchar>(i, j)/255.f;
            }
        }
    } else if (this->channels == 3) {
        for (int k = 0; k < this->channels; k++) {
            for (int i = 0; i < this->height; i++) {
                for (int j = 0; j < this->width; j++) {
                    this->image[j + i * this->width + k * this->height * this->width] = img.at<Vec3b>(i, j)[k]/255.f;
                }
            }
        }
    } else if (this->channels == 4) {
        for (int k = 0; k < this->channels; k++) {
            for (int i = 0; i < this->height; i++) {
                for (int j = 0; j < this->width; j++) {
                    this->image[j + i * this->width + k * this->height * this->width] = img.at<Vec4b>(i, j)[k]/255.f;
                }
            }
        }
    }
}

template<typename aType>
void Image<aType>::Read(const string filename, bool gray) {
    Mat img = imread(filename, (gray ? 0 : -1));
    this->channels = img.channels();
    this->width = img.cols;
    this->height = img.rows;
    this->type = img.type();
    this->image = (aType*)malloc(this->width * this->height * this->channels*sizeof(aType));
    if (this->channels == 1) {
        for (int i = 0; i < this->height; i++) {
            for (int j = 0; j < this->width; j++) {
                this->image[j + i * this->width] = img.at<uchar>(i, j)/255.f;
            }
        }
    } else if (this->channels == 3) {
        for (int k = 0; k < this->channels; k++) {
            for (int i = 0; i < this->height; i++) {
                for (int j = 0; j < this->width; j++) {
                    this->image[j + i * this->width + k * this->height * this->width] = img.at<Vec3b>(i, j)[k]/255.f;
                }
            }
        }
    } else if (this->channels == 4) {
        for (int k = 0; k < this->channels; k++) {
            for (int i = 0; i < this->height; i++) {
                for (int j = 0; j < this->width; j++) {
                    this->image[j + i * this->width + k * this->height * this->width] = img.at<Vec4b>(i, j)[k]/255.f;
                }
            }
        }
    }
}

template<typename aType>
void Image<aType>::Write(const string filename) {
    Mat img(this->height, this->width, this->type);
    if (this->channels == 1) {
        for (int i = 0; i < this->height; i++) {
            for (int j = 0; j < this->width; j++) {
                img.at<uchar>(i, j) = this->image[j + i * this->width]*255.f;
            }
        }
    } else if (this-> channels == 3) {
        for (int k = 0; k < this->channels; k++) {
            for (int i = 0; i < this->height; i++) {
                for (int j = 0; j < this->width; j++) {
                    img.at<Vec3b>(i, j)[k] = this->image[j + i * this->width + k * this->height * this->width]*255.f;
                }
            }
        }
    } else if (this-> channels == 4) {
        for (int k = 0; k < this->channels; k++) {
            for (int i = 0; i < this->height; i++) {
                for (int j = 0; j < this->width; j++) {
                    img.at<Vec4b>(i, j)[k] = this->image[j + i * this->width + k * this->height * this->width]*255.f;
                }
            }
        }
    }
    imwrite(filename, img);
}

template<typename aType>
void Image<aType>::Reset(int height, int width, int channels, char type) {
    free(image);
    image = NULL;
    this->channels = channels;
    this->height = height;
    this->width = width;
    this->type = type;
    this->image = (aType*)malloc(height * width * this->channels*sizeof(aType));
}
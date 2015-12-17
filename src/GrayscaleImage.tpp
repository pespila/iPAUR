template<typename aType>
GrayscaleImage<aType>::~GrayscaleImage() {
    free(image);
    image = NULL;
    this->type = 0;
    this->height = 0;
    this->width = 0;
    this->channels = 0;
}

template<typename aType>
void GrayscaleImage<aType>::Read(const string filename) {
    Mat img = imread(filename, 0); // force gray scale
    this->channels = 1;
    this->width = img.cols;
    this->height = img.rows;
    this->type = img.type();
    this->image = (aType*)malloc(this->width * this->height*sizeof(aType));
    for (int i = 0; i < this->height; i++) {
        for (int j = 0; j < this->width; j++) {
            this->image[j + i * this->width] = img.at<uchar>(i, j);
        }
    }
}

template<typename aType>
void GrayscaleImage<aType>::Write(const string filename) {
    Mat img(this->height, this->width, this->type);
    for (int i = 0; i < this->height; i++) {
        for (int j = 0; j < this->width; j++) {
            img.at<uchar>(i, j) = this->image[j + i * this->width];
        }
    }
    imwrite(filename, img);
}

template<typename aType>
void GrayscaleImage<aType>::Reset(int height, int width, char type) {
    free(image);
    this->image = NULL;
    this->channels = 1;
    this->height = height;
    this->width = width;
    this->type = type;
    this->image = (aType*)malloc(this->width * this->height*sizeof(aType));
}
template<typename aType>
RGBImage<aType>::~RGBImage() {
    free(this->image);
    this->image = NULL;
    this->type = 0;
    this->height = 0;
    this->width = 0;
    this->channels = 0;
}

template<typename aType>
void RGBImage<aType>::Read(const string filename) {
    Mat img = imread(filename);
    this->channels = 3;
    this->width = img.cols;
    this->height = img.rows;
    this->type = img.type();
    this->image = (aType*)malloc(this->width * this->height * this->channels*sizeof(aType));
    for (int k = 0; k < this->channels; k++) {
        for (int i = 0; i < this->height; i++) {
            for (int j = 0; j < this->width; j++) {
                this->image[j + i * this->width + k * this->height * this->width] = img.at<Vec3b>(i, j)[k];
            }
        }
    }
}

template<typename aType>
void RGBImage<aType>::Write(const string filename) {
    Mat img(this->height, this->width, this->type);
    for (int k = 0; k < this->channels; k++) {
        for (int i = 0; i < this->height; i++) {
            for (int j = 0; j < this->width; j++) {
                img.at<Vec3b>(i, j)[k] = this->image[j + i * this->width + k * this->height * this->width];
            }
        }
    }
    imwrite(filename, img);
}

template<typename aType>
void RGBImage<aType>::Reset(int height, int width, char type) {
    free(image);
    image = NULL;
    this->channels = 3;
    this->height = height;
    this->width = width;
    this->type = type;
    this->image = (aType*)malloc(height * width * this->channels*sizeof(aType));
}
template<typename aType>
LinearFilter<aType>::LinearFilter(Image<aType>& src) {
    this->height = src.Height();
    this->width = src.Width();
    this->channel = src.Channels();
    this->type = src.Type();
    this->filtered = new aType[this->height*this->width*this->channel];
}

template<typename aType>
LinearFilter<aType>::~LinearFilter() {
    delete[] filtered;
}

template<typename aType>
void LinearFilter<aType>::CreateGaussFilter(aType* kernel, aType sigma, int radius, int diameter) {
    int i;
    aType sum = 0.0;
    for (i = -radius; i <= radius; i++) {
        kernel[i + radius] = 1.0 / (sqrt(2 * PI) * sigma) * exp(pow(i, 2) * (-1.0) / (2.0 * pow(sigma, 2)));
        sum += kernel[i + radius];
    }
    for (i = 0; i < diameter; i++) {
        kernel[i] /= sum;
    }
}

template<typename aType>
void LinearFilter<aType>::CreateBinomialFilter(aType* kernel, int radius, int diameter) {
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

template<typename aType>
void LinearFilter<aType>::CreateBoxFilter(aType* kernel, int radius, int diameter) {
    for (int i = -radius; i <= radius; i++)
        kernel[i + radius] = 1.0 / diameter;
}

template<typename aType>
void LinearFilter<aType>::FilterDx(Image<aType>& src, aType* kernel, int radius) {
    int i, j, k, l, sum;
    for (k = 0; k < channel; k++) {
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                sum = 0;
                for (l = -radius; l <= radius; l++) {
                    if (j + l < 0) {
                        sum += kernel[l + radius] * src.Get(i, 0, k);
                    } else if (j + l >= width) {
                        sum += kernel[l + radius] * src.Get(i, width - 1, k);
                    } else {
                        sum += kernel[l + radius] * src.Get(i, j + l, k);
                    }
                }
                filtered[j + i * width + k * height * width] = sum;
            }
        }
    }
}

template<typename aType>
void LinearFilter<aType>::FilterDy(Image<aType>& dst, aType* kernel, int radius) {
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
                dst.Set(i, j, k, sum);
            }
        }
    }
}

template<typename aType>
void LinearFilter<aType>::Gauss(Image<aType>& src, Image<aType>& dst, int radius, aType sigma) {
    dst.Reset(src.Height(), src.Width(), src.Channels(), src.Type());
    int diameter = 2 * radius + 1;
    aType* kernel = new aType[diameter];
    CreateGaussFilter(kernel, sigma, radius, diameter);
    FilterDx(src, kernel, radius);
    FilterDy(dst, kernel, radius);
    delete[] kernel;
}

template<typename aType>
void LinearFilter<aType>::Binomial(Image<aType>& src, Image<aType>& dst, int radius) {
    dst.Reset(src.Height(), src.Width(), src.Channels(), src.Type());
    int diameter = 2 * radius + 1;
    aType* kernel = new aType[diameter];
    CreateBinomialFilter(kernel, radius, diameter);
    FilterDx(src, kernel, radius);
    FilterDy(dst, kernel, radius);
    delete[] kernel;
}

template<typename aType>
void LinearFilter<aType>::Box(Image<aType>& src, Image<aType>& dst, int radius) {
    dst.Reset(src.Height(), src.Width(), src.Channels(), src.Type());
    int diameter = 2 * radius + 1;
    aType* kernel = new aType[diameter];
    CreateBoxFilter(kernel, radius, diameter);
    FilterDx(src, kernel, radius);
    FilterDy(dst, kernel, radius);
    delete[] kernel;
}

template<typename aType>
void LinearFilter<aType>::Duto(Image<aType>& src, Image<aType>& dst, int radius, aType sigma, aType lambda) {
    dst.Reset(src.Height(), src.Width(), src.Channels(), src.Type());
    Gauss(src, dst, radius, sigma);
    // for (int k = 0; k < channel; k++)
    //     for (int i = 0; i < height; i++)
    //         for (int j = 0; j < width; j++)
    //             dst.Set(i, j, k, (1 - lambda) * src.Get(i, j, k));

    for (int k = 0; k < channel; k++)
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                dst.Set(i, j, k, lambda * src.Get(i, j, k) + (1 - lambda) * dst.Get(i, j, k));
}
template<typename aType>
EdgeDetector<aType>::EdgeDetector(Image<aType>& src) {
    this->height = src.Height();
    this->width = src.Width();
    this->type = src.Type();
    this->gx = new aType[this->height*this->width];
    this->gy = new aType[this->height*this->width];
}

template<typename aType>
EdgeDetector<aType>::~EdgeDetector() {
    delete[] gx;
    delete[] gy;
}

template<typename aType>
void EdgeDetector<aType>::NablaX(aType* gx, Image<aType>& src, int normalizer) {
    int i, j, x, y, which;
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            x = j + 1 >= width ? width - 1: j + 1;
            y = j - 1 < 0 ? 0 : j - 1;
            gx[j + i * width] = (src.Get(i, y, 0) - src.Get(i, x, 0)) / 2;
        }
    }
    for (i = 0; i < height; i++) {
        x = i + 1 >= height ? height - 1 : i + 1;
        y = i - 1 < 0 ? 0 : i - 1;
        for (j = 0; j < width; j++) {
            which = normalizer == 4 ? gx[j + i * width] * 2 : gx[j + i * width] * 3;
            gx[j + i * width] = (gx[j + y * width] + which + gx[j + x * width]) / normalizer;
        }
    }
}

template<typename aType>
void EdgeDetector<aType>::NablaY(aType* gy, Image<aType>& src, int normalizer) {
    int i, j, x, y, which;
    for (i = 0; i < height; i++) {
        x = i + 1 >= height ? height - 1 : i + 1;
        y = i - 1 < 0 ? 0 : i - 1;
        for (j = 0; j < width; j++) {
            gy[j + i * width] = (src.Get(y, j, 0) - src.Get(x, j, 0)) / 2;
        }
    }
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            x = j + 1 >= width ? width - 1: j + 1;
            y = j - 1 < 0 ? 0 : j - 1;
            which = normalizer == 4 ? gy[j + i * width] * 2 : gy[j + i * width] * 3;
            gy[j + i * width] = (gy[y + i * width] + which + gy[x + i * width]) / normalizer;
        }
    }
}

template<typename aType>
void EdgeDetector<aType>::EvalGradient(aType* gradient_angle, aType* gx, aType* gy) {
    aType angle = 0;
    for (int i = 0; i < height*width; i++) {
        angle = (aType)(atan2(gy[i], gx[i]) * 180 / PI);
        angle = angle > 255 ? 360 - angle : angle;
        angle = (angle >= 0 && angle < 23 ) || ( angle >= 158 && angle <= 180) ? 0 : angle;
        angle = angle >= 23 && angle < 68 ? 45 : angle;
        angle = angle >= 68 && angle < 123 ? 90 : angle;
        angle = angle >= 113 && angle < 158 ? 135 : angle;
        gradient_angle[i] = angle;
    }
}

template<typename aType>
void EdgeDetector<aType>::NonMaximumSupression(aType* nms, aType* sobel, aType* gradient_angle) {
    int i, j, v, w, x, y, angle;
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            v = i + 1 >= height ? height - 2 : i + 1;
            w = i - 1 < 0 ? 1 : i - 1;
            x = j + 1 >= width ? width - 2 : j + 1;
            y = j - 1 < 0 ? 1 : j - 1;
            angle = gradient_angle[j + i * width];
            if (angle == 0) {
                if (sobel[j + i * width] >= sobel[y + i * width] && sobel[j + i * width] >= sobel[x + i * width]) {
                    nms[j + i * width] = sobel[j + i * width];
                } else {
                    nms[j + i * width] = 0;
                }
            } else if (angle == 45) {
                if (sobel[j + i * width] >= sobel[x + w * width] && sobel[j + i * width] >= sobel[y + v * width]) {
                    nms[j + i * width] = sobel[j + i * width];
                } else {
                    nms[j + i * width] = 0;
                }
            } else if (angle == 90) {
                if (sobel[j + i * width] >= sobel[j + v * width] && sobel[j + i * width] >= sobel[j + w * width]) {
                    nms[j + i * width] = sobel[j + i * width];
                } else {
                    nms[j + i * width] = 0;
                }
            } else if (angle == 135) {
                if (sobel[j + i * width] >= sobel[x + v * width] && sobel[j + i * width] >= sobel[y + w * width]) {
                    nms[j + i * width] = sobel[j + i * width];
                } else {
                    nms[j + i * width] = 0;
                }
            } else {
                nms[j + i * width] = sobel[j + i * width];
            }
        }
    }
}

template<typename aType>
void EdgeDetector<aType>::Hysteresis(Image<aType>& dst, aType* nms, const int TL, const int TH) {
    int i, j, k, l, x, y;
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            if (nms[j + i * width] > TH) {
                dst.Set(i, j, 0, 255);
            } else {
                for (k = -1; k <= 1; k++) {
                    for (l = -1; l <= 1; l++) {
                        x = i + k < 0 ? 0 : (i + k >= height ? height - 1 : i + k);
                        y = j + l < 0 ? 0 : (j + l >= width ? width - 1 : j + l);
                        if (nms[y + x * width] > TL) {
                            dst.Set(x, y, 0, 255);
                        } else {
                            dst.Set(x, y, 0, 0);
                        }
                    }
                }
            }
        }
    }
}

template<typename aType>
void EdgeDetector<aType>::SetEdges(Image<aType>& dst, aType* gx, aType* gy) {
    int i, j, sum = 0;
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            sum = abs(gx[j + i * width]) + abs(gy[j + i * width]);
            sum = sum > 255 ? 255 : sum;
            dst.Set(i, j, 0, sum);
        }
    }
}

template<typename aType>
void EdgeDetector<aType>::Sobel(Image<aType>& src, Image<aType>& dst) {
    dst.Reset(src.Height(), src.Width(), src.Channels(), src.Type());
    NablaX(gx, src, 4);
    NablaY(gy, src, 4);
    SetEdges(dst, gx, gy);
}

template<typename aType>
void EdgeDetector<aType>::Prewitt(Image<aType>& src, Image<aType>& dst) {
    dst.Reset(src.Height(), src.Width(), src.Channels(), src.Type());
    NablaX(gx, src, 3);
    NablaY(gy, src, 3);
    SetEdges(dst, gx, gy);
}

template<typename aType>
void EdgeDetector<aType>::RobertsCross(Image<aType>& src, Image<aType>& dst) {
    dst.Reset(src.Height(), src.Width(), src.Channels(), src.Type());
    int i, j, x, y, dx, dy, sum;
    for (i = 0; i < height; i++) {
        y = i + 1 >= height ? height - 1 : i + 1;
        for (j = 0; j < width; j++) {
            x = j + 1 >= width ? width - 1 : j + 1;
            dx = abs(src.Get(i, j, 0) - src.Get(y, x, 0));
            dy = abs(src.Get(y, j, 0) - src.Get(i, x, 0));
            sum = dx + dy;
            sum = sum > 255 ? 255 : sum;
            dst.Set(i, j, 0, sum);
        }
    }
}

template<typename aType>
void EdgeDetector<aType>::Laplace(Image<aType>& src, Image<aType>& dst) {
    dst.Reset(src.Height(), src.Width(), src.Channels(), src.Type());
    int i, j, v, w, x, y, sum;
    for (i = 0; i < height; i++) {
        v = i + 1 >= height ? height - 1 : i + 1;
        w = i - 1 < 0 ? 0 : i - 1;
        for (j = 0; j < width; j++) {
            x = j + 1 >= width ? width - 1 : j + 1;
            y = j - 1 < 0 ? 0 : j - 1;
            sum = (src.Get(w, j, 0) + src.Get(v, j, 0) + src.Get(i, x, 0) + src.Get(i, y, 0) - 4 * src.Get(i, j, 0)) / 8;
            sum = fabs(sum) > 5 ? 255 : 0;
            dst.Set(i, j, 0, sum);
        }
    }
}

template<typename aType>
void EdgeDetector<aType>::Canny(Image<aType>& src, Image<aType>& dst, const int TL, const int TH) {
    dst.Reset(src.Height(), src.Width(), src.Channels(), src.Type());
    int i;
    aType* gradient_angle = new aType[this->height*this->width];
    aType* sobel = new aType[this->height*this->width];
    aType* nms = new aType[this->height*this->width];
    NablaX(gx, src, 4);
    NablaY(gy, src, 4);
    for (i = 0; i < height*width; i++)
        sobel[i] = abs(gx[i]) + abs(gy[i]);
    EvalGradient(gradient_angle, gx, gy);
    NonMaximumSupression(nms, sobel, gradient_angle);
    Hysteresis(dst, nms, TL, TH);
}
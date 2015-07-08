#include "image.h"
#include "edgedetector.h"

EdgeDetector::EdgeDetector(int height, int width, char type) {
    this->height = height;
    this->width = width;
    this->type = type;
    this->gx = new short[height*width];
    this->gy = new short[height*width];
}
EdgeDetector::EdgeDetector(Image& src) {
    this->height = src.get_height();
    this->width = src.get_width();
    this->type = src.get_type();
    this->gx = new short[this->height*this->width];
    this->gy = new short[this->height*this->width];
}
EdgeDetector::~EdgeDetector() {
    delete[] gx;
    delete[] gy;
}

void EdgeDetector::NablaX(short* gx, Image& src, unsigned char normalizer) {
    int i, j, x, y, which;
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            x = j + 1 >= width ? width - 1: j + 1;
            y = j - 1 < 0 ? 0 : j - 1;
            gx[j + i * width] = (src.get_pixel(i, y, 0) - src.get_pixel(i, x, 0)) / 2;
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

void EdgeDetector::NablaY(short* gy, Image& src, unsigned char normalizer) {
    int i, j, x, y, which;
    for (i = 0; i < height; i++) {
        x = i + 1 >= height ? height - 1 : i + 1;
        y = i - 1 < 0 ? 0 : i - 1;
        for (j = 0; j < width; j++) {
            gy[j + i * width] = (src.get_pixel(y, j, 0) - src.get_pixel(x, j, 0)) / 2;
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

void EdgeDetector::EvalGradient(short* gradient_angle, short* gx, short* gy) {
    short angle = 0;
    for (int i = 0; i < height*width; i++) {
        angle = (short)(atan2(gy[i], gx[i]) * 180 / PI);
        angle = angle > 255 ? 360 - angle : angle;
        angle = (angle >= 0 && angle < 23 ) || ( angle >= 158 && angle <= 180) ? 0 : angle;
        angle = angle >= 23 && angle < 68 ? 45 : angle;
        angle = angle >= 68 && angle < 123 ? 90 : angle;
        angle = angle >= 113 && angle < 158 ? 135 : angle;
        gradient_angle[i] = angle;
    }
}

void EdgeDetector::NonMaximumSupression(short* nms, short* sobel, short* gradient_angle) {
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

void EdgeDetector::Hysteresis(WriteableImage& dst, short* nms, const int TL, const int TH) {
    int i, j, k, l, x, y;
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            if (nms[j + i * width] > TH) {
                dst.set_pixel(i, j, 0, 255);
            } else {
                for (k = -1; k <= 1; k++) {
                    for (l = -1; l <= 1; l++) {
                        x = i + k < 0 ? 0 : (i + k >= height ? height - 1 : i + k);
                        y = j + l < 0 ? 0 : (j + l >= width ? width - 1 : j + l);
                        if (nms[y + x * width] > TL) {
                            dst.set_pixel(x, y, 0, 255);
                        } else {
                            dst.set_pixel(x, y, 0, 0);
                        }
                    }
                }
            }
        }
    }
}

void EdgeDetector::SetEdges(WriteableImage& dst, short* gx, short* gy) {
    int i, j, sum = 0;
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            sum = abs(gx[j + i * width]) + abs(gy[j + i * width]);
            // sum = sum > 100 ? 255 : 0;
            dst.set_pixel(i, j, 0, sum);
        }
    }
}

void EdgeDetector::Sobel(Image& src, WriteableImage& dst) {
    dst.reset_image(this->height, this->width, this->type);
    NablaX(gx, src, 4);
    NablaY(gy, src, 4);
    SetEdges(dst, gx, gy);
}

void EdgeDetector::Prewitt(Image& src, WriteableImage& dst) {
    dst.reset_image(this->height, this->width, this->type);
    NablaX(gx, src, 3);
    NablaY(gy, src, 3);
    SetEdges(dst, gx, gy);
}

void EdgeDetector::RobertsCross(Image& src, WriteableImage& dst) {
    dst.reset_image(this->height, this->width, this->type);
    int i, j, x, y, dx, dy, sum;
    for (i = 0; i < height; i++) {
        y = i + 1 >= height ? height - 1 : i + 1;
        for (j = 0; j < width; j++) {
            x = j + 1 >= width ? width - 1 : j + 1;
            dx = abs(src.get_pixel(i, j, 0) - src.get_pixel(y, x, 0));
            dy = abs(src.get_pixel(y, j, 0) - src.get_pixel(i, x, 0));
            sum = dx + dy > 255 ? 255 : dx + dy;
            dst.set_pixel(i, j, 0, sum);
        }
    }
}

void EdgeDetector::Laplace(Image& src, WriteableImage& dst) {
    dst.reset_image(this->height, this->width, this->type);
    int i, j, v, w, x, y, sum;
    for (i = 0; i < height; i++) {
        v = i + 1 >= height ? height - 1 : i + 1;
        w = i - 1 < 0 ? 0 : i - 1;
        for (j = 0; j < width; j++) {
            x = j + 1 >= width ? width - 1 : j + 1;
            y = j - 1 < 0 ? 0 : j - 1;
            sum = (src.get_pixel(w, j, 0) + src.get_pixel(v, j, 0) + src.get_pixel(i, x, 0) + src.get_pixel(i, y, 0) - 4 * src.get_pixel(i, j, 0)) / 8;
            sum = sum > 10 ? 255 : 0;
            dst.set_pixel(i, j, 0, sum);
        }
    }
}

void EdgeDetector::Canny(Image& src, WriteableImage& dst, const int TL, const int TH) {
    dst.reset_image(this->height, this->width, this->type);
    int i;
    short* gradient_angle = new short[this->height*this->width];
    short* sobel = new short[this->height*this->width];
    short* nms = new short[this->height*this->width];
    NablaX(gx, src, 4);
    NablaY(gy, src, 4);
    for (i = 0; i < height*width; i++)
        sobel[i] = abs(gx[i]) + abs(gy[i]);
    EvalGradient(gradient_angle, gx, gy);
    NonMaximumSupression(nms, sobel, gradient_angle);
    Hysteresis(dst, nms, TL, TH);
}
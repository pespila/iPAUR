#include <math.h>
#include <Rcpp.h>

using namespace Rcpp;
using namespace std;

#ifndef __EDGEDETECTOR_H__
#define __EDGEDETECTOR_H__

template<typename aType>
class EdgeDetector
{
private:
	int height;
	int width;
	int channel;
	aType* gx;
	aType* gy;

	void NablaX(aType*, const NumericMatrix&, int);
	void NablaY(aType*, const NumericMatrix&, int);
	void EvalGradient(aType*, aType*, aType*);
	void NonMaximumSupression(aType*, aType*, aType*);
	void Hysteresis(NumericMatrix&, aType*, const int, const int);
	void SetEdges(NumericMatrix&, aType*, aType*);
public:
	EdgeDetector():height(0), width(0), channel(0), gx(NULL), gy(NULL) {}
	EdgeDetector(int, int, int);
	EdgeDetector(const NumericMatrix&, int, int, int);
	~EdgeDetector();

	void Sobel(const NumericMatrix&, NumericMatrix&);
	void Prewitt(const NumericMatrix&, NumericMatrix&);
	void RobertsCross(const NumericMatrix&, NumericMatrix&);
	void Laplace(const NumericMatrix&, NumericMatrix&);
	void Canny(const NumericMatrix&, NumericMatrix&, const int, const int);
};

template<typename aType>
EdgeDetector<aType>::EdgeDetector(int height, int width, int channel) {
    this->height = height;
    this->width = width;
    this->channel = channel;
    this->gx = new aType[height*width];
    this->gy = new aType[height*width];
}

template<typename aType>
EdgeDetector<aType>::EdgeDetector(const NumericMatrix& src, int height, int width, int channel) {
    this->height = height;
    this->width = width;
    this->channel = channel;
    this->gx = new aType[this->height*this->width];
    this->gy = new aType[this->height*this->width];
}

template<typename aType>
EdgeDetector<aType>::~EdgeDetector() {
    delete[] gx;
    delete[] gy;
}

template<typename aType>
void EdgeDetector<aType>::NablaX(aType* gx, const NumericMatrix& src, int normalizer) {
    int i, j, x, y, which;
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            x = j + 1 >= width ? width - 1: j + 1;
            y = j - 1 < 0 ? 0 : j - 1;
            gx[j + i * width] = (src(i, y) - src(i, x)) / 2;
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
void EdgeDetector<aType>::NablaY(aType* gy, const NumericMatrix& src, int normalizer) {
    int i, j, x, y, which;
    for (i = 0; i < height; i++) {
        x = i + 1 >= height ? height - 1 : i + 1;
        y = i - 1 < 0 ? 0 : i - 1;
        for (j = 0; j < width; j++) {
            gy[j + i * width] = (src(y, j) - src(x, j)) / 2;
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
void EdgeDetector<aType>::Hysteresis(NumericMatrix& dst, aType* nms, const int TL, const int TH) {
    int i, j, k, l, x, y;
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            if (nms[j + i * width] > TH) {
            	dst(i, j) = 255;
            } else {
                for (k = -1; k <= 1; k++) {
                    for (l = -1; l <= 1; l++) {
                        x = i + k < 0 ? 0 : (i + k >= height ? height - 1 : i + k);
                        y = j + l < 0 ? 0 : (j + l >= width ? width - 1 : j + l);
                        if (nms[y + x * width] > TL) {
                        	dst(i, j) = 255;
                        } else {
                        	dst(i, j) = 0;
                        }
                    }
                }
            }
        }
    }
}

template<typename aType>
void EdgeDetector<aType>::SetEdges(NumericMatrix& dst, aType* gx, aType* gy) {
    int i, j, sum = 0;
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            sum = abs(gx[j + i * width]) + abs(gy[j + i * width]);
            sum = sum > 10 ? 255 : 0;
            dst(i, j) = sum;
        }
    }
}

template<typename aType>
void EdgeDetector<aType>::Sobel(const NumericMatrix& src, NumericMatrix& dst) {
    NablaX(gx, src, 4);
    NablaY(gy, src, 4);
    SetEdges(dst, gx, gy);
}

template<typename aType>
void EdgeDetector<aType>::Prewitt(const NumericMatrix& src, NumericMatrix& dst) {
    NablaX(gx, src, 3);
    NablaY(gy, src, 3);
    SetEdges(dst, gx, gy);
}

template<typename aType>
void EdgeDetector<aType>::RobertsCross(const NumericMatrix& src, NumericMatrix& dst) {
    int i, j, x, y, dx, dy, sum;
    for (i = 0; i < height; i++) {
        y = i + 1 >= height ? height - 1 : i + 1;
        for (j = 0; j < width; j++) {
            x = j + 1 >= width ? width - 1 : j + 1;
            dx = abs(src(i, j) - src(y, x));
            dy = abs(src(y, j) - src(i, x));
            sum = dx + dy > 255 ? 255 : dx + dy;
            sum = sum > 10 ? 255 : 0;
            dst(i, j) = sum;
        }
    }
}

template<typename aType>
void EdgeDetector<aType>::Laplace(const NumericMatrix& src, NumericMatrix& dst) {
    int i, j, v, w, x, y, sum;
    for (i = 0; i < height; i++) {
        v = i + 1 >= height ? height - 1 : i + 1;
        w = i - 1 < 0 ? 0 : i - 1;
        for (j = 0; j < width; j++) {
            x = j + 1 >= width ? width - 1 : j + 1;
            y = j - 1 < 0 ? 0 : j - 1;
            sum = (src(w, j) + src(v, j) + src(i, x) + src(i, y) - 4 * src(i, j)) / 8;
            sum = sum > 10 ? 255 : 0;
            dst(i, j) = sum;
        }
    }
}

template<typename aType>
void EdgeDetector<aType>::Canny(const NumericMatrix& src, NumericMatrix& dst, const int TL, const int TH) {
    aType* gradient_angle = new aType[this->height*this->width];
    aType* sobel = new aType[this->height*this->width];
    aType* nms = new aType[this->height*this->width];
    NablaX(gx, src, 4);
    NablaY(gy, src, 4);
    for (int i = 0; i < height*width; i++)
        sobel[i] = abs(gx[i]) + abs(gy[i]);
    EvalGradient(gradient_angle, gx, gy);
    NonMaximumSupression(nms, sobel, gradient_angle);
    Hysteresis(dst, nms, TL, TH);
}

#endif //__EDGEDETECTOR_H__
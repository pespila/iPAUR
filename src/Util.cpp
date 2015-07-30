#include "Util.h"

Util::Util(int height, int width, int channel, char type) {
    this->height = height;
    this->width = width;
    this->channel = channel;
    this->type = type;
}

Util::Util(Image& src) {
    this->height = src.GetHeight();
    this->width = src.GetWidth();
    this->channel = src.GetChannels();
    this->type = src.GetType();
}

Util::~Util() {}

void Util::MarkRed(RGBImage& src, GrayscaleImage& edges, RGBImage& dst) {
    dst.Reset(this->height, this->width, this->type);
    if (src.GetHeight() != edges.GetHeight() || src.GetWidth() != edges.GetWidth()) {
        printf("Height, width and number of channels do not match!\n");
    } else {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if (edges.Get(i, j, 0) > 50) {
                    dst.Set(i, j, 0, 0);
                    dst.Set(i, j, 1, 0);
                    dst.Set(i, j, 2, 255);
                } else {
                    for (int k = 0; k < channel; k++) {
                        dst.Set(i, j, k, src.Get(i, j, k));
                    }
                }
            }
        }
    }
}

void Util::AddImages(Image& src1, Image& src2, WriteableImage& dst) {
    dst.Reset(this->height, this->width, this->type);
    if (src1.GetHeight() != src2.GetHeight() || src1.GetWidth() != src2.GetWidth() || src1.GetChannels() != src2.GetChannels()) {
        printf("Height, width and number of channels do not match!\n");
    } else {
        int value;
        for (int k = 0; k < channel; k++) {
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    value = src1.Get(i, j, k) + src2.Get(i, j, k) > 255 ? 255 : src1.Get(i, j, k) + src2.Get(i, j, k);
                    dst.Set(i, j, k, value);
                }
            }
        }
    }
}

void Util::InverseImage(Image& src, WriteableImage& dst) {
    dst.Reset(this->height, this->width, this->type);
    for (int k = 0; k < channel; k++)
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                dst.Set(i, j, k, 255 - src.Get(i, j, k));
}
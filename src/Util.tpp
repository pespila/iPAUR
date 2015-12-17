template<typename aType>
void Util<aType>::MarkRed(RGBImage<aType>& src, GrayscaleImage<aType>& edges, RGBImage<aType>& dst) {
    dst.Reset(src.GetHeight(), src.GetWidth(), src.GetType());
    if (src.GetHeight() != edges.GetHeight() || src.GetWidth() != edges.GetWidth()) {
        printf("Height, width and number of channels do not match!\n");
    } else {
        for (int i = 0; i < src.GetHeight(); i++) {
            for (int j = 0; j < src.GetWidth(); j++) {
                if (edges.Get(i, j, 0) > 50) {
                    dst.Set(i, j, 0, 0);
                    dst.Set(i, j, 1, 0);
                    dst.Set(i, j, 2, 255);
                } else {
                    for (int k = 0; k < src.GetChannels(); k++) {
                        dst.Set(i, j, k, src.Get(i, j, k));
                    }
                }
            }
        }
    }
}

template<typename aType>
void Util<aType>::AddImages(Image<aType>& src1, Image<aType>& src2, WriteableImage<aType>& dst) {
    dst.Reset(src1.GetHeight(), src1.GetWidth(), src1.GetType());
    if (src1.GetHeight() != src2.GetHeight() || src1.GetWidth() != src2.GetWidth() || src1.GetChannels() != src2.GetChannels()) {
        printf("Height, width and number of channels do not match!\n");
    } else {
        int value;
        for (int k = 0; k < src1.GetChannels(); k++) {
            for (int i = 0; i < src1.GetHeight(); i++) {
                for (int j = 0; j < src1.GetWidth(); j++) {
                    value = src1.Get(i, j, k) + src2.Get(i, j, k) > 255 ? 255 : src1.Get(i, j, k) + src2.Get(i, j, k);
                    dst.Set(i, j, k, value);
                }
            }
        }
    }
}

template<typename aType>
void Util<aType>::InverseImage(Image<aType>& src, WriteableImage<aType>& dst) {
    dst.Reset(src.GetHeight(), src.GetWidth(), src.GetType());
    for (int k = 0; k < src.GetChannels(); k++)
        for (int i = 0; i < src.GetHeight(); i++)
            for (int j = 0; j < src.GetWidth(); j++)
                dst.Set(i, j, k, 255 - src.Get(i, j, k));
}
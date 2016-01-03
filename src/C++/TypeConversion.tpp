template<typename aType>
void TypeConversion<aType>::RGB2Gray(Image<aType>& src, Image<aType>& dst) {
    dst.Reset(src.Height(), src.Width(), 1, CV_8UC1);
    for (int i = 0; i < src.Height()*src.Width(); i++)
        dst.Set(0, i, 0, src.Get(0, i, 2) * 0.299 + src.Get(0, i, 1) * 0.587 + src.Get(0, i, 0) * 0.114);
}

template<typename aType>
void TypeConversion<aType>::Gray2RGB(Image<aType>& src, Image<aType>& dst) {
    dst.Reset(src.Height(), src.Width(), 3, CV_8UC3);
    for (int k = 0; k < 3; k++) {
        for (int i = 0; i < src.Height(); i++) {
            for (int j = 0; j < src.Width(); j++) {
                dst.Set(i, j, k, src.Get(i, j, 0));
            }
        }
    }
}

template<typename aType>
void TypeConversion<aType>::RGB2YCrCb(Image<aType>& src, Image<aType>& dst) {
    dst.Reset(src.Height(), src.Width(), src.Channels(), src.Type());
    for (int i = 0; i < src.Height()*src.Width(); i++) {
        dst.Set(0, i, 0, src.Get(0, i, 2) * 0.299 + src.Get(0, i, 1) * 0.587 + src.Get(0, i, 0) * 0.114);
        dst.Set(0, i, 1, (src.Get(0, i, 2) - dst.Get(0, i, 0)) * 0.713 + 128); // delta = 128
        dst.Set(0, i, 2, (src.Get(0, i, 0) - dst.Get(0, i, 0)) * 0.564 + 128); // delta = 128
    }
}

template<typename aType>
void TypeConversion<aType>::RGB2HSI(Image<aType>& src, Image<aType>& dst) {
    dst.Reset(src.Height(), src.Width(), src.Channels(), src.Type());
    int max, min, max_minus_min, s_value, h_value;
    for (int i = 0; i < src.Height(); i++) {
        for (int j = 0; j < src.Width(); j++) {
            max = 0;
            min = 255;
            for (int k = 0; k < src.Channels(); k++) {
                max = src.Get(i, j, k) <= max ? max : src.Get(i, j, k);
                min = src.Get(i, j, k) >= min ? min : src.Get(i, j, k);
            }
            max_minus_min = max - min;
            dst.Set(i, j, 2, max);
            s_value = max != 0 ? 255 * (max_minus_min) / max : 0;
            dst.Set(i, j, 1, s_value);
            h_value = max == min ? 0 : -1;
            if (h_value == -1) {
                if (max == src.Get(i, j, 2)) {
                    h_value = 60 * (src.Get(i, j, 1) - src.Get(i, j, 0)) / max_minus_min;
                    h_value = h_value < 0 ? h_value + 360 : h_value;
                } else if (max == src.Get(i, j, 1)) {
                    h_value = 120 + 60 * (src.Get(i, j, 0) - src.Get(i, j, 2)) / max_minus_min;
                    h_value = h_value < 0 ? h_value + 360 : h_value;
                } else if (max == src.Get(i, j, 0)) {
                    h_value = 240 + 60 * (src.Get(i, j, 2) - src.Get(i, j, 1)) / max_minus_min;
                    h_value = h_value < 0 ? h_value + 360 : h_value;
                }
            }
            dst.Set(i, j, 0, h_value / 2);
        }
    }
}
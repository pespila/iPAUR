// #include "Util.h"

// #ifndef __MORPHOLOGICALFILTER_H__
// #define __MORPHOLOGICALFILTER_H__

// template<typename aType>
// class MorphologicalFilter
// {
// private:
// 	int height;
// 	int width;
// 	int channel;
// 	char type;
// 	aType* filtered;

// 	void MedianOfArray(aType*, int);
// 	void CreateOnes(aType*, int);
// 	void FilterDx(Image<aType>&, aType*, int, char);
// 	void FilterDy(WriteableImage<aType>&, aType*, int, char);

// public:
// 	MorphologicalFilter():height(0), width(0), channel(0), type(0) {}
// 	MorphologicalFilter(int, int, int, char);
// 	MorphologicalFilter(Image<aType>&);
// 	~MorphologicalFilter();

// 	void Erosion(Image<aType>&, WriteableImage<aType>&, int);
// 	void Dilatation(Image<aType>&, WriteableImage<aType>&, int);
// 	void Median(Image<aType>&, WriteableImage<aType>&, int);
// 	void Open(Image<aType>&, WriteableImage<aType>&, int);
// 	void Close(Image<aType>&, WriteableImage<aType>&, int);
// 	void WhiteTopHat(Image<aType>&, WriteableImage<aType>&, int);
// 	void BlackTopHat(Image<aType>&, WriteableImage<aType>&, int);
// };

// template<typename aType>
// MorphologicalFilter<aType>::MorphologicalFilter(int height, int width, int channel, char type) {
//     this->height = height;
//     this->width = width;
//     this->channel = channel;
//     this->type = type;
//     this->filtered = new aType[this->height*this->width*this->channel];
// }

// template<typename aType>
// MorphologicalFilter<aType>::MorphologicalFilter(Image<aType>& src) {
//     this->height = src.GetHeight();
//     this->width = src.GetWidth();
//     this->channel = src.GetChannels();
//     this->type = src.GetType();
//     this->filtered = new aType[this->height*this->width*this->channel];
// }

// template<typename aType>
// MorphologicalFilter<aType>::~MorphologicalFilter() {
//     delete[] filtered;
// }

// template<typename aType>
// void MorphologicalFilter<aType>::MedianOfArray(aType* kernel, int diameter) {
//     int tmp;
//     for (int i = diameter - 1; i > 0; --i) {
//         for (int j = 0; j < i; ++j) {
//             if (kernel[j] > kernel[j + 1]) {
//                 tmp = kernel[j + 1];
//                 kernel[j] = kernel[j + 1];
//                 kernel[j + 1] = tmp;
//             }
//         }
//     }
// }

// template<typename aType>
// void MorphologicalFilter<aType>::CreateOnes(aType* kernel, int diameter) {
//     for (int i = 0; i < diameter; i++)
//         kernel[i] = 1.0;
// }

// template<typename aType>
// void MorphologicalFilter<aType>::FilterDx(Image<aType>& src, aType* kernel, int radius, char factor) {
//     int i, j, k, l, inf;
//     for (k = 0; k < channel; k++) {
//         for (i = 0; i < height; i++) {
//             for (j = 0; j < width; j++) {
//                 inf = 255;
//                 for (l = -radius; l <= radius; l++) {
//                     if (j + l < 0) {
//                         inf = (kernel[l + radius] == 1 && factor * src.Get(i, 0, k) < inf) ? factor * src.Get(i, 0, k) : inf;
//                     } else if (j + l >= width) {
//                         inf = (kernel[l + radius] == 1 && factor * src.Get(i, width - 1, k) < inf) ? factor * src.Get(i, width - 1, k) : inf;
//                     } else {
//                         inf = (kernel[l + radius] == 1 && factor * src.Get(i, j + l, k) < inf) ? factor * src.Get(i, j + l, k) : inf;
//                     }
//                 }
//                 filtered[j + i * width + k * height * width] = factor * inf;
//             }
//         }
//     }
// }

// template<typename aType>
// void MorphologicalFilter<aType>::FilterDy(WriteableImage<aType>& dst, aType* kernel, int radius, char factor) {
//     int i, j, k, l, inf;
//     for (k = 0; k < channel; k++) {
//         for (i = 0; i < height; i++) {
//             for (j = 0; j < width; j++) {
//                 inf = filtered[j + i * width + k * height * width];
//                 for (l = -radius; l <= radius; l++) {
//                     if (i + l < 0) {
//                         inf = (kernel[l + radius] == 1 && factor * filtered[j + k * height * width] < inf) ? factor * filtered[j + k * height * width] : inf;
//                     } else if (i + l >= height) {
//                         inf = (kernel[l + radius] == 1 && factor * filtered[j + (height - 1) * width + k * height * width] < inf) ? factor * filtered[j + (height - 1) * width + k * height * width] : inf;
//                     } else {
//                         inf = (kernel[l + radius] == 1 && factor * filtered[j + (i + l) * width + k * height * width] < inf) ? factor * filtered[j + (i + l) * width + k * height * width] : inf;
//                     }
//                 }
//                 dst.Set(i, j, k, factor * inf);
//             }
//         }
//     }
// }

// template<typename aType>
// void MorphologicalFilter<aType>::Erosion(Image<aType>& src, WriteableImage<aType>& dst, int radius) {
//     dst.Reset(height, width, type);
//     int diameter = 2 * radius + 1;
//     aType* kernel = new aType[diameter];
//     CreateOnes(kernel, diameter);
//     FilterDx(src, kernel, radius, 1);
//     FilterDy(dst, kernel, radius, 1);
//     delete[] kernel;
// }

// template<typename aType>
// void MorphologicalFilter<aType>::Dilatation(Image<aType>& src, WriteableImage<aType>& dst, int radius) {
//     dst.Reset(height, width, type);
//     int diameter = 2 * radius + 1;
//     aType* kernel = new aType[diameter];
//     CreateOnes(kernel, diameter);
//     FilterDx(src, kernel, radius, -1);
//     FilterDy(dst, kernel, radius, -1);
//     delete[] kernel;
// }

// template<typename aType>
// void MorphologicalFilter<aType>::Median(Image<aType>& src, WriteableImage<aType>& dst, int radius) {
//     dst.Reset(height, width, type);
//     int i, j, k, l, m, x, y;
//     int diameter = pow(2 * radius + 1, 2);
//     aType* kernel = new aType[diameter];
//     for (k = 0; k < channel; k++) {
//         for (i = 0; i < height; i++) {
//             for (j = 0; j < width; j++) {
//                 for (l = -radius; l <= radius; l++) {
//                     for (m = -radius; m <= radius; m++) {
//                         x = i + l >= height ? height - 1 : (i + l < 0 ? 0 : i + l);
//                         y = j + m >= width ? width - 1 : (j + m < 0 ? 0 : j + m);
//                         kernel[(m + radius) + (l + radius) * (2 * radius + 1)] = src.Get(x, y, k);
//                     }
//                 }
//                 MedianOfArray(kernel, diameter);
//                 dst.Set(i, j, k, kernel[(diameter - 1) / 2]);
//             }
//         }
//     }
//     delete[] kernel;
// }

// template<typename aType>
// void MorphologicalFilter<aType>::Open(Image<aType>& src, WriteableImage<aType>& dst, int radius) {
//     Dilatation(src, dst, radius);
//     Erosion(dst, dst, radius);
// }

// template<typename aType>
// void MorphologicalFilter<aType>::Close(Image<aType>& src, WriteableImage<aType>& dst, int radius) {
//     Erosion(src, dst, radius);
//     Dilatation(dst, dst, radius);
// }

// template<typename aType>
// void MorphologicalFilter<aType>::WhiteTopHat(Image<aType>& src, WriteableImage<aType>& dst, int radius) {
//     Open(src, dst, radius);
//     int value;
//     for (int k = 0; k < channel; k++) {
//         for (int i = 0; i < height; i++) {
//             for (int j = 0; j < width; j++) {
//                 value = src.Get(i, j, k) - dst.Get(i, j, k) < 0 ? 0 : src.Get(i, j, k) - dst.Get(i, j, k);
//                 dst.Set(i, j, k, value);
//             }
//         }
//     }
// }

// template<typename aType>
// void MorphologicalFilter<aType>::BlackTopHat(Image<aType>& src, WriteableImage<aType>& dst, int radius) {
//     Close(src, dst, radius);
//     int value;
//     for (int k = 0; k < channel; k++) {
//         for (int i = 0; i < height; i++) {
//             for (int j = 0; j < width; j++) {
//                 value = dst.Get(i, j, k) - src.Get(i, j, k) < 0 ? 0 : dst.Get(i, j, k) - src.Get(i, j, k);
//                 dst.Set(i, j, k, value);
//             }
//         }
//     }
// }

// #endif //__MORPHOLOGICALFILTER_H__
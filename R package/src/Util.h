// #include <Rcpp.h>

// using namespace Rcpp;
// using namespace std;

// #ifndef __UTIL_H__
// #define __UTIL_H__

// #define PI 3.14159265359

// template <typename T> int sgn(T val) {
//     return (T(0) < val) - (val < T(0));
// }

// template<typename aType>
// class Util
// {
// public:
// 	Util();
// 	~Util();

// 	void MarkRed(NumericMatrix&, NumericMatrix&, NumericMatrix&);
// 	void AddImages(NumericMatrix&, NumericMatrix&, NumericMatrix&);
// 	void InverseImage(NumericMatrix&, NumericMatrix&);
// };

// template<typename aType>
// void Util<aType>::MarkRed(NumericMatrix& src, NumericMatrix& edges, NumericMatrix& dst) {
//     if (src.nrow() != edges.nrow() || src.ncol() != edges.ncol()) {
//         printf("Height, width and number of channels do not match!\n");
//     } else {
//         for (int i = 0; i < src.nrow(); i++) {
//             for (int j = 0; j < src.ncol(); j++) {
//                 if (edges.Get(i, j, 0) > 50) {
//                     dst.Set(i, j, 0, 0);
//                     dst.Set(i, j, 1, 0);
//                     dst.Set(i, j, 2, 255);
//                 } else {
//                     for (int k = 0; k < src.GetChannels(); k++) {
//                         dst.Set(i, j, k, src.Get(i, j, k));
//                     }
//                 }
//             }
//         }
//     }
// }

// template<typename aType>
// void Util<aType>::AddImages(NumericMatrix& src1, NumericMatrix& src2, NumericMatrix& dst) {
//     dst.Reset(src1.nrow(), src1.ncol(), src1.GetType());
//     if (src1.nrow() != src2.nrow() || src1.ncol() != src2.ncol() || src1.GetChannels() != src2.GetChannels()) {
//         printf("Height, width and number of channels do not match!\n");
//     } else {
//         int value;
//         for (int k = 0; k < src1.GetChannels(); k++) {
//             for (int i = 0; i < src1.nrow(); i++) {
//                 for (int j = 0; j < src1.ncol(); j++) {
//                     value = src1.Get(i, j, k) + src2.Get(i, j, k) > 255 ? 255 : src1.Get(i, j, k) + src2.Get(i, j, k);
//                     dst.Set(i, j, k, value);
//                 }
//             }
//         }
//     }
// }

// template<typename aType>
// void Util<aType>::InverseImage(NumericMatrix& src, NumericMatrix& dst) {
//     dst.Reset(src.nrow(), src.ncol(), src.GetType());
//     for (int k = 0; k < src.GetChannels(); k++)
//         for (int i = 0; i < src.nrow(); i++)
//             for (int j = 0; j < src.ncol(); j++)
//                 dst.Set(i, j, k, 255 - src.Get(i, j, k));
// }

// #endif //__UTIL_H__
#include "EdgeDetector.h"
#include "HuberROFModel.h"
#include "ImageInpainting.h"
#include "LinearFilter.h"
#include "MorphologicalFilter.h"
#include "RealTimeMinimizer.h"
#include "TypeConversion.h"
#include "Util.h"
#include "TVL1Model.h"
#include "ROFModel.h"

//  [[Rcpp::export]]
void ROF(const NumericMatrix& src, NumericMatrix& dst, float lambda, float tau, int iter, int height, int width, int channel) {
  ROFModel<float> rof(src, iter, height, width, channel);
  rof.ROF(src, dst, lambda, tau);
}

//  [[Rcpp::export]]
void TVL1(const NumericMatrix& src, NumericMatrix& dst, float lambda, float tau, int iter, int height, int width, int channel) {
  TVL1Model<float> tvl1(src, iter, height, width, channel);
  tvl1.TVL1(src, dst, lambda, tau);
}

//  [[Rcpp::export]]
void SobelOperator(const NumericMatrix& src, NumericMatrix& dst, int height, int width, int channel) {
  EdgeDetector<float> sobel(src, height, width, channel);
  sobel.Sobel(src, dst);
}

//  [[Rcpp::export]]
void HuberROF(const NumericMatrix& src, NumericMatrix& dst, float alpha, float lambda, float tau, int iter, int height, int width, int channel) {
  HuberROFModel<float> hrof(src, iter, height, width, channel);
  hrof.HuberROF(src, dst, alpha, lambda, tau);
}

//  [[Rcpp::export]]
void Inpaint(const NumericMatrix& src, NumericMatrix& dst, float lambda, float tau, int iter, int height, int width, int channel) {
  ImageInpainting<float> inpaint(src, iter, height, width, channel);
  inpaint.Inpaint(src, dst, lambda, tau);
}

//  [[Rcpp::export]]
void RealTime(const NumericMatrix& src, NumericMatrix& dst, float alpha, float lambda, float tau, int cartoon, int iter, int height, int width, int channel) {
  RealTimeMinimizer<float> realtime(src, iter, height, width, channel);
  realtime.RTMinimizer(src, dst, alpha, lambda, tau, cartoon);
}

// //  [[Rcpp::export]]
// void LinearFilter(const NumericMatrix& src, NumericMatrix& dst, float lambda, float tau, int iter, int height, int width, int channel) {
//   LinearFilter<float> tvl1(src, iter, height, width, channel);
//   tvl1.TVL1(src, dst, lambda, tau);
// }

// //  [[Rcpp::export]]
// void MorphologicalFilter(const NumericMatrix& src, NumericMatrix& dst, float lambda, float tau, int iter, int height, int width, int channel) {
//   MorphologicalFilter<float> tvl1(src, iter, height, width, channel);
//   tvl1.TVL1(src, dst, lambda, tau);
// }

// //  [[Rcpp::export]]
// void RealTimeMinimizer(const NumericMatrix& src, NumericMatrix& dst, float lambda, float tau, int iter, int height, int width, int channel) {
//   RealTimeMinimizer<float> tvl1(src, iter, height, width, channel);
//   tvl1.TVL1(src, dst, lambda, tau);
// }

// //  [[Rcpp::export]]
// void TypeConversion(const NumericMatrix& src, NumericMatrix& dst, float lambda, float tau, int iter, int height, int width, int channel) {
//   TypeConversion<float> tvl1(src, iter, height, width, channel);
//   tvl1.TVL1(src, dst, lambda, tau);
// }

// //  [[Rcpp::export]]
// void Util(const NumericMatrix& src, NumericMatrix& dst, float lambda, float tau, int iter, int height, int width, int channel) {
//   Util<float> tvl1(src, iter, height, width, channel);
//   tvl1.TVL1(src, dst, lambda, tau);
// }
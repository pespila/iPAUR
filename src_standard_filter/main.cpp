#include "image.h"
#include "rgb.h"
#include "hsi.h"
#include "ycrcb.h"
#include "grayscale.h"
#include "type_conversion.h"
#include "bluring.h"
#include "gradient_filter.h"
#include "robertscross.h"
#include "prewitt.h"
#include "sobel.h"
#include "canny.h"
#include "laplacian_of_gaussian.h"
#include "laplace.h"
#include "create_filter.h"
#include "linear_filter.h"
#include "duto.h"
#include "dilatation.h"
#include "erosion.h"
#include "inverse.h"
#include "openclose.h"
#include "top_hats.h"
#include "median.h"
#include "arithmetic_functions.h"
#include "rof_model.h"

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        printf("ERROR: Pass in file reading file, writing file, algorithm & filter size!\n");
        return 1;
    }

    GrayscaleImage in, out;
    in.read_image(argv[1]);
    Huber_ROF_Model Op(in.get_height(), in.get_width(), 0.01, 8.0, 1.0, 0.0); // tau, lambda, theta, alpha
    
    Op.primal_dual_algorithm(in, out);

    out.write_image(argv[2]);

    // GrayscaleImage gray, edge1, edge2, edge;
    // RGBImage rgb_original, rgb;
    // HSIImage hsi;
    // YCrCbImage ycrcb;
    // GrayscaleImage gray;
    
    // rgb.read_image(argv[1]);
    // rgb_original.read_image(argv[3]);

    // rgb2hsi(rgb, hsi);
    // rgb2ycrcb(rgb, ycrcb);

    // rgb2gray(rgb, gray);
    // robertsCross(gray, edge1);
    // sobel(gray, edge1);
    // prewitt(gray, edge2);
    // addGrayscaleImages(edge1, edge2, edge);
    // markRed(edge, rgb_original, rgb.get_type());
    // canny(gray, edge, 1, 9);

    // rgb_original.write_image(argv[2]);
    // edge1.write_image(argv[2]);
    // color.read_image(argv[1]);

    // Image detection
    //TI
    // rgb2hsi(color, hsi);
    // rgb2ycrcb(color, ycrcb);
    // rgb2gray(hsi, gray);

    // canny(gray, can, 1, 9);
    // prewitt(gray, prew);
    // add(can, prew, sum1);

    // rgb2gray(ycrcb, gray);

    // canny(gray, can, 1, 9);
    // prewitt(gray, prew);
    // addGrayscaleImages(can, prew, sum2);

    // rgb2gray(color, gray);

    // canny(gray, can, 1, 9);
    // prewitt(gray, prew);
    // addGrayscaleImages(can, prew, sum3);

    // addGrayscaleImages(sum1, sum2, save1);
    // addGrayscaleImages(save1, sum3, save2);
    // dilatationGrayscaleImage(save2, save3, morphologicalStandardFilter(1), 1);
    // markRed(save3, color, color.get_type());
    //TI

    // rgb2gray(color, gray);
    // canny(gray, g_edge, 1, 9);

    // sobel(gray, g_edge);
    // prewitt(gray, g_edge);
    // robertsCross(gray, g_edge);
    // laplace(gray, g_edge);
    // laplaceSharpener(gray, g_edge, 1.2);
    // gradientFilter(gray, g_edge);
    // laplacianOfGaussian(gray, g_edge, laplacianOfGaussianFilter(3, 1.2), 3);

    // dutoGrayscaleImage(gray, g_edge, gaussKernel(5, 1.8), 5, 0.5);
    // linearFilterColorImage(color, c_edge, gaussKernel(5, 1.8), 5);
    // dutoColorImage(color, c_edge, gaussKernel(5, 1.8), 5, 0.5);

    // inverseGrayscaleImage(g_edge, g_dilate);
    // inverseColorImage(color, c_edge);
    // dilatationGrayscaleImage(gray, g_dilate, morphologicalStandardFilter(3), 3);
    // erosionGrayscaleImage(gray, g_dilate, morphologicalStandardFilter(1), 1);
    // openGrayscaleImage(gray, g_dilate, morphologicalStandardFilter(2), 2);
    // closeGrayscaleImage(gray, g_dilate, morphologicalStandardFilter(2), 2);
    // openColorImage(color, c_edge, morphologicalStandardFilter(4), 4);
    // closeColorImage(color, c_edge, morphologicalStandardFilter(5), 5);
    // rgb2gray(c_edge, gray);
    // dilatationColorImage(color, c_edge, morphologicalStandardFilter(3), 3);
    // erosionColorImage(color, c_edge, morphologicalStandardFilter(5), 5);

    // sobel(g_dilate, g_edge);

    // whiteTopHatGrayscaleImage(gray, g_dilate, morphologicalStandardFilter(1), 1);
    // blackTopHatGrayscaleImage(gray, g_dilate, morphologicalStandardFilter(1), 1);
    // whiteTopHatColorImage(color, c_edge, morphologicalStandardFilter(1), 1);
    // blackTopHatColorImage(color, c_edge, morphologicalStandardFilter(1), 1);
    
    return 0;
}
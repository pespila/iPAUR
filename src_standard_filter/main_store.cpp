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

#include <vector>
#include <iostream>

void FindBlobs(const cv::Mat &binary, std::vector < std::vector<cv::Point2i> > &blobs)
{
    blobs.clear();

    // Fill the label_image with the blobs
    // 0  - background
    // 1  - unlabelled foreground
    // 2+ - labelled foreground

    cv::Mat label_image;
    binary.convertTo(label_image, CV_32SC1);

    int label_count = 2; // starts at 2 because 0,1 are used already

    for(int y=0; y < label_image.rows; y++) {
        int *row = (int*)label_image.ptr(y);
        for(int x=0; x < label_image.cols; x++) {
            if(row[x] != 1) {
                continue;
            }

            cv::Rect rect;
            cv::floodFill(label_image, cv::Point(x,y), label_count, &rect, 0, 0, 4);

            std::vector <cv::Point2i> blob;

            for(int i=rect.y; i < (rect.y+rect.height); i++) {
                int *row2 = (int*)label_image.ptr(i);
                for(int j=rect.x; j < (rect.x+rect.width); j++) {
                    if(row2[j] != label_count) {
                        continue;
                    }

                    blob.push_back(cv::Point2i(j,i));
                }
            }

            blobs.push_back(blob);

            label_count++;
        }
    }
}

void connectedComponentLabeling(GrayscaleImage& src, RGBImage& dst)
{
    // cv::Mat img = cv::imread("blob.png", 0); // force greyscale
    int height = src.get_height(), width = src.get_width();
    dst.reset_image(height, width, CV_8UC3);
    Mat mySrc(height, width, src.get_type());
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            mySrc.at<Vec3b>(i, j)[0] = src.get_gray_pixel_at_position(i, j);
            mySrc.at<Vec3b>(i, j)[1] = src.get_gray_pixel_at_position(i, j);
            mySrc.at<Vec3b>(i, j)[2] = src.get_gray_pixel_at_position(i, j);
        }
    }

    Mat img;
    cvtColor(mySrc, img, CV_BGR2GRAY);

    // if(!img.data) {
    //     std::cout << "File not found" << std::endl;
    //     return -1;
    // }
    // cv::namedWindow("binary");

    // cv::Mat output = cv::Mat::zeros(img.size(), CV_8UC3);

    cv::Mat binary;
    std::vector < std::vector<cv::Point2i > > blobs;

    cv::threshold(img, binary, 0.0, 1.0, cv::THRESH_BINARY);

    FindBlobs(binary, blobs);

    int sum = 0, norm = 0;
    // Randomy color the blobs
    for(size_t i=0; i < blobs.size(); i++) {
        // unsigned char r = 255 * (rand()/(1.0 + RAND_MAX));
        // unsigned char g = 255 * (rand()/(1.0 + RAND_MAX));
        // unsigned char b = 255 * (rand()/(1.0 + RAND_MAX));
        // unsigned char r = 0;
        // unsigned char g = 0;
        // unsigned char b = 0;
        sum = 0;
        norm = 0;
        for(size_t j=0; j < blobs[i].size(); j++) {
            norm++;
            int x = blobs[i][j].x;
            int y = blobs[i][j].y;
                sum += src.get_gray_pixel_at_position(x, y);
            // cout << i << " " << j << endl;

            // dst.set_color_pixel_at_position(y, x, 0, b);
            // dst.set_color_pixel_at_position(y, x, 1, g);
            // dst.set_color_pixel_at_position(y, x, 2, r);

            // output.at<cv::Vec3b>(y,x)[0] = b;
            // output.at<cv::Vec3b>(y,x)[1] = g;
            // output.at<cv::Vec3b>(y,x)[2] = r;
        }
        sum /= norm;
        if (sum > 154 && sum < 160) {
            unsigned char r = 0;
            unsigned char g = 255;
            unsigned char b = 0;
            for(size_t j=0; j < blobs[i].size(); j++) {
                int x = blobs[i][j].x;
                int y = blobs[i][j].y;
                dst.set_color_pixel_at_position(y, x, 0, b);
                dst.set_color_pixel_at_position(y, x, 1, g);
                dst.set_color_pixel_at_position(y, x, 2, r);
            }
        }
        // sum /= norm;
        // if (sum > 154 && sum < 160) {
        //     cout << min_element(blobs[i][0].begin(), blobs[i][0].end()) << endl;
        //     // int min_i = min(blobs[i][0].y);
        //     // int min_j = min(blobs[i][0].x);
        //     int min_i = blobs[i][0].y, min_j = blobs[i][0].x;
        //     int max_i = blobs[i][blobs.size()-1].y, max_j = blobs[i][blobs.size()-1].x;
        //     for (int k = min_i; k < max_i; k++)
        //     {
        //         dst.set_color_pixel_at_position(k, min_j, 0, 0);
        //         dst.set_color_pixel_at_position(k, max_j, 0, 0);
        //         dst.set_color_pixel_at_position(k, min_j, 1, 255);
        //         dst.set_color_pixel_at_position(k, max_j, 1, 255);
        //         dst.set_color_pixel_at_position(k, min_j, 2, 0);
        //         dst.set_color_pixel_at_position(k, max_j, 2, 0);
        //     }
        //     for (int j = min_j; j < max_j; j++)
        //     {
        //         dst.set_color_pixel_at_position(min_i, j, 0, 0);
        //         dst.set_color_pixel_at_position(max_i, j, 0, 0);
        //         dst.set_color_pixel_at_position(min_i, j, 1, 255);
        //         dst.set_color_pixel_at_position(max_i, j, 1, 255);
        //         dst.set_color_pixel_at_position(min_i, j, 2, 0);
        //         dst.set_color_pixel_at_position(max_i, j, 2, 0);
        //     }
        // }
    }
    // cv::imshow("binary", output);
    // cv::waitKey(0);
}

int sumOfArea(GrayscaleImage& src, int start_h, int start_w, int end_h, int end_w) {
    int norm = (end_h - start_h)*(end_w - start_w);
    int sum = 0;
    for (int i = start_h; i < end_h; i++) {
        for (int j = start_w; j < end_w; j++) {
            sum += src.get_gray_pixel_at_position(i, j);
        }
    }
    sum /= norm;
    return sum;
}

void trackArea(GrayscaleImage& src, RGBImage& dst) {
    // if (src.get_height() > 0 && dst.get_height() > 0) {
        int height = src.get_height(), width = src.get_width();
        // dst.reset_image(height, width, src.get_type());
        int sum = 0;
        int x = 0, y = 0, stop = 0;
        int min_i = -1, min_j = -1, max_i = -1, max_j = -1;
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                x = i + 3;
                y = j + 200;
                if (x < height && y < width) {
                    sum = sumOfArea(src, i, j, x, y);
                    if (sum < 124 && min_i == -1 && min_j == -1) {
                        min_i = i;
                        min_j = j;
                        stop = 1;
                        break;
                    }
                }
            }
            if (stop) {
                break;
            }
        }
        stop = 0;
        for (int i = height-1; i >= 0; i--)
        {
            for (int j = width-1; j >= 0; j--)
            {
                x = i - 3;
                y = j - 200;
                if (x >= 0 && y >= 0) {
                    sum = sumOfArea(src, x, y, i, j);
                    if (sum < 124 && max_i == -1 && max_j == -1) {
                        max_i = i;
                        max_j = j;
                        stop = 1;
                        break;
                    }
                }
            }
            if (stop) {
                break;
            }
        }
        if (max_j < min_j) {
            int tmp = max_j;
            max_j = min_j;
            min_j = tmp;
        }
        // cout << min_i << " " << min_j << " " << max_i << " " << max_j << endl;
        for (int i = min_i; i < max_i; i++)
        {
            dst.set_color_pixel_at_position(i, min_j, 0, 0);
            dst.set_color_pixel_at_position(i, max_j, 0, 0);
            dst.set_color_pixel_at_position(i, min_j, 1, 255);
            dst.set_color_pixel_at_position(i, max_j, 1, 255);
            dst.set_color_pixel_at_position(i, min_j, 2, 0);
            dst.set_color_pixel_at_position(i, max_j, 2, 0);
        }
        for (int j = min_j; j < max_j; j++)
        {
            dst.set_color_pixel_at_position(min_i, j, 0, 0);
            dst.set_color_pixel_at_position(max_i, j, 0, 0);
            dst.set_color_pixel_at_position(min_i, j, 1, 255);
            dst.set_color_pixel_at_position(max_i, j, 1, 255);
            dst.set_color_pixel_at_position(min_i, j, 2, 0);
            dst.set_color_pixel_at_position(max_i, j, 2, 0);
        }
    // }
}

// void connectedComponentLabeling(GrayscaleImage& src) {
//     int height = src.get_height(), width = src.get_width();
//     // dst.reset_image(height, width, src.get_type());
//     short* label = (short*)malloc(height*width*sizeof(short));

//     for (int i = 0; i < height; i++)
//     {
//         for (int j = 0; j < width; j++)
//         {
//             if (src.get_gray_pixel_at_position(i, j) != 0) {

//             }
//         }
//     }
// }

void medianFilter(GrayscaleImage& src, GrayscaleImage& dst) {
    int height = src.get_height(), width = src.get_width();
    dst.reset_image(height, width, src.get_type());
    vector<short> array(9, 0);
    // short* array = (short*)malloc(9*sizeof(short));
    for (int i = 1; i < height-1; i++)
    {
        for (int j = 1; j < width-1; j++)
        {
            for (int k = -1; k <= 1; k++)
            {
                for (int l = -1; l <= 1; l++)
                {
                    array[(l+1)+(k+1)*3] = src.get_gray_pixel_at_position(i+k, j+l);
                }
            }
            sort(array.begin(), array.end());
            dst.set_gray_pixel_at_position(i, j, array[4]);
        }
    }
}

void piv(GrayscaleImage& src, GrayscaleImage& dst) {
    int height = src.get_height(), width = src.get_width();
    dst.reset_image(height, width, src.get_type());
    GrayscaleImage LoG, med;
    laplacianOfGaussian(src, LoG, laplacianOfGaussianFilter(3, 1.2), 3);
    medianFilter(LoG, med);
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            short tmp = LoG.get_gray_pixel_at_position(i, j) - med.get_gray_pixel_at_position(i, j);
            dst.set_gray_pixel_at_position(i, j, tmp);
        }
    }
}

void imageDetection(const char* argv[]) {
    // Image detection
    RGBImage color, afterHSI, afterYCrCb, save, labelled, tracked;
    YCrCbImage ycrcb;
    HSIImage hsi;
    GrayscaleImage gray, fromHSI, fromYCrCb, HSIclose, YCrCbclose, saving, prew, can, edge, mult, inv, inv2, blubb, clo;
    color.read_image(argv[1]);
    rgb2hsi(color, hsi);
    rgb2ycrcb(color, ycrcb);

    inRangeColorImage(hsi, afterHSI, 0, 20, 10, 150, 60, 255);
    inRangeColorImage(ycrcb, afterYCrCb, 0, 255, 133, 173, 77, 127);
    rgb2gray(color, gray);
    rgb2gray(afterHSI, fromHSI);
    rgb2gray(afterYCrCb, fromYCrCb);
    closeGrayscaleImage(fromHSI, HSIclose, morphologicalStandardFilter(3), 3);
    closeGrayscaleImage(fromYCrCb, YCrCbclose, morphologicalStandardFilter(3), 3);
    // erosionGrayscaleImage(fromHSI, HSIerode, morphologicalStandardFilter(2), 2);
    // dilatationGrayscaleImage(HSIerode, HSIdilate, morphologicalStandardFilter(2), 2);
    // erosionGrayscaleImage(fromYCrCb, YCrCberode, morphologicalStandardFilter(2), 2);
    // dilatationGrayscaleImage(YCrCberode, YCrCbdilate, morphologicalStandardFilter(2), 2);
    addGrayscaleImages(HSIclose, YCrCbclose, saving);
    prewitt(gray, prew);
    // sobel(gray, prew);
    canny(gray, can, 5, 20);
    // thresh_callback(gray, can);
    addGrayscaleImages(prew, can, edge);
    // openGrayscaleImage(edge, clo, morphologicalStandardFilter(3), 3);
    // multiplyGrayscaleImages(saving, edge, mult);
    inverseGrayscaleImage(edge, inv);
    // addGrayscaleImages(saving, inv, mult);
    multiplyGrayscaleImages(saving, inv, mult);
    closeGrayscaleImage(inv, mult, morphologicalStandardFilter(3), 3);
    // addGrayscaleImages(inv, clo, mult);
    // closeGrayscaleImage(mult, clo, morphologicalStandardFilter(10), 10);
    connectedComponentLabeling(mult, color);
    // inverseGrayscaleImage(mult, inv2);
    // rgb2gray(labelled, blubb);
    // closeGrayscaleImage(blubb, clo, morphologicalStandardFilter(3), 3);
    // trackArea(blubb, color);
    // addGrayscaleImages(saving, inv, mult);
    // addColorImages(afterHSI, afterYCrCb, save);
    // addGrayscaleImages(fromHSI, fromYCrCb, saving);
    color.write_image(argv[2]);
    // Image detection
}

struct components
{
    double dual_x;
    double dual_y;
};

double project_x(double x) {
    return (min(1.0, max(0.0, x)));
}

double project_y(double y) {
    return (min(1.0, max(0.0, y)));
}

void primal_dual(GrayscaleImage& src, GrayscaleImage& dst) {
    int height = src.get_height(), width = src.get_width(), w = 0;
    dst.reset_image(height, width, src.get_type());

    double tau = 1.0/sqrt(12), sigma = 1.0/sqrt(12), height_h = 1.0/(height), width_h = 1.0/(width);

    struct components* dual = (struct components*)malloc(height*width*sizeof(struct components));
    double* primal = (double*)malloc(height*width*sizeof(double));
    double* primal_hat = (double*)malloc(height*width*sizeof(double));

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            primal[j + i*width] = (double)src.get_gray_pixel_at_position(i, j)/255.0;
            primal_hat[j + i*width] = primal[j + i*width];
            dual[j + i*width].dual_x = 0.0;
            dual[j + i*width].dual_y = 0.0;
        }
    }

    double grad_x = 0.0, grad_y = 0.0, div_x = 0.0, div_y = 0.0, x0 = 0.0, y0 = 0.0, u = 0.0, ux = 0.0, uy = 0.0, divergence = 0.0, u_cur = 0.0;
    
    while (w < 32) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                x0 = i+1 == height ? 0.0 : primal_hat[j + (i+1)*width];
                y0 = j+1 == width ? 0.0 : primal_hat[j + 1 + i*width];
                u = primal_hat[j + i*width];
                grad_x = (x0 - u) * height_h * sigma;
                grad_y = (y0 - u) * width_h * sigma;
                dual[j + i*width].dual_x += grad_x;
                dual[j + i*width].dual_y += grad_y;
                dual[j + i*width].dual_x = project_y(dual[j + i*width].dual_x);
                dual[j + i*width].dual_y = project_y(dual[j + i*width].dual_y);
            }
        }
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                x0 = i+1 == height ? 0.0 : dual[j + (i+1)*width].dual_x;
                y0 = j+1 == width ? 0.0 : dual[j + 1 + i*width].dual_y;
                ux = dual[j + i*width].dual_x;
                uy = dual[j + i*width].dual_y;
                div_x = (x0 - ux) * height_h * tau;
                div_y = (y0 - uy) * width_h * tau;
                divergence = div_x + div_y;
                u_cur = primal[j + i*width];
                primal[j + i*width] -= divergence;
                primal[j + i*width] = project_x(primal[j + i*width]);
                primal_hat[j + i*width] = 2.0*primal[j + i*width] - u_cur;
            }
        }
        w++;
    }
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            dst.set_gray_pixel_at_position(i, j, (short)(255.0*primal[j + i*width]));
        }
    }
    free(dual);
    free(primal);
    free(primal_hat);
}

// void primal_dual(GrayscaleImage& src, GrayscaleImage& dst) {
//     int height = src.get_height(), width = src.get_width();
//     double tau = 0.02, sigma = 0.0, height_h = 1.0/height, width_h = 1.0/width;
//     sigma = 6.25 * height_h * width_h;
//     // double tau = 1.0/100.0, sigma = 1.0/50.0, height_h = 1.0/height, width_h = 1.0/width;
//     // double tau = 1.0/sqrt(12), sigma = 1.0/sqrt(12), height_h = 1.0/(height-1), width_h = 1.0/(width-1);
//     dst.reset_image(height, width, src.get_type());
//     // GrayscaleImage upd;
//     // upd.reset_image(height, width, src.get_type());

//     struct components* dual = (struct components*)malloc(height*width*sizeof(struct components));
//     int* primal = (int*)malloc(height*width*sizeof(int));
//     int* primal_hat = (int*)malloc(height*width*sizeof(int));
//     for (int i = 0; i < height; i++) {
//         for (int j = 0; j < width; j++) {
//             primal[j + i*width] = src.get_gray_pixel_at_position(i, j);
//             primal_hat[j + i*width] = primal[j + i*width];
//             dual[j + i*width].dual_x = 0;
//             dual[j + i*width].dual_y = 0;
//         }
//     }

//     int x = 0, y = 0, M = 1024, w = 0;
//     double eps = 1.0;
//     double grad_x = 0, grad_y = 0, div_x = 0, div_y = 0, tmp_primal;
//     // for (int k = 1; k <= M; k++) {
//     while (eps > pow(10, -4) || w < 2) {
//         eps = 0.0;
//         for (int i = 0; i < height; i++) {
//             for (int j = 0; j < width; j++) {
//                 int x0 = i+1 == height ? 0 : primal_hat[j + (i+1)*width];
//                 int y0 = j+1 == width ? 0 : primal_hat[j + 1 + i*width];
//                 int u = primal_hat[j + i*width];
//                 grad_x = (x0 - u) * height_h * sigma;
//                 grad_y = (y0 - u) * width_h * sigma;
//                 if (w == 1) {
//                     cout << grad_x << endl;
//                     // cout << primal_hat[j + i*width] << " " << u_cur << " " << primal_hat[j + i*width] - u_cur << endl;
//                 }
//                 dual[j + i*width].dual_x += grad_x;
//                 dual[j + i*width].dual_y += grad_y;
//             }
//         }
//         for (int i = 0; i < height; i++) {
//             for (int j = 0; j < width; j++) {
//                 int x0 = i+1 == height ? 0 : dual[j + (i+1)*width].dual_x;
//                 int y0 = j+1 == width ? 0 : dual[j + 1 + i*width].dual_y;
//                 int ux = dual[j + i*width].dual_x;
//                 int uy = dual[j + i*width].dual_y;
//                 div_x = (x0 - ux) * height_h * tau;
//                 div_y = (y0 - uy) * width_h * tau;
//                 double divergence = div_x + div_y;
//                 int u_cur = primal[j + i*width];
//                 primal[j + i*width] -= divergence;
//                 primal_hat[j + i*width] = 2.0*primal[j + i*width] - u_cur;

//                 eps += pow(primal_hat[j + i*width] - u_cur, 2);
//                 // if (w == 1) {
//                 //     cout << divergence << endl;
//                 //     // cout << primal_hat[j + i*width] << " " << u_cur << " " << primal_hat[j + i*width] - u_cur << endl;
//                 // }



//                 // dual[j + i*width].dual_x += grad_x;
//                 // dual[j + i*width].dual_y += grad_y;



//                 // x = i+1 == height ? i : i+1;
//                 // y = j+1 == width ? j : j+1;
                
//                 // grad_x = (primal_hat[j + x*width] - primal_hat[j + i*width]) * height_h * sigma;
//                 // grad_y = (primal_hat[y + i*width] - primal_hat[j + i*width]) * width_h * sigma;

//                 // dual[j + i*width].dual_x += grad_x;
//                 // dual[j + i*width].dual_y += grad_y;

//                 // div_x = (dual[j + x*width].dual_x - dual[j + i*width].dual_x) * height_h * tau;
//                 // div_y = (dual[y + i*width].dual_y - dual[j + i*width].dual_y) * width_h * tau;

//                 // tmp_primal = primal[j + i*width];
//                 // primal[j + i*width] = primal[j + i*width] - (div_x + div_y);

//                 // primal_hat[j + i*width] = 2.0*primal[j + i*width] - tmp_primal;
//             }
//         }
//         eps = sqrt(eps);
//         cout << eps << endl;
//         w++;
//     }
//     for (int i = 0; i < height; i++) {
//         for (int j = 0; j < width; j++) {
//             dst.set_gray_pixel_at_position(i, j, primal_hat[j + i*width]);
//         }
//     }
//     free(dual);
//     free(primal);
//     free(primal_hat);
// }

struct duality
{
    double x;
    double y;
};

double norm(GrayscaleImage& src1, GrayscaleImage& src2) {
    double nrm = 0.0;
    for (int i = 0; i < src1.get_height(); i++) {
        for (int j = 0; j < src1.get_width(); j++) {
            nrm += pow(src1.get_gray_pixel_at_position(i, j) - src2.get_gray_pixel_at_position(i, j), 2);
        }
    }
    return sqrt(nrm);
}

void fastms(GrayscaleImage& src, GrayscaleImage& dst) {
    GrayscaleImage u, u_hat, u_tmp;

    int x0 = 0, y0 = 0, u0 = 0;
    double alpha = 10000, lambda = 0.01, tau = 0.25, sigma = 0.5, nrm = 0.0;
    
    int height = src.get_height(), width = src.get_width(), isTrue = 0;
    double grad_x = 0.0, grad_y = 0.0, height_h = 1.0/height, width_h = 1.0/width, px_hat = 0.0, py_hat = 0.0;
    double div_x = 0.0, div_y = 0.0, x1 = 0.0, y1 = 0.0, ux = 0.0, uy = 0.0, div_sum = 0.0;
    double rho = 0.0;

    dst.reset_image(height, width, src.get_type());
    u.reset_image(height, width, src.get_type());
    u_hat.reset_image(height, width, src.get_type());
    u_tmp.reset_image(height, width, src.get_type());
    struct duality* p = (struct duality*)malloc(height*width*sizeof(struct duality));

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            u.set_gray_pixel_at_position(i, j, src.get_gray_pixel_at_position(i, j));
            u_hat.set_gray_pixel_at_position(i, j, src.get_gray_pixel_at_position(i, j));
            u_tmp.set_gray_pixel_at_position(i, j, src.get_gray_pixel_at_position(i, j));
            p[j + i*width].x = 0.0;
            p[j + i*width].y = 0.0;
        }
    }

    nrm = 1.0;
    for (int n = 0; n < 128; n++) {
        // Primal ascent
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                x0 = i+1 == height ? 0 : u_hat.get_gray_pixel_at_position(i+1, j);
                y0 = j+1 == width ? 0 : u_hat.get_gray_pixel_at_position(i, j+1);
                u0 = u_hat.get_gray_pixel_at_position(i, j);
                grad_x = (x0 - u0) * height_h * sigma;
                grad_y = (y0 - u0) * width_h * sigma;
                px_hat = p[j + i*width].x + grad_x;
                py_hat = p[j + i*width].y + grad_y;
                isTrue = sqrt(pow(px_hat, 2)+pow(py_hat, 2)) <= sqrt(lambda/alpha*sigma*(sigma + 2.0*alpha)) ? 1 : 0;
                p[j + i*width].x = isTrue ? (2.0*alpha)/(sigma+2.0*alpha) * px_hat : 0.0;
                p[j + i*width].y = isTrue ? (2.0*alpha)/(sigma+2.0*alpha) * py_hat : 0.0;
            }
        }

        // Dual descent
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                x1 = i+1 == height ? 0.0 : p[j + (i+1)*width].x;
                y1 = j+1 == width ? 0.0 : p[j + 1 + i*width].y;
                ux = p[j + i*width].x;
                uy = p[j + i*width].y;
                div_x = (x0 - ux) * height_h * tau;
                div_y = (y0 - uy) * width_h * tau;
                div_sum = u.get_gray_pixel_at_position(i, j) - div_x + div_y;
                u.set_gray_pixel_at_position(i, j, (short)((div_sum + 2.0*tau*src.get_gray_pixel_at_position(i, j))/(1.0 + 2.0*tau)));
            }
        }

        // Extrapolation
        rho = 1.0/sqrt(1.0 + 4.0*tau);
        tau = rho * tau;
        sigma = sigma/rho;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                short tmp = u.get_gray_pixel_at_position(i, j) + (short)(rho*(u.get_gray_pixel_at_position(i, j) - u_tmp.get_gray_pixel_at_position(i, j)));
                u_hat.set_gray_pixel_at_position(i, j, tmp);
                dst.set_gray_pixel_at_position(i, j, tmp);
                u_tmp.set_gray_pixel_at_position(i, j, u.get_gray_pixel_at_position(i, j));
            }
        }
    }
}

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        printf("ERROR: Pass in file reading file, writing file, algorithm & filter size!\n");
        return 1;
    }
    // PIV
    // GrayscaleImage gray, PIV;
    // gray.read_image(argv[1]);
    // piv(gray, PIV);
    // PIV.write_image(argv[2]);
    // PIV

    // Primal-Dual
    GrayscaleImage gray, pd;
    gray.read_image(argv[1]);
    primal_dual(gray, pd);
    // fastms(gray, pd);
    pd.write_image(argv[2]);
    // Primal-Dual

    // Extract and Convert
    // RGBImage color, newcol;
    // color.read_image(argv[1]);
    // GrayscaleImage gray;
    // gray.reset_image(380, 380, color.get_type());
    // newcol.reset_image(380, 380, color.get_type());
    // for (int i = 0; i < 380; i++) {
    //     for (int j = 150; j < 530; j++) {
    //         for (int k = 0; k < 3; k++) {
    //             newcol.set_color_pixel_at_position(i, j-150, k, color.get_color_pixel_at_position(i, j, k));
    //         }
    //     }
    // }
    // rgb2gray(newcol, gray);
    // gray.write_image(argv[2]);
    // Extract and Convert

    // Image Detection
    // imageDetection(argv);
    // Image Detection


    // RGBImage color, afterHSI, afterYCrCb, save, labelled, tracked;
    // YCrCbImage ycrcb;
    // HSIImage hsi;
    // GrayscaleImage gray, fromHSI, fromYCrCb, HSIclose, YCrCbclose, saving, prew, can, edge, mult, inv, inv2, blubb, clo;
    // // GrayscaleImage gray, g_edge, g_dilate;
    // // GrayscaleImage gray, prew, can, sum1, sum2, sum3, save1, save2, save3;

    // color.read_image(argv[1]);
    // // color.read_image(argv[1]);

    

    // medianFilterColorImage(color, c_edge);
    // medianFilterGrayscaleImage(gray, g_edge);

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

    // rgb2gray(c_edge, gray);
    // sobel(gray, g_edge);

    // g_dilate.write_image(argv[2]);
    // gray.write_image(argv[2]);
    // c_edge.write_image(argv[2]);
    // for (int i = 0; i < afterHSI.get_height(); i++)
    // {
    //     for (int j = 0; j < afterHSI.get_width(); j++)
    //     {
    //         for (int k = 0; k < 3; k++)
    //         {
    //             afterHSI.set_color_pixel_at_position(i, j, k, 0);
    //         }
    //     }
    // }
    // can.write_image(argv[2]);
    // hsi.write_image(argv[2]);
    // ycrcb.write_image(argv[2]);
    return 0;
}
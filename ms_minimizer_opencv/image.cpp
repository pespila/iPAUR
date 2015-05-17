#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "main.h"
#include "image.h"

using namespace cv;

gray_img *read_image_data(const char* filename) {
    if (filename == NULL) {
        printf("HI, geil das es dich gibt! :)\n");
        return NULL;
    } else {
        Mat opencv_image = imread(filename, 0); // force gray scale

        gray_img *image = (gray_img*)malloc(sizeof(gray_img));
        assert(image);

        image->image_data = alloc_image_data(opencv_image.rows, opencv_image.cols);
        image->approximation = alloc_image_data(opencv_image.rows, opencv_image.cols);
        image->iterative_data = alloc_double_array(opencv_image.rows, opencv_image.cols);
        image->image_height = opencv_image.rows;
        image->image_width = opencv_image.cols;
        image->image_type = opencv_image.type();

        for (int i = 0; i < opencv_image.rows; i++) {
            for (int j = 0; j < opencv_image.cols; j++) {
                image->image_data[j + i * opencv_image.cols] = opencv_image.at<uchar>(i, j);
                image->approximation[j + i * opencv_image.cols] = opencv_image.at<uchar>(i, j);
                image->iterative_data[j + i * opencv_image.cols] = (double)opencv_image.at<uchar>(i, j);
            }
        }

        return image;
    }
}

gray_img *initalize_raw_image(int rows, int cols, char type) {
    gray_img *image = (gray_img*)malloc(sizeof(gray_img));
    assert(image);

    image->image_data = alloc_image_data(rows, cols);
    image->iterative_data = alloc_double_array(rows, cols);
    image->image_height = rows;
    image->image_width = cols;
    image->image_type = type;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            image->image_data[j + i * cols] = 0;
            image->iterative_data[j + i * cols] = 0.0;
        }
    }

    return image;
}

void write_image_data(gray_img* image, const char* filename) {
    Mat img(image->image_height, image->image_width, image->image_type);
    for (int i = 0; i < image->image_height; i++) {
        for (int j = 0; j < image->image_width; j++) {
            img.at<uchar>(i, j) = image->approximation[j + i * image->image_width];
        }
    }
    imwrite(filename, img);
}

void destroy_image(gray_img* image) {
    assert(image);

    free(image->image_data);
    free(image->iterative_data);
    free(image);
}
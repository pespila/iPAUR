#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "main.h"
#include "image.h"

using namespace cv;

color_img *read_image_data(const char* filename) {
    if (filename == NULL) {
        printf("HI, geil das es dich gibt! :)\n");
        return NULL;
    } else {
        Mat opencv_image = imread(filename);

        color_img *image = (color_img*)malloc(sizeof(color_img));
        assert(image);

        image->red_data = (unsigned char*)malloc(opencv_image.rows*opencv_image.cols*sizeof(unsigned char));
        image->green_data = (unsigned char*)malloc(opencv_image.rows*opencv_image.cols*sizeof(unsigned char));
        image->blue_data = (unsigned char*)malloc(opencv_image.rows*opencv_image.cols*sizeof(unsigned char));
        image->red_approximation = (unsigned char*)malloc(opencv_image.rows*opencv_image.cols*sizeof(unsigned char));
        image->green_approximation = (unsigned char*)malloc(opencv_image.rows*opencv_image.cols*sizeof(unsigned char));
        image->blue_approximation = (unsigned char*)malloc(opencv_image.rows*opencv_image.cols*sizeof(unsigned char));
        image->red_iterative_data = (double*)malloc(opencv_image.rows*opencv_image.cols*sizeof(double));
        image->green_iterative_data = (double*)malloc(opencv_image.rows*opencv_image.cols*sizeof(double));
        image->blue_iterative_data = (double*)malloc(opencv_image.rows*opencv_image.cols*sizeof(double));
        image->image_height = opencv_image.rows;
        image->image_width = opencv_image.cols;
        image->image_type = opencv_image.type();

        for (int i = 0; i < opencv_image.rows; i++) {
            for (int j = 0; j < opencv_image.cols; j++) {
                image->red_data[j + i * opencv_image.cols] = opencv_image.at<Vec3b>(i, j)[2];
                image->green_data[j + i * opencv_image.cols] = opencv_image.at<Vec3b>(i, j)[1];
                image->blue_data[j + i * opencv_image.cols] = opencv_image.at<Vec3b>(i, j)[0];
                image->red_approximation[j + i * opencv_image.cols] = opencv_image.at<Vec3b>(i, j)[2];
                image->green_approximation[j + i * opencv_image.cols] = opencv_image.at<Vec3b>(i, j)[1];
                image->blue_approximation[j + i * opencv_image.cols] = opencv_image.at<Vec3b>(i, j)[0];
                image->red_iterative_data[j + i * opencv_image.cols] = (double)opencv_image.at<Vec3b>(i, j)[2];
                image->green_iterative_data[j + i * opencv_image.cols] = (double)opencv_image.at<Vec3b>(i, j)[1];
                image->blue_iterative_data[j + i * opencv_image.cols] = (double)opencv_image.at<Vec3b>(i, j)[0];
            }
        }

        return image;
    }
}

color_img *initalize_raw_image(int rows, int cols, char type) {
    color_img *image = (color_img*)malloc(sizeof(color_img));
    assert(image);

    image->red_data = (unsigned char*)malloc(rows*cols*sizeof(unsigned char));
    image->green_data = (unsigned char*)malloc(rows*cols*sizeof(unsigned char));
    image->blue_data = (unsigned char*)malloc(rows*cols*sizeof(unsigned char));
    image->red_approximation = (unsigned char*)malloc(rows*cols*sizeof(unsigned char));
    image->green_approximation = (unsigned char*)malloc(rows*cols*sizeof(unsigned char));
    image->blue_approximation = (unsigned char*)malloc(rows*cols*sizeof(unsigned char));
    image->red_iterative_data = (double*)malloc(rows*cols*sizeof(double));
    image->green_iterative_data = (double*)malloc(rows*cols*sizeof(double));
    image->blue_iterative_data = (double*)malloc(rows*cols*sizeof(double));
    image->image_height = rows;
    image->image_width = cols;
    image->image_type = type;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            image->red_data[j + i * cols] = 0.0;
            image->green_data[j + i * cols] = 0.0;
            image->blue_data[j + i * cols] = 0.0;
            image->red_approximation[j + i * cols] = 0.0;
            image->green_approximation[j + i * cols] = 0.0;
            image->blue_approximation[j + i * cols] = 0.0;
            image->red_iterative_data[j + i * cols] = 0.0;
            image->green_iterative_data[j + i * cols] = 0.0;
            image->blue_iterative_data[j + i * cols] = 0.0;
        }
    }

    return image;
}

void write_image_data(color_img* image, const char* filename) {
    Mat img(image->image_height, image->image_width, image->image_type);
    for (int i = 0; i < image->image_height; i++) {
        for (int j = 0; j < image->image_width; j++) {
            img.at<Vec3b>(i, j)[0] = image->blue_approximation[j + i * image->image_width];
            img.at<Vec3b>(i, j)[1] = image->green_approximation[j + i * image->image_width];
            img.at<Vec3b>(i, j)[2] = image->red_approximation[j + i * image->image_width];
        }
    }
    imwrite(filename, img);
}

void destroy_image(color_img* image) {
    assert(image);

    free(image->red_data);
    free(image->green_data);
    free(image->blue_data);
    free(image->red_iterative_data);
    free(image->green_iterative_data);
    free(image->blue_iterative_data);
    free(image->red_approximation);
    free(image->green_approximation);
    free(image->blue_approximation);
    free(image);
}
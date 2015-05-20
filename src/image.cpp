#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
// #include "main.h"
#include "image.h"

using namespace cv;

gray_img *read_image_data(const char* filename) {
    if (filename == NULL) {
        printf("HI, geil das es dich gibt! :)\n");
        return NULL;
    } else {
        Mat opencv_image = imread(filename, 0);

        gray_img *image = (gray_img*)malloc(sizeof(gray_img));
        assert(image);

        image->approximation = (float*)malloc(opencv_image.rows*opencv_image.cols*sizeof(float));
        image->image_height = opencv_image.rows;
        image->image_width = opencv_image.cols;
        image->image_type = opencv_image.type();

        for (int i = 0; i < opencv_image.rows; i++) {
            for (int j = 0; j < opencv_image.cols; j++) {
                image->approximation[j + i * opencv_image.cols] = opencv_image.at<uchar>(i, j);
            }
        }

        return image;
    }
}

gray_img *initalize_raw_image(int rows, int cols, char type) {
    gray_img *image = (gray_img*)malloc(sizeof(gray_img));
    assert(image);

    image->image_height = rows;
    image->image_width = cols;
    image->image_type = type;
    image->approximation = (float*)malloc(image->image_height*image->image_width*sizeof(float));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            image->approximation[j + i * image->image_width] = 0.0;
        }
    }

    return image;
}

void write_image_data(gray_img* image, const char* filename) {
    Mat img(image->image_height, image->image_width, image->image_type);
    for (int i = 0; i < image->image_height; i++) {
        for (int j = 0; j < image->image_width; j++) {
            img.at<uchar>(i, j) = (unsigned char)image->approximation[j + i * image->image_width];
        }
    }
    imwrite(filename, img);
}

void destroy_image(gray_img* image) {
    assert(image);
    free(image->approximation);
    free(image);
}

color_img *read_image_data_color(const char* filename) {
    if (filename == NULL) {
        printf("HI, geil das es dich gibt! :)\n");
        return NULL;
    } else {
        Mat opencv_image = imread(filename);

        color_img *image = (color_img*)malloc(sizeof(color_img));
        assert(image);

        image->red = (float*)malloc(opencv_image.rows*opencv_image.cols*sizeof(float));
        image->green = (float*)malloc(opencv_image.rows*opencv_image.cols*sizeof(float));
        image->blue = (float*)malloc(opencv_image.rows*opencv_image.cols*sizeof(float));
        image->image_height = opencv_image.rows;
        image->image_width = opencv_image.cols;
        image->image_type = opencv_image.type();

        for (int i = 0; i < opencv_image.rows; i++) {
            for (int j = 0; j < opencv_image.cols; j++) {
                image->red[j + i * opencv_image.cols] = opencv_image.at<Vec3b>(i, j)[0];
                image->green[j + i * opencv_image.cols] = opencv_image.at<Vec3b>(i, j)[1];
                image->blue[j + i * opencv_image.cols] = opencv_image.at<Vec3b>(i, j)[2];
            }
        }

        return image;
    }
}

color_img *initalize_raw_image_color(int rows, int cols, char type) {
    color_img *image = (color_img*)malloc(sizeof(color_img));
    assert(image);
    image->image_height = rows;
    image->image_width = cols;
    image->image_type = type;
    image->red = (float*)malloc(image->image_height*image->image_width*sizeof(float));
    image->green = (float*)malloc(image->image_height*image->image_width*sizeof(float));
    image->blue = (float*)malloc(image->image_height*image->image_width*sizeof(float));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            image->red[j + i * image->image_width] = 0.0;
            image->green[j + i * image->image_width] = 0.0;
            image->blue[j + i * image->image_width] = 0.0;
        }
    }

    return image;
}

void write_image_data_color(color_img* image, const char* filename) {
    Mat img(image->image_height, image->image_width, image->image_type);
    for (int i = 0; i < image->image_height; i++) {
        for (int j = 0; j < image->image_width; j++) {
            img.at<cv::Vec3b>(i, j)[2] = (unsigned char)image->blue[j + i * image->image_width];
            img.at<cv::Vec3b>(i, j)[1] = (unsigned char)image->green[j + i * image->image_width];
            img.at<cv::Vec3b>(i, j)[0] = (unsigned char)image->red[j + i * image->image_width];
        }
    }
    imwrite(filename, img);
}

void destroy_image_color(color_img* image) {
    assert(image);
    free(image->red);
    free(image->green);
    free(image->blue);
    free(image);
}
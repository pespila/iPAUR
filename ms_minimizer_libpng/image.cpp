#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <png.h>
#include "image.h"

gray_img *read_image_data(const char* filename) {
    unsigned int sig_read = 0;
    png_structp png_ptr;
    png_infop info_ptr;
    png_bytep * row_pointers;
    gray_img *image = (gray_img*)malloc(sizeof(gray_img));
    FILE *fp = fopen(filename, "rb");

    if (!fp)
        printf("Cannot open file: %s\n", filename);

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (png_ptr == NULL)
        fclose(fp);

    info_ptr = png_create_info_struct(png_ptr);
    if (info_ptr == NULL) {
        fclose(fp);
        png_destroy_read_struct(&png_ptr, png_infopp_NULL, png_infopp_NULL);
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, png_infopp_NULL);
        fclose(fp);
    }

    png_init_io(png_ptr, fp);

    png_set_sig_bytes(png_ptr, sig_read);

    png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, png_voidp_NULL);

    image->image_height = info_ptr->height;
    image->image_width = info_ptr->width;
    image->approximation = (float*)malloc(info_ptr->height*info_ptr->width*sizeof(float));
    image->bit_depth = info_ptr->bit_depth;
    image->color_type = info_ptr->color_type;

    row_pointers = png_get_rows(png_ptr, info_ptr);
    for (int i = 0; i < image->image_height; i++) {
        for (int j = 0; j < image->image_width; j++) {
            image->approximation[j + i * image->image_width] = (float)*(row_pointers[i] + j*info_ptr->pixel_depth/8);
        }
    }

    png_destroy_read_struct(&png_ptr, &info_ptr, png_infopp_NULL);

    fclose(fp);
    return image;
}

void write_image_data(gray_img* image, const char* filename) {
    int i, j;
    FILE *fp = fopen(filename, "wb");

    if (!fp)
        printf("Cannot open file: %s\n", filename);

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (png_ptr == NULL)
        fclose(fp);

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (info_ptr == NULL) {
        png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
        fclose(fp);
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
    }

    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, image->image_width, image->image_height,
                image->bit_depth, image->color_type, PNG_INTERLACE_NONE,
                PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
    
    png_write_info(png_ptr, info_ptr);

    png_bytepp str = (png_bytep*)malloc(sizeof(png_bytep) * image->image_height);
    for (i = 0; i < image->image_height; i++) {
        str[i] = (png_byte*)malloc(sizeof(png_bytep) * image->image_width);
    }

    for (i = 0; i < image->image_height; i++) {
        png_byte* row = str[i];
        for (j = 0; j < image->image_width; j++) {
            png_byte* ptr = &row[j];
            ptr[0] = image->approximation[j + i * image->image_width];
        }
    }

    png_write_image(png_ptr, str);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);

    for (i = 0; i < image->image_height; i++)
        free(str[i]);
    free(str);
    
    fclose(fp);
}

gray_img *initalize_raw_image(int rows, int cols, char type) {
    gray_img *image = (gray_img*)malloc(sizeof(gray_img));
    assert(image);
    image->approximation = (float*)malloc(rows*cols*sizeof(float));
    image->image_height = rows;
    image->image_width = cols;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            image->approximation[j + i * cols] = 0;
        }
    }
    return image;
}

void destroy_image(gray_img* image) {
    assert(image);
    free(image->approximation);
    free(image);
}
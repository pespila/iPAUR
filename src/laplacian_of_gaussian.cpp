#include "laplacian_of_gaussian.h"

void laplacianOfGaussian(GrayscaleImage& src, GrayscaleImage& dst, float** filter, int radius) {
    int x = 0, y = 0, a = 0, b = 0, sum = 0;
    unsigned short height = src.get_height(), width = src.get_width();
    dst.reset_image(height, width, src.get_type());

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            sum = 0;
            for (int k = (-1)*radius; k <= radius; k++) {
                for (int l = (-1)*radius; l <= radius; l++) {
                    x = i+k;
                    y = j+l;
                    a = k+radius;
                    b = l+radius;
                    if (y < 0 && x < 0) {
                        sum += filter[a][b] * src.get_gray_pixel_at_position(0, 0);
                    } else if (y >= width && x < 0) {
                        sum += filter[a][b] * src.get_gray_pixel_at_position(0, width-1);
                    } else if (x >= height && y < 0) {
                        sum += filter[a][b] * src.get_gray_pixel_at_position(height-1, 0);
                    } else if (x >= height && y >= width) {
                        sum += filter[a][b] * src.get_gray_pixel_at_position(height-1, width-1);
                    } else if (x < 0 && y >= 0 && y < width) {
                        sum += filter[a][b] * src.get_gray_pixel_at_position(0, y);
                    } else if (y < 0 && x >= 0 && x < height) {
                        sum += filter[a][b] * src.get_gray_pixel_at_position(x, 0);
                    } else if (y >= width && x >= 0 && x < height) {
                        sum += filter[a][b] * src.get_gray_pixel_at_position(x, width-1);
                    } else if (x >= height && y >= 0 && y < width) {
                        sum += filter[a][b] * src.get_gray_pixel_at_position(height-1, y);
                    } else {
                        sum += filter[a][b] * src.get_gray_pixel_at_position(x, y);
                    }
                }
            }
            sum -= src.get_gray_pixel_at_position(i, j);
            sum = sum < 0 ? 0 : sum;
            sum = sum > 255 ? 255 : sum;
            dst.set_gray_pixel_at_position(i, j, sum);
        }
    }
}
#include "laplacian_of_gaussian.h"

void laplacian_of_gaussian(GrayscaleImage& src, GrayscaleImage& dst, float** filter, int radius) {
    unsigned short height = src.get_height();
    unsigned short width = src.get_width();
    dst.reset_image(height, width, src.get_type());
    int x = 0, y = 0;
    int a = 0, b = 0;
    int sum = 0;
    int i, j, k, l;

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            sum = 0;
            for (k = (-1) * radius; k <= radius; k++) {
                for (l = (-1) * radius; l <= radius; l++) {
                    x = i + k; y = j + l;
                    a = k + radius; b = l + radius;
                    if (y < 0 && x < 0) {
                        sum += filter[a][b] * src.get_pixel(0, 0);
                    } else if (y >= width && x < 0) {
                        sum += filter[a][b] * src.get_pixel(0, width - 1);
                    } else if (x >= height && y < 0) {
                        sum += filter[a][b] * src.get_pixel(height - 1, 0);
                    } else if (x >= height && y >= width) {
                        sum += filter[a][b] * src.get_pixel(height - 1, width - 1);
                    } else if (x < 0 && y >= 0 && y < width) {
                        sum += filter[a][b] * src.get_pixel(0, y);
                    } else if (y < 0 && x >= 0 && x < height) {
                        sum += filter[a][b] * src.get_pixel(x, 0);
                    } else if (y >= width && x >= 0 && x < height) {
                        sum += filter[a][b] * src.get_pixel(x, width - 1);
                    } else if (x >= height && y >= 0 && y < width) {
                        sum += filter[a][b] * src.get_pixel(height - 1, y);
                    } else {
                        sum += filter[a][b] * src.get_pixel(x, y);
                    }
                }
            }
            sum -= src.get_pixel(i, j);
            sum = sum < 0 ? 0 : sum;
            sum = sum > 255 ? 255 : sum;
            dst.set_pixel(i, j, sum);
        }
    }
}
#include "image.h"
#include "rgb.h"
#include "hsi.h"
#include "ycrcb.h"
#include "grayscale.h"
#include "type_conversion.h"
#include "bluring.h"
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
#include "util.h"
#include "huber_rof_model.h"
#include "tv_l1_model.h"
#include "inpainting.h"
#include "util_color.h"
#include "huber_rof_model_color.h"
#include "tv_l1_model_color.h"
#include "inpainting_color.h"
#include "primal_dual.h"

void run(int, const char*[]);

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        printf("ERROR: Pass in filenames for reading and writing!\n");
        return 1;
    }
    // int category = 0;
    // int algorithm = 0;
    // int colored_in = 0;
    // int colored_out = 0;


    // GrayscaleImage in, out;
    // in.read_image(argv[1]);
    // roberts_cross(in, out);
    // out.write_image(argv[2]);

    // scanf("%d\n", &category);
    // prewitt(in, out);
    // out.write_image(argv[2]);

    // scanf("%d\n", &category);
    // sobel(in, out);
    // out.write_image(argv[2]);
    
    // scanf("%d\n", &category);
    // canny(in, out, 0, 9);
    // out.write_image(argv[2]);

    

    // do {
    //     print_init();
    //     printf("Input Image:\n");
    //     colored_in = check_color_channels();
    //     printf("Output Image:\n");
    //     colored_out = check_color_channels();
    //     category = return_category();
    //     if (return_algorithm_choice(category, colored_in, colored_out, argv[1], argv[2])) printf("Algorithm successfull.\n");

    // }

    // int doit = 1;

    // do {
    //     printf("Hallo\n");
    //     scanf("%d", &doit);
    // } while (doit == 1);

    run(argc, argv);

    // GrayscaleImage gray;
    // synthimage(gray, 128, 128);
    // gray.write_image("../../img/crack_tip.png");

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

void synthimage(WriteableImage& gray, int height, int width) {
    gray.reset_image(height, width, CV_8UC1);
    int center_x = width/2;
    int center_y = height/2;

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int x = fabs(i - center_y);
            int y = fabs(j - center_x);
            float absval = sqrtf(x * x + y * y); // nothing...
            float angle = atan((float)(j+1)/(float)(i+1));
            // float absval = fabs(x * x + y * y); // crack_tip_cool.png
            // float absval = x + y;
            // printf("%f\n", sin(angle/2.0) * 180 / PI);
            gray.set_pixel(i, j, (short)(sqrtf(absval) * sin(angle/2.0)  * 180 / PI));
            // if (j % 10 == 0) {
            //     gray.set_pixel(i, j, 255);
            // }
        }
    }
}

// void print_init() {
//     printf("\nWelcome to iPAUR! Choose your prefered (type of) category:\n\n");
//     printf("[1] Bluring\n[2] Edge Detection\n[3] Dilatation\n[4] Erosion\n[5] Color conversion\n[6] Inverse Image\n\
// [7] Filtering\n[8] Open-Close Operator\n[9] Top-Hat Operator\n[10] Advanced Techniques\n");
// }

// int check_color_channels() {
//     int colored
//     printf("Of which type is your image?\n\n [0] Gray valued or \n [1] Color valued\n");
//     scanf("%d", &colored);
//     return colored;
// }

// int return_category() {
//     int category = 0;
//     printf("Your choice: ");
//     scanf("%d", &category);
//     return category;
// }

// int return_algorithm_choice(int category, int colored_in, int colored_out, const char* in_file, const char* out_file) {
//     int success = 0;
//     switch category: {
//         case 1:
//             success = run_bluring(colored_in, colored_out, in_file, out_file); return;
//         case 2:
//             success = run_edge_detection(colored_in, colored_out, in_file, out_file); return;
//         case 3:
//             success = run_dilatation(colored_in, colored_out, in_file, out_file); return;
//         case 4:
//             success = run_erosion(colored_in, colored_out, in_file, out_file); return;
//         case 5:
//             success = run_color_conversion(colored_in, colored_out, in_file, out_file); return;
//         case 6:
//             success = run_inverse_image(colored_in, colored_out, in_file, out_file); return;
//         case 7:
//             success = run_filtering(colored_in, colored_out, in_file, out_file); return;
//         case 8:
//             success = run_open_close_operator(colored_in, colored_out, in_file, out_file); return;
//         case 9:
//             success = run_top_hat_operator(colored_in, colored_out, in_file, out_file); return;
//         case 10:
//             success = run_ms_minimizer(colored_in, colored_out, in_file, out_file); return;
//         default:
//             return;
//     }
//     return success;
// }

// int run_bluring(int colored_in, int colored_out, const char* in_file, const char* out_file) {
//     int success = 0;
//     int blur_type = 0;
//     int radius = 0;
//     float sigma = 0.0;
//     printf("Which blur do you choose? [1] Box blur, [2] Gaussian Blur\n");
//     scanf("%d", &blur_type);
//     printf("Which radius of your filter should be applied? Radius: ");
//     scanf("%d\n", &radius);
//     if (blur_type == 2) {
//         printf("Choose the weighting sigma (floating point number) for the gaussian kernel: ");
//         scanf("%d\n", &sigma);
//     }
//     switch blur_type: {
//         case 1:

//     }
//     return success;
// }

// int run_edge_detection(int colored_in, int colored_out, const char* in_file, const char* out_file) {
//     int success = 0;

//     return success;
// }

// int run_dilatation(int colored_in, int colored_out, const char* in_file, const char* out_file) {
//     int success = 0;

//     return success;
// }

// int run_erosion(int colored_in, int colored_out, const char* in_file, const char* out_file) {
//     int success = 0;

//     return success;
// }

// int run_color_conversion(int colored_in, int colored_out, const char* in_file, const char* out_file) {
//     int success = 0;

//     return success;
// }

// int run_inverse_image(int colored_in, int colored_out, const char* in_file, const char* out_file) {
//     int success = 0;

//     return success;
// }

// int run_filtering(int colored_in, int colored_out, const char* in_file, const char* out_file) {
//     int success = 0;

//     return success;
// }

// int run_open_close_operator(int colored_in, int colored_out, const char* in_file, const char* out_file) {
//     int success = 0;

//     return success;
// }

// int run_top_hat_operator(int colored_in, int colored_out, const char* in_file, const char* out_file) {
//     int success = 0;

//     return success;
// }

// int run_ms_minimizer(int colored_in, int colored_out, const char* in_file, const char* out_file) {
//     int success = 0;

//     return success;
// }


void run(int argc, const char* argv[]) {
    printf("\nStarting algorithm. Just a few seconds please:\n");
    float start_watch = clock();
    if (atoi(argv[3]) == 1) {
        gray_img* gray;
        gray = read_image_data(argv[1]);
        if (atoi(argv[4]) == 1) {
            param* parameter = set_input_parameter(0.35, 0.7, 1.0, 0.05, 1);
            tv_l1_model(gray, parameter, argv[5], 200);
        } else if (atoi(argv[4]) == 2) {
            param* parameter = set_input_parameter(0.01, 8.0, 1.0, 0.0, gray->image_height * gray->image_width);
            huber_rof_model(gray, parameter, argv[5], 1000);
        } else if (atoi(argv[4]) == 3) {
            param* parameter = set_input_parameter(0.01, 8.0, 1.0, 0.05, gray->image_height * gray->image_width);
            huber_rof_model(gray, parameter, argv[5], 1000);
        } else if (atoi(argv[4]) == 4) {
            param* parameter = set_input_parameter(0.01, 128.0, 1.0, 0.0, gray->image_height * gray->image_width);
            image_inpainting(gray, parameter, argv[5], 1000);
        } else if (atoi(argv[4]) == 5) {
            param* parameter = set_input_parameter(0.35, 0.7, 1.0, 0.0, gray->image_height * gray->image_width);
            primal_dual(gray, parameter, argv[5], 500);
        } else {
            printf("Too few arguments!\n");
        }
        write_image_data(gray, argv[2]);
        destroy_image(gray);
    } else {
        color_img* color;
        color = read_image_data_color(argv[1]);
        if (atoi(argv[4]) == 1) {
            param* parameter = set_input_parameter(0.35, 0.7, 1.0, 0.05, 1);
            tv_l1_model_color(color, parameter, argv[5], 200);
        } else if (atoi(argv[4]) == 2) {
            param* parameter = set_input_parameter(0.01, 8.0, 1.0, 0.0, color->image_height * color->image_width);
            huber_rof_model_color(color, parameter, argv[5], 1000);
        } else if (atoi(argv[4]) == 3) {
            param* parameter = set_input_parameter(0.01, 8.0, 1.0, 0.05, color->image_height * color->image_width);
            huber_rof_model_color(color, parameter, argv[5], 1000);
        } else if (atoi(argv[4]) == 4) {
            param* parameter = set_input_parameter(0.01, 128.0, 1.0, 0.0, color->image_height * color->image_width);
            image_inpainting_color(color, parameter, argv[5], 1000);
        } else {
            printf("Too few arguments!\n");
        }
        write_image_data_color(color, argv[2]);
        destroy_image_color(color);
    }
    float stop_watch = clock();
    printf("Algorithm finished in %f seconds.\n", (stop_watch - start_watch)/CLOCKS_PER_SEC);
}
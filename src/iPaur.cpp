/* FIT THE (GOOD DEFAULT) PARAMETER TO THE ALGORTIHMS
    - ROF-Model: alpha = 0.f, lambda = 0.03f, tau = 0.25f, sigma = 1.0 / (tau * 8.0), theta = 1.f, L = sqrt(8), nu = 0.f, cartoon = -1
    - Huber-ROF-Model: alpha = 0.05, lambda = 0.03f, tau = 0.2f, sigma = 1.0 / (tau * 8.0), theta = 1.0, L = sqrt(8), nu = 0.f, cartoon = -1
    - Image Inpainting: alpha = 0.05, lambda = 32.0, tau = 0.01, sigma = 1.0 / (tau * 8.0), theta = 1.0, L = sqrt(8), nu = 0.f, cartoon = -1
    - TVL1-Model: alpha = 0.05, lambda = 0.7, tau = 0.35, sigma = 1.0 / (0.35 * 8.0), theta = 1.0, L = sqrt(8), nu = 5, cartoon = -1
    - Real-Time-Minimizer: alpha = 20.0, lambda = 0.1, tau = 0.25, sigma = 0.5, theta = 1.0, L = sqrt(8), nu = 5, cartoon = 1/0
*/

#include "Image.h"
#include "GrayscaleImage.h"
#include "RGBImage.h"
#include "RGBAImage.h"
#include "HSIImage.h"
#include "YCrCbImage.h"
#include "EdgeDetector.h"
#include "LinearFilter.h"
#include "MorphologicalFilter.h"
#include "TypeConversion.h"
#include "Parameter.h"
#include "Util.h"
#include "HuberROFModel.h"
#include "ImageInpainting.h"
#include "RealTimeMinimizer.h"
#include "TVL1Model.h"
#include "ROFModel.h"
// #include "iPaur.h"

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        printf("ERROR: Pass in filenames for reading and writing!\n");
        return 1;
    }

    printf("\nStarting algorithm. Just a few seconds please:\n");
    float start_watch = clock();

    GrayscaleImage<float> in, out;
    in.Read(argv[1]);
    float alpha = atof(argv[3]);
    float beta = atof(argv[4]);
    int iter = atoi(argv[5]);

    // TVL1Model<float> tvl1(in, iter);
    RealTimeMinimizer<float> rt(in, iter);
    // ROFModel<float> rof(in, iter);
    // HuberROFModel<float> rof(in, iter);
    // iPAUR ipaur(in, iter);

    // Parameter<float> par(0.f, alpha, 0.35, 1.0 / (0.35 * 8.0), 1.0, sqrt(8), 0.f, -1); // TVL1
    Parameter<float> par(alpha, beta, 0.25, 1.0 / (0.25 * 8.0), 1.0, sqrt(8), 0.f, -1); // RealTimeMinimizer
    // Parameter<float> par(0.f, alpha, beta, 1.0 / (beta * 8.0), 1.0, sqrt(8), 0.f, -1); // ROF
    // Parameter<float> par(0.25f, alpha, beta, 1.0 / (beta * 8.0), 1.0, sqrt(8), 0.f, -1); // HuberROF
    // Parameter par(0.f, alpha, 0.01, 1.0 / (0.01 * 8.0), 1.0, sqrt(8), 0.f, -1); // IPAUR

    // tvl1.TVL1(in, out, par);
    rt.RTMinimizer(in, out, par);
    // rof.ROF(in, out, par);
    // rof.HuberROF(in, out, par);
    // ipaur.iPAURmodel(in, out, alpha, beta);
    
    out.Write(argv[2]);
    
    float stop_watch = clock();
    printf("Algorithm finished in %f seconds.\n", (stop_watch - start_watch)/CLOCKS_PER_SEC);
    
    return 0;
}
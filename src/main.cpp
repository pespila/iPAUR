/* FIT THE (GOOD DEFAULT) PARAMETER TO THE ALGORTIHMS
    - Huber-ROF-Model: alpha = 0.05, lambda = 32.0, tau = 0.01, sigma = 0.0, theta = 1.0, L = sqrt(8), nu = 5, cartoon = -1
    - Image Inpainting: alpha = 0.05, lambda = 32.0, tau = 0.01, sigma = 0.0, theta = 1.0, L = sqrt(8), nu = 5, cartoon = -1
    - TVL1-Model: alpha = 0.05, lambda = 0.7, tau = 0.35, sigma = 1.0 / (0.35 * 8.0), theta = 1.0, L = sqrt(8), nu = 5, cartoon = -1
    - Real-Time-Minimizer: alpha = 20.0, lambda = 0.1, tau = 0.25, sigma = 0.5, theta = 1.0, L = sqrt(8), nu = 5, cartoon = 1/0
*/

#include "main.h"

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        printf("ERROR: Pass in filenames for reading and writing!\n");
        return 1;
    }

    printf("\nStarting algorithm. Just a few seconds please:\n");
    float start_watch = clock();
    GrayscaleImage in, out;
    in.Read(argv[1]);
    RealTimeMinimizer rtm(in, 50);
    PrimalDualAlgorithm pd(in, 16, 100);
    // Parameter par;
    // rtm.RTMinimizer(in, out, par);
    Parameter par(0.0, 0.1, 1.0/sqrt(12), 1.0/sqrt(12), 1.0, sqrt(12), 5.0, -1);
    pd.PrimalDual(in, out, par, 10);
    out.Write(argv[2]);
    float stop_watch = clock();
    printf("Algorithm finished in %f seconds.\n", (stop_watch - start_watch)/CLOCKS_PER_SEC);
    
    return 0;
}
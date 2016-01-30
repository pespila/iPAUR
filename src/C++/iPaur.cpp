#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include "Image.h"
#include "EdgeDetector.h"
#include "LinearFilter.h"
#include "MorphologicalFilter.h"
#include "TypeConversion.h"
#include "HuberROFModel.h"
#include "ImageInpainting.h"
#include "RealTimeMinimizer.h"
#include "TVL1Model.h"
#include "ROFModel.h"
#include "URModel.h"
#include "iPaurModel.h"

using namespace std;

// parameter processing
template<typename T>
bool getParam(string param, T &var, int argc, const char* argv[]) {
    for(int i = (argc-1); i >= 1; i--) {
        if (argv[i][0] != '-') continue;
        if (param.compare(argv[i]+1) == 0) {
            if (!(i+1 < argc)) continue;
            stringstream ss;
            ss << argv[i+1];
            ss >> var;
            return (bool)ss;
        }
    }
    return false;
}

template<typename aType>
aType MSE(Image<aType>& src, Image<aType>& dst) {
    aType h = (aType)1 / (aType)(src.Height()*src.Width()*src.Channels());
    aType mse = 0;
    for (int k = 0; k < src.Channels(); k++) {
        for (int i = 0; i < src.Height(); i++) {
            for (int j = 0; j < src.Width(); j++) {
                mse += pow(src.Get(i, j, k) - dst.Get(i, j, k), 2);
            }
        }
    }
    return mse*h;
}

template<typename aType>
aType PSNR(Image<aType>& src, Image<aType>& dst) {
    return 10*log10((255*255) / MSE(src, dst));
}

int main(int argc, const char* argv[]) {
    string err_msg = "Usage: ./iPaur -i <image_in> -o <image_out> [-model <model>] [-iter <iterations>] [-tl <lower_threshold>] [-tu <upper_threshold>] [-radius <radius>] [-alpha <alpha>] [-beta <beta>] [-lambda <lambda>] [-tau <tau>] [-gray <gray>]";
    if (argc <= 1) {
        cout << err_msg << endl;
        return 1;
    }

    string input = "";
    bool retI = getParam("i", input, argc, argv);
    if (!retI) cerr << "ERROR I: no input image specified" << endl;
    cout << "Image input path: " << input << endl;
    
    string output = "";
    bool retO = getParam("o", output, argc, argv);
    if (!retO) cerr << "ERROR II: no output image specified" << endl;
    cout << "Image output path: " << output << endl;

    string inputC = "";
    getParam("c", inputC, argc, argv);
    cout << "Image inputC path: " << inputC << endl;
    
    string model = "";
    getParam("model", model, argc, argv);
    cout << "Model: " << model << endl;

    int iter = 10000;
    getParam("iter", iter, argc, argv);
    cout << "Number of iterations: " << iter << endl;

    int tl = 50;
    getParam("tl", tl, argc, argv);
    cout << "Lower Threshold: " << tl << endl;

    int tu = 150;
    getParam("tu", tu, argc, argv);
    cout << "Upper Threshold: " << tu << endl;

    int radius = 1;
    getParam("radius", radius, argc, argv);
    cout << "Radius: " << radius << endl;

    float sigma = 2.4f;
    getParam("sigma", sigma, argc, argv);
    cout << "sigma = " << sigma << endl;

    float alpha = 0.03f;
    getParam("alpha", alpha, argc, argv);
    cout << "alpha = " << alpha << endl;

    float beta = 0.7f;
    getParam("beta", beta, argc, argv);
    cout << "beta = " << beta << endl;

    float gamma = 0.f;
    getParam("gamma", gamma, argc, argv);
    cout << "gamma = " << gamma << endl;

    float lambda = 0.7f;
    getParam("lambda", lambda, argc, argv);
    cout << "lambda = " << lambda << endl;

    float tau = 0.25f;
    getParam("tau", tau, argc, argv);
    cout << "tau = " << tau << endl;

    bool gray = false;
    getParam("gray", gray, argc, argv);
    if (gray) {
        cout << "Output image will be written as gray-scaled image." << endl;
    }

    Image<float> in(input, gray);
    Image<float> out;

    float start_watch = clock();

    if (model.compare("rof") == 0) {
        printf("\nStarting ROF Model. Just a few seconds please:\n");
        ROFModel<float> rof(in, iter);
        rof.ROF(in, out, lambda, tau);
    } else if (model.compare("tvl1") == 0) {
        printf("\nStarting TVL1 Model. Just a few seconds please:\n");
        TVL1Model<float> tvl1(in, iter);
        tvl1.TVL1(in, out, lambda, tau);
    } else if (model.compare("realtime") == 0) {
        printf("\nStarting Real-Time Minimizer. Just a few seconds please:\n");
        RealTimeMinimizer<float> rt(in, iter);
        rt.RTMinimizer(in, out, alpha, lambda, gamma, tau);
    } else if (model.compare("ur") == 0) {
        printf("\nStarting UR Model. Just a few seconds please:\n");
        URModel<float> ur(in, iter);
        // ur.UR(in, out, alpha, beta, gamma, lambda, tau);
        ur.UR(in, out, alpha, beta, tau);
    } else if (model.compare("ipaur") == 0) {
        printf("\nStarting iPaur Model. Just a few seconds please:\n");
        iPaurModel<float> ipaur(in, iter);
        ipaur.iPaur(in, out, alpha, beta, gamma, lambda, tau);
    } else if (model.compare("inpaint") == 0) {
        printf("\nStarting Inpainting. Just a few seconds please:\n");
        ImageInpainting<float> ii(in, iter);
        ii.Inpaint(in, out, lambda, tau);
    } else if (model.compare("rgb2hsi") == 0) {
        printf("\nStarting RGB2HSI. Just a few seconds please:\n");
        TypeConversion<float> tp;
        tp.RGB2HSI(in, out);
    } else if (model.compare("rgb2ycrcb") == 0) {
        printf("\nStarting RGB2YCrCb. Just a few seconds please:\n");
        TypeConversion<float> tp;
        tp.RGB2YCrCb(in, out);
    } else if (model.compare("rgb2gray") == 0) {
        printf("\nStarting RGB2Gray. Just a few seconds please:\n");
        TypeConversion<float> tp;
        tp.RGB2Gray(in, out);
    } else if (model.compare("gray2rgb") == 0) {
        printf("\nStarting Gray2RGB. Just a few seconds please:\n");
        TypeConversion<float> tp;
        tp.Gray2RGB(in, out);
    } else if (model.compare("prewitt") == 0) {
        printf("\nStarting Prewitt Edge Detector. Just a few seconds please:\n");
        EdgeDetector<float> ed(in);
        ed.Prewitt(in, out);
    } else if (model.compare("sobel") == 0) {
        printf("\nStarting Sobel Edge Detector. Just a few seconds please:\n");
        EdgeDetector<float> ed(in);
        ed.Sobel(in, out);
    } else if (model.compare("laplace") == 0) {
        printf("\nStarting Laplace Edge Detector. Just a few seconds please:\n");
        EdgeDetector<float> ed(in);
        ed.Laplace(in, out);
    } else if (model.compare("robertscross") == 0) {
        printf("\nStarting Robert's Cross Edge Detector. Just a few seconds please:\n");
        EdgeDetector<float> ed(in);
        ed.RobertsCross(in, out);
    } else if (model.compare("canny") == 0) {
        printf("\nStarting Canny Edge Detector. Just a few seconds please:\n");
        EdgeDetector<float> ed(in);
        ed.Canny(in, out, tl, tu);
    } else if (model.compare("inverse") == 0) {
        printf("\nStarting Inverse Image. Just a few seconds please:\n");
        MorphologicalFilter<float> mf(in);
        mf.Inverse(in, out);
    } else if (model.compare("erosion") == 0) {
        printf("\nStarting Erosion Filter. Just a few seconds please:\n");
        MorphologicalFilter<float> mf(in);
        mf.Erosion(in, out, radius);
    } else if (model.compare("dilatation") == 0) {
        printf("\nStarting Dilatation Filter. Just a few seconds please:\n");
        MorphologicalFilter<float> mf(in);
        mf.Dilatation(in, out, radius);
    } else if (model.compare("median") == 0) {
        printf("\nStarting Median Filter. Just a few seconds please:\n");
        MorphologicalFilter<float> mf(in);
        mf.Median(in, out, radius);
    } else if (model.compare("open") == 0) {
        printf("\nStarting Open Filter. Just a few seconds please:\n");
        MorphologicalFilter<float> mf(in);
        mf.Open(in, out, radius);
    } else if (model.compare("close") == 0) {
        printf("\nStarting Close Filter. Just a few seconds please:\n");
        MorphologicalFilter<float> mf(in);
        mf.Close(in, out, radius);
    } else if (model.compare("gauss") == 0) {
        printf("\nStarting Gauss Filter. Just a few seconds please:\n");
        LinearFilter<float> lf(in);
        lf.Gauss(in, out, radius, sigma);
    } else if (model.compare("binomial") == 0) {
        printf("\nStarting Binomial Filter. Just a few seconds please:\n");
        LinearFilter<float> lf(in);
        lf.Binomial(in, out, radius);
    } else if (model.compare("box") == 0) {
        printf("\nStarting Box Filter. Just a few seconds please:\n");
        LinearFilter<float> lf(in);
        lf.Box(in, out, radius);
    } else if (model.compare("duto") == 0) {
        printf("\nStarting Duto Filter. Just a few seconds please:\n");
        LinearFilter<float> lf(in);
        lf.Duto(in, out, radius, sigma, lambda);
    } else if (model.compare("mse") == 0) {
        printf("\nStarting MSE. Just a few seconds please:\n");
        Image<float> inC(inputC, gray);
        cout << "Estimated MSE for file at path " << input << " and file at path " << inputC << ": " << MSE(in, inC) << endl;
        out.Read(input, gray);
    } else if (model.compare("psnr") == 0) {
        printf("\nStarting PSNR. Just a few seconds please:\n");
        Image<float> inC(inputC, gray);
        cout << "Estimated PSNR for file at path " << input << " and file at path " << inputC << ": " << PSNR(in, inC) << endl;
        out.Read(input, gray);
    } else {
        out.Read(input, gray);
    }
    
    float stop_watch = clock();
    // cout << "Estimated MSE: " << MSE(in, out) << endl;
    // cout << "Estimated PSNR: " << PSNR(in, out) << " db" << endl;
    cout << "Estimated Run-Time: " << (stop_watch - start_watch)/CLOCKS_PER_SEC << endl << endl;
    
    out.Write(output);
    
    return 0;
}
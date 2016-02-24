// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Summer Semester 2015, September 7 - October 6
// ###
// ###
// ### Thomas Moellenhoff, Robert Maier, Caner Hazirbas
// ###
// ###
// ###
// ### THIS FILE IS SUPPOSED TO REMAIN UNCHANGED
// ###
// ###

#include "aux.h"
#include <iostream>
#include <stdio.h>
using namespace std;

void parameterToFile(string filename,int repeats,bool gray,int level,float tau,float sigma,float lambda,float nu,int w,int h,int nc,size_t available,size_t total,float t,int iter) {
    FILE *file;
    file = fopen(filename.c_str(), "w");
    if(file == NULL)
        printf("ERROR: Could not open file!");
    else {
        fprintf(file, "image: %d x %d x %d\n", w, h, nc);
        fprintf(file,"repeats: %d\n", repeats);
        fprintf(file,"gray: %d\n", gray);
        fprintf(file,"level: %d\n", level);
        fprintf(file,"tau: %f\n", tau);
        fprintf(file,"sigma: %f\n", sigma);
        fprintf(file,"lambda: %f\n", lambda);
        fprintf(file,"nu: %f\n", nu);
        fprintf(file, "GPU Memory: %zd - %zd = %f GB\n", total, available, (total-available)/pow(10,9));
        fprintf(file, "time: %f s\n", t);
        fprintf(file, "iterations: %d\n", iter);
    }
    fclose (file);
}

void parameterToConsole(string filename,int repeats,bool gray,int level,float tau,float sigma,float lambda,float nu,int w,int h,int nc,size_t available,size_t total,float t,int iter) {
    printf( "image: %d x %d x %d\n", w, h, nc);
    printf("repeats: %d\n", repeats);
    printf("gray: %d\n", gray);
    printf("level: %d\n", level);
    printf("tau: %f\n", tau);
    printf("sigma: %f\n", sigma);
    printf("lambda: %f\n", lambda);
    printf("nu: %f\n", nu);
    printf( "GPU Memory: %zd - %zd = %f GB\n", total, available, (total-available)/pow(10,9));
    printf( "time: %f s\n", t);
    printf( "iterations: %d\n", iter);
}

__device__ float bound(float x1, float x2, float lambda, float k, float l, float f)
{
    return 0.25f * (x1*x1 + x2*x2) - lambda * pow(k / l - f, 2);
}

__device__ float interpolate(float k, float uk0, float uk1, float l)
{
    return (k + (0.5 - uk0) / (uk1 - uk0)) / l;
}

__device__ void on_parabola(float* u1, float* u2, float* u3, float x1, float x2, float x3, float f, float L, float lambda, float k, int j, float l)
{
    float y = x3 + lambda * pow(k / l - f, 2);
    float norm = sqrtf(x1*x1+x2*x2);
    float v = 0.f;
    float a = 2.f * 0.25f * norm;
    float b = 2.f / 3.f * (1.f - 2.f * 0.25f * y);
    float d = b < 0 ? (a - pow(sqrt(-b), 3)) * (a + pow(sqrt(-b), 3)) : a*a + b*b*b;
    float c = pow((a + sqrt(d)), 1.f/3.f);
    if (d >= 0) {
        v = c == 0 ? 0.f : c - b / c;
    } else {
        v = 2.f * sqrt(-b) * cos((1.f / 3.f) * acos(a / (pow(sqrt(-b), 3))));
    }
    u1[j] = norm == 0 ? 0.f : (v / (2.0 * 0.25f)) * x1 / norm;
    u2[j] = norm == 0 ? 0.f : (v / (2.0 * 0.25f)) * x2 / norm;
    u3[j] = bound(u1[j], u2[j], lambda, k, l, f);
}

__global__ void project_on_parabola(float* u1, float* u2, float* u3, float* v1, float* v2, float* v3, float* img, float L, float lambda, int k, int w, int h, int l)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    
    if (x < w && y < h && z < l)
    {
        int i = x + w * y + w * h * z;
        int j = x + w * y + w * h * z + (k-1) * w * h * l;
        u1[i] = u1[j];
        u2[i] = u2[j];
        u3[i] = u3[j];
    }

    if (x < w && y < h && z < l)
    {
        int index = x + w * y;
        int i = x + w * y + w * h * z + (k-1) * w * h * l;
        int j = x + w * y + w * h * z + k * w * h * l;

        float f = img[index];
        float x1 = u1[i] - v1[j];
        float x2 = u2[i] - v2[j];
        float x3 = u3[i] - v3[j];
        float bound_val = bound(x1, x2, lambda, (z+1.f), l, f);

        if (x3 < bound_val) {
            on_parabola(u1, u2, u3, x1, x2, x3, f, L, lambda, (z+1.f), j, l);
        } else {
            u1[j] = x1;
            u2[j] = x2;
            u3[j] = x3;
        }
    }

    if (x < w && y < h && z < l)
    {
        int i = x + w * y + w * h * z + k * w * h * l;
        int j = x + w * y + w * h * z + (k-1) * w * h * l;
        v1[i] = u1[i] - (u1[j] - v1[i]);
        v2[i] = u2[i] - (u2[j] - v2[i]);
        v3[i] = u3[i] - (u3[j] - v3[i]);
    }
}

__global__ void soft_shrinkage(float* u1, float* u2, float* u3, float* v1, float* v2, float* v3, float nu, int k1, int k2, int P, int w, int h, int l)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x < w && y < h)
    {
        int count_p = 2;
        float K;         
            
        for (int k1 = 0; k1 < l; k1++)
        {
            for (int k2 = k1; k2 < l; k2++)
            {
                K = (float)(k2 - k1 + 1);
                int i, j;
                float s1 = 0.f;
                float s2 = 0.f;
                float s01 = 0.f;
                float s02 = 0.f;
                float x1 = 0.f;
                float x2 = 0.f;

                for (int k = k1; k <= k2; k++)
                {
                    i = x + w * y + k * w * h + (count_p-1) * w * h * l;
                    j = x + w * y + k * w * h + count_p * w * h * l;
                    x1 = u1[i] - v1[j];
                    x2 = u2[i] - v2[j];
                    s01 += x1;
                    s02 += x2;
                }

                float norm = sqrtf(s01*s01+s02*s02);

                s1 = norm <= nu ? s01 : (nu * s01 / norm);
                s2 = norm <= nu ? s02 : (nu * s02 / norm);

                for (int k = 0; k < l; k++)
                {
                    i = x + w * y + k * w * h + (count_p-1) * w * h * l;
                    j = x + w * y + k * w * h + count_p * w * h * l;
                    x1 = u1[i] - v1[j];
                    x2 = u2[i] - v2[j];
                    if (k >= k1 && k <= k2) {
                        u1[j] = x1 + (s1 - s01) / K;
                        u2[j] = x2 + (s2 - s02) / K;
                    } else {
                        u1[j] = x1;
                        u2[j] = x2;
                    }
                    u3[j] = u3[i] - v3[j];
                }
                for (int z = 0; z < l; z++)
                {
                    int i = x + w * y + w * h * z + count_p * w * h * l;
                    int j = x + w * y + w * h * z + (count_p-1) * w * h * l;
                    v1[i] = u1[i] - (u1[j] - v1[i]);
                    v2[i] = u2[i] - (u2[j] - v2[i]);
                    v3[i] = u3[i] - (u3[j] - v3[i]);
                }
                count_p++;
            }
        }
    }
}

__global__ void init(float* u,float* ubar,float* p1,float* p2,float* p3,float* f,int w,int h,int l,int nc) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (x < w && y < h) {
        int I;
        float img;

        for (int c = 0; c < nc; c++) {   
            img = f[x+y*w+c*w*h];

            for (int k = 0; k < l; k++) {
                I = x+y*w+k*w*h+c*w*h*l;
                
                u[I] = img;
                ubar[I] = img;
                p1[I] = 0.f;
                p2[I] = 0.f;
                p3[I] = 0.f;
            }
        }
    }
}

__global__ void isosurface(float* f,float* u,int w,int h,int l,int nc)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x < w && y < h)
    {
        float uk0, uk1, val;
        for (int c = 0; c < nc; c++)
        {
            for (int k = 0; k < l-1; k++)
            {
                uk0 = u[x+y*w+k*h*w+c*h*w*l];
                uk1 = u[x+y*w+(k+1)*h*w+c*h*w*l];
                if (uk0 > 0.5 && uk1 <= 0.5) {
                    val = interpolate(k+1, uk0, uk1, l);
                    break;
                } else {
                    val = uk1;
                }
            }
            f[x+y*w+c*h*w] = val;
        }
    }
}

__global__ void set_y(float* y1, float* y2, float* y3, float* u1, float* u2, float* u3, int w, int h, int l, int p)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    

    if (x < w && y < h && z < l)
    {
        int i = x + w * y + w * h * z;
        int j = x + w * y + w * h * z + (p-1) * w * h * l;
        y1[i] = u1[j];
        y2[i] = u2[j];
        y3[i] = u3[j];
    }
}

__global__ void set_u_v(float* u1,float* u2,float* u3,float* v1,float* v2,float* v3,float* ubar,float* p1, float* p2, float* p3,float sigma,int w,int h,int l,int p,int nc) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    
    if (x < w && y < h && z < l) {
        float x1, x2, x3, val;

        for (int c = 0; c < nc; c++) {
            int I = x+y*w+z*w*h+c*w*h*l;
            int J;

            val = ubar[I];
            x1 = (x+1<w) ? (ubar[(x+1)+y*w+z*w*h+c*w*h*l] - val) : 0.f;
            x2 = (y+1<h) ? (ubar[x+(y+1)*w+z*w*h+c*w*h*l] - val) : 0.f;
            x3 = (z+1<l) ? (ubar[x+y*w+(z+1)*w*h+c*w*h*l] - val) : 0.f;

            x1 = p1[I] + sigma * x1;
            x2 = p2[I] + sigma * x2;
            x3 = p3[I] + sigma * x3;

            for (int k = 0; k < p; k++) {
                J = x+y*w+z*w*h+k*w*h*l+c*w*h*l*p;

                u1[J] = k < p-1 ? 0.f : x1;
                u2[J] = k < p-1 ? 0.f : x2;
                u3[J] = k < p-1 ? 0.f : x3;

                v1[J] = 0.f;
                v2[J] = 0.f;
                v3[J] = 0.f;
                
            }
        }
    }
}

__global__ void clipping(float* u,float* un,float* p1,float* p2,float* p3,float tau,int w,int h,int l,int nc)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    
    if (x < w && y < h && z < l)
    {
        int I;
        float cur;
        float d1,d2,d3,D;
        for (int c = 0; c < nc; c++)
        {
            I = x+y*w+z*h*w+c*h*w*l;
            cur = u[I];
            un[I] = cur;
            d1 = (y+1<h ? p1[I] : 0.f) - (y>0 ? p1[x+(y-1)*w+z*h*w+c*h*w*l] : 0.f);
            d2 = (x+1<w ? p2[I] : 0.f) - (x>0 ? p2[(x-1)+y*w+z*h*w+c*h*w*l] : 0.f);
            d3 = (z+1<l ? p3[I] : 0.f) - (z>0 ? p3[x+y*w+(z-1)*h*w+c*h*w*l] : 0.f);
            D = cur+tau*(d1+d2+d3);
            if (z==0) {
                u[I]=1.f;
            } else if (z==l-1) {
                u[I]=0.f;
            } else {
                u[I]=fmin(1.f, fmax(0.f, D));
            }
        }
    }
}

__global__ void extrapolate(float* xbar, float* xcur, float* xn, int w, int h, int l)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    
    int i = x + w * y + w * h * z;

    if (x < w && y < h && z < l) {
        xbar[i] = 2 * xn[i] - xcur[i];
    }
}

int main(int argc, char **argv)
{
    // Before the GPU can process your kernels, a so called "CUDA context" must be initialized
    // This happens on the very first call to a CUDA function, and takes some time (around half a second)
    // We will do it right here, so that the run time measurements are accurate
    cudaDeviceSynchronize();  CUDA_CHECK;

    // Reading command line parameters:

    if (argc <= 2) { cout << "Usage: " << argv[0] << " -i <image> -o <output_image> [-repeats <repeats>] [-gray]" << endl; return 1; }
    
    // input image
    string image = "";
    bool ret = getParam("i", image, argc, argv);
    if (!ret) cerr << "ERROR: no image specified" << endl;
    if (argc <= 1) { cout << "Usage: " << argv[0] << " -i <image> -o <output_image> -data <data.txt> -parm <parameter.txt> [-repeats <repeats>] [-gray]" << endl; return 1; }

    // output image
    string output = "";
    bool retO = getParam("o", output, argc, argv);
    if (!retO) cerr << "ERROR: no output image specified" << endl;

    // parameter values
    string parm = "";
    bool ret2 = getParam("parm", parm, argc, argv);
    
    // number of computation repetitions to get a better run time measurement
    int repeats = 1000;
    getParam("repeats", repeats, argc, argv);

    // number of computation repetitions to get a better run time measurement
    int dykstra = 10;
    getParam("dykstra", dykstra, argc, argv);
    
    // load the input image as grayscale if "-gray" is specifed
    bool gray = false;
    getParam("gray", gray, argc, argv);

    // load the input image as grayscale if "-gray" is specifed
    int level = 16;
    getParam("level", level, argc, argv);

    // load the input image as grayscale if "-gray" is specifed
    float L = sqrtf(12);
    getParam("L", L, argc, argv);

    // load the input image as grayscale if "-gray" is specifed
    // float tau = 1000;
    float tau = 1.f/L;
    getParam("tau", tau, argc, argv);
    
    // load the input image as grayscale if "-gray" is specifed
    float sigma = 1.f/(L*L*tau);
    getParam("sigma", sigma, argc, argv);

    // load the input image as grayscale if "-gray" is specifed
    float lambda = 0.1;
    getParam("lambda", lambda, argc, argv);

    // load the input image as grayscale if "-gray" is specifed
    float nu = 0.01f;
    getParam("nu", nu, argc, argv);
    
    // Load the input image using opencv (load as grayscale if "gray==true", otherwise as is (may be color or grayscale))
    cv::Mat mIn = cv::imread(image.c_str(), (gray? CV_LOAD_IMAGE_GRAYSCALE : -1));
    // check
    if (mIn.data == NULL) { cerr << "ERROR: Could not load image " << image << endl; return 1; }

    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn,CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;

    // get image dimensions
    int w = mIn.cols;         // width
    int h = mIn.rows;         // height
    int nc = mIn.channels();  // number of channels
    int dim = w*h*nc;
    int size = w*h*nc*level;
    int projections = level*(level-1)/2 + level + 2;
    // int projections = level * (level+1) / 2 + 1 + 1;
    int nbytes = size*sizeof(float);
    int nbyted = dim*sizeof(float);
    int nbytep = projections*size*sizeof(float);

    cv::Mat mOut(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers

    float nrj = 0.f;

    // allocate raw input image array
    // allocate raw input image array
    float* h_u = new float[(size_t)size];
    float* h_un = new float[(size_t)size];
    float* h_img  = new float[(size_t)dim];

    // allocate raw input image for GPU
    float* d_f; cudaMalloc(&d_f, nbyted); CUDA_CHECK;
    // float* d_imgOut;cudaMalloc(&d_imgOut, nbyted); CUDA_CHECK;

    float* d_u; cudaMalloc(&d_u, nbytes); CUDA_CHECK;
    float* d_ubar; cudaMalloc(&d_ubar, nbytes); CUDA_CHECK;
    float* d_un; cudaMalloc(&d_un, nbytes); CUDA_CHECK;

    float* d_p1; cudaMalloc(&d_p1, nbytes); CUDA_CHECK;
    float* d_p2; cudaMalloc(&d_p2, nbytes); CUDA_CHECK;
    float* d_p3; cudaMalloc(&d_p3, nbytes); CUDA_CHECK;

    float* d_u1; cudaMalloc(&d_u1, nbytep); CUDA_CHECK;
    float* d_u2; cudaMalloc(&d_u2, nbytep); CUDA_CHECK;
    float* d_u3; cudaMalloc(&d_u3, nbytep); CUDA_CHECK;

    float* d_v1; cudaMalloc(&d_v1, nbytep); CUDA_CHECK;
    float* d_v2; cudaMalloc(&d_v2, nbytep); CUDA_CHECK;
    float* d_v3; cudaMalloc(&d_v3, nbytep); CUDA_CHECK;

    size_t available, total;
    cudaMemGetInfo(&available, &total);

    // Init raw input image array
    // opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
    // But for CUDA it's better to work with layered images: rrr... ggg... bbb...
    // So we will convert as necessary, using interleaved "cv::Mat" for loading/saving/displaying, and layered "float*" for CUDA computations
    convert_mat_to_layered (h_img, mIn);

    // copy host memory
    cudaMemcpy(d_f, h_img, nbyted, cudaMemcpyHostToDevice); CUDA_CHECK;

    // launch kernel
    dim3 block = dim3(32, 8, 4);
    dim3 grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, (level + block.z - 1) / block.z);
    dim3 block_iso = dim3(32, 8, 1);
    dim3 grid_iso = dim3((w + block_iso.x - 1) / block_iso.x, (h + block_iso.y - 1) / block_iso.y, 1);

    Timer timer; timer.start();

    int count_p = projections;
    int iter;

    init <<<grid_iso, block_iso>>> (d_u,d_ubar,d_p1,d_p2,d_p3,d_f,w,h,level,nc);
    for (iter = 0; iter < repeats; iter++)
    {
        set_u_v <<<grid, block>>> (d_u1,d_u2,d_u3,d_v1,d_v2,d_v3,d_ubar,d_p1,d_p2,d_p3,sigma,w,h,level,projections,nc);
        for (int j = 0; j < dykstra; j++)
        {            
            count_p = 1;
            // set_u_zero <<<grid, block>>> (d_u1, d_u2, d_u3, w, h, level, projections);
            project_on_parabola <<<grid, block>>> (d_u1, d_u2, d_u3, d_v1, d_v2, d_v3, d_f, L, lambda, count_p, w, h, level);
            // update_v <<<grid, block>>> (d_v1, d_v2, d_v3, d_u1, d_u2, d_u3, w, h, level, count_p);
            // count_p++;
            
            // for (int k1 = 0; k1 < level; k1++)
            // {
            //     for (int k2 = k1; k2 < level; k2++)
            //     {
                    soft_shrinkage <<<grid_iso, block_iso>>> (d_u1, d_u2, d_u3, d_v1, d_v2, d_v3, nu, 0, 0, count_p, w, h, level);
                    // update_v <<<grid, block>>> (d_v1, d_v2, d_v3, d_u1, d_u2, d_u3, w, h, level, count_p);
                    // count_p++;
            //     }
            // }
        }
        set_y <<<grid, block>>> (d_p1, d_p2, d_p3, d_u1, d_u2, d_u3, w, h, level, projections);
        clipping <<<grid, block>>> (d_u,d_un,d_p1,d_p2,d_p3,tau,w,h,level,nc);
        // cudaMemcpy(h_u, d_u, nbytes, cudaMemcpyDeviceToHost); CUDA_CHECK;
        // cudaMemcpy(h_un, d_un, nbytes, cudaMemcpyDeviceToHost); CUDA_CHECK;
        // nrj = 0.f;
        // for (int i = 0; i < size; i++)
        // {
        //     nrj += fabs(h_u[i] - h_un[i]);
        // }
        // if (nrj/(w*h*level) <= 5*1E-5) break;
        extrapolate <<<grid, block>>> (d_ubar, d_un, d_u, w, h, level);
    }
    isosurface <<<grid_iso, block_iso>>> (d_f,d_u,w,h,level,nc);
    
    timer.end();  float t = timer.get();  // elapsed time in seconds

    cudaMemcpy(h_img, d_f, nbyted, cudaMemcpyDeviceToHost); CUDA_CHECK;
    // dualEnergy(data, h_energy, count);
    if (!ret2) {
        parameterToConsole(parm,repeats,gray,level,tau,sigma,lambda,nu,w,h,nc,available,total,t,iter);
    } else {
        parameterToFile(parm,repeats,gray,level,tau,sigma,lambda,nu,w,h,nc,available,total,t,iter);
    }

    // free GPU memory
    cudaFree(d_f); CUDA_CHECK;
    // cudaFree(d_f); CUDA_CHECK;
    
    cudaFree(d_u); CUDA_CHECK;
    cudaFree(d_ubar); CUDA_CHECK;
    cudaFree(d_un); CUDA_CHECK;

    cudaFree(d_p1); CUDA_CHECK;
    cudaFree(d_p2); CUDA_CHECK;
    cudaFree(d_p3); CUDA_CHECK;

    cudaFree(d_u1); CUDA_CHECK;
    cudaFree(d_u2); CUDA_CHECK;
    cudaFree(d_u3); CUDA_CHECK;

    cudaFree(d_v1); CUDA_CHECK;
    cudaFree(d_v2); CUDA_CHECK;
    cudaFree(d_v3); CUDA_CHECK;

    // show input image
    // showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

    // show output image: first convert to interleaved opencv format from the layered raw array
    convert_layered_to_mat(mOut, h_img);
    // showImage("Output", mOut, 100+w+40, 100);

    // wait for key inputs
    // cv::waitKey(0);

    // save input and result
    // cv::imwrite("image_input.png",mIn*255.f);  // "imwrite" assumes channel range [0,255]
    cv::imwrite(output, mOut*255.f);

    // free allocated arrays
    delete[] h_u;
    delete[] h_un;
    delete[] h_img;
    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}

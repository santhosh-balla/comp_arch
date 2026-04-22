#include "opencv2/imgcodecs.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#if 0
/**
 From https://docs.opencv.org/3.4.0/d3/dc1/tutorial_basic_linear_transform.html
 Check the webpage for description
 */
#include "opencv2/highgui.hpp"
using namespace std;
using namespace cv;
int main( int argc, char** argv )
{
    double alpha = 1.0; /*< Simple contrast control */
    int beta = 0;       /*< Simple brightness control */
    String imageName("../data/lena.jpg"); // by default
    if (argc > 1)
    {
        imageName = argv[1];
    }
    Mat image = imread( imageName );
    Mat new_image = Mat::zeros( image.size(), image.type() );
    cout << " Basic Linear Transforms " << endl;
    cout << "-------------------------" << endl;
    cout << "* Enter the alpha value [1.0-3.0]: "; cin >> alpha;
    cout << "* Enter the beta value [0-100]: ";    cin >> beta;
    for( int y = 0; y < image.rows; y++ ) {
        for( int x = 0; x < image.cols; x++ ) {
            for( int c = 0; c < 3; c++ ) {
                new_image.at<Vec3b>(y,x)[c] =
                  saturate_cast<uchar>( alpha*( image.at<Vec3b>(y,x)[c] ) + beta );
            }
        }
    }
    namedWindow("Original Image", WINDOW_AUTOSIZE);
    namedWindow("New Image", WINDOW_AUTOSIZE);
    imshow("Original Image", image);
    imshow("New Image", new_image);
    waitKey();
    return 0;
}
#endif

using namespace std;
using namespace cv;

#ifdef ENABLE_CUDA
bool convolution3x3_cuda(const unsigned char *input,
                         unsigned char *output,
                         int rows,
                         int cols,
                         const int *kernel,
                         int divisor,
                         int iterations,
                         float *kernel_ms,
                         float *total_ms,
                         string *err);
#endif

static bool get_kernel(const string &name, vector<int> &kernel, int &divisor)
{
    if (name == "lpf6")
    {
        kernel = {0, 1, 0, 1, 2, 1, 0, 1, 0};
        divisor = 6;
        return true;
    }
    if (name == "lpf9")
    {
        kernel = {1, 1, 1, 1, 1, 1, 1, 1, 1};
        divisor = 9;
        return true;
    }
    if (name == "lpf10")
    {
        kernel = {1, 1, 1, 1, 2, 1, 1, 1, 1};
        divisor = 10;
        return true;
    }
    if (name == "lpf16")
    {
        kernel = {1, 2, 1, 2, 4, 2, 1, 2, 1};
        divisor = 16;
        return true;
    }
    if (name == "lpf32")
    {
        kernel = {1, 4, 1, 4, 12, 4, 1, 4, 1};
        divisor = 32;
        return true;
    }
    if (name == "hpf1")
    {
        kernel = {0, -1, 0, -1, 5, -1, 0, -1, 0};
        divisor = 1;
        return true;
    }
    if (name == "hpf2")
    {
        kernel = {-1, -1, -1, -1, 9, -1, -1, -1, -1};
        divisor = 1;
        return true;
    }
    if (name == "hpf3")
    {
        kernel = {1, -2, 1, -2, 5, -2, 1, -2, 1};
        divisor = 1;
        return true;
    }
    return false;
}

static void convolution3x3_cpu(const Mat &input,
                               Mat &output,
                               const vector<int> &kernel,
                               int divisor,
                               int iterations,
                               double &elapsed_ms)
{
    Mat temp(input.rows, input.cols, input.type());
    auto start = chrono::high_resolution_clock::now();
    for (int it = 0; it < iterations; ++it)
    {
        input.copyTo(temp);
        for (int y = 1; y < input.rows - 1; ++y)
        {
            for (int x = 1; x < input.cols - 1; ++x)
            {
                for (int c = 0; c < 3; ++c)
                {
                    int sum = 0;
                    int k = 0;
                    for (int ky = -1; ky <= 1; ++ky)
                    {
                        for (int kx = -1; kx <= 1; ++kx)
                        {
                            const Vec3b &px = input.at<Vec3b>(y + ky, x + kx);
                            sum += static_cast<int>(px[c]) * kernel[k++];
                        }
                    }
                    sum /= divisor;
                    sum = max(0, min(255, sum));
                    temp.at<Vec3b>(y, x)[c] = static_cast<uchar>(sum);
                }
            }
        }
    }
    auto end = chrono::high_resolution_clock::now();
    elapsed_ms = chrono::duration<double, milli>(end - start).count();
    output = temp;
}

static double max_abs_diff(const Mat &a, const Mat &b)
{
    double maxv = 0.0;
    for (int y = 0; y < a.rows; ++y)
    {
        for (int x = 0; x < a.cols; ++x)
        {
            const Vec3b &pa = a.at<Vec3b>(y, x);
            const Vec3b &pb = b.at<Vec3b>(y, x);
            for (int c = 0; c < 3; ++c)
            {
                maxv = max(maxv, static_cast<double>(abs(static_cast<int>(pa[c]) - static_cast<int>(pb[c]))));
            }
        }
    }
    return maxv;
}

int main(int argc, char **argv)
{
    string image_name = "../data/lena.jpg";
    string kernel_name = "lpf10";
#ifdef ENABLE_CUDA
    string mode = "both";
#else
    string mode = "cpu";
#endif
    int iterations = 20;
    string prefix = "filter_out";

    if (argc > 1)
        image_name = argv[1];
    if (argc > 2)
        kernel_name = argv[2];
    if (argc > 3)
        mode = argv[3];
    if (argc > 4)
        iterations = stoi(argv[4]);
    if (argc > 5)
        prefix = argv[5];

    vector<int> kernel;
    int divisor = 1;
    if (!get_kernel(kernel_name, kernel, divisor))
    {
        cerr << "Invalid kernel. Use one of: lpf6 lpf9 lpf10 lpf16 lpf32 hpf1 hpf2 hpf3" << endl;
        return 1;
    }

    Mat image = imread(image_name, IMREAD_COLOR);
    if (image.empty())
    {
        cerr << "Failed to read image: " << image_name << endl;
        return 1;
    }

    Mat cpu_out;
    Mat gpu_out;
    bool ran_cpu = false;
    bool ran_gpu = false;

    if (mode == "cpu" || mode == "both")
    {
        double cpu_ms = 0.0;
        convolution3x3_cpu(image, cpu_out, kernel, divisor, iterations, cpu_ms);
        ran_cpu = true;
        cout << "CPU total ms: " << cpu_ms << endl;
        cout << "CPU avg ms/iter: " << (cpu_ms / iterations) << endl;
        imwrite(prefix + "_cpu.png", cpu_out);
    }

#ifdef ENABLE_CUDA
    if (mode == "gpu" || mode == "both")
    {
        gpu_out = Mat(image.rows, image.cols, image.type());
        float gpu_kernel_ms = 0.0f;
        float gpu_total_ms = 0.0f;
        string err;
        bool ok = convolution3x3_cuda(image.ptr<unsigned char>(),
                                      gpu_out.ptr<unsigned char>(),
                                      image.rows,
                                      image.cols,
                                      kernel.data(),
                                      divisor,
                                      iterations,
                                      &gpu_kernel_ms,
                                      &gpu_total_ms,
                                      &err);
        if (!ok)
        {
            cerr << "GPU run failed: " << err << endl;
            return 1;
        }
        ran_gpu = true;
        cout << "GPU kernel total ms: " << gpu_kernel_ms << endl;
        cout << "GPU kernel avg ms/iter: " << (gpu_kernel_ms / iterations) << endl;
        cout << "GPU end-to-end total ms: " << gpu_total_ms << endl;
        cout << "GPU end-to-end avg ms/iter: " << (gpu_total_ms / iterations) << endl;
        imwrite(prefix + "_gpu.png", gpu_out);
    }
#else
    if (mode == "gpu" || mode == "both")
    {
        cerr << "GPU mode requested but binary is not built with CUDA support." << endl;
        return 1;
    }
#endif

    if (ran_cpu && ran_gpu)
    {
        double diff = max_abs_diff(cpu_out, gpu_out);
        cout << "Max abs diff CPU vs GPU: " << diff << endl;
    }

    return 0;
}

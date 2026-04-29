#include "opencv2/imgcodecs.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <utility>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

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

__global__ static void convolution3x3_kernel(const unsigned char *input,
                                             unsigned char *output,
                                             int rows,
                                             int cols,
                                             const int *kernel,
                                             int divisor)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows)
    {
        return;
    }

    int idx = (y * cols + x) * 3;

    if (x == 0 || y == 0 || x == cols - 1 || y == rows - 1)
    {
        output[idx] = input[idx];
        output[idx + 1] = input[idx + 1];
        output[idx + 2] = input[idx + 2];
        return;
    }

    for (int c = 0; c < 3; ++c)
    {
        int sum = 0;
        int k = 0;
        for (int ky = -1; ky <= 1; ++ky)
        {
            for (int kx = -1; kx <= 1; ++kx)
            {
                int nidx = ((y + ky) * cols + (x + kx)) * 3 + c;
                sum += static_cast<int>(input[nidx]) * kernel[k++];
            }
        }
        sum /= divisor;
        if (sum < 0)
        {
            sum = 0;
        }
        if (sum > 255)
        {
            sum = 255;
        }
        output[idx + c] = static_cast<unsigned char>(sum);
    }
}

bool convolution3x3_cuda(const unsigned char *input,
                         unsigned char *output,
                         int rows,
                         int cols,
                         const int *kernel,
                         int divisor,
                         int iterations,
                         float *kernel_ms,
                         float *total_ms,
                         string *err)
{
    const size_t image_bytes = static_cast<size_t>(rows) * static_cast<size_t>(cols) * 3;
    const size_t kernel_bytes = 9 * sizeof(int);

    unsigned char *d_input = nullptr;
    unsigned char *d_output = nullptr;
    int *d_kernel = nullptr;

    cudaError_t st = cudaSuccess;

    st = cudaMalloc(&d_input, image_bytes);
    if (st != cudaSuccess)
    {
        if (err)
        {
            *err = cudaGetErrorString(st);
        }
        return false;
    }

    st = cudaMalloc(&d_output, image_bytes);
    if (st != cudaSuccess)
    {
        if (err)
        {
            *err = cudaGetErrorString(st);
        }
        cudaFree(d_input);
        return false;
    }

    st = cudaMalloc(&d_kernel, kernel_bytes);
    if (st != cudaSuccess)
    {
        if (err)
        {
            *err = cudaGetErrorString(st);
        }
        cudaFree(d_input);
        cudaFree(d_output);
        return false;
    }

    st = cudaMemcpy(d_kernel, kernel, kernel_bytes, cudaMemcpyHostToDevice);
    if (st != cudaSuccess)
    {
        if (err)
        {
            *err = cudaGetErrorString(st);
        }
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_kernel);
        return false;
    }

    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    cudaEvent_t ev_start;
    cudaEvent_t ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    float kernel_total = 0.0f;
    auto t0 = chrono::high_resolution_clock::now();

    for (int it = 0; it < iterations; ++it)
    {
        st = cudaMemcpy(d_input, input, image_bytes, cudaMemcpyHostToDevice);
        if (st != cudaSuccess)
        {
            if (err)
            {
                *err = cudaGetErrorString(st);
            }
            cudaEventDestroy(ev_start);
            cudaEventDestroy(ev_stop);
            cudaFree(d_input);
            cudaFree(d_output);
            cudaFree(d_kernel);
            return false;
        }

        cudaEventRecord(ev_start);
        convolution3x3_kernel<<<grid, block>>>(d_input, d_output, rows, cols, d_kernel, divisor);
        cudaEventRecord(ev_stop);
        st = cudaEventSynchronize(ev_stop);
        if (st != cudaSuccess)
        {
            if (err)
            {
                *err = cudaGetErrorString(st);
            }
            cudaEventDestroy(ev_start);
            cudaEventDestroy(ev_stop);
            cudaFree(d_input);
            cudaFree(d_output);
            cudaFree(d_kernel);
            return false;
        }

        st = cudaGetLastError();
        if (st != cudaSuccess)
        {
            if (err)
            {
                *err = cudaGetErrorString(st);
            }
            cudaEventDestroy(ev_start);
            cudaEventDestroy(ev_stop);
            cudaFree(d_input);
            cudaFree(d_output);
            cudaFree(d_kernel);
            return false;
        }

        float iter_kernel_ms = 0.0f;
        cudaEventElapsedTime(&iter_kernel_ms, ev_start, ev_stop);
        kernel_total += iter_kernel_ms;

        st = cudaMemcpy(output, d_output, image_bytes, cudaMemcpyDeviceToHost);
        if (st != cudaSuccess)
        {
            if (err)
            {
                *err = cudaGetErrorString(st);
            }
            cudaEventDestroy(ev_start);
            cudaEventDestroy(ev_stop);
            cudaFree(d_input);
            cudaFree(d_output);
            cudaFree(d_kernel);
            return false;
        }
    }

    auto t1 = chrono::high_resolution_clock::now();
    double total = chrono::duration<double, milli>(t1 - t0).count();

    if (kernel_ms)
    {
        *kernel_ms = kernel_total;
    }
    if (total_ms)
    {
        *total_ms = static_cast<float>(total);
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    return true;
}
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

static void convolution3x3_cpu_parallel(const Mat &input,
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
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
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

struct TimingResult
{
    string name;
    double total_ms = 0.0;
    double avg_ms = 0.0;
    double extra_ms = 0.0;
    string extra_label;
    bool has_extra = false;
};

static void write_performance_report(const string &prefix,
                                     const string &image_name,
                                     const string &kernel_name,
                                     int iterations,
                                     const vector<TimingResult> &results,
                                     const vector<pair<string, double>> &diffs)
{
    cout << "\n=== Performance Report ===\n";
    cout << "Output prefix: " << prefix << '\n';
    cout << "Image: " << image_name << '\n';
    cout << "Kernel: " << kernel_name << '\n';
    cout << "Iterations: " << iterations << '\n';
    cout << left << setw(22) << "Implementation"
         << right << setw(14) << "Total ms"
         << setw(18) << "Avg ms/iter"
         << setw(16) << "Extra ms"
         << setw(18) << "Extra label" << '\n';
    cout << string(88, '-') << '\n';
    for (const auto &result : results)
    {
        cout << left << setw(22) << result.name
             << right << setw(14) << fixed << setprecision(3) << result.total_ms
             << setw(18) << fixed << setprecision(3) << result.avg_ms
             << setw(16);
        if (result.has_extra)
        {
            cout << fixed << setprecision(3) << result.extra_ms;
        }
        cout << setw(18);
        if (result.has_extra)
        {
            cout << result.extra_label;
        }
        cout << '\n';
    }

    if (!diffs.empty())
    {
        cout << "\nComparisons:\n";
        for (const auto &diff : diffs)
        {
            cout << "  " << diff.first << ": " << fixed << setprecision(3) << diff.second << '\n';
        }
    }

    cout << "=== End Performance Report ===\n";
}

int main(int argc, char **argv)
{
    string image_name = "../data/lena.jpg";
    string kernel_name = "lpf10";
#ifdef ENABLE_CUDA
    string mode = "all";
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

    bool run_cpu = (mode == "cpu" || mode == "both" || mode == "all");
    bool run_cpu_parallel = (mode == "cpu-par" || mode == "cpu-parallel" || mode == "all");
    bool run_gpu = (mode == "gpu" || mode == "both" || mode == "all");

    Mat cpu_out;
    Mat cpu_parallel_out;
    Mat gpu_out;
    bool ran_cpu = false;
    bool ran_cpu_parallel = false;
    bool ran_gpu = false;
    vector<TimingResult> results;
    vector<pair<string, double>> diffs;

    if (run_cpu)
    {
        double cpu_ms = 0.0;
        convolution3x3_cpu(image, cpu_out, kernel, divisor, iterations, cpu_ms);
        ran_cpu = true;
        cout << "CPU total ms: " << cpu_ms << endl;
        cout << "CPU avg ms/iter: " << (cpu_ms / iterations) << endl;
        imwrite(prefix + "_cpu.png", cpu_out);
        results.push_back({"cpu_sequential", cpu_ms, cpu_ms / iterations, 0.0, "", false});
    }

    if (run_cpu_parallel)
    {
        double cpu_parallel_ms = 0.0;
        convolution3x3_cpu_parallel(image, cpu_parallel_out, kernel, divisor, iterations, cpu_parallel_ms);
        ran_cpu_parallel = true;
        cout << "CPU parallel total ms: " << cpu_parallel_ms << endl;
        cout << "CPU parallel avg ms/iter: " << (cpu_parallel_ms / iterations) << endl;
        imwrite(prefix + "_cpu_parallel.png", cpu_parallel_out);
        results.push_back({"cpu_parallel", cpu_parallel_ms, cpu_parallel_ms / iterations, 0.0, "", false});
    }

#ifdef ENABLE_CUDA
    if (run_gpu)
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
        results.push_back({"gpu_kernel", gpu_kernel_ms, gpu_kernel_ms / iterations, gpu_total_ms, "end_to_end_ms", true});
    }
#else
    if (run_gpu)
    {
        cerr << "GPU mode requested but binary is not built with CUDA support." << endl;
        if (!run_cpu && !run_cpu_parallel)
        {
            return 1;
        }
    }
#endif

    if (ran_cpu && ran_cpu_parallel)
    {
        double diff = max_abs_diff(cpu_out, cpu_parallel_out);
        diffs.push_back({"cpu_sequential_vs_cpu_parallel", diff});
        cout << "Max abs diff CPU sequential vs CPU parallel: " << diff << endl;
    }

    if (ran_cpu && ran_gpu)
    {
        double diff = max_abs_diff(cpu_out, gpu_out);
        diffs.push_back({"cpu_sequential_vs_gpu", diff});
        cout << "Max abs diff CPU vs GPU: " << diff << endl;
    }

    if (!results.empty())
    {
        write_performance_report(prefix, image_name, kernel_name, iterations, results, diffs);
    }

    return 0;
}

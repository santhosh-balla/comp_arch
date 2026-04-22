#include <cuda_runtime.h>
#include <chrono>
#include <string>

using namespace std;

__global__ static void convolution3x3_kernel(const unsigned char* input,
                                             unsigned char* output,
                                             int rows,
                                             int cols,
                                             const int* kernel,
                                             int divisor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows) return;

    int idx = (y * cols + x) * 3;

    if (x == 0 || y == 0 || x == cols - 1 || y == rows - 1) {
        output[idx] = input[idx];
        output[idx + 1] = input[idx + 1];
        output[idx + 2] = input[idx + 2];
        return;
    }

    for (int c = 0; c < 3; ++c) {
        int sum = 0;
        int k = 0;
        for (int ky = -1; ky <= 1; ++ky) {
            for (int kx = -1; kx <= 1; ++kx) {
                int nidx = ((y + ky) * cols + (x + kx)) * 3 + c;
                sum += static_cast<int>(input[nidx]) * kernel[k++];
            }
        }
        sum /= divisor;
        if (sum < 0) sum = 0;
        if (sum > 255) sum = 255;
        output[idx + c] = static_cast<unsigned char>(sum);
    }
}

bool convolution3x3_cuda(const unsigned char* input,
                        unsigned char* output,
                        int rows,
                        int cols,
                        const int* kernel,
                        int divisor,
                        int iterations,
                        float* kernel_ms,
                        float* total_ms,
                        string* err) {
    const size_t image_bytes = static_cast<size_t>(rows) * static_cast<size_t>(cols) * 3;
    const size_t kernel_bytes = 9 * sizeof(int);

    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;
    int* d_kernel = nullptr;

    cudaError_t st = cudaSuccess;

    st = cudaMalloc(&d_input, image_bytes);
    if (st != cudaSuccess) {
        if (err) *err = cudaGetErrorString(st);
        return false;
    }

    st = cudaMalloc(&d_output, image_bytes);
    if (st != cudaSuccess) {
        if (err) *err = cudaGetErrorString(st);
        cudaFree(d_input);
        return false;
    }

    st = cudaMalloc(&d_kernel, kernel_bytes);
    if (st != cudaSuccess) {
        if (err) *err = cudaGetErrorString(st);
        cudaFree(d_input);
        cudaFree(d_output);
        return false;
    }

    st = cudaMemcpy(d_kernel, kernel, kernel_bytes, cudaMemcpyHostToDevice);
    if (st != cudaSuccess) {
        if (err) *err = cudaGetErrorString(st);
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

    for (int it = 0; it < iterations; ++it) {
        st = cudaMemcpy(d_input, input, image_bytes, cudaMemcpyHostToDevice);
        if (st != cudaSuccess) {
            if (err) *err = cudaGetErrorString(st);
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
        if (st != cudaSuccess) {
            if (err) *err = cudaGetErrorString(st);
            cudaEventDestroy(ev_start);
            cudaEventDestroy(ev_stop);
            cudaFree(d_input);
            cudaFree(d_output);
            cudaFree(d_kernel);
            return false;
        }

        st = cudaGetLastError();
        if (st != cudaSuccess) {
            if (err) *err = cudaGetErrorString(st);
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
        if (st != cudaSuccess) {
            if (err) *err = cudaGetErrorString(st);
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

    if (kernel_ms) *kernel_ms = kernel_total;
    if (total_ms) *total_ms = static_cast<float>(total);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    return true;
}

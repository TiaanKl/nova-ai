#include <cuda_runtime.h>
#include <stdint.h>

__global__ void nova_mean_k_kernel(
    const float* __restrict__ columns,
    float* __restrict__ output,
    int batch, int channels, int k, int height, int width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * height * width;
    if (idx >= total) return;

    int w = idx % width;
    int tmp = idx / width;
    int h = tmp % height;
    tmp = tmp / height;
    int c = tmp % channels;
    int b = tmp / channels;

    float sum = 0.0f;
    for (int ki = 0; ki < k; ++ki) {
        size_t idx_col = (((size_t)(c * k + ki) * batch + b) * height * width) + (h * width + w);
        sum += columns[idx_col];
    }
    output[idx] = sum / k;
}

extern "C" void nova_mean_k_kernel_launcher(const float* columns, float* output, int batch, int channels, int k, int height, int width, cudaStream_t stream)
{
    int total = batch * channels * height * width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    nova_mean_k_kernel<<<blocks, threads, 0, stream>>>(columns, output, batch, channels, k, height, width);
}

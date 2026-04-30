#include <cuda_runtime.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include "nova_dcnv2_api.h"

#if defined(__has_include)
#if __has_include(<ATen/cuda/CUDAContext.h>)
#include <ATen/cuda/CUDAContext.h>
#define NOVA_HAVE_ATEN_CUDA 1
#endif
#endif

#if defined(__CUDACC__)
#include <cuda_fp16.h>
#if defined(__has_include)
#if __has_include(<cuda_bf16.h>)
#include <cuda_bf16.h>
#define NOVA_HAVE_BF16 1
#endif
#endif
#endif

#define CUDA_NUM_THREADS 256

#ifndef NOVA_DCNV2_SYNC_AFTER_KERNEL
#define NOVA_DCNV2_SYNC_AFTER_KERNEL 0
#endif

enum NovaDcnv2Status
{
  NOVA_DCNV2_SUCCESS = 0,
  NOVA_DCNV2_ERROR_NULL_POINTER = 1,
  NOVA_DCNV2_ERROR_INVALID_ARGUMENT = 2,
  NOVA_DCNV2_ERROR_INVALID_OUTPUT_SHAPE = 3,
  NOVA_DCNV2_ERROR_SIZE_OVERFLOW = 4,
  NOVA_DCNV2_ERROR_CUDA_LAUNCH = 5,
  NOVA_DCNV2_ERROR_CUDA_EXECUTION = 6,
};

inline int GET_BLOCKS(const int64_t N) {
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

#define CUDA_KERNEL_LOOP(i, n) \
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; \
         i < (n); \
     i += static_cast<int64_t>(blockDim.x) * gridDim.x)

inline bool checked_mul_size_t(size_t lhs, size_t rhs, size_t* result)
{
  if (lhs != 0 && rhs > SIZE_MAX / lhs)
    return false;
  *result = lhs * rhs;
  return true;
}

inline int compute_modulated_deform_conv2d_output_shape(
    const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    int* height_out, int* width_out)
{
  const int64_t kernel_extent_h = static_cast<int64_t>(dilation_h) * (kernel_h - 1) + 1;
  const int64_t kernel_extent_w = static_cast<int64_t>(dilation_w) * (kernel_w - 1) + 1;
  const int64_t numerator_h = static_cast<int64_t>(height) + 2LL * pad_h - kernel_extent_h;
  const int64_t numerator_w = static_cast<int64_t>(width) + 2LL * pad_w - kernel_extent_w;
  if (numerator_h < 0 || numerator_w < 0)
    return NOVA_DCNV2_ERROR_INVALID_OUTPUT_SHAPE;

  const int64_t computed_height_out = numerator_h / stride_h + 1;
  const int64_t computed_width_out = numerator_w / stride_w + 1;
  if (computed_height_out <= 0 || computed_width_out <= 0 ||
      computed_height_out > INT_MAX || computed_width_out > INT_MAX)
    return NOVA_DCNV2_ERROR_INVALID_OUTPUT_SHAPE;

  *height_out = static_cast<int>(computed_height_out);
  *width_out = static_cast<int>(computed_width_out);
  return NOVA_DCNV2_SUCCESS;
}

__device__ float modulated_deform_conv2d_im2col_bilinear(
  const float *bottom_data, const int data_width,
        const int height, const int width, float h, float w)
{
  int h_low = floorf(h);
  int w_low = floorf(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  float lh = h - static_cast<float>(h_low);
  float lw = w - static_cast<float>(w_low);
  float hh = 1.0f - lh, hw = 1.0f - lw;

  float v1 = 0.0f;
  if (h_low >= 0 && w_low >= 0)
    v1 = bottom_data[h_low * data_width + w_low];
  float v2 = 0.0f;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = bottom_data[h_low * data_width + w_high];
  float v3 = 0.0f;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = bottom_data[h_high * data_width + w_low];
  float v4 = 0.0f;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = bottom_data[h_high * data_width + w_high];

  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  return fmaf(w1, v1, fmaf(w2, v2, fmaf(w3, v3, w4 * v4)));
}

__device__ __forceinline__ float nova_element_to_float(float value)
{
  return value;
}

__device__ __forceinline__ float nova_element_to_float(__half value)
{
  return __half2float(value);
}

#if defined(NOVA_HAVE_BF16)
__device__ __forceinline__ float nova_element_to_float(__nv_bfloat16 value)
{
  return __bfloat162float(value);
}
#endif

template <typename ElementT>
__device__ __forceinline__ ElementT nova_float_to_element(float value);

template <>
__device__ __forceinline__ float nova_float_to_element<float>(float value)
{
  return value;
}

template <>
__device__ __forceinline__ __half nova_float_to_element<__half>(float value)
{
  return __float2half(value);
}

#if defined(NOVA_HAVE_BF16)
template <>
__device__ __forceinline__ __nv_bfloat16 nova_float_to_element<__nv_bfloat16>(float value)
{
  return __float2bfloat16(value);
}
#endif

template <typename ElementT>
__device__ float modulated_deform_conv2d_bilinear_typed(
  const ElementT* bottom_data, const int data_width,
  const int height, const int width, float h, float w)
{
  int h_low = floorf(h);
  int w_low = floorf(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  float lh = h - static_cast<float>(h_low);
  float lw = w - static_cast<float>(w_low);
  float hh = 1.0f - lh;
  float hw = 1.0f - lw;

  float v1 = 0.0f;
  if (h_low >= 0 && w_low >= 0)
    v1 = nova_element_to_float(bottom_data[h_low * data_width + w_low]);
  float v2 = 0.0f;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = nova_element_to_float(bottom_data[h_low * data_width + w_high]);
  float v3 = 0.0f;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = nova_element_to_float(bottom_data[h_high * data_width + w_low]);
  float v4 = 0.0f;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = nova_element_to_float(bottom_data[h_high * data_width + w_high]);

  float w1 = hh * hw;
  float w2 = hh * lw;
  float w3 = lh * hw;
  float w4 = lh * lw;

  return fmaf(w1, v1, fmaf(w2, v2, fmaf(w3, v3, w4 * v4)));
}

template <typename ElementT, int KernelH, int KernelW>
__device__ __forceinline__ float modulated_deform_conv2d_fused_mean_accumulate_fixed(
    const ElementT* __restrict__ data_im_ptr,
    const ElementT* __restrict__ data_offset_ptr,
    const ElementT* __restrict__ data_mask_ptr,
    const int height, const int width,
    const int h_in, const int w_in,
    const int dilation_h, const int dilation_w,
    const size_t spatial_size,
    const size_t spatial_index)
{
  float acc = 0.0f;

#pragma unroll
  for (int i = 0; i < KernelH; ++i)
  {
    const float h_base = static_cast<float>(h_in + i * dilation_h);

#pragma unroll
    for (int j = 0; j < KernelW; ++j)
    {
      const int kernel_index = i * KernelW + j;
      const size_t offset_base = static_cast<size_t>(2 * kernel_index) * spatial_size + spatial_index;
      const float offset_h = nova_element_to_float(data_offset_ptr[offset_base]);
      const float offset_w = nova_element_to_float(data_offset_ptr[offset_base + spatial_size]);
      const float mask = nova_element_to_float(data_mask_ptr[static_cast<size_t>(kernel_index) * spatial_size + spatial_index]);
      const float h_im = h_base + offset_h;
      const float w_im = static_cast<float>(w_in + j * dilation_w) + offset_w;

      if (h_im > -1.0f && w_im > -1.0f && h_im < static_cast<float>(height) && w_im < static_cast<float>(width))
      {
        const float val = modulated_deform_conv2d_bilinear_typed(data_im_ptr, width, height, width, h_im, w_im);
        acc = fmaf(val, mask, acc);
      }
    }
  }

  return acc;
}

template <typename ElementT>
__device__ __forceinline__ float modulated_deform_conv2d_fused_mean_accumulate_generic(
    const ElementT* __restrict__ data_im_ptr,
    const ElementT* __restrict__ data_offset_ptr,
    const ElementT* __restrict__ data_mask_ptr,
    const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int h_in, const int w_in,
    const int dilation_h, const int dilation_w,
    const size_t spatial_size,
    const size_t spatial_index)
{
  float acc = 0.0f;

  for (int i = 0; i < kernel_h; ++i)
  {
    const float h_base = static_cast<float>(h_in + i * dilation_h);

    for (int j = 0; j < kernel_w; ++j)
    {
      const int kernel_index = i * kernel_w + j;
      const size_t offset_base = static_cast<size_t>(2 * kernel_index) * spatial_size + spatial_index;
      const float offset_h = nova_element_to_float(data_offset_ptr[offset_base]);
      const float offset_w = nova_element_to_float(data_offset_ptr[offset_base + spatial_size]);
      const float mask = nova_element_to_float(data_mask_ptr[static_cast<size_t>(kernel_index) * spatial_size + spatial_index]);
      const float h_im = h_base + offset_h;
      const float w_im = static_cast<float>(w_in + j * dilation_w) + offset_w;

      if (h_im > -1.0f && w_im > -1.0f && h_im < static_cast<float>(height) && w_im < static_cast<float>(width))
      {
        const float val = modulated_deform_conv2d_bilinear_typed(data_im_ptr, width, height, width, h_im, w_im);
        acc = fmaf(val, mask, acc);
      }
    }
  }

  return acc;
}

template <typename ElementT, int FixedKernelH, int FixedKernelW>
__global__ void modulated_deform_conv2d_fused_mean_gpu_kernel(
  const int64_t n, const ElementT* __restrict__ data_im,
  const ElementT* __restrict__ data_offset,
  const ElementT* __restrict__ data_mask,
  const int height, const int width,
  const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w,
  const int stride_h, const int stride_w,
  const int dilation_h, const int dilation_w,
  const int channel_per_deformable_group,
  const int batch_size, const int num_channels, const int deformable_group,
  const int height_col, const int width_col,
  ElementT* __restrict__ data_out)
{
  const size_t spatial_size = static_cast<size_t>(height_col) * width_col;
  const size_t group_offset_stride = static_cast<size_t>(2) * kernel_h * kernel_w * spatial_size;
  const size_t group_mask_stride = static_cast<size_t>(kernel_h) * kernel_w * spatial_size;
  const size_t im_channel_stride = static_cast<size_t>(height) * width;
  const size_t im_batch_stride = static_cast<size_t>(num_channels) * im_channel_stride;

  CUDA_KERNEL_LOOP(index, n)
  {
    const int w_col = static_cast<int>(index % width_col);
    const int h_col = static_cast<int>((index / width_col) % height_col);
    const int b_col = static_cast<int>((index / width_col / height_col) % batch_size);
    const int c_im = static_cast<int>((index / width_col / height_col) / batch_size);
    const size_t spatial_index = static_cast<size_t>(h_col) * width_col + w_col;

    const int deformable_group_index = c_im / channel_per_deformable_group;
    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;

    const ElementT* data_im_ptr = data_im + static_cast<size_t>(b_col) * im_batch_stride
                                  + static_cast<size_t>(c_im) * im_channel_stride;
    const ElementT* data_offset_ptr = data_offset +
        (static_cast<size_t>(b_col) * deformable_group + deformable_group_index) * group_offset_stride;
    const ElementT* data_mask_ptr = data_mask +
        (static_cast<size_t>(b_col) * deformable_group + deformable_group_index) * group_mask_stride;

    float acc = 0.0f;
    if constexpr (FixedKernelH > 0 && FixedKernelW > 0)
    {
      acc = modulated_deform_conv2d_fused_mean_accumulate_fixed<ElementT, FixedKernelH, FixedKernelW>(
          data_im_ptr, data_offset_ptr, data_mask_ptr,
          height, width, h_in, w_in, dilation_h, dilation_w,
          spatial_size, spatial_index);
    }
    else
    {
      acc = modulated_deform_conv2d_fused_mean_accumulate_generic(
          data_im_ptr, data_offset_ptr, data_mask_ptr,
          height, width, kernel_h, kernel_w, h_in, w_in,
          dilation_h, dilation_w, spatial_size, spatial_index);
    }

    data_out[index] = nova_float_to_element<ElementT>(acc / static_cast<float>(kernel_h * kernel_w));
  }
}

template <int KernelH, int KernelW>
__device__ __forceinline__ void modulated_deform_conv2d_im2col_write_fixed(
    float* data_col_ptr,
    const float* __restrict__ data_im_ptr,
    const float* __restrict__ data_offset_ptr,
    const float* __restrict__ data_mask_ptr,
    const int height, const int width,
    const int h_in, const int w_in,
    const int dilation_h, const int dilation_w,
    const size_t spatial_size,
    const size_t col_step,
    const size_t spatial_index)
{
#pragma unroll
  for (int i = 0; i < KernelH; ++i)
  {
    const float h_base = static_cast<float>(h_in + i * dilation_h);

#pragma unroll
    for (int j = 0; j < KernelW; ++j)
    {
      const int kernel_index = i * KernelW + j;
      const size_t offset_base = static_cast<size_t>(2 * kernel_index) * spatial_size + spatial_index;
      const float offset_h = data_offset_ptr[offset_base];
      const float offset_w = data_offset_ptr[offset_base + spatial_size];
      const float mask = data_mask_ptr[static_cast<size_t>(kernel_index) * spatial_size + spatial_index];
      const float h_im = h_base + offset_h;
      const float w_im = static_cast<float>(w_in + j * dilation_w) + offset_w;

      float val = 0.0f;
      if (h_im > -1.0f && w_im > -1.0f && h_im < static_cast<float>(height) && w_im < static_cast<float>(width))
        val = modulated_deform_conv2d_im2col_bilinear(data_im_ptr, width, height, width, h_im, w_im);

      *data_col_ptr = val * mask;
      data_col_ptr += col_step;
    }
  }
}

__device__ __forceinline__ void modulated_deform_conv2d_im2col_write_generic(
    float* data_col_ptr,
    const float* __restrict__ data_im_ptr,
    const float* __restrict__ data_offset_ptr,
    const float* __restrict__ data_mask_ptr,
    const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int h_in, const int w_in,
    const int dilation_h, const int dilation_w,
    const size_t spatial_size,
    const size_t col_step,
    const size_t spatial_index)
{
  for (int i = 0; i < kernel_h; ++i)
  {
    const float h_base = static_cast<float>(h_in + i * dilation_h);

    for (int j = 0; j < kernel_w; ++j)
    {
      const int kernel_index = i * kernel_w + j;
      const size_t offset_base = static_cast<size_t>(2 * kernel_index) * spatial_size + spatial_index;
      const float offset_h = data_offset_ptr[offset_base];
      const float offset_w = data_offset_ptr[offset_base + spatial_size];
      const float mask = data_mask_ptr[static_cast<size_t>(kernel_index) * spatial_size + spatial_index];
      const float h_im = h_base + offset_h;
      const float w_im = static_cast<float>(w_in + j * dilation_w) + offset_w;

      float val = 0.0f;
      if (h_im > -1.0f && w_im > -1.0f && h_im < static_cast<float>(height) && w_im < static_cast<float>(width))
        val = modulated_deform_conv2d_im2col_bilinear(data_im_ptr, width, height, width, h_im, w_im);

      *data_col_ptr = val * mask;
      data_col_ptr += col_step;
    }
  }
}

template <int FixedKernelH, int FixedKernelW>
__global__ void modulated_deform_conv2d_im2col_gpu_kernel(
  const int64_t n, const float *__restrict__ data_im, const float *__restrict__ data_offset, const float *__restrict__ data_mask,
        const int height, const int width, const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w,
        const int channel_per_deformable_group,
        const int batch_size, const int num_channels, const int deformable_group,
        const int height_col, const int width_col,
        float *data_col)
{
  const size_t spatial_size = static_cast<size_t>(height_col) * width_col;
  const size_t col_step = static_cast<size_t>(batch_size) * spatial_size;
  const size_t group_offset_stride = static_cast<size_t>(2) * kernel_h * kernel_w * spatial_size;
  const size_t group_mask_stride = static_cast<size_t>(kernel_h) * kernel_w * spatial_size;

  const size_t col_channel_stride = static_cast<size_t>(batch_size) * height_col * width_col;
  const size_t col_batch_stride   = static_cast<size_t>(height_col) * width_col;
  const size_t col_row_stride     = static_cast<size_t>(width_col);

  const size_t im_channel_stride = static_cast<size_t>(height) * width;
  const size_t im_batch_stride   = static_cast<size_t>(num_channels) * im_channel_stride;

  CUDA_KERNEL_LOOP(index, n)
  {
    const int w_col = static_cast<int>(index % width_col);
    const int h_col = static_cast<int>((index / width_col) % height_col);
    const int b_col = static_cast<int>((index / width_col / height_col) % batch_size);
    const int c_im = static_cast<int>((index / width_col / height_col) / batch_size);
    const size_t c_col = static_cast<size_t>(c_im) * kernel_h * kernel_w;
    const size_t spatial_index = static_cast<size_t>(h_col) * width_col + w_col;

    const int deformable_group_index = c_im / channel_per_deformable_group;
    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;

    float *data_col_ptr = data_col + c_col * col_channel_stride
                          + static_cast<size_t>(b_col) * col_batch_stride
                          + static_cast<size_t>(h_col) * col_row_stride
                          + static_cast<size_t>(w_col);

    const float *data_im_ptr = data_im + static_cast<size_t>(b_col) * im_batch_stride
                               + static_cast<size_t>(c_im) * im_channel_stride;

    const float *data_offset_ptr = data_offset + (static_cast<size_t>(b_col) * deformable_group + deformable_group_index) * group_offset_stride;
    const float *data_mask_ptr = data_mask + (static_cast<size_t>(b_col) * deformable_group + deformable_group_index) * group_mask_stride;

    if constexpr (FixedKernelH > 0 && FixedKernelW > 0)
    {
      modulated_deform_conv2d_im2col_write_fixed<FixedKernelH, FixedKernelW>(
          data_col_ptr, data_im_ptr, data_offset_ptr, data_mask_ptr,
          height, width, h_in, w_in, dilation_h, dilation_w,
          spatial_size, col_step, spatial_index);
    }
    else
    {
      modulated_deform_conv2d_im2col_write_generic(
          data_col_ptr, data_im_ptr, data_offset_ptr, data_mask_ptr,
          height, width, kernel_h, kernel_w, h_in, w_in,
          dilation_h, dilation_w, spatial_size, col_step, spatial_index);
    }
  }
}

int modulated_deform_conv2d_im2col_cuda(
    const float* data_im, const float* data_offset, const float* data_mask,
    const int batch_size, const int channels, const int height_im, const int width_im,
    const int height_col, const int width_col, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int deformable_group, float* data_col,
    cudaStream_t stream)
{
  if (data_im == nullptr || data_offset == nullptr || data_mask == nullptr || data_col == nullptr)
    return NOVA_DCNV2_ERROR_NULL_POINTER;

  if (batch_size <= 0 || channels <= 0 || height_im <= 0 || width_im <= 0 ||
      height_col <= 0 || width_col <= 0 || kernel_h <= 0 || kernel_w <= 0 ||
      pad_h < 0 || pad_w < 0 || stride_h <= 0 || stride_w <= 0 ||
      dilation_h <= 0 || dilation_w <= 0 || deformable_group <= 0 ||
      channels % deformable_group != 0)
    return NOVA_DCNV2_ERROR_INVALID_ARGUMENT;

  size_t element_count = static_cast<size_t>(batch_size);
  if (!checked_mul_size_t(element_count, static_cast<size_t>(channels), &element_count) ||
      !checked_mul_size_t(element_count, static_cast<size_t>(height_im), &element_count) ||
      !checked_mul_size_t(element_count, static_cast<size_t>(width_im), &element_count))
    return NOVA_DCNV2_ERROR_SIZE_OVERFLOW;

  element_count = static_cast<size_t>(batch_size);
  if (!checked_mul_size_t(element_count, static_cast<size_t>(channels), &element_count) ||
      !checked_mul_size_t(element_count, static_cast<size_t>(kernel_h), &element_count) ||
      !checked_mul_size_t(element_count, static_cast<size_t>(kernel_w), &element_count) ||
      !checked_mul_size_t(element_count, static_cast<size_t>(height_col), &element_count) ||
      !checked_mul_size_t(element_count, static_cast<size_t>(width_col), &element_count))
    return NOVA_DCNV2_ERROR_SIZE_OVERFLOW;

  element_count = static_cast<size_t>(batch_size);
  if (!checked_mul_size_t(element_count, static_cast<size_t>(deformable_group), &element_count) ||
      !checked_mul_size_t(element_count, 2, &element_count) ||
      !checked_mul_size_t(element_count, static_cast<size_t>(kernel_h), &element_count) ||
      !checked_mul_size_t(element_count, static_cast<size_t>(kernel_w), &element_count) ||
      !checked_mul_size_t(element_count, static_cast<size_t>(height_col), &element_count) ||
      !checked_mul_size_t(element_count, static_cast<size_t>(width_col), &element_count))
    return NOVA_DCNV2_ERROR_SIZE_OVERFLOW;

  element_count = static_cast<size_t>(batch_size);
  if (!checked_mul_size_t(element_count, static_cast<size_t>(deformable_group), &element_count) ||
      !checked_mul_size_t(element_count, static_cast<size_t>(kernel_h), &element_count) ||
      !checked_mul_size_t(element_count, static_cast<size_t>(kernel_w), &element_count) ||
      !checked_mul_size_t(element_count, static_cast<size_t>(height_col), &element_count) ||
      !checked_mul_size_t(element_count, static_cast<size_t>(width_col), &element_count))
    return NOVA_DCNV2_ERROR_SIZE_OVERFLOW;

  const int channel_per_deformable_group = channels / deformable_group;
  const int64_t num_kernels = static_cast<int64_t>(channels) * batch_size * height_col * width_col;
  if (num_kernels <= 0 || num_kernels > static_cast<int64_t>(INT_MAX) * CUDA_NUM_THREADS)
    return NOVA_DCNV2_ERROR_SIZE_OVERFLOW;

  if (kernel_h == 3 && kernel_w == 3)
  {
    modulated_deform_conv2d_im2col_gpu_kernel<3, 3><<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
        num_kernels, data_im, data_offset, data_mask, height_im, width_im, kernel_h, kernel_w,
        pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, channel_per_deformable_group,
        batch_size, channels, deformable_group, height_col, width_col, data_col);
  }
  else if (kernel_h == 1 && kernel_w == 1)
  {
    modulated_deform_conv2d_im2col_gpu_kernel<1, 1><<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
        num_kernels, data_im, data_offset, data_mask, height_im, width_im, kernel_h, kernel_w,
        pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, channel_per_deformable_group,
        batch_size, channels, deformable_group, height_col, width_col, data_col);
  }
  else
  {
    modulated_deform_conv2d_im2col_gpu_kernel<-1, -1><<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
        num_kernels, data_im, data_offset, data_mask, height_im, width_im, kernel_h, kernel_w,
        pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, channel_per_deformable_group,
        batch_size, channels, deformable_group, height_col, width_col, data_col);
  }

  cudaError_t err = cudaPeekAtLastError();
  if (err != cudaSuccess)
    return NOVA_DCNV2_ERROR_CUDA_LAUNCH;

#if NOVA_DCNV2_SYNC_AFTER_KERNEL
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess)
    return NOVA_DCNV2_ERROR_CUDA_EXECUTION;
#endif

  return NOVA_DCNV2_SUCCESS;
}

template <typename ElementT>
int modulated_deform_conv2d_fused_mean_cuda(
    const ElementT* data_im, const ElementT* data_offset, const ElementT* data_mask,
    const int batch_size, const int channels, const int height_im, const int width_im,
    const int height_col, const int width_col, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int deformable_group, ElementT* data_out,
    cudaStream_t stream)
{
  if (data_im == nullptr || data_offset == nullptr || data_mask == nullptr || data_out == nullptr)
    return NOVA_DCNV2_ERROR_NULL_POINTER;

  if (batch_size <= 0 || channels <= 0 || height_im <= 0 || width_im <= 0 ||
      height_col <= 0 || width_col <= 0 || kernel_h <= 0 || kernel_w <= 0 ||
      pad_h < 0 || pad_w < 0 || stride_h <= 0 || stride_w <= 0 ||
      dilation_h <= 0 || dilation_w <= 0 || deformable_group <= 0 ||
      channels % deformable_group != 0)
    return NOVA_DCNV2_ERROR_INVALID_ARGUMENT;

  size_t element_count = static_cast<size_t>(batch_size);
  if (!checked_mul_size_t(element_count, static_cast<size_t>(channels), &element_count) ||
      !checked_mul_size_t(element_count, static_cast<size_t>(height_im), &element_count) ||
      !checked_mul_size_t(element_count, static_cast<size_t>(width_im), &element_count))
    return NOVA_DCNV2_ERROR_SIZE_OVERFLOW;

  element_count = static_cast<size_t>(batch_size);
  if (!checked_mul_size_t(element_count, static_cast<size_t>(deformable_group), &element_count) ||
      !checked_mul_size_t(element_count, 2, &element_count) ||
      !checked_mul_size_t(element_count, static_cast<size_t>(kernel_h), &element_count) ||
      !checked_mul_size_t(element_count, static_cast<size_t>(kernel_w), &element_count) ||
      !checked_mul_size_t(element_count, static_cast<size_t>(height_col), &element_count) ||
      !checked_mul_size_t(element_count, static_cast<size_t>(width_col), &element_count))
    return NOVA_DCNV2_ERROR_SIZE_OVERFLOW;

  element_count = static_cast<size_t>(batch_size);
  if (!checked_mul_size_t(element_count, static_cast<size_t>(deformable_group), &element_count) ||
      !checked_mul_size_t(element_count, static_cast<size_t>(kernel_h), &element_count) ||
      !checked_mul_size_t(element_count, static_cast<size_t>(kernel_w), &element_count) ||
      !checked_mul_size_t(element_count, static_cast<size_t>(height_col), &element_count) ||
      !checked_mul_size_t(element_count, static_cast<size_t>(width_col), &element_count))
    return NOVA_DCNV2_ERROR_SIZE_OVERFLOW;

  const int channel_per_deformable_group = channels / deformable_group;
  const int64_t num_kernels = static_cast<int64_t>(channels) * batch_size * height_col * width_col;
  if (num_kernels <= 0 || num_kernels > static_cast<int64_t>(INT_MAX) * CUDA_NUM_THREADS)
    return NOVA_DCNV2_ERROR_SIZE_OVERFLOW;

  if (kernel_h == 3 && kernel_w == 3)
  {
    modulated_deform_conv2d_fused_mean_gpu_kernel<ElementT, 3, 3><<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
        num_kernels, data_im, data_offset, data_mask, height_im, width_im, kernel_h, kernel_w,
        pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, channel_per_deformable_group,
        batch_size, channels, deformable_group, height_col, width_col, data_out);
  }
  else if (kernel_h == 1 && kernel_w == 1)
  {
    modulated_deform_conv2d_fused_mean_gpu_kernel<ElementT, 1, 1><<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
        num_kernels, data_im, data_offset, data_mask, height_im, width_im, kernel_h, kernel_w,
        pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, channel_per_deformable_group,
        batch_size, channels, deformable_group, height_col, width_col, data_out);
  }
  else
  {
    modulated_deform_conv2d_fused_mean_gpu_kernel<ElementT, -1, -1><<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
        num_kernels, data_im, data_offset, data_mask, height_im, width_im, kernel_h, kernel_w,
        pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, channel_per_deformable_group,
        batch_size, channels, deformable_group, height_col, width_col, data_out);
  }

  cudaError_t err = cudaPeekAtLastError();
  if (err != cudaSuccess)
    return NOVA_DCNV2_ERROR_CUDA_LAUNCH;

#if NOVA_DCNV2_SYNC_AFTER_KERNEL
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess)
    return NOVA_DCNV2_ERROR_CUDA_EXECUTION;
#endif

  return NOVA_DCNV2_SUCCESS;
}

template <typename ElementT>
int execute_modulated_deform_conv2d_fused_mean_stream(
    const ElementT* input,
    const ElementT* offset,
    const ElementT* mask,
    ElementT* output,
    int batch, int channels, int height, int width,
    int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int dilation_h, int dilation_w, int deformable_groups,
    void* stream)
{
  if (input == nullptr || offset == nullptr || mask == nullptr || output == nullptr)
    return NOVA_DCNV2_ERROR_NULL_POINTER;

  if (batch <= 0 || channels <= 0 || height <= 0 || width <= 0 ||
      kernel_h <= 0 || kernel_w <= 0 || pad_h < 0 || pad_w < 0 ||
      stride_h <= 0 || stride_w <= 0 || dilation_h <= 0 || dilation_w <= 0 ||
      deformable_groups <= 0 || channels % deformable_groups != 0)
    return NOVA_DCNV2_ERROR_INVALID_ARGUMENT;

  int height_out = 0;
  int width_out = 0;
  const int shape_status = compute_modulated_deform_conv2d_output_shape(
      height, width, kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w,
      dilation_h, dilation_w,
      &height_out, &width_out);
  if (shape_status != NOVA_DCNV2_SUCCESS)
    return shape_status;

  return modulated_deform_conv2d_fused_mean_cuda(
      input, offset, mask,
      batch, channels, height, width,
      height_out, width_out,
      kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w,
      dilation_h, dilation_w,
      deformable_groups, output,
      static_cast<cudaStream_t>(stream));
}

extern "C" int NOVA_CDECL Nova_ExecuteModulatedDCNv2_Im2Col_Stream(
    const float* input,
    const float* offset,
    const float* mask,
    float* columns,
    int batch, int channels, int height, int width,
    int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int dilation_h, int dilation_w, int deformable_groups,
    void* stream)
{
    if (input == nullptr || offset == nullptr || mask == nullptr || columns == nullptr)
      return NOVA_DCNV2_ERROR_NULL_POINTER;

    if (batch <= 0 || channels <= 0 || height <= 0 || width <= 0 ||
      kernel_h <= 0 || kernel_w <= 0 || pad_h < 0 || pad_w < 0 ||
      stride_h <= 0 || stride_w <= 0 || dilation_h <= 0 || dilation_w <= 0 ||
      deformable_groups <= 0 || channels % deformable_groups != 0)
      return NOVA_DCNV2_ERROR_INVALID_ARGUMENT;

    int height_out = 0;
    int width_out = 0;
    const int shape_status = compute_modulated_deform_conv2d_output_shape(
        height, width, kernel_h, kernel_w,
        pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w,
        &height_out, &width_out);
    if (shape_status != NOVA_DCNV2_SUCCESS)
      return shape_status;

    return modulated_deform_conv2d_im2col_cuda(
        input, offset, mask,
        batch, channels, height, width,
        height_out, width_out,
        kernel_h, kernel_w,
        pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w,
        deformable_groups, columns,
        static_cast<cudaStream_t>(stream));
}

extern "C" int NOVA_CDECL Nova_ExecuteModulatedDCNv2_FusedMean_Stream(
    const float* input,
    const float* offset,
    const float* mask,
    float* output,
    int batch, int channels, int height, int width,
    int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int dilation_h, int dilation_w, int deformable_groups,
    void* stream)
{
  return execute_modulated_deform_conv2d_fused_mean_stream(
      input, offset, mask, output,
      batch, channels, height, width,
      kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w,
      dilation_h, dilation_w, deformable_groups,
      stream);
}

extern "C" int NOVA_CDECL Nova_ExecuteModulatedDCNv2_FusedMean_Half_Stream(
    const void* input,
    const void* offset,
    const void* mask,
    void* output,
    int batch, int channels, int height, int width,
    int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int dilation_h, int dilation_w, int deformable_groups,
    void* stream)
{
  return execute_modulated_deform_conv2d_fused_mean_stream(
      static_cast<const __half*>(input),
      static_cast<const __half*>(offset),
      static_cast<const __half*>(mask),
      static_cast<__half*>(output),
      batch, channels, height, width,
      kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w,
      dilation_h, dilation_w, deformable_groups,
      stream);
}

extern "C" int NOVA_CDECL Nova_ExecuteModulatedDCNv2_FusedMean_BFloat16_Stream(
    const void* input,
    const void* offset,
    const void* mask,
    void* output,
    int batch, int channels, int height, int width,
    int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int dilation_h, int dilation_w, int deformable_groups,
    void* stream)
{
#if defined(NOVA_HAVE_BF16)
  return execute_modulated_deform_conv2d_fused_mean_stream(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(offset),
      static_cast<const __nv_bfloat16*>(mask),
      static_cast<__nv_bfloat16*>(output),
      batch, channels, height, width,
      kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w,
      dilation_h, dilation_w, deformable_groups,
      stream);
#else
  (void)input;
  (void)offset;
  (void)mask;
  (void)output;
  (void)batch;
  (void)channels;
  (void)height;
  (void)width;
  (void)kernel_h;
  (void)kernel_w;
  (void)pad_h;
  (void)pad_w;
  (void)stride_h;
  (void)stride_w;
  (void)dilation_h;
  (void)dilation_w;
  (void)deformable_groups;
  (void)stream;
  return NOVA_DCNV2_ERROR_INVALID_ARGUMENT;
#endif
}

extern "C" int Nova_ExecuteModulatedDCNv2_Im2Col(
    const float* input,
    const float* offset,
    const float* mask,
    float* columns,
    int batch, int channels, int height, int width,
    int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int dilation_h, int dilation_w, int deformable_groups)
{
  #if defined(NOVA_HAVE_ATEN_CUDA)
    auto stream = at::cuda::getCurrentCUDAStream();
    return Nova_ExecuteModulatedDCNv2_Im2Col_Stream(
      input, offset, mask, columns,
      batch, channels, height, width,
      kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w,
      dilation_h, dilation_w, deformable_groups,
      reinterpret_cast<void*>(stream.stream()));
  #else
    return Nova_ExecuteModulatedDCNv2_Im2Col_Stream(
      input, offset, mask, columns,
      batch, channels, height, width,
      kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w,
      dilation_h, dilation_w, deformable_groups,
      nullptr);
  #endif
}

// -----------------------------
// Cast kernels and extern "C" wrappers
// Compile only with NVCC
// -----------------------------
#if defined(__CUDACC__)

#include <cuda_fp16.h>
#if defined(__has_include)
# if __has_include(<cuda_bf16.h>)
#  include <cuda_bf16.h>
#  define NOVA_HAVE_BF16 1
# endif
#endif

// Half -> float
__global__ void nova_cast_half_to_float_kernel(const __half* __restrict__ src, float* __restrict__ dst, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

__global__ void nova_cast_float_to_half_kernel(const float* __restrict__ src, __half* __restrict__ dst, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

extern "C" int NOVA_CDECL Nova_CastHalfToFloat(const void* src_half, void* dst_float, size_t n, void* stream)
{
    if (!src_half || !dst_float) return NOVA_DCNV2_ERROR_NULL_POINTER;
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    nova_cast_half_to_float_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __half*>(src_half), static_cast<float*>(dst_float), n);
    return cudaPeekAtLastError() == cudaSuccess ? NOVA_DCNV2_SUCCESS : NOVA_DCNV2_ERROR_CUDA_LAUNCH;
}

extern "C" int NOVA_CDECL Nova_CastFloatToHalf(const void* src_float, void* dst_half, size_t n, void* stream)
{
    if (!src_float || !dst_half) return NOVA_DCNV2_ERROR_NULL_POINTER;
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    nova_cast_float_to_half_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float*>(src_float), static_cast<__half*>(dst_half), n);
    return cudaPeekAtLastError() == cudaSuccess ? NOVA_DCNV2_SUCCESS : NOVA_DCNV2_ERROR_CUDA_LAUNCH;
}

#if defined(NOVA_HAVE_BF16)
// BF16 -> float
__global__ void nova_cast_bf16_to_float_kernel(const __nv_bfloat16* __restrict__ src, float* __restrict__ dst, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __bfloat162float(src[i]);
}

__global__ void nova_cast_float_to_bf16_kernel(const float* __restrict__ src, __nv_bfloat16* __restrict__ dst, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16(src[i]);
}

extern "C" int NOVA_CDECL Nova_CastBFloat16ToFloat(const void* src_bf16, void* dst_float, size_t n, void* stream)
{
    if (!src_bf16 || !dst_float) return NOVA_DCNV2_ERROR_NULL_POINTER;
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    nova_cast_bf16_to_float_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(src_bf16), static_cast<float*>(dst_float), n);
    return cudaPeekAtLastError() == cudaSuccess ? NOVA_DCNV2_SUCCESS : NOVA_DCNV2_ERROR_CUDA_LAUNCH;
}

extern "C" int NOVA_CDECL Nova_CastFloatToBFloat16(const void* src_float, void* dst_bf16, size_t n, void* stream)
{
    if (!src_float || !dst_bf16) return NOVA_DCNV2_ERROR_NULL_POINTER;
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    nova_cast_float_to_bf16_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float*>(src_float), static_cast<__nv_bfloat16*>(dst_bf16), n);
    return cudaPeekAtLastError() == cudaSuccess ? NOVA_DCNV2_SUCCESS : NOVA_DCNV2_ERROR_CUDA_LAUNCH;
}
#else
// BF16 not available in this toolkit
extern "C" int NOVA_CDECL Nova_CastBFloat16ToFloat(const void*, void*, size_t, void*) { return NOVA_DCNV2_ERROR_INVALID_ARGUMENT; }
extern "C" int NOVA_CDECL Nova_CastFloatToBFloat16(const void*, void*, size_t, void*) { return NOVA_DCNV2_ERROR_INVALID_ARGUMENT; }
#endif // NOVA_HAVE_BF16

#else // not __CUDACC__

// Stubs for host-only builds so the TU links when compiled by MSVC.
// These should never be called at runtime.
extern "C" int NOVA_CDECL Nova_ExecuteModulatedDCNv2_FusedMean_Stream(const float*, const float*, const float*, float*, int, int, int, int, int, int, int, int, int, int, int, int, int, void*) { return NOVA_DCNV2_ERROR_INVALID_ARGUMENT; }
extern "C" int NOVA_CDECL Nova_ExecuteModulatedDCNv2_FusedMean_Half_Stream(const void*, const void*, const void*, void*, int, int, int, int, int, int, int, int, int, int, int, int, int, void*) { return NOVA_DCNV2_ERROR_INVALID_ARGUMENT; }
extern "C" int NOVA_CDECL Nova_ExecuteModulatedDCNv2_FusedMean_BFloat16_Stream(const void*, const void*, const void*, void*, int, int, int, int, int, int, int, int, int, int, int, int, int, void*) { return NOVA_DCNV2_ERROR_INVALID_ARGUMENT; }
extern "C" int NOVA_CDECL Nova_CastHalfToFloat(const void*, void*, size_t, void*) { return NOVA_DCNV2_ERROR_INVALID_ARGUMENT; }
extern "C" int NOVA_CDECL Nova_CastFloatToHalf(const void*, void*, size_t, void*) { return NOVA_DCNV2_ERROR_INVALID_ARGUMENT; }
extern "C" int NOVA_CDECL Nova_CastBFloat16ToFloat(const void*, void*, size_t, void*) { return NOVA_DCNV2_ERROR_INVALID_ARGUMENT; }
extern "C" int NOVA_CDECL Nova_CastFloatToBFloat16(const void*, void*, size_t, void*) { return NOVA_DCNV2_ERROR_INVALID_ARGUMENT; }

#endif // __CUDACC__

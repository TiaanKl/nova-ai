#pragma once
#include <stddef.h>

#if defined(_WIN32) || defined(_WIN64)
#  ifdef NOVA_KERNELS_EXPORTS
#    define NOVA_API extern "C" __declspec(dllexport)
#  else
#    define NOVA_API extern "C" __declspec(dllimport)
#  endif
#else
#  define NOVA_API extern "C"
#endif

#if defined(_MSC_VER)
#define NOVA_CDECL __cdecl
#else
#define NOVA_CDECL
#endif

NOVA_API int NOVA_CDECL Nova_ExecuteModulatedDCNv2_Im2Col_Stream(
    const float* input,
    const float* offset,
    const float* mask,
    float* columns,
    int batch, int channels, int height, int width,
    int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int dilation_h, int dilation_w, int deformable_groups,
    void* stream);

NOVA_API int NOVA_CDECL Nova_ExecuteModulatedDCNv2_FusedMean_Stream(
    const float* input,
    const float* offset,
    const float* mask,
    float* output,
    int batch, int channels, int height, int width,
    int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int dilation_h, int dilation_w, int deformable_groups,
    void* stream);

NOVA_API int NOVA_CDECL Nova_ExecuteModulatedDCNv2_FusedMean_Half_Stream(
    const void* input,
    const void* offset,
    const void* mask,
    void* output,
    int batch, int channels, int height, int width,
    int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int dilation_h, int dilation_w, int deformable_groups,
    void* stream);

NOVA_API int NOVA_CDECL Nova_ExecuteModulatedDCNv2_FusedMean_BFloat16_Stream(
    const void* input,
    const void* offset,
    const void* mask,
    void* output,
    int batch, int channels, int height, int width,
    int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int dilation_h, int dilation_w, int deformable_groups,
    void* stream);

NOVA_API int NOVA_CDECL Nova_CastHalfToFloat(const void* src_half, void* dst_float, size_t n, void* stream);
NOVA_API int NOVA_CDECL Nova_CastFloatToHalf(const void* src_float, void* dst_half, size_t n, void* stream);
NOVA_API int NOVA_CDECL Nova_CastBFloat16ToFloat(const void* src_bf16, void* dst_float, size_t n, void* stream);
NOVA_API int NOVA_CDECL Nova_CastFloatToBFloat16(const void* src_float, void* dst_bf16, size_t n, void* stream);

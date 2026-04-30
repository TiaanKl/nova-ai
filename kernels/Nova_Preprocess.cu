#include <cuda_runtime.h>
#include <stdint.h>

#if defined(_MSC_VER)
#define NOVA_API extern "C" __declspec(dllexport)
#define NOVA_CDECL __cdecl
#else
#define NOVA_API extern "C" __attribute__((visibility("default")))
#define NOVA_CDECL
#endif

namespace
{
    constexpr int kThreads = 256;
    constexpr int kNormalizeZeroToOne = 0;
    constexpr int kNormalizeNegativeOneToOne = 1;

    inline int blocks(int64_t n)
    {
        return static_cast<int>((n + kThreads - 1) / kThreads);
    }

    inline bool is_valid_normalization_mode(int mode)
    {
        return mode == kNormalizeZeroToOne || mode == kNormalizeNegativeOneToOne;
    }

    __device__ __forceinline__ float normalize_channel(uint8_t value, int normalizationMode)
    {
        const float unit = static_cast<float>(value) * (1.0f / 255.0f);
        if (normalizationMode == kNormalizeNegativeOneToOne)
        {
            return fmaf(unit, 2.0f, -1.0f);
        }

        return unit;
    }

    __device__ __forceinline__ float denormalize_channel(float value, int normalizationMode)
    {
        if (normalizationMode == kNormalizeNegativeOneToOne)
        {
            value = fmaf(value, 0.5f, 0.5f);
        }

        return fminf(fmaxf(value, 0.0f), 1.0f);
    }
}

NOVA_API int NOVA_CDECL Nova_HostAllocPinned(void** out_ptr, size_t bytes)
{
    if (out_ptr == nullptr) return 1;
    *out_ptr = nullptr;
    return cudaHostAlloc(out_ptr, bytes, cudaHostAllocPortable) == cudaSuccess ? 0 : 2;
}

NOVA_API int NOVA_CDECL Nova_HostFree(void* ptr)
{
    if (ptr == nullptr) return 0;
    return cudaFreeHost(ptr) == cudaSuccess ? 0 : 2;
}

NOVA_API int NOVA_CDECL Nova_DeviceAlloc(void** out_ptr, size_t bytes)
{
    if (out_ptr == nullptr) return 1;
    *out_ptr = nullptr;
    return cudaMalloc(out_ptr, bytes) == cudaSuccess ? 0 : 2;
}

NOVA_API int NOVA_CDECL Nova_DeviceFree(void* ptr)
{
    if (ptr == nullptr) return 0;
    return cudaFree(ptr) == cudaSuccess ? 0 : 2;
}

NOVA_API int NOVA_CDECL Nova_StreamCreate(void** out_stream)
{
    if (out_stream == nullptr) return 1;
    cudaStream_t s = nullptr;
    cudaError_t err = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    *out_stream = static_cast<void*>(s);
    return err == cudaSuccess ? 0 : 2;
}

NOVA_API int NOVA_CDECL Nova_StreamDestroy(void* stream)
{
    if (stream == nullptr) return 0;
    return cudaStreamDestroy(static_cast<cudaStream_t>(stream)) == cudaSuccess ? 0 : 2;
}

NOVA_API int NOVA_CDECL Nova_StreamSynchronize(void* stream)
{
    return cudaStreamSynchronize(static_cast<cudaStream_t>(stream)) == cudaSuccess ? 0 : 2;
}

NOVA_API int NOVA_CDECL Nova_DeviceSynchronize()
{
    return cudaDeviceSynchronize() == cudaSuccess ? 0 : 2;
}

NOVA_API int NOVA_CDECL Nova_MemcpyAsync(void* dst, const void* src, size_t bytes, int kind, void* stream)
{
    cudaMemcpyKind k;
    switch (kind)
    {
    case 0: k = cudaMemcpyHostToDevice; break;
    case 1: k = cudaMemcpyDeviceToHost; break;
    case 2: k = cudaMemcpyDeviceToDevice; break;
    default: return 1;
    }
    return cudaMemcpyAsync(dst, src, bytes, k, static_cast<cudaStream_t>(stream)) == cudaSuccess ? 0 : 2;
}

__global__ void nova_bgr8_to_chw_fp32_kernel(
    const uint8_t* __restrict__ src,
    float* __restrict__ dst,
    int height, int width,
    int normalizationMode)
{
    const int hw = height * width;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hw) return;

    const int base = idx * 3;
    const float b = normalize_channel(src[base], normalizationMode);
    const float g = normalize_channel(src[base + 1], normalizationMode);
    const float r = normalize_channel(src[base + 2], normalizationMode);

    dst[idx]            = r;
    dst[hw + idx]       = g;
    dst[2 * hw + idx]   = b;
}

NOVA_API int NOVA_CDECL Nova_PreprocessBgr8ToChwFp32Ex(
    const void* src_host_or_device,
    void* dst_device_fp32,
    int height, int width,
    int frame_index, int,
    int normalization_mode,
    void* stream)
{
    if (src_host_or_device == nullptr || dst_device_fp32 == nullptr || height <= 0 || width <= 0)
        return 1;
    if (!is_valid_normalization_mode(normalization_mode))
        return 1;

    const int hw = height * width;
    float* dst = static_cast<float*>(dst_device_fp32) + static_cast<size_t>(frame_index) * 3 * hw;

    nova_bgr8_to_chw_fp32_kernel<<<blocks(hw), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint8_t*>(src_host_or_device), dst, height, width, normalization_mode);

    return cudaPeekAtLastError() == cudaSuccess ? 0 : 3;
}

NOVA_API int NOVA_CDECL Nova_PreprocessBgr8ToChwFp32(
    const void* src_host_or_device,
    void* dst_device_fp32,
    int height, int width,
    int frame_index, int frame_count,
    void* stream)
{
    return Nova_PreprocessBgr8ToChwFp32Ex(
        src_host_or_device,
        dst_device_fp32,
        height,
        width,
        frame_index,
        frame_count,
        kNormalizeZeroToOne,
        stream);
}

__global__ void nova_chw_fp32_to_bgr8_kernel(
    const float* __restrict__ src,
    uint8_t* __restrict__ dst,
    int height, int width,
    int normalizationMode)
{
    const int hw = height * width;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hw) return;

    float r = denormalize_channel(src[idx], normalizationMode) * 255.0f;
    float g = denormalize_channel(src[hw + idx], normalizationMode) * 255.0f;
    float b = denormalize_channel(src[2 * hw + idx], normalizationMode) * 255.0f;

    const int base = idx * 3;
    dst[base]     = static_cast<uint8_t>(__float2int_rn(b));
    dst[base + 1] = static_cast<uint8_t>(__float2int_rn(g));
    dst[base + 2] = static_cast<uint8_t>(__float2int_rn(r));
}

NOVA_API int NOVA_CDECL Nova_PostprocessChwFp32ToBgr8Ex(
    const void* src_device_fp32,
    void* dst_host_or_device,
    int height, int width,
    int frame_index, int,
    int normalization_mode,
    void* stream)
{
    if (src_device_fp32 == nullptr || dst_host_or_device == nullptr || height <= 0 || width <= 0)
        return 1;
    if (!is_valid_normalization_mode(normalization_mode))
        return 1;

    const int hw = height * width;
    const float* src = static_cast<const float*>(src_device_fp32) + static_cast<size_t>(frame_index) * 3 * hw;

    nova_chw_fp32_to_bgr8_kernel<<<blocks(hw), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        src, static_cast<uint8_t*>(dst_host_or_device), height, width, normalization_mode);

    return cudaPeekAtLastError() == cudaSuccess ? 0 : 3;
}

NOVA_API int NOVA_CDECL Nova_PostprocessChwFp32ToBgr8(
    const void* src_device_fp32,
    void* dst_host_or_device,
    int height, int width,
    int frame_index, int frame_count,
    void* stream)
{
    return Nova_PostprocessChwFp32ToBgr8Ex(
        src_device_fp32,
        dst_host_or_device,
        height,
        width,
        frame_index,
        frame_count,
        kNormalizeZeroToOne,
        stream);
}

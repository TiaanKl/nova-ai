#include "nova_dcnv2_plugin.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <iostream>
#include "nova_dcnv2_api.h"

namespace {
    const char* PLUGIN_NAME = "NovaDcnv2";
    const char* PLUGIN_VERSION = "1";
}

NovaDcnv2Plugin::NovaDcnv2Plugin(const std::string& name, int kernel_h, int kernel_w, int deformable_groups)
    : mName(name), mKernelH(kernel_h), mKernelW(kernel_w), mDeformableGroups(deformable_groups) {}

NovaDcnv2Plugin::NovaDcnv2Plugin(const void* data, size_t length) {
    const char* d = reinterpret_cast<const char*>(data);
    const char* a = d;
    int name_len = 0;
    std::memcpy(&name_len, d, sizeof(int)); d += sizeof(int);
    mName.assign(d, name_len); d += name_len;
    std::memcpy(&mKernelH, d, sizeof(int)); d += sizeof(int);
    std::memcpy(&mKernelW, d, sizeof(int)); d += sizeof(int);
    std::memcpy(&mDeformableGroups, d, sizeof(int)); d += sizeof(int);
    assert(static_cast<size_t>(d - a) == length);
}

NovaDcnv2Plugin::~NovaDcnv2Plugin() {}

const char* NovaDcnv2Plugin::getPluginType() const noexcept { return PLUGIN_NAME; }
const char* NovaDcnv2Plugin::getPluginVersion() const noexcept { return PLUGIN_VERSION; }
int NovaDcnv2Plugin::getNbOutputs() const noexcept { return 1; }

DimsExprs NovaDcnv2Plugin::getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept {
    DimsExprs out;
    out.nbDims = inputs[0].nbDims;
    for (int i = 0; i < out.nbDims; ++i) out.d[i] = inputs[0].d[i];
    return out;
}

bool NovaDcnv2Plugin::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {
    const PluginTensorDesc& desc = inOut[pos];
    if (desc.format != TensorFormat::kLINEAR) return false;
    if (pos > 0 && desc.type != inOut[0].type) return false;
    return desc.type == DataType::kFLOAT || desc.type == DataType::kHALF || desc.type == DataType::kBF16;
}

void NovaDcnv2Plugin::configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) noexcept {}

size_t NovaDcnv2Plugin::getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept {
    (void)inputs;
    (void)nbInputs;
    (void)outputs;
    (void)nbOutputs;
    return 0;
}

int NovaDcnv2Plugin::enqueue(
    const PluginTensorDesc* inputDesc,
    const PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept
{
    (void)workspace;
    const auto dtype = inputDesc[0].type;

    auto volume = [](const Dims& d) -> size_t {
        size_t v = 1;
        for (int i = 0; i < d.nbDims; ++i) v *= static_cast<size_t>(d.d[i]);
        return v;
    };

    const int batch    = inputDesc[0].dims.d[0];
    const int channels = inputDesc[0].dims.d[1];
    const int height   = inputDesc[0].dims.d[2];
    const int width    = inputDesc[0].dims.d[3];
    const size_t nOut    = volume(outputDesc[0].dims);

    // Cache env var lookup once — TRT calls enqueue() at every layer, every window.
    static const bool kPassthrough = (std::getenv("NOVA_DCNV2_PLUGIN_PASSTHROUGH") != nullptr);
    if (kPassthrough)
    {
        size_t bytesPerElem =
            (dtype == DataType::kFLOAT) ? sizeof(float) :
            (dtype == DataType::kHALF)  ? sizeof(uint16_t) :
            (dtype == DataType::kBF16)  ? sizeof(uint16_t) : 0;

        cudaMemcpyAsync(outputs[0], inputs[0],
                        nOut * bytesPerElem,
                        cudaMemcpyDeviceToDevice, stream);
        return 0;
    }

    int pad_h = (mKernelH - 1) / 2;
    int pad_w = (mKernelW - 1) / 2;
    int stride_h = 1, stride_w = 1;
    int dilation_h = 1, dilation_w = 1;
    void* streamVoid = static_cast<void*>(stream);
    int status = 0;

    if (dtype == DataType::kFLOAT)
    {
        const float* input  = static_cast<const float*>(inputs[0]);
        const float* offset = static_cast<const float*>(inputs[1]);
        const float* mask   = static_cast<const float*>(inputs[2]);
        float* output       = static_cast<float*>(outputs[0]);

        status = Nova_ExecuteModulatedDCNv2_FusedMean_Stream(
            input, offset, mask, output,
            batch, channels, height, width,
            mKernelH, mKernelW,
            pad_h, pad_w, stride_h, stride_w,
            dilation_h, dilation_w, mDeformableGroups,
            streamVoid
        );
    }
    else if (dtype == DataType::kHALF || dtype == DataType::kBF16)
    {
        if (dtype == DataType::kHALF)
        {
            status = Nova_ExecuteModulatedDCNv2_FusedMean_Half_Stream(
                inputs[0], inputs[1], inputs[2], outputs[0],
                batch, channels, height, width,
                mKernelH, mKernelW,
                pad_h, pad_w, stride_h, stride_w,
                dilation_h, dilation_w, mDeformableGroups,
                streamVoid
            );
        }
        else
        {
            status = Nova_ExecuteModulatedDCNv2_FusedMean_BFloat16_Stream(
                inputs[0], inputs[1], inputs[2], outputs[0],
                batch, channels, height, width,
                mKernelH, mKernelW,
                pad_h, pad_w, stride_h, stride_w,
                dilation_h, dilation_w, mDeformableGroups,
                streamVoid
            );
        }
    }
    else
    {
        return -1;
    }

    return status == 0 ? 0 : -1;
}

int NovaDcnv2Plugin::initialize() noexcept { return 0; }
void NovaDcnv2Plugin::terminate() noexcept {}
void NovaDcnv2Plugin::destroy() noexcept { delete this; }
IPluginV2DynamicExt* NovaDcnv2Plugin::clone() const noexcept {
    auto* plugin = new NovaDcnv2Plugin(mName, mKernelH, mKernelW, mDeformableGroups);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void NovaDcnv2Plugin::serialize(void* buffer) const noexcept {
    char* d = reinterpret_cast<char*>(buffer);
    int name_len = static_cast<int>(mName.size());
    std::memcpy(d, &name_len, sizeof(int)); d += sizeof(int);
    std::memcpy(d, mName.data(), name_len); d += name_len;
    std::memcpy(d, &mKernelH, sizeof(int)); d += sizeof(int);
    std::memcpy(d, &mKernelW, sizeof(int)); d += sizeof(int);
    std::memcpy(d, &mDeformableGroups, sizeof(int)); d += sizeof(int);
}

size_t NovaDcnv2Plugin::getSerializationSize() const noexcept {
    return sizeof(int) + mName.size() + sizeof(int) * 3;
}

void NovaDcnv2Plugin::setPluginNamespace(const char* libNamespace) noexcept { mNamespace = libNamespace; }
const char* NovaDcnv2Plugin::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

DataType NovaDcnv2Plugin::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept {
    return inputTypes[0];
}

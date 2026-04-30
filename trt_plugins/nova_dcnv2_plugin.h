#pragma once
#include "NvInfer.h"
#include <string>

using namespace nvinfer1;

class NovaDcnv2Plugin : public IPluginV2DynamicExt
{
public:
    NovaDcnv2Plugin(const std::string& name, int kernel_h, int kernel_w, int deformable_groups);
    NovaDcnv2Plugin(const void* data, size_t length);
    ~NovaDcnv2Plugin() override;

    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
                const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    void destroy() noexcept override;
    IPluginV2DynamicExt* clone() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    size_t getSerializationSize() const noexcept override;
    void setPluginNamespace(const char* libNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;
    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override;

private:
    std::string mName;
    std::string mNamespace;
    int mKernelH;
    int mKernelW;
    int mDeformableGroups;
};

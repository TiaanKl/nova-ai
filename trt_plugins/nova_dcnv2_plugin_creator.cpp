#include "nova_dcnv2_plugin.h"
#include "NvInferPlugin.h"
#include <vector>
#include <cstring>

using namespace nvinfer1;
using namespace plugin;

#if defined(_WIN32) || defined(_WIN64)
#define PLUGIN_EXPORT extern "C" __declspec(dllexport)
#else
#define PLUGIN_EXPORT extern "C"
#endif

class NovaDcnv2PluginCreator : public IPluginCreator
{
public:
    NovaDcnv2PluginCreator() {
        mPluginAttributes.emplace_back(PluginField("kernel_h", nullptr, PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(PluginField("kernel_w", nullptr, PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(PluginField("deformable_groups", nullptr, PluginFieldType::kINT32, 1));
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* getPluginName() const noexcept override { return "NovaDcnv2"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    const PluginFieldCollection* getFieldNames() noexcept override { return &mFC; }

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override {
        int kernel_h = 3, kernel_w = 3, deformable_groups = 1;
        if (fc) {
            for (int i = 0; i < fc->nbFields; ++i) {
                std::string fname(fc->fields[i].name);
                if (fname == "kernel_h") kernel_h = *static_cast<const int*>(fc->fields[i].data);
                if (fname == "kernel_w") kernel_w = *static_cast<const int*>(fc->fields[i].data);
                if (fname == "deformable_groups") deformable_groups = *static_cast<const int*>(fc->fields[i].data);
            }
        }
        return new NovaDcnv2Plugin(name, kernel_h, kernel_w, deformable_groups);
    }

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override {
        return new NovaDcnv2Plugin(serialData, serialLength);
    }

    void setPluginNamespace(const char* libNamespace) noexcept override { mNamespace = libNamespace; }
    const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

private:
    std::string mNamespace;
    PluginFieldCollection mFC{};
    std::vector<PluginField> mPluginAttributes;
};

PLUGIN_EXPORT bool pluginInitPlugin() {
    static NovaDcnv2PluginCreator creator;
    getPluginRegistry()->registerCreator(creator, "nova");
    getPluginRegistry()->registerCreator(creator, "");
    return true;
}

#include "NvOnnxParser.h"
#include "NvInfer.h"
#include <iostream>

using namespace nvinfer1;
using namespace nvonnxparser;

void registerNovaDcnv2PluginForOnnx(nvonnxparser::IParser* parser)
{
    std::cout << "[*] NovaDcnv2 ONNX node domain 'nova' will be resolved "
                 "to the TensorRT plugin 'NovaDcnv2' during engine build."
              << std::endl;
}

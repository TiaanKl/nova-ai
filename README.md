# RealBasicPlusPlus

# TensorRT Model Optimizations

## 1.) Increase Accuracy:
Validate layer outputs:

Use Polygraphy to dump layer outputs and verify no NaNs or Infs. The --validate option can check for NaNs and Infs. Also, we can compare layer outputs with golden values from, such as ONNX runtime.

For FP16 and BF16, a model might require retraining to ensure that intermediate layer output can be represented in FP16/BF16 precision without overflow or underflow.

For INT8, consider recalibrating with a more representative calibration data set. If your model comes from PyTorch, we also provide the TensorRT Model Optimizer for QAT in the framework besides PTQ in TensorRT. You can try both approaches and choose the one with more accuracy.

Manipulate layer precision:

Sometimes, running a layer with a certain precision results in incorrect output. This can be due to inherent layer constraints (such as LayerNorm output should not be INT8) or model constraints (output gets diverged, resulting in poor accuracy).

You can control layer execution precision and output precision.

An experimental debug precision tool can help automatically find layers to run with high precision.

Use the Editable Timing Cache to select a proper tactic.

When accuracy changes between two built engines for the same model, it might be due to a bad tactic being selected for a layer.

Use Editable Timing Cache to dump available tactics. Update the cache with a proper one.

## 2.) Overhead Layer Optimization
https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/overhead-layer-optimization.html

## 3.) Optimize TensorRT Performance
https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html

## 4.) Best Practices
https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html#best-practices

## 5.) Python API Documentation
https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/python-api-docs.html#python-api-docs

## 6.) C++ API Documentation
https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/c-api-docs.html#c-api-docs

## 7.) FAQ
https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/troubleshooting.html#troubleshooting

## 8.) Model-Optimizer
https://github.com/NVIDIA/Model-Optimizer
https://nvidia.github.io/Model-Optimizer/getting_started/1_overview.html

## 9.) TensorRT GitHub
https://github.com/NVIDIA/TensorRT/tree/main

## 10.) Code Examples/Tutorials
https://github.com/nikil-ravi/trt_tutorial
https://medium.com/@kish.imss/tensorrt-optimizing-model-inference-for-maximum-performance-8be266f78ec0
https://github.com/butaixianran/VIdeo_Upscaler_by_GFPGAN
https://github.com/willermo/video-enhancer
https://github.com/cliffordkleinsr/DE-SRFREN
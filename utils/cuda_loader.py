import os
import sys
import torch
from torch.utils.cpp_extension import load

def load_nova_kernels():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.environ['TORCH_EXTENSIONS_DIR'] = os.path.join(project_root, 'build')

    kernel_dir = os.path.join(project_root, 'kernels')

    bridge_source = os.path.join(kernel_dir, "nova_bridge.cpp")

    with open(bridge_source, "w") as f:
        f.write("""
#include <torch/extension.h>
#include <cuda_runtime.h>

extern "C" int Nova_ExecuteModulatedDCNv2_Im2Col(
    const float* input,
    const float* offset,
    const float* mask,
    float* columns,
    int batch, int channels, int height, int width,
    int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int dilation_h, int dilation_w, int deformable_groups);

int dcn_forward(at::Tensor input, at::Tensor offset, at::Tensor mask,
                int batch, int channels, int h_im, int w_im,
                int kh, int kw, int ph, int pw, int sh, int sw,
                int dh, int dw, int d_grp, at::Tensor output) {

    return Nova_ExecuteModulatedDCNv2_Im2Col(
        input.data_ptr<float>(),
        offset.data_ptr<float>(),
        mask.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, channels, h_im, w_im,
        kh, kw, ph, pw, sh, sw, dh, dw, d_grp);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dcn_forward", &dcn_forward, "Nova DCNv2 Forward");
}
""")

    sources = [
        bridge_source,
        os.path.join(kernel_dir, 'Nova_DCNv2.cu'),
        os.path.join(kernel_dir, 'Nova_Preprocess.cu')
    ]

    if sys.platform == 'win32':
        torch_dll_path = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "torch", "lib")
        if os.path.exists(torch_dll_path):
            os.add_dll_directory(torch_dll_path)

        cuda_bin_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin"
        if os.path.exists(cuda_bin_path):
            os.add_dll_directory(cuda_bin_path)

    return load(
        name="nova_native",
        sources=sources,
        extra_cflags=['/Zc:preprocessor'],
        extra_cuda_cflags=[
            '-O3',
            '--use_fast_math',
            '-allow-unsupported-compiler',
            '-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING',
            '-Xcompiler=/Zc:preprocessor',
        ],
        verbose=True
    )
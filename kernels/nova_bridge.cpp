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

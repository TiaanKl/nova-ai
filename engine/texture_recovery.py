import torch
import torch.nn.functional as F

def _box_blur(image, kernel_size):
    padding = kernel_size // 2
    image = F.pad(image, (padding, padding, padding, padding), mode="reflect")
    return F.avg_pool2d(image, kernel_size=kernel_size, stride=1)


def _to_luma(image):
    weights = image.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


def recover_biological_texture(upscaled, original_lr, alpha=0.24, detail_clip=0.03, lr_resized=None):
    """
    Re-injects high-frequency detail from the LR frame into flatter regions
    while suppressing dark halos around strong edges.

    Args:
        upscaled: SR output at target resolution (B, C, H, W)
        original_lr: LR input at native resolution (B, C, h, w)
        alpha: texture recovery strength
        detail_clip: maximum detail amplitude to inject
        lr_resized: pre-computed bicubic-upsampled LR reference (same shape as upscaled).
                    When provided, skips the costly F.interpolate on 4x output.
    """
    _, _, h, w = upscaled.shape
    if lr_resized is not None:
        lr_ref = lr_resized
    else:
        lr_ref = F.interpolate(original_lr, size=(h, w), mode="bicubic", align_corners=False).clamp(0.0, 1.0)

    reference_luma = _to_luma(lr_ref)
    upscaled_luma = _to_luma(upscaled)

    base_kernel = 9
    detail_kernel = 7
    reference_base = _box_blur(reference_luma, kernel_size=base_kernel)
    reference_detail = (reference_luma - reference_base).clamp(-detail_clip, detail_clip)

    edge_response = (reference_luma - _box_blur(reference_luma, kernel_size=3)).abs()
    flat_region_mask = (1.0 - edge_response * 28.0).clamp(0.0, 1.0)

    detail_energy = _box_blur(reference_detail.abs(), kernel_size=detail_kernel)
    texture_presence = ((detail_energy - 0.0015) * 80.0).clamp(0.0, 1.0)
    detail_mask = flat_region_mask * texture_presence

    luma_detail = alpha * reference_detail * detail_mask
    recovered = upscaled + luma_detail

    recovered_luma = upscaled_luma + luma_detail
    edge_mask = (edge_response * 34.0).clamp(0.0, 1.0)
    dark_halo = (reference_luma - recovered_luma - 0.01).clamp(min=0.0)
    recovered = recovered + 0.75 * dark_halo * edge_mask

    return recovered.clamp(0.0, 1.0)

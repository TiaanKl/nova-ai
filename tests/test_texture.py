import torch
import cv2
import numpy as np
from engine.texture_recovery import recover_biological_texture
import torchvision.transforms.v2.functional as F_vision

def test_texture_fix(sample_image_path):
    img = cv2.imread(sample_image_path)
    lr_frame = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    lr_frame = lr_frame.to('cuda')

    smooth_ai = torch.nn.functional.interpolate(lr_frame, scale_factor=4, mode='bilinear')
    smooth_ai = F_vision.gaussian_blur(smooth_ai, kernel_size=[5, 5], sigma=[1.5, 1.5])

    refined = recover_biological_texture(smooth_ai, lr_frame, alpha=0.15)

    comparison = (refined.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite("texture_test_result.png", comparison)
    print("[+] Texture test complete. Check 'texture_test_result.png'.")

if __name__ == "__main__":
    test_texture_fix("test_data/skin_sample.png")
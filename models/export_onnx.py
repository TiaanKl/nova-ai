import argparse
import torch
import torch.nn as nn
from arch.vsr_backbone import NovaVSRBackbone
from utils.checkpoint import load_model_checkpoint

class NovaONNXWrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

def export_nova_onnx(
    weights_path="./weights/realbasicvsr.pth",
    onnx_path="./weights/nova_480.onnx",
    T=7,
    H=480,
    W=854,
    dynamic_shapes=False
):
    device = torch.device("cpu")
    model = NovaVSRBackbone().to(device)

    load_model_checkpoint(model, weights_path, device)
    model.eval()

    wrapper = NovaONNXWrapper(model).to(device).eval()

    dummy = torch.randn(1, T, 3, H, W, device=device)

    print("[*] Exporting checkpoint-compatible NovaVSRBackbone to ONNX...")
    with torch.no_grad():
        export_kwargs = {
            "opset_version": 18,
            "dynamo": True,
            "input_names": ["input"],
            "output_names": ["output"],
        }

        if dynamic_shapes:
            export_kwargs["dynamic_shapes"] = {
                "input": {1: torch.export.Dim("T"), 3: torch.export.Dim("H"), 4: torch.export.Dim("W")},
                "output": {1: torch.export.Dim("T"), 3: torch.export.Dim("H4"), 4: torch.export.Dim("W4")},
            }

        torch.onnx.export(wrapper, (dummy,), onnx_path, **export_kwargs)

    print(f"[+] Export complete: {onnx_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export NovaVSRBackbone to ONNX")
    parser.add_argument("--dynamic-shapes", action="store_true",
                        help="Export with dynamic spatial and temporal dimensions (default: fixed shapes)")
    parser.add_argument("--onnx-path", type=str, default="./weights/nova_480.onnx",
                        help="Output ONNX file path")
    parser.add_argument("--weights-path", type=str, default="./weights/realbasicvsr.pth",
                        help="Path to pretrained weights")
    parser.add_argument("--T", type=int, default=7, help="Number of frames (sequence length)")
    parser.add_argument("--H", type=int, default=480, help="Frame height")
    parser.add_argument("--W", type=int, default=854, help="Frame width")
    args = parser.parse_args()

    export_nova_onnx(
        weights_path=args.weights_path,
        onnx_path=args.onnx_path,
        T=args.T,
        H=args.H,
        W=args.W,
        dynamic_shapes=args.dynamic_shapes,
    )
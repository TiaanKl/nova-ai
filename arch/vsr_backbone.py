from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_layer(block: type[nn.Module], num_blocks: int, **kwargs) -> nn.Sequential:
    return nn.Sequential(*(block(**kwargs) for _ in range(num_blocks)))


def flow_warp(
    x: torch.Tensor,
    flow: torch.Tensor,
    interp_mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> torch.Tensor:
    n, _, h, w = x.shape
    device = x.device
    dtype = x.dtype

    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h, device=device, dtype=dtype),
        torch.arange(0, w, device=device, dtype=dtype),
        indexing="ij",
    )
    base_grid = torch.stack((grid_x, grid_y), dim=2).unsqueeze(0).expand(n, -1, -1, -1)
    grid = base_grid + flow

    grid_x = 2.0 * grid[..., 0] / max(w - 1, 1) - 1.0
    grid_y = 2.0 * grid[..., 1] / max(h - 1, 1) - 1.0
    sampling_grid = torch.stack((grid_x, grid_y), dim=3)

    return F.grid_sample(
        x,
        sampling_grid,
        mode=interp_mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


class ResidualBlockNoBN(nn.Module):
    def __init__(self, mid_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out


class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 64, num_blocks: int = 30):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_blocks, mid_channels=out_channels),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.main(feat)


class PixelShufflePack(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int, upsample_kernel: int = 3):
        super().__init__()
        padding = (upsample_kernel - 1) // 2
        self.upsample_conv = nn.Conv2d(
            in_channels,
            out_channels * scale_factor * scale_factor,
            upsample_kernel,
            1,
            padding,
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pixel_shuffle(self.upsample_conv(x))


class ConvAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 7, 1, 3, bias=True)
        self.activate = nn.ReLU(inplace=False) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.conv(x))


class SPyNetBasicModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_module = nn.Sequential(
            ConvAct(8, 32),
            ConvAct(32, 64),
            ConvAct(64, 32),
            ConvAct(32, 16),
            ConvAct(16, 2, act=False),
        )

    def forward(self, tensor_input: torch.Tensor) -> torch.Tensor:
        return self.basic_module(tensor_input)


class SPyNet(nn.Module):
    def __init__(self, pretrained: str | None = None):
        super().__init__()
        self.basic_module = nn.ModuleList([SPyNetBasicModule() for _ in range(6)])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        if isinstance(pretrained, str) and pretrained and Path(pretrained).is_file():
            state_dict = torch.load(pretrained, map_location="cpu", weights_only=True)
            self.load_state_dict(state_dict, strict=True)

    def compute_flow(self, ref: torch.Tensor, supp: torch.Tensor) -> torch.Tensor:
        n, _, h, w = ref.size()

        mean = self.mean
        std = self.std
        if not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor):
            raise TypeError("SPyNet buffers mean/std were not initialized as tensors")

        ref_pyramid = [(ref - mean) / std]
        supp_pyramid = [(supp - mean) / std]

        for _ in range(5):
            ref_pyramid.append(F.avg_pool2d(ref_pyramid[-1], kernel_size=2, stride=2, count_include_pad=False))
            supp_pyramid.append(F.avg_pool2d(supp_pyramid[-1], kernel_size=2, stride=2, count_include_pad=False))

        ref_pyramid = ref_pyramid[::-1]
        supp_pyramid = supp_pyramid[::-1]

        flow = ref_pyramid[0].new_zeros(n, 2, h // 32, w // 32)
        for level, (ref_level, supp_level) in enumerate(zip(ref_pyramid, supp_pyramid)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(flow, scale_factor=2, mode="bilinear", align_corners=True) * 2.0

            flow = flow_up + self.basic_module[level](
                torch.cat(
                    [
                        ref_level,
                        flow_warp(supp_level, flow_up.permute(0, 2, 3, 1), padding_mode="border"),
                        flow_up,
                    ],
                    dim=1,
                )
            )

        return flow

    def forward(self, ref: torch.Tensor, supp: torch.Tensor) -> torch.Tensor:
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)

        ref = F.interpolate(ref, size=(h_up, w_up), mode="bilinear", align_corners=False)
        supp = F.interpolate(supp, size=(h_up, w_up), mode="bilinear", align_corners=False)

        flow = F.interpolate(self.compute_flow(ref, supp), size=(h, w), mode="bilinear", align_corners=False)
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)
        return flow


class BasicVSRNet(nn.Module):
    def __init__(self, mid_channels: int = 64, num_blocks: int = 20, spynet_pretrained: str | None = None):
        super().__init__()
        self.mid_channels = mid_channels
        self.spynet = SPyNet(pretrained=spynet_pretrained)

        self.backward_resblocks = ResidualBlocksWithInputConv(mid_channels + 3, mid_channels, num_blocks)
        self.forward_resblocks = ResidualBlocksWithInputConv(mid_channels + 3, mid_channels, num_blocks)

        self.fusion = nn.Conv2d(mid_channels * 2, mid_channels, 1, 1, 0, bias=True)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.is_mirror_extended = False

    def check_if_mirror_extended(self, lrs: torch.Tensor) -> None:
        self.is_mirror_extended = False
        try:
            if torch.onnx.is_in_onnx_export():
                return
        except Exception:
            pass

        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lrs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)

        # Backward flow: frame t+1 -> frame t
        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)
        # Forward flow: frame t -> frame t+1
        flows_forward  = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, lrs: torch.Tensor) -> torch.Tensor:
        n, t, _, h, w = lrs.size()

        self.check_if_mirror_extended(lrs)
        flows_forward, flows_backward = self.compute_flow(lrs)

        outputs: list[torch.Tensor] = []
        feat_prop = lrs.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            if i < t - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([lrs[:, i, :, :, :], feat_prop], dim=1)
            feat_prop = self.backward_resblocks(feat_prop)
            outputs.append(feat_prop)
        outputs = outputs[::-1]

        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, t):
            lr_curr = lrs[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([lr_curr, feat_prop], dim=1)
            feat_prop = self.forward_resblocks(feat_prop)

            out = torch.cat([outputs[i], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.upsample1(out))
            out = self.lrelu(self.upsample2(out))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            out += self.img_upsample(lr_curr)
            outputs[i] = out

        return torch.stack(outputs, dim=1)


class NovaVSRBackbone(nn.Module):
    """Checkpoint-compatible two-stage BasicVSR backbone."""

    def __init__(
        self,
        native_ops: object | None = None,
        num_feat: int = 64,
        num_block: int = 20,
        num_cleaning_blocks: int = 20,
        num_cleaning_passes: int = 3,
        dynamic_refine_thres: float = 255.0,
        is_sequential_cleaning: bool = False,
        spynet_pretrained: str | None = None,
    ):
        super().__init__()
        self.native_ops = native_ops
        self.num_cleaning_passes = max(1, int(num_cleaning_passes))
        self.dynamic_refine_thres = float(dynamic_refine_thres) / 255.0
        self.is_sequential_cleaning = is_sequential_cleaning

        self.image_cleaning = nn.Sequential(
            ResidualBlocksWithInputConv(3, num_feat, num_cleaning_blocks),
            nn.Conv2d(num_feat, 3, 3, 1, 1, bias=True),
        )
        self.basicvsr = BasicVSRNet(num_feat, num_block, spynet_pretrained)
        self.basicvsr.spynet.requires_grad_(False)

    def forward(self, lrs: torch.Tensor) -> torch.Tensor:
        n, t, c, h, w = lrs.shape
        cleaned = lrs
        try:
            exporting = torch.onnx.is_in_onnx_export()
        except Exception:
            exporting = False

        for _ in range(self.num_cleaning_passes):
            if self.is_sequential_cleaning:
                residues: list[torch.Tensor] = []
                updated_frames: list[torch.Tensor] = []
                for i in range(t):
                    frame = cleaned[:, i, :, :, :]
                    residue_i = self.image_cleaning(frame)
                    updated_frames.append(frame + residue_i)
                    residues.append(residue_i)
                residues_tensor = torch.stack(residues, dim=1)
                cleaned = torch.stack(updated_frames, dim=1)
            else:
                residues_tensor = self.image_cleaning(cleaned.reshape(-1, c, h, w)).reshape(n, t, c, h, w)
                cleaned = cleaned + residues_tensor

            if not exporting and torch.mean(torch.abs(residues_tensor)) < self.dynamic_refine_thres:
                break
        return self.basicvsr(cleaned)
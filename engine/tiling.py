import torch
import torch.nn.functional as F

class TileHandler:
    def __init__(self, tile_size=1024, overlap=64):
        self.tile_size = tile_size
        self.overlap = overlap

    def _compute_tiles(self, H, W):
        tiles = []

        y = 0
        while y < H:
            y_end = min(y + self.tile_size, H)
            tiles.append((y, y_end))
            if y_end == H:
                break
            y = y_end - self.overlap

        x = 0
        x_tiles = []
        while x < W:
            x_end = min(x + self.tile_size, W)
            x_tiles.append((x, x_end))
            if x_end == W:
                break
            x = x_end - self.overlap

        return [(y0, y1, x0, x1) for (y0, y1) in tiles for (x0, x1) in x_tiles]

    def process_tiles(self, lrs, model):
        b, t, c, h, w = lrs.shape
        preferred_input_shape = getattr(model, "input_spatial_shape", None)
        tile_h_limit = self.tile_size
        tile_w_limit = self.tile_size
        if preferred_input_shape is not None:
            tile_h_limit = min(tile_h_limit, int(preferred_input_shape[0]))
            tile_w_limit = min(tile_w_limit, int(preferred_input_shape[1]))

        stride_h = tile_h_limit - self.overlap
        stride_w = tile_w_limit - self.overlap
        if stride_h <= 0 or stride_w <= 0:
            raise ValueError("Tile size must be larger than overlap")

        scale_h = None
        scale_w = None
        output = None
        weight_mask = None

        for y in range(0, h, stride_h):
            for x in range(0, w, stride_w):
                y1, x1 = y, x
                y2, x2 = min(y + tile_h_limit, h), min(x + tile_w_limit, w)

                tile_lr = lrs[:, :, :, y1:y2, x1:x2]
                model_input = tile_lr

                if preferred_input_shape is not None:
                    pad_h = int(preferred_input_shape[0]) - tile_lr.shape[-2]
                    pad_w = int(preferred_input_shape[1]) - tile_lr.shape[-1]
                    if pad_h < 0 or pad_w < 0:
                        raise ValueError(
                            "Tile dimensions exceed the TensorRT engine input shape; rebuild the engine or reduce tile_size."
                        )
                    if pad_h or pad_w:
                        model_input = tile_lr.reshape(b * t, c, tile_lr.shape[-2], tile_lr.shape[-1])
                        model_input = F.pad(model_input, (0, pad_w, 0, pad_h), mode="replicate")
                        model_input = model_input.reshape(b, t, c, preferred_input_shape[0], preferred_input_shape[1])

                with torch.no_grad():
                    tile_sr = model(model_input)

                if scale_h is None or scale_w is None:
                    input_h, input_w = model_input.shape[-2:]
                    out_h, out_w = tile_sr.shape[-2:]
                    if out_h % input_h != 0 or out_w % input_w != 0:
                        raise ValueError(
                            f"Model output shape {tile_sr.shape[-2:]} is not an integer upscale of input shape {model_input.shape[-2:]}"
                        )

                    scale_h = out_h // input_h
                    scale_w = out_w // input_w
                    oh, ow = h * scale_h, w * scale_w
                    output = torch.zeros((tile_sr.shape[0], tile_sr.shape[1], tile_sr.shape[2], oh, ow), device=lrs.device, dtype=tile_sr.dtype)
                    weight_mask = torch.zeros((1, 1, 1, oh, ow), device=lrs.device, dtype=tile_sr.dtype)

                if output is None or weight_mask is None or scale_h is None or scale_w is None:
                    raise RuntimeError("Failed to initialize tile accumulation buffers")

                valid_sr_h = (y2 - y1) * scale_h
                valid_sr_w = (x2 - x1) * scale_w
                tile_sr = tile_sr[:, :, :, :valid_sr_h, :valid_sr_w]

                sy1, sx1 = y1 * scale_h, x1 * scale_w
                sy2, sx2 = sy1 + valid_sr_h, sx1 + valid_sr_w

                tile_h, tile_w = tile_sr.shape[-2:]
                tile_weight = torch.ones((1, 1, 1, tile_h, tile_w), device=lrs.device, dtype=tile_sr.dtype)

                output[:, :, :, sy1:sy2, sx1:sx2] += tile_sr
                weight_mask[:, :, :, sy1:sy2, sx1:sx2] += tile_weight

        if output is None or weight_mask is None:
            raise ValueError("No tiles were processed")

        return output / weight_mask.clamp_min(1)

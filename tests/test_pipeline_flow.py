import torch
from engine.pipeline import VideoProcessor

def test_window_stacking():
    dummy_frames = [torch.full((3, 64, 64), i, device='cuda') for i in range(20)]

    processor = VideoProcessor(model=None, native_ops=None, window_size=5)

    window = []
    for i, frame in enumerate(dummy_frames):
        window.append(frame)
        if len(window) == 5:
            stack = torch.stack(window)
            center_val = stack[2, 0, 0, 0].item()
            print(f"Window processed. Input Frame Index: {i}, Center Frame Value: {center_val}")

            window.pop(0)

if __name__ == "__main__":
    test_window_stacking()
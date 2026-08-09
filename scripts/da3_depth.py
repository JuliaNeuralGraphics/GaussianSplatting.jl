import argparse
import os

import imageio
import numpy as np
import torch

from depth_anything_3.api import DepthAnything3

IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Depth Anything 3 on a directory of images.")
    parser.add_argument("input_dir", type=str, help="Directory containing input images.")
    parser.add_argument("output_dir", type=str, help="Directory to write depth map PNGs to.")
    parser.add_argument(
        "--model",
        type=str,
        default="depth-anything/DA3NESTED-GIANT-LARGE",
        help="Pretrained model name or path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    valid_exts = {ext.lstrip("*").lower() for ext in IMAGE_EXTENSIONS}
    images = sorted(
        entry.path
        for entry in os.scandir(args.input_dir)
        if entry.is_file() and os.path.splitext(entry.name)[1].lower() in valid_exts
    )
    if not images:
        raise ValueError(f"No images found in {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained(args.model)
    model = model.to(device=device)

    for image_path in images:
        prediction = model.inference([image_path])
        depth = prediction.depth[0]

        valid_mask = depth > 0
        if valid_mask.sum() > 0:
            depth_min = depth[valid_mask].min()
            depth_max = depth[valid_mask].max()
        else:
            depth_min, depth_max = 0.0, 1.0
        depth_norm = ((depth - depth_min) / (depth_max - depth_min + 1e-6)).clip(0, 1)
        depth_u16 = (depth_norm * 65535).astype(np.uint16)

        out_name = os.path.splitext(os.path.basename(image_path))[0] + ".png"
        imageio.imwrite(os.path.join(args.output_dir, out_name), depth_u16)


if __name__ == "__main__":
    main()

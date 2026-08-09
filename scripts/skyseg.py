"""Sky segmentation with the U-2-Net skyseg ONNX model.

Model: https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing
Weights are fetched from the HuggingFace mirror on first run if not present.
"""

import argparse
import os
import urllib.request

import cv2 as cv
import numpy as np
import onnxruntime

IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif")

MODEL_URL = "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Segment sky in a directory of images.")
    parser.add_argument("input_dir", type=str, help="Directory containing input images.")
    parser.add_argument("output_dir", type=str, help="Directory to write sky masks to.")
    parser.add_argument(
        "--model",
        type=str,
        default="skyseg.onnx",
        help="Path to skyseg.onnx (downloaded if missing).",
    )
    parser.add_argument("--input-size", type=int, default=320, help="Network input resolution.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="If set, binarize the mask at this probability (0-1) instead of saving soft values.",
    )
    parser.add_argument(
        "--minmax",
        action="store_true",
        help="Min-max stretch each mask, as in the reference script. Off by default because it "
        "amplifies noise into a full-contrast mask on images that contain no sky.",
    )
    parser.add_argument("--overlay", action="store_true", help="Also write a colored overlay preview.")
    return parser.parse_args()


def download_model(path):
    print(f"Downloading skyseg model to {path} ...")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, path)


def run_inference(session, input_size, image):
    """Return a float32 sky probability map in [0, 1] at the original image size."""
    # Pre process: resize, BGR->RGB, ImageNet standardization, HWC->NCHW
    resized = cv.resize(image, (input_size, input_size), interpolation=cv.INTER_LINEAR)
    x = cv.cvtColor(resized, cv.COLOR_BGR2RGB).astype(np.float32)
    x = (x / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    x = x.transpose(2, 0, 1)[None].astype(np.float32)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: x})[0]

    # Post process: the network already emits a sigmoid saliency map
    mask = np.asarray(result).squeeze().astype(np.float32)
    mask = cv.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv.INTER_LINEAR)
    return mask


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

    if not os.path.isfile(args.model):
        download_model(args.model)

    os.makedirs(args.output_dir, exist_ok=True)

    providers = onnxruntime.get_available_providers()
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in providers]
    session = onnxruntime.InferenceSession(args.model, providers=providers)

    for image_path in images:
        image = cv.imread(image_path, cv.IMREAD_COLOR)
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        mask = run_inference(session, args.input_size, image)

        if args.minmax:
            lo, hi = float(mask.min()), float(mask.max())
            mask = (mask - lo) / (hi - lo + 1e-6)
        mask = np.clip(mask, 0.0, 1.0)

        if args.threshold is not None:
            mask = (mask >= args.threshold).astype(np.float32)
        mask_u8 = (mask * 255).round().astype(np.uint8)

        stem = os.path.splitext(os.path.basename(image_path))[0]
        cv.imwrite(os.path.join(args.output_dir, stem + ".png"), mask_u8)

        if args.overlay:
            tint = np.zeros_like(image)
            tint[:, :] = (255, 128, 0)  # BGR
            alpha = (mask * 0.5)[..., None]
            blended = (image * (1 - alpha) + tint * alpha).astype(np.uint8)
            cv.imwrite(os.path.join(args.output_dir, stem + "_overlay.jpg"), blended)

        print(f"{image_path} -> sky coverage {float(mask.mean()) * 100:.1f}%")


if __name__ == "__main__":
    main()

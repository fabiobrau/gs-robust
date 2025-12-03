r"""Visualize splats and fitting error."""
from argparse import ArgumentParser

import torch
import torchvision.transforms.functional as tf
from PIL import Image
from splats import GaussianSplats
from torch import Tensor


def parse_args():
    r"""Parse command line arguments."""
    parser = ArgumentParser(description=("Create Gif for Gaussian Splatting ",
                                            "Image rendering demonstration."))
    parser.add_argument("img_path", type=str, help="Path to image to render.")
    parser.add_argument("--n_gaussians", default=500, type=int, help="Number of splats.")
    parser.add_argument("--iters", default = 200, type=int, help="Iterations for training.")
    parser.add_argument("--out_path", type=str, default="splat_rendering.gif",
                        help="Path to output gif.")
    return parser.parse_args()

def import_image_as_tensor(img_path: str)-> Tensor:
    r"""Import image as tensor."""
    img = Image.open(img_path).convert("RGB")
    return tf.to_tensor(img)

def main():
    args = parse_args()
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using GPU for fitting.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS for fitting.")

    img = import_image_as_tensor(args.img_path).to(device)
    model = GaussianSplats(n_gaussians=args.n_gaussians, n_channels=3).to(device)
    iters = list(model._fit_iterable(img, n_iters=args.iters))
    frames = [tf.to_pil_image(it[0].to("cpu")) for it in iters]
    # Save gif from list
    frames[0].save(args.out_path, save_all=True, append_images=frames[1:], 
                   duration=args.iters//30)

if __name__ == "__main__":
    import sys
    sys.argv += ["../Downloads/gauss.jpg"]  # Example default argument
    main()



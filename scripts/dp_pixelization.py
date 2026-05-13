import numpy as np
import cv2
from pathlib import Path

def dp_pixelization(img: np.ndarray, block_size: int, epsilon: float) -> np.ndarray:
    """
    Apply differential privacy pixelization to the input image.

    Parameters:
    img (np.ndarray): Input image as a NumPy array.
    block_size (int): Size of the blocks to pixelate.
    epsilon (float): Privacy budget for differential privacy.

    Returns:
    np.ndarray: Pixelated image with differential privacy applied.
    """
    # Get the dimensions of the image
    h, w = img.shape
    img_f = img.astype(np.float64)
    # Create a copy of the image to modify
    pixelated_img = img_f.copy()

    sensitivity = 255.0 / (block_size**2)  # Sensitivity for mean color calculation
    scale = sensitivity / epsilon

    # Loop through the image in blocks
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            # Define the block boundaries
            block = img_f[i:i+block_size, j:j+block_size]

            # Calculate the mean color of the block
            mean_color = block.mean()

            # Add Laplace noise to the mean color for differential privacy
            noise = np.random.laplace(loc=0.0, scale=scale)
            noised_val = mean_color + noise

            # Assign the noisy mean color to the block
            pixelated_img[i:i+block_size, j:j+block_size] = noised_val

    pixelated_img = np.clip(pixelated_img, 0, 255).astype(np.uint8)
    return pixelated_img
def process_dataset(
    input_dir: str,
    output_dir: str,
    block_size: int,
    epsilon: float,
    seed: int | None = None,
):
    """Apply DP-Pixelization to every PGM under ``input_dir``."""
    if seed is not None:
        np.random.seed(seed)

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for pgm in sorted(input_path.rglob("*.pgm")):
        img = cv2.imread(str(pgm), cv2.IMREAD_GRAYSCALE)
        pixelated_img = dp_pixelization(img, block_size, epsilon)
        rel = pgm.relative_to(input_path)
        output_file = output_path / rel.with_suffix(".png")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_file), pixelated_img)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--src",   default="data/att_faces")
    parser.add_argument("--dst",   default="output/dp_pix")
    parser.add_argument("--block", type=int, default=8)
    parser.add_argument("--eps",   type=float, default=1.0)
    parser.add_argument("--seed",  type=int, default=None, help="Optional NumPy seed for reproducible noise.")
    args = parser.parse_args()

    out_dir = f"{args.dst}_b{args.block}/eps{args.eps}"
    process_dataset(args.src, out_dir, args.block, args.eps, seed=args.seed)
    print(f"Done → {out_dir}")
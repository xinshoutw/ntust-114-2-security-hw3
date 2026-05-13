import numpy as np
import cv2
from pathlib import Path


def dp_blur(img: np.ndarray, epsilon: float, kernel_size: int = 5, sigma: float = 1.0, m: int = 1) -> np.ndarray:
    """
    Apply differential privacy blurring to the input image (Fan 2018).

    Sensitivity follows Fan (2018) DP-Pix: Δf = 255 * m / k²,
    where k = kernel_size and m = neighborhood parameter.

    Parameters:
    img (np.ndarray): Input image as a NumPy array.
    epsilon (float): Privacy budget for differential privacy.
    kernel_size (int): Size of the Gaussian kernel for blurring.
    sigma (float): Standard deviation for Gaussian kernel.
    m (int): Neighborhood parameter (number of pixels that may differ).

    Returns:
    np.ndarray: Blurred image with differential privacy applied.
    """
    img_f = img.astype(np.float64)
    blurred_img = cv2.GaussianBlur(img_f, (kernel_size, kernel_size), sigma)

    sensitivity = 255.0 * m / (kernel_size ** 2)
    scale = sensitivity / epsilon

    noise = np.random.laplace(loc=0.0, scale=scale, size=blurred_img.shape)
    noised_blurred_img = blurred_img + noise

    noised_blurred_img = np.clip(noised_blurred_img, 0, 255).astype(np.uint8)
    return noised_blurred_img

def process_dataset(input_dir: str, output_dir: str, epsilon: float, kernel_size: int = 5, sigma: float = 1.0):
    """
    Process a dataset of images by applying differential privacy blurring.

    Parameters:
    input_dir (str): Directory containing the input images.
    output_dir (str): Directory to save the processed images.
    epsilon (float): Privacy budget for differential privacy.
    kernel_size (int): Size of the Gaussian kernel for blurring.
    sigma (float): Standard deviation for Gaussian kernel.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for pgm in sorted(input_path.rglob("*.pgm")):
        img = cv2.imread(str(pgm), cv2.IMREAD_GRAYSCALE)
        blurred_img = dp_blur(img, epsilon, kernel_size, sigma)
        rel = pgm.relative_to(input_path)
        output_file = output_path / rel.with_suffix(".png")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_file), blurred_img)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/att_faces")
    parser.add_argument("--dst", default="output/dp_blur")
    parser.add_argument("--eps", type=float, default=1.0)
    args = parser.parse_args()

    out_dir = f"{args.dst}/eps{args.eps}"
    process_dataset(args.src, out_dir, args.eps)
    print(f"Done → {out_dir}")
"""
preprocessing.py
=================
Image preprocessing utilities for the OCR pipeline.

Two preprocessing modes are provided:

1. `preprocess_document()` — a general-purpose pipeline (grayscale ->
   denoise -> adaptive threshold -> deskew -> resize) suited to normal
   documents, screenshots, receipts, signage, etc. This is the default
   mode used by `src/pipeline.py`.

2. `dilate_image()`, `erode_image()`, `compute_gradient()` and
   `preprocess_code_crop()` — carried over from `notebooks/CIG_MODEL.ipynb`.
   That notebook trains a custom CTC recognizer for short alphanumeric
   codes printed on cigarette packaging, and it arrived at a
   "morphological close + Sobel-gradient second channel" preprocessing
   recipe after several failed attempts (plain morphology, contour-based
   blob removal, ESRGAN+SAM segmentation — see the notebook's Section 3a
   for the full write-up). `preprocess_code_crop()` re-implements that
   same cleaned-image output (channel 0 of the notebook's two-channel
   tensor) as a standalone, Tesseract-friendly preprocessing step, so the
   lessons from that notebook are actually reachable from this pipeline
   instead of being stranded in a training-only notebook.
"""

from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# General-purpose document preprocessing
# ---------------------------------------------------------------------------

def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert a BGR (or already-gray) image to single-channel grayscale."""
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()


def denoise(gray: np.ndarray) -> np.ndarray:
    """Light denoising that preserves character edges."""
    return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)


def adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    """Binarize using Gaussian-weighted adaptive thresholding.

    Adaptive (rather than global Otsu) thresholding is used because it
    copes much better with uneven lighting / scan gradients across a
    photographed or screenshotted document.
    """
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=15,
    )


def deskew(binary: np.ndarray) -> np.ndarray:
    """Estimate and correct small rotational skew using the minimum-area
    bounding rectangle of all foreground (text) pixels.
    """
    coords = np.column_stack(np.where(binary < 255))
    if coords.shape[0] < 20:
        return binary  # not enough foreground pixels to estimate a reliable angle

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Skip correction for negligible angles to avoid introducing blur on
    # already-straight images.
    if abs(angle) < 0.5:
        return binary

    (h, w) = binary.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        binary, matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def upscale_if_small(gray: np.ndarray, min_height: int = 600) -> np.ndarray:
    """Tesseract accuracy drops sharply on small text; upscale small images."""
    h, w = gray.shape[:2]
    if h >= min_height:
        return gray
    scale = min_height / float(h)
    return cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)


def preprocess_document(img: np.ndarray) -> np.ndarray:
    """Full general-purpose preprocessing pipeline.

    Steps: grayscale -> upscale (if small) -> denoise -> adaptive
    threshold -> deskew. Returns a binary (0/255) image ready for
    Tesseract.
    """
    gray = to_grayscale(img)
    gray = upscale_if_small(gray)
    gray = denoise(gray)
    binary = adaptive_threshold(gray)
    binary = deskew(binary)
    return binary


# ---------------------------------------------------------------------------
# Morphological pipeline carried over from notebooks/CIG_MODEL.ipynb
# (Section 3b: "Preprocessing Helper Functions")
# ---------------------------------------------------------------------------

def dilate_image(img: np.ndarray, ksize_x: int = 3, ksize_y: int = 1, iterations: int = 1) -> np.ndarray:
    """Morphological dilation — expands bright (foreground) pixels."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize_x, ksize_y))
    return cv2.dilate(img, kernel, iterations=iterations)


def erode_image(img: np.ndarray, ksize_x: int = 3, ksize_y: int = 1, iterations: int = 1) -> np.ndarray:
    """Morphological erosion — shrinks bright (foreground) pixels.

    Dilation followed by erosion (a morphological "closing") removes
    speckle noise around text strokes while mostly preserving character
    shape — this is the cleaning step the CIG_MODEL notebook settled on
    after contour-based blob removal and ESRGAN+SAM segmentation both
    failed on this dataset.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize_x, ksize_y))
    return cv2.erode(img, kernel, iterations=iterations)


def compute_gradient(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Sobel gradient magnitude — highlights character stroke edges.

    In the notebook this becomes the second input channel fed to the CTC
    model; here it is only used internally as an optional sharpening
    signal before thresholding.
    """
    img_f = img.astype(np.float32)
    grad_x = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=ksize)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return magnitude.astype(np.uint8)


def preprocess_code_crop(img: np.ndarray) -> np.ndarray:
    """Preprocessing for small, noisy crops of short printed codes
    (e.g. batch/date codes), reusing the morphological recipe from
    `notebooks/CIG_MODEL.ipynb`.

    Pipeline: grayscale -> dilate -> erode (closing) -> Otsu threshold.
    Returns a binary image tuned for Tesseract's single-line / single-word
    page-segmentation modes.
    """
    gray = to_grayscale(img)
    dilated = dilate_image(gray, ksize_x=3, ksize_y=1, iterations=1)
    closed = erode_image(dilated, ksize_x=3, ksize_y=1, iterations=1)
    _, binary = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

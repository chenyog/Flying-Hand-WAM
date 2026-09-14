"""RGB JPEG helpers for current-format Flying-Hand HDF5 files.

OpenCV encodes and decodes color images in BGR order, while SAPIEN and every
policy adapter in this repository use RGB arrays. Flying-Hand HDF5 inputs must
explicitly declare the current codec; unmarked and legacy files are rejected.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


RGB_JPEG_CODEC_ATTR = "rgb_jpeg_codec"
RGB_JPEG_CODEC = "opencv_bgr_from_rgb_v2"


def require_current_rgb_jpeg_codec(root: Any) -> None:
    """Reject HDF5 inputs that do not explicitly use the current RGB codec."""
    source = getattr(root, "filename", None) or "Flying-Hand HDF5 file"
    if RGB_JPEG_CODEC_ATTR not in root.attrs:
        raise ValueError(
            f"{source}: missing required HDF5 root attribute "
            f"{RGB_JPEG_CODEC_ATTR!r}; regenerate this legacy dataset"
        )
    value = root.attrs[RGB_JPEG_CODEC_ATTR]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    value = str(value)
    if value != RGB_JPEG_CODEC:
        raise ValueError(
            f"{source}: unsupported Flying-Hand RGB JPEG codec {value!r}; "
            f"expected {RGB_JPEG_CODEC!r}"
        )


def encode_rgb_jpeg(image: np.ndarray) -> bytes:
    """Encode one uint8 RGB image as a standards-compliant JPEG byte string."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image [H, W, 3], got {image.shape}")
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".jpg", bgr)
    if not success:
        raise ValueError("OpenCV failed to encode an RGB image as JPEG")
    return encoded.tobytes()


def decode_hdf5_rgb(value: Any) -> np.ndarray:
    """Decode one current-format Flying-Hand HDF5 JPEG as RGB."""
    image = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV failed to decode a Flying-Hand RGB JPEG")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

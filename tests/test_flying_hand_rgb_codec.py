import unittest

import numpy as np

from envs.utils.rgb_codec import (
    RGB_JPEG_CODEC,
    RGB_JPEG_CODEC_ATTR,
    decode_hdf5_rgb,
    encode_rgb_jpeg,
    require_current_rgb_jpeg_codec,
)


class FlyingHandRgbCodecTest(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        self.rgb[..., 0] = 220
        self.rgb[..., 1] = 70
        self.rgb[..., 2] = 15

    def assert_channel_order(self, decoded):
        means = decoded.mean(axis=(0, 1))
        self.assertGreater(means[0], means[1])
        self.assertGreater(means[1], means[2])

    def test_current_round_trip_preserves_rgb_channel_order(self):
        decoded = decode_hdf5_rgb(encode_rgb_jpeg(self.rgb))
        self.assert_channel_order(decoded)

    def test_current_codec_attribute_is_accepted(self):
        root = type("Root", (), {"attrs": {RGB_JPEG_CODEC_ATTR: RGB_JPEG_CODEC}})()
        require_current_rgb_jpeg_codec(root)

    def test_missing_codec_attribute_is_rejected(self):
        root = type("Root", (), {"attrs": {}})()
        with self.assertRaisesRegex(ValueError, "missing required HDF5 root attribute"):
            require_current_rgb_jpeg_codec(root)

    def test_legacy_codec_attribute_is_rejected(self):
        root = type(
            "Root",
            (),
            {"attrs": {RGB_JPEG_CODEC_ATTR: "legacy_rgb_passed_as_bgr_v1"}},
        )()
        with self.assertRaisesRegex(ValueError, "unsupported Flying-Hand RGB JPEG codec"):
            require_current_rgb_jpeg_codec(root)


if __name__ == "__main__":
    unittest.main()

import sys

from PIL import Image

import numpy as np
from einops import rearrange


def main(args):
    features = np.load(args.features, allow_pickle=False)

    assert len(features.shape) == 2
    assert features.shape[1] == 512
    assert features.dtype == np.float32

    w, h = features.shape

    image = Image.new(mode="HSV", size=(w, h))

    for i, feature in enumerate(features):
        feature = feature / np.linalg.norm(feature)
        assert (feature >= 0).all()
        assert (feature <= 1).all()
        assert feature.dtype == np.float32

        # Map color values in HSV color space
        # Range for H, S, V is [0, 255] uint8

        h = ((feature * 255 + 20) % 255).astype(np.uint8)

        s = np.ones(512, dtype=np.uint8) * 128
        v = np.ones(512, dtype=np.uint8) * 255

        hsv = np.array((h, s, v))
        hsv = rearrange(hsv, "c h -> h () c")

        assert hsv.shape == (512, 1, 3)
        assert hsv.dtype == np.uint8

        strip = Image.fromarray(hsv, mode="HSV")
        assert strip.size == (1, 512)

        image.paste(strip, box=(i, 0))

    image.convert("RGB").save(args.image, optimize=True)
    print("🔰 Done", file=sys.stderr)

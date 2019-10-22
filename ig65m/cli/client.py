import sys
import json
from xmlrpc.client import ServerProxy

import numpy as np
from einops import rearrange


def main(args):
    feats = np.load(args.features, allow_pickle=False)
    assert len(feats.shape) == 2
    assert feats.shape[1] == args.dimension
    assert feats.dtype == np.float32

    assert 0 <= args.clip < feats.shape[0]

    feat = feats[args.clip]
    feat = rearrange(feat, "n -> () n")

    print("📱 Calling similarity server on {}:{}".format(args.host, args.port), file=sys.stderr)

    with ServerProxy("http://{}:{}".format(args.host, args.port)) as client:
        batch_dists, batch_indices, batch_metas = client.query(feat.tobytes(), args.num_results)

        assert len(batch_dists) == 1
        assert len(batch_indices) == 1
        assert len(batch_metas) == 1

        dists = batch_dists[0]
        indices = batch_indices[0]
        metas = batch_metas[0]

        features = []

        for dist, index, meta in zip(dists, indices, metas):
            path = meta["path"]
            clip = meta["clip"]

            feature = {"distance": round(dist, 2), "path": path, "clip": clip}

            features.append(feature)

        print(json.dumps(features))

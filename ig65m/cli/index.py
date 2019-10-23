import sys
import json

import numpy as np

from tqdm import tqdm

from faiss import IndexFlatL2, IndexIVFPQ, write_index

from ig65m.samplers import StreamSampler


# GPU index constraints
# - https://github.com/facebookresearch/faiss/blob/a8118acbc516b0263dde610862c806400cc48bf5/gpu/impl/IVFPQ.cu#L69-L92
# - https://github.com/facebookresearch/faiss/blob/a8118acbc516b0263dde610862c806400cc48bf5/ProductQuantizer.cpp#L189


def main(args):
    # https://github.com/facebookresearch/faiss/blob/a8118acbc516b0263dde610862c806400cc48bf5/Clustering.cpp#L78-L80
    if args.num_train < max(args.num_centroids, args.code_size):
        sys.exit("💥 Require at least {} training samples".format(max(args.num_centroids, args.code_size)))

    paths = [path for path in args.features.iterdir() if path.is_file()]

    print("🍪 Loading clip features from {} video feature files at {}"
          .format(len(paths), args.features), file=sys.stderr)

    # Two passes over videos of arbitrary number of clips:
    # - First pass reservoir samples clip features, trains index
    # - Second pass adds remaining clip features to index
    #
    # This way we can properly randomly select train samples and
    # at the same time keep our peak memory consumption reasonable.

    train_samples = StreamSampler(args.num_train)

    # 1st pass

    total_clips = 0

    for i, path in enumerate(tqdm(paths)):
        feats = np.load(path, allow_pickle=False)

        assert len(feats.shape) == 2
        assert feats.shape[1] == args.dimension
        assert feats.dtype == np.float32

        for j, feat in enumerate(feats):
            # Keep train and index datasets disjoint
            # Track train clips: ith video, jth clip
            train_samples.add((feat, (i, j)))
            total_clips += 1

    if len(train_samples) < args.num_train:
        sys.exit("💥 Not enough samples in dataset to train on; loaded {}".format(len(train_samples)))

    train_feats = [k for k, _ in train_samples]
    train_clips = {v for _, v in train_samples}

    train_feats = np.array(train_feats)
    assert train_feats.shape == (args.num_train, args.dimension)
    assert train_feats.dtype == np.float32

    quantizer = IndexFlatL2(args.dimension)

    index = IndexIVFPQ(quantizer, args.dimension, args.num_centroids, args.code_size, args.num_bits)

    print("🚄 Training index on {} out of {} total {}-dimensional clip features"
          .format(args.num_train, total_clips, args.dimension), file=sys.stderr)

    index.train(train_feats)

    del train_feats
    del train_samples

    # 2nd pass

    assert index.is_trained

    print("🔖 Adding to index {} out of {} total {}-dimensional clip features"
          .format(total_clips - len(train_clips), total_clips, args.dimension), file=sys.stderr)

    metadata = []
    batch_feats = []

    for i, path in enumerate(tqdm(paths)):
        feats = np.load(path, allow_pickle=False)

        assert len(feats.shape) == 2
        assert feats.shape[1] == args.dimension
        assert feats.dtype == np.float32

        for j, feat in enumerate(feats):
            if (i, j) in train_clips:
                continue

            batch_feats.append(feat)

            # Could be more efficient than one entry per clip.
            # This way it's simple to use in the client for now.
            metadata.append({"path": str(path), "clip": j})

            if len(batch_feats) % args.batch_size == 0:
                feats = np.array(batch_feats)
                batch_feats.clear()
                index.add(feats)

    if batch_feats:
        feats = np.array(batch_feats)
        batch_feats.clear()
        index.add(feats)

    assert index.ntotal == total_clips - len(train_clips)

    write_index(index, str(args.index.with_suffix(".idx")))

    with args.index.with_suffix(".json").open("w") as fp:
        json.dump(metadata, fp)

    print("📖 Done", file=sys.stderr)

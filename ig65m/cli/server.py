import sys
import json
from xmlrpc.server import SimpleXMLRPCServer

import numpy as np
from einops import rearrange

from faiss import read_index


def main(args):
    index = read_index(str(args.index.with_suffix(".idx")))
    index.nprobe = args.num_probes

    with args.index.with_suffix(".json").open() as fp:
        metadata = json.load(fp)

    def query(batch, n):
        feats = np.frombuffer(batch.data, dtype=np.float32)
        feats = rearrange(feats, "(n d) -> n d", d=args.dimension)
        assert len(feats.shape) == 2
        assert feats.shape[1] == args.dimension
        assert feats.dtype == np.float32

        dists, indices = index.search(feats, n)

        meta = [[metadata[i] for i in batch] for batch in indices]

        return dists.tolist(), indices.tolist(), meta

    with SimpleXMLRPCServer((args.host, args.port), logRequests=False) as server:
        server.register_function(query)

        try:
            print("⏳ Waiting for similarity calls on {}:{}".format(args.host, args.port), file=sys.stderr)
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n⌛ Done", file=sys.stderr)

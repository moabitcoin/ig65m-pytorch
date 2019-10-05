import argparse
from pathlib import Path

import ig65m.cli.convert
import ig65m.cli.extract
import ig65m.cli.semcode
import ig65m.cli.dreamer


parser = argparse.ArgumentParser(prog="ig65m")

subcmd = parser.add_subparsers(dest="command")
subcmd.required = True

Formatter = argparse.ArgumentDefaultsHelpFormatter


convert = subcmd.add_parser("convert", help="🍝 converts model weights", formatter_class=Formatter)
convert.add_argument("pkl", type=Path, help=".pkl file to read the R(2+1)D 34 layer weights from")
convert.add_argument("out", type=Path, help="prefix to save converted R(2+1)D 34 layer weights to")
convert.add_argument("--frames", type=int, choices=(8, 32), required=True, help="clip frames for video model")
convert.add_argument("--classes", type=int, choices=(359, 400, 487), required=True, help="classes in last layer")
convert.set_defaults(main=ig65m.cli.convert.main)


extract = subcmd.add_parser("extract", help="🍪 extracts video features", formatter_class=Formatter)
extract.add_argument("video", type=Path, help="video to run feature extraction on")
extract.add_argument("features", type=Path, help="file to save video features to")
extract.add_argument("--frame-size", type=int, default=128, help="size of smaller edge for frame resizing")
extract.add_argument("--batch-size", type=int, default=1, help="number of sequences per batch for inference")
extract.add_argument("--pool-spatial", type=str, choices=("mean", "max"), default="mean", help="spatial pooling")
extract.add_argument("--pool-temporal", type=str, choices=("mean", "max"), default="mean", help="temporal pooling")
extract.set_defaults(main=ig65m.cli.extract.main)


semcode = subcmd.add_parser("semcode", help="🔰 generates semantic codes", formatter_class=Formatter)
semcode.add_argument("features", type=Path, help="file to read video features from")
semcode.add_argument("image", type=Path, help="file to save semantic code image to")
semcode.add_argument("--color", type=int, default=20, help="HSV hue in angle [0, 360]")
semcode.set_defaults(main=ig65m.cli.semcode.main)


dreamer = subcmd.add_parser("dreamer", help="💤 dream of electric sheep", formatter_class=Formatter)
dreamer.add_argument("image", type=Path, help="file to save dream animation to")
dreamer.add_argument("--lr", type=float, default=1e-2, help="how lucid the dream is")
dreamer.add_argument("--num-epochs", type=int, default=100, help="how long to dream")
dreamer.set_defaults(main=ig65m.cli.dreamer.main)


args = parser.parse_args()
args.main(args)

import argparse
from pathlib import Path

import ig65m.cli.convert
import ig65m.cli.extract


parser = argparse.ArgumentParser(prog="ig65m")

subcmd = parser.add_subparsers(title="commands", metavar="")
subcmd.required = True

Formatter = argparse.ArgumentDefaultsHelpFormatter


convert = subcmd.add_parser("convert", help="converts model weights", formatter_class=Formatter)
convert.add_argument("pkl", type=Path, help=".pkl file to read the R(2+1)D 34 layer weights from")
convert.add_argument("out", type=Path, help="prefix to save converted R(2+1)D 34 layer weights to")
convert.add_argument("--frames", type=int, choices=(8, 32), required=True, help="clip frames for video model")
convert.add_argument("--classes", type=int, choices=(359, 400, 487), required=True, help="classes in last layer")
convert.set_defaults(main=ig65m.cli.convert.main)


extract = subcmd.add_parser("extract", help="extracts video features", formatter_class=Formatter)
extract.add_argument("model", type=Path, help=".pth file to load model weights from")
extract.add_argument("video", type=Path, help="video file to run feature extraction on")
extract.add_argument("--frames", type=int, choices=(8, 32), required=True, help="clip frames for video model")
extract.add_argument("--classes", type=int, choices=(359, 400, 487), required=True, help="classes in last layer")
extract.add_argument("--batch-size", type=int, default=1, help="number of sequences per batch for inference")
extract.add_argument("--num-workers", type=int, default=0, help="number of workers for data loading")
extract.add_argument("--labels", type=Path, help="JSON file with label map array")
extract.set_defaults(main=ig65m.cli.extract.main)


args = parser.parse_args()
args.main(args)

#!/usr/bin/env python3
"""Download the AIMET ResNet-18 model card, AdaRound encodings and optional weights."""

import argparse
import json
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

CARD_URL = (
    "https://raw.githubusercontent.com/quic/aimet-model-zoo/develop/"
    "aimet_zoo_torch/resnet/model/model_cards/resnet18_w4a8.json"
)


def fetch(url, path):
    if not path.is_file():
        with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            partial = Path(temporary.name)
            try:
                with urlopen(url, timeout=120) as source:
                    copyfileobj(source, temporary)
                temporary.close()
                partial.replace(path)
            finally:
                partial.unlink(missing_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", action="store_true", help="Also download the folded checkpoint")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    card = fetch(CARD_URL, args.output / "aimet-resnet18-w4a8.json")
    artifacts = json.loads(card.read_text())["artifacts"]
    fetch(artifacts["url_adaround_encodings"], args.output / "resnet18_W4A8.encodings")
    if args.weights:
        fetch(artifacts["url_post_opt_weights"], args.output / "resnet18_w4a8_state_dict.pth")


if __name__ == "__main__":
    main()

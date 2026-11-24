"""Versioned public layout of an exported quantized ImageNet model."""

import struct
import json
from pathlib import Path
from typing import NamedTuple


def checkpoint(folder):
    path = Path(folder) / "model.json"
    meta = json.loads(path.read_text()) if path.exists() else {}
    arch = meta.get("architecture", "resnet18")
    return arch, meta.get("source", "aimet" if arch == "resnet18" else "torchvision")


class Arithmetic(NamedTuple):
    bits: int = 20
    shift: int = 9
    weight_bits: int = 4
    activation_bits: int = 8
    fc_bits: int = 8

    def validate(self):
        if not (
            20 <= self.bits <= 32
            and 2 <= self.weight_bits <= 8
            and 2 <= self.fc_bits <= 8
            and 2 <= self.activation_bits <= 16
            and 0 < self.shift < self.bits - 1
            and self.activation_bits + self.shift < self.bits
        ):
            raise ValueError("Require W/FC 2..8, A 2..16, work ring 20..32 and A+shift < work bits")
        return self


def read_arithmetic(folder):
    with (Path(folder) / "meta.bin").open("rb") as stream:
        magic = stream.read(4)
        if magic == b"QIM1":
            return Arithmetic(*struct.unpack("<2I", stream.read(8))).validate()
        if magic == b"QIM2":
            return Arithmetic(*struct.unpack("<5I", stream.read(20))).validate()
        raise ValueError("Invalid quantized bundle header")

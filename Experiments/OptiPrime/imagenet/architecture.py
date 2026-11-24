"""Public ImageNet topology shared by calibration, clear inference and the MPC bundle."""

from collections import Counter
from dataclasses import dataclass


@dataclass
class Step:
    op: str
    src: int
    name: str = ""
    shape: tuple = ()
    signed: bool = False
    skip: int = 0
    projection: bool = False


BLOCKS = {
    "resnet18": (2, 2, 2, 2),
    "resnet34": (3, 4, 6, 3),
    "resnet50": (3, 4, 6, 3),
    "vgg16": (2, 2, 3, 3, 3),
    "vgg19": (2, 2, 4, 4, 4),
}


def topology(arch):
    steps, shapes = [], [(3, 224)]

    def append(step, shape):
        steps.append(step)
        shapes.append(shape)
        return len(steps)

    def conv(src, name, co, kernel, stride=1, pad=0, signed=False):
        ci, h = shapes[src]
        s = (h, ci, co, kernel, stride, pad)
        return append(
            Step("conv", src, name, s, signed), (co, (h + 2 * pad - kernel) // stride + 1)
        )

    if arch.startswith("vgg"):
        x = index = 0
        for repeats, channels in zip(BLOCKS[arch], (64, 128, 256, 512, 512)):
            for _ in range(repeats):
                x = conv(x, f"features.{index}", channels, 3, pad=1)
                index += 2
            x = append(Step("pool2", x), (channels, shapes[x][1] // 2))
            index += 1
        x = append(Step("flatten", x), (512 * 7 * 7, 1))
        for i, channels in [(0, 4096), (3, 4096), (6, 1000)]:
            x = conv(x, f"classifier.{i}", channels, 1)
    else:
        x = conv(0, "conv1", 64, 7, 2, 3)
        x = append(Step("pool3", x), (64, 56))
        bottleneck = arch == "resnet50"
        for i, n in enumerate(BLOCKS[arch]):
            width = 64 << i
            output = width * (4 if bottleneck else 1)
            for j in range(n):
                name = f"layer{i+1}.{j}"
                skip = x
                stride = 2 if i and j == 0 else 1
                project = shapes[x][0] != output or stride != 1
                if bottleneck:
                    x = conv(x, name + ".conv1", width, 1)
                    x = conv(x, name + ".conv2", width, 3, stride, 1)
                    x = conv(x, name + ".conv3", output, 1, signed=True)
                else:
                    x = conv(x, name + ".conv1", width, 3, stride, 1)
                    x = conv(x, name + ".conv2", output, 3, pad=1, signed=True)
                if project:
                    skip = conv(skip, name + ".downsample.0", output, 1, stride, signed=True)
                x = append(Step("add", x, name + ".add", skip=skip, projection=project), shapes[x])
        x = append(Step("mean", x), (shapes[x][0], 1))
        conv(x, "fc", 1000, 1)
    return steps


def forward(steps, image, conv, residual, mean, observe_layout=None):
    import torch.nn.functional as F

    values = {0: image}
    uses = Counter(s.src for s in steps)
    uses.update(s.skip for s in steps if s.op == "add")

    def take(i):
        uses[i] -= 1
        return values.pop(i) if uses[i] == 0 else values[i]

    for i, s in enumerate(steps, 1):
        x = take(s.src)
        if s.op == "conv":
            x = conv(s, x)
        elif s.op == "add":
            x = residual(s, x, take(s.skip))
        elif s.op == "pool3":
            x = F.max_pool2d(x, 3, 2, 1)
        elif s.op == "pool2":
            x = F.max_pool2d(x, 2, 2)
        elif s.op == "flatten":
            x = x.flatten(1)[:, :, None, None]
        elif s.op == "mean":
            x = mean(x)
        if observe_layout is not None and s.op not in ("conv", "add"):
            observe_layout(f"{i}/{s.op}", x)
        values[i] = x
    return x.flatten(1)

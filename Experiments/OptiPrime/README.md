# OptiPrime 实验：量化视觉模型的两方推理

本模块的 CPU/GPU 密文卷积采用 **OptiPrime** 论文第三节提出的 **OptiEncode** 协议。研究中使用该协议时，请引用 Jiangrui Yu 等人的 MICRO 2026 论文。本文档中的精确和近似量化是此 SEC-PPDL 集成提供的两种推理模式。[OptiPrime 论文](https://arxiv.org/abs/2609.16898) · [BibTeX 引用](../../README.md#cite-optiprime)

支持 ImageNet 的 `resnet18`、`resnet34`、`resnet50`、`vgg16` 和 `vgg19`。精确版执行有符号向下截断，再将输出夹到激活范围。近似版执行符号判断、MUX 和已知非负的随机截断，保留超出激活上界的值。两版共用权重、激活尺度和安全 argmax。

## 代码入口

| 用途 | 入口 |
|---|---|
| 查看网络结构 | [architecture.py](imagenet/architecture.py) 中的 `topology()` 定义各阶段、残差连接及分类头。 |
| 运行整网实验 | [evaluate.py](imagenet/evaluate.py) 导出模型，[run.py](run.py) 启动两方推理，[check_secure.py](imagenet/check_secure.py) 核验逐层结果和 logits。 |
| 在 C++ 中调用整网 | [Model.h](include/QuantExperiment/Model.h) 提供 `QuantizedImageNetModel` 接口。[调用示例](#从-c-调用) |
| 单独调用卷积算子 | [ConvOptiEncode.h](../../Layer/LinearLayer/include/LinearLayer/ConvOptiEncode.h) 提供 `LinearLayer::Conv2DOptiEncode` 接口。[C++ 示例](../../README.md#optiencode-convolution) |

以下命令从 SEC-PPDL 仓库根目录执行。

## 编译并导出模型

先按[根目录安装说明](../../README.md)准备 C++ 依赖。导出和明文校验还需要 PyTorch、torchvision、Pillow 和 NumPy。在下方的 `$data` 目录准备 `imagenette2-320/{train,val}`、`imagenetv2-matched-frequency-format-val/{0,...,999}` 和 `imagenet_class_index.json`。校准使用 Imagenette 训练集，公开核验图像来自 ImageNetV2。[Imagenette 320](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz) · [ImageNetV2 matched-frequency](https://huggingface.co/datasets/vaishaal/ImageNetV2) · [ImageNet 类别映射](https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json)

```bash
exp=Experiments/OptiPrime
assets=build/quant-model-research
data=build/real-rn18
bundle=build/rn18-w4a8-k20
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_HE_GPU=OFF -DSEC_PPDL_BUILD_QUANT_EXPERIMENTS=ON
cmake --build build --target test_quant_imagenet -j8
python3 "$exp/model_assets.py" --output "$assets" --weights
python3 "$exp/imagenet/evaluate.py" --model resnet18 --assets "$assets" --data "$data" \
  --output "$bundle" --weight-bits 4 --activation-bits 8 --fc-bits 8 --work-bits 20 \
  --secure-images 2 --export-only
```

ResNet-18 使用 AIMET 已折叠的权重和 AdaRound 编码。其他模型首次导出时通过 torchvision 下载官方 ImageNet-1K V1 权重，再离线量化。更改 W 会重新计算权重尺度。提高已量化检查点的 W 无法恢复先前丢失的精度。要从浮点权重比较不同 W，可在 ResNet-18 导出命令中添加 `--weights-source torchvision`。VGG 保留五次池化和完整的三个分类器线性层，大分类器矩阵使用系数打包点积来控制内存。

| 导出参数 | 范围 | 含义 |
|---|---|---|
| `--weight-bits` | 2–8，默认 4 | 卷积及 VGG 分类器中间层的有符号权重位宽 |
| `--fc-bits` | 2–8，默认 8 | 最后一层 1,000 类分类器的权重位宽 |
| `--activation-bits` | 2–16，默认 8 | 输入和激活量化宽度 A。精确版上界为 `2^A−1` |
| `--work-bits` | 20–32，VGG19 默认 26，其余默认 20 | HE 明文模为 `2^k`，MPC 份额使用 k 位工作环 |
| `--shift` | 默认 9 | 整数倍率的分数位数。要求大于 0 且 `A + shift < k` |

例如 W6A6 使用 `--weight-bits 6 --activation-bits 6 --fc-bits 6 --work-bits 26`。W8A16 可使用 `--weight-bits 8 --activation-bits 16 --work-bits 32`。更换位宽时导出到新目录，两个参与方和明文参考均从同一 bundle 读取参数。省略 `--export-only` 可以在完整评估集上测浮点、精确和近似三组明文准确率。

## 推理并验证

```bash
for mode in exact approx; do
  out="build/rn18-$mode"
  python3 "$exp/run.py" --bundle "$bundle" --mode "$mode" --repeat 2 \
    --threads 16 --trace-count 2 --output "$out"
  python3 "$exp/imagenet/check_secure.py" --assets "$assets" --results "$out" --device cpu
done
```

切换模型时，将导出命令中的 `--model` 改为 `resnet34`、`resnet50`、`vgg16` 或 `vgg19`，选择新的 bundle 目录，并省略 `--work-bits` 以使用各模型的默认值。相同的推理与校验命令可用于这四个模型。

`run.py` 在本机启动两个参与方，默认使用 CPU OptiEncode 和预处理 Ferret。`--ot iknp|ferret` 可选 OT 后端，`--preprocess 0|1` 可切换在线生成与离线预处理。GPU 版在 `build-gpu` 以 `-DUSE_HE_GPU=ON` 编译相同 target，然后传 `--linear gpu`。比较原框架的 CPU 卷积与在线 IKNP 时，使用 `--linear cheetah --mode exact`。自定义构建目录可传 `--binary`。`--qbits` 接受 HE 系数模数链，例如 `50,50`；默认 k=20 使用两个 50 位素数，更宽的工作环使用三个。更改模数后可以通过精确 logits 和近似逐层校验检查噪声余量。

`--repeat` 指定实际使用的图片数，上限是导出时的 `--secure-images`。`accuracy.json` 给出 Top‑1 / Top‑5 命中数，`predictions.jsonl` 保存逐图输出，`summary.json` 保存参数与计时。这个验证程序会在服务端打开公开测试图片的 logits。`--trace-count N` 还会打开前 N 张图片的中间激活。测性能时使用默认值 0。`check_secure.py` 将精确模式与整数明文参考逐项比较。近似模式逐层验证随机截断的允许区间，用实际中间值重算后续层和最终 logits，并核对 argmax。

`summary.json` 按图像记录两方发送字节之和、离线 OT 开销、在线耗时及线性/非线性分项。此命令使用本机回环网络。

## 从 C++ 调用

链接 CMake target `QuantExperiment`，包含 `<QuantExperiment/Bundle.h>`。两方以相同布局和模式执行以下代码，`input_share` 是形状为 `{3,224,224}` 的无符号整数加法份额。模型归属方为 party 1。公开的 `meta.bin` 和 `graph.txt` 可供双方读取，模型归属方还需读取 `weights.bin`。

```cpp
using namespace QuantExperiment;
ImageNetBundle bundle(directory);
auto cfg = bundle.InferenceConfig();
cfg.party = party;
cfg.port = port;
cfg.address = server_address;
cfg.approximate_relu = approximate;
Context ctx(cfg);
auto weights = party == 1 ? bundle.LoadWeights() : Weights{};
QuantizedImageNetModel net(ctx, party == 1 ? &weights : nullptr, bundle.metadata, bundle.steps);
QuantizedImageNetModel::Plan(ctx, bundle.metadata, bundle.steps);  // 同一布局规划一次
ctx.Preprocess();                                                // 每张图片使用新相关随机数
Profile profile;
auto logits = net.Forward(input_share, profile);
auto index_share = Classify(ctx, logits);
ctx.CheckConsumed();
```

`net.Forward` 返回加法分享的 logits。工作环模式下 `Classify` 先对 logits 做有符号扩展，再进行比较，`index_share` 为类别索引的份额。

# OptiPrime experiments for quantized vision inference

Run two-party ImageNet inference with ResNet-18, ResNet-34, ResNet-50, VGG-16, or VGG-19. The CPU and GPU convolution backends implement OptiEncode from Section III of OptiPrime (Jiangrui Yu et al., MICRO 2026). Please cite OptiPrime when using this protocol. [Paper](https://arxiv.org/abs/2609.16898) · [BibTeX](../../README.md#citation)

Both inference modes use the same weights, activation scales, and secure argmax. Exact mode truncates signed values toward negative infinity and clamps the result to the activation range. Approximate mode uses a sign comparison and a multiplexer, then applies stochastic truncation to the nonnegative result. Values can exceed the upper activation limit.

| Task | Entry point |
|---|---|
| Inspect a network | `topology()` defines the stages, residual connections, and classifier in [architecture.py](imagenet/architecture.py). |
| Run an experiment | Export with [evaluate.py](imagenet/evaluate.py), start both parties with [run.py](run.py), and verify activations and logits with [check_secure.py](imagenet/check_secure.py). |
| Use a network from C++ | Use `QuantizedImageNetModel` from [Model.h](include/QuantExperiment/Model.h). [C++ example](#model-api) |
| Use just the convolution | Link `LinearLayer` and call `LinearLayer::Conv2DOptiEncode`. [Public header](../../Layer/LinearLayer/include/LinearLayer/ConvOptiEncode.h) · [Operator example](../../README.md#optiencode-convolution) |

## Build and export a model

Run the commands below from the repository root after installing the [C++ dependencies](../../Extern/README.md). Model export and cleartext verification also require PyTorch, torchvision, Pillow, and NumPy.

Place `imagenette2-320/{train,val}`, `imagenetv2-matched-frequency-format-val/{0,...,999}`, and `imagenet_class_index.json` under the data directory assigned to `$data` below. Calibration uses the Imagenette training split. The exported public test images come from ImageNetV2. [Imagenette 320](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz) · [ImageNetV2 matched-frequency](https://huggingface.co/datasets/vaishaal/ImageNetV2) · [ImageNet class index](https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json)

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

ResNet-18 defaults to AIMET's batch-normalization-folded weights and AdaRound weight encodings. The other models use torchvision's ImageNet-1K V1 weights and quantize them at export. VGG retains all five pooling stages and all three fully connected classifier layers. The large classifier matrices use coefficient-packed dot products to limit memory use.

| Export option | Range and default | Meaning |
|---|---|---|
| `--weight-bits` | 2–8, default 4 | Signed weight width for convolutions and the intermediate VGG classifier layers |
| `--fc-bits` | 2–8, default 8 | Weight width for the final 1,000-class layer |
| `--activation-bits` | 2–16, default 8 | Input and activation quantizer width A. Exact mode clamps to `[0, 2^A − 1]`. |
| `--work-bits` | 20–32, default 26 for VGG-19 and 20 for the others | HE plaintext modulus `2^k` and k-bit MPC share ring |
| `--shift` | Default 9 | Fractional bits for the integer multiplier. Requires `shift > 0` and `A + shift < k`. |

Changing the weight width recomputes the weight scales. For comparisons starting from floating-point ResNet-18 weights, select `--weights-source torchvision`. Increasing the width of an already quantized checkpoint cannot recover precision lost during its original quantization.

For W6A6, use `--weight-bits 6 --activation-bits 6 --fc-bits 6 --work-bits 26`. For W8A16, use `--weight-bits 8 --activation-bits 16 --work-bits 32`. Export each configuration to a separate bundle so that both parties and the cleartext reference read the same parameters. Omit `--export-only` to measure floating-point, exact integer, and approximate integer accuracy on the full evaluation splits.

## Run and verify inference

```bash
for mode in exact approx; do
  out="build/rn18-$mode"
  python3 "$exp/run.py" --bundle "$bundle" --mode "$mode" --repeat 2 \
    --threads 16 --trace-count 2 --output "$out"
  python3 "$exp/imagenet/check_secure.py" --assets "$assets" --results "$out" --device cpu
done
```

To switch networks, select `resnet34`, `resnet50`, `vgg16`, or `vgg19` with `--model` at export, choose a new bundle directory, and omit `--work-bits` to use the model default. Run the same inference and verification commands on the new bundle.

- **Backend.** The runner starts both parties on loopback. It defaults to CPU OptiEncode with preprocessed Ferret OT. Select the OT backend with `--ot iknp|ferret` and toggle preprocessing with `--preprocess 0|1`. For GPU inference, build the same target in `build-gpu` with `-DUSE_HE_GPU=ON` and pass `--linear gpu`. Use `--binary` for a custom build directory. To run the Cheetah CPU baseline with online IKNP, use `--linear cheetah --mode exact`.
- **HE parameters.** `--qbits` sets the coefficient-modulus prime widths. The default is `50,50` at k=20 and `50,50,50` for wider work rings. After changing the chain, verify exact logits and approximate layer traces to check numerical correctness on the chosen inputs.
- **Correctness.** `--repeat` selects up to the number of cases exported with `--secure-images`. The verification harness opens the public test logits at the server. `--trace-count N` also opens intermediate activations for the first N images. In exact mode, the checker compares against the integer reference. In approximate mode, it checks each layer's rounding bounds, replays the observed values through the classifier, and checks the secure argmax against the logits.
- **Outputs and timing.** `predictions.jsonl` contains per-image predictions. The checker writes Top-1 and Top-5 hit counts to `accuracy.json`. For performance measurements, use the default `--trace-count 0`. Per image, `summary.json` records offline OT cost, online latency, and the linear/nonlinear breakdown. Communication totals sum bytes sent by both parties.

## Model API

Link the CMake target `QuantExperiment` and include `<QuantExperiment/Bundle.h>`. Both parties use the same public `meta.bin`, `graph.txt`, and inference mode. Party 1 owns the model and reads `weights.bin`. Each party passes its additive share of the quantized input as a `{3,224,224}` unsigned integer tensor.

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
QuantizedImageNetModel::Plan(ctx, bundle.metadata, bundle.steps);  // Plan once per graph.
ctx.Preprocess();                                                // Prepare fresh OT correlations for each image.
Profile profile;
auto logits = net.Forward(input_share, profile);
auto index_share = Classify(ctx, logits);
ctx.CheckConsumed();
```

`Forward` returns additive shares of the logits. In the power-of-two work ring, `Classify` sign-extends the logits before comparison and returns an additive share of the winning class index.

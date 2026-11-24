#include <QuantExperiment/Bundle.h>
#include <fstream>

namespace QuantExperiment {
using Datatype::Tensor;
namespace fs = std::filesystem;
namespace {
template <typename T> std::vector<T> Read(std::istream &in, size_t n) {
    std::vector<T> v(n);
    if (!in.read(reinterpret_cast<char *>(v.data()), n * sizeof(T)))
        throw std::runtime_error("Incomplete model or test vector");
    return v;
}
void End(std::istream &in) {
    if (in.peek() != std::istream::traits_type::eof())
        throw std::runtime_error("Unexpected trailing bundle data");
}
// QIM1 predates graph.txt and uses the fixed ImageNet ResNet-18 layout.
void ReadLegacyLayout(std::vector<LayerMetadata> &layers, std::vector<Step> &steps) {
    auto conv = [&](size_t input, LayerShape shape, int input_zero = 0, int output_zero = 0) {
        LayerMetadata layer{};
        layer.shape = shape;
        layer.input_zero_point = input_zero;
        layer.output_zero_point = output_zero;
        layers.push_back(layer);
        steps.push_back({Operation::Conv, input});
        return steps.size();
    };
    conv(0, {224, 3, 64, 7, 2, 3}, 128);
    steps.push_back({Operation::MaxPool3, 1});
    size_t size = 56, channels = 64;
    for (size_t outputs : {64, 128, 256, 512}) {
        for (size_t block = 0; block < 2; ++block) {
            size_t skip = steps.size();
            const bool projection = channels != outputs;
            const size_t stride = projection ? 2 : 1;
            auto first = conv(skip, {size, channels, outputs, 3, stride, 1});
            auto branch = conv(first, {size / stride, outputs, outputs, 3, 1, 1}, 0, 128);
            if (projection)
                skip = conv(skip, {size, channels, outputs, 1, stride, 0}, 0, 128);
            steps.push_back({Operation::Residual, branch, skip, projection});
            size /= stride;
            channels = outputs;
        }
    }
    steps.push_back({Operation::GlobalAverage, steps.size()});
    conv(steps.size(), {1, 512, 1000, 1, 1, 0});
}
void ReadLayout(const fs::path &root, std::vector<LayerMetadata> &layers,
                std::vector<Step> &steps) {
    std::ifstream in(root / "graph.txt");
    if (!in)
        throw std::runtime_error("Missing public graph layout");
    for (std::string op; in >> op;) {
        Step s{};
        in >> s.input;
        if (op == "conv") {
            s.op = Operation::Conv;
            LayerMetadata m;
            auto &a = m.shape;
            in >> a.size >> a.input_channels >> a.output_channels >> a.kernel >> a.stride >>
                a.padding >> m.input_zero_point >> m.output_zero_point;
            layers.push_back(m);
        } else if (op == "add") {
            s.op = Operation::Residual;
            in >> s.skip >> s.projection;
        } else if (op == "pool2")
            s.op = Operation::MaxPool2;
        else if (op == "pool3")
            s.op = Operation::MaxPool3;
        else if (op == "mean")
            s.op = Operation::GlobalAverage;
        else if (op == "flatten")
            s.op = Operation::Flatten;
        else
            throw std::runtime_error("Unknown model operation: " + op);
        if (!in || s.input > steps.size() || s.skip > steps.size())
            throw std::runtime_error("Invalid model graph");
        steps.push_back(s);
    }
    if (layers.empty())
        throw std::runtime_error("Empty model graph");
}
} // namespace
ImageNetBundle::ImageNetBundle(const fs::path &root) : root_(root) {
    std::ifstream meta(root / "meta.bin", std::ios::binary);
    const auto magic = Read<char>(meta, 4);
    const std::string version(magic.begin(), magic.end());
    if (version != "QIM1" && version != "QIM2")
        throw std::runtime_error("Unknown model format");
    const auto header = Read<uint32_t>(meta, 2);
    if (header[0] < 20 || header[0] > 32 || !header[1] || header[1] >= header[0] - 1)
        throw std::runtime_error("Unsupported work ring or shift");
    uint32_t weight_bits = 4, fc_bits = 8;
    if (version == "QIM2") {
        const auto widths = Read<uint32_t>(meta, 3);
        weight_bits = widths[0];
        activation_bits_ = widths[1];
        fc_bits = widths[2];
        if (weight_bits < 2 || weight_bits > 8 || fc_bits < 2 || fc_bits > 8 || widths[1] < 2 ||
            widths[1] > 16 || widths[1] + header[1] >= header[0])
            throw std::runtime_error("Incompatible model bit widths");
        ReadLayout(root, metadata, steps);
    } else {
        ReadLegacyLayout(metadata, steps);
    }
    for (auto &m : metadata) {
        const auto &s = m.shape;
        if (&m == &metadata.back())
            m.weight_bits = fc_bits;
        else {
            m.weight_bits = weight_bits;
            m.shift = header[1];
            m.clamp_bits = header[0];
            m.multipliers = Read<uint32_t>(meta, s.output_channels);
        }
    }
    End(meta);
    work_bits_ = header[0];
}
Weights ImageNetBundle::LoadWeights() const {
    Weights weights;
    std::ifstream weight_file(root_ / "weights.bin", std::ios::binary);
    for (const auto &m : metadata) {
        const auto &s = m.shape;
        Parameters p{Tensor<uint64_t>({s.output_channels, s.input_channels, s.kernel, s.kernel}),
                     Tensor<uint64_t>({s.output_channels})};
        auto w = Read<int8_t>(weight_file, p.weight.size());
        auto b = Read<int32_t>(weight_file, p.bias.size());
        std::copy(w.begin(), w.end(), p.weight.data().begin());
        std::copy(b.begin(), b.end(), p.bias.data().begin());
        weights.push_back(std::move(p));
    }
    End(weight_file);
    return weights;
}
Config ImageNetBundle::InferenceConfig() const {
    Config cfg;
    cfg.linear = LinearBackend::CPU;
    cfg.activation_bits = activation_bits_;
    cfg.activation_ring_bits = cfg.accumulator_bits = work_bits_;
    cfg.plaintext_bits = work_bits_ + 1;
    cfg.power_of_two = cfg.ciphertext_inverse = cfg.empirical_range = true;
    cfg.preprocess = true;
    cfg.ot = OTBackend::Ferret;
    cfg.ciphertext_prime_bits = {50, 50};
    if (work_bits_ > 20)
        cfg.ciphertext_prime_bits.push_back(50);
    return cfg;
}
} // namespace QuantExperiment

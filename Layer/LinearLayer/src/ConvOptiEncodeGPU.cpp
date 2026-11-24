#include <LinearLayer/ConvOptiEncode.h>
#ifdef USE_HE_GPU
#include <phantom/evaluate.cuh>
#include <chrono>

namespace LinearLayer {
void Conv2DOptiEncode::ApplyAutomorphismGPU(UnifiedCiphertext &ciphertext, uint32_t element) const {
    if (element == 1) return;
    if (HE->galoisKeys->hgalois().has_key(element)) {
        phantom::apply_galois_inplace(HE->context->dcontext(), ciphertext.dcipher(), element,
                                     HE->galoisKeys->dgalois());
    } else {
        const auto step = rotation_steps_.find(element);
        if (step == rotation_steps_.end()) throw std::invalid_argument("Missing OptiEncode GPU Galois key");
        HE->evaluator->rotate_rows_inplace(ciphertext, step->second, *HE->galoisKeys);
    }
}

Tensor<UnifiedCiphertext> Conv2DOptiEncode::HEComputeGPU(const Tensor<UnifiedPlaintext> &weights,
                                                      Tensor<UnifiedCiphertext> &inputs) {
    Tensor<UnifiedCiphertext> output({layout_.TileCount(), layout_.output_blocks}, HE->server ? DEVICE : HOST);
    if (!HE->server) return output;
    HE->evaluator->sync();
    const auto start = std::chrono::steady_clock::now();
    auto &evaluator = *HE->evaluator;
    const uint32_t giant_element = static_cast<uint32_t>(Operator::OptiEncodePowerMod(
        layout_.generator, layout_.baby_steps, 2 * layout_.degree));
    for (size_t tile = 0; tile < layout_.TileCount(); ++tile) {
        std::vector<UnifiedCiphertext> babies(layout_.input_blocks * layout_.baby_steps);
        for (size_t i = 0; i < layout_.input_blocks; ++i) {
            auto current = inputs({tile, i});
            for (size_t a = 0; a < layout_.baby_steps; ++a) {
                if (a) ApplyAutomorphismGPU(current, layout_.generator);
                auto &baby = babies[i * layout_.baby_steps + a];
                baby = current;
                evaluator.transform_to_ntt_inplace(baby);
            }
        }
        for (size_t o = 0; o < layout_.output_blocks; ++o) {
            UnifiedCiphertext result;
            bool have_result = false;
            for (size_t giant = layout_.giant_steps; giant-- > 0;) {
                if (have_result) ApplyAutomorphismGPU(result, giant_element);
                UnifiedCiphertext subtotal;
                bool have_subtotal = false;
                for (size_t i = 0; i < layout_.input_blocks; ++i)
                    for (size_t a = 0; a < layout_.baby_steps; ++a) {
                        const size_t index = (i * layout_.output_blocks + o) * layout_.channels + giant * layout_.baby_steps + a;
                        if (!nonzero_weights_[index]) continue;
                        UnifiedCiphertext product;
                        evaluator.multiply_plain_ntt(babies[i * layout_.baby_steps + a], weights(index), product);
                        if (have_subtotal) evaluator.add_inplace(subtotal, product);
                        else { subtotal = std::move(product); have_subtotal = true; }
                    }
                if (have_subtotal) {
                    evaluator.transform_from_ntt_inplace(subtotal);
                    if (have_result) evaluator.add_inplace(result, subtotal);
                    else { result = std::move(subtotal); have_result = true; }
                }
            }
            if (!have_result) result = HE->GenerateZeroCiphertext(DEVICE);
            for (size_t level = 0; level < options_.output_mod_switches; ++level) {
                const size_t limbs = result.coeff_modulus_size();
                // SEAL's valid chain can stop before the smallest Phantom level.
                auto data = HE->context->hcontext().first_context_data();
                while (data && data->parms().coeff_modulus().size() != limbs) data = data->next_context_data();
                if (!data || !data->next_context_data()) break;
                evaluator.mod_switch_to_next_inplace(result);
            }
            output({tile, o}) = std::move(result);
        }
    }
    // Include kernel completion, not merely asynchronous launch time.
    evaluator.sync();
    stats_.compute_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return output;
}
} // namespace LinearLayer
#endif

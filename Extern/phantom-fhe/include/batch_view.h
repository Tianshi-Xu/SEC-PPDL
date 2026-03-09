#pragma once

#include <cstddef>
#include <stdexcept>

#include <cuda_runtime.h>

#include "ciphertext.h"
#include "context.cuh"
#include "cuda_wrapper.cuh"
#include "plaintext.h"

class PhantomCiphertextView {
public:
    PhantomCiphertextView() = default;

    explicit PhantomCiphertextView(PhantomCiphertext &ciphertext)
        : data_(ciphertext.data()), chain_index_(ciphertext.chain_index()),
          size_(ciphertext.size()),
          poly_modulus_degree_(ciphertext.poly_modulus_degree()),
          coeff_modulus_size_(ciphertext.coeff_modulus_size()),
          scale_(ciphertext.scale()),
          correction_factor_(ciphertext.correction_factor()),
          noiseScaleDeg_(ciphertext.GetNoiseScaleDeg()),
          is_ntt_form_(ciphertext.is_ntt_form()),
          is_asymmetric_(ciphertext.is_asymmetric()) {}

    PhantomCiphertextView(
        uint64_t *data, std::size_t chain_index, std::size_t size,
        std::size_t poly_modulus_degree, std::size_t coeff_modulus_size,
        double scale, uint64_t correction_factor, std::size_t noiseScaleDeg,
        bool is_ntt_form, bool is_asymmetric)
        : data_(data), chain_index_(chain_index), size_(size),
          poly_modulus_degree_(poly_modulus_degree),
          coeff_modulus_size_(coeff_modulus_size), scale_(scale),
          correction_factor_(correction_factor), noiseScaleDeg_(noiseScaleDeg),
          is_ntt_form_(is_ntt_form), is_asymmetric_(is_asymmetric) {}

    uint64_t *data() const {
        if (data_ == nullptr) {
            throw std::runtime_error("PhantomCiphertextView: null data");
        }
        return data_;
    }

    std::size_t chain_index() const { return chain_index_; }
    std::size_t size() const { return size_; }
    std::size_t poly_modulus_degree() const { return poly_modulus_degree_; }
    std::size_t coeff_modulus_size() const { return coeff_modulus_size_; }
    std::size_t coeff_count() const {
        return poly_modulus_degree_ * coeff_modulus_size_;
    }
    std::size_t data_count() const { return size_ * coeff_count(); }
    bool is_ntt_form() const { return is_ntt_form_; }
    bool is_asymmetric() const { return is_asymmetric_; }
    double scale() const { return scale_; }
    uint64_t correction_factor() const { return correction_factor_; }
    std::size_t noiseScaleDeg() const { return noiseScaleDeg_; }

    void set_scale(double scale) { scale_ = scale; }
    void set_ntt_form(bool is_ntt_form) { is_ntt_form_ = is_ntt_form; }
    void set_correction_factor(uint64_t correction_factor) {
        correction_factor_ = correction_factor;
    }
    void set_noiseScaleDeg(std::size_t noiseScaleDeg) {
        noiseScaleDeg_ = noiseScaleDeg;
    }

private:
    uint64_t *data_ = nullptr;
    std::size_t chain_index_ = 0;
    std::size_t size_ = 0;
    std::size_t poly_modulus_degree_ = 0;
    std::size_t coeff_modulus_size_ = 0;
    double scale_ = 1.0;
    uint64_t correction_factor_ = 1;
    std::size_t noiseScaleDeg_ = 1;
    bool is_ntt_form_ = true;
    bool is_asymmetric_ = false;
};

class ConstPhantomCiphertextView {
public:
    ConstPhantomCiphertextView() = default;

    explicit ConstPhantomCiphertextView(const PhantomCiphertext &ciphertext)
        : data_(ciphertext.data()), chain_index_(ciphertext.chain_index()),
          size_(ciphertext.size()),
          poly_modulus_degree_(ciphertext.poly_modulus_degree()),
          coeff_modulus_size_(ciphertext.coeff_modulus_size()),
          scale_(ciphertext.scale()),
          correction_factor_(ciphertext.correction_factor()),
          noiseScaleDeg_(ciphertext.GetNoiseScaleDeg()),
          is_ntt_form_(ciphertext.is_ntt_form()),
          is_asymmetric_(ciphertext.is_asymmetric()) {}

    explicit ConstPhantomCiphertextView(const PhantomCiphertextView &view)
        : data_(view.data()), chain_index_(view.chain_index()),
          size_(view.size()), poly_modulus_degree_(view.poly_modulus_degree()),
          coeff_modulus_size_(view.coeff_modulus_size()),
          scale_(view.scale()), correction_factor_(view.correction_factor()),
          noiseScaleDeg_(view.noiseScaleDeg()), is_ntt_form_(view.is_ntt_form()),
          is_asymmetric_(view.is_asymmetric()) {}

    ConstPhantomCiphertextView(
        const uint64_t *data, std::size_t chain_index, std::size_t size,
        std::size_t poly_modulus_degree, std::size_t coeff_modulus_size,
        double scale, uint64_t correction_factor, std::size_t noiseScaleDeg,
        bool is_ntt_form, bool is_asymmetric)
        : data_(data), chain_index_(chain_index), size_(size),
          poly_modulus_degree_(poly_modulus_degree),
          coeff_modulus_size_(coeff_modulus_size), scale_(scale),
          correction_factor_(correction_factor), noiseScaleDeg_(noiseScaleDeg),
          is_ntt_form_(is_ntt_form), is_asymmetric_(is_asymmetric) {}

    const uint64_t *data() const {
        if (data_ == nullptr) {
            throw std::runtime_error("ConstPhantomCiphertextView: null data");
        }
        return data_;
    }

    std::size_t chain_index() const { return chain_index_; }
    std::size_t size() const { return size_; }
    std::size_t poly_modulus_degree() const { return poly_modulus_degree_; }
    std::size_t coeff_modulus_size() const { return coeff_modulus_size_; }
    std::size_t coeff_count() const {
        return poly_modulus_degree_ * coeff_modulus_size_;
    }
    std::size_t data_count() const { return size_ * coeff_count(); }
    bool is_ntt_form() const { return is_ntt_form_; }
    bool is_asymmetric() const { return is_asymmetric_; }
    double scale() const { return scale_; }
    uint64_t correction_factor() const { return correction_factor_; }
    std::size_t noiseScaleDeg() const { return noiseScaleDeg_; }

private:
    const uint64_t *data_ = nullptr;
    std::size_t chain_index_ = 0;
    std::size_t size_ = 0;
    std::size_t poly_modulus_degree_ = 0;
    std::size_t coeff_modulus_size_ = 0;
    double scale_ = 1.0;
    uint64_t correction_factor_ = 1;
    std::size_t noiseScaleDeg_ = 1;
    bool is_ntt_form_ = true;
    bool is_asymmetric_ = false;
};

class PhantomPlaintextView {
public:
    PhantomPlaintextView() = default;

    explicit PhantomPlaintextView(PhantomPlaintext &plaintext)
        : data_(plaintext.data()), chain_index_(plaintext.chain_index()),
          poly_modulus_degree_(plaintext.coeff_count() / plaintext.coeff_modulus_size()),
          coeff_modulus_size_(plaintext.coeff_modulus_size()),
          scale_(plaintext.scale()), is_ntt_form_(plaintext.is_ntt_form()) {}

    PhantomPlaintextView(
        uint64_t *data, std::size_t chain_index, std::size_t poly_modulus_degree,
        std::size_t coeff_modulus_size, double scale, bool is_ntt_form)
        : data_(data), chain_index_(chain_index),
          poly_modulus_degree_(poly_modulus_degree),
          coeff_modulus_size_(coeff_modulus_size), scale_(scale),
          is_ntt_form_(is_ntt_form) {}

    uint64_t *data() const {
        if (data_ == nullptr) {
            throw std::runtime_error("PhantomPlaintextView: null data");
        }
        return data_;
    }

    std::size_t chain_index() const { return chain_index_; }
    std::size_t poly_modulus_degree() const { return poly_modulus_degree_; }
    std::size_t coeff_modulus_size() const { return coeff_modulus_size_; }
    std::size_t coeff_count() const {
        return poly_modulus_degree_ * coeff_modulus_size_;
    }
    bool is_ntt_form() const { return is_ntt_form_; }
    double scale() const { return scale_; }

    void set_scale(double scale) { scale_ = scale; }
    void set_ntt_form(bool is_ntt_form) { is_ntt_form_ = is_ntt_form; }

private:
    uint64_t *data_ = nullptr;
    std::size_t chain_index_ = 0;
    std::size_t poly_modulus_degree_ = 0;
    std::size_t coeff_modulus_size_ = 0;
    double scale_ = 1.0;
    bool is_ntt_form_ = false;
};

class ConstPhantomPlaintextView {
public:
    ConstPhantomPlaintextView() = default;

    explicit ConstPhantomPlaintextView(const PhantomPlaintext &plaintext)
        : data_(plaintext.data()), chain_index_(plaintext.chain_index()),
          poly_modulus_degree_(plaintext.coeff_count() / plaintext.coeff_modulus_size()),
          coeff_modulus_size_(plaintext.coeff_modulus_size()),
          scale_(plaintext.scale()), is_ntt_form_(plaintext.is_ntt_form()) {}

    explicit ConstPhantomPlaintextView(const PhantomPlaintextView &view)
        : data_(view.data()), chain_index_(view.chain_index()),
          poly_modulus_degree_(view.poly_modulus_degree()),
          coeff_modulus_size_(view.coeff_modulus_size()), scale_(view.scale()),
          is_ntt_form_(view.is_ntt_form()) {}

    ConstPhantomPlaintextView(
        const uint64_t *data, std::size_t chain_index,
        std::size_t poly_modulus_degree, std::size_t coeff_modulus_size,
        double scale, bool is_ntt_form)
        : data_(data), chain_index_(chain_index),
          poly_modulus_degree_(poly_modulus_degree),
          coeff_modulus_size_(coeff_modulus_size), scale_(scale),
          is_ntt_form_(is_ntt_form) {}

    const uint64_t *data() const {
        if (data_ == nullptr) {
            throw std::runtime_error("ConstPhantomPlaintextView: null data");
        }
        return data_;
    }

    std::size_t chain_index() const { return chain_index_; }
    std::size_t poly_modulus_degree() const { return poly_modulus_degree_; }
    std::size_t coeff_modulus_size() const { return coeff_modulus_size_; }
    std::size_t coeff_count() const {
        return poly_modulus_degree_ * coeff_modulus_size_;
    }
    bool is_ntt_form() const { return is_ntt_form_; }
    double scale() const { return scale_; }

private:
    const uint64_t *data_ = nullptr;
    std::size_t chain_index_ = 0;
    std::size_t poly_modulus_degree_ = 0;
    std::size_t coeff_modulus_size_ = 0;
    double scale_ = 1.0;
    bool is_ntt_form_ = false;
};

class BatchCipherView {
public:
    BatchCipherView() = default;

    BatchCipherView(
        uint64_t *data, std::size_t batch_size, std::size_t chain_index,
        std::size_t size, std::size_t poly_modulus_degree,
        std::size_t coeff_modulus_size, double scale, uint64_t correction_factor,
        std::size_t noiseScaleDeg, bool is_ntt_form, bool is_asymmetric)
        : data_(data), batch_size_(batch_size), chain_index_(chain_index),
          size_(size), poly_modulus_degree_(poly_modulus_degree),
          coeff_modulus_size_(coeff_modulus_size), scale_(scale),
          correction_factor_(correction_factor), noiseScaleDeg_(noiseScaleDeg),
          is_ntt_form_(is_ntt_form), is_asymmetric_(is_asymmetric) {}

    uint64_t *data() const {
        if (data_ == nullptr) {
            throw std::runtime_error("BatchCipherView: null data");
        }
        return data_;
    }

    std::size_t batch_size() const { return batch_size_; }
    std::size_t chain_index() const { return chain_index_; }
    std::size_t size() const { return size_; }
    std::size_t poly_modulus_degree() const { return poly_modulus_degree_; }
    std::size_t coeff_modulus_size() const { return coeff_modulus_size_; }
    std::size_t coeff_count() const {
        return poly_modulus_degree_ * coeff_modulus_size_;
    }
    std::size_t item_data_count() const { return size_ * coeff_count(); }
    std::size_t total_data_count() const { return batch_size_ * item_data_count(); }
    bool is_ntt_form() const { return is_ntt_form_; }
    bool is_asymmetric() const { return is_asymmetric_; }
    double scale() const { return scale_; }
    uint64_t correction_factor() const { return correction_factor_; }
    std::size_t noiseScaleDeg() const { return noiseScaleDeg_; }

    PhantomCiphertextView operator[](std::size_t index) const {
        if (index >= batch_size_) {
            throw std::out_of_range("BatchCipherView index out of range");
        }
        return PhantomCiphertextView(
            data() + index * item_data_count(), chain_index_, size_,
            poly_modulus_degree_, coeff_modulus_size_, scale_,
            correction_factor_, noiseScaleDeg_, is_ntt_form_, is_asymmetric_);
    }

    void set_scale(double scale) { scale_ = scale; }
    void set_ntt_form(bool is_ntt_form) { is_ntt_form_ = is_ntt_form; }
    void set_correction_factor(uint64_t correction_factor) {
        correction_factor_ = correction_factor;
    }
    void set_noiseScaleDeg(std::size_t noiseScaleDeg) {
        noiseScaleDeg_ = noiseScaleDeg;
    }

private:
    uint64_t *data_ = nullptr;
    std::size_t batch_size_ = 0;
    std::size_t chain_index_ = 0;
    std::size_t size_ = 0;
    std::size_t poly_modulus_degree_ = 0;
    std::size_t coeff_modulus_size_ = 0;
    double scale_ = 1.0;
    uint64_t correction_factor_ = 1;
    std::size_t noiseScaleDeg_ = 1;
    bool is_ntt_form_ = true;
    bool is_asymmetric_ = false;
};

class ConstBatchCipherView {
public:
    ConstBatchCipherView() = default;

    explicit ConstBatchCipherView(const BatchCipherView &view)
        : data_(view.data()), batch_size_(view.batch_size()),
          chain_index_(view.chain_index()), size_(view.size()),
          poly_modulus_degree_(view.poly_modulus_degree()),
          coeff_modulus_size_(view.coeff_modulus_size()), scale_(view.scale()),
          correction_factor_(view.correction_factor()),
          noiseScaleDeg_(view.noiseScaleDeg()),
          is_ntt_form_(view.is_ntt_form()),
          is_asymmetric_(view.is_asymmetric()) {}

    ConstBatchCipherView(
        const uint64_t *data, std::size_t batch_size, std::size_t chain_index,
        std::size_t size, std::size_t poly_modulus_degree,
        std::size_t coeff_modulus_size, double scale, uint64_t correction_factor,
        std::size_t noiseScaleDeg, bool is_ntt_form, bool is_asymmetric)
        : data_(data), batch_size_(batch_size), chain_index_(chain_index),
          size_(size), poly_modulus_degree_(poly_modulus_degree),
          coeff_modulus_size_(coeff_modulus_size), scale_(scale),
          correction_factor_(correction_factor), noiseScaleDeg_(noiseScaleDeg),
          is_ntt_form_(is_ntt_form), is_asymmetric_(is_asymmetric) {}

    const uint64_t *data() const {
        if (data_ == nullptr) {
            throw std::runtime_error("ConstBatchCipherView: null data");
        }
        return data_;
    }

    std::size_t batch_size() const { return batch_size_; }
    std::size_t chain_index() const { return chain_index_; }
    std::size_t size() const { return size_; }
    std::size_t poly_modulus_degree() const { return poly_modulus_degree_; }
    std::size_t coeff_modulus_size() const { return coeff_modulus_size_; }
    std::size_t coeff_count() const {
        return poly_modulus_degree_ * coeff_modulus_size_;
    }
    std::size_t item_data_count() const { return size_ * coeff_count(); }
    std::size_t total_data_count() const { return batch_size_ * item_data_count(); }
    bool is_ntt_form() const { return is_ntt_form_; }
    bool is_asymmetric() const { return is_asymmetric_; }
    double scale() const { return scale_; }
    uint64_t correction_factor() const { return correction_factor_; }
    std::size_t noiseScaleDeg() const { return noiseScaleDeg_; }

    ConstPhantomCiphertextView operator[](std::size_t index) const {
        if (index >= batch_size_) {
            throw std::out_of_range("ConstBatchCipherView index out of range");
        }
        return ConstPhantomCiphertextView(
            data() + index * item_data_count(), chain_index_, size_,
            poly_modulus_degree_, coeff_modulus_size_, scale_,
            correction_factor_, noiseScaleDeg_, is_ntt_form_, is_asymmetric_);
    }

private:
    const uint64_t *data_ = nullptr;
    std::size_t batch_size_ = 0;
    std::size_t chain_index_ = 0;
    std::size_t size_ = 0;
    std::size_t poly_modulus_degree_ = 0;
    std::size_t coeff_modulus_size_ = 0;
    double scale_ = 1.0;
    uint64_t correction_factor_ = 1;
    std::size_t noiseScaleDeg_ = 1;
    bool is_ntt_form_ = true;
    bool is_asymmetric_ = false;
};

class BatchPlaintextView {
public:
    BatchPlaintextView() = default;

    BatchPlaintextView(
        uint64_t *data, std::size_t batch_size, std::size_t chain_index,
        std::size_t poly_modulus_degree, std::size_t coeff_modulus_size,
        double scale, bool is_ntt_form)
        : data_(data), batch_size_(batch_size), chain_index_(chain_index),
          poly_modulus_degree_(poly_modulus_degree),
          coeff_modulus_size_(coeff_modulus_size), scale_(scale),
          is_ntt_form_(is_ntt_form) {}

    uint64_t *data() const {
        if (data_ == nullptr) {
            throw std::runtime_error("BatchPlaintextView: null data");
        }
        return data_;
    }

    std::size_t batch_size() const { return batch_size_; }
    std::size_t chain_index() const { return chain_index_; }
    std::size_t poly_modulus_degree() const { return poly_modulus_degree_; }
    std::size_t coeff_modulus_size() const { return coeff_modulus_size_; }
    std::size_t coeff_count() const {
        return poly_modulus_degree_ * coeff_modulus_size_;
    }
    std::size_t item_data_count() const { return coeff_count(); }
    std::size_t total_data_count() const { return batch_size_ * coeff_count(); }
    bool is_ntt_form() const { return is_ntt_form_; }
    double scale() const { return scale_; }

    PhantomPlaintextView operator[](std::size_t index) const {
        if (index >= batch_size_) {
            throw std::out_of_range("BatchPlaintextView index out of range");
        }
        return PhantomPlaintextView(
            data() + index * coeff_count(), chain_index_, poly_modulus_degree_,
            coeff_modulus_size_, scale_, is_ntt_form_);
    }

    void set_scale(double scale) { scale_ = scale; }
    void set_ntt_form(bool is_ntt_form) { is_ntt_form_ = is_ntt_form; }

private:
    uint64_t *data_ = nullptr;
    std::size_t batch_size_ = 0;
    std::size_t chain_index_ = 0;
    std::size_t poly_modulus_degree_ = 0;
    std::size_t coeff_modulus_size_ = 0;
    double scale_ = 1.0;
    bool is_ntt_form_ = false;
};

class ConstBatchPlaintextView {
public:
    ConstBatchPlaintextView() = default;

    explicit ConstBatchPlaintextView(const BatchPlaintextView &view)
        : data_(view.data()), batch_size_(view.batch_size()),
          chain_index_(view.chain_index()),
          poly_modulus_degree_(view.poly_modulus_degree()),
          coeff_modulus_size_(view.coeff_modulus_size()),
          scale_(view.scale()), is_ntt_form_(view.is_ntt_form()) {}

    ConstBatchPlaintextView(
        const uint64_t *data, std::size_t batch_size, std::size_t chain_index,
        std::size_t poly_modulus_degree, std::size_t coeff_modulus_size,
        double scale, bool is_ntt_form)
        : data_(data), batch_size_(batch_size), chain_index_(chain_index),
          poly_modulus_degree_(poly_modulus_degree),
          coeff_modulus_size_(coeff_modulus_size), scale_(scale),
          is_ntt_form_(is_ntt_form) {}

    const uint64_t *data() const {
        if (data_ == nullptr) {
            throw std::runtime_error("ConstBatchPlaintextView: null data");
        }
        return data_;
    }

    std::size_t batch_size() const { return batch_size_; }
    std::size_t chain_index() const { return chain_index_; }
    std::size_t poly_modulus_degree() const { return poly_modulus_degree_; }
    std::size_t coeff_modulus_size() const { return coeff_modulus_size_; }
    std::size_t coeff_count() const {
        return poly_modulus_degree_ * coeff_modulus_size_;
    }
    std::size_t item_data_count() const { return coeff_count(); }
    std::size_t total_data_count() const { return batch_size_ * coeff_count(); }
    bool is_ntt_form() const { return is_ntt_form_; }
    double scale() const { return scale_; }

    ConstPhantomPlaintextView operator[](std::size_t index) const {
        if (index >= batch_size_) {
            throw std::out_of_range("ConstBatchPlaintextView index out of range");
        }
        return ConstPhantomPlaintextView(
            data() + index * coeff_count(), chain_index_, poly_modulus_degree_,
            coeff_modulus_size_, scale_, is_ntt_form_);
    }

private:
    const uint64_t *data_ = nullptr;
    std::size_t batch_size_ = 0;
    std::size_t chain_index_ = 0;
    std::size_t poly_modulus_degree_ = 0;
    std::size_t coeff_modulus_size_ = 0;
    double scale_ = 1.0;
    bool is_ntt_form_ = false;
};

class PhantomBatchCiphertext {
public:
    PhantomBatchCiphertext() = default;
    PhantomBatchCiphertext(const PhantomBatchCiphertext &) = default;
    PhantomBatchCiphertext &operator=(const PhantomBatchCiphertext &) = default;
    PhantomBatchCiphertext(PhantomBatchCiphertext &&) = default;
    PhantomBatchCiphertext &operator=(PhantomBatchCiphertext &&) = default;
    ~PhantomBatchCiphertext() = default;

    void resize(
        const PhantomContext &context, std::size_t chain_index,
        std::size_t cipher_size, std::size_t batch_size,
        const cudaStream_t &stream = cudaStreamPerThread) {
        auto &context_data = context.get_context_data(chain_index);
        auto &parms = context_data.parms();
        chain_index_ = chain_index;
        size_ = cipher_size;
        batch_size_ = batch_size;
        poly_modulus_degree_ = parms.poly_modulus_degree();
        coeff_modulus_size_ = parms.coeff_modulus().size();
        is_ntt_form_ = true;
        scale_ = 1.0;
        correction_factor_ = 1;
        noiseScaleDeg_ = 1;
        is_asymmetric_ = false;

        data_ = phantom::util::make_cuda_auto_ptr<uint64_t>(total_data_count(), stream);
    }

    void resize_like(
        const PhantomContext &, const PhantomCiphertext &prototype,
        std::size_t batch_size,
        const cudaStream_t &stream = cudaStreamPerThread) {
        chain_index_ = prototype.chain_index();
        size_ = prototype.size();
        batch_size_ = batch_size;
        poly_modulus_degree_ = prototype.poly_modulus_degree();
        coeff_modulus_size_ = prototype.coeff_modulus_size();
        is_ntt_form_ = prototype.is_ntt_form();
        scale_ = prototype.scale();
        correction_factor_ = prototype.correction_factor();
        noiseScaleDeg_ = prototype.GetNoiseScaleDeg();
        is_asymmetric_ = prototype.is_asymmetric();

        data_ = phantom::util::make_cuda_auto_ptr<uint64_t>(total_data_count(), stream);
    }

    std::size_t batch_size() const { return batch_size_; }
    std::size_t chain_index() const { return chain_index_; }
    std::size_t size() const { return size_; }
    std::size_t poly_modulus_degree() const { return poly_modulus_degree_; }
    std::size_t coeff_modulus_size() const { return coeff_modulus_size_; }
    std::size_t coeff_count() const {
        return poly_modulus_degree_ * coeff_modulus_size_;
    }
    std::size_t item_data_count() const { return size_ * coeff_count(); }
    std::size_t total_data_count() const { return batch_size_ * item_data_count(); }
    bool is_ntt_form() const { return is_ntt_form_; }
    bool is_asymmetric() const { return is_asymmetric_; }
    double scale() const { return scale_; }
    uint64_t correction_factor() const { return correction_factor_; }
    std::size_t noiseScaleDeg() const { return noiseScaleDeg_; }

    uint64_t *data() const {
        if (data_.get() == nullptr) {
            throw std::runtime_error("PhantomBatchCiphertext: null data");
        }
        return data_.get();
    }

    BatchCipherView view() {
        return BatchCipherView(
            data(), batch_size_, chain_index_, size_, poly_modulus_degree_,
            coeff_modulus_size_, scale_, correction_factor_, noiseScaleDeg_,
            is_ntt_form_, is_asymmetric_);
    }

    ConstBatchCipherView view() const {
        return ConstBatchCipherView(
            data(), batch_size_, chain_index_, size_, poly_modulus_degree_,
            coeff_modulus_size_, scale_, correction_factor_, noiseScaleDeg_,
            is_ntt_form_, is_asymmetric_);
    }

    operator BatchCipherView() { return view(); }
    operator ConstBatchCipherView() const { return view(); }

    PhantomCiphertextView operator[](std::size_t index) {
        return view()[index];
    }

    ConstPhantomCiphertextView operator[](std::size_t index) const {
        return view()[index];
    }

    void copy_from(
        std::size_t index, const PhantomCiphertext &src,
        const cudaStream_t &stream = cudaStreamPerThread) {
        if (index >= batch_size_) {
            throw std::out_of_range("PhantomBatchCiphertext::copy_from index");
        }
        if (src.size() != size_ ||
            src.poly_modulus_degree() != poly_modulus_degree_ ||
            src.coeff_modulus_size() != coeff_modulus_size_) {
            throw std::invalid_argument(
                "PhantomBatchCiphertext::copy_from shape mismatch");
        }
        if (src.chain_index() != chain_index_ ||
            src.is_ntt_form() != is_ntt_form_ ||
            src.scale() != scale_ ||
            src.correction_factor() != correction_factor_ ||
            src.GetNoiseScaleDeg() != noiseScaleDeg_ ||
            src.is_asymmetric() != is_asymmetric_) {
            throw std::invalid_argument(
                "PhantomBatchCiphertext::copy_from metadata mismatch");
        }

        auto *dst_ptr = data() + index * item_data_count();
        const auto bytes = item_data_count() * sizeof(uint64_t);
        PHANTOM_CHECK_CUDA(
            cudaMemcpyAsync(dst_ptr, src.data(), bytes, cudaMemcpyDeviceToDevice, stream));
    }

    void copy_from(
        const PhantomBatchCiphertext &src,
        const cudaStream_t &stream = cudaStreamPerThread) {
        if (src.batch_size() != batch_size_ || src.size() != size_ ||
            src.poly_modulus_degree() != poly_modulus_degree_ ||
            src.coeff_modulus_size() != coeff_modulus_size_) {
            throw std::invalid_argument(
                "PhantomBatchCiphertext::copy_from(batch) shape mismatch");
        }
        const auto bytes = total_data_count() * sizeof(uint64_t);
        PHANTOM_CHECK_CUDA(
            cudaMemcpyAsync(data(), src.data(), bytes, cudaMemcpyDeviceToDevice, stream));
        chain_index_ = src.chain_index_;
        scale_ = src.scale_;
        is_ntt_form_ = src.is_ntt_form_;
        correction_factor_ = src.correction_factor_;
        noiseScaleDeg_ = src.noiseScaleDeg_;
        is_asymmetric_ = src.is_asymmetric_;
    }

    void set_scale(double scale) { scale_ = scale; }
    void set_ntt_form(bool is_ntt_form) { is_ntt_form_ = is_ntt_form; }
    void set_correction_factor(uint64_t correction_factor) {
        correction_factor_ = correction_factor;
    }
    void set_noiseScaleDeg(std::size_t noiseScaleDeg) {
        noiseScaleDeg_ = noiseScaleDeg;
    }
    void set_asymmetric(bool is_asymmetric) { is_asymmetric_ = is_asymmetric; }

private:
    std::size_t chain_index_ = 0;
    std::size_t batch_size_ = 0;
    std::size_t size_ = 0;
    std::size_t poly_modulus_degree_ = 0;
    std::size_t coeff_modulus_size_ = 0;
    double scale_ = 1.0;
    uint64_t correction_factor_ = 1;
    std::size_t noiseScaleDeg_ = 1;
    bool is_ntt_form_ = true;
    bool is_asymmetric_ = false;
    phantom::util::cuda_auto_ptr<uint64_t> data_;
};

class PhantomBatchPlaintext {
public:
    PhantomBatchPlaintext() = default;
    PhantomBatchPlaintext(const PhantomBatchPlaintext &) = default;
    PhantomBatchPlaintext &operator=(const PhantomBatchPlaintext &) = default;
    PhantomBatchPlaintext(PhantomBatchPlaintext &&) = default;
    PhantomBatchPlaintext &operator=(PhantomBatchPlaintext &&) = default;
    ~PhantomBatchPlaintext() = default;

    void resize_like(
        const PhantomContext &, const PhantomPlaintext &prototype,
        std::size_t batch_size,
        const cudaStream_t &stream = cudaStreamPerThread) {
        chain_index_ = prototype.chain_index();
        batch_size_ = batch_size;
        coeff_modulus_size_ = prototype.coeff_modulus_size();
        poly_modulus_degree_ = prototype.coeff_count() / coeff_modulus_size_;
        scale_ = prototype.scale();
        is_ntt_form_ = prototype.is_ntt_form();

        data_ = phantom::util::make_cuda_auto_ptr<uint64_t>(total_data_count(), stream);
    }

    void resize(
        std::size_t chain_index, std::size_t poly_modulus_degree,
        std::size_t coeff_modulus_size, std::size_t batch_size,
        const cudaStream_t &stream = cudaStreamPerThread) {
        chain_index_ = chain_index;
        batch_size_ = batch_size;
        coeff_modulus_size_ = coeff_modulus_size;
        poly_modulus_degree_ = poly_modulus_degree;
        is_ntt_form_ = (chain_index != 0);
        scale_ = 1.0;
        data_ = phantom::util::make_cuda_auto_ptr<uint64_t>(total_data_count(), stream);
    }

    std::size_t batch_size() const { return batch_size_; }
    std::size_t chain_index() const { return chain_index_; }
    std::size_t poly_modulus_degree() const { return poly_modulus_degree_; }
    std::size_t coeff_modulus_size() const { return coeff_modulus_size_; }
    std::size_t coeff_count() const {
        return poly_modulus_degree_ * coeff_modulus_size_;
    }
    std::size_t item_data_count() const { return coeff_count(); }
    std::size_t total_data_count() const { return batch_size_ * coeff_count(); }
    bool is_ntt_form() const { return is_ntt_form_; }
    double scale() const { return scale_; }

    uint64_t *data() const {
        if (data_.get() == nullptr) {
            throw std::runtime_error("PhantomBatchPlaintext: null data");
        }
        return data_.get();
    }

    BatchPlaintextView view() {
        return BatchPlaintextView(
            data(), batch_size_, chain_index_, poly_modulus_degree_,
            coeff_modulus_size_, scale_, is_ntt_form_);
    }

    ConstBatchPlaintextView view() const {
        return ConstBatchPlaintextView(
            data(), batch_size_, chain_index_, poly_modulus_degree_,
            coeff_modulus_size_, scale_, is_ntt_form_);
    }

    operator BatchPlaintextView() { return view(); }
    operator ConstBatchPlaintextView() const { return view(); }

    PhantomPlaintextView operator[](std::size_t index) { return view()[index]; }
    ConstPhantomPlaintextView operator[](std::size_t index) const {
        return view()[index];
    }

    void copy_from(
        std::size_t index, const PhantomPlaintext &src,
        const cudaStream_t &stream = cudaStreamPerThread) {
        if (index >= batch_size_) {
            throw std::out_of_range("PhantomBatchPlaintext::copy_from index");
        }
        if (src.coeff_modulus_size() != coeff_modulus_size_ ||
            src.coeff_count() != coeff_count()) {
            throw std::invalid_argument(
                "PhantomBatchPlaintext::copy_from shape mismatch");
        }
        if (src.chain_index() != chain_index_ ||
            src.is_ntt_form() != is_ntt_form_ ||
            src.scale() != scale_) {
            throw std::invalid_argument(
                "PhantomBatchPlaintext::copy_from metadata mismatch");
        }

        auto *dst_ptr = data() + index * item_data_count();
        const auto bytes = item_data_count() * sizeof(uint64_t);
        PHANTOM_CHECK_CUDA(
            cudaMemcpyAsync(dst_ptr, src.data(), bytes, cudaMemcpyDeviceToDevice, stream));
    }

    void copy_from(
        const PhantomBatchPlaintext &src,
        const cudaStream_t &stream = cudaStreamPerThread) {
        if (src.batch_size() != batch_size_ ||
            src.poly_modulus_degree() != poly_modulus_degree_ ||
            src.coeff_modulus_size() != coeff_modulus_size_) {
            throw std::invalid_argument(
                "PhantomBatchPlaintext::copy_from(batch) shape mismatch");
        }
        const auto bytes = total_data_count() * sizeof(uint64_t);
        PHANTOM_CHECK_CUDA(
            cudaMemcpyAsync(data(), src.data(), bytes, cudaMemcpyDeviceToDevice, stream));
        chain_index_ = src.chain_index_;
        is_ntt_form_ = src.is_ntt_form_;
        scale_ = src.scale_;
    }

    void set_scale(double scale) { scale_ = scale; }
    void set_ntt_form(bool is_ntt_form) { is_ntt_form_ = is_ntt_form; }

private:
    std::size_t chain_index_ = 0;
    std::size_t batch_size_ = 0;
    std::size_t poly_modulus_degree_ = 0;
    std::size_t coeff_modulus_size_ = 0;
    bool is_ntt_form_ = false;
    double scale_ = 1.0;
    phantom::util::cuda_auto_ptr<uint64_t> data_;
};

#pragma once

#include "common.h"

class DBaseConverter {

private:

    phantom::arith::DRNSBase ibase_;
    phantom::arith::DRNSBase obase_;

    phantom::util::cuda_auto_ptr<uint64_t> qiHat_mod_pj_;
    phantom::util::cuda_auto_ptr<uint64_t> alpha_Q_mod_pj_;
    phantom::util::cuda_auto_ptr<uint64_t> negPQHatInvModq_;
    phantom::util::cuda_auto_ptr<uint64_t> negPQHatInvModq_shoup_;
    phantom::util::cuda_auto_ptr<uint64_t> QInvModp_;
    phantom::util::cuda_auto_ptr<uint64_t> PModq_;
    phantom::util::cuda_auto_ptr<uint64_t> PModq_shoup_;

public:

    DBaseConverter() = default;

    explicit DBaseConverter(phantom::arith::BaseConverter &cpu_base_converter, const cudaStream_t &stream) {
        init(cpu_base_converter, stream);
    }

    void init(phantom::arith::BaseConverter &cpu_base_converter, const cudaStream_t &stream) {
        ibase_.init(cpu_base_converter.ibase(), stream);
        obase_.init(cpu_base_converter.obase(), stream);

        qiHat_mod_pj_ = phantom::util::make_cuda_auto_ptr<uint64_t>(obase_.size() * ibase_.size(), stream);
        for (size_t idx = 0; idx < obase_.size(); idx++)
            cudaMemcpyAsync(qiHat_mod_pj_.get() + idx * ibase_.size(), cpu_base_converter.QHatModp(idx),
                            ibase_.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice, stream);

        alpha_Q_mod_pj_ = phantom::util::make_cuda_auto_ptr<uint64_t>((ibase_.size() + 1) * obase_.size(), stream);
        for (size_t idx = 0; idx < ibase_.size() + 1; idx++)
            cudaMemcpyAsync(alpha_Q_mod_pj_.get() + idx * obase_.size(), cpu_base_converter.alphaQModp(idx),
                            obase_.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice, stream);

        negPQHatInvModq_ = phantom::util::make_cuda_auto_ptr<uint64_t>(ibase_.size(), stream);
        cudaMemcpyAsync(negPQHatInvModq_.get(), cpu_base_converter.negPQHatInvModq(),
                        ibase_.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);

        negPQHatInvModq_shoup_ = phantom::util::make_cuda_auto_ptr<uint64_t>(ibase_.size(), stream);
        cudaMemcpyAsync(negPQHatInvModq_shoup_.get(), cpu_base_converter.negPQHatInvModq_shoup(),
                        ibase_.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);

        QInvModp_ = phantom::util::make_cuda_auto_ptr<uint64_t>(obase_.size() * ibase_.size(), stream);
        for (size_t idx = 0; idx < obase_.size(); idx++)
            cudaMemcpyAsync(QInvModp_.get() + idx * ibase_.size(), cpu_base_converter.QInvModp(idx),
                            ibase_.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice, stream);

        PModq_ = phantom::util::make_cuda_auto_ptr<uint64_t>(ibase_.size(), stream);
        cudaMemcpyAsync(PModq_.get(), cpu_base_converter.PModq(), ibase_.size() * sizeof(uint64_t),
                        cudaMemcpyHostToDevice, stream);

        PModq_shoup_ = phantom::util::make_cuda_auto_ptr<uint64_t>(ibase_.size(), stream);
        cudaMemcpyAsync(PModq_shoup_.get(), cpu_base_converter.PModq_shoup(),
                        ibase_.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
    }

    void bConv_BEHZ(uint64_t *dst, const uint64_t *src, size_t n, const cudaStream_t &stream) const;

    void bConv_BEHZ_var1(uint64_t *dst, const uint64_t *src, size_t n, const cudaStream_t &stream) const;

    void bConv_HPS(uint64_t *dst, const uint64_t *src, size_t n, const cudaStream_t &stream) const;

    void exact_convert_array(uint64_t *dst, const uint64_t *src, uint64_t poly_degree, const cudaStream_t &stream) const;

    __host__ inline auto &ibase() const { return ibase_; }

    __host__ inline auto &obase() const { return obase_; }

    __host__ inline uint64_t *QHatModp() const { return qiHat_mod_pj_.get(); }

    __host__ inline uint64_t *alpha_Q_mod_pj() const { return alpha_Q_mod_pj_.get(); }

    __host__ inline uint64_t *negPQHatInvModq() const { return negPQHatInvModq_.get(); }

    __host__ inline uint64_t *negPQHatInvModq_shoup() const { return negPQHatInvModq_shoup_.get(); }

    __host__ inline uint64_t *QInvModp() const { return QInvModp_.get(); }

    __host__ inline uint64_t *PModq() const { return PModq_.get(); }

    __host__ inline uint64_t *PModq_shoup() const { return PModq_shoup_.get(); }
};
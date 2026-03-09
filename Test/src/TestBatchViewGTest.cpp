#include "PhantomBatchTestUtils.h"

#include "batch_view.h"

#include <stdexcept>
#include <vector>

namespace {

using namespace phantom_batch_test;

TEST(BatchViewGTest, CipherBatchCopyFromSingleAndMetadataValidation) {
    PhantomContext context = make_test_context();
    const std::size_t chain = data_chain_index(context);

    auto proto = make_random_ntt_cipher(context, chain, 2, 11, 8.0);

    PhantomBatchCiphertext batch;
    batch.resize_like(context, proto, 3);

    std::vector<PhantomCiphertext> inputs;
    inputs.emplace_back(make_random_ntt_cipher(context, chain, 2, 101, 8.0));
    inputs.emplace_back(make_random_ntt_cipher(context, chain, 2, 102, 8.0));
    inputs.emplace_back(make_random_ntt_cipher(context, chain, 2, 103, 8.0));

    for (std::size_t i = 0; i < inputs.size(); ++i) {
        batch.copy_from(i, inputs[i]);
    }
    sync_stream();

    EXPECT_EQ(batch.batch_size(), 3U);
    EXPECT_EQ(batch.size(), 2U);
    EXPECT_EQ(batch.chain_index(), chain);
    EXPECT_TRUE(batch.is_ntt_form());

    for (std::size_t i = 0; i < inputs.size(); ++i) {
        expect_device_buffer_eq(batch[i].data(), inputs[i].data(), batch.item_data_count());
    }

    auto bad_scale = make_random_ntt_cipher(context, chain, 2, 201, 4.0);
    EXPECT_THROW(batch.copy_from(0, bad_scale), std::invalid_argument);
    EXPECT_THROW(batch.copy_from(3, inputs[0]), std::out_of_range);
}

TEST(BatchViewGTest, CipherBatchCopyFromBatchCopiesRuntimeMetadata) {
    PhantomContext context = make_test_context();
    const std::size_t chain = data_chain_index(context);

    auto proto = make_random_ntt_cipher(context, chain, 2, 21, 2.0);

    PhantomBatchCiphertext src;
    src.resize_like(context, proto, 2);
    src.copy_from(0, make_random_ntt_cipher(context, chain, 2, 211, 2.0));
    src.copy_from(1, make_random_ntt_cipher(context, chain, 2, 212, 2.0));
    src.set_ntt_form(false);
    src.set_scale(16.0);
    src.set_correction_factor(7);
    src.set_noiseScaleDeg(3);
    src.set_asymmetric(true);

    PhantomBatchCiphertext dst;
    dst.resize_like(context, proto, 2);
    dst.set_scale(1.0);
    dst.set_ntt_form(true);
    dst.set_correction_factor(1);
    dst.set_noiseScaleDeg(1);
    dst.set_asymmetric(false);

    dst.copy_from(src);
    sync_stream();

    EXPECT_EQ(dst.scale(), src.scale());
    EXPECT_EQ(dst.is_ntt_form(), src.is_ntt_form());
    EXPECT_EQ(dst.correction_factor(), src.correction_factor());
    EXPECT_EQ(dst.noiseScaleDeg(), src.noiseScaleDeg());
    EXPECT_EQ(dst.is_asymmetric(), src.is_asymmetric());

    expect_device_buffer_eq(dst[0].data(), src[0].data(), dst.item_data_count());
    expect_device_buffer_eq(dst[1].data(), src[1].data(), dst.item_data_count());
}

TEST(BatchViewGTest, PlainBatchCopyAndMetadataValidation) {
    PhantomContext context = make_test_context();
    const std::size_t chain = data_chain_index(context);

    auto proto = make_random_ntt_plain(context, chain, 31, 4.0);

    PhantomBatchPlaintext batch;
    batch.resize_like(context, proto, 2);

    auto p0 = make_random_ntt_plain(context, chain, 301, 4.0);
    auto p1 = make_random_ntt_plain(context, chain, 302, 4.0);
    batch.copy_from(0, p0);
    batch.copy_from(1, p1);
    sync_stream();

    EXPECT_EQ(batch.batch_size(), 2U);
    EXPECT_EQ(batch.chain_index(), chain);
    EXPECT_TRUE(batch.is_ntt_form());

    expect_device_buffer_eq(batch[0].data(), p0.data(), batch.item_data_count());
    expect_device_buffer_eq(batch[1].data(), p1.data(), batch.item_data_count());

    auto bad_scale = make_random_ntt_plain(context, chain, 303, 2.0);
    EXPECT_THROW(batch.copy_from(0, bad_scale), std::invalid_argument);

    PhantomBatchPlaintext src;
    src.resize_like(context, proto, 2);
    src.copy_from(0, p0);
    src.copy_from(1, p1);
    src.set_ntt_form(false);
    src.set_scale(9.0);

    PhantomBatchPlaintext dst;
    dst.resize_like(context, proto, 2);
    dst.copy_from(src);
    sync_stream();

    EXPECT_EQ(dst.scale(), src.scale());
    EXPECT_EQ(dst.is_ntt_form(), src.is_ntt_form());
    EXPECT_EQ(dst.chain_index(), src.chain_index());
}

} // namespace

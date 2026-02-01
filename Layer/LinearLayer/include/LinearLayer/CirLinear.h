
#include <seal/seal.h>
#include <Datatype/Tensor.h>
#include <HE/HE.h>
#include <LinearOperator/Conversion.h>
#include "../../../Layer/Module.h"

using namespace seal;
using namespace Datatype;
using namespace HE;
using namespace HE::unified;

namespace LinearLayer {

/**
 * CirLinearNest: Block Circulant Linear Layer with Nested Encoding
 * 
 * This implements the CirEncode algorithm from protocol.tex for block circulant GEMMs.
 * Key idea: Y = WX where W is a block circulant matrix with block size b.
 * 
 * Two-level encoding:
 * 1. Within each circulant block: coefficient encoding (Theorem 1)
 *    - For circulant W (b×b), only encode first column: ŵ[i*d1] = W[i,0]
 *    - For input X (b×d1): x̂[i*d1+j] = X[i,j]
 *    - Polynomial multiplication gives Y = WX
 * 
 * 2. Across circulant blocks: SIMD encoding with DFT/NTT (Theorem 2)
 *    - Pack multiple circulant blocks in parallel
 *    - Use BSGS for efficient rotation
 * 
 * Dimensions:
 *   Input activation: (dim_0, dim_1) where dim_1 must be divisible by block_size
 *   Weight matrix: (dim_1, dim_2) block circulant with block_size
 *   Output: (dim_0, dim_2)
 * 
 * When block_size=1, this reduces to LinearNest.
 */
class CirLinearNest : public Module {
public:
    // Original dimensions
    uint64_t dim_0;         // Batch size (number of rows in input)
    uint64_t dim_1;         // Input channels (must be multiple of block_size)
    uint64_t dim_2;         // Output channels (must be multiple of block_size)
    uint64_t block_size;    // Circulant block size (b)
    
    // Derived dimensions for block-level GEMM
    uint64_t num_blocks_1;  // dim_1 / block_size (input blocks)
    uint64_t num_blocks_2;  // dim_2 / block_size (output blocks)
    
    // HE parameters
    uint64_t padded_dim_0;      // dim_0 padded to power of 2
    uint64_t ntt_size;          // padded_dim_0 * block_size (NTT block size)
    uint64_t tile_size;         // N / (2 * ntt_size) - number of blocks per ciphertext
    uint64_t tiled_blocks_1;    // ceil(num_blocks_1 / tile_size)
    uint64_t tiled_blocks_2;    // ceil(num_blocks_2 / tile_size)
    uint64_t padded_blocks_1;   // tiled_blocks_1 * tile_size
    uint64_t padded_blocks_2;   // tiled_blocks_2 * tile_size
    uint64_t input_rot;         // sqrt(tile_size) for BSGS
    
    // Data
    Tensor<uint64_t> weight;
    Tensor<uint64_t> padded_weight;
    Tensor<uint64_t> bias;
    Tensor<HE::unified::UnifiedPlaintext> weight_pt;
    HE::HEEvaluator* HE;

    // Constructors
    CirLinearNest(uint64_t dim_0, uint64_t block_size, 
                  const Tensor<uint64_t>& weight, const Tensor<uint64_t>& bias, 
                  HE::HEEvaluator* HE);
    CirLinearNest(uint64_t dim_0, uint64_t dim_1, uint64_t dim_2, uint64_t block_size,
                  HE::HEEvaluator* HE);
    
    virtual ~CirLinearNest() = default;
    
    Tensor<uint64_t> operator()(Tensor<uint64_t> &x);
    
    // Statistics
    uint64_t rotation_count = 0;  // Count of HE rotations performed
    uint64_t multiply_count = 0;  // Count of HE multiply_plain operations
    double rotation_time_ms = 0;  // Total time for rotations (ms)
    double multiply_time_ms = 0;  // Total time for multiply_plain (ms)
    
    uint64_t getRotationCount() const { return rotation_count; }
    uint64_t getMultiplyCount() const { return multiply_count; }
    double getRotationTimeMs() const { return rotation_time_ms; }
    double getMultiplyTimeMs() const { return multiply_time_ms; }

private:
    void compute_he_params();
    
    /**
     * PackWeight: Encode weight matrix with nested encoding
     * For each circulant block, only encode the first column using coefficient encoding.
     * Then apply NTT to convert to SIMD format for parallel computation.
     */
    Tensor<HE::unified::UnifiedPlaintext> PackWeight();
    
    /**
     * PackActivation: Encode input activation with nested encoding
     * Pack input into blocks of size (block_size × padded_dim_0).
     * Apply NTT for SIMD parallel computation.
     */
    Tensor<uint64_t> PackActivation(Tensor<uint64_t> &x);
    
    /**
     * HECompute: Perform HE computation using BSGS
     * Rotations are at block granularity (ntt_size slots per block).
     */
    Tensor<HE::unified::UnifiedCiphertext> HECompute(
        const Tensor<HE::unified::UnifiedPlaintext> &weight_pt, 
        Tensor<HE::unified::UnifiedCiphertext> &ac_ct);
    
    /**
     * DepackResult: Extract result from HE output
     * Apply iNTT and extract according to coefficient encoding rule.
     */
    Tensor<uint64_t> DepackResult(Tensor<uint64_t> &out_msg);
};

} // namespace LinearLayer

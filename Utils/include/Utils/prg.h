#include "aes-ni.h"
#include <emp-tool/utils/aes.h>
#include "block.h"
#include <emp-tool/utils/constants.h>
#include <random>

#ifdef EMP_USE_RANDOM_DEVICE
#else
#include <x86intrin.h>
#endif

/** @addtogroup BP
  @{
 */
namespace Utils
{

  class PRG128
  {
  public:
    uint64_t counter = 0;
    AES_KEY aes;
    PRG128(const void *seed = nullptr, int id = 0)
    {
      if (seed != nullptr)
      {
        reseed(seed, id);
      }
      else
      {
        block128 v;
#ifdef EMP_USE_RANDOM_DEVICE
        int *data = (int *)(&v);
        std::random_device rand_div;
        for (size_t i = 0; i < sizeof(block128) / sizeof(int); ++i)
          data[i] = rand_div();
#else
        unsigned long long r0, r1;
        _rdseed64_step(&r0);
        _rdseed64_step(&r1);
        v = makeBlock128(r0, r1);
#endif
        reseed(&v);
      }
    }
    void reseed(const void *key, uint64_t id = 0)
    {
      block128 v = _mm_loadu_si128((block128 *)key);
      v = xorBlocks(v, makeBlock128(0LL, id));
      AES_set_encrypt_key(v, &aes);
      counter = 0;
    }

    void random_data(void *data, int nbytes)
    {
      random_block((block128 *)data, nbytes / 16);
      if (nbytes % 16 != 0)
      {
        block128 extra;
        random_block(&extra, 1);
        memcpy((nbytes / 16 * 16) + (char *)data, &extra, nbytes % 16);
      }
    }

    void random_bool(bool *data, int length)
    {
      uint8_t *uint_data = (uint8_t *)data;
      random_data(uint_data, length);
      for (int i = 0; i < length; ++i)
        data[i] = uint_data[i] & 1;
    }

    void random_data_unaligned(void *data, int nbytes)
    {
      block128 tmp[AES_BATCH_SIZE];
      for (int i = 0; i < nbytes / (AES_BATCH_SIZE * 16); i++)
      {
        random_block(tmp, AES_BATCH_SIZE);
        memcpy((16 * i * AES_BATCH_SIZE) + (uint8_t *)data, tmp,
               16 * AES_BATCH_SIZE);
      }
      if (nbytes % (16 * AES_BATCH_SIZE) != 0)
      {
        random_block(tmp, AES_BATCH_SIZE);
        memcpy((nbytes / (16 * AES_BATCH_SIZE) * (16 * AES_BATCH_SIZE)) +
                   (uint8_t *)data,
               tmp, nbytes % (16 * AES_BATCH_SIZE));
      }
    }

    void random_block(block128 *data, int nblocks = 1)
    {
      for (int i = 0; i < nblocks; ++i)
      {
        data[i] = makeBlock128(0LL, counter++);
      }
      int i = 0;
      for (; i < nblocks - AES_BATCH_SIZE; i += AES_BATCH_SIZE)
      {
        AES_ecb_encrypt_blks(data + i, AES_BATCH_SIZE, &aes);
      }
      AES_ecb_encrypt_blks(
          data + i, (AES_BATCH_SIZE > nblocks - i) ? nblocks - i : AES_BATCH_SIZE,
          &aes);
    }

    void random_block(block256 *data, int nblocks = 1)
    {
      nblocks = nblocks * 2;
      block128 tmp[nblocks];
      for (int i = 0; i < nblocks; ++i)
      {
        tmp[i] = makeBlock128(0LL, counter++);
      }
      int i = 0;
      for (; i < nblocks - AES_BATCH_SIZE; i += AES_BATCH_SIZE)
      {
        AES_ecb_encrypt_blks(tmp + i, AES_BATCH_SIZE, &aes);
      }
      AES_ecb_encrypt_blks(
          tmp + i, (AES_BATCH_SIZE > nblocks - i) ? nblocks - i : AES_BATCH_SIZE,
          &aes);
      for (int i = 0; i < nblocks / 2; ++i)
      {
        data[i] = makeBlock256(tmp[2 * i], tmp[2 * i + 1]);
      }
    }

    template <typename T>
    void random_mod_p(T *arr, uint64_t size, T prime_mod)
    {
      T boundary = (((-1 * prime_mod) / prime_mod) + 1) *
                   prime_mod; // prime_mod*floor((2^l)/prime_mod)
      int tries_before_resampling = 2;
      uint64_t size_total = tries_before_resampling * size;
      T *randomness = new T[size_total];
      uint64_t rptr = 0, arrptr = 0;
      while (arrptr < size)
      {
        this->random_data(randomness, sizeof(T) * size_total);
        rptr = 0;
        for (; (arrptr < size) && (rptr < size_total); arrptr++, rptr++)
        {
          while (randomness[rptr] > boundary)
          {
            rptr++;
            if (rptr >= size_total)
            {
              this->random_data(randomness, sizeof(T) * size_total);
              rptr = 0;
            }
          }
          arr[arrptr] = randomness[rptr] % prime_mod;
        }
      }
      delete[] randomness;
    }
  };

  class PRG256
  {
  public:
    uint64_t counter = 0;
    AESNI_KEY aes;
    PRG256(const void *seed = nullptr, int id = 0)
    {
      if (seed != nullptr)
      {
        reseed(seed, id);
      }
      else
      {
        alignas(32) block256 v;
#ifdef EMP_USE_RANDOM_DEVICE
        int *data = (int *)(&v);
        std::random_device rand_div;
        for (size_t i = 0; i < sizeof(block256) / sizeof(int); ++i)
          data[i] = rand_div();
#else
        unsigned long long r0, r1, r2, r3;
        _rdseed64_step(&r0);
        _rdseed64_step(&r1);
        _rdseed64_step(&r2);
        _rdseed64_step(&r3);
        v = makeBlock256(r0, r1, r2, r3);
#endif
        reseed(&v);
      }
    }
    void reseed(const void *key, uint64_t id = 0)
    {
      alignas(32) block256 v = _mm256_load_si256((block256 *)key);
      v = xorBlocks(v, makeBlock256(0LL, 0LL, 0LL, id));
      AESNI_set_encrypt_key(&aes, v);
      counter = 0;
    }

    void random_data(void *data, int nbytes)
    {
      random_block((block128 *)data, nbytes / 16);
      if (nbytes % 16 != 0)
      {
        block128 extra;
        random_block(&extra, 1);
        memcpy((nbytes / 16 * 16) + (char *)data, &extra, nbytes % 16);
      }
    }

    void random_bool(bool *data, int length)
    {
      uint8_t *uint_data = (uint8_t *)data;
      random_data(uint_data, length);
      for (int i = 0; i < length; ++i)
        data[i] = uint_data[i] & 1;
    }

    void random_data_unaligned(void *data, int nbytes)
    {
      block128 tmp[AES_BATCH_SIZE];
      for (int i = 0; i < nbytes / (AES_BATCH_SIZE * 16); i++)
      {
        random_block(tmp, AES_BATCH_SIZE);
        memcpy((16 * i * AES_BATCH_SIZE) + (uint8_t *)data, tmp,
               16 * AES_BATCH_SIZE);
      }
      if (nbytes % (16 * AES_BATCH_SIZE) != 0)
      {
        random_block(tmp, AES_BATCH_SIZE);
        memcpy((nbytes / (16 * AES_BATCH_SIZE) * (16 * AES_BATCH_SIZE)) +
                   (uint8_t *)data,
               tmp, nbytes % (16 * AES_BATCH_SIZE));
      }
    }

    void random_block(block128 *data, int nblocks = 1)
    {
      for (int i = 0; i < nblocks; ++i)
      {
        data[i] = makeBlock128(0LL, counter++);
      }
      int i = 0;
      for (; i < nblocks - AES_BATCH_SIZE; i += AES_BATCH_SIZE)
      {
        AESNI_ecb_encrypt_blks(data + i, AES_BATCH_SIZE, &aes);
      }
      AESNI_ecb_encrypt_blks(
          data + i, (AES_BATCH_SIZE > nblocks - i) ? nblocks - i : AES_BATCH_SIZE,
          &aes);
    }

    void random_block(block256 *data, int nblocks = 1)
    {
      nblocks = nblocks * 2;
      block128 tmp[nblocks];
      for (int i = 0; i < nblocks; ++i)
      {
        tmp[i] = makeBlock128(0LL, counter++);
      }
      int i = 0;
      for (; i < nblocks - AES_BATCH_SIZE; i += AES_BATCH_SIZE)
      {
        AESNI_ecb_encrypt_blks(tmp + i, AES_BATCH_SIZE, &aes);
      }
      AESNI_ecb_encrypt_blks(
          tmp + i, (AES_BATCH_SIZE > nblocks - i) ? nblocks - i : AES_BATCH_SIZE,
          &aes);
      for (int i = 0; i < nblocks / 2; ++i)
      {
        data[i] = makeBlock256(tmp[2 * i], tmp[2 * i + 1]);
      }
    }
  };
} // namespace Utils

#include "Datatype/UnifiedType.h"
#include "HE/unified/UnifiedEncoder.h"
#include "HE/unified/UnifiedPlaintext.h"
#include <HE/unified/UnifiedEvaluator.h>
#include <seal/seal.h>

using namespace std;
using namespace seal;
using namespace HE::unified;

inline void print_line(int line_number) {
  std::cout << "Line " << std::setw(3) << line_number << " --> ";
}

template <typename T>
inline void print_matrix(std::vector<T> matrix, std::size_t row_size) {
  /*
  We're not going to print every column of the matrix (there are 2048). Instead
  print this many slots from beginning and end of the matrix.
  */
  std::size_t print_size = 5;

  std::cout << std::endl;
  std::cout << "    [";
  for (std::size_t i = 0; i < print_size; i++) {
    std::cout << std::setw(3) << std::right << matrix[i] << ",";
  }
  std::cout << std::setw(3) << " ...,";
  for (std::size_t i = row_size - print_size; i < row_size; i++) {
    std::cout << std::setw(3) << matrix[i]
              << ((i != row_size - 1) ? "," : " ]\n");
  }
  std::cout << "    [";
  for (std::size_t i = row_size; i < row_size + print_size; i++) {
    std::cout << std::setw(3) << matrix[i] << ",";
  }
  std::cout << std::setw(3) << " ...,";
  for (std::size_t i = 2 * row_size - print_size; i < 2 * row_size; i++) {
    std::cout << std::setw(3) << matrix[i]
              << ((i != 2 * row_size - 1) ? "," : " ]\n");
  }
  std::cout << std::endl;
}

void bfv_rotation_example() {
  uint64_t polyModulusDegree = 8192;
  uint64_t plainWidth = 20;

  UnifiedContext context(polyModulusDegree, plainWidth, true, Datatype::DEVICE);
  UnifiedBatchEncoder encoder(context);
  UnifiedEvaluator evaluator(context);

  SecretKey *secretKeys = new SecretKey();
  PublicKey *publicKeys = new PublicKey();
  UnifiedGaloisKeys *galoisKeys = new UnifiedGaloisKeys(HOST);

  KeyGenerator keygen(context);
  *secretKeys = keygen.secret_key();
  keygen.create_public_key(*publicKeys);
  keygen.create_galois_keys(*galoisKeys);
  galoisKeys->to_device(context);

  Encryptor encryptor(context, *publicKeys);
  Decryptor decryptor(context, *secretKeys);

  size_t slot_count = encoder.slot_count();
  size_t row_size = slot_count / 2;
  cout << "Plaintext matrix row size: " << row_size << endl;

  vector<uint64_t> pod_matrix(slot_count, 0ULL);
  pod_matrix[0] = 0ULL;
  pod_matrix[1] = 1ULL;
  pod_matrix[2] = 2ULL;
  pod_matrix[3] = 3ULL;
  pod_matrix[row_size] = 4ULL;
  pod_matrix[row_size + 1] = 5ULL;
  pod_matrix[row_size + 2] = 6ULL;
  pod_matrix[row_size + 3] = 7ULL;

  cout << "Input plaintext matrix:" << endl;
  print_matrix(pod_matrix, row_size);

  int step = 2;
  /*
  First we use BatchEncoder to encode the matrix into a plaintext. We encrypt
  the plaintext as usual.
  */
  UnifiedPlaintext plain_matrix(HOST);
  print_line(__LINE__);
  cout << "Encode and encrypt." << endl;
  encoder.encode(pod_matrix, plain_matrix);
  UnifiedCiphertext encrypted_matrix(HOST);
  encryptor.encrypt(plain_matrix, encrypted_matrix);
  cout << "    + Noise budget in fresh encryption: "
       << decryptor.invariant_noise_budget(encrypted_matrix) << " bits" << endl;
  cout << endl;

  UnifiedCiphertext d_encrypted_matrix = encrypted_matrix;
  d_encrypted_matrix.to_device(context);

  /*
    Now rotate both matrix rows 3 steps to the left, decrypt, decode, and print.
    */
  print_line(__LINE__);
  cout << "Rotate rows " << step << " steps left." << endl;
  evaluator.rotate_rows_inplace(encrypted_matrix, step, *galoisKeys);
  Plaintext plain_result;
  cout << "    + Noise budget after rotation: "
       << decryptor.invariant_noise_budget(encrypted_matrix) << " bits" << endl;
  cout << "    + Decrypt and decode ...... Correct." << endl;
  decryptor.decrypt(encrypted_matrix, plain_result);
  encoder.decode(plain_result, pod_matrix);
  print_matrix(pod_matrix, row_size);

  print_line(__LINE__);
  cout << "Rotate rows " << step << " steps left (on DEVICE)." << endl;
  evaluator.rotate_rows_inplace(d_encrypted_matrix, step, *galoisKeys);
  d_encrypted_matrix.to_host(context);
  cout << "    + Noise budget after rotation: "
       << decryptor.invariant_noise_budget(d_encrypted_matrix) << " bits"
       << endl;
  cout << "    + Decrypt and decode ...... Correct." << endl;
  decryptor.decrypt(d_encrypted_matrix, plain_result);
  encoder.decode(plain_result, pod_matrix);
  print_matrix(pod_matrix, row_size);
}

void bfv_ct_ct_mult_example() {
  uint64_t polyModulusDegree = 8192;
  uint64_t plainWidth = 20;

  UnifiedContext context(polyModulusDegree, plainWidth, true, Datatype::DEVICE);
  UnifiedBatchEncoder encoder(context);
  UnifiedEvaluator evaluator(context);

  SecretKey *secretKeys = new SecretKey();
  PublicKey *publicKeys = new PublicKey();
  UnifiedRelinKeys *relinKeys = new UnifiedRelinKeys(HOST);

  KeyGenerator keygen(context);
  *secretKeys = keygen.secret_key();
  keygen.create_public_key(*publicKeys);
  keygen.create_relin_keys(*relinKeys);
  relinKeys->to_device(context);

  Encryptor encryptor(context, *publicKeys);
  Decryptor decryptor(context, *secretKeys);

  size_t slot_count = encoder.slot_count();
  size_t row_size = slot_count / 2;
  cout << "Plaintext matrix row size: " << row_size << endl;

  vector<uint64_t> pod_matrix(slot_count, 0ULL);
  pod_matrix[0] = 1ULL;
  pod_matrix[1] = 2ULL;
  pod_matrix[2] = 3ULL;
  pod_matrix[3] = 4ULL;
  pod_matrix[row_size] = 5ULL;
  pod_matrix[row_size + 1] = 6ULL;
  pod_matrix[row_size + 2] = 7ULL;
  pod_matrix[row_size + 3] = 8ULL;

  cout << "Input plaintext matrix:" << endl;
  print_matrix(pod_matrix, row_size);

  /*
  First we use BatchEncoder to encode the matrix into a plaintext. We encrypt
  the plaintext as usual.
  */
  UnifiedPlaintext plain_matrix(HOST);
  print_line(__LINE__);
  cout << "Encode and encrypt." << endl;
  encoder.encode(pod_matrix, plain_matrix);
  UnifiedCiphertext encrypted_matrix(HOST);
  encryptor.encrypt(plain_matrix, encrypted_matrix);
  cout << "    + Noise budget in fresh encryption: "
       << decryptor.invariant_noise_budget(encrypted_matrix) << " bits" << endl;
  cout << endl;

  UnifiedCiphertext d_encrypted_matrix = encrypted_matrix;
  d_encrypted_matrix.to_device(context);

  /*
    Now rotate both matrix rows 3 steps to the left, decrypt, decode, and print.
    */
  print_line(__LINE__);
  cout << "Square (on HOST)." << endl;
  evaluator.square_inplace(encrypted_matrix);
  evaluator.relinearize_inplace(encrypted_matrix, *relinKeys);
  Plaintext plain_result;
  cout << "    + Noise budget after rotation: "
       << decryptor.invariant_noise_budget(encrypted_matrix) << " bits" << endl;
  cout << "    + Decrypt and decode ...... Correct." << endl;
  decryptor.decrypt(encrypted_matrix, plain_result);
  encoder.decode(plain_result, pod_matrix);
  print_matrix(pod_matrix, row_size);

  print_line(__LINE__);
  cout << "Square (on DEVICE)." << endl;
  evaluator.square_inplace(d_encrypted_matrix);
  evaluator.relinearize_inplace(d_encrypted_matrix, *relinKeys);
  d_encrypted_matrix.to_host(context);
  cout << "    + Noise budget after rotation: "
       << decryptor.invariant_noise_budget(d_encrypted_matrix) << " bits"
       << endl;
  cout << "    + Decrypt and decode ...... Correct." << endl;
  decryptor.decrypt(d_encrypted_matrix, plain_result);
  encoder.decode(plain_result, pod_matrix);
  print_matrix(pod_matrix, row_size);
}

int main() {
  // bfv_rotation_example();
  bfv_ct_ct_mult_example();
  return 0;
}
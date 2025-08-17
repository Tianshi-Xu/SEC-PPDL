#include <HE/unified/UnifiedEvaluator.h>
#include <cassert>
#include <iomanip>
#include <seal/seal.h>
#include "Datatype/UnifiedType.h"
#include "HE/unified/UnifiedEncoder.h"
#include "HE/unified/UnifiedPlaintext.h"

using namespace std;
using namespace seal;
using namespace HE::unified;

// Test result verification functions
bool verify_results(const vector<uint64_t> &expected, const vector<uint64_t> &actual, const string &test_name)
{
    if (expected.size() != actual.size())
    {
        cout << "\n\033[1;31m" << string(60, '=') << "\033[0m" << endl;
        cout << "\033[1;31mFAILED: " << test_name << "\033[0m" << endl;
        cout << "\033[1;31mSize mismatch: expected " << expected.size() << ", got " << actual.size() << "\033[0m"
             << endl;
        cout << "\033[1;31m" << string(60, '=') << "\033[0m" << endl;
        return false;
    }

    for (size_t i = 0; i < expected.size(); ++i)
    {
        if (expected[i] != actual[i])
        {
            cout << "\n\033[1;31m" << string(60, '=') << "\033[0m" << endl;
            cout << "\033[1;31mFAILED: " << test_name << "\033[0m" << endl;
            cout << "\033[1;31mMismatch at index " << i << ": expected " << expected[i] << ", got " << actual[i]
                 << "\033[0m" << endl;
            cout << "\033[1;31m" << string(60, '=') << "\033[0m" << endl;
            return false;
        }
    }

    cout << "\033[1;32mPASSED: " << test_name << "\033[0m" << endl;
    return true;
}

void print_banner(std::string msg)
{
    cout << "==========================================" << endl;
    cout << msg << endl;
    cout << "==========================================" << endl;
}

/*
Helper function: Prints the parameters in a SEALContext.
*/
inline void print_parameters(const seal::SEALContext &context)
{
    auto &context_data = *context.key_context_data();

    /*
    Which scheme are we using?
    */
    std::string scheme_name;
    switch (context_data.parms().scheme())
    {
    case seal::scheme_type::bfv:
        scheme_name = "BFV";
        break;
    case seal::scheme_type::ckks:
        scheme_name = "CKKS";
        break;
    case seal::scheme_type::bgv:
        scheme_name = "BGV";
        break;
    default:
        throw std::invalid_argument("unsupported scheme");
    }
    std::cout << "/" << std::endl;
    std::cout << "| Encryption parameters :" << std::endl;
    std::cout << "|   scheme: " << scheme_name << std::endl;
    std::cout << "|   poly_modulus_degree: " << context_data.parms().poly_modulus_degree() << std::endl;

    /*
    Print the size of the true (product) coefficient modulus.
    */
    std::cout << "|   coeff_modulus size: ";
    std::cout << context_data.total_coeff_modulus_bit_count() << " (";
    auto coeff_modulus = context_data.parms().coeff_modulus();
    std::size_t coeff_modulus_size = coeff_modulus.size();
    for (std::size_t i = 0; i < coeff_modulus_size - 1; i++)
    {
        std::cout << coeff_modulus[i].bit_count() << " + ";
    }
    std::cout << coeff_modulus.back().bit_count();
    std::cout << ") bits" << std::endl;

    /*
    For the BFV scheme print the plain_modulus parameter.
    */
    if (context_data.parms().scheme() == seal::scheme_type::bfv)
    {
        std::cout << "|   plain_modulus: " << context_data.parms().plain_modulus().value() << std::endl;
    }

    std::cout << "\\" << std::endl;
}

inline void print_line(int line_number)
{
    std::cout << "Line " << std::setw(3) << line_number << " --> ";
}

template <typename T>
inline void print_matrix(std::vector<T> matrix, std::size_t row_size)
{
    /*
    We're not going to print every column of the matrix (there are 2048). Instead
    print this many slots from beginning and end of the matrix.
    */
    std::size_t print_size = 5;

    std::cout << std::endl;
    std::cout << "    [";
    for (std::size_t i = 0; i < print_size; i++)
    {
        std::cout << std::setw(3) << std::right << matrix[i] << ",";
    }
    std::cout << std::setw(3) << " ...,";
    for (std::size_t i = row_size - print_size; i < row_size; i++)
    {
        std::cout << std::setw(3) << matrix[i] << ((i != row_size - 1) ? "," : " ]\n");
    }
    std::cout << "    [";
    for (std::size_t i = row_size; i < row_size + print_size; i++)
    {
        std::cout << std::setw(3) << matrix[i] << ",";
    }
    std::cout << std::setw(3) << " ...,";
    for (std::size_t i = 2 * row_size - print_size; i < 2 * row_size; i++)
    {
        std::cout << std::setw(3) << matrix[i] << ((i != 2 * row_size - 1) ? "," : " ]\n");
    }
    std::cout << std::endl;
}

void bfv_rotation_example()
{
    print_banner("Example: BFV rotation");

    uint64_t polyModulusDegree = 8192;
    uint64_t plainWidth = 20;

    UnifiedContext context(polyModulusDegree, plainWidth, true, Datatype::DEVICE);
    print_parameters(context);
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
    cout << "    + Noise budget in fresh encryption: " << decryptor.invariant_noise_budget(encrypted_matrix) << " bits"
         << endl;
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
    cout << "    + Noise budget after rotation: " << decryptor.invariant_noise_budget(encrypted_matrix) << " bits"
         << endl;
    decryptor.decrypt(encrypted_matrix, plain_result);
    vector<uint64_t> expected(slot_count, 0ULL);
    encoder.decode(plain_result, expected);
    cout << "    + Expected result (HOST):" << endl;
    print_matrix(expected, row_size);

    print_line(__LINE__);
    cout << "Rotate rows " << step << " steps left (on DEVICE)." << endl;
    evaluator.rotate_rows_inplace(d_encrypted_matrix, step, *galoisKeys);
    d_encrypted_matrix.to_host(context);
    cout << "    + Noise budget after rotation: " << decryptor.invariant_noise_budget(d_encrypted_matrix) << " bits"
         << endl;
    decryptor.decrypt(d_encrypted_matrix, plain_result);
    vector<uint64_t> actual(slot_count, 0ULL);
    encoder.decode(plain_result, actual);
    cout << "    + Actual result (DEVICE):" << endl;
    print_matrix(actual, row_size);

    // Verify results
    verify_results(expected, actual, "BFV Rotation Test");
}

void bfv_ct_ct_mult_example()
{
    print_banner("Example: BFV ct-ct multiplication");

    uint64_t polyModulusDegree = 8192;
    uint64_t plainWidth = 20;

    UnifiedContext context(polyModulusDegree, plainWidth, true, Datatype::DEVICE);
    print_parameters(context);
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
    cout << "    + Noise budget in fresh encryption: " << decryptor.invariant_noise_budget(encrypted_matrix) << " bits"
         << endl;
    cout << endl;

    UnifiedCiphertext d_encrypted_matrix = encrypted_matrix;
    d_encrypted_matrix.to_device(context);
    cout << "chain index: " << d_encrypted_matrix.dcipher().chain_index()
         << ", coeff_size: " << d_encrypted_matrix.coeff_modulus_size() << endl;

    /*
      Now rotate both matrix rows 3 steps to the left, decrypt, decode, and print.
      */
    print_line(__LINE__);
    cout << "Square (on HOST)." << endl;
    evaluator.square_inplace(encrypted_matrix);
    evaluator.relinearize_inplace(encrypted_matrix, *relinKeys);
    Plaintext plain_result;
    cout << "    + Noise budget after squaring: " << decryptor.invariant_noise_budget(encrypted_matrix) << " bits"
         << endl;
    decryptor.decrypt(encrypted_matrix, plain_result);
    vector<uint64_t> expected(slot_count, 0ULL);
    encoder.decode(plain_result, expected);
    cout << "    + Expected result (HOST):" << endl;
    print_matrix(expected, row_size);

    print_line(__LINE__);
    cout << "Square (on DEVICE)." << endl;
    evaluator.square_inplace(d_encrypted_matrix);
    evaluator.relinearize_inplace(d_encrypted_matrix, *relinKeys);
    d_encrypted_matrix.to_host(context);
    cout << "    + Noise budget after squaring: " << decryptor.invariant_noise_budget(d_encrypted_matrix) << " bits"
         << endl;
    decryptor.decrypt(d_encrypted_matrix, plain_result);
    vector<uint64_t> actual(slot_count, 0ULL);
    encoder.decode(plain_result, actual);
    cout << "    + Actual result (DEVICE):" << endl;
    print_matrix(actual, row_size);

    // Verify results
    verify_results(expected, actual, "BFV Ciphertext-Ciphertext Multiplication Test");
}

void bfv_pt_ct_mult_example()
{
    print_banner("Example: BFV pt-ct multiplication");

    uint64_t polyModulusDegree = 8192;
    uint64_t plainWidth = 20;

    UnifiedContext context(polyModulusDegree, plainWidth, true, Datatype::DEVICE);
    print_parameters(context);
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
    UnifiedPlaintext d_plain_matrix = plain_matrix;
    d_plain_matrix.to_device(context);

    UnifiedCiphertext encrypted_matrix(HOST);
    encryptor.encrypt(plain_matrix, encrypted_matrix);
    cout << "    + Noise budget in fresh encryption: " << decryptor.invariant_noise_budget(encrypted_matrix) << " bits"
         << endl;
    cout << endl;

    UnifiedCiphertext d_encrypted_matrix = encrypted_matrix;
    d_encrypted_matrix.to_device(context);

    /*
      Now rotate both matrix rows 3 steps to the left, decrypt, decode, and print.
    */
    print_line(__LINE__);
    cout << "Plaintext-Ciphertext multiplication (on HOST)." << endl;
    evaluator.multiply_plain_inplace(encrypted_matrix, plain_matrix);
    Plaintext plain_result;
    cout << "    + Noise budget after pmult: " << decryptor.invariant_noise_budget(encrypted_matrix) << " bits" << endl;
    decryptor.decrypt(encrypted_matrix, plain_result);
    vector<uint64_t> expected(slot_count, 0ULL);
    encoder.decode(plain_result, expected);
    cout << "    + Expected result (HOST):" << endl;
    print_matrix(expected, row_size);

    print_line(__LINE__);
    cout << "Plaintext-Ciphertext multiplication (on DEVICE)." << endl;
    evaluator.multiply_plain_inplace(d_encrypted_matrix, d_plain_matrix);
    d_encrypted_matrix.to_host(context);
    cout << "    + Noise budget after pmult: " << decryptor.invariant_noise_budget(d_encrypted_matrix) << " bits"
         << endl;
    decryptor.decrypt(d_encrypted_matrix, plain_result);
    vector<uint64_t> actual(slot_count, 0ULL);
    encoder.decode(plain_result, actual);
    cout << "    + Actual result (DEVICE):" << endl;
    print_matrix(actual, row_size);

    // Verify results
    verify_results(expected, actual, "BFV Plaintext-Ciphertext Multiplication Test");
}

void bfv_pt_ct_mult_with_pre_ntt_example()
{
    print_banner("Example: BFV pt-ct multiplication with pre-ntt");

    uint64_t polyModulusDegree = 8192;
    uint64_t plainWidth = 20;

    UnifiedContext context(polyModulusDegree, plainWidth, true, Datatype::DEVICE);
    print_parameters(context);
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
    UnifiedPlaintext d_plain_matrix = plain_matrix;
    d_plain_matrix.to_device(context);

    UnifiedCiphertext encrypted_matrix(HOST);
    encryptor.encrypt(plain_matrix, encrypted_matrix);
    cout << "    + Noise budget in fresh encryption: " << decryptor.invariant_noise_budget(encrypted_matrix) << " bits"
         << endl;
    cout << endl;

    UnifiedCiphertext d_encrypted_matrix = encrypted_matrix;
    d_encrypted_matrix.to_device(context);

    // Pre-ntt
    evaluator.transform_to_ntt_inplace(d_plain_matrix, d_encrypted_matrix.dcipher().chain_index());
    evaluator.transform_to_ntt_inplace(d_encrypted_matrix);

    /*
      Now rotate both matrix rows 3 steps to the left, decrypt, decode, and print.
    */
    print_line(__LINE__);
    cout << "Plaintext-Ciphertext multiplication with pre-NTT (on HOST)." << endl;
    evaluator.multiply_plain_inplace(encrypted_matrix, plain_matrix);
    Plaintext plain_result;
    cout << "    + Noise budget after pmult: " << decryptor.invariant_noise_budget(encrypted_matrix) << " bits" << endl;
    decryptor.decrypt(encrypted_matrix, plain_result);
    vector<uint64_t> expected(slot_count, 0ULL);
    encoder.decode(plain_result, expected);
    cout << "    + Expected result (HOST):" << endl;
    print_matrix(expected, row_size);

    print_line(__LINE__);
    cout << "Plaintext-Ciphertext multiplication with pre-NTT (on DEVICE)." << endl;
    evaluator.multiply_plain_ntt_inplace(d_encrypted_matrix, d_plain_matrix);
    evaluator.transform_from_ntt_inplace(d_encrypted_matrix);
    d_encrypted_matrix.to_host(context);
    cout << "    + Noise budget after pmult: " << decryptor.invariant_noise_budget(d_encrypted_matrix) << " bits"
         << endl;
    decryptor.decrypt(d_encrypted_matrix, plain_result);
    vector<uint64_t> actual(slot_count, 0ULL);
    encoder.decode(plain_result, actual);
    cout << "    + Actual result (DEVICE):" << endl;
    print_matrix(actual, row_size);

    // Verify results
    verify_results(expected, actual, "BFV Plaintext-Ciphertext Multiplication with Pre-NTT Test");
}

// Wrapper function to run tests with exception handling
bool run_test_with_exception_handling(const string &test_name, function<void()> test_func)
{
    try
    {
        test_func();
        return true;
    }
    catch (const exception &e)
    {
        cout << "\n\033[1;31m" << string(60, '=') << "\033[0m" << endl;
        cout << "\033[1;31mEXCEPTION in " << test_name << "\033[0m" << endl;
        cout << "\033[1;31mError: " << e.what() << "\033[0m" << endl;
        cout << "\033[1;31m" << string(60, '=') << "\033[0m" << endl;
        return false;
    }
    catch (...)
    {
        cout << "\n\033[1;31m" << string(60, '=') << "\033[0m" << endl;
        cout << "\033[1;31mUNKNOWN EXCEPTION in " << test_name << "\033[0m" << endl;
        cout << "\033[1;31m" << string(60, '=') << "\033[0m" << endl;
        return false;
    }
}

int main()
{
    cout << "\033[1;36mStarting HE GPU Tests...\033[0m" << endl;
    cout << "\033[1;36m" << string(60, '=') << "\033[0m" << endl;

    int total_tests = 4;
    int passed_tests = 0;

    if (run_test_with_exception_handling("BFV Rotation Test", bfv_rotation_example))
    {
        passed_tests++;
    }

    if (run_test_with_exception_handling("BFV Ciphertext-Ciphertext Multiplication Test", bfv_ct_ct_mult_example))
    {
        passed_tests++;
    }

    if (run_test_with_exception_handling("BFV Plaintext-Ciphertext Multiplication Test", bfv_pt_ct_mult_example))
    {
        passed_tests++;
    }

    if (run_test_with_exception_handling(
            "BFV Plaintext-Ciphertext Multiplication with Pre-NTT Test", bfv_pt_ct_mult_with_pre_ntt_example))
    {
        passed_tests++;
    }

    cout << "\n\033[1;36m" << string(60, '=') << "\033[0m" << endl;
    cout << "\033[1;36mTest Summary:\033[0m" << endl;
    cout << "\033[1;36mTotal Tests: " << total_tests << "\033[0m" << endl;
    cout << "\033[1;36mPassed: " << passed_tests << "\033[0m" << endl;
    cout << "\033[1;36mFailed: " << (total_tests - passed_tests) << "\033[0m" << endl;

    if (passed_tests == total_tests)
    {
        cout << "\033[1;32mAll tests passed!\033[0m" << endl;
    }
    else
    {
        cout << "\033[1;31mSome tests failed!\033[0m" << endl;
    }
    cout << "\033[1;36m" << string(60, '=') << "\033[0m" << endl;

    return (passed_tests == total_tests) ? 0 : 1;
}
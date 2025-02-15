#include "layer/linear/MatCheetah.h"

inline std::string uint64_to_hex_string(std::uint64_t value)
{
    return seal::util::uint_to_hex_string(&value, std::size_t(1));
}

inline void print_line(int line_number)
{
    std::cout << "Line " << " " << line_number << " --> ";
}


int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <server/client> <IP> <port>\n";
        return 1;
    }

    bool is_server = std::string(argv[1]) == "server";
    const char* ip = argv[2];
    int port = std::stoi(argv[3]);
    NetIO netio(ip, port, is_server);
    std::cout << "start test" << std::endl;
    HEEvaluator HE(netio,is_server);
    std::cout << "build";
    HE.GenerateNewKey();
    std::cout << "gen";
    if (!is_server){
        int in_features = 4; // 输入特征维度
        int out_features = 4; // 输出特征维度
        Tensor<int> weightMatrix({4, 4}); // 创建权重矩阵
        Tensor<int> biasVec({4}); // 创建偏置向量
        unsigned long ni = 4, no = 4, niw = 2, now = 2;
        // // 初始化权重矩阵和偏置向量
        for (size_t i = 0; i < 4; ++i) {
            biasVec({i}) = i;
            for (size_t j = 0; j < 4; ++j) {
                weightMatrix({i, j}) = i * 4 + j;
            }
        }
        // 创建 MatCheetah 对象
        matCheetah mat(in_features, out_features, weightMatrix, &HE, biasVec, ni, no, niw, now);

        // 测试 encodeInputVector
        Tensor<int64_t> inputVector({4});
        for (size_t i = 0; i < 4; ++i) {
            inputVector({i}) = i + 1; // 简单初始化
        }

        std::cout << "Testing encodeInputVector..." << std::endl;
        Tensor<seal::Plaintext> encodedInput = mat.encodeInputVector(inputVector);
        std::cout<< "poly coeff" << *encodedInput({0}).data() << *(encodedInput({0}).data() + 1);
        std::cout<< "poly coeff" << *encodedInput({1}).data() << *(encodedInput({1}).data() + 1);

        // 测试 encodeWeightMatrix
        std::vector<std::vector<int64_t>> weightMatrixVec(4, std::vector<int64_t>(4));
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                weightMatrixVec[i][j] = i * 4 + j;
            }
        }
        mat.encodeWeightMatrix();

        // std::vector<std::vector<seal::Plaintext>> encodedWeightMatrix(4);
        // mat.encodeWeightMatrix(weightMatrixVec, encodedWeightMatrix, HE.plain);

        // // 测试 matrix_multiplication
        // std::vector<seal::Ciphertext> vec(4);       // 输入密文向量
        // std::vector<seal::Ciphertext> output(4);    // 输出密文向量
        // seal::Evaluator evaluator(nullptr);         // 需要实际初始化
        // seal::Decryptor decryptor(nullptr, nullptr); // 需要实际初始化

        // std::cout << "Testing matrix_multiplication..." << std::endl;
        // mat.matrix_multiplication(encodedWeightMatrix, vec, output, HE.plain, evaluator, decryptor);

        // std::cout << "Tests completed." << std::endl;
    }

    

    return 0;
}
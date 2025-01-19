#include "Linear.h"

class matCheetah : public Linear{
    public:
        unsigned long ni = 4;
        unsigned long no = 4;
        unsigned long niw = 2;
        unsigned long now = 2;


    matCheetah (int in_features, int out_features, Tensor<int> weightMatrix, HEEvaluator* HE, Tensor<int> biasVec, unsigned long ni, unsigned long no, unsigned long niw, unsigned long now)
        : Linear(in_features, out_features, weightMatrix, HE, biasVec){
            this->ni = ni;
            this->no = no;
            this->niw = niw;
            this->now = now;
        };

    Tensor<seal::Plaintext> encodeInputVector(
    Tensor<int64_t> input_vector)
    {
        auto niprime = ((ni + niw - 1) / niw);
        //this is ni' in paper,这里是用到了上取整
        auto noprime = ((no + now - 1) / now);
        //this is no' in paper，这里同样也是用到了上取整
        //encoded_polynomial_vector.resize(noprime);
        //make sure the polynomial vector's size is OK.
        std::cout<< "ni prime" << niprime << std::endl;
        std::cout<< "no prime" << noprime << std::endl;
        int plain = this->he->plain;
        std::vector<size_t> shape = {(ni + niw - 1) / niw}; 
        Tensor<seal::Plaintext> encoded_polynomial_vector(shape);
        for (unsigned long i = 0; i < 4; ++i) {
            std::cout << input_vector({i}) << " ";
        }
        std::cout <<std::endl;
        std::vector<uint64_t> tmp(this->he->polyModulusDegree);
        for (size_t i = 0; i < niprime; ++i) {
            std::cout<<"in encodeInputVectorLoop"<< i << std::endl;
            auto start = i * niw;
            auto end = std::min<size_t>(ni, start + niw);
            auto len = end - start;
            std::transform(input_vector.data().data()+start,input_vector.data().data() + end,tmp.begin(),[plain](uint64_t u){return u;});
            if (len < tmp.size()){
                std::fill_n(tmp.begin() + len, tmp.size() - len - 1, 0);
            }
            std::cout<<"start: " << start << "end: " << end << "len: " << len << std::endl;
            //then we have to switch the vector to the polynomial.
            //vec2poly
            seal::util::modulo_poly_coeffs(tmp, len, plain, encoded_polynomial_vector({i}).data());
            std::fill_n(encoded_polynomial_vector({i}).data() + len, encoded_polynomial_vector({i}).coeff_count() - len, 0);
        }   
        std::cout<< "poly coeff" << *encoded_polynomial_vector({0}).data() << *(encoded_polynomial_vector({0}).data() + 1);
        //std::cout<< "poly coeff" << *encoded_polynomial_vector.at(1).data() << *(encoded_polynomial_vector.at(1).data() + 1);
        return encoded_polynomial_vector;
    };

    void encodeWeightMatrix(
        const std::vector<std::vector<int64_t>> &weight_matrix, std::vector<std::vector<seal::Plaintext>> &encoded_matrix,uint64_t plain)
    {
        std::cout<< "start encode weight matrix" << std::endl;
        auto niprime = ((ni + niw - 1) / niw);
        //this is ni' in paper
        auto noprime = ((no + now - 1) / now);
        //notice the size of the matrix is no * ni;
        std::vector<uint64_t> tmp(this->he->polyModulusDegree);
        std::cout << "noprime" << noprime << std::endl;
        std::cout << "niprime" << niprime << std::endl;
        for (size_t r_blk = 0; r_blk < noprime; ++r_blk){
            encoded_matrix[r_blk].resize(niprime);
            //它的行确定下来了以后再resize它的列。
            auto up_row = r_blk * now;
        //   起点的行位置数
            auto bottom_row = std::min<size_t>(up_row + now, no);
        //     //末尾的行数，min是考虑不整除的情况防止溢出。
            auto row_extent = bottom_row - up_row;
            //行个数
            for (size_t c_blk = 0; c_blk < niprime; ++c_blk) {
            //有了行和列的编号，我们可以定位到一个具体的block number。
                auto left_col = c_blk * niw;
                auto right_col = std::min<size_t>(left_col + niw, ni);
                //这个对应会原来矩阵的行列数
                auto col_extent = right_col - left_col;
                //这个对应于行个数
                //Encode the sub-matrix start ad (top_left_row, top_left_col) with
                //确定元素的位置。
                for (size_t r = 0; r < row_extent; ++r) {
                    size_t nzero_pad = niw - col_extent;
                    if (nzero_pad > 0) {
                        for (size_t c = 0; c < nzero_pad; ++c){
                            tmp[r * niw + c] = 0;
                        }
                        for (size_t c = 0; c < col_extent; ++c){
                            tmp[nzero_pad + r * niw + c] = weight_matrix[up_row + r][left_col + col_extent - 1 - c];
                            // For the right-most submatrtix, we might need zero-padding.
                        }
                    }else{
                        for (size_t c = 0; c < col_extent; ++c){
                            tmp[r * niw + c] = weight_matrix[up_row + r][left_col + col_extent - 1 - c];
                            // For the right-most submatrtix, we might need zero-padding.
                        }
                    }
                // zero-out the other coefficients
                }
                std::cout << std::endl;
                std::fill(tmp.begin() + row_extent * niw, tmp.end(), 0);
                std::cout << r_blk << " " << c_blk << " " << col_extent << " " << tmp[2] << std::endl;
                for (size_t i = 0; i < now * niw; i++){
                    std::cout << tmp[i] <<" ";
                }
                seal::util::modulo_poly_coeffs(tmp, tmp.size(), plain, encoded_matrix.at(r_blk).at(c_blk).data());
            }
        }

    }

    int div_upper(
        int a,
        int b
    ){
        return ((a + b - 1) / b);
    }

    void matrix_multiplication(
        const std::vector<std::vector<seal::Plaintext>> &weight_matrix,
        const std::vector<seal::Ciphertext> &vec,
        std::vector<seal::Ciphertext> &output,
        const uint64_t plain,
        const Evaluator &evaluate,
        Decryptor &decry 
    ){
        auto niprime = ((ni + niw - 1) / niw);
        //this is ni' in paper
        auto noprime = ((no + now - 1) / now);
        std::vector<std::vector<seal::Ciphertext>> multiple_inter(noprime,std::vector<seal::Ciphertext>(niprime));
        seal::Plaintext decrypt_in;
        for (size_t r_blk = 0; r_blk < noprime; r_blk++){
            for (size_t c_blk = 0; c_blk < niprime; c_blk++){
                evaluate.multiply_plain(vec[c_blk],weight_matrix[r_blk][c_blk],multiple_inter[r_blk][c_blk]);
                decry.decrypt(multiple_inter[r_blk][c_blk],decrypt_in);
                std::cout << "r and c" << r_blk << " " << c_blk;
                std::cout << *(decrypt_in.data()) << " " << *(decrypt_in.data() + 1) << " " << *(decrypt_in.data() + 2) << " " << *(decrypt_in.data() + 3) <<std::endl; 
            }
        }

        for (size_t r_blk = 0; r_blk < noprime; r_blk++){
            evaluate.add_many(multiple_inter[r_blk],output[r_blk]);
        }


    }
        
};
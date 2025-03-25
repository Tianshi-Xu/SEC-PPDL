#include <LinearOperator/Polynomial.h>
#include <LinearOperator/Conversion.h>

namespace LinearOperator {

// input and output are both secret shares, also supports square when x==y, the input can be any shape
// TODO: support x.size() % HE->polyModulusDegree != 0
Tensor<uint64_t> ElementWiseMul(Tensor<uint64_t> &x, Tensor<uint64_t> &y, HE::HEEvaluator* HE){
    if (x.size() != y.size()) {
        throw std::invalid_argument("x and y must have the same size in ElementWiseMul");
    }
    auto shape = x.shape();
    x.reshape({x.size()/HE->polyModulusDegree, HE->polyModulusDegree});
    Tensor<HE::unified::UnifiedCiphertext> x_ct = Operator::SSToHE(x, HE);
    Tensor<HE::unified::UnifiedCiphertext> z(x_ct.shape(), HE->GenerateZeroCiphertext(HE->Backend()));
    x.reshape(shape);
    if(&x==&y){
        // cout << "x==y" << endl;
        for(size_t i = 0; i < x_ct.size(); i++){
            HE->evaluator->square(x_ct(i), z(i));
        }
    }else{
        y.reshape({y.size()/HE->polyModulusDegree, HE->polyModulusDegree});
        Tensor<HE::unified::UnifiedCiphertext> y_ct = Operator::SSToHE(y, HE);
        for(size_t i = 0; i < x_ct.size(); i++){
            HE->evaluator->multiply(x_ct(i), y_ct(i), z(i));
        }
        y.reshape(shape);
    }
    Tensor<uint64_t> z_ss = Operator::HEToSS(z, HE);
    z_ss.reshape(shape);
    return z_ss;
}


} // namespace LinearOperator
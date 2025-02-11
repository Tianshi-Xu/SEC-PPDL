#include "LinearLayer/Conv.h"

using namespace seal;
using namespace LinearLayer;

// 计算上取整除法
int Conv2DCheetah::DivUpper(int a, int b) {
    return ((a + b - 1) / b);
}

// 计算计算开销
int Conv2DCheetah::CalculateCost(int H, int W, int h, int Hw, int Ww, int C, int N) {
    return (int)ceil((double)C / (N / (Hw * Ww))) *
           (int)ceil((double)(H - h + 1) / (Hw - h + 1)) *
           (int)ceil((double)(W - h + 1) / (Ww - h + 1));
}

// 查找最佳分块方式
void Conv2DCheetah::FindOptimalPartition(int H, int W, int h, int C, int N, int* optimalHw, int* optimalWw) {
    int min_cost = (1 << 30);
    for (int Hw = h; Hw <= H; Hw++) {
        for (int Ww = h; Ww <= W; Ww++) {
            if (Hw * Ww > N) continue;
            int cost = CalculateCost(H, W, h, Hw, Ww, C, N);
            if (cost < min_cost) {
                min_cost = cost;
                *optimalHw = Hw;
                *optimalWw = Ww;
            }
        }
    }
}

Conv2DCheetah::Conv2DCheetah (size_t inputHeight, size_t inputWeight, HEEvaluator* he, Tensor<int64_t> kernel){
    this->he = he;
    std::vector<size_t> shape = kernel.shape();
    C = shape[1];
    M = shape[0];
    h = shape[2];
    //const std::vector<size_t>& inputTensorShape = inputTensor.shape();
    H = inputHeight;
    W = inputWeight;
    int optimalHw = H, optimalWw = W;
    //FindOptimalPartition(H, W, h, C, N, &optimalHw, &optimalWw);
    HW = 8;
    WW = 8;
    CW = 16;
    MW = 8;
    dM = DivUpper(M,MW);
    dC = DivUpper(C,CW);
    dH = DivUpper(H - h + 1 , HW - h + 1);
    dW = DivUpper(W - h + 1 , WW - h + 1);
    OW = HW * WW * (MW * CW - 1) + WW * (h - 1) + h - 1;
    // Hprime = (H - h + s) / s;
    // Wprime = (W - h + s) / s;
    // HWprime = (HW - h + s) / s;
    // WWprime = (WW - h + s) / s;
    std::cout << "in Conv2DCheetah:" << he->polyModulusDegree << std::endl;
    polyModulusDegree = he->polyModulusDegree;
    std::cout << "in Conv2DCheetah:" << polyModulusDegree << std::endl;
    plain = he->plain_mod;
    std::cout << "setup success" << std::endl;

    };

// 加密张量
Tensor<seal::Ciphertext> Conv2DCheetah::EncryptTensor(Tensor<seal::Plaintext> plainTensor) {
    std::vector<size_t> shapeTab = {dC, dH, dW};
    Tensor<seal::Ciphertext> TalphabetaCipher(shapeTab, he->GenerateZeroCiphertext());

    for (unsigned long gama = 0; gama < dC; gama++) {
        for (unsigned long alpha = 0; alpha < dH; alpha++) {
            for (unsigned long beta = 0; beta < dW; beta++) {
                he->encryptor->encrypt(plainTensor({gama, alpha, beta}), TalphabetaCipher({gama, alpha, beta}));
            }
        }
    }
    return TalphabetaCipher;
}

// 计算输入张量的 Pack 版本
Tensor<seal::Plaintext> Conv2DCheetah::PackTensor(Tensor<int64_t> x) {
    std::vector<size_t> shapeTab = {dC, dH, dW};
    Tensor<seal::Plaintext> Talphabeta(shapeTab);
    int len = CW * HW * WW;
    Tensor<uint64_t> Tsub ({CW, HW, WW});
    for (unsigned long gama = 0; gama < dC; gama++){
        for (unsigned long alpha = 0; alpha < dH; alpha++){
            for (unsigned long beta = 0; beta < dW; beta++){
                
                //traverse 
                for (unsigned long ic = 0; ic < CW; ic++){
                    if ((ic + gama * CW) >= C){
                        for (unsigned long jh = 0; jh < HW; jh++){
                            for (unsigned long kw = 0; kw < WW; kw++){
                                Tsub({ic,jh,kw}) = 0;
                            }
                        }
                        //对于超出的channel部分应该设置为0
                    }
                    else{
                        for (unsigned long jh = 0; jh < HW; jh++){
                            if ((jh + alpha * (HW - h + 1)) >= H){
                                for (unsigned long kw = 0; kw < WW; kw++){
                                    Tsub({ic,jh,kw}) = 0;
                                }
                                //超出的HW部分应该为0
                            }
                            else{
                                for (unsigned long kw = 0; kw <WW; kw++){
                                    if ((kw + beta * (WW - h + 1)) >= W){
                                        Tsub({ic,jh,kw}) = 0;
                                    }
                                    else{
                                        int64_t element = x({gama * CW + ic, alpha * (HW - h + 1) + jh, beta * (WW - h + 1) + kw});
                                        Tsub({ic,jh,kw}) = (element >= 0) ? unsigned(element) : unsigned(element + plain);
                                    }
                                }
                            }
                        }
                    }
                }
                Tensor<uint64_t> Tsubflatten = Tsub;
                Tsubflatten.flatten();
                vector<uint64_t> Tsubv = Tsubflatten.data(); 
                std::cout << "gama alpha beta:" << gama << " " << alpha << " " << beta << std::endl;
                std::cout << "poly:" << polyModulusDegree << std::endl;
                Talphabeta({gama,alpha,beta}).resize(polyModulusDegree);
                seal::util::modulo_poly_coeffs(Tsubv, len, plain, Talphabeta({gama,alpha,beta}).data());
                std::fill_n(Talphabeta({gama,alpha,beta}).data() + len, polyModulusDegree - len, 0);
                //encryptor.encrypt(Talphabeta({gama,alpha,beta}),TalphabetaCipher({gama,alpha,beta}));
                std::cout << len << " " << plain << " ";
            }
        }
    }
    return Talphabeta;
}

// 计算卷积核的 Pack 版本
Tensor<seal::Plaintext> Conv2DCheetah::PackKernel(Tensor<int64_t> x) {
    std::vector<size_t> shapeTab = {dM, dC};
    Tensor<seal::Plaintext> Ktg(shapeTab);
    size_t len = OW + 1;
    std::cout << dM << " " << dC << " " << MW <<" " << CW << std::endl;


    for (unsigned long theta = 0; theta < dM; theta++){
        for (unsigned long gama = 0; gama < dC; gama++){

            vector<uint64_t> Tsubv (polyModulusDegree,0); 
            for (unsigned long it = 0; it < MW; it++){
                for (unsigned long jg = 0; jg < CW; jg++){
                    if (((theta * MW + it) >= M) || ((gama * CW + jg) >= C)){
                        for (unsigned hr = 0; hr < h; hr++){
                            for (unsigned hc = 0; hc < h; hc++){
                                //Ksub({it,jg,hr,hc}) = 0;
                                Tsubv[OW - it * CW * HW * WW - jg * HW * WW - hr * WW - hc] = 0;
                            }
                        }
                        std::cout << "execute expand! theta MW it" << theta << " " << MW << " " << it <<std::endl;
                        std::cout << "gama CW jg" << gama << " " << CW << " " << jg << std::endl;
                    }else{
                        for (unsigned hr = 0; hr < h; hr++){
                            for (unsigned hc = 0; hc < h; hc++){
                                int64_t element = x({theta * MW + it, gama * CW + jg, hr, hc});
                                std::cout << "element theta MW it" << element << " " << theta << " " << MW << " " << it << std::endl;
                                std::cout << "gama CW jg" << gama << " " << CW << " " << jg << std::endl;
                                // Ksub({it,jg,hr,hc}) = (element >= 0) ? unsigned(element) : unsigned(element + plain);
                                // Tsubv[OW - it * CW * HW * WW - jg * HW * WW - hr * WW - hc] = Ksub({it,jg,hr,hc});
                                Tsubv[OW - it * CW * HW * WW - jg * HW * WW - hr * WW - hc] = (element >= 0) ? unsigned(element) : unsigned(element + plain);
                            }
                        }
                    }
                }
            }
            std::cout << "theta:" << theta << "gama:" << gama << std::endl;
            for (size_t check = 0; check <= OW; check++){
                std::cout << Tsubv[check] << " ";
            }
            std::cout << std::endl;
            Ktg({theta,gama}).resize(polyModulusDegree);
            //batch_encoder.encode(vecIni, Talphabeta[gama][alpha][beta]);
            seal::util::modulo_poly_coeffs(Tsubv, len, plain, Ktg({theta, gama}).data());
            for (size_t check = 0; check <= OW; check++){
                std::cout << Ktg({theta,gama}).data()[check] << " ";
            }
            if (len < polyModulusDegree){
                std::fill_n(Ktg({theta,gama}).data() + len, polyModulusDegree - len, 0);
            }
        }
    }

    std::cout << "finish encode";

    return Ktg;
}

// 计算同态卷积
Tensor<seal::Ciphertext> Conv2DCheetah::Conv(Tensor<seal::Ciphertext> T, Tensor<seal::Plaintext> K) {
    std::vector<size_t> shapeTab = {dM, dH, dW};
    Tensor<seal::Ciphertext> ConvRe(shapeTab,he->GenerateZeroCiphertext());
    seal::Ciphertext interm;

    for (size_t theta = 0; theta < dM; theta++) {
        for (size_t alpha = 0; alpha < dH; alpha++) {
            for (size_t beta = 0; beta < dW; beta++) {
                he->evaluator->multiply_plain(T({0, alpha, beta}), K({theta, 0}), ConvRe({theta, alpha, beta}));
                for (size_t gama = 1; gama < dC; gama++) {
                    he->evaluator->multiply_plain(T({gama, alpha, beta}), K({theta, gama}), interm);
                    he->evaluator->add_inplace(ConvRe({theta, alpha, beta}), interm);
                }
            }
        }
    }
    return ConvRe;
}

Tensor<int64_t> Conv2DCheetah::ExtractResult(Tensor<Plaintext> ConvResultPlain){
    Tensor<int64_t> finalResult ({M, Hprime, Wprime});
    int checkl = 0;

    for (size_t cprime = 0; cprime < M; cprime++){
        for (size_t iprime = 0; iprime < Hprime; iprime++){
            for (size_t jprime = 0; jprime < Wprime; jprime++){
                size_t c = cprime % MW;
                size_t i = (iprime * s) % (HW - h + 1);
                size_t j = (jprime * s) % (WW - h + 1);
                size_t theta = cprime / MW;
                size_t alpha = (iprime * s) / (HW - h + 1);
                size_t beta = (jprime * s) / (WW - h + 1);
                size_t des = OW - c * CW * HW * WW + i  * WW + j;
                auto interm = *(ConvResultPlain({theta, alpha, beta}).data() + des);
                interm = (interm > plain / 2) ? (interm - plain) : interm;
                finalResult({cprime, iprime, jprime}) = interm;
            }
        }
    }
    return finalResult;
}

 // namespace LinearLayer
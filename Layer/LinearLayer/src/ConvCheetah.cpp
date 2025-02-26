#include <LinearLayer/Conv.h>
#include <seal/util/polyarithsmallmod.h>
#include <algorithm>


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

Conv2DCheetah::Conv2DCheetah (size_t inputHeight, size_t inputWeight, HEEvaluator* he, const Tensor<uint64_t>& kernel, size_t stride, const Tensor<uint64_t>& bias, uint64_t padding)
    : Conv2D(inputHeight, stride, padding, kernel, bias, he)
{
    this->he = he;
    std::vector<size_t> shape = kernel.shape();
    C = shape[1];
    M = shape[0];
    h = shape[2];
    polyModulusDegree = he->polyModulusDegree;
    this->padding = padding;
    H = inputHeight + 2 * padding;
    W = inputWeight + 2 * padding;
    s = stride;
    int optimalHw = H, optimalWw = W;
    FindOptimalPartition(H, W, h, C, polyModulusDegree, &optimalHw, &optimalWw);
    HW = optimalHw;
    WW = optimalWw;
    CW = min(C, (polyModulusDegree / (HW * WW)));
    MW = min(M, (polyModulusDegree / (CW * HW * WW)));
    dM = DivUpper(M,MW);
    dC = DivUpper(C,CW);
    dH = DivUpper(H - h + 1 , HW - h + 1);
    dW = DivUpper(W - h + 1 , WW - h + 1);
    OW = HW * WW * (MW * CW - 1) + WW * (h - 1) + h - 1;
    Hprime = (H - h + s) / s;
    Wprime = (W - h + s) / s;
    HWprime = (HW - h + s) / s;
    WWprime = (WW - h + s) / s;
    polyModulusDegree = he->polyModulusDegree;
    plain = he->plain_mod;
    weight_pt = this->PackWeight();

};


Conv2DCheetah::Conv2DCheetah (size_t Height, size_t Width, HEEvaluator* he, const Tensor<uint64_t>& kernel, 
                  size_t stride, const Tensor<uint64_t>& bias, uint64_t padding, Tensor<uint64_t> *gamma, Tensor<uint64_t> *beta)
    : Conv2D(Height, stride, padding, kernel, bias, he)
{
    this->fused_bn = true;
    this->he = he;
    std::vector<size_t> shape = kernel.shape();
    C = shape[1];
    M = shape[0];
    h = shape[2];
    Tensor<uint64_t> kernelFuse({M, C, h, h}, 0);
    for (size_t i = 0; i < C; i++){
        for (size_t j = 0; j < M; j++){
            for (size_t k = 0; k < h; k++){
                for (size_t l = 0; l < h; l++){
                    kernelFuse({i, j, k, l}) = kernel({i, j, k, l}) * (*gamma)({j});
                }
            }
        }
    }

    this->weight = kernelFuse;


    polyModulusDegree = he->polyModulusDegree;
    this->padding = padding;
    H = Height + 2 * padding;
    W = Width + 2 * padding;
    s = stride;
    int optimalHw = H, optimalWw = W;
    FindOptimalPartition(H, W, h, C, polyModulusDegree, &optimalHw, &optimalWw);
    HW = optimalHw;
    WW = optimalWw;
    CW = min(C, (polyModulusDegree / (HW * WW)));
    MW = min(M, (polyModulusDegree / (CW * HW * WW)));
    dM = DivUpper(M,MW);
    dC = DivUpper(C,CW);
    dH = DivUpper(H - h + 1 , HW - h + 1);
    dW = DivUpper(W - h + 1 , WW - h + 1);
    OW = HW * WW * (MW * CW - 1) + WW * (h - 1) + h - 1;
    Hprime = (H - h + s) / s;
    Wprime = (W - h + s) / s;

    Tensor<uint64_t> biasFuse({M, Hprime, Wprime}, 0);

    for (size_t i = 0; i < M; i++){
        for (size_t j = 0; j < Hprime; j++){
            for (size_t k = 0; k < Wprime; k++){
                biasFuse({i, j, k}) = bias({i, j, k}) * (*gamma)({j}) + (*beta)({j});
            }
        }
    }
    this->bias = biasFuse;

    HWprime = (HW - h + s) / s;
    WWprime = (WW - h + s) / s;
    polyModulusDegree = he->polyModulusDegree;
    plain = he->plain_mod;
    weight_pt = this->PackWeight();

};



// 加密张量
Tensor<UnifiedCiphertext> Conv2DCheetah::EncryptTensor(Tensor<UnifiedPlaintext> plainTensor) {
    std::vector<size_t> shapeTab = {dC, dH, dW};
    Tensor<UnifiedCiphertext> TalphabetaCipher(shapeTab, he->GenerateZeroCiphertext());
    for (unsigned long gama = 0; gama < dC; gama++) {
        for (unsigned long alpha = 0; alpha < dH; alpha++) {
            for (unsigned long beta = 0; beta < dW; beta++) {
                he->encryptor->encrypt(plainTensor({gama, alpha, beta}), TalphabetaCipher({gama, alpha, beta}));
            }
        }
    }
    return TalphabetaCipher;
}

Tensor<uint64_t> Conv2DCheetah::HETOTensor (Tensor<UnifiedCiphertext> inputCipher){
    if (he->server) {
        //add mask 
        //send ciphertext
        Tensor<UnifiedCiphertext> cipherMask({dM, dH, dW},he->GenerateZeroCiphertext());
        Tensor<UnifiedPlaintext> plainMask({dM, dH, dW},HOST);
        Tensor<uint64_t> tensorMask({dM, dH, dW, polyModulusDegree}, 0);
        UnifiedPlaintext plainMaskInv(HOST);
        int64_t mask;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int64_t> dist(0, plain - 1);


        for (size_t i = 0; i < dM; i++){
            for (size_t j = 0; j < dH; j++){
                for (size_t k = 0; k < dW; k++){
                    plainMask({i,j,k}).hplain().resize(polyModulusDegree);
                    plainMaskInv.hplain().resize(polyModulusDegree);
                    for (size_t l = 0; l < polyModulusDegree; l++){
                        mask = dist(gen);
                        *(plainMask({i,j,k}).hplain().data() + l) = mask;
                        tensorMask({i, j, k, l}) = mask;
                        mask = plain - mask;
                        *(plainMaskInv.hplain().data() + l) = mask;   
                    }
                    he->evaluator->add_plain(inputCipher({i,j,k}), plainMaskInv, cipherMask({i,j,k}));
                }
            }
        }
        cipherMask.flatten();
        he->SendEncVec(cipherMask);
        return tensorMask;

    }else{
        //receive ciphertext and decry.
        Tensor<UnifiedCiphertext> cipherMask({dM * dH * dW},he->GenerateZeroCiphertext());
        he->ReceiveEncVec(cipherMask);
        Tensor<UnifiedPlaintext>  plainMask({dM, dH, dW},HOST);
        Tensor<uint64_t> tensorMask({dM, dH, dW, polyModulusDegree}, 0);
        for (size_t i = 0; i < dM; i++){
            for (size_t j = 0; j < dH; j++){
                for (size_t k = 0; k < dW; k++){
                    he->decryptor->decrypt(cipherMask({i * dH * dW + j * dW + k}), plainMask({i, j, k}));
                    for (size_t l = 0; l < polyModulusDegree; l++){
                        tensorMask({i, j, k, l}) = *(plainMask({i,j,k}).hplain().data() + l);
                    }
                }
            }
        }
        return tensorMask;
    }
}

// 计算输入张量的 Pack 版本
Tensor<uint64_t> Conv2DCheetah::PackActivation(Tensor<uint64_t> x){
    Tensor<uint64_t> padded_x ({C, H, W} ,0);
    for (size_t i = 0; i < C; i++){
        for (size_t j = 0; j < (H - 2 * padding); j++){
            for (size_t k = 0; k < (W - 2 * padding); k++){
                padded_x({i, j + padding, k + padding}) = x({i, j, k});
            }
        }
    }
    size_t len = CW * HW * WW;
    Tensor<uint64_t> Tsub ({CW, HW, WW});
    Tensor<uint64_t> PackActivationTensor({dC, dH, dW, len},0);
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
                                        int64_t element = padded_x({gama * CW + ic, alpha * (HW - h + 1) + jh, beta * (WW - h + 1) + kw});
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
                for (size_t i = 0; i < len; i++){
                    PackActivationTensor({gama, alpha, beta, i}) = Tsubv[i];
                }
            }
        }
    }
    return PackActivationTensor;
}

Tensor<UnifiedCiphertext> Conv2DCheetah::TensorTOHE(Tensor<uint64_t> PackActivationTensor) {
    std::vector<size_t> shapeTab = {dC, dH, dW};
    Tensor<UnifiedPlaintext> Talphabeta(shapeTab,Datatype::HOST);
    size_t len = CW * HW * WW;
    for (unsigned long gama = 0; gama < dC; gama++){
        for (unsigned long alpha = 0; alpha < dH; alpha++){
            for (unsigned long beta = 0; beta < dW; beta++){
                //traverse 
                vector<uint64_t> Tsubv(len, 0); 
                for (size_t i = 0; i < len; i++){
                    Tsubv[i] = PackActivationTensor({gama, alpha, beta, i});
                }
                Talphabeta({gama,alpha,beta}).hplain().resize(polyModulusDegree);
                seal::util::modulo_poly_coeffs(Tsubv, len, plain, Talphabeta({gama,alpha,beta}).hplain().data());
                std::fill_n(Talphabeta({gama,alpha,beta}).hplain().data() + len, polyModulusDegree - len, 0);
            }
        }
    }
    Tensor<UnifiedCiphertext> finalpack({dC, dH, dW}, he->GenerateZeroCiphertext());
    if (!he->server){
        //客服端
        Tensor<UnifiedCiphertext> enc({dC, dH, dW}, he->GenerateZeroCiphertext());
        enc = this->EncryptTensor(Talphabeta);
        enc.flatten();
        he->SendEncVec(enc);
    }else{
        //服务器端
        Tensor<UnifiedCiphertext> encflatten({dC * dH * dW}, this->he->GenerateZeroCiphertext());
        he->ReceiveEncVec(encflatten);
        Tensor<UnifiedCiphertext> enc({dC, dH, dW}, he->GenerateZeroCiphertext());
        for (size_t i = 0; i < dC; i++){
            for (size_t j = 0; j < dH; j++){
                for (size_t k = 0; k < dW; k++){
                    enc({i,j,k}) = encflatten({i * dH * dW + j * dW + k});
                }
            }
        }
        finalpack = this->sumCP(enc,Talphabeta);
    }
    return finalpack;
}


// 计算卷积核的 Pack 版本
Tensor<UnifiedPlaintext> Conv2DCheetah::PackWeight() {
    std::vector<size_t> shapeTab = {dM, dC};
    Tensor<UnifiedPlaintext> Ktg(shapeTab,Datatype::HOST);
    size_t len = OW + 1;
    if (!he->server){
        return Ktg;
    }

    for (unsigned long theta = 0; theta < dM; theta++){
        for (unsigned long gama = 0; gama < dC; gama++){

            vector<uint64_t> Tsubv (polyModulusDegree,0); 
            for (unsigned long it = 0; it < MW; it++){
                for (unsigned long jg = 0; jg < CW; jg++){
                    if (((theta * MW + it) >= M) || ((gama * CW + jg) >= C)){
                        for (unsigned hr = 0; hr < h; hr++){
                            for (unsigned hc = 0; hc < h; hc++){
                                Tsubv[OW - it * CW * HW * WW - jg * HW * WW - hr * WW - hc] = 0;
                            }
                        }
                    }else{
                        for (unsigned hr = 0; hr < h; hr++){
                            for (unsigned hc = 0; hc < h; hc++){
                                int64_t element = this->weight({theta * MW + it, gama * CW + jg, hr, hc});
                                Tsubv[OW - it * CW * HW * WW - jg * HW * WW - hr * WW - hc] = (element >= 0) ? unsigned(element) : unsigned(element + plain);
                            }
                        }
                    }
                }
            }
            Ktg({theta,gama}).hplain().resize(polyModulusDegree);
            seal::util::modulo_poly_coeffs(Tsubv, len, plain, Ktg({theta, gama}).hplain().data());
            if (len < polyModulusDegree){
                std::fill_n(Ktg({theta,gama}).hplain().data() + len, polyModulusDegree - len, 0);
            }
        }
    }


    return Ktg;
}
Tensor<UnifiedCiphertext> Conv2DCheetah::sumCP(Tensor<UnifiedCiphertext> cipherTensor, Tensor<UnifiedPlaintext> plainTensor){
    Tensor<UnifiedCiphertext> Talphabeta({dC, dH, dW}, HOST);
    for (size_t gama = 0; gama < dC; gama++){
        for (size_t alpha = 0; alpha < dH; alpha++){
            for (size_t beta = 0; beta < dW; beta++){
                he->evaluator->add_plain(cipherTensor({gama,alpha,beta}), plainTensor({gama,alpha,beta}), Talphabeta({gama,alpha,beta}));
            }
        }
    }
    return Talphabeta;
}
   

// 计算同态卷积
Tensor<UnifiedCiphertext> Conv2DCheetah::HECompute(Tensor<UnifiedPlaintext> weight_pt, Tensor<UnifiedCiphertext> ac_ct)
{
//Tensor<UnifiedCiphertext> Conv2DCheetah::ConvCP(Tensor<UnifiedCiphertext> T, Tensor<UnifiedPlaintext> K) {
    std::vector<size_t> shapeTab = {dM, dH, dW};
    Tensor<UnifiedCiphertext> ConvRe(shapeTab,he->GenerateZeroCiphertext());
    UnifiedCiphertext interm = he->GenerateZeroCiphertext();
    if (!he->server){
        return ConvRe;
    }

    for (size_t theta = 0; theta < dM; theta++) {
        for (size_t alpha = 0; alpha < dH; alpha++) {
            for (size_t beta = 0; beta < dW; beta++) {
                he->evaluator->multiply_plain(ac_ct({0, alpha, beta}), weight_pt({theta, 0}), ConvRe({theta, alpha, beta}));
                for (size_t gama = 1; gama < dC; gama++) {
                    he->evaluator->multiply_plain(ac_ct({gama, alpha, beta}), weight_pt({theta, gama}), interm);
                    he->evaluator->add_inplace(ConvRe({theta, alpha, beta}), interm);
                }
            }
        }
    }
    return ConvRe;
}

Tensor<uint64_t> Conv2DCheetah::DepackResult(Tensor<uint64_t> out){
    Tensor<uint64_t> finalResult ({M, Hprime, Wprime});
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
                finalResult({cprime, iprime, jprime}) = out({theta, alpha, beta, des});
            }
        }
    }
    return finalResult;

}

Tensor<uint64_t> Conv2DCheetah::operator()(Tensor<uint64_t> x){
    std::cout << "1";
    auto pack = this->PackActivation(x);
    std::cout << "2";
    auto Cipher = this->TensorTOHE(pack);
    std::cout << "3";
    auto ConvResult = this->HECompute(weight_pt, Cipher);
    std::cout << "4";
    auto share = this->HETOTensor(ConvResult);
    std::cout << "5";
    auto finalR = this->DepackResult(share);
    std::cout << "6";
    return finalR;
}

 // namespace LinearLayer

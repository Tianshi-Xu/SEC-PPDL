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


void Conv2DCheetah::compute_he_params(uint64_t in_feature_size) {
    this->in_feature_size = this->in_feature_size + 2 * padding;
    int optimalHw = this->in_feature_size, optimalWw = this->in_feature_size;
    FindOptimalPartition(this->in_feature_size, this->in_feature_size, kernel_size, in_channels, polyModulusDegree, &optimalHw, &optimalWw);
    HW = optimalHw;
    WW = optimalWw;
    CW = min(in_channels, (polyModulusDegree / (HW * WW)));
    MW = min(out_channels, (polyModulusDegree / (CW * HW * WW)));
    dM = DivUpper(out_channels,MW);
    dC = DivUpper(in_channels,CW);
    dH = DivUpper(this->in_feature_size - kernel_size + 1 , HW - kernel_size + 1);
    dW = DivUpper(this->in_feature_size - kernel_size + 1 , WW - kernel_size + 1);
    OW = HW * WW * (MW * CW - 1) + WW * (kernel_size - 1) + kernel_size - 1;
    HOut = (this->in_feature_size - kernel_size + stride) / stride;
    WOut = (this->in_feature_size - kernel_size + stride) / stride;
    HWprime = (HW - kernel_size + stride) / stride;
    WWprime = (WW - kernel_size + stride) / stride;
    this->fused_bn = false;
}


Conv2DCheetah::Conv2DCheetah(uint64_t in_feature_size, uint64_t stride, uint64_t padding, const Tensor<uint64_t>& weight, const Tensor<uint64_t>& bias, HE::HEEvaluator* HE)
    : Conv2D(in_feature_size + 2 * padding, stride, padding, weight, bias, HE)
{
    std::vector<size_t> shape = weight.shape();
    in_channels = shape[1];
    out_channels = shape[0];
    kernel_size = shape[2];
    polyModulusDegree = HE->polyModulusDegree;
    this->padding = padding;
    //in_feature_size = in_feature_size + 2 * padding;
    int optimalHw = this->in_feature_size, optimalWw = this->in_feature_size;
    cout << "in_feature_size:" << this->in_feature_size << endl;
    FindOptimalPartition(this->in_feature_size, this->in_feature_size, kernel_size, in_channels, polyModulusDegree, &optimalHw, &optimalWw);
    cout << "optimalHw:" << optimalHw << endl;
    HW = optimalHw;
    WW = optimalWw;
    CW = min(in_channels, (polyModulusDegree / (HW * WW)));
    MW = min(out_channels, (polyModulusDegree / (CW * HW * WW)));
    dM = DivUpper(out_channels,MW);
    dC = DivUpper(in_channels,CW);
    cout << "dC,dM:" << dC << "," << dM << endl;
    dH = DivUpper(this->in_feature_size - kernel_size + 1 , HW - kernel_size + 1);
    dW = DivUpper(this->in_feature_size - kernel_size + 1 , WW - kernel_size + 1);
    OW = HW * WW * (MW * CW - 1) + WW * (kernel_size - 1) + kernel_size - 1;
    HOut = (this->in_feature_size - kernel_size + stride) / stride;
    WOut = (this->in_feature_size - kernel_size + stride) / stride;
    HWprime = (HW - kernel_size + stride) / stride;
    WWprime = (WW - kernel_size + stride) / stride;
    polyModulusDegree = HE->polyModulusDegree;
    plain = HE->plain_mod;
    // std::cout << "plain" << plain;
    weight_pt = this->PackWeight();
    this->fused_bn = false;
    cout << "feature_size:" << this->in_feature_size << endl;
    cout << "Hprime:" << HOut << endl;
    cout << "Wprime:" << WOut << endl;
    cout << "HWPrime:" << HWprime << endl;
    cout << "WWprime:" << WWprime << endl;
    cout << "Conv2DCheetah constructor done" << endl;
};

Conv2DCheetah::Conv2DCheetah(uint64_t in_feature_size, uint64_t in_channels, uint64_t out_channels, uint64_t kernel_size, uint64_t stride, HE::HEEvaluator* HE)
    : Conv2D(in_feature_size, in_channels, out_channels, kernel_size, stride, HE)
{
    polyModulusDegree = HE->polyModulusDegree;
    plain = HE->plain_mod;
    compute_he_params(in_feature_size);
    // this->weight.print_shape();
    if(HE->server) {
        weight_pt = PackWeight();
    }
    cout << "padding:" << padding << endl;
    // weight_pt.print_shape();
}

void Conv2DCheetah::fuse_bn(Tensor<uint64_t> *gamma, Tensor<uint64_t> *beta){
    Tensor<uint64_t> kernelFuse({out_channels, in_channels, kernel_size, kernel_size}, 0);
    for (size_t i = 0; i < out_channels; i++){
        for (size_t j = 0; j < in_channels; j++){
            for (size_t k = 0; k < kernel_size; k++){
                for (size_t l = 0; l < kernel_size; l++){
                    kernelFuse({i, j, k, l}) = this->weight({i, j, k, l}) * (*gamma)({i});
                }
            }
        }
    }
    this->weight = kernelFuse;
    Tensor<uint64_t> biasFuse({out_channels, HOut, WOut}, 0);
    std::cout << "Hprime:" << HOut << std::endl;
    std::cout << "Wprime:" << WOut << std::endl; 

    for (size_t i = 0; i < out_channels; i++){
        for (size_t j = 0; j < HOut; j++){
            for (size_t k = 0; k < WOut; k++){
                biasFuse({i, j, k}) = this->bias({i, j, k}) * (*gamma)({i}) + (*beta)({i});
            }
        }
    }
    this->bias = biasFuse;
}


Conv2DCheetah::Conv2DCheetah (uint64_t in_feature_size, uint64_t stride, uint64_t padding, const Tensor<uint64_t>& weight, const Tensor<uint64_t>& bias, HE::HEEvaluator* HE, Tensor<uint64_t> *gamma, Tensor<uint64_t> *beta)
    : Conv2DCheetah(in_feature_size, stride, padding, weight, bias, HE)
{
    this->fused_bn = true;
    this->fuse_bn(gamma, beta);
    weight_pt = this->PackWeight();
};


// 加密张量
Tensor<UnifiedCiphertext> Conv2DCheetah::EncryptTensor(Tensor<UnifiedPlaintext> plainTensor) {
    std::vector<size_t> shapeTab = {dC ,dH , dW};
    Tensor<UnifiedCiphertext> TalphabetaCipher(shapeTab, HE->GenerateZeroCiphertext());
    for (unsigned long gama = 0; gama < dC; gama++) {
        for (unsigned long alpha = 0; alpha < dH; alpha++) {
            for (unsigned long beta = 0; beta < dW; beta++) {
                HE->encryptor->encrypt(plainTensor(gama * dH * dW + alpha + beta), TalphabetaCipher(gama * dH * dW + alpha * dW + beta));
            }
        }
    }
    return TalphabetaCipher;
}

Tensor<uint64_t> Conv2DCheetah::HETOTensor (Tensor<UnifiedCiphertext> inputCipher){
    auto shapeTab = inputCipher.shape();
    Tensor<UnifiedCiphertext> cipherMask(shapeTab,HE->GenerateZeroCiphertext());
    Tensor<UnifiedPlaintext> plainMask(shapeTab,HOST);
    size_t numPoly = 1;
    for (int num : shapeTab) {
        numPoly *= num;
    }
    auto tensorShapeTab = shapeTab;
    tensorShapeTab.push_back(polyModulusDegree);

    Tensor<uint64_t> tensorMask(tensorShapeTab, 0);
    UnifiedPlaintext plainMaskInv(HOST);
    if (HE->server) {
        int64_t mask;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int64_t> dist(0, plain - 1);
        for (size_t i = 0; i < numPoly; i++){
            plainMask(i).hplain().resize(polyModulusDegree);
            plainMaskInv.hplain().resize(polyModulusDegree);
            for (size_t l = 0; l < polyModulusDegree; l++){
                mask = dist(gen);
                *(plainMask(i).hplain().data() + l) = mask;
                tensorMask((i) * polyModulusDegree + l) = mask;
                mask = plain - mask;
                *(plainMaskInv.hplain().data() + l) = mask;   
            }
            HE->evaluator->add_plain(inputCipher(i), plainMaskInv, cipherMask(i));
        }
        cipherMask.flatten();
        HE->SendEncVec(cipherMask);
        return tensorMask;

    }else{
        HE->ReceiveEncVec(cipherMask);
        for (size_t i = 0; i < numPoly; i++){
            this->HE->decryptor->decrypt(cipherMask(i), plainMask(i));
            for (size_t j = 0; j < polyModulusDegree; j++){
                tensorMask(i * polyModulusDegree + j) = *(plainMask(i).hplain().data() + j);
            }
        }
        return tensorMask;
    }
}

// 计算输入张量的 Pack 版本
Tensor<uint64_t> Conv2DCheetah::PackActivation(Tensor<uint64_t> &x){
    Tensor<uint64_t> padded_x ({in_channels, in_feature_size, in_feature_size} ,0);
    for (size_t i = 0; i < in_channels; i++){
        for (size_t j = 0; j < (in_feature_size - 2 * padding); j++){
            for (size_t k = 0; k < (in_feature_size - 2 * padding); k++){
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
                    if ((ic + gama * CW) >= in_channels){
                        for (unsigned long jh = 0; jh < HW; jh++){
                            for (unsigned long kw = 0; kw < WW; kw++){
                                Tsub({ic,jh,kw}) = 0;
                            }
                        }
                        //对于超出的channel部分应该设置为0
                    }
                    else{
                        for (unsigned long jh = 0; jh < HW; jh++){
                            if ((jh + alpha * (HW - kernel_size + 1)) >= in_feature_size){
                                for (unsigned long kw = 0; kw < WW; kw++){
                                    Tsub({ic,jh,kw}) = 0;
                                }
                                //超出的HW部分应该为0
                            }
                            else{
                                for (unsigned long kw = 0; kw <WW; kw++){
                                    if ((kw + beta * (WW - kernel_size + 1)) >= in_feature_size){
                                        Tsub({ic,jh,kw}) = 0;
                                    }
                                    else{
                                        int64_t element = padded_x({gama * CW + ic, alpha * (HW - kernel_size + 1) + jh, beta * (WW - kernel_size + 1) + kw});
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
    std::vector<size_t> shapeTab = PackActivationTensor.shape();
    size_t numPoly = 1;
    for (int num : shapeTab) {
        numPoly *= num;
    }

    int len = shapeTab.back(); 
    shapeTab.pop_back();
    numPoly /= len;
    Tensor<UnifiedPlaintext> T(shapeTab,Datatype::HOST);
    for (size_t i = 0; i < numPoly; i++){
        vector<uint64_t> Tsubv(len, 0);
        for (size_t j = 0; j < len; j++){
            Tsubv[j] = PackActivationTensor(i * len + j);
        }
        T(i).hplain().resize(polyModulusDegree);
        seal::util::modulo_poly_coeffs(Tsubv, len, plain, T(i).hplain().data());
        std::fill_n(T(i).hplain().data() + len, polyModulusDegree - len, 0);
    }
    Tensor<UnifiedCiphertext> finalpack(shapeTab, HE->GenerateZeroCiphertext());
    if (!HE->server){
        //客服端
        Tensor<UnifiedCiphertext> enc(shapeTab, HE->GenerateZeroCiphertext());
        for (size_t i = 0; i < numPoly; i++){
            this->HE->encryptor->encrypt(T(i), enc(i));
        }
        // enc.flatten();
        HE->SendEncVec(enc);
    }else{
        //服务器端
        Tensor<UnifiedCiphertext> encflatten({numPoly}, this->HE->GenerateZeroCiphertext());
        HE->ReceiveEncVec(encflatten);
        Tensor<UnifiedCiphertext> enc(shapeTab, HE->GenerateZeroCiphertext());
        for (size_t i = 0; i < numPoly; i++){
            this->HE->evaluator->add_plain(encflatten(i), T(i), enc(i));
        }
        finalpack = enc;
    }
    return finalpack;
}

// 计算卷积核的 Pack 版本
Tensor<UnifiedPlaintext> Conv2DCheetah::PackWeight() {
    cout << "pack weight begin" << endl;
    std::vector<size_t> shapeTab = {dM, dC};
    Tensor<UnifiedPlaintext> Ktg(shapeTab,HOST);
    size_t len = OW + 1;
    if (!HE->server){
        return Ktg;
    }
    for (unsigned long theta = 0; theta < dM; theta++){
        for (unsigned long gama = 0; gama < dC; gama++){
            vector<uint64_t> Tsubv (polyModulusDegree,0); 
            for (unsigned long it = 0; it < MW; it++){
                for (unsigned long jg = 0; jg < CW; jg++){
                    if (((theta * MW + it) >= out_channels) || ((gama * CW + jg) >= in_channels)){
                        for (unsigned hr = 0; hr < kernel_size; hr++){
                            for (unsigned hc = 0; hc < kernel_size; hc++){
                                Tsubv[OW - it * CW * HW * WW - jg * HW * WW - hr * WW - hc] = 0;
                            }
                        }
                    }else{
                        for (unsigned hr = 0; hr < kernel_size; hr++){
                            for (unsigned hc = 0; hc < kernel_size; hc++){
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
    cout << "transfer weight to device" << endl;
    if (HE->Backend() == DEVICE){
        for (size_t i = 0; i < Ktg.size(); i++){
            Ktg(i).to_device(*HE->context);
        }
    }
    cout << "pack weight done" << endl;
    return Ktg;
}

Tensor<UnifiedCiphertext> Conv2DCheetah::sumCP(Tensor<UnifiedCiphertext> cipherTensor, Tensor<UnifiedPlaintext> plainTensor){
    Tensor<UnifiedCiphertext> Talphabeta({dC, dH, dW}, HOST);
    for (size_t gama = 0; gama < dC; gama++){
        for (size_t alpha = 0; alpha < dH; alpha++){
            for (size_t beta = 0; beta < dW; beta++){
                HE->evaluator->add_plain(cipherTensor({gama,alpha,beta}), plainTensor({gama,alpha,beta}), Talphabeta({gama,alpha,beta}));
            }
        }
    }
    return Talphabeta;
}
   

// 计算同态卷积
// #define MULTI_STRAEM
Tensor<UnifiedCiphertext> Conv2DCheetah::HECompute(const Tensor<UnifiedPlaintext> &weight_pt, Tensor<UnifiedCiphertext> &ac_ct)
{
    const auto target = HE->server ? HE->Backend() : HOST;
    cout << "target:" << target << endl;
    std::vector<size_t> shapeTab = {dM, dH, dW};
    Tensor<UnifiedCiphertext> out_ct(shapeTab,HE->GenerateZeroCiphertext(target));
    if (!HE->server){
        return out_ct;
    }

    #ifndef MULTI_STRAEM
    UnifiedCiphertext interm(target);
    for (size_t theta = 0; theta < dM; theta++) {
        for (size_t alpha = 0; alpha < dH; alpha++) {
            for (size_t beta = 0; beta < dW; beta++) {
                HE->evaluator->multiply_plain(ac_ct({0, alpha, beta}), weight_pt({theta, 0}), out_ct({theta, alpha, beta}));
                for (size_t gama = 1; gama < dC; gama++) {
                    HE->evaluator->multiply_plain(ac_ct({gama, alpha, beta}), weight_pt({theta, gama}), interm);
                    HE->evaluator->add_inplace(out_ct({theta, alpha, beta}), interm);
                }
            }
        }
    }
    #else
    // 定义线程工作函数
    auto worker = [&](size_t theta_start, size_t theta_end) {
        UnifiedCiphertext interm(target);
        for (size_t theta = theta_start; theta < theta_end; theta++) {
            for (size_t alpha = 0; alpha < dH; alpha++) {
                for (size_t beta = 0; beta < dW; beta++) {
                    HE->evaluator->multiply_plain(ac_ct({0, alpha, beta}), 
                                                weight_pt({theta, 0}), 
                                                out_ct({theta, alpha, beta}));
                    for (size_t gama = 1; gama < dC; gama++) {
                        HE->evaluator->multiply_plain(ac_ct({gama, alpha, beta}), 
                                                    weight_pt({theta, gama}), 
                                                    interm);
                        HE->evaluator->add_inplace(out_ct({theta, alpha, beta}), interm);
                    }
                }
            }
        }
    };

    const size_t num_threads = 4; // [可修改]
    const size_t theta_per_thread = (dM + num_threads - 1) / num_threads;

    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; t++) {
        size_t start = t * theta_per_thread;
        size_t end = std::min(start + theta_per_thread, dM);
        if (start < dM) {
            threads.emplace_back(worker, start, end);
        }
    }

    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }
    #endif
    return out_ct;
}

Tensor<uint64_t> Conv2DCheetah::DepackResult(Tensor<uint64_t> &out){
    Tensor<uint64_t> finalResult ({out_channels, HOut, WOut});
    int checkl = 0;

    for (size_t cprime = 0; cprime < out_channels; cprime++){
        for (size_t iprime = 0; iprime < HOut; iprime++){
            for (size_t jprime = 0; jprime < WOut; jprime++){
                size_t c = cprime % MW;
                size_t i = (iprime * stride) % (HW - kernel_size + 1);
                size_t j = (jprime * stride) % (WW - kernel_size + 1);
                size_t theta = cprime / MW;
                size_t alpha = (iprime * stride) / (HW - kernel_size + 1);
                size_t beta = (jprime * stride) / (WW - kernel_size + 1);
                size_t des = OW - c * CW * HW * WW + i  * WW + j;
                finalResult({cprime, iprime, jprime}) = out({theta, alpha, beta, des});
            }
        }
    }
    return finalResult;

}

Tensor<uint64_t> Conv2DCheetah::operator()(Tensor<uint64_t> &x){
    cout << "in Conv2D, x.shape:" << endl;
    x.print_shape();
    auto pack = this->PackActivation(x);
    // cout << "PackActivation done" << endl;
    auto Cipher = Operator::SSToHE_coeff(pack, HE);
    // cout << "SSTOHE done" << endl;
    auto ConvResult = this->HECompute(weight_pt, Cipher);
    // cout << "HECompute done" << endl;
    auto share = Operator::HEToSS_coeff(ConvResult, HE);
    // cout << "HEToSS done" << endl;
    auto finalR = this->DepackResult(share);
    // cout << "DepackResult done" << endl;
    return finalR;
}

 // namespace LinearLayer

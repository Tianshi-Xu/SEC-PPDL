#include <LinearLayer/Conv.h>

using namespace LinearLayer;

namespace LinearLayer {
class Conv2DCheetah : public LinearLayer::Conv2D{
public:

    unsigned long M = 3;
    //M是input channel
    unsigned long C = 3;
    //C是output channel
    unsigned long H = 7;
    //这个是input height。
    unsigned long W = 7;
    //这个是input width。
    unsigned long h = 3;
    //kernel size
    unsigned long s = 2;
    //this is stride.
    unsigned long N;
    unsigned long HW;
    unsigned long WW;
    unsigned long CW;
    unsigned long MW;
    unsigned long dM;
    unsigned long dC;
    unsigned long dH;
    unsigned long dW;
    unsigned long OW;
    unsigned long Hprime;
    unsigned long Wprime;
    unsigned long HWprime;
    unsigned long WWprime;
    size_t polyModulusDegree;
    uint64_t plain;

    int div_upper(
        int a,
        int b
    ){
        return ((a + b - 1) / b);
    }

    int calculate_cost(int H, int W, int h, int Hw, int Ww, int C, int N) {
    return (int)ceil((double)C / (N / (Hw * Ww))) *
           (int)ceil((double)(H - h + 1) / (Hw - h + 1)) *
           (int)ceil((double)(W - h + 1) / (Ww - h + 1));
}

    // Function to find the optimal Hw and Ww
    void find_optimal_partition(int H, int W, int h, int C, int N, int *optimal_Hw, int *optimal_Ww) {
        int min_cost = (1 << 30);

        // Iterate over all possible Hw and Ww within the constraints
        for (int Hw = h; Hw <= H; Hw++) {
            for (int Ww = h; Ww <= W; Ww++) {
                if (Hw * Ww > N) {
                    continue; // Skip if Hw * Ww exceeds N
                }
                int cost = calculate_cost(H, W, h, Hw, Ww, C, N);
                if (cost < min_cost) {
                    min_cost = cost;
                    *optimal_Hw = Hw;
                    *optimal_Ww = Ww;
                }
            }
        }
    }



    Conv2DCheetah (int in_channels, int out_channels, int kernel_size, int stride, int padding, HEEvaluator* he, Tensor<int> inputTensor, Tensor<int> kernel)
        : Conv2D(in_channels, out_channels, kernel_size, stride, padding, he, inputTensor, kernel){
            C = out_channels;
            M = in_channels;
            s = stride;
            h = kernel_size;
            const std::vector<size_t>& inputTensorShape = inputTensor.shape();
            H = inputTensorShape[0];
            W = inputTensorShape[1];
            int optimal_Hw = 0, optimal_Ww = 0;
            find_optimal_partition(H, W, h, C, N, &optimal_Hw, &optimal_Ww);
            HW = optimal_Hw;
            WW = optimal_Ww;
            CW = 2;
            MW = 2;
            dM = div_upper(M,MW);
            dC = div_upper(C,CW);
            dH = div_upper(H - h + 1 , HW - h + 1);
            dW = div_upper(W - h + 1 , WW - h + 1);
            OW = HW * WW * (MW * CW - 1) + WW * (h - 1) + h - 1;
            Hprime = (H - h + s) / s;
            Wprime = (W - h + s) / s;
            HWprime = (HW - h + s) / s;
            WWprime = (WW - h + s) / s;
            auto polyModulusDegree = he->polyModulusDegree;
            auto plain = he->plain;

        };

    Tensor<seal::Ciphertext> EncryptTensor(Tensor<seal::Plaintext> plainTensor){
        std::vector<size_t> shapeTab = {dC, dH, dW};
        Tensor<seal::Ciphertext> TalphabetaCipher(shapeTab);
        int len = CW * HW * WW;
        Tensor<uint64_t> Tsub ({ CW, HW, WW});
        for (unsigned long gama = 0; gama < dC; gama++){
            for (unsigned long alpha = 0; alpha < dH; alpha++){
                for (unsigned long beta = 0; beta < dW; beta++){
                    he->encryptor->encrypt(plainTensor({gama,alpha,beta}),TalphabetaCipher({gama,alpha,beta}));
                }
            }
        }
        return TalphabetaCipher;
    };
        

    Tensor<seal::Plaintext> PackTensor(Tensor<int> x) {
        std::vector<size_t> shapeTab = {dC, dH, dW};
        Tensor<seal::Plaintext> Talphabeta(shapeTab);
        //Tensor<seal::Ciphertext> TalphabetaCipher(shapeTab);
        int len = CW * HW * WW;
        Tensor<uint64_t> Tsub ({ CW, HW, WW});
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
                        }
                        else{
                            for (unsigned long jh = 0; jh < HW; jh++){
                                if ((jh + alpha * (HW - h + 1)) >= H){
                                    for (unsigned long kw = 0; kw < WW; kw++){
                                        Tsub({ic,jh,kw}) = 0;
                                    }
                                }
                                else{
                                    for (unsigned long kw = 0; kw <WW; kw++){
                                        if ((kw + beta * (WW - h + 1)) >= W){
                                            Tsub({ic,jh,kw}) = 0;
                                        }
                                        else{
                                            int64_t element = inputTensor({gama * CW + ic, alpha * (HW - h + 1) + jh, beta * (WW - h + 1) + kw});
                                            Tsub({ic,jh,kw}) = (element >= 0) ? unsigned(element) : unsigned(element + plain);
                                        }
                                    }
                                }
                            }
                        }

                        // Tsub.print();
                        Tensor<uint64_t> Tsubflatten = Tsub;
                        Tsubflatten.flatten();
                        //std::cout<< "1234567" << std::endl;
                        vector<uint64_t> Tsubv = Tsubflatten.data(); 
                        // vector<uint64_t> tmp(poly_modulus_degree);
                        // std::transform(Tsubv.data(), Tsubv.end(),tmp.begin(),[plain](uint64_t u) { return u > 0 ? plain - u : 0; });
                        //std::cout<< "1234567" << std::endl;
                        Talphabeta({gama,alpha,beta}).resize(polyModulusDegree);
                        //Talphabeta[gama][alpha][beta].resize(polyModulusDegree);
                        //batch_encoder.encode(vecIni, Talphabeta[gama][alpha][beta]);
                        //seal::util::modulo_poly_coeffs(Tsubv, len, plain, Talphabeta[gama][alpha][beta].data());
                        seal::util::modulo_poly_coeffs(Tsubv, len, plain, Talphabeta({gama,alpha,beta}).data());
                        //std::cout<< "1234567" << std::endl;
                        //Talphabeta[gama][alpha][beta].resize(poly_modulus_degree);
                        //batch_encoder.encode(vecIni, Talphabeta[gama][alpha][beta]);
                        std::fill_n(Talphabeta({gama,alpha,beta}).data() + len, polyModulusDegree - len, 0);
                        //std::cout<< "1234567" << std::endl;
                        //he->encryptor->encrypt(Talphabeta({gama,alpha,beta}),TalphabetaCipher[gama][alpha][beta]);
                    }
                }
            }
        }
        return Talphabeta;
    }

    Tensor<Plaintext> PackKernel(Tensor<int> x){
        std::vector<std::vector<seal::Plaintext>> Ktg (
            dM, std::vector<seal::Plaintext>{
                dC, seal::Plaintext()
            }
        );

        Tensor<uint64_t> Ksub ({ MW, CW, h, h});
        int len = MW * CW * h * h;

        for (unsigned long theta = 0; theta < dM; theta++){
            for (unsigned long gama = 0; gama < dC; gama++){
                for (unsigned long it = 0; it < MW; it++){
                    for (unsigned long jg = 0; jg < CW; jg++){
                        if (((theta * MW + it) >= M) || ((gama * CW + jg) >= C)){
                            for (unsigned hr = 0; hr < h; hr++){
                                for (unsigned hc = 0; hc < h; hc++){
                                    Ksub({it,jg,hr,hc}) = 0;
                                }
                            }
                        }else{
                            for (unsigned hr = 0; hr < h; hr++){
                                for (unsigned hc = 0; hc < h; hc++){
                                    int64_t element = kernel({theta * MW + it, gama * CW + jg, hr, hc});
                                    Ksub({it,jg,hr,hc}) = (element >= 0) ? unsigned(element) : unsigned(element + plain);
                                }
                            }
                        }
                    }
                }
                Tensor<uint64_t> Ksubflatten = Ksub;
                Ksubflatten.flatten();
                //std::cout<< "1234567" << std::endl;
                vector<uint64_t> Tsubv = Ksubflatten.data(); 
                // vector<uint64_t> tmp(poly_modulus_degree);
                // std::transform(Tsubv.data(), Tsubv.end(),tmp.begin(),[plain](uint64_t u) { return u > 0 ? plain - u : 0; });
                //std::cout<< "1234567" << std::endl;
                Ktg[theta][gama].resize(polyModulusDegree);
                //batch_encoder.encode(vecIni, Talphabeta[gama][alpha][beta]);
                seal::util::modulo_poly_coeffs(Tsubv, len, plain, Ktg[theta][gama].data());
                //std::cout<< "1234567" << std::endl;
                //Talphabeta[gama][alpha][beta].resize(poly_modulus_degree);
                //batch_encoder.encode(vecIni, Talphabeta[gama][alpha][beta]);
                std::fill_n(Ktg[theta][gama].data() + len, polyModulusDegree - len, 0);
                //std::cout<< "1234567" << std::endl;
            }
        }
    }; 

    //暂时先不区分Alice和Bob，先只做密文向量和明文kernel。

    Tensor<Ciphertext> Conv(Tensor<seal::Ciphertext> T, Tensor<seal::Plaintext> K){
        std::vector<size_t> shapeTab = {dM, dH, dW};
        Tensor<seal::Ciphertext> ConvRe(shapeTab);

        seal::Ciphertext interm;
        for (size_t theta = 0; theta < dM; theta++){
            for (size_t alpha = 0; alpha < dH; alpha++){
                for (size_t beta = 0; beta < dW; beta++){
                    he->evaluator->multiply_plain(T({0,alpha,beta}),K({theta,0}),ConvRe({theta,alpha,beta}));
                    for (size_t gama = 1; gama < dC; gama++){
                        he->evaluator->multiply_plain(T({gama,alpha,beta}),K({theta,gama}),interm);
                        he->evaluator->add_inplace(ConvRe({theta,alpha,beta}),interm);
                    }       
                }
            }
        }
    };
};

}
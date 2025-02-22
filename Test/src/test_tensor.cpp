#include <Datatype/Tensor.h>
using namespace Datatype;
using namespace std;


int main(){
    Tensor<double> tensor({2, 3});
    tensor.randomize();
    tensor.print();
    Tensor<uint64_t> tensor2({2, 3});
    tensor2.randomize(8);
    tensor2.print();
    Tensor<int> tensor3({2, 3});
    tensor3.randomize(4);
    tensor3.print();
    cout << tensor3({0,1}) << endl;
    printf("%d\n", tensor3({0,1}));
    // Tensor<double> *tensor2 = &tensor;
    // tensor({0,0}) = 100;
    // tensor2->print();
    // tensor.print();
    return 0;
}
#include <Datatype/Tensor.h>
using namespace Datatype;
using namespace std;


int main(){
    // Tensor<double> tensor1({2, 3});
    // tensor1.randomize();
    // tensor1.print();
    // tensor1.print_shape();
    // Tensor<double> tensor2({2, 3});
    // Tensor<double> tensor3 = tensor1 + tensor2;
    // tensor3({0,1}) = 100;
    // cout << tensor3({0,1}) << endl;
    // tensor3.reshape({3, 2});
    // tensor3.flatten();
    Tensor<int64_t> tensor1({2, 3}, 10, 5);
    tensor1.randomize(12);
    tensor1.print();
    cout << (tensor1(0) & tensor1.get_mask()) << endl;
    return 0;
}

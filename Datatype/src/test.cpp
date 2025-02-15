#include <Datatype/Tensor.h>
using namespace Datatype;


int main(){
    Tensor<double> tensor({2, 3}, {1.1, 2.2, 3.3, 4.4, 5.5, 6.6});
    tensor.print();
    return 0;
}
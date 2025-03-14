//#include "NetIO.h"
#include <NonlinearLayer/Pool.h>
using namespace std;
using namespace NonlinearLayer;

// avg_pool is correct!
int main(int argc, char* argv[]) {
    Tensor<uint64_t> x({1,1,4,4});
    x.randomize(4);
    x.print();
    std::cout << "gen";
    AvgPool2D<uint64_t> avg_pool(2);
    Tensor<uint64_t> y = avg_pool(x);
    y.print();
    return 0;
}
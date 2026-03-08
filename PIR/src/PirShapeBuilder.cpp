#include "PIR/PirShapeBuilder.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace secppdl::pir
{

PirUserInput read_user_input()
{
    PirUserInput input;
    std::cout << "INFO: Input the number of items (e.g., 2^15 (32768)):" << std::endl;
    std::cin >> input.num;
    std::cout << "INFO: Input the bit width (e.g., 64):" << std::endl;
    std::cin >> input.bit_width;
    std::cout << "INFO: Input the query id (e.g., 1, in [0,num-1]):" << std::endl;
    std::cin >> input.query_id;
    return input;
}

PirShape build_shape(std::size_t num, std::size_t bit_width, std::size_t poly_modulus_degree, std::size_t slot_count)
{
    if (num == 0 || (num % poly_modulus_degree) != 0)
    {
        throw std::invalid_argument("num must be positive and divisible by poly_modulus_degree.");
    }
    if (bit_width == 0 || (bit_width % 16) != 0)
    {
        throw std::invalid_argument("bit width must be a positive multiple of 16.");
    }

    PirShape shape;
    shape.poly_modulus_degree = poly_modulus_degree;
    shape.slot_count = slot_count;

    const std::size_t dim1 = num / poly_modulus_degree;
    const std::size_t h = static_cast<std::size_t>(std::llround(std::sqrt(static_cast<long double>(dim1))));
    if (h * h != dim1)
    {
        throw std::invalid_argument("SmartPIR setup requires num/poly_modulus_degree to be a perfect square.");
    }

    shape.h = h;
    shape.w = h * poly_modulus_degree;
    shape.blocks_per_row = shape.w / poly_modulus_degree;
    shape.chunks = bit_width / 16;

    if (shape.blocks_per_row == 0 || shape.blocks_per_row > poly_modulus_degree)
    {
        throw std::invalid_argument("blocks_per_row must be in [1, poly_modulus_degree].");
    }

    return shape;
}

PirShape build_shape(const PirUserInput &input, std::size_t poly_modulus_degree, std::size_t slot_count)
{
    if (input.query_id >= input.num)
    {
        throw std::invalid_argument("query id must be in [0, num-1].");
    }

    PirShape shape = build_shape(input.num, input.bit_width, poly_modulus_degree, slot_count);
    QueryIndex q = map_query(input.query_id, shape);

    shape.r_idx = q.r_idx;
    shape.c_idx = q.c_idx;
    shape.c_block = q.c_block;
    shape.oft = q.oft;
    return shape;
}

QueryIndex map_query(std::size_t query_id, const PirShape &shape)
{
    if (query_id >= shape.h * shape.w)
    {
        throw std::invalid_argument("query_id out of range.");
    }

    QueryIndex q;
    q.query_id = query_id;
    q.r_idx = query_id / shape.w;
    q.c_idx = query_id % shape.w;
    q.c_block = q.c_idx / shape.poly_modulus_degree;
    q.oft = query_id % shape.poly_modulus_degree;

    if (q.r_idx >= shape.h || q.c_block >= shape.blocks_per_row)
    {
        throw std::invalid_argument("Computed SmartPIR indices are out of range.");
    }

    return q;
}

void print_usage(const char *prog)
{
    std::cout << "Usage: " << prog << " [num] [bit_width] [batch_size] [seed] [gpu_id]" << std::endl;
    std::cout << "  num: total DB items, must be divisible by poly_modulus_degree(32768), default=32768" << std::endl;
    std::cout << "  bit_width: must be multiple of 16, default=16" << std::endl;
    std::cout << "  batch_size: number of concurrent queries, default=2" << std::endl;
    std::cout << "  seed: RNG seed, default=7" << std::endl;
    std::cout << "  gpu_id: CUDA device id, default=0" << std::endl;
}

CliInput parse_cli(int argc, char **argv)
{
    CliInput input;

    if (argc > 1 && std::string(argv[1]) == "--help")
    {
        print_usage(argv[0]);
        std::exit(0);
    }

    if (argc > 1)
    {
        input.num = static_cast<std::size_t>(std::stoull(argv[1]));
    }
    if (argc > 2)
    {
        input.bit_width = static_cast<std::size_t>(std::stoull(argv[2]));
    }
    if (argc > 3)
    {
        input.batch_size = static_cast<std::size_t>(std::stoull(argv[3]));
    }
    if (argc > 4)
    {
        input.seed = static_cast<uint64_t>(std::stoull(argv[4]));
    }
    if (argc > 5)
    {
        input.gpu_id = static_cast<int>(std::stoi(argv[5]));
    }

    if (input.batch_size == 0)
    {
        throw std::invalid_argument("batch_size must be positive.");
    }
    if (input.gpu_id < 0)
    {
        throw std::invalid_argument("gpu_id must be non-negative.");
    }

    return input;
}

} // namespace secppdl::pir

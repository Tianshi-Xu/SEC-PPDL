#pragma once

#include <cstddef>

#include "PIR/PirTypes.h"

namespace secppdl::pir
{

PirUserInput read_user_input();
PirShape build_shape(const PirUserInput &input, std::size_t poly_modulus_degree, std::size_t slot_count);
PirShape build_shape(std::size_t num, std::size_t bit_width, std::size_t poly_modulus_degree, std::size_t slot_count);
QueryIndex map_query(std::size_t query_id, const PirShape &shape);
void print_usage(const char *prog);
CliInput parse_cli(int argc, char **argv);

} // namespace secppdl::pir

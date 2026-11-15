// q36_pp_i8.hpp - the fat products (a prompt chunk's GEMMs on the matrix engine) are the B70 card library's
// (strata/b70/fat.hpp); the dense family's sources go on calling them under their first name.
#pragma once
#include "strata/b70/fat.hpp"

namespace strata {
namespace q36pp = b70::fat;
}

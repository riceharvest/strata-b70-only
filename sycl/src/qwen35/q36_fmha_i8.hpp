// q36_fmha_i8.hpp - the 8-bit cache's prompt attention on the matrix engine is the B70 card library's
// (strata/b70/attn_prompt.hpp); the dense family's sources go on calling it under its first name.
#pragma once
#include "strata/b70/attn_prompt.hpp"

namespace strata {
namespace q36fmha = b70::attn_prompt;
}

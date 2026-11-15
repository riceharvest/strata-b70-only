// strata/b70/env.hpp - the B70 card library: how its kernels read a tuning switch.
#pragma once
#include <cstdlib>

namespace strata::b70 {

/// A switch of the card library by its name (`STRATA_B70_...`).  `legacy` is the name the switch had when the kernel belonged
/// to one model family; it is still read when the new name is not set, so existing configs keep working.
inline const char* env(const char* name, const char* legacy) {
    const char* e = std::getenv(name);
    return e ? e : std::getenv(legacy);
}

}   // namespace strata::b70

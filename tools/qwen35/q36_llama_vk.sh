#!/usr/bin/env bash
# q36_llama_vk.sh <q36_llama_dump args...> - llama.cpp's side of the parity and KL checks on the card: q36_llama_dump
# (q36_llama_dump.cpp, built against a llama.cpp release with its Vulkan backend) with that release's backend libraries.
#   Q36_LLAMA_DUMP        the q36_llama_dump binary (required)
#   STRATA_DUMP_BACKENDS  the folder of llama.cpp's backend libraries, libggml-vulkan.so among them (required)
#   STRATA_DUMP_CTX       the context cells it allocates (default 1024)
#   Q36_LOCK              a command put before it (e.g. "flock /var/lock/gpu.lock", when other jobs share the card)
set -uo pipefail
: "${Q36_LLAMA_DUMP:?set Q36_LLAMA_DUMP to the q36_llama_dump binary}"
: "${STRATA_DUMP_BACKENDS:?set STRATA_DUMP_BACKENDS to the folder of the llama.cpp backend libraries}"
export STRATA_DUMP_BACKENDS
export STRATA_DUMP_CTX=${STRATA_DUMP_CTX:-1024}
# (Q36_LOCK is split into words on purpose)
exec ${Q36_LOCK:-} "$Q36_LLAMA_DUMP" "$@"

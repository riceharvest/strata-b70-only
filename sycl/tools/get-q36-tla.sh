#!/usr/bin/env bash
# get-q36-tla.sh: the two upstream projects that strata-q36's long-prompt attention (libq36_tla.so) is compiled against.
# They are not part of this repository: this fetches them, at the commits below, into third_party/ (git-ignored).
#
#   third_party/sycl-tla          https://github.com/intel/sycl-tla (BSD-3-Clause)
#                                 87f6850680a580654b9ea2c80dbc01aeb36ad231  (2026-08-10, "FIX BMG GEMM Performance regression (#846)", v0.9.2 + 7 commits)
#   third_party/vllm-xpu-kernels  https://github.com/vllm-project/vllm-xpu-kernels (Apache-2.0)
#                                 ddf336d86e3c8602888572a3502f951abd51df12  (2026-09-18, v0.1.15 + 4 commits)
#                                 plus sycl/third_party/q36-tla/vllm-fmha_utils-no-torch.patch (takes the torch glue out of one header)
#
# Only the folders the build reads are checked out (sycl-tla: include, tools/util/include, applications/flash_attention_v2;
# vllm-xpu-kernels: csrc/xpu/attn).  About 40 MB of sycl-tla.  Needs git (2.25 or newer) and a network.  Run it once from
# anywhere; running it again changes nothing.  Q36_TLA_DEST=<dir> puts the two folders somewhere else (the build wants third_party/).
# Licences and what was changed: THIRD_PARTY_NOTICES.md.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
dest=${Q36_TLA_DEST:-$repo/third_party}
patch=$repo/sycl/third_party/q36-tla/vllm-fmha_utils-no-torch.patch

SYCL_TLA_URL=https://github.com/intel/sycl-tla.git
SYCL_TLA_COMMIT=87f6850680a580654b9ea2c80dbc01aeb36ad231
VLLM_URL=https://github.com/vllm-project/vllm-xpu-kernels.git
VLLM_COMMIT=ddf336d86e3c8602888572a3502f951abd51df12

# fetch <name> <url> <commit> <folder>...
fetch() {
    local name=$1 url=$2 commit=$3 d tmp
    shift 3
    d=$dest/$name
    if [ -d "$d/.git" ]; then
        if [ "$(git -C "$d" rev-parse HEAD 2>/dev/null)" = "$commit" ]; then
            echo "$name: already at $commit"
            return 0
        fi
        echo "get-q36-tla.sh: $d is a checkout of another commit; move it away and run this again" >&2
        return 1
    fi
    if [ -e "$d" ]; then
        echo "get-q36-tla.sh: $d exists and is not a checkout this script made; move it away and run this again" >&2
        return 1
    fi
    tmp=$d.part
    rm -rf "$tmp"
    mkdir -p "$tmp"
    git -C "$tmp" init -q
    git -C "$tmp" config advice.detachedHead false
    git -C "$tmp" remote add origin "$url"
    git -C "$tmp" sparse-checkout set --cone "$@"
    echo "$name: fetching $commit"
    git -C "$tmp" fetch -q --depth 1 --filter=blob:none origin "$commit"
    git -C "$tmp" checkout -q --detach FETCH_HEAD
    if [ "$(git -C "$tmp" rev-parse HEAD)" != "$commit" ]; then
        echo "get-q36-tla.sh: $name: the fetch gave another commit than $commit" >&2
        rm -rf "$tmp"
        return 1
    fi
    mkdir -p "$dest"
    mv "$tmp" "$d"
}

mkdir -p "$dest"
fetch sycl-tla "$SYCL_TLA_URL" "$SYCL_TLA_COMMIT" include tools/util/include applications/flash_attention_v2
fetch vllm-xpu-kernels "$VLLM_URL" "$VLLM_COMMIT" csrc/xpu/attn

# the one patch on vllm-xpu-kernels (applied once: a second run finds it in place)
v=$dest/vllm-xpu-kernels
if git -C "$v" apply --check --reverse "$patch" >/dev/null 2>&1; then
    echo "vllm-xpu-kernels: the patch is in place"
elif git -C "$v" apply --check "$patch" >/dev/null 2>&1; then
    git -C "$v" apply "$patch"
    echo "vllm-xpu-kernels: applied $(basename "$patch")"
else
    echo "get-q36-tla.sh: $patch does not apply to $v (changed by hand?). Remove $v and run this again" >&2
    exit 1
fi
echo "done: $dest/sycl-tla and $dest/vllm-xpu-kernels"

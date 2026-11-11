# Strata SYCL port - development image for the Intel Arc path.
# Base: the llama.cpp SYCL image already on the box (oneAPI 2026.1 compiler, MKL, Level Zero for Xe2).
# Adds SYCLomatic (dpct, "Intel DPC++ Compatibility Tool") for the CUDA -> SYCL migration, and the build tools.
FROM ghcr.io/snailium/llama.cpp-sycl-intel-b70/llama-sycl-b70:stable
# Pinned device compiler (IGC + ocloc). The engine is compiled ahead of time for the card at link time by ocloc/IGC, so the
# compiler version decides how fast the answers are. Measured 2026-11-11 on the B70, same sources and flags, same fixed text:
#   IGC 2.41.5 + ocloc 26.35.39758.10:  127-133 tok/s (reasoning prompt), 108-110 (code prompt)
#   IGC 2.42.6 + ocloc 26.40.40196.8:   120-125 and 103-106  (3-5% slower; prompt reading is the same)
# The base image's tag moves, and from 2026-11-04 it carries the slower pair. So this image puts the 2.41.5 / 26.35 packages over
# the base's, from the vendors' GitHub release pages, checked by SHA-256, and holds them so a later apt upgrade cannot undo it.
# The compute runtime packages (opencl-icd, level-zero) must match the IGC version exactly, so the same 26.35 release replaces those too
# (this is the set that was measured). A binary built with 2.41.5 also runs at full speed on the newer 26.40 runtime.
# To move to a newer compiler on purpose: change the three versions and hashes below, measure, and update docs/B70.md.
ARG IGC_VERSION=2.41.5
ARG IGC_BUILD=22716
ARG OCLOC_VERSION=26.35.39758.10
ARG IGC_CORE_SHA256=0a6e64a663ae65a0fa02d6912ae3b6b37cf85b90c21cc423fd9fef70aaf4f628
ARG IGC_OPENCL_SHA256=779e1b9e88098eb25711e9a8f67c2752665bad22f134aa40ed5649f6e1b87058
ARG OCLOC_SHA256=c64bff586edf2bd9b49f3e4c9c2be25e05c43466e4dafa2a2c3948d498c736b7
ARG ICD_SHA256=61712caaddeba3d38e4f79e2a0fb23fea25596ca2d72c3144c6eea2331ec4301
ARG ZE_SHA256=c19a641b953d55aebbf1d51bec364a84bf629f985e02fbbe6dc70224c0e88470
RUN set -eu; apt-get update -qq; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends ca-certificates curl; \
    mkdir /tmp/pin && cd /tmp/pin; \
    g=https://github.com/intel/intel-graphics-compiler/releases/download/v${IGC_VERSION}; \
    c=https://github.com/intel/compute-runtime/releases/download/${OCLOC_VERSION}; \
    curl -fsSLO "$g/intel-igc-core-2_${IGC_VERSION}+${IGC_BUILD}_amd64.deb"; \
    curl -fsSLO "$g/intel-igc-opencl-2_${IGC_VERSION}+${IGC_BUILD}_amd64.deb"; \
    curl -fsSLO "$c/intel-ocloc_${OCLOC_VERSION}-0_amd64.deb"; \
    curl -fsSLO "$c/intel-opencl-icd_${OCLOC_VERSION}-0_amd64.deb"; \
    curl -fsSLO "$c/libze-intel-gpu1_${OCLOC_VERSION}-0_amd64.deb"; \
    printf '%s  %s\n' "$IGC_CORE_SHA256" "intel-igc-core-2_${IGC_VERSION}+${IGC_BUILD}_amd64.deb" \
                       "$IGC_OPENCL_SHA256" "intel-igc-opencl-2_${IGC_VERSION}+${IGC_BUILD}_amd64.deb" \
                       "$OCLOC_SHA256" "intel-ocloc_${OCLOC_VERSION}-0_amd64.deb" \
                       "$ICD_SHA256" "intel-opencl-icd_${OCLOC_VERSION}-0_amd64.deb" \
                       "$ZE_SHA256" "libze-intel-gpu1_${OCLOC_VERSION}-0_amd64.deb" | sha256sum -c -; \
    dpkg -i --force-downgrade ./*.deb; \
    apt-mark hold intel-igc-core-2 intel-igc-opencl-2 intel-ocloc intel-opencl-icd libze-intel-gpu1; \
    cd /; rm -rf /tmp/pin /var/lib/apt/lists/*
RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
      intel-oneapi-dpcpp-ct ninja-build unzip python3-pip python3-venv rsync \
    && rm -rf /var/lib/apt/lists/*
LABEL strata.igc="2.41.5" strata.ocloc="26.35.39758.10"
ENV ONEAPI_DEVICE_SELECTOR=level_zero:0 SYCL_CACHE_PERSISTENT=0 ZES_ENABLE_SYSMAN=1
ENTRYPOINT ["/bin/bash", "-lc"]

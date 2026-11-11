// q36_tla_fmha.cpp - libq36_tla.so: vllm-xpu-kernels 0.1.15.4's Xe2 chunk_prefill (csrc/xpu/attn/xe_2/chunk_prefill.hpp,
// on sycl-tla 87f68506), the template instantiated as its policy_dispatch_impl instantiates it for an FP16 query and
// an FP16 K / V cache: chunk_policy_head256 (256 query rows a work-group, 32 sub-groups of 8 rows, 32-cell K / V
// tiles), 2 pipeline stages, not paged, no sliding window, no sinks, the softmax LSE written; causal and not.  Nothing
// of theirs is changed (fmha_utils.hpp without its torch glue, see that file); this file only fills their
// chunk_prefill_args_t the way their cutlass_chunk_prefill_impl does for a varlen, non-paged call (fmha_xe2.cpp).
// Built on its own (sycl/CMakeLists.txt): their compile defines and AOT link options, not strata's.
#include <sycl/sycl.hpp>
#include "q36_tla.hpp"
#include "csrc/xpu/attn/xe_2/chunk_prefill.hpp"

namespace {
using namespace cute;
using Pol = chunk_policy_head256;
template <bool Causal>
using Cfg = vllm::xpu::xe2::FMHAConfig<Pol::ShapeQK, Pol::ShapePV, Pol::ShapeOut,
                                       Pol::SubgroupLayoutQK, void, /*PipelineStages*/ 2,
                                       /*Paged*/ false, Causal, /*Local*/ false, /*Sink*/ false, /*SoftmaxLSE*/ true,
                                       half_t, half_t, half_t, half_t>;
}  // namespace

namespace q36tla {

void fmha(sycl::queue& q, const Fmha& f) {
    vllm::xpu::xe2::chunk_prefill_args_t a{};
    a.query = const_cast<sycl::half*>(f.q);
    a.key = const_cast<sycl::half*>(f.k);
    a.value = const_cast<sycl::half*>(f.v);
    a.out = f.o;
    a.block_table = nullptr;
    a.cu_seqlens_q = const_cast<int32_t*>(f.cu_q);
    a.cu_seqlens_k = const_cast<int32_t*>(f.cu_k);
    a.max_queries = f.T;
    a.max_keys = f.L;
    a.total_seqlen_q = f.T;
    a.total_seqlen_k = f.L;
    a.k_scale = nullptr;
    a.v_scale = nullptr;
    a.sm_scale = f.scale;
    a.sm_sink = nullptr;
    a.batch_size = 1;
    a.num_heads_q = f.H;
    a.num_heads_k = f.HKV;
    a.head_size = kHeadDim;
    a.max_blocks_per_seq = 0;
    a.block_size = 0;
    a.window_size_left = -1;
    a.window_size_right = -1;
    a.is_varlen = true;
    a.is_paged = false;
    a.is_causal = f.causal;
    a.is_local = false;
    a.is_sink = false;
    // softmax_lse (num_heads_q, total_seqlen_q): the stride along the query rows is total_seqlen_q (fmha_xe2.cpp)
    a.softmax_lse = f.lse;
    a.lse_stride = f.T;
    // varlen Q / O [total_seq, num_heads, head_size]; non-paged K / V [total_seq_k, num_heads_kv, head_size]
    a.q_stride_seq = f.H * kHeadDim;
    a.q_stride_heads = kHeadDim;
    a.q_stride_batch = 0;
    a.o_stride_seq = f.H * kHeadDim;
    a.o_stride_heads = kHeadDim;
    a.o_stride_batch = 0;
    a.k_stride_seq = f.HKV * kHeadDim;
    a.k_stride_heads = kHeadDim;
    a.k_stride_batch = 0;
    a.v_stride_seq = f.HKV * kHeadDim;
    a.v_stride_heads = kHeadDim;
    a.v_stride_batch = 0;
    a.is_prefill = nullptr;
    a.page_stride_elements = 0;
    if (f.causal)
        Cfg<true>::kernel_dispatch(q, a);
    else
        Cfg<false>::kernel_dispatch(q, a);
}

}  // namespace q36tla

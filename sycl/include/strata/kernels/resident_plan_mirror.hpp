// SYCL port: experts mirrored in pinned host memory (GgufExpertSource::mirror) in the device-built verify plan.
#pragma once
#include <cstdint>
#include <string>
namespace strata::kernels {
/// From now on (and in every graph captured after this call) the device-built plan (resident_plan) treats an expert
/// that is not in the VRAM cache but has a non-zero entry in `mirror_table` ([n_layers][n_expert] device-readable
/// addresses, 0 = none) as planned: the expert kernels read it from that address, over PCIe. `d_res` is the residency
/// table whose per-layer slices resident_plan receives, so the layer of a call is found from its pointer.
/// `entries` (#1980): the length of `d_res`, `mirror_table` and `usage` (n_layers x n_expert).  A call whose residency
/// slice lies outside `d_res` - a later layer-split stage's own table - gets no mirror and no counts (a plain miss): the
/// slice used to be taken as an offset from `d_res` anyway and read another allocation (xe page fault on the second card).
/// 0: unchecked, as before.
void resident_plan_set_mirror(const int32_t* d_res, const unsigned long long* mirror_table, int32_t* usage = nullptr,
                              long long entries = 0);
/// STRATA_MIRROR_STAGE: the resident_plan calls that follow (until reset with nullptr) number the host-mirrored groups
/// after every VRAM group (counts[3] = the VRAM groups), point mirrored group j at base + j * stride (VRAM) in both lists
/// and list them on their own (ptr2 / start2 / counts[2], the PCIe share's slots, which the device plan leaves empty),
/// their mirror addresses in src[j].
void resident_plan_set_stage(uint8_t* base, long long stride, unsigned long long* src, bool ring = false);
/// STRATA_MIRROR_RING: one VRAM ring of `slots` blobs (`stride` bytes, the stage stride) shared by every verifier, made
/// once after resident_plan_set_mirror; resident_plan_set_stage(..., true) then plans mirrored groups through it.
bool mirror_ring_create(int slots, long long stride, std::string& err);
/// The ring's size in blobs: STRATA_MIRROR_RING=N (0 off), default 1024; 0 without the lean device plan
/// (STRATA_VERIFY_DEVICE_PLAN + STRATA_VERIFY_NO_HOST) or with STRATA_MIRROR_STAGE=0.  The auto-sized expert cache leaves it.
int mirror_ring_slots();
/// The ring the expert cache made room for (generate: the slots it gave up, after the mirror check); 0 = none, and then
/// the verifier makes no ring, so the ring never takes VRAM the cache's budget did not leave it.
void mirror_ring_grant(int slots);
int mirror_ring_granted();
/// Copy the plan's mirrored list (src[j] -> ptr2[j], counts[2] blobs of blob_bytes) into VRAM.
void mirror_stage(const int32_t* plan, long long capx, long long blob_bytes, const unsigned long long* src, void* stream);
}

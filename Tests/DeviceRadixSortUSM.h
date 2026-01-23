#pragma once
#include <cstdint>

namespace ARBD {
// Forward declaration to avoid pulling in full Resource.h
class Resource;

#ifndef ARBD_SORT_CONFIG_H
#define ARBD_SORT_CONFIG_H

#include <cstdint>

// ----------------------------------------------------------------------------
// 1. Hardware Detection (Wave32 vs Wave64)
// ----------------------------------------------------------------------------
// You can also define SG_SIZE via compiler flags (-DSG_SIZE=64)
#ifndef SG_SIZE
#if defined(__AMDGCN_WAVEFRONT_SIZE) && (__AMDGCN_WAVEFRONT_SIZE == 64)
#define SG_SIZE 64
#else
#define SG_SIZE 32 // Default for NVIDIA (Delta) and Intel
#endif
#endif

// ----------------------------------------------------------------------------
// 2. Dynamic Configuration
// ----------------------------------------------------------------------------
constexpr uint32_t DRS_SUBGROUP_SIZE = SG_SIZE;

constexpr uint32_t DRS_RADIX = 256;
constexpr uint32_t DRS_RADIX_MASK = 255;
constexpr uint32_t DRS_RADIX_LOG = 8;

constexpr uint32_t DRS_BIN_KEYS_PER_THREAD = 16;

// Number of Warps/Wavefronts per Threadblock
// Note: On AMD (Wave64), 16 waves * 64 threads = 1024 threads (Max block size).
// This is safe, but high. If you hit register limits, reduce this to 8 for AMD.
constexpr uint32_t DRS_BIN_WARPS = 16;

// Threads per block: (16 * 32 = 512) or (16 * 64 = 1024)
constexpr uint32_t DRS_BIN_THREADS = DRS_BIN_WARPS * DRS_SUBGROUP_SIZE;

// Items processed per single warp/wavefront: (15 * 32 = 480) or (15 * 64 = 960)
constexpr uint32_t DRS_BIN_SUB_PART_SIZE =
    DRS_BIN_KEYS_PER_THREAD * DRS_SUBGROUP_SIZE;

// Total items processed per threadblock (Partition Size)
// (16 * 480 = 7680) or (16 * 960 = 15360)
constexpr uint32_t DRS_BIN_PART_SIZE = DRS_BIN_WARPS * DRS_BIN_SUB_PART_SIZE;

// Usually Upsweep uses the same partition size logic
constexpr uint32_t DRS_PART_SIZE = DRS_BIN_PART_SIZE;

// Vector part size (usually 1/4th of part size for vectorized loads, check your
// kernel logic)
constexpr uint32_t DRS_VEC_PART_SIZE = DRS_PART_SIZE / 4;

// Shared Memory for Histograms (16 warps * 256 bins = 4096 integers)
// This size is independent of Subgroup Size, so it stays constant.
constexpr uint32_t DRS_BIN_HISTS_SIZE = DRS_BIN_WARPS * DRS_RADIX;

// These are usually standalone and safe to keep fixed
constexpr uint32_t DRS_SCAN_THREADS = 128;
constexpr uint32_t DRS_UPSWEEP_THREADS = 512;

#endif // ARBD_SORT_CONFIG_H
// DeviceRadixSort using SYCL USM - simpler and more direct implementation

float device_radix_sort_pairs_usm(const Resource &device, uint32_t *keys,
                                  uint32_t *payloads, uint32_t *alt_keys,
                                  uint32_t *alt_payloads,
                                  uint32_t *globalHistogram,
                                  uint32_t *passHistogram, uint32_t size);

float device_radix_sort_pairs_cub(int device_id, uint32_t *keys,
                                  uint32_t *payloads, uint32_t *alt_keys,
                                  uint32_t *alt_payloads, uint32_t size);
} // namespace ARBD


#ifdef USE_SYCL
#include "DeviceRadixSortUSM.h"
#include "ARBDLogger.h"
#include "Backend/Resource.h"
#include <iostream>
#include <sycl/sycl.hpp>

namespace ARBD {

float device_radix_sort_pairs_usm(const Resource &device, uint32_t *keys,
                                  uint32_t *payloads, uint32_t *alt_keys,
                                  uint32_t *alt_payloads,
                                  uint32_t *globalHistogram,
                                  uint32_t *passHistogram, uint32_t size) {
  // Get SYCL queue from resource
  sycl::queue &q = *static_cast<sycl::queue *>(device.get_stream());

  const uint32_t threadBlocks = (size + DRS_PART_SIZE - 1) / DRS_PART_SIZE;
  if (threadBlocks > 10000000) {
    LOGWARN("Launching {} blocks. Input size may be too large.", threadBlocks);
  }
  LOGWARN("SYCL Radix Sort should be performed on empty GPU. If not this won't "
          "sort. This function "
          "did not check racing conditions.");

  LOGINFO("DeviceRadixSortUSM: size={}, threadBlocks={} on device {}", size,
          threadBlocks, device.id());

  auto start_event = q.submit([&](sycl::handler &h) {
    h.single_task([]() {}); // Empty marker kernel
  });
  auto E_global_clear = q.submit([&](sycl::handler &h) {
    h.depends_on(start_event);
    h.memset(globalHistogram, 0, DRS_RADIX * 4 * sizeof(uint32_t));
  });
  sycl::event E_last = E_global_clear;
  // Four radix passes
  for (uint32_t pass = 0; pass < 4; ++pass) {
    const uint32_t radixShift = pass * 8;
    // Reset histograms
    auto E_pass = q.submit([&](sycl::handler &h) {
      h.depends_on(E_last);
      h.memset(passHistogram, 0, DRS_RADIX * threadBlocks * sizeof(uint32_t));
    });

    uint32_t *input_keys = (pass % 2 == 0) ? keys : alt_keys;
    uint32_t *input_payloads = (pass % 2 == 0) ? payloads : alt_payloads;
    uint32_t *output_keys = (pass % 2 == 0) ? alt_keys : keys;
    uint32_t *output_payloads = (pass % 2 == 0) ? alt_payloads : payloads;

    // ========================================================================
    // Upsweep: Compute histograms per work-group
    // ========================================================================
    E_last = q.submit([&](sycl::handler &h) {
      h.depends_on({E_pass});
      sycl::local_accessor<uint32_t, 1> s_globalHist(DRS_RADIX * 2, h);

      h.parallel_for(
          sycl::nd_range<1>(threadBlocks * DRS_UPSWEEP_THREADS,
                            DRS_UPSWEEP_THREADS),
          [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t local_id = item.get_local_id(0);
            const uint32_t group_id = item.get_group(0);
            const uint32_t block_dim = item.get_local_range(0);

            // Clear shared memory
            for (uint32_t i = local_id; i < DRS_RADIX * 2; i += block_dim)
              s_globalHist[i] = 0;
            item.barrier(sycl::access::fence_space::local_space);

            // Histogram - 256 threads per histogram
            uint32_t *s_wavesHist = &s_globalHist[(local_id / 256) * DRS_RADIX];

            if (group_id < threadBlocks - 1) {
              const uint32_t partEnd = (group_id + 1) * DRS_VEC_PART_SIZE;
              for (uint32_t i = local_id + (group_id * DRS_VEC_PART_SIZE);
                   i < partEnd; i += block_dim) {
                const uint32_t base_idx = i * 4;
                if (base_idx + 3 < size) {
                  const uint32_t k0 = input_keys[base_idx + 0];
                  const uint32_t k1 = input_keys[base_idx + 1];
                  const uint32_t k2 = input_keys[base_idx + 2];
                  const uint32_t k3 = input_keys[base_idx + 3];

                  sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                   sycl::memory_scope::work_group,
                                   sycl::access::address_space::local_space>
                      a0(s_wavesHist[(k0 >> radixShift) & DRS_RADIX_MASK]);
                  sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                   sycl::memory_scope::work_group,
                                   sycl::access::address_space::local_space>
                      a1(s_wavesHist[(k1 >> radixShift) & DRS_RADIX_MASK]);
                  sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                   sycl::memory_scope::work_group,
                                   sycl::access::address_space::local_space>
                      a2(s_wavesHist[(k2 >> radixShift) & DRS_RADIX_MASK]);
                  sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                   sycl::memory_scope::work_group,
                                   sycl::access::address_space::local_space>
                      a3(s_wavesHist[(k3 >> radixShift) & DRS_RADIX_MASK]);

                  a0.fetch_add(1);
                  a1.fetch_add(1);
                  a2.fetch_add(1);
                  a3.fetch_add(1);
                }
              }
            }

            if (group_id == threadBlocks - 1) {
              for (uint32_t i = local_id + (group_id * DRS_PART_SIZE); i < size;
                   i += block_dim) {
                const uint32_t t = input_keys[i];
                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                 sycl::memory_scope::work_group,
                                 sycl::access::address_space::local_space>
                    a(s_wavesHist[(t >> radixShift) & DRS_RADIX_MASK]);
                a.fetch_add(1);
              }
            }
            item.barrier(sycl::access::fence_space::local_space);

            // Reduce to first hist, pass out, begin prefix sum
            for (uint32_t i = local_id; i < DRS_RADIX; i += block_dim) {
              s_globalHist[i] += s_globalHist[i + DRS_RADIX];
              passHistogram[i * threadBlocks + group_id] = s_globalHist[i];
            }
            item.barrier(sycl::access::fence_space::local_space);

            // Compute exclusive prefix sum within work-group
            for (uint32_t i = local_id; i < DRS_RADIX; i += block_dim) {
              uint32_t val = s_globalHist[i];
              auto sg = item.get_sub_group();
              s_globalHist[i] = sycl::exclusive_scan_over_group(
                  sg, val, sycl::plus<uint32_t>());
            }
            item.barrier(sycl::access::fence_space::local_space);

            // Add to global histogram (just the count, not inclusive sum)
            for (uint32_t i = local_id; i < DRS_RADIX; i += block_dim) {
              uint32_t original = passHistogram[i * threadBlocks + group_id];
              sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                               sycl::memory_scope::device,
                               sycl::access::address_space::global_space>
                  a(globalHistogram[i + (radixShift << 5)]);
              a.fetch_add(original);
            }
          });
    });
    // q.wait();
    LOGDEBUG("Pass {} upsweep completed", pass);

    // ========================================================================
    // Scan: Compute prefix sums across work-groups
    // ========================================================================
    E_last = q.submit([&](sycl::handler &h) {
      h.depends_on(E_last);
      sycl::local_accessor<uint32_t, 1> s_scan(DRS_SCAN_THREADS, h);

      h.parallel_for(
          sycl::nd_range<1>(DRS_RADIX * DRS_SCAN_THREADS, DRS_SCAN_THREADS),
          [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t local_id = item.get_local_id(0);
            const uint32_t group_id = item.get_group(0);
            const uint32_t block_dim = item.get_local_range(0);
            auto group = item.get_group(); // ← Use work-group, not sub-group!

            uint32_t reduction = 0;
            const uint32_t partitionsEnd = threadBlocks / block_dim * block_dim;
            const uint32_t digitOffset = group_id * threadBlocks;

            uint32_t i = local_id;
            for (; i < partitionsEnd; i += block_dim) {
              uint32_t original = passHistogram[i + digitOffset];

              // Exclusive scan over the ENTIRE work-group (all 128 threads)
              uint32_t exclusive = sycl::exclusive_scan_over_group(
                  group, original, sycl::plus<uint32_t>());

              // Also get the total sum (for reduction)
              uint32_t inclusive = sycl::inclusive_scan_over_group(
                  group, original, sycl::plus<uint32_t>());

              passHistogram[i + digitOffset] = exclusive + reduction;

              // Use the last thread's inclusive value as the total
              uint32_t total =
                  sycl::group_broadcast(group, inclusive, block_dim - 1);
              reduction += total;
            }

            // Handle remaining elements
            if (i < threadBlocks) {
              uint32_t original = passHistogram[i + digitOffset];

              uint32_t exclusive = sycl::exclusive_scan_over_group(
                  group, original, sycl::plus<uint32_t>());

              uint32_t inclusive = sycl::inclusive_scan_over_group(
                  group, original, sycl::plus<uint32_t>());

              passHistogram[i + digitOffset] = exclusive + reduction;
            }
          });
    });
    // q.wait();
    LOGDEBUG("Pass {} scan completed", pass);

    // Compute exclusive prefix sum of global histogram on device
    // This is needed because globalHistogram is used as base offset in
    // downsweep
    E_last = q.submit([&](sycl::handler &h) {
      h.depends_on(E_last);
      sycl::local_accessor<uint32_t, 1> s_counts(DRS_RADIX, h);

      h.parallel_for(
          sycl::nd_range<1>(DRS_RADIX, DRS_RADIX),
          [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t local_id = item.get_local_id(0);
            const uint32_t global_offset = radixShift << 5;

            // Load counts into local memory
            s_counts[local_id] = globalHistogram[global_offset + local_id];
            item.barrier(sycl::access::fence_space::local_space);

            // Compute exclusive prefix sum sequentially (single thread)
            if (local_id == 0) {
              uint32_t sum = 0;
              for (uint32_t i = 0; i < DRS_RADIX; i++) {
                uint32_t count = s_counts[i];
                s_counts[i] = sum; // exclusive prefix sum
                sum += count;
              }
            }
            item.barrier(sycl::access::fence_space::local_space);

            // Write back to global memory
            globalHistogram[global_offset + local_id] = s_counts[local_id];
          });
    });
    // q.wait();

    // ========================================================================
    // Downsweep: Scatter keys and payloads
    // ========================================================================
    E_last = q.submit([&](sycl::handler &h) {
      h.depends_on(E_last);
      sycl::local_accessor<uint32_t, 1> s_warpHistograms(DRS_BIN_PART_SIZE, h);
      sycl::local_accessor<uint32_t, 1> s_localHistogram(DRS_RADIX, h);

      h.parallel_for(
          sycl::nd_range<1>(threadBlocks * DRS_BIN_THREADS, DRS_BIN_THREADS),
          [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t local_id = item.get_local_id(0);
            const uint32_t group_id = item.get_group(0);
            auto sg = item.get_sub_group();
            const uint32_t warp_index = local_id / SG_SIZE;
            const uint32_t lane_id = sg.get_local_linear_id();

            uint32_t *s_warpHist = &s_warpHistograms[warp_index * DRS_RADIX];

            // Clear shared memory
            for (uint32_t i = local_id; i < DRS_BIN_HISTS_SIZE;
                 i += DRS_BIN_THREADS)
              s_warpHistograms[i] = 0;
            item.barrier(sycl::access::fence_space::local_space);

            // Load keys
            uint32_t keys[DRS_BIN_KEYS_PER_THREAD];
            const uint32_t bin_sub_part_start =
                warp_index * DRS_BIN_SUB_PART_SIZE;
            const uint32_t bin_part_start = group_id * DRS_BIN_PART_SIZE;

            if (group_id < threadBlocks - 1) {
#pragma unroll
              for (uint32_t i = 0,
                            t = lane_id + bin_sub_part_start + bin_part_start;
                   i < DRS_BIN_KEYS_PER_THREAD; ++i, t += SG_SIZE)
                keys[i] = input_keys[t];
            }

            if (group_id == threadBlocks - 1) {
#pragma unroll
              for (uint32_t i = 0,
                            t = lane_id + bin_sub_part_start + bin_part_start;
                   i < DRS_BIN_KEYS_PER_THREAD; ++i, t += SG_SIZE)
                keys[i] = t < size ? input_keys[t] : 0xffffffff;
            }
            item.barrier(sycl::access::fence_space::local_space);

            // Simple histogram-based offset calculation
            bool is_valid[DRS_BIN_KEYS_PER_THREAD];
            uint16_t offsets[DRS_BIN_KEYS_PER_THREAD];
#pragma unroll
            for (uint32_t i = 0; i < DRS_BIN_KEYS_PER_THREAD; ++i) {
              uint32_t t =
                  lane_id + bin_sub_part_start + bin_part_start + (i * SG_SIZE);
              is_valid[i] = (t < size);
              if (is_valid[i]) {
                const uint32_t digit = (keys[i] >> radixShift) & DRS_RADIX_MASK;
                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                 sycl::memory_scope::work_group,
                                 sycl::access::address_space::local_space>
                    a(s_warpHist[digit]);
                offsets[i] = a.fetch_add(1);
              } else {
                offsets[i] = 0xffff; // Padding gets invalid offset
              }
            }
            item.barrier(sycl::access::fence_space::local_space);

            // Exclusive prefix sum up the warp histograms
            if (local_id < DRS_RADIX) {
              uint32_t reduction = s_warpHistograms[local_id];
              const uint32_t num_warps = DRS_BIN_HISTS_SIZE / DRS_RADIX;
              for (uint32_t warp = 1; warp < num_warps; ++warp) {
                const uint32_t idx = local_id + warp * DRS_RADIX;
                const uint32_t warp_val = s_warpHistograms[idx];
                reduction += warp_val;
                s_warpHistograms[idx] = reduction - warp_val;
              }
              s_localHistogram[local_id] = reduction;
            }
            item.barrier(sycl::access::fence_space::local_space);

            // Exclusive scan: use sub-group scan for each digit
            if (local_id == 0) {
              uint32_t sum = 0;
              for (uint32_t i = 0; i < DRS_RADIX; ++i) {
                uint32_t val = s_localHistogram[i];
                s_warpHistograms[i] = sum; // exclusive
                sum += val;
              }
            }
            item.barrier(sycl::access::fence_space::local_space);

            // Update offsets
            if (warp_index) {
#pragma unroll
              for (uint32_t i = 0; i < DRS_BIN_KEYS_PER_THREAD; ++i) {
                if (is_valid[i]) {
                  const uint32_t t2 = (keys[i] >> radixShift) & DRS_RADIX_MASK;
                  offsets[i] += s_warpHist[t2] + s_warpHistograms[t2];
                }
              }
            } else {
#pragma unroll
              for (uint32_t i = 0; i < DRS_BIN_KEYS_PER_THREAD; ++i) {
                if (is_valid[i]) {
                  offsets[i] += s_warpHistograms[(keys[i] >> radixShift) &
                                                 DRS_RADIX_MASK];
                }
              }
            }

            // Load threadblock reductions
            if (local_id < DRS_RADIX) {
              s_localHistogram[local_id] =
                  globalHistogram[local_id + (radixShift << 5)] +
                  passHistogram[local_id * threadBlocks + group_id] -
                  s_warpHistograms[local_id];
            }
            item.barrier(sycl::access::fence_space::local_space);

        // Scatter keys into shared memory
#pragma unroll
            for (uint32_t i = 0; i < DRS_BIN_KEYS_PER_THREAD; ++i) {
              if (is_valid[i]) { // FIX: Guard Shared Mem Write
                s_warpHistograms[offsets[i]] = keys[i];
              }
            }
            item.barrier(sycl::access::fence_space::local_space);

            // Scatter keys to device memory
            uint8_t digits[DRS_BIN_KEYS_PER_THREAD];
            if (group_id < threadBlocks - 1) {
#pragma unroll
              for (uint32_t i = 0, t = local_id; i < DRS_BIN_KEYS_PER_THREAD;
                   ++i, t += DRS_BIN_THREADS) {
                digits[i] =
                    (s_warpHistograms[t] >> radixShift) & DRS_RADIX_MASK;
                output_keys[s_localHistogram[digits[i]] + t] =
                    s_warpHistograms[t];
              }
              item.barrier(sycl::access::fence_space::local_space);

          // Load payloads
#pragma unroll
              for (uint32_t i = 0,
                            t = lane_id + bin_sub_part_start + bin_part_start;
                   i < DRS_BIN_KEYS_PER_THREAD; ++i, t += SG_SIZE) {
                keys[i] = input_payloads[t];
              }

          // Scatter payloads into shared memory
#pragma unroll
              for (uint32_t i = 0; i < DRS_BIN_KEYS_PER_THREAD; ++i) {
                if (is_valid[i]) { // FIX: Guard Shared Mem Write
                  s_warpHistograms[offsets[i]] = keys[i];
                }
              }
              item.barrier(sycl::access::fence_space::local_space);

          // Scatter payloads to device
#pragma unroll
              for (uint32_t i = 0, t = local_id; i < DRS_BIN_KEYS_PER_THREAD;
                   ++i, t += DRS_BIN_THREADS) {
                output_payloads[s_localHistogram[digits[i]] + t] =
                    s_warpHistograms[t];
              }
            }

            if (group_id == threadBlocks - 1) {
              const uint32_t finalPartSize = size - bin_part_start;
#pragma unroll
              for (uint32_t i = 0, t = local_id; i < DRS_BIN_KEYS_PER_THREAD;
                   ++i, t += DRS_BIN_THREADS) {
                if (t < finalPartSize) {
                  digits[i] =
                      (s_warpHistograms[t] >> radixShift) & DRS_RADIX_MASK;
                  output_keys[s_localHistogram[digits[i]] + t] =
                      s_warpHistograms[t];
                }
              }
              item.barrier(sycl::access::fence_space::local_space);

#pragma unroll
              for (uint32_t i = 0,
                            t = lane_id + bin_sub_part_start + bin_part_start;
                   i < DRS_BIN_KEYS_PER_THREAD; ++i, t += SG_SIZE) {
                if (t < size)
                  keys[i] = input_payloads[t];
              }

          // Scatter remaining payloads into shared memory
#pragma unroll
              for (uint32_t i = 0; i < DRS_BIN_KEYS_PER_THREAD; ++i) {
                if (is_valid[i]) { // FIX: Guard Shared Mem Write
                  s_warpHistograms[offsets[i]] = keys[i];
                }
              }
              item.barrier(sycl::access::fence_space::local_space);

#pragma unroll
              for (uint32_t i = 0, t = local_id; i < DRS_BIN_KEYS_PER_THREAD;
                   ++i, t += DRS_BIN_THREADS) {
                if (t < finalPartSize)
                  output_payloads[s_localHistogram[digits[i]] + t] =
                      s_warpHistograms[t];
              }
            }
          });
    });
    // E_last.wait();
    LOGDEBUG("Pass {} downsweep completed", pass);
  }
  auto end_event = q.submit([&](sycl::handler &h) {
    h.single_task([]() {}); // Empty marker kernel
  });
  end_event.wait();

  // Calculate elapsed time
  auto start_time =
      start_event
          .get_profiling_info<sycl::info::event_profiling::command_start>();
  auto end_time =
      end_event.get_profiling_info<sycl::info::event_profiling::command_end>();

  double milliseconds =
      (end_time - start_time) / 1e6; // nanoseconds to milliseconds
                                     // Determine final output location (after 4
                                     // passes: pass 0->alt, 1->keys,
  // 2->alt, 3->keys) So after pass 3, output is in keys
  uint32_t *final_keys = keys;
  uint32_t *final_payloads = payloads;
  return milliseconds;
}

} // namespace ARBD
#endif // USE_SYCL

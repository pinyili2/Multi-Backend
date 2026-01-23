#include "Backend/Buffer.h"
#include "Backend/KernelConfig.h"
#include "Backend/Kernels.h"
#include "Backend/Resource.h"
#include "DeviceRadixSortUSM.h"
#include "Random/Random.h"
#include "catch_boiler.h"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <vector>

using namespace ARBD;
using namespace Tests;

static int device_id = 1;
void generate_random_data_drs(const Resource &device,
                              std::vector<uint32_t> &data, uint32_t seed,
                              float entropy = 1.0f) {
  Random<Resource> rng(device, 12);
  rng.init(seed, 0);
  DeviceBuffer<uint32_t> d_data(data.size(), device.id());
  uint32_t min_val = 0;
  uint32_t max_val;
  if (entropy >= 0.99f) {
    max_val = UINT32_MAX;
  } else if (entropy >= 0.8f) {
    max_val = (1u << 26) - 1;
  } else if (entropy >= 0.5f) {
    max_val = (1u << 18) - 1;
  } else {
    max_val = (1u << 7) - 1;
  }
  Event gen_event = rng.generate_uniform(d_data, min_val, max_val);
  gen_event.wait();
  d_data.copy_to_host(data.data(), data.size());
}

TEST_CASE("DeviceRadixSort Key-Value Pairs - Small",
          "[deviceradix][sort][pairs][small]") {
  initialize_backend_once();
  Resource device(device_id);
  const uint32_t size = 1024;

  SECTION("Sort 1K elements") {
    std::vector<uint32_t> h_keys(size);
    std::vector<uint32_t> h_payloads(size);
    generate_random_data_drs(device, h_keys, 12345);
    std::iota(h_payloads.begin(), h_payloads.end(), 0);

    DeviceBuffer<uint32_t> d_keys(size, device.id());
    DeviceBuffer<uint32_t> d_payloads(size, device.id());
    d_keys.copy_from_host(h_keys.data(), size);
    d_payloads.copy_from_host(h_payloads.data(), size);

    DeviceBuffer<uint32_t> d_alt_keys(size, device.id());
    DeviceBuffer<uint32_t> d_alt_payloads(size, device.id());
    DeviceBuffer<uint32_t> d_globalHistogram(DRS_RADIX * 8, device.id());
    const uint32_t threadBlocks = (size + DRS_PART_SIZE - 1) / DRS_PART_SIZE;
    DeviceBuffer<uint32_t> d_passHistogram(DRS_RADIX * threadBlocks,
                                           device.id());
    device_radix_sort_pairs_usm(device, d_keys.data(), d_payloads.data(),
                                d_alt_keys.data(), d_alt_payloads.data(),
                                d_globalHistogram.data(),
                                d_passHistogram.data(), size);

    std::vector<uint32_t> h_sorted_keys(size);
    std::vector<uint32_t> h_sorted_payloads(size);
    d_keys.copy_to_host(h_sorted_keys.data(), size);
    d_payloads.copy_to_host(h_sorted_payloads.data(), size);

    // Verify keys are sorted
    for (uint32_t i = 1; i < size; ++i) {
      REQUIRE(h_sorted_keys[i - 1] <= h_sorted_keys[i]);
    }

    // Verify stability: payloads should maintain relative order for equal keys
    std::vector<std::pair<uint32_t, uint32_t>> original(size);
    for (uint32_t i = 0; i < size; ++i) {
      original[i] = {h_keys[i], h_payloads[i]};
    }
    std::stable_sort(
        original.begin(), original.end(),
        [](const auto &a, const auto &b) { return a.first < b.first; });

    for (uint32_t i = 0; i < size; ++i) {
      REQUIRE(h_sorted_keys[i] == original[i].first);
      REQUIRE(h_sorted_payloads[i] == original[i].second);
    }
  }
}

TEST_CASE("DeviceRadixSort Key-Value Pairs - Medium",
          "[cub][sort][pairs][medium]") {
  initialize_backend_once();
  Resource device(device_id);
  const size_t size = 1024 * 1024 * 1024;

  SECTION("Sort 1M elements") {
    std::vector<uint32_t> h_keys(size);
    std::vector<uint32_t> h_payloads(size);
    generate_random_data_drs(device, h_keys, 54321);
    std::iota(h_payloads.begin(), h_payloads.end(), 0);

    DeviceBuffer<uint32_t> d_keys(size, device.id());
    DeviceBuffer<uint32_t> d_payloads(size, device.id());
    d_keys.copy_from_host(h_keys.data(), size);
    d_payloads.copy_from_host(h_payloads.data(), size);

    DeviceBuffer<uint32_t> d_alt_keys(size, device.id());
    DeviceBuffer<uint32_t> d_alt_payloads(size, device.id());

    auto start = std::chrono::high_resolution_clock::now();
    auto stream = device.get_stream();

    device_radix_sort_pairs_cub(device.id(), d_keys.data(), d_payloads.data(),
                                d_alt_keys.data(), d_alt_payloads.data(), size);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "CUB: Sorted " << size << " elements in " << duration.count()
              << " ms" << std::endl;

    std::vector<uint32_t> h_sorted_keys(size);
    d_keys.copy_to_host(h_sorted_keys.data(), size);

    // Verify keys are sorted
    for (uint32_t i = 1; i < size; ++i) {
      REQUIRE(h_sorted_keys[i - 1] <= h_sorted_keys[i]);
    }
  }
}
TEST_CASE("DeviceRadixSort Key-Value Pairs - Medium",
          "[usm][sort][pairs][medium]") {
  initialize_backend_once();
  Resource device(device_id);
  const size_t size = 1024 * 1024 * 1024;

  SECTION("Sort 1G elements") {
    std::vector<uint32_t> h_keys(size);
    std::vector<uint32_t> h_payloads(size);
    generate_random_data_drs(device, h_keys, 54321);
    std::iota(h_payloads.begin(), h_payloads.end(), 0);

    DeviceBuffer<uint32_t> d_keys(size, device.id());
    DeviceBuffer<uint32_t> d_payloads(size, device.id());
    d_keys.copy_from_host(h_keys.data(), size);
    d_payloads.copy_from_host(h_payloads.data(), size);

    DeviceBuffer<uint32_t> d_alt_keys(size, device.id());
    DeviceBuffer<uint32_t> d_alt_payloads(size, device.id());
    DeviceBuffer<uint32_t> d_globalHistogram(DRS_RADIX * 8, device.id());
    const uint32_t threadBlocks = (size + DRS_PART_SIZE - 1) / DRS_PART_SIZE;
    DeviceBuffer<uint32_t> d_passHistogram(DRS_RADIX * threadBlocks,
                                           device.id());
    auto start = std::chrono::high_resolution_clock::now();

    device_radix_sort_pairs_usm(device, d_keys.data(), d_payloads.data(),
                                d_alt_keys.data(), d_alt_payloads.data(),
                                d_globalHistogram.data(),
                                d_passHistogram.data(), size);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "USM: Sorted " << size << " elements in " << duration.count()
              << " ms" << std::endl;

    std::vector<uint32_t> h_sorted_keys(size);
    d_keys.copy_to_host(h_sorted_keys.data(), size);

    // Verify keys are sorted
    for (uint32_t i = 1; i < size; ++i) {
      REQUIRE(h_sorted_keys[i - 1] <= h_sorted_keys[i]);
      if (h_sorted_keys[i - 1] > h_sorted_keys[i]) {
        std::cout << "Sorting error at index " << i - 1 << "->" << i << ": "
                  << h_sorted_keys[i - 1] << " > " << h_sorted_keys[i]
                  << std::endl;
        break;
      }
    }
  }
}

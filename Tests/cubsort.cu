#include <cstdint>
#include <cuda_runtime.h>

#ifdef Debug
#undef Debug
#include <cub/cub.cuh>
#define Debug(x) static_cast<void>(0)
#else
#include <cub/cub.cuh>
#endif

#include "DeviceRadixSortUSM.h"

namespace ARBD {

// CUB-based radix sort implementation
float device_radix_sort_pairs_cub(int device_id, uint32_t *d_keys,
                                  uint32_t *d_payloads, uint32_t *d_alt_keys,
                                  uint32_t *d_alt_payloads, uint32_t size) {
  // Set the CUDA device

  cudaSetDevice(device_id);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  // Use default CUDA stream (nullptr)
  // Caller should synchronize if needed
  cudaStream_t stream = nullptr;
  // 1. Wrap pointers in DoubleBuffer
  // This tells CUB it can swap between d_keys <-> d_alt_keys as needed
  cub::DoubleBuffer<uint32_t> d_keys_db(d_keys, d_alt_keys);
  cub::DoubleBuffer<uint32_t> d_values_db(d_payloads, d_alt_payloads);

  // 2. Determine temporary device storage requirements
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  cudaError_t err = cub::DeviceRadixSort::SortPairs(
      d_temp_storage, temp_storage_bytes, d_keys_db, d_values_db, size, 0,
      sizeof(uint32_t) * 8, stream);
  if (err != cudaSuccess) {
    printf("CUB size query error: %s\n", cudaGetErrorString(err));
    return 0;
  }
  printf("CUB requires %zu bytes of temp storage for %u elements\n",
         temp_storage_bytes, size);

  // 3. Allocate temporary storage
  // In production, you might want to use a caching allocator here
  err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
  if (err != cudaSuccess) {
    printf("cudaMalloc error: %s\n", cudaGetErrorString(err));
    return 0;
  }

  cudaEventRecord(start, stream);
  // 4. Run sorting operation
  err = cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                        d_keys_db, d_values_db, size, 0,
                                        sizeof(uint32_t) * 8, stream);
  if (err != cudaSuccess) {
    printf("CUB sort error: %s\n", cudaGetErrorString(err));
  }
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  // ============ END TIMER ============

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // 5. Cleanup Temp Storage
  cudaFree(d_temp_storage);

  // 6. Ensure result is in the original buffers
  // CUB might leave the sorted data in the "alt" buffer depending on pass
  // count. If so, copy it back to the primary buffer to match your interface
  // expectation.
  if (d_keys_db.Current() != d_keys) {
    cudaError_t err1 =
        cudaMemcpy(d_keys, d_keys_db.Current(), size * sizeof(uint32_t),
                   cudaMemcpyDeviceToDevice);
    cudaError_t err2 =
        cudaMemcpy(d_payloads, d_values_db.Current(), size * sizeof(uint32_t),
                   cudaMemcpyDeviceToDevice);
    if (err1 != cudaSuccess || err2 != cudaSuccess) {
      printf("CUDA memcpy error: %s %s\n", cudaGetErrorString(err1),
             cudaGetErrorString(err2));
    }
  }

  // 7. Synchronize to ensure all CUDA operations complete before returning
  // This is critical when called from SYCL context to avoid race conditions
  cudaError_t sync_err = cudaDeviceSynchronize();
  if (sync_err != cudaSuccess) {
    printf("CUDA sync error: %s\n", cudaGetErrorString(sync_err));
  }

  // 8. Clear any pending CUDA errors before returning to SYCL
  // This prevents SYCL from seeing stale CUDA error states
  cudaGetLastError();
  return milliseconds;
}
} // namespace ARBD

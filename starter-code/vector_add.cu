#include "cuda_check.cuh"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

__global__ void vector_add_kernel(const float* x, const float* y, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = x[i] + y[i];
  }
}

int main() {
  const int n = 1 << 20;
  const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

  std::vector<float> h_x(n), h_y(n), h_out(n);
  for (int i = 0; i < n; ++i) {
    h_x[i] = static_cast<float>(i) * 0.5f;
    h_y[i] = static_cast<float>(i) * 2.0f;
  }

  float* d_x = nullptr;
  float* d_y = nullptr;
  float* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, bytes));
  CUDA_CHECK(cudaMalloc(&d_y, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));

  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  const int block_size = 256;
  const int grid_size = (n + block_size - 1) / block_size;

  CUDA_CHECK(cudaEventRecord(start));
  vector_add_kernel<<<grid_size, block_size>>>(d_x, d_y, d_out, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

  bool ok = true;
  for (int i = 0; i < n; ++i) {
    float expected = h_x[i] + h_y[i];
    if (std::fabs(h_out[i] - expected) > 1e-5f) {
      std::cerr << "Mismatch at " << i << ": got " << h_out[i]
                << ", expected " << expected << "\n";
      ok = false;
      break;
    }
  }

  if (ok) {
    std::cout << "OK: vector_add for n=" << n << "\n";
  }
  std::cout << "Kernel time: " << elapsed_ms << " ms\n";

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaFree(d_out));
  return ok ? 0 : 1;
}

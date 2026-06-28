#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

inline void cuda_check(cudaError_t status, const char* expr, const char* file, int line) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(status) +
                             " at " + file + ":" + std::to_string(line) +
                             " for " + expr);
  }
}

#define CUDA_CHECK(expr) cuda_check((expr), #expr, __FILE__, __LINE__)

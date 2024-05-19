#include <stdio.h>

extern int performCpuReduction(const size_t);
extern int performCudaReductionV0(const size_t);
extern int performCudaReductionV1(const size_t);
extern int performCudaReductionV2(const size_t);
extern int performCudaReductionV3(const size_t);
extern int performCudaReductionV4(const size_t);
extern int performCudaReductionV5(const size_t);
extern int performCudaReductionV6(const size_t);

int main(int args, char** argv) {
  size_t array_size = 1 << 28;
  printf("start profiling reduction with array size %zu    \n", array_size);
  performCpuReduction(array_size);
  performCudaReductionV0(array_size);
  performCudaReductionV1(array_size);
  performCudaReductionV2(array_size);
  performCudaReductionV3(array_size);
  performCudaReductionV4(array_size);
  performCudaReductionV5(array_size);
  performCudaReductionV6(array_size);
}

#include <stdio.h>

extern int performCpuReduction();
extern int performCudaReductionV0();
extern int performCudaReductionV1();
extern int performCudaReductionV2();
extern int performCudaReductionV3();
extern int performCudaReductionV4();
extern int performCudaReductionV5();
extern int performCudaReductionV6(const size_t);

int main(int args, char** argv) {
  size_t array_size = 1 << 28;
  // performCpuReduction();
  // performCudaReductionV0();
  // performCudaReductionV1();
  // performCudaReductionV2();
  // performCudaReductionV3();
  // performCudaReductionV4();
  // performCudaReductionV5();
  performCudaReductionV6(array_size);
}

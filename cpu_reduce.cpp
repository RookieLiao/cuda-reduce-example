#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "utils.h"

int cpuReduction(int* idata, size_t size) {
  if (size == 1) { return idata[0]; }
  size_t stride = (size + 1) / 2;
  for (size_t i = 0; i < stride; ++i) {
    if (i + stride < size) { idata[i] += idata[i + stride]; }
  }
  return cpuReduction(idata, stride);
}

int performCpuReduction() {
  // set up device
  int dev = 0;
  printf("starting reduction at cpu");

  size_t elem_size = 1 << 28;
  printf("    with array size %zu    \n", elem_size);
  size_t bytes = elem_size * sizeof(int);

  // allocate host memory
  int* idata_h = (int*)malloc(bytes);
  int* temp = (int*)malloc(bytes);

  initialize_array(idata_h, elem_size);

  double iStart, iElaps;

  // cpu reduction
  memcpy(temp, idata_h, bytes);

  iStart = seconds();
  int cpu_sum = cpuReduction(temp, elem_size);
  iElaps = seconds() - iStart;
  float cpu_bw = bytes / iElaps / 1e9;
  printf("cpu reduce elapsed %lf ms, bandwidth %lf GB/s, cpu_sum: %d\n", iElaps * 1e3, cpu_bw,
         cpu_sum);

  return EXIT_SUCCESS;
}

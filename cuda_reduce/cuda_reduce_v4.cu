#include <stdio.h>
#include "../utils.h"

extern int cpuReduction(int* idata, size_t size);

// unroll the last warp
__global__ void reduce4(int* idata_d, int* odata_d, size_t size) {
  extern __shared__ int sdata[];
  size_t tid = threadIdx.x;
  size_t g_idx = (2 * blockDim.x) * blockIdx.x + tid;

  // if out of boundary, just return
  if (g_idx + blockDim.x >= size) { return; }

  sdata[tid] = idata_d[g_idx] + idata_d[g_idx + blockDim.x];
  __syncthreads();

  for (size_t stride = blockDim.x / 2; stride > 32; stride >>= 1) {
    if (tid < stride) { sdata[tid] += sdata[tid + stride]; }
    __syncthreads();
  }

  // unroll last warp
  if (tid < 32) {
    volatile int* temp = static_cast<volatile int*>(sdata);
#pragma unroll
    for (size_t stride = 32; stride > 0; stride >>= 1) { temp[tid] += temp[tid + stride]; }
  }

  if (tid == 0) { odata_d[blockIdx.x] = sdata[0]; }
}

int performCudaReductionV4(const size_t elem_size) {
  // set up device
  int dev = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  printf("starting reduction at cuda_v4 device %d: %s\n", dev, deviceProp.name);

  size_t bytes = elem_size * sizeof(int);

  // allocate host memory
  int* idata_h = (int*)malloc(bytes);
  int* temp = (int*)malloc(bytes);

  initialize_array(idata_h, elem_size);

  double iStart, iElaps;

  // cpu reduction
  memcpy(temp, idata_h, bytes);

  int cpu_sum = cpuReduction(temp, elem_size);

  // cuda reduce
  size_t block_size = 128;

  dim3 block(block_size);
  dim3 grid0(((elem_size - 1) / (2 * block_size) + 1));
  dim3 grid1((grid0.x - 1) / (2 * block_size) + 1);
  dim3 grid2((grid1.x - 1) / (2 * block_size) + 1);

  // allocate device memory
  int* idata_d = NULL;
  int* odata_d0 = NULL;
  int* odata_d1 = NULL;
  int* odata_d2 = NULL;

  cudaMalloc((void**)(&idata_d), bytes);
  cudaMalloc((void**)(&odata_d0), grid0.x * sizeof(int));
  cudaMalloc((void**)(&odata_d1), grid1.x * sizeof(int));
  cudaMalloc((void**)(&odata_d2), grid2.x * sizeof(int));

  int* odata_h = (int*)malloc(grid2.x * sizeof(int));

  cudaMemcpy(idata_d, idata_h, bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  iStart = seconds();

  size_t smem_size = block_size * sizeof(int);
  reduce4<<<grid0, block, smem_size>>>(idata_d, odata_d0, elem_size);
  reduce4<<<grid1, block, smem_size>>>(odata_d0, odata_d1, grid0.x);
  reduce4<<<grid2, block, smem_size>>>(odata_d1, odata_d2, grid1.x);

  cudaMemcpy(odata_h, odata_d2, grid2.x * sizeof(int), cudaMemcpyDeviceToHost);
  int gpu_sum = cpuReduction(odata_h, grid2.x);
  iElaps = seconds() - iStart;
  float gpu_bw = bytes / iElaps / 1e9;
  printf("reduction_v4 elapsed %lf ms, bandwidth %lf GB/s\n", iElaps * 1e3, gpu_bw);

  free(idata_h);
  free(temp);

  cudaFree(idata_d);
  cudaFree(odata_d0);
  cudaFree(odata_d1);
  cudaFree(odata_d2);

  // check results
  bool bResult = (gpu_sum == cpu_sum);
  if (!bResult) printf("Test failed!\n");
  return EXIT_SUCCESS;
}

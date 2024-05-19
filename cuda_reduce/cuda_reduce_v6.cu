#include <stdio.h>
#include "../utils.h"

extern int cpuReduction(int* idata, size_t size);

// multiple add
template<size_t blockSize>
__global__ void reduce6(int* idata_d, int* odata_d, size_t size) {
  extern __shared__ int sdata[];
  size_t tid = threadIdx.x;
  size_t g_idx = (2 * blockSize) * blockIdx.x + tid;
  size_t gridSize = (blockSize * 2) * gridDim.x;

  sdata[tid] = 0;
  while (g_idx + blockSize < size) {
    sdata[g_idx] += idata_d[g_idx] + idata_d[g_idx+blockSize];
    g_idx += gridSize;
  }
  __syncthreads();

  sdata[tid] = idata_d[g_idx] + idata_d[g_idx + blockDim.x];
  __syncthreads();

  if (blockSize >= 512) {
    if (tid < 256) {
      sdata[tid] += sdata[tid + 256];
    }
    __syncthreads();
  }

  if (blockSize >= 256) {
    if (tid < 128) {
      sdata[tid] += sdata[tid + 128];
    }
    __syncthreads();
  }

  if (blockSize >= 128) {
    if (tid < 64) {
      sdata[tid] += sdata[tid + 64];
    }
    __syncthreads();
  }

  // unroll last warp
  if (tid < 32) {
    volatile int *temp = static_cast<volatile int *>(sdata);
#pragma unroll
    for (size_t stride = 32; stride > 0; stride >>= 1) {
      temp[tid] += temp[tid + stride];
    }
  }

  if (tid == 0) {
    odata_d[blockIdx.x] = sdata[0];
  }

}

int performCudaReductionV6(const size_t elem_size) {
  // set up device
  int dev = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  printf("starting reduction at cuda_v6 ");
  printf("device %d: %s ", dev, deviceProp.name);

  printf("    with array size %zu    \n", elem_size);
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
  constexpr size_t block_size = 128;

  dim3 block(block_size);
  dim3 grid0(128);

  // allocate device memory
  int* idata_d = NULL;
  int* odata_d0 = NULL;

  cudaMalloc((void**)(&idata_d), bytes);
  cudaMalloc((void**)(&odata_d0), grid0.x * sizeof(int));

  int* odata_h = (int*)malloc(grid0.x * sizeof(int));

  cudaMemcpy(idata_d, idata_h, bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  iStart = seconds();

  size_t smem_size = block_size * sizeof(int);
  reduce6<block_size><<<grid0, block, smem_size>>>(idata_d, odata_d0, elem_size);

  cudaMemcpy(odata_h, odata_d0, grid0.x * sizeof(int), cudaMemcpyDeviceToHost);
  int gpu_sum = cpuReduction(odata_h, grid0.x);
  iElaps = seconds() - iStart;
  float gpu_bw = bytes / iElaps / 1e9;
  printf("reduction_v6 elapsed %lf ms, bandwidth %lf GB/s\n", iElaps * 1e3, gpu_bw);

  free(idata_h);
  free(temp);

  cudaFree(idata_d);
  cudaFree(odata_d0);

  // check results
  bool bResult = (gpu_sum == cpu_sum);
  if (!bResult) printf("Test failed!\n");
  printf("cpu sum: %d, gpu sum: %d", cpu_sum, gpu_sum);
  return EXIT_SUCCESS;
}

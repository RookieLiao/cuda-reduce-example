extern int performCpuReduction();
extern int performCudaReductionV0();
extern int performCudaReductionV1();
extern int performCudaReductionV2();

int main(int args, char** argv) {
  // performCpuReduction();
  performCudaReductionV0();
  performCudaReductionV1();
  performCudaReductionV2();
}

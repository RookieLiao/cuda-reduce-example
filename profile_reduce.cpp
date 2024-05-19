extern int performCpuReduction();
extern int performCudaReductionV0();
extern int performCudaReductionV1();

int main(int args, char** argv) {
  // performCpuReduction();
  performCudaReductionV0();
  performCudaReductionV1();
}

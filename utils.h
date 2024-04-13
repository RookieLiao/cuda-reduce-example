#ifndef UTILS_H
#define UTILS_H

#include <cstdlib>
#include <sys/time.h>

inline double seconds() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

inline void initialize_array(int* idata_h, std::size_t elem_size) {
  // initialize array
  for (size_t i = 0; i < elem_size; ++i) { idata_h[i] = (int)(rand() & 0xFF); }
}

#endif  // utils_h

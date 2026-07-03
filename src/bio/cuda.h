#ifndef BIO_CUDA_H
#define BIO_CUDA_H

#include "io/input.h"
#include "io/output.h"

bool cuda_memory(size_t bytes);

bool cuda_align(struct input, struct output);

#endif /* BIO_CUDA_H */

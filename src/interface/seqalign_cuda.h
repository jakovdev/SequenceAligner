#ifndef INTERFACE_SEQALIGN_CUDA_H
#define INTERFACE_SEQALIGN_CUDA_H

#include "io/input.h"
#include "io/output.h"

bool cuda_memory(size_t bytes);
[[gnu::nonnull]]
bool cuda_align(struct input, struct output);

#endif /* INTERFACE_SEQALIGN_CUDA_H */

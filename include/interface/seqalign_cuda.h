#ifndef INTERFACE_SEQALIGN_CUDA_H
#define INTERFACE_SEQALIGN_CUDA_H

#include <stddef.h>

struct input;
struct output;

bool cuda_memory(size_t bytes);
[[gnu::nonnull]]
bool cuda_align(const struct input *, struct output *);

#endif /* INTERFACE_SEQALIGN_CUDA_H */

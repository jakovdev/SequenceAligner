#pragma once
#ifndef INTERFACE_SEQALIGN_CUDA_H
#define INTERFACE_SEQALIGN_CUDA_H

#include <stddef.h>

#include "bio/sequence/sequences.h"

bool arg_mode_cuda(void);
bool cuda_device_init(void);
void cuda_device_close(void);
bool cuda_memory(size_t bytes);
[[gnu::nonnull]]
bool cuda_align(struct sequences *);

#endif /* INTERFACE_SEQALIGN_CUDA_H */

#pragma once
#ifndef INTERFACE_SEQALIGN_CUDA_H
#define INTERFACE_SEQALIGN_CUDA_H

#include <stdbool.h>
#include <stddef.h>

bool arg_mode_cuda(void);
bool cuda_device_init(void);
void cuda_device_close(void);
bool cuda_memory(size_t bytes);
bool cuda_align(void);

#endif /* INTERFACE_SEQALIGN_CUDA_H */

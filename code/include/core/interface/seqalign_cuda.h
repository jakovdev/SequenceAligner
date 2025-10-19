#pragma once
#ifndef INTERFACE_SEQALIGN_CUDA_H
#define INTERFACE_SEQALIGN_CUDA_H

#include <stdbool.h>

bool cuda_init(void);
bool cuda_align(void);

#endif // INTERFACE_SEQALIGN_CUDA_H

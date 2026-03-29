#pragma once
#ifndef INTERFACE_SEQALIGN_HDF5_H
#define INTERFACE_SEQALIGN_HDF5_H

#include <stdbool.h>
#include <stddef.h>

#include "system/types.h"

bool h5_open(void);

void h5_matrix_column_set(s32 col, const s32 *values);

void h5_checksum_set(s64 checksum);

s64 h5_checksum(void);

void h5_close(int skip_flush);

#ifdef USE_CUDA
s32 *h5_matrix_data(void);
size_t h5_matrix_bytes(void);
#endif

#endif /* INTERFACE_SEQALIGN_HDF5_H */

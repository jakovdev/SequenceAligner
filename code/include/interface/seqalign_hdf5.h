#pragma once
#ifndef INTERFACE_SEQALIGN_HDF5_H
#define INTERFACE_SEQALIGN_HDF5_H

#include <stdbool.h>
#include <stddef.h>

#include "bio/types.h"
#include "system/types.h"

bool h5_open(const char *file_path, u64 mat_dim, u8 compression, bool write);

bool h5_sequences_store(sequence_t *sequences, u32 seq_count);

void h5_matrix_set(u32 row, u32 col, s32 value);

void h5_checksum_set(s64 checksum);

s64 h5_checksum(void);

void h5_close(int skip_flush);

#ifdef USE_CUDA
s32 *h5_matrix_data(void);
size_t h5_matrix_bytes(void);
#endif

#endif // INTERFACE_SEQALIGN_HDF5_H

#pragma once
#ifndef SEQALIGN_HDF5_H
#define SEQALIGN_HDF5_H

#include "sequences.h"

#include "stdbool.h"
#include "stddef.h"
#include "stdint.h"

extern bool h5_open(const char* fname, size_t mat_dim, unsigned int compression, bool write);
extern bool h5_sequences_store(sequence_t* sequences, size_t seq_count);
extern void h5_matrix_set(size_t row, size_t col, int value);
extern void h5_checksum_set(int64_t checksum);
extern int64_t h5_checksum(void);
extern void h5_close(int skip_flush);

#ifdef USE_CUDA
extern int* h5_matrix_data(void);
extern size_t h5_matrix_bytes(void);
extern bool h5_triangle_indices_64_bit(void);
extern half_t* h5_triangle_indices_32(void);
extern size_t* h5_triangle_indices_64(void);
#endif

#endif // SEQALIGN_HDF5_H
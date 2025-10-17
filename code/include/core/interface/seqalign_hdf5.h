#pragma once
#ifndef INTERFACE_SEQALIGN_HDF5_H
#define INTERFACE_SEQALIGN_HDF5_H

#include "core/bio/types.h"

extern bool h5_open(const char* file_path, size_t mat_dim, unsigned int compression, bool write);
extern bool h5_sequences_store(sequences_t sequences, sequence_count_t seq_count);
extern void h5_matrix_set(sequence_index_t row, sequence_index_t col, score_t value);
extern void h5_checksum_set(int64_t checksum);
extern int64_t h5_checksum(void);
extern void h5_close(int skip_flush);

#ifdef USE_CUDA
extern score_t* h5_matrix_data(void);
extern size_t h5_matrix_bytes(void);
extern bool h5_triangle_indices_64_bit(void);
extern half_t* h5_triangle_indices_32(void);
extern size_t* h5_triangle_indices_64(void);
#endif

#endif // INTERFACE_SEQALIGN_HDF5_H
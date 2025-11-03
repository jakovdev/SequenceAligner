#pragma once
#ifndef INTERFACE_SEQALIGN_HDF5_H
#define INTERFACE_SEQALIGN_HDF5_H

#include "core/bio/types.h"

bool h5_open(const char *file_path, size_t mat_dim, unsigned int compression,
	     bool write);

bool h5_sequences_store(sequences_t sequences, sequence_count_t seq_count);

void h5_matrix_set(sequence_index_t row, sequence_index_t col, score_t value);

void h5_checksum_set(int64_t checksum);

int64_t h5_checksum(void);

void h5_close(int skip_flush);

#ifdef USE_CUDA
score_t *h5_matrix_data(void);

size_t h5_matrix_bytes(void);

size_t *h5_triangle_indices(void);
#endif

#endif // INTERFACE_SEQALIGN_HDF5_H

#ifndef SEQALIGN_HDF5_H
#define SEQALIGN_HDF5_H

#include "stdbool.h"
#include "stddef.h"

#include "sequence.h"

extern void h5_initialize(const char* fname, size_t matsize, int compression, bool write);
extern void h5_set_matrix_value(size_t row, size_t col, int value);
extern void h5_set_checksum(int64_t checksum);
extern int64_t h5_get_checksum(void);
extern bool h5_store_sequences(sequence_t* sequences, size_t seq_count);
extern void h5_close(void);

#endif // SEQALIGN_HDF5_H
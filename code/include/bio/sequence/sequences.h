#pragma once
#ifndef BIO_SEQUENCE_SEQUENCES_H
#define BIO_SEQUENCE_SEQUENCES_H

#include <stdbool.h>

#include "bio/types.h"
#include "system/compiler.h"

extern s32 *g_restrict g_lengths;
extern s64 *g_restrict g_offsets;
extern char *g_restrict g_letters;
extern sequence_t *g_restrict g_seqs;
extern s64 g_alignments;
extern size_t g_seq_len_max;
extern s32 g_seq_n;

bool sequences_load_from_file(void);

#endif /* BIO_SEQUENCE_SEQUENCES_H */

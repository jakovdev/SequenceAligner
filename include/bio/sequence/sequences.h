#pragma once
#ifndef BIO_SEQUENCE_SEQUENCES_H
#define BIO_SEQUENCE_SEQUENCES_H

#include <stdbool.h>

#include "bio/types.h"
#include "system/compiler.h"

extern s64 ALIGNMENTS;
extern size_t LENGTHS_MAX;
extern s32 *g_restrict LENGTHS;
extern s64 *g_restrict OFFSETS;
extern char *g_restrict LETTERS;
extern sequence_t *g_restrict SEQS;
extern s32 SEQS_N;

bool sequences_load_from_file(void);

#endif /* BIO_SEQUENCE_SEQUENCES_H */

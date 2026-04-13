#pragma once
#ifndef BIO_SEQUENCE_SEQUENCES_H
#define BIO_SEQUENCE_SEQUENCES_H

#include <stdbool.h>
#include <stddef.h>

#include "bio/types.h"

#define TABLE_SIZE ((LENGTHS_MAX + 1) * (LENGTHS_MAX + 1))
extern s64 ALIGNMENTS;
extern size_t LENGTHS_MAX;
extern s32 *LENGTHS;
extern s64 *OFFSETS;
extern char *LETTERS;
extern struct seq *SEQS;
extern s32 SEQS_N;

bool sequences_load_from_file(void);

#endif /* BIO_SEQUENCE_SEQUENCES_H */

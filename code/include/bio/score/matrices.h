#pragma once
#ifndef BIO_SCORE_MATRICES_H
#define BIO_SCORE_MATRICES_H
/* clang-format off */

#define AMINO_SIZE 24
#define AMINO_MATSIZE (AMINO_SIZE * AMINO_SIZE * sizeof(int))
extern const char AMINO_ALPHABET[];

#define NUCLEO_SIZE 16
#define NUCLEO_MATSIZE (NUCLEO_SIZE * NUCLEO_SIZE * sizeof(int))
extern const char NUCLEO_ALPHABET[];

#define SUBMAT_MAX AMINO_SIZE

#define NUM_AMINO_MATRICES 65
#define NUM_NUCLEO_MATRICES 2

typedef struct {
	const char *name;
	const int (*matrix)[AMINO_SIZE];
} AminoMatrix;

typedef struct {
	const char *name;
	const int (*matrix)[NUCLEO_SIZE];
} NucleoMatrix;

extern const AminoMatrix AMINO_MATRIX[NUM_AMINO_MATRICES];
extern const NucleoMatrix NUCLEO_MATRIX[NUM_NUCLEO_MATRICES];

/* clang-format on */
#endif /* BIO_SCORE_MATRICES_H */

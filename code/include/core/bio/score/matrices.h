#pragma once
#ifndef CORE_BIO_SCORE_MATRICES_H
#define CORE_BIO_SCORE_MATRICES_H
// clang-format off

#define AMINO_SIZE 24
extern const char AMINO_ALPHABET[];

#define NUCLEOTIDE_SIZE 16
extern const char NUCLEOTIDE_ALPHABET[];

#define MAX_MATRIX_DIM AMINO_SIZE

#define NUM_AMINO_MATRICES 65
#define NUM_NUCLEOTIDE_MATRICES 2

typedef struct
{
    const char* name;
    const int (*matrix)[AMINO_SIZE];
} AminoMatrix;

typedef struct
{
    const char* name;
    const int (*matrix)[NUCLEOTIDE_SIZE];
} NucleotideMatrix;

extern const AminoMatrix ALL_AMINO_MATRICES[NUM_AMINO_MATRICES];
extern const NucleotideMatrix ALL_NUCLEOTIDE_MATRICES[NUM_NUCLEOTIDE_MATRICES];

// clang-format on
#endif // CORE_BIO_SCORE_MATRICES_H

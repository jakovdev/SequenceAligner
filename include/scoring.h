#pragma once
#ifndef SCORING_H
#define SCORING_H

#include "args.h"
#include "biotypes.h"
#include "matrices.h"

#include <limits.h>

#ifdef USE_SIMD
static veci_t g_first_row_indices;
static veci_t g_gap_penalty_vec;
static veci_t g_gap_start_vec;
static veci_t g_gap_extend_vec;
#endif

static int SEQUENCE_LOOKUP[SCHAR_MAX + 1];
static int SCORING_MATRIX[MAX_MATRIX_DIM][MAX_MATRIX_DIM];

static inline void
scoring_matrix_init()
{
    int matrix_id = args_scoring_matrix();
    int sequence_type = args_sequence_type();

    switch (sequence_type)
    {
        case SEQ_TYPE_NUCLEOTIDE:
        {
            const int(*src_matrix)[NUCLEOTIDE_SIZE] = ALL_NUCLEOTIDE_MATRICES[matrix_id].matrix;
            for (int i = 0; i < NUCLEOTIDE_SIZE; i++)
            {
                for (int j = 0; j < NUCLEOTIDE_SIZE; j++)
                {
                    SCORING_MATRIX[i][j] = src_matrix[i][j];
                }
            }

            break;
        }

        // Expandable
        case SEQ_TYPE_AMINO:
        default:
        {
            const int(*src_matrix)[AMINO_SIZE] = ALL_AMINO_MATRICES[matrix_id].matrix;
            for (int i = 0; i < AMINO_SIZE; i++)
            {
                for (int j = 0; j < AMINO_SIZE; j++)
                {
                    SCORING_MATRIX[i][j] = src_matrix[i][j];
                }
            }

            break;
        }
    }

    memset(SEQUENCE_LOOKUP, -1, sizeof(SEQUENCE_LOOKUP));
    switch (sequence_type)
    {
        case SEQ_TYPE_NUCLEOTIDE:
        {
            for (int i = 0; i < (int)strlen(NUCLEOTIDE_ALPHABET); i++)
            {
                SEQUENCE_LOOKUP[(int)NUCLEOTIDE_ALPHABET[i]] = i;
            }

            break;
        }

        // Expandable
        case SEQ_TYPE_AMINO:
        default:
        {
            for (int i = 0; i < (int)strlen(AMINO_ALPHABET); i++)
            {
                SEQUENCE_LOOKUP[(int)AMINO_ALPHABET[i]] = i;
            }

            break;
        }
    }

#ifdef USE_SIMD
    g_first_row_indices = set_row_indices();
    g_gap_penalty_vec = set1_epi32(args_gap_penalty());
    g_gap_start_vec = set1_epi32(args_gap_start());
    g_gap_extend_vec = set1_epi32(args_gap_extend());
#endif
}

#endif
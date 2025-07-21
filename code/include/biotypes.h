#pragma once
#ifndef TYPES_H
#define TYPES_H

typedef enum
{
    SEQ_TYPE_AMINO,
    SEQ_TYPE_NUCLEOTIDE,
    // NOTE: This enum is kept minimal by design. Only standard biological sequence types
    //       are included as they're the only ones with established substitution matrices.
    SEQ_TYPE_COUNT
} SequenceType;

typedef enum
{
    ALIGN_NEEDLEMAN_WUNSCH,
    ALIGN_GOTOH_AFFINE,
    ALIGN_SMITH_WATERMAN,
    // NOTE: Additional alignment methods can be added here if needed.
    //       However, this requires implementing the corresponding algorithm.
    // Expandable
    ALIGN_COUNT
} AlignmentMethod;

typedef enum
{
    GAP_TYPE_LINEAR,
    GAP_TYPE_AFFINE,
} GapPenaltyType;

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef USE_CUDA
#include "host_types.h"
#else
#if SIZE_MAX == UINT64_MAX
typedef uint32_t HALF_OF_SIZE_T;
typedef uint16_t HALF_OF_HALF_T;
#define HALF_MAX (UINT32_MAX)
#define QUAR_MAX (UINT16_MAX)
#elif SIZE_MAX == UINT32_MAX
typedef uint16_t HALF_OF_SIZE_T;
typedef uint8_t HALF_OF_HALF_T;
#define HALF_MAX (UINT16_MAX)
#define QUAR_MAX (UINT8_MAX)
#else
#error "Unsupported platform: size_t width not 32 or 64 bits"
#endif

typedef HALF_OF_SIZE_T half_t;
typedef HALF_OF_HALF_T quar_t;

typedef quar_t sequence_length_t;
#define MAX_SEQUENCE_LENGTH (QUAR_MAX)

typedef half_t sequence_index_t;
typedef half_t sequence_count_t;
#define MAX_SEQUENCE_COUNT (HALF_MAX)

typedef size_t alignment_size_t;

typedef int score_t;

#define SCORE_MIN (INT_MIN / 2)

#endif

typedef struct
{
    char* letters;
    sequence_length_t length;
} sequence_t;

// Collection of sequences
typedef sequence_t* sequences_t;

// Pointer to a specific sequence
typedef sequence_t* restrict sequence_ptr_t;

extern const char* alignment_name(int method);
extern bool alignment_linear(int method);
extern bool alignment_affine(int method);
extern const char* gap_type_name(int method);
extern int alignment_arg(const char* arg);
extern void alignment_list(void);
extern const char* matrix_id_name(int seq_type, int matrix_id);
extern int matrix_name_id(int seq_type, const char* name);
extern void matrix_seq_type_list(int seq_type);
extern const char* sequence_type_name(int type);
extern int sequence_type_arg(const char* arg);
extern void sequence_types_list(void);

#endif // TYPES_H
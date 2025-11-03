#pragma once
#ifndef CORE_BIO_TYPES_H
#define CORE_BIO_TYPES_H

typedef enum {
	SEQ_TYPE_INVALID = -1,
	SEQ_TYPE_AMINO,
	SEQ_TYPE_NUCLEOTIDE,
	// NOTE: This enum is kept minimal by design. Only standard biological sequence types
	//       are included as they're the only ones with established substitution matrices.
	SEQ_TYPE_COUNT
} SequenceType;

typedef enum {
	ALIGN_INVALID = -1,
	ALIGN_GOTOH_AFFINE,
	ALIGN_NEEDLEMAN_WUNSCH,
	ALIGN_SMITH_WATERMAN,
	// NOTE: Additional alignment methods can be added here if needed.
	//       However, this requires implementing the corresponding algorithm.
	// Expandable
	ALIGN_COUNT
} AlignmentMethod;

typedef enum {
	GAP_TYPE_LINEAR,
	GAP_TYPE_AFFINE,
} GapPenaltyType;

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if SIZE_MAX == UINT64_MAX
typedef uint32_t HALF_OF_SIZE_T;
typedef uint16_t HALF_OF_HALF_T;
#define HALF_MAX (UINT32_MAX)
#define QUAR_MAX (UINT16_MAX)
#else
#error "Unsupported platform: size_t width not 64 bits"
#endif

typedef HALF_OF_SIZE_T half_t;
typedef HALF_OF_HALF_T quar_t;

// Keep at DWORD_PTR size
typedef size_t sequence_length_t;
// Can be up to ~ 2^32 - 1
#define SEQUENCE_LENGTH_MAX (QUAR_MAX)

typedef half_t sequence_index_t;
typedef half_t sequence_count_t;
#define SEQUENCE_COUNT_MAX (HALF_MAX)

typedef half_t sequence_offset_t;
typedef size_t alignment_size_t;
typedef int score_t;
#define SCORE_MIN (INT_MIN / 2)

typedef struct {
	char *letters;
	sequence_length_t length;
} sequence_t;

// Collection of sequences
typedef sequence_t *sequences_t;

// Pointer to a specific sequence
typedef const sequence_t *const restrict sequence_ptr_t;
typedef score_t (*align_func_t)(sequence_ptr_t, sequence_ptr_t);

align_func_t align_function(AlignmentMethod method);
const char *alignment_name(AlignmentMethod method);
bool alignment_linear(AlignmentMethod method);
bool alignment_affine(AlignmentMethod method);
const char *gap_type_name(AlignmentMethod method);
AlignmentMethod alignment_arg(const char *arg);
void alignment_list(void);

const char *matrix_id_name(SequenceType seq_type, int matrix_id);
int matrix_name_id(SequenceType seq_type, const char *name);
void matrix_seq_type_list(SequenceType seq_type);

const char *sequence_type_name(SequenceType seq_type);
SequenceType sequence_type_arg(const char *arg);
void sequence_types_list(void);

#endif // CORE_BIO_TYPES_H

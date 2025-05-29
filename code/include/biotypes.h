#pragma once
#ifndef TYPES_H
#define TYPES_H

#include <stdbool.h>

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
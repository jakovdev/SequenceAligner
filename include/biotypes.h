#ifndef TYPES_H
#define TYPES_H

#include "arch.h"
#include "matrices.h"
#include <ctype.h>

#define MAX_MATRIX_DIM AMINO_SIZE

typedef enum
{
    SEQ_TYPE_AMINO,
    SEQ_TYPE_NUCLEOTIDE,
    // NOTE: This enum is kept minimal by design. Only standard biological sequence types
    //       are included as they're the only ones with established scoring matrices.
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

typedef struct
{
    SequenceType type;
    const char* name;
    const char* description;
    const char** aliases;
} SequenceTypeInfo;

typedef struct
{
    AlignmentMethod method;
    const char* name;
    const char* description;
    const char** aliases;
    GapPenaltyType gap_type;
} AlignmentMethodInfo;

static const AlignmentMethodInfo ALIGNMENT_METHODS[] = {
    { ALIGN_NEEDLEMAN_WUNSCH,
      "Needleman-Wunsch",
      "global alignment",
      (const char*[]){ "nw", "needleman", NULL },
      GAP_TYPE_LINEAR },
    { ALIGN_GOTOH_AFFINE,
      "Gotoh (affine)",
      "global alignment",
      (const char*[]){ "ga", "gotoh", NULL },
      GAP_TYPE_AFFINE },
    { ALIGN_SMITH_WATERMAN,
      "Smith-Waterman",
      "local alignment",
      (const char*[]){ "sw", "smith", NULL },
      GAP_TYPE_AFFINE }
};

static const SequenceTypeInfo SEQUENCE_TYPES[] = {
    { SEQ_TYPE_AMINO,
      "Amino acids",
      "protein sequences",
      (const char*[]){ "amino", "aa", "protein", NULL } },
    { SEQ_TYPE_NUCLEOTIDE,
      "Nucleotides",
      "DNA/RNA sequences",
      (const char*[]){ "nucleotide", "dna", "rna", "nt", NULL } },
};

static inline const char*
alignment_name(int method)
{
    return ALIGNMENT_METHODS[method].name;
}

static inline bool
alignment_linear(int method)
{
    return ALIGNMENT_METHODS[method].gap_type == GAP_TYPE_LINEAR;
}

static inline bool
alignment_affine(int method)
{
    return ALIGNMENT_METHODS[method].gap_type == GAP_TYPE_AFFINE;
}

static inline const char*
gap_type_name(int method)
{
    switch (ALIGNMENT_METHODS[method].gap_type)
    {
        case GAP_TYPE_LINEAR:
            return "Linear";
        case GAP_TYPE_AFFINE:
            return "Affine";
        default:
            UNREACHABLE();
    }
}

static inline int
alignment_arg(const char* arg)
{
    if (!arg)
    {
        return -1;
    }

    // Check for numeric method
    if (isdigit(arg[0]) || (arg[0] == '-' && isdigit(arg[1])))
    {
        int method = atoi(arg);
        if (method >= 0 && method < ALIGN_COUNT)
        {
            return method;
        }

        return -1;
    }

    // Search by name/alias
    for (int i = 0; i < ALIGN_COUNT; i++)
    {
        for (const char** alias = ALIGNMENT_METHODS[i].aliases; *alias != NULL; alias++)
        {
            if (strcasecmp(arg, *alias) == 0)
            {
                return ALIGNMENT_METHODS[i].method;
            }
        }
    }

    return -1;
}

static inline void
alignment_list(void)
{
    for (int i = 0; i < ALIGN_COUNT; i++)
    {
        printf("                           %s: %s (%s, %s gap)\n",
               ALIGNMENT_METHODS[i].aliases[0],
               ALIGNMENT_METHODS[i].name,
               ALIGNMENT_METHODS[i].description,
               gap_type_name(i));
    }
}

static inline const char*
matrix_id_name(int seq_type, int matrix_id)
{
    if (seq_type < 0 || matrix_id < 0)
    {
        return "Unknown";
    }

    if (seq_type == SEQ_TYPE_AMINO && matrix_id < NUM_AMINO_MATRICES)
    {
        return ALL_AMINO_MATRICES[matrix_id].name;
    }

    else if (seq_type == SEQ_TYPE_NUCLEOTIDE && matrix_id < NUM_NUCLEOTIDE_MATRICES)
    {
        return ALL_NUCLEOTIDE_MATRICES[matrix_id].name;
    }

    return "Unknown";
}

static inline int
matrix_name_id(int seq_type, const char* name)
{
    if (!name)
    {
        return -1;
    }

    int num_matrices = 0;
    const void* matrices = NULL;

    if (seq_type == SEQ_TYPE_AMINO)
    {
        num_matrices = NUM_AMINO_MATRICES;
        matrices = ALL_AMINO_MATRICES;
    }

    else if (seq_type == SEQ_TYPE_NUCLEOTIDE)
    {
        num_matrices = NUM_NUCLEOTIDE_MATRICES;
        matrices = ALL_NUCLEOTIDE_MATRICES;
    }

    else
    {
        return -1;
    }

    for (int i = 0; i < num_matrices; i++)
    {
        const char* matrix_name = NULL;
        if (seq_type == SEQ_TYPE_AMINO)
        {
            matrix_name = ((const AminoMatrix*)matrices)[i].name;
        }

        else
        {
            matrix_name = ((const NucleotideMatrix*)matrices)[i].name;
        }

        if (strcasecmp(name, matrix_name) == 0)
        {
            return i;
        }
    }

    return -1;
}

static inline void
matrix_seq_type_list(int seq_type)
{
    if (seq_type == SEQ_TYPE_AMINO)
    {
        for (int i = 0; i < NUM_AMINO_MATRICES; i++)
        {
            printf("  %s%s",
                   ALL_AMINO_MATRICES[i].name,
                   (i + 1) % 5 == 0                ? "\n"
                   : (i == NUM_AMINO_MATRICES - 1) ? "\n"
                                                   : ", ");
        }
    }

    else if (seq_type == SEQ_TYPE_NUCLEOTIDE)
    {
        for (int i = 0; i < NUM_NUCLEOTIDE_MATRICES; i++)
        {
            printf("  %s%s",
                   ALL_NUCLEOTIDE_MATRICES[i].name,
                   (i + 1) % 5 == 0                     ? "\n"
                   : (i == NUM_NUCLEOTIDE_MATRICES - 1) ? "\n"
                                                        : ", ");
        }
    }
}

static inline const char*
sequence_type_name(int type)
{
    return SEQUENCE_TYPES[type].name;
}

static inline int
sequence_type_arg(const char* arg)
{
    if (!arg)
    {
        return -1;
    }

    if (isdigit(arg[0]) || (arg[0] == '-' && isdigit(arg[1])))
    {
        int type = atoi(arg);
        if (type >= 0 && type < SEQ_TYPE_COUNT)
        {
            return type;
        }

        return -1;
    }

    for (int i = 0; i < SEQ_TYPE_COUNT; i++)
    {
        for (const char** alias = SEQUENCE_TYPES[i].aliases; *alias != NULL; alias++)
        {
            if (strcasecmp(arg, *alias) == 0)
            {
                return SEQUENCE_TYPES[i].type;
            }
        }
    }

    return -1;
}

static inline void
sequence_types_list(void)
{
    for (int i = 0; i < SEQ_TYPE_COUNT; i++)
    {
        printf("                           %s: %s (%s)\n",
               SEQUENCE_TYPES[i].aliases[0],
               SEQUENCE_TYPES[i].name,
               SEQUENCE_TYPES[i].description);
    }
}

#endif // TYPES_H
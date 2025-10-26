#include "core/bio/types.h"

#include <ctype.h>
#include <stdio.h>

#include "core/bio/algorithm/method/ga.h"
#include "core/bio/algorithm/method/nw.h"
#include "core/bio/algorithm/method/sw.h"
#include "core/bio/score/matrices.h"
#include "system/arch.h"

static struct
{
    SequenceType type;
    const char* name;
    const char* description;
    const char** aliases;
} SEQUENCE_TYPES[] = {
    { SEQ_TYPE_AMINO,
      "Amino acids",
      "protein sequences",
      (const char*[]){ "amino", "aa", "protein", NULL } },
    { SEQ_TYPE_NUCLEOTIDE,
      "Nucleotides",
      "DNA/RNA sequences",
      (const char*[]){ "nucleotide", "dna", "rna", "nt", NULL } },
};

static struct
{
    AlignmentMethod method;
    const char* name;
    const char* description;
    const char** aliases;
    GapPenaltyType gap_type;
} ALIGNMENT_METHODS[] = { //
    { ALIGN_GOTOH_AFFINE,
      "Gotoh (affine)",
      "global alignment",
      (const char*[]){ "ga", "gotoh", NULL },
      GAP_TYPE_AFFINE },
    { ALIGN_NEEDLEMAN_WUNSCH,
      "Needleman-Wunsch",
      "global alignment",
      (const char*[]){ "nw", "needleman", NULL },
      GAP_TYPE_LINEAR },
    { ALIGN_SMITH_WATERMAN,
      "Smith-Waterman",
      "local alignment",
      (const char*[]){ "sw", "smith", NULL },
      GAP_TYPE_AFFINE }
};

align_func_t
align_function(AlignmentMethod method)
{
    switch (method)
    {
        case ALIGN_GOTOH_AFFINE:
            return align_ga;

        case ALIGN_NEEDLEMAN_WUNSCH:
            return align_nw;

        case ALIGN_SMITH_WATERMAN:
            return align_sw;

        case ALIGN_INVALID:
        case ALIGN_COUNT:
        default:
            return NULL;
    }
}

const char*
alignment_name(AlignmentMethod method)
{
    return ALIGNMENT_METHODS[method].name;
}

bool
alignment_linear(AlignmentMethod method)
{
    return ALIGNMENT_METHODS[method].gap_type == GAP_TYPE_LINEAR;
}

bool
alignment_affine(AlignmentMethod method)
{
    return ALIGNMENT_METHODS[method].gap_type == GAP_TYPE_AFFINE;
}

const char*
gap_type_name(AlignmentMethod method)
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

AlignmentMethod
alignment_arg(const char* arg)
{
    if (!arg)
    {
        return ALIGN_INVALID;
    }

    // Check for numeric method
    if (isdigit(arg[0]) || (arg[0] == '-' && isdigit(arg[1])))
    {
        int method = atoi(arg);
        if (method >= 0 && method < ALIGN_COUNT)
        {
            return (AlignmentMethod)method;
        }

        return ALIGN_INVALID;
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

    return ALIGN_INVALID;
}

void
alignment_list(void)
{
    for (int i = 0; i < ALIGN_COUNT; i++)
    {
        printf("                           %s: %s (%s, %s gap)\n",
               ALIGNMENT_METHODS[i].aliases[0],
               ALIGNMENT_METHODS[i].name,
               ALIGNMENT_METHODS[i].description,
               gap_type_name((AlignmentMethod)i));
    }
}

const char*
matrix_id_name(SequenceType seq_type, int matrix_id)
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

int
matrix_name_id(SequenceType seq_type, const char* name)
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

void
matrix_seq_type_list(SequenceType seq_type)
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

const char*
sequence_type_name(SequenceType seq_type)
{
    return SEQUENCE_TYPES[seq_type].name;
}

SequenceType
sequence_type_arg(const char* arg)
{
    if (!arg)
    {
        return SEQ_TYPE_INVALID;
    }

    if (isdigit(arg[0]) || (arg[0] == '-' && isdigit(arg[1])))
    {
        int type = atoi(arg);
        if (type >= 0 && type < SEQ_TYPE_COUNT)
        {
            return (SequenceType)type;
        }

        return SEQ_TYPE_INVALID;
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

    return SEQ_TYPE_INVALID;
}

void
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

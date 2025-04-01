#ifndef METHODS_H
#define METHODS_H

#include "macros.h"

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

INLINE const char*
get_alignment_method_name(int method)
{
    return ALIGNMENT_METHODS[method].name;
}

INLINE int
requires_linear_gap(int method)
{
    return ALIGNMENT_METHODS[method].gap_type == GAP_TYPE_LINEAR;
}

INLINE int
requires_affine_gap(int method)
{
    return ALIGNMENT_METHODS[method].gap_type == GAP_TYPE_AFFINE;
}

INLINE const char*
get_gap_type_name(int method)
{
    switch (ALIGNMENT_METHODS[method].gap_type)
    {
        case GAP_TYPE_LINEAR:
            return "Linear";
        case GAP_TYPE_AFFINE:
            return "Affine";
        default:
            unreachable();
    }
}

INLINE int
find_alignment_method_by_name(const char* arg)
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

INLINE void
list_alignment_methods(void)
{
    for (int i = 0; i < ALIGN_COUNT; i++)
    {
        printf("  %s: %s (%s, %s gap)\n",
               ALIGNMENT_METHODS[i].aliases[0],
               ALIGNMENT_METHODS[i].name,
               ALIGNMENT_METHODS[i].description,
               get_gap_type_name(i));
    }
}

#endif // METHODS_H
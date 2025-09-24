#include "core/biology/algorithm/alignment.h"

#include "core/biology/algorithm/method/ga.h"
#include "core/biology/algorithm/method/nw.h"
#include "core/biology/algorithm/method/sw.h"

#include "core/app/args.h"
#include "system/arch.h"

score_t
align_pairwise(const sequence_ptr_t seq1, const sequence_ptr_t seq2)
{
    switch (args_align_method())
    {
        case ALIGN_GOTOH_AFFINE:
            return align_ga(seq1, seq2);

        case ALIGN_NEEDLEMAN_WUNSCH:
            return align_nw(seq1, seq2);

        case ALIGN_SMITH_WATERMAN:
            return align_sw(seq1, seq2);

        case ALIGN_INVALID:
        case ALIGN_COUNT:
        default:
            UNREACHABLE();
    }
}
#ifndef SEQALIGN_H
#define SEQALIGN_H

#include "align_nw.h"
#include "align_ga.h"
#include "align_sw.h"

INLINE Alignment align_sequences(const char seq1[MAX_SEQ_LEN],
                                 const size_t len1,
                                 const char seq2[MAX_SEQ_LEN], 
                                 const size_t len2,
                                 const ScoringMatrix* restrict scoring) {    
    switch(get_alignment_method()) {
        case ALIGN_GOTOH_AFFINE:
            return ga_align(seq1, len1, seq2, len2, scoring);
        case ALIGN_SMITH_WATERMAN:
            return sw_align(seq1, len1, seq2, len2, scoring);
        // Expandable
        case ALIGN_NEEDLEMAN_WUNSCH:
        default:
            return nw_align(seq1, len1, seq2, len2, scoring);
    }
}

#endif
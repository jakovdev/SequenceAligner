#ifndef SCORING_H
#define SCORING_H

#include "fstring.h"

static const char AMINO_ACIDS[] = "ARNDCQEGHILKMFPSTWYV";

#ifdef USE_AVX
static veci_t FIRST_ROW_INDICES;
static veci_t GAP_PENALTY_VEC;
static veci_t GAP_START_VEC;
static veci_t GAP_EXTEND_VEC;
#endif

static int AMINO_LOOKUP[SCHAR_MAX + 1];

INLINE void init_scoring_matrix(ScoringMatrix* restrict matrix) {
    // TEMPORARY
    const int blosum50[BLOSUM_SIZE][BLOSUM_SIZE] = {
        { 5,-2,-1,-2,-1,-1,-1, 0,-2,-1,-2,-1,-1,-3,-1, 1, 0,-3,-2, 0}, // A
        {-2, 7,-1,-2,-4, 1, 0,-3, 0,-4,-3, 3,-2,-3,-3,-1,-1,-3,-1,-3}, // R
        {-1,-1, 7, 2,-2, 0, 0, 0, 1,-3,-4, 0,-2,-4,-2, 1, 0,-4,-2,-3}, // N
        {-2,-2, 2, 8,-4, 0, 2,-1,-1,-4,-4,-1,-4,-5,-1, 0,-1,-5,-3,-4}, // D
        {-1,-4,-2,-4,13,-3,-3,-3,-3,-2,-2,-3,-2,-2,-4,-1,-1,-5,-3,-1}, // C
        {-1, 1, 0, 0,-3, 7, 2,-2, 1,-3,-2, 2, 0,-4,-1, 0,-1,-1,-1,-3}, // Q
        {-1, 0, 0, 2,-3, 2, 6,-3, 0,-4,-3, 1,-2,-3,-1,-1,-1,-3,-2,-3}, // E
        { 0,-3, 0,-1,-3,-2,-3, 8,-2,-4,-4,-2,-3,-4,-2, 0,-2,-3,-3,-4}, // G
        {-2, 0, 1,-1,-3, 1, 0,-2,10,-4,-3, 0,-1,-1,-2,-1,-2,-3, 2,-4}, // H
        {-1,-4,-3,-4,-2,-3,-4,-4,-4, 5, 2,-3, 2, 0,-3,-3,-1,-3,-1, 4}, // I
        {-2,-3,-4,-4,-2,-2,-3,-4,-3, 2, 5,-3, 3, 1,-4,-3,-1,-2,-1, 1}, // L
        {-1, 3, 0,-1,-3, 2, 1,-2, 0,-3,-3, 6,-2,-4,-1, 0,-1,-3,-2,-3}, // K
        {-1,-2,-2,-4,-2, 0,-2,-3,-1, 2, 3,-2, 7, 0,-3,-2,-1,-1, 0, 1}, // M
        {-3,-3,-4,-5,-2,-4,-3,-4,-1, 0, 1,-4, 0, 8,-4,-3,-2, 1, 4,-1}, // F
        {-1,-3,-2,-1,-4,-1,-1,-2,-2,-3,-4,-1,-3,-4,10,-1,-1,-4,-3,-3}, // P
        { 1,-1, 1, 0,-1, 0,-1, 0,-1,-3,-3, 0,-2,-3,-1, 5, 2,-4,-2,-2}, // S
        { 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 2, 5,-3,-2, 0}, // T
        {-3,-3,-4,-5,-5,-1,-3,-3,-3,-3,-2,-3,-1, 1,-4,-4,-3,15, 2,-3}, // W
        {-2,-1,-2,-3,-3,-1,-2,-3, 2,-1,-1,-2, 0, 4,-3,-2,-2, 2, 8,-1}, // Y
        { 0,-3,-3,-4,-1,-3,-3,-4,-4, 4, 1,-3, 1,-1,-3,-2, 0,-3,-1, 5}};// V
    
    memcpy(matrix->matrix, blosum50, sizeof(blosum50));

    static bool initialized = false;
    if (UNLIKELY(!initialized)) {
        memset(AMINO_LOOKUP, -1, sizeof(AMINO_LOOKUP));
        for (int i = 0; i < (int)strlen(AMINO_ACIDS); i++) {
            AMINO_LOOKUP[(int)AMINO_ACIDS[i]] = i;
        }
        #ifdef USE_AVX
        FIRST_ROW_INDICES = setr_indicies;
        GAP_PENALTY_VEC = set1_epi32(get_gap_penalty());
        GAP_START_VEC = set1_epi32(get_gap_start());
        GAP_EXTEND_VEC = set1_epi32(get_gap_extend());
        #endif
        initialized = true;
    }
}

#endif
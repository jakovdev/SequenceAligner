#ifndef ALIGN_H
#define ALIGN_H

#include "scoring.h"
#include "methods.h"

#define STACK_MATRIX_THRESHOLD (128 * KiB)
#define MATRIX_SIZE(len1, len2) ((len1 + 1) * (len2 + 1))
#define MATRIX_BYTES(len1, len2) (MATRIX_SIZE(len1, len2) * sizeof(int))
#define MATRICES_3X_BYTES(len1, len2) (3 * MATRIX_BYTES(len1, len2))
#define USE_STACK_MATRIX(bytes) ((bytes) <= STACK_MATRIX_THRESHOLD)

// Direction enums for traceback
#define DIAG 0
#define UP 1
#define LEFT 2

static const int8_t next_i[] = {-1, -1, 0};    // DIAG, UP, LEFT
static const int8_t next_j[] = {-1, 0, -1};

typedef struct {
    int gap_penalty;
    int gap_open;
    int gap_extend;
} GapPenalties;

static GapPenalties g_gap_penalties = {0, 0, 0};

INLINE void init_gap_penalties(void) {
    int method = get_alignment_method();
    GapPenaltyType gap_type = ALIGNMENT_METHODS[method].gap_type;
    
    if (gap_type == GAP_TYPE_LINEAR) {
        g_gap_penalties.gap_penalty = get_gap_penalty();
    } else if (gap_type == GAP_TYPE_AFFINE) {
        g_gap_penalties.gap_open = get_gap_start();
        g_gap_penalties.gap_extend = get_gap_extend();
    }
}

typedef struct {
    int* data;
    size_t size;
    bool is_stack;
} SeqIndices;

INLINE void precompute_seq_indices(SeqIndices* indices, const char* restrict seq, size_t len) {
    indices->size = len;
    
    if (len <= MAX_STACK_SEQUENCE_LENGTH) {
        int* stack_buffer = alloca(len * sizeof(int));
        indices->data = stack_buffer;
        indices->is_stack = true;
    } else {
        indices->data = (int*)malloc(len * sizeof(int));
        indices->is_stack = false;
    }
    
    #pragma GCC unroll 8
    for (size_t i = 0; i < len; ++i) {
        indices->data[i] = SEQUENCE_LOOKUP[(unsigned char)seq[i]];
    }
}

INLINE void free_seq_indices(SeqIndices* indices) {
    if (!indices->is_stack && indices->data) {
        free(indices->data);
        indices->data = NULL;
    }
}

INLINE int* allocate_matrix(int* stack_matrix, size_t bytes) {
    if (USE_STACK_MATRIX(bytes)) {
        return stack_matrix;
    } else {
        return (int*)huge_page_alloc(bytes);
    }
}

INLINE void free_matrix(int* matrix, int* stack_matrix) {
    if (matrix != stack_matrix) {
        aligned_free(matrix);
    }
}

INLINE float calculate_similarity(const char* seq1, size_t len1, const char* seq2, size_t len2) {
    // Use the shorter length as reference for percentage calculation
    size_t min_len = len1 < len2 ? len1 : len2;
    if (min_len == 0) return 0.0f;
    
    size_t matches = 0;
    size_t compare_len = min_len;
    
    #ifdef USE_SIMD
    // Process in vector chunks
    size_t vec_limit = (compare_len / BYTES) * BYTES;
    for (size_t i = 0; i < vec_limit; i += BYTES) {
        veci_t v1 = loadu((veci_t*)(seq1 + i));
        veci_t v2 = loadu((veci_t*)(seq2 + i));
        veci_t eq = cmpeq_epi8(v1, v2);
        num_t mask = movemask_epi8(eq);
        matches += __builtin_popcount(mask);
    }
    
    // Process remaining characters
    for (size_t i = vec_limit; i < compare_len; i++) {
        if (seq1[i] == seq2[i]) {
            matches++;
        }
    }
    #else
    // Basic character-by-character comparison
    for (size_t i = 0; i < compare_len; i++) {
        if (seq1[i] == seq2[i]) {
            matches++;
        }
    }
    #endif
    
    return (float)matches / min_len;
}

#endif
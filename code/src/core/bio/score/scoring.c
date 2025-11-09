#include "core/bio/score/scoring.h"

#include <string.h>

#include "core/app/args.h"
#include "core/bio/types.h"
#include "system/compiler.h"

#if USE_SIMD == 1
veci_t g_first_row_indices;
veci_t g_gap_penalty_vec;
veci_t g_gap_open_vec;
veci_t g_gap_extend_vec;
#endif

int SEQ_LUP[SCHAR_MAX + 1];
int SUB_MAT[SUBMAT_MAX][SUBMAT_MAX];

#define SEQ_TYPE_INIT(TYPE)                             \
	for (int i = 0; i < TYPE##_SIZE; i++)           \
		SEQ_LUP[(uchar)TYPE##_ALPHABET[i]] = i; \
	memcpy(SUB_MAT, TYPE##_MATRIX[args_sub_matrix()].matrix, TYPE##_MATSIZE)

void scoring_init(void)
{
	memset(SEQ_LUP, -1, sizeof(SEQ_LUP));
	switch (args_sequence_type()) {
	case SEQ_TYPE_AMINO:
		SEQ_TYPE_INIT(AMINO);
		break;
	case SEQ_TYPE_NUCLEO:
		SEQ_TYPE_INIT(NUCLEO);
		break;
	/* EXPANDABLE: enum SequenceType */
	default:
	case SEQ_TYPE_COUNT:
	case SEQ_TYPE_INVALID:
		UNREACHABLE();
	}

#if USE_SIMD == 1
	g_first_row_indices = set_row_indices();
	g_gap_penalty_vec = set1_epi32(args_gap_penalty());
	g_gap_open_vec = set1_epi32(args_gap_open());
	g_gap_extend_vec = set1_epi32(args_gap_extend());
#endif
}

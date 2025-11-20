#include "bio/score/scoring.h"

#include <string.h>

#include "bio/types.h"
#include "system/compiler.h"

#if /* USE_SIMD == 1 */ 0
veci_t g_first_row_indices;
veci_t g_gap_pen_vec;
veci_t g_gap_open_vec;
veci_t g_gap_ext_vec;
#endif

s32 SEQ_LUP[SCHAR_MAX + 1];
s32 SUB_MAT[SUBMAT_MAX][SUBMAT_MAX];

#define SEQ_TYPE_INIT(TYPE)                             \
	for (int i = 0; i < TYPE##_SIZE; i++)           \
		SEQ_LUP[(uchar)TYPE##_ALPHABET[i]] = i; \
	memcpy(SUB_MAT, TYPE##_MATRIX[arg_sub_matrix()].matrix, TYPE##_MATSIZE)

void scoring_init(void)
{
	memset(SEQ_LUP, -1, sizeof(SEQ_LUP));
	switch (arg_sequence_type()) {
	case SEQ_TYPE_AMINO:
		SEQ_TYPE_INIT(AMINO);
		break;
	case SEQ_TYPE_NUCLEO:
		SEQ_TYPE_INIT(NUCLEO);
		break;
	default: /* NOTE: EXPANDABLE enum SequenceType */
	case SEQ_TYPE_COUNT:
	case SEQ_TYPE_INVALID:
		UNREACHABLE();
	}

#if /* USE_SIMD == 1 */ 0
	g_first_row_indices = set_row_indices();
	g_gap_pen_vec = set1_epi32(arg_gap_pen());
	g_gap_open_vec = set1_epi32(arg_gap_open());
	g_gap_ext_vec = set1_epi32(arg_gap_ext());
#endif
}

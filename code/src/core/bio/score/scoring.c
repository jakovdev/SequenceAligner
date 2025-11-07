#include "core/bio/score/scoring.h"

#include <string.h>

#include "core/app/args.h"
#include "core/bio/types.h"
#include "util/print.h"

#if USE_SIMD == 1
veci_t g_first_row_indices;
veci_t g_gap_penalty_vec;
veci_t g_gap_open_vec;
veci_t g_gap_extend_vec;
#endif

int SEQ_LUP[SCHAR_MAX + 1];
int SUB_MAT[SUBMAT_MAX][SUBMAT_MAX];

void scoring_matrix_init(void)
{
	const int m_id = args_scoring_matrix();
	memset(SEQ_LUP, -1, sizeof(SEQ_LUP));
	switch (args_sequence_type()) {
	case SEQ_TYPE_NUCLEO: {
		memcpy(SUB_MAT, NUCLEO_MATRIX[m_id].matrix, NUCLEO_MATSIZE);
		for (int i = 0; i < NUCLEO_SIZE; i++)
			SEQ_LUP[(uchar)NUCLEO_ALPHABET[i]] = i;
		break;
	}
	// Expandable
	case SEQ_TYPE_AMINO: {
		memcpy(SUB_MAT, AMINO_MATRIX[m_id].matrix, AMINO_MATSIZE);
		for (int i = 0; i < AMINO_SIZE; i++)
			SEQ_LUP[(uchar)AMINO_ALPHABET[i]] = i;
		break;
	}
	default:
	case SEQ_TYPE_COUNT:
	case SEQ_TYPE_INVALID:
		print(M_NONE, ERR "Invalid sequence type for scoring matrix");
		exit(EXIT_FAILURE);
	}

#if USE_SIMD == 1
	g_first_row_indices = set_row_indices();
	g_gap_penalty_vec = set1_epi32(args_gap_penalty());
	g_gap_open_vec = set1_epi32(args_gap_open());
	g_gap_extend_vec = set1_epi32(args_gap_extend());
#endif
}

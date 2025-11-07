#include "core/bio/score/scoring.h"

#include <string.h>

#include "core/app/args.h"
#include "core/bio/types.h"

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
	int matrix_id = args_scoring_matrix();
	int sequence_type = args_sequence_type();

	switch (sequence_type) {
	case SEQ_TYPE_NUCLEO: {
		const int (*src_matrix)[NUCLEOTIDE_SIZE] =
			ALL_NUCLEOTIDE_MATRICES[matrix_id].matrix;
		for (int i = 0; i < NUCLEOTIDE_SIZE; i++) {
			for (int j = 0; j < NUCLEOTIDE_SIZE; j++)
				SUB_MAT[i][j] = src_matrix[i][j];
		}

		break;
	}
	// Expandable
	case SEQ_TYPE_AMINO:
	default: {
		const int (*src_matrix)[AMINO_SIZE] =
			ALL_AMINO_MATRICES[matrix_id].matrix;
		for (int i = 0; i < AMINO_SIZE; i++) {
			for (int j = 0; j < AMINO_SIZE; j++)
				SUB_MAT[i][j] = src_matrix[i][j];
		}

		break;
	}
	}

	memset(SEQ_LUP, -1, sizeof(SEQ_LUP));
	switch (sequence_type) {
	case SEQ_TYPE_NUCLEO:
		for (int i = 0; i < NUCLEOTIDE_SIZE; i++)
			SEQ_LUP[(int)NUCLEOTIDE_ALPHABET[i]] = i;
		break;
	// Expandable
	case SEQ_TYPE_AMINO:
	default:
		for (int i = 0; i < AMINO_SIZE; i++)
			SEQ_LUP[(int)AMINO_ALPHABET[i]] = i;
		break;
	}

#if USE_SIMD == 1
	g_first_row_indices = set_row_indices();
	g_gap_penalty_vec = set1_epi32(args_gap_penalty());
	g_gap_open_vec = set1_epi32(args_gap_open());
	g_gap_extend_vec = set1_epi32(args_gap_extend());
#endif
}

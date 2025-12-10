#include "bio/algorithm/local/affine.h"

#include "bio/algorithm/indices.h"
#include "bio/algorithm/matrix.h"
#include "system/compiler.h"
#include "system/simd.h"

#if USE_SIMD == 1
static void affine_local_init_simd(sequence_ptr_t seq1, sequence_ptr_t seq2)
{
	if (SEQ_INVALID(seq1) || SEQ_INVALID(seq2) || !g_match || !g_gap_x ||
	    !g_gap_y)
		unreachable();

	const s32 len1 = seq1->length;
	const s32 len2 = seq2->length;
	const s32 cols = len1 + 1;

	veci_t zero_vec = setzero_si();
	veci_t score_min = set1_epi32(SCORE_MIN);

	for (s32 j = 0; j <= len1; j += NUM_ELEMS) {
		s32 remaining = cols - j;
		if (remaining >= NUM_ELEMS) {
			storeu((veci_t *)&g_match[j], zero_vec);
			storeu((veci_t *)&g_gap_x[j], score_min);
			storeu((veci_t *)&g_gap_y[j], score_min);
		} else {
			for (s32 k = 0; k < remaining; k++) {
				g_match[j + k] = 0;
				g_gap_x[j + k] = g_gap_y[j + k] = SCORE_MIN;
			}
		}
	}

	for (s32 i = 1; i <= len2; i++) {
		s64 idx = (s64)cols * i;
		g_match[idx] = 0;
		g_gap_x[idx] = SCORE_MIN;
		g_gap_y[idx] = SCORE_MIN;
	}
}
#endif

void affine_local_init(sequence_ptr_t seq1, sequence_ptr_t seq2)
{
	if (SEQ_INVALID(seq1) || SEQ_INVALID(seq2) || !g_match || !g_gap_x ||
	    !g_gap_y)
		unreachable();

	const s32 len1 = seq1->length;
#if USE_SIMD == 1
	if (len1 >= NUM_ELEMS) {
		affine_local_init_simd(seq1, seq2);
		return;
	}
#endif
	const s32 len2 = seq2->length;
	const s64 cols = len1 + 1;

	g_match[0] = 0;
	g_gap_x[0] = g_gap_y[0] = SCORE_MIN;

	for (s32 j = 1; j <= len1; j++) {
		g_match[j] = 0;
		g_gap_x[j] = g_gap_y[j] = SCORE_MIN;
	}

	for (s32 i = 1; i <= len2; i++) {
		s64 idx = cols * i;
		g_match[idx] = 0;
		g_gap_x[idx] = g_gap_y[idx] = SCORE_MIN;
	}
}

s32 affine_local_fill(sequence_ptr_t seq1, sequence_ptr_t seq2)
{
	if (SEQ_INVALID(seq1) || SEQ_INVALID(seq2) || !g_match || !g_gap_x ||
	    !g_gap_y || !g_seq1_i)
		unreachable();

	const s32 len1 = seq1->length;
	const s32 len2 = seq2->length;
	const s64 cols = len1 + 1;
	const s32 gap_open = arg_gap_open();
	const s32 gap_ext = arg_gap_ext();

	s32 score = 0;
	for (s32 i = 1; i <= len2; ++i) {
		const s64 row = cols * i;
		const s64 p_row = cols * (i - 1);
		const s32 c2_idx = SEQ_LUP[(uchar)seq2->letters[i - 1]];

		for (s32 j = 1; j <= len1; j++) {
			const s32 similarity = SUB_MAT[g_seq1_i[j - 1]][c2_idx];
			const s32 d_score = g_match[p_row + j - 1] + similarity;

			const s32 p_match_x = g_match[row + j - 1];
			const s32 p_gap_x = g_gap_x[row + j - 1];
			const s32 p_match_y = g_match[p_row + j];
			const s32 p_gap_y = g_gap_y[p_row + j];

			const s32 open_x = p_match_x + gap_open;
			const s32 extend_x = p_gap_x + gap_ext;
			const s32 open_y = p_match_y + gap_open;
			const s32 extend_y = p_gap_y + gap_ext;

			const s32 curr_gap_x = open_x > extend_x ? open_x :
								   extend_x;
			const s32 curr_gap_y = open_y > extend_y ? open_y :
								   extend_y;

			g_gap_x[row + j] = curr_gap_x;
			g_gap_y[row + j] = curr_gap_y;

			s32 best = 0;
			best = d_score > best ? d_score : best;
			best = curr_gap_x > best ? curr_gap_x : best;
			best = curr_gap_y > best ? curr_gap_y : best;
			g_match[row + j] = best;
			if (best > score)
				score = best;
		}
	}

	return score;
}

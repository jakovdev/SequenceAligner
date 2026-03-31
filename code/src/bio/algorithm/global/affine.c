#include "bio/algorithm/global/affine.h"

#include "bio/algorithm/indices.h"
#include "bio/algorithm/matrix.h"
#include "bio/score/matrices.h"
#include "system/compiler.h"
#include "system/os.h"

void affine_global_init(sequence_ptr_t seq1, sequence_ptr_t seq2)
{
	if (SEQ_INVALID(seq1) || SEQ_INVALID(seq2) || !g_match || !g_gap_x ||
	    !g_gap_y)
		unreachable_release();

	const s32 len1 = seq1->length;
	const s32 len2 = seq2->length;
	const s64 cols = len1 + 1;

	g_match[0] = 0;
	g_gap_x[0] = g_gap_y[0] = SCORE_MIN;

	for (s32 j = 1; j <= len1; j++) {
		g_gap_x[j] = max(g_match[j - 1] + GAP_OPEN,
				 g_gap_x[j - 1] + GAP_EXT);
		g_match[j] = g_gap_x[j];
		g_gap_y[j] = SCORE_MIN;
	}

	for (s32 i = 1; i <= len2; i++) {
		const s64 idx = cols * i;
		g_gap_y[idx] = max(g_match[idx - cols] + GAP_OPEN,
				   g_gap_y[idx - cols] + GAP_EXT);
		g_match[idx] = g_gap_y[idx];
		g_gap_x[idx] = SCORE_MIN;
	}
}

s32 affine_global_fill(sequence_ptr_t seq1, sequence_ptr_t seq2)
{
	if (SEQ_INVALID(seq1) || SEQ_INVALID(seq2) || !g_match || !g_gap_x ||
	    !g_gap_y || !g_seq1_i)
		unreachable_release();

	const s32 len1 = seq1->length;
	const s32 len2 = seq2->length;
	const s64 cols = len1 + 1;

	for (s32 i = 1; i <= len2; ++i) {
		const s64 row = cols * i;
		const s64 p_row = cols * (i - 1);
		const s32 c2_idx = SEQ_LUT[(uchar)seq2->letters[i - 1]];

		for (s32 j = 1; j <= len1; j++) {
			const s32 similarity = SUB_MAT[g_seq1_i[j - 1]][c2_idx];
			const s32 d_score = g_match[p_row + j - 1] + similarity;

			const s32 p_match_x = g_match[row + j - 1];
			const s32 p_gap_x = g_gap_x[row + j - 1];
			const s32 p_match_y = g_match[p_row + j];
			const s32 p_gap_y = g_gap_y[p_row + j];

			const s32 open_x = p_match_x + GAP_OPEN;
			const s32 extend_x = p_gap_x + GAP_EXT;
			const s32 open_y = p_match_y + GAP_OPEN;
			const s32 extend_y = p_gap_y + GAP_EXT;

			const s32 curr_gap_x = open_x > extend_x ? open_x :
								   extend_x;
			const s32 curr_gap_y = open_y > extend_y ? open_y :
								   extend_y;

			g_gap_x[row + j] = curr_gap_x;
			g_gap_y[row + j] = curr_gap_y;

			s32 best = d_score;
			best = curr_gap_x > best ? curr_gap_x : best;
			best = curr_gap_y > best ? curr_gap_y : best;
			g_match[row + j] = best;
		}
	}

	return g_match[(s64)len2 * (len1 + 1) + len1];
}

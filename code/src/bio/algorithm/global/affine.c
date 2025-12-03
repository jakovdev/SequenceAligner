#include "bio/algorithm/global/affine.h"

#include "bio/algorithm/indices.h"
#include "bio/algorithm/matrix.h"
#include "system/compiler.h"
#include "system/os.h"
#include "system/simd.h"

void affine_global_init(sequence_ptr_t seq1, sequence_ptr_t seq2)
{
	g_match[0] = 0;
	g_gap_x[0] = g_gap_y[0] = SCORE_MIN;

	const u64 len1 = seq1->length;
	const u64 len2 = seq2->length;
	const u64 cols = len1 + 1;
	const s32 gap_open = arg_gap_open();
	const s32 gap_ext = arg_gap_ext();

	UNROLL(8)
	for (u64 j = 1; j <= len1; j++) {
		g_gap_x[j] = max(g_match[j - 1] + gap_open,
				 g_gap_x[j - 1] + gap_ext);
		g_match[j] = g_gap_x[j];
		g_gap_y[j] = SCORE_MIN;
	}

	UNROLL(8)
	for (u64 i = 1; i <= len2; i++) {
		const u64 idx = i * cols;
		g_gap_y[idx] = max(g_match[idx - cols] + gap_open,
				   g_gap_y[idx - cols] + gap_ext);
		g_match[idx] = g_gap_y[idx];
		g_gap_x[idx] = SCORE_MIN;
	}
}

s32 affine_global_fill(sequence_ptr_t seq1, sequence_ptr_t seq2)
{
	const u64 len1 = seq1->length;
	const u64 len2 = seq2->length;
	const u64 cols = len1 + 1;
	const s32 gap_open = arg_gap_open();
	const s32 gap_ext = arg_gap_ext();

	for (u64 i = 1; i <= len2; ++i) {
		const u64 row = i * cols;
		const u64 p_row = (i - 1) * cols;
		const s32 c2_idx = SEQ_LUP[(uchar)seq2->letters[i - 1]];

		prefetch(&g_match[row + PREFETCH_DISTANCE]);
		prefetch(&g_gap_x[row + PREFETCH_DISTANCE]);
		prefetch(&g_gap_y[row + PREFETCH_DISTANCE]);

		VECTORIZE
		for (u64 j = 1; j <= len1; j++) {
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

			s32 best = d_score;
			best = curr_gap_x > best ? curr_gap_x : best;
			best = curr_gap_y > best ? curr_gap_y : best;
			g_match[row + j] = best;
		}
	}

	return g_match[len2 * (len1 + 1) + len1];
}

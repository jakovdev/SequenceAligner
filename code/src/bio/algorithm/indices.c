#include "bio/algorithm/indices.h"

#include "bio/score/matrices.h"
#include "bio/sequence/sequences.h"
#include "system/compiler.h"
#include "system/memory.h"
#include "util/print.h"

thread_local s32 *g_restrict g_seq1_i;

void indices_buffers_init(void)
{
	if (g_seq_len_max < SEQ_LEN_MIN || g_seq_len_max > SEQ_LEN_MAX) {
		pdev("Invalid seq_len_max in indices_buffers_init()");
		perr("Internal error during sequence indices allocation");
		pabort();
	}

	MALLOCA_AL(g_seq1_i, CACHE_LINE, g_seq_len_max);
	if unlikely (!g_seq1_i) {
		perr("Out of memory allocating sequence indices");
		exit(EXIT_FAILURE);
	}
}

void indices_buffers_free(void)
{
	if (!g_seq1_i) {
		pdev("Call indices_buffers_init() before indices_buffers_free()");
		perr("Internal error during sequence indices deallocation");
		pabort();
	}

	free_aligned(g_seq1_i);
	g_seq1_i = NULL;
}

void indices_precompute(sequence_ptr_t seq)
{
	if (SEQ_INVALID(seq) || !g_seq1_i)
		unreachable_release();

	for (s32 i = 0; i < seq->length; ++i)
		g_seq1_i[i] = SEQ_LUT[(uchar)seq->letters[i]];
}

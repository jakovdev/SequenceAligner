#include "bio/algorithm/indices.h"

#include "system/compiler.h"
#include "system/simd.h"
#include "system/memory.h"
#include "system/os.h"
#include "util/print.h"

_Thread_local s32 *restrict g_seq1_i;

void indices_buffers_init(s32 seq_len_max)
{
	if (seq_len_max < SEQ_LEN_MIN || seq_len_max > SEQ_LEN_MAX) {
		pdev("Invalid seq_len_max in indices_buffers_init()");
		perr("Internal error during sequence indices allocation");
		exit(EXIT_FAILURE);
	}

	MALLOCA_CL(g_seq1_i, (size_t)seq_len_max);
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
		exit(EXIT_FAILURE);
	}

	free_aligned(g_seq1_i);
	g_seq1_i = NULL;
}

void indices_precompute(sequence_ptr_t seq)
{
	if (SEQ_INVALID(seq) || !g_seq1_i)
		unreachable();

#if USE_SIMD == 1
	const s32 vector_len = (seq->length / BYTES) * BYTES;
	s32 i = 0;

	for (; i < vector_len; i += BYTES) {
		prefetch(seq->letters + i + BYTES * 2);
		VECTORIZE
		for (s32 j = 0; j < BYTES; j++)
			g_seq1_i[i + j] = SEQ_LUP[(uchar)seq->letters[i + j]];
	}

	for (; i < seq->length; i++)
		g_seq1_i[i] = SEQ_LUP[(uchar)seq->letters[i]];
#else
	VECTORIZE
	UNROLL(8)
	for (s32 i = 0; i < seq->length; ++i)
		g_seq1_i[i] = SEQ_LUP[(uchar)seq->letters[i]];
#endif
}

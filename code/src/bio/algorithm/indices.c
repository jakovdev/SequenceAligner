#include "bio/algorithm/indices.h"

#include "system/compiler.h"
#include "system/simd.h"
#include "system/memory.h"
#include "system/os.h"
#include "util/print.h"

_Thread_local s32 *restrict g_seq1_i;

void indices_buffers_init(u32 lenmax)
{
	MALLOC_CL(g_seq1_i, lenmax);
	if (UNLIKELY(!g_seq1_i)) {
		perr("Failed to allocate memory for sequence indices");
		exit(EXIT_FAILURE);
	}
}

void indices_buffers_free(void)
{
	if (g_seq1_i)
		free_aligned(g_seq1_i);
	g_seq1_i = NULL;
}

void indices_precompute(sequence_ptr_t seq)
{
#if USE_SIMD == 1
	const u64 vector_len = (seq->length / BYTES) * BYTES;
	u64 i = 0;

	for (; i < vector_len; i += BYTES) {
		prefetch(seq->letters + i + BYTES * 2);
		VECTORIZE
		for (u64 j = 0; j < BYTES; j++)
			g_seq1_i[i + j] = SEQ_LUP[(uchar)seq->letters[i + j]];
	}

	for (; i < seq->length; i++)
		g_seq1_i[i] = SEQ_LUP[(uchar)seq->letters[i]];
#else
	VECTORIZE
	UNROLL(8)
	for (u64 i = 0; i < seq->length; ++i)
		g_seq1_i[i] = SEQ_LUP[(uchar)seq->letters[i]];
#endif
}

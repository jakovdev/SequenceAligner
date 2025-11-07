#include "core/bio/algorithm/indices.h"

#include "core/bio/score/scoring.h"
#include "system/compiler.h"
#include "system/simd.h"

void seq_indices_precompute(s32 *restrict indices, sequence_ptr_t seq)
{
#if USE_SIMD == 1
	const u64 vector_len = (seq->length / BYTES) * BYTES;
	u64 i = 0;

	for (; i < vector_len; i += BYTES) {
		prefetch(seq->letters + i + BYTES * 2);
		VECTORIZE for (u64 j = 0; j < BYTES; j++) indices[i + j] =
			SEQUENCE_LOOKUP[(uchar)seq->letters[i + j]];
	}

	for (; i < seq->length; i++)
		indices[i] = SEQUENCE_LOOKUP[(uchar)seq->letters[i]];
#else
	VECTORIZE UNROLL(8) for (u64 i = 0; i < seq->length; ++i)
		indices[i] = SEQUENCE_LOOKUP[(uchar)seq->letters[i]];
#endif
}

void seq_indices_free(s32 *restrict indices, bool is_stack)
{
	if (!is_stack && indices) {
		free(indices);
		indices = NULL;
	}
}

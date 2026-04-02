#include "bio/algorithm/indices.h"

#include "bio/score/matrices.h"
#include "bio/sequence/sequences.h"
#include "system/compiler.h"
#include "system/memory.h"
#include "util/print.h"

thread_local s32 *g_restrict SEQ1I;

void indices_buffers_init(void)
{
	if (LENGTHS_MAX < SEQ_LEN_MIN || LENGTHS_MAX > SEQ_LEN_MAX) {
		pdev("Invalid LENGTHS_MAX in indices_buffers_init()");
		perr("Internal error during sequence indices allocation");
		pabort();
	}

	MALLOCA_AL(SEQ1I, CACHE_LINE, LENGTHS_MAX);
	if unlikely (!SEQ1I) {
		perr("Out of memory allocating sequence indices");
		exit(EXIT_FAILURE);
	}
}

void indices_buffers_free(void)
{
	if (!SEQ1I) {
		pdev("Call indices_buffers_init() before indices_buffers_free()");
		perr("Internal error during sequence indices deallocation");
		pabort();
	}

	free_aligned(SEQ1I);
	SEQ1I = NULL;
}

void indices_precompute(SEQUENCE_PTR_T(seq))
{
	if (SEQ_BAD(seq) || !SEQ1I)
		unreachable_release();

	for (s32 i = 0; i < seq->length; ++i)
		SEQ1I[i] = SEQ_LUT[(uchar)seq->letters[i]];
}

#include "bio/algorithm/matrix.h"

#include "bio/sequence/sequences.h"
#include "bio/types.h"
#include "system/memory.h"
#include "util/print.h"

thread_local s32 *g_restrict MATRIX;
thread_local s32 *g_restrict MATCH;
thread_local s32 *g_restrict GAP_X;
thread_local s32 *g_restrict GAP_Y;

#define MATRIX_SIZE(len) ((len + 1) * (len + 1))

void matrix_buffers_init(void)
{
	if (LENGTHS_MAX < SEQ_LEN_MIN || LENGTHS_MAX > SEQ_LEN_MAX) {
		pdev("Invalid LENGTHS_MAX in matrix_buffers_init()");
		perr("Internal error during matrix allocation");
		pabort();
	}

	MALLOCA_AL(MATRIX, CACHE_LINE, 3 * MATRIX_SIZE(LENGTHS_MAX));
	if unlikely (!MATRIX) {
		perr("Out of memory allocating alignment matrices");
		exit(EXIT_FAILURE);
	}

	MATCH = MATRIX;
	GAP_X = MATRIX + MATRIX_SIZE(LENGTHS_MAX);
	GAP_Y = MATRIX + 2 * MATRIX_SIZE(LENGTHS_MAX);
}

void matrix_buffers_free(void)
{
	if (!MATRIX) {
		pdev("Call matrix_buffers_init() before matrix_buffers_free()");
		perr("Internal error during alignment matrices deallocation");
		pabort();
	}

	free_aligned(MATRIX);
	MATRIX = NULL;
	MATCH = NULL;
	GAP_X = NULL;
	GAP_Y = NULL;
}

#include "bio/algorithm/matrix.h"

#include "bio/types.h"
#include "system/memory.h"
#include "system/os.h"
#include "util/print.h"

_Thread_local s32 *restrict g_matrix;
_Thread_local s32 *restrict g_match;
_Thread_local s32 *restrict g_gap_x;
_Thread_local s32 *restrict g_gap_y;

#define MATRIX_SIZE(len) ((size_t)(len + 1) * (size_t)(len + 1))

void matrix_buffers_init(s32 seq_len_max)
{
	if (seq_len_max < SEQ_LEN_MIN || seq_len_max > SEQ_LEN_MAX) {
		pdev("Invalid seq_len_max in matrix_buffers_init()");
		perr("Internal error during matrix allocation");
		exit(EXIT_FAILURE);
	}

	MALLOCA_CL(g_matrix, 3 * MATRIX_SIZE(seq_len_max));
	if unlikely (!g_matrix) {
		perr("Out of memory allocating alignment matrices");
		exit(EXIT_FAILURE);
	}

	g_match = g_matrix;
	g_gap_x = g_matrix + MATRIX_SIZE(seq_len_max);
	g_gap_y = g_matrix + 2 * MATRIX_SIZE(seq_len_max);
}

void matrix_buffers_free(void)
{
	if (!g_matrix) {
		pdev("Call matrix_buffers_init() before matrix_buffers_free()");
		perr("Internal error during alignment matrices deallocation");
		exit(EXIT_FAILURE);
	}

	free_aligned(g_matrix);
	g_matrix = NULL;
	g_match = NULL;
	g_gap_x = NULL;
	g_gap_y = NULL;
}

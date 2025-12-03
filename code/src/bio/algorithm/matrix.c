#include "bio/algorithm/matrix.h"

#include "bio/types.h"
#include "system/memory.h"
#include "system/os.h"
#include "util/print.h"

_Thread_local s32 *restrict g_matrix;
_Thread_local s32 *restrict g_match;
_Thread_local s32 *restrict g_gap_x;
_Thread_local s32 *restrict g_gap_y;

#define MATRIX_SIZE(len) ((len + 1) * (len + 1))

void matrix_buffers_init(u32 lenmax)
{
	g_matrix = alloc_huge_page(3 * MATRIX_SIZE(lenmax) * sizeof(*g_matrix));
	if (UNLIKELY(!g_matrix)) {
		perr("Failed to allocate memory for alignment matrix");
		exit(EXIT_FAILURE);
	}

	g_match = g_matrix;
	g_gap_x = g_matrix + MATRIX_SIZE(lenmax);
	g_gap_y = g_matrix + 2 * MATRIX_SIZE(lenmax);
}

void matrix_buffers_free(void)
{
	if (g_matrix)
		aligned_free(g_matrix);
	g_matrix = NULL;
	g_match = NULL;
	g_gap_x = NULL;
	g_gap_y = NULL;
}

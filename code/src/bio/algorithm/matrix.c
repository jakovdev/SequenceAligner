#include "bio/algorithm/matrix.h"

#include "system/memory.h"
#include "system/os.h"

s32 *matrix_alloc(s32 *stack_matrix, size_t bytes)
{
	return USE_STACK_MATRIX(bytes) ? stack_matrix : alloc_huge_page(bytes);
}

void matrix_free(s32 *matrix, s32 *stack_matrix)
{
	if (matrix != stack_matrix)
		aligned_free(matrix);
}

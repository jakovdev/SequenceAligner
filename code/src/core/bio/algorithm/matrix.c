#include "core/bio/algorithm/matrix.h"

score_t*
matrix_alloc(score_t* stack_matrix, size_t bytes)
{
    if (USE_STACK_MATRIX(bytes))
    {
        return stack_matrix;
    }

    else
    {
        return (score_t*)(alloc_huge_page(bytes));
    }
}

void
matrix_free(score_t* matrix, score_t* stack_matrix)
{
    if (matrix != stack_matrix)
    {
        aligned_free(matrix);
    }
}

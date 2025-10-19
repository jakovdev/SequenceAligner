#include "core/bio/algorithm/indices.h"

#include "core/bio/score/scoring.h"
#include "system/arch.h"

void
seq_indices_precompute(SeqIndices* indices, const sequence_ptr_t seq)
{
#ifdef USE_SIMD
    const sequence_length_t vector_len = (seq->length / BYTES) * BYTES;
    sequence_length_t i = 0;

    for (; i < vector_len; i += BYTES)
    {
        prefetch(seq->letters + i + BYTES * 2);

        VECTORIZE for (sequence_length_t j = 0; j < BYTES; j++)
        {
            indices->data[i + j] = SEQUENCE_LOOKUP[(unsigned char)seq->letters[i + j]];
        }
    }

    for (; i < seq->length; i++)
    {
        indices->data[i] = SEQUENCE_LOOKUP[(unsigned char)seq->letters[i]];
    }

#else

    VECTORIZE UNROLL(8) for (sequence_length_t i = 0; i < seq->length; ++i)
    {
        indices->data[i] = SEQUENCE_LOOKUP[(unsigned char)seq->letters[i]];
    }

#endif
}

void
seq_indices_free(SeqIndices* indices)
{
    if (!indices->is_stack && indices->data)
    {
        free(indices->data);
        indices->data = NULL;
    }
}

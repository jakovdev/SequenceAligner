#ifndef SEQUENCE_H
#define SEQUENCE_H

#include "args.h"
#include "csv.h"
#include "print.h"

typedef struct
{
    char* data;
    size_t length;
} Sequence;

#define SEQ_POOL_BLOCK_SIZE (4 * MiB)

typedef struct SeqMemBlock
{
    char* data;
    size_t used;
    size_t capacity;
    struct SeqMemBlock* next;
} SeqMemBlock;

typedef struct
{
    SeqMemBlock* head;
    SeqMemBlock* current;
    size_t total_bytes;
    size_t total_allocated;
    size_t block_count;
} SeqMemPool;

typedef struct
{
    Sequence* sequences;
    size_t count;
    size_t total_alignments;
} SequenceData;

static SeqMemPool g_seq_pool = { 0 };

INLINE void
seq_pool_init(void)
{
    if (g_seq_pool.head)
    {
        return;
    }

    g_seq_pool.head = malloc(sizeof(*g_seq_pool.head));
    if (!g_seq_pool.head)
    {
        return;
    }

    g_seq_pool.head->data = alloc_huge_page(SEQ_POOL_BLOCK_SIZE);
    if (!g_seq_pool.head->data)
    {
        free(g_seq_pool.head);
        g_seq_pool.head = NULL;
        return;
    }

    g_seq_pool.head->used = 0;
    g_seq_pool.head->capacity = SEQ_POOL_BLOCK_SIZE;
    g_seq_pool.head->next = NULL;
    g_seq_pool.current = g_seq_pool.head;
    g_seq_pool.total_bytes = SEQ_POOL_BLOCK_SIZE;
    g_seq_pool.total_allocated = 0;
    g_seq_pool.block_count = 1;
}

INLINE char*
seq_pool_alloc(size_t size)
{
    if (!g_seq_pool.head)
    {
        seq_pool_init();
        if (!g_seq_pool.head)
        {
            return NULL;
        }
    }

    size = (size + 7) & ~7;

    if (g_seq_pool.current->used + size > g_seq_pool.current->capacity)
    {
        size_t new_block_size = SEQ_POOL_BLOCK_SIZE;
        if (size > new_block_size)
        {
            new_block_size = size;
        }

        SeqMemBlock* new_block = malloc(sizeof(*new_block));
        if (!new_block)
        {
            return NULL;
        }

        new_block->data = malloc(new_block_size);
        if (!new_block->data)
        {
            free(new_block);
            return NULL;
        }

        new_block->used = 0;
        new_block->capacity = new_block_size;
        new_block->next = NULL;

        g_seq_pool.current->next = new_block;
        g_seq_pool.current = new_block;
        g_seq_pool.total_bytes += new_block_size;
        g_seq_pool.block_count++;
    }

    char* result = g_seq_pool.current->data + g_seq_pool.current->used;
    g_seq_pool.current->used += size;
    g_seq_pool.total_allocated += size;

    return result;
}

INLINE __attribute__((destructor)) void
seq_pool_free(void)
{
    if (!g_seq_pool.head)
    {
        return;
    }

    SeqMemBlock* block = g_seq_pool.head;
    while (block)
    {
        SeqMemBlock* next = block->next;
        aligned_free(block->data);
        free(block);
        block = next;
    }

    g_seq_pool.head = NULL;
    g_seq_pool.current = NULL;
    g_seq_pool.total_bytes = 0;
    g_seq_pool.total_allocated = 0;
    g_seq_pool.block_count = 0;
}

INLINE void
sequence_init(Sequence* seq, const char* data, size_t length)
{
    seq->length = length;

    seq->data = seq_pool_alloc(length + 1);
    if (seq->data)
    {
        memcpy(seq->data, data, length);
        seq->data[length] = '\0';
    }
}

INLINE float
similarity_pairwise(const char* restrict seq1, size_t len1, const char* restrict seq2, size_t len2)
{
    if (UNLIKELY(!len1 || !len2))
    {
        return 0.0f;
    }

    size_t min_len = len1 < len2 ? len1 : len2;
    size_t matches = 0;

#ifdef USE_SIMD
    size_t vec_limit = (min_len / BYTES) * BYTES;

    for (size_t i = 0; i < vec_limit; i += BYTES * 2)
    {
        prefetch(seq1 + i + BYTES);
        prefetch(seq2 + i + BYTES);
    }

    for (size_t i = 0; i < vec_limit; i += BYTES)
    {
        veci_t v1 = loadu((const veci_t*)(seq1 + i));
        veci_t v2 = loadu((const veci_t*)(seq2 + i));

#if defined(__AVX512F__) && defined(__AVX512BW__)
        num_t mask = cmpeq_epi8(v1, v2);
        matches += __builtin_popcountll(mask);
#else
        num_t mask = movemask_epi8(cmpeq_epi8(v1, v2));
        matches += __builtin_popcount(mask);
#endif
    }

    for (size_t i = vec_limit; i < min_len; i++)
    {
        matches += (seq1[i] == seq2[i]);
    }

#else
    for (size_t i = 0; i < min_len; i++)
    {
        matches += (seq1[i] == seq2[i]);
    }

#endif

    return (float)matches / min_len;
}

INLINE void
sequences_alloc_from_file(SequenceData* seq_data,
                          char* current,
                          char* end,
                          size_t total_seqs_in_file)
{
    float filter_threshold = args_filter_threshold();
    bool apply_filtering = args_mode_filter();

    Sequence* seqs = malloc(total_seqs_in_file * sizeof(*seqs));
    if (!seqs)
    {
        return;
    }

    size_t idx = 0;
    size_t filtered_count = 0;
    size_t total_sequence_length = 0;

    char* temp_seq = NULL;
    size_t temp_seq_capacity = 0;

    while (current < end && *current)
    {
        char* line_end = current;
        while (*line_end && *line_end != '\n' && *line_end != '\r')
        {
            line_end++;
        }

        size_t max_line_len = line_end - current;

        if (max_line_len + 16 > temp_seq_capacity)
        {
            size_t new_capacity = max_line_len + 64;
            char* new_buffer = malloc(new_capacity);
            if (!new_buffer)
            {
                free(temp_seq);
                free(seqs);
                return;
            }

            free(temp_seq);
            temp_seq = new_buffer;
            temp_seq_capacity = new_capacity;
        }

        size_t seq_len = csv_line_parse(&current, temp_seq);

        if (!seq_len)
        {
            continue;
        }

        bool should_include = true;

        if (apply_filtering && idx > 0)
        {
            for (size_t j = 0; j < idx; j++)
            {
                float similarity = similarity_pairwise(temp_seq,
                                                       seq_len,
                                                       seqs[j].data,
                                                       seqs[j].length);

                if (similarity >= filter_threshold)
                {
                    should_include = false;
                    filtered_count++;
                    break;
                }
            }
        }

        if (should_include)
        {
            sequence_init(&seqs[idx], temp_seq, seq_len);
            total_sequence_length += seq_len;
            idx++;
        }

        print(PROGRESS,
              MSG_PROPORTION((float)(idx + filtered_count) / total_seqs_in_file),
              apply_filtering ? "Filtering sequences" : "Loading sequences");
    }

    free(temp_seq);

    size_t seq_count = idx;

    if (apply_filtering && filtered_count > 0 && filtered_count >= total_seqs_in_file / 4)
    {
        print(VERBOSE, MSG_NONE, "Reallocating memory to save %zu sequence slots", filtered_count);
        Sequence* new_seqs = realloc(seqs, seq_count * sizeof(*seqs));
        if (new_seqs)
        {
            seqs = new_seqs;
        }
    }

    if (apply_filtering)
    {
        print(DNA, MSG_NONE, "Loaded %zu sequences (filtered out %zu)", seq_count, filtered_count);
    }

    print(INFO,
          MSG_NONE,
          "Average sequence length: %.2f",
          (float)total_sequence_length / seq_count);

    size_t total_alignments = (seq_count * (seq_count - 1)) / 2;

    seq_data->sequences = seqs;
    seq_data->count = seq_count;
    seq_data->total_alignments = total_alignments;

    return;
}

INLINE void
seq_get_pair(SequenceData* seq_data,
             size_t first,
             size_t second,
             char** seq1,
             size_t* len1,
             char** seq2,
             size_t* len2)
{
    *seq1 = seq_data->sequences[first].data;
    *len1 = seq_data->sequences[first].length;
    *seq2 = seq_data->sequences[second].data;
    *len2 = seq_data->sequences[second].length;

    prefetch(*seq1);
    prefetch(*seq2);
}

#endif // SEQUENCE_H
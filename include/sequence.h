#ifndef SEQUENCE_H
#define SEQUENCE_H

#include "args.h"
#include "csv.h"
#include "print.h"

typedef struct
{
    char* letters;
    size_t length;
} sequence_t;

#define SEQ_POOL_BLOCK_SIZE (4 * MiB)

typedef struct SeqMemBlock
{
    char* sequence_block;
    size_t used;
    size_t capacity;
    struct SeqMemBlock* next;
} SeqMemBlock;

static struct
{
    SeqMemBlock* head;
    SeqMemBlock* current;
    size_t total_bytes;
    size_t total_allocated;
    size_t block_count;
} g_sequence_pool = { 0 };

static struct
{
    sequence_t* sequences;
    size_t sequence_count;
    size_t alignment_count;
} g_sequence_dataset = { 0 };

INLINE void
seq_pool_init(void)
{
    if (g_sequence_pool.head)
    {
        return;
    }

    g_sequence_pool.head = malloc(sizeof(*g_sequence_pool.head));
    if (!g_sequence_pool.head)
    {
        return;
    }

    g_sequence_pool.head->sequence_block = alloc_huge_page(SEQ_POOL_BLOCK_SIZE);
    if (!g_sequence_pool.head->sequence_block)
    {
        free(g_sequence_pool.head);
        g_sequence_pool.head = NULL;
        return;
    }

    g_sequence_pool.head->used = 0;
    g_sequence_pool.head->capacity = SEQ_POOL_BLOCK_SIZE;
    g_sequence_pool.head->next = NULL;
    g_sequence_pool.current = g_sequence_pool.head;
    g_sequence_pool.total_bytes = SEQ_POOL_BLOCK_SIZE;
    g_sequence_pool.total_allocated = 0;
    g_sequence_pool.block_count = 1;
}

INLINE char*
seq_pool_alloc(size_t size)
{
    if (!g_sequence_pool.head)
    {
        seq_pool_init();
        if (!g_sequence_pool.head)
        {
            return NULL;
        }
    }

    size = (size + 7) & ~7;

    if (g_sequence_pool.current->used + size > g_sequence_pool.current->capacity)
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

        new_block->sequence_block = malloc(new_block_size);
        if (!new_block->sequence_block)
        {
            free(new_block);
            return NULL;
        }

        new_block->used = 0;
        new_block->capacity = new_block_size;
        new_block->next = NULL;

        g_sequence_pool.current->next = new_block;
        g_sequence_pool.current = new_block;
        g_sequence_pool.total_bytes += new_block_size;
        g_sequence_pool.block_count++;
    }

    char* result = g_sequence_pool.current->sequence_block + g_sequence_pool.current->used;
    g_sequence_pool.current->used += size;
    g_sequence_pool.total_allocated += size;

    return result;
}

INLINE __attribute__((destructor)) void
seq_pool_free(void)
{
    if (!g_sequence_pool.head)
    {
        return;
    }

    SeqMemBlock* block = g_sequence_pool.head;
    while (block)
    {
        SeqMemBlock* next = block->next;
        aligned_free(block->sequence_block);
        free(block);
        block = next;
    }

    g_sequence_pool.head = NULL;
    g_sequence_pool.current = NULL;
    g_sequence_pool.total_bytes = 0;
    g_sequence_pool.total_allocated = 0;
    g_sequence_pool.block_count = 0;
}

INLINE void
sequence_init(sequence_t* sequence, const char* data, size_t sequence_length)
{
    sequence->length = sequence_length;

    sequence->letters = seq_pool_alloc(sequence_length + 1);
    if (sequence->letters)
    {
        memcpy(sequence->letters, data, sequence_length);
        sequence->letters[sequence_length] = '\0';
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
sequences_alloc_from_file(char* file_cursor, char* file_end, size_t sequences_total)
{
    float filter_threshold = args_filter_threshold();
    bool apply_filtering = args_mode_filter();

    sequence_t* sequences = malloc(sequences_total * sizeof(*sequences));
    if (!sequences)
    {
        return;
    }

    const size_t buffer_margin = 16;
    const size_t buffer_padding = 64;

    size_t sequence_count_current = 0;
    size_t filtered_count = 0;
    size_t total_sequence_length = 0;

    char* temp_seq = NULL;
    size_t temp_seq_capacity = 0;

    while (file_cursor < file_end && *file_cursor)
    {
        char* line_end = file_cursor;
        while (*line_end && *line_end != '\n' && *line_end != '\r')
        {
            line_end++;
        }

        size_t max_line_len = line_end - file_cursor;

        if (max_line_len + buffer_margin > temp_seq_capacity)
        {
            size_t new_capacity = max_line_len + buffer_padding;
            char* new_buffer = malloc(new_capacity);
            if (!new_buffer)
            {
                free(temp_seq);
                free(sequences);
                return;
            }

            free(temp_seq);
            temp_seq = new_buffer;
            temp_seq_capacity = new_capacity;
        }

        size_t sequence_length = csv_line_parse(&file_cursor, temp_seq);

        if (!sequence_length)
        {
            continue;
        }

        bool should_include = true;

        if (apply_filtering && sequence_count_current > 0)
        {
            for (size_t j = 0; j < sequence_count_current; j++)
            {
                float similarity = similarity_pairwise(temp_seq,
                                                       sequence_length,
                                                       sequences[j].letters,
                                                       sequences[j].length);

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
            sequence_init(&sequences[sequence_count_current], temp_seq, sequence_length);
            total_sequence_length += sequence_length;
            sequence_count_current++;
        }

        print(PROGRESS,
              MSG_PROPORTION((float)(sequence_count_current + filtered_count) / sequences_total),
              apply_filtering ? "Filtering sequences" : "Loading sequences");
    }

    free(temp_seq);

    size_t sequence_count = sequence_count_current;

    if (apply_filtering && filtered_count > 0 && filtered_count >= sequences_total / 4)
    {
        print(VERBOSE, MSG_NONE, "Reallocating memory to save %zu sequence slots", filtered_count);
        sequence_t* _sequences_new = realloc(sequences, sequence_count * sizeof(*sequences));
        if (_sequences_new)
        {
            sequences = _sequences_new;
        }
    }

    if (apply_filtering)
    {
        print(DNA, MSG_NONE, "Loaded %zu sequences (filtered %zu)", sequence_count, filtered_count);
    }

    float sequence_average_length = (float)total_sequence_length / sequence_count;

    print(INFO, MSG_NONE, "Average sequence length: %.2f", sequence_average_length);

    size_t alignment_count = (sequence_count * (sequence_count - 1)) / 2;

    g_sequence_dataset.sequences = sequences;
    g_sequence_dataset.sequence_count = sequence_count;
    g_sequence_dataset.alignment_count = alignment_count;

    return;
}

INLINE void
seq_get_pair(size_t first_index,
             char** first_sequence_out,
             size_t* first_length_out,
             size_t second_index,
             char** second_sequence_out,
             size_t* second_length_out)
{
    *first_sequence_out = g_sequence_dataset.sequences[first_index].letters;
    *first_length_out = g_sequence_dataset.sequences[first_index].length;
    *second_sequence_out = g_sequence_dataset.sequences[second_index].letters;
    *second_length_out = g_sequence_dataset.sequences[second_index].length;

    prefetch(*first_sequence_out);
    prefetch(*second_sequence_out);
}

#endif // SEQUENCE_H
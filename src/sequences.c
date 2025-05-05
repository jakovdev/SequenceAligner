#include "sequences.h"

#include "arch.h"
#include "csv.h"
#include "print.h"

typedef struct SeqMemBlock
{
#define SB_SIZE (4 * MiB)
    char* block;
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
} g_pool = { 0 };

static struct
{
    sequence_t* sequences;
    size_t sequence_count;
    size_t alignment_count;
} g_sequence_dataset = { 0 };

static void
seq_pool_init(void)
{
    if (g_pool.head)
    {
        return;
    }

    g_pool.head = MALLOC(g_pool.head, 1);
    if (!g_pool.head)
    {
        return;
    }

    g_pool.head->block = CAST(g_pool.head->block)(alloc_huge_page(SB_SIZE));
    if (!g_pool.head->block)
    {
        free(g_pool.head);
        g_pool.head = NULL;
        return;
    }

    g_pool.head->used = 0;
    g_pool.head->capacity = SB_SIZE;
    g_pool.head->next = NULL;
    g_pool.current = g_pool.head;
    g_pool.total_bytes = SB_SIZE;
    g_pool.total_allocated = 0;
    g_pool.block_count = 1;
}

static char*
seq_pool_alloc(size_t size)
{
    if (!g_pool.head)
    {
        seq_pool_init();
        if (!g_pool.head)
        {
            return NULL;
        }
    }

    size = (size + 7) & ~7;

    if (g_pool.current->used + size > g_pool.current->capacity)
    {
        size_t new_block_size = SB_SIZE;
        if (size > new_block_size)
        {
            new_block_size = size;
        }

        SeqMemBlock* new_block = MALLOC(new_block, 1);
        if (!new_block)
        {
            return NULL;
        }

        new_block->block = MALLOC(new_block->block, new_block_size);
        if (!new_block->block)
        {
            free(new_block);
            return NULL;
        }

        new_block->used = 0;
        new_block->capacity = new_block_size;
        new_block->next = NULL;

        g_pool.current->next = new_block;
        g_pool.current = new_block;
        g_pool.total_bytes += new_block_size;
        g_pool.block_count++;
    }

    char* result = g_pool.current->block + g_pool.current->used;
    g_pool.current->used += size;
    g_pool.total_allocated += size;

    return result;
}

DESTRUCTOR static void
seq_pool_free(void)
{
    if (!g_pool.head)
    {
        return;
    }

    SeqMemBlock* curr = g_pool.head;
    while (curr)
    {
        SeqMemBlock* next = curr->next;
        aligned_free(curr->block);
        free(curr);
        curr = next;
    }

    g_pool.head = NULL;
    g_pool.current = NULL;
    g_pool.total_bytes = 0;
    g_pool.total_allocated = 0;
    g_pool.block_count = 0;
}

DESTRUCTOR static void
sequences_free(void)
{
    if (g_sequence_dataset.sequences)
    {
        free(g_sequence_dataset.sequences);
        g_sequence_dataset.sequences = NULL;
    }

    g_sequence_dataset.sequence_count = 0;
    g_sequence_dataset.alignment_count = 0;
}

static void
sequence_init(sequence_t* sequence, const char* letters, size_t sequence_length)
{
    sequence->length = sequence_length;

    sequence->letters = seq_pool_alloc(sequence_length + 1);
    if (sequence->letters)
    {
        memcpy(sequence->letters, letters, sequence_length);
        sequence->letters[sequence_length] = '\0';
    }
}

static float
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

void
sequences_alloc_from_file(char* file_cursor,
                          char* file_end,
                          size_t sequences_total,
                          float filter_threshold,
                          bool apply_filtering,
                          int sequence_column)
{
    seq_pool_init();
    sequence_t* sequences = MALLOC(sequences, sequences_total);
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
            char* new_buffer = MALLOC(new_buffer, new_capacity);
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

        size_t sequence_length = csv_line_column_extract(&file_cursor, temp_seq, sequence_column);

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
        sequence_t* _sequences_new = REALLOC(sequences, sequence_count);
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

void
sequences_get_pair(size_t first_index,
                   char* restrict* first_sequence_out,
                   size_t* restrict first_length_out,
                   size_t second_index,
                   char* restrict* second_sequence_out,
                   size_t* restrict second_length_out)
{
    *first_sequence_out = g_sequence_dataset.sequences[first_index].letters;
    *first_length_out = g_sequence_dataset.sequences[first_index].length;
    *second_sequence_out = g_sequence_dataset.sequences[second_index].letters;
    *second_length_out = g_sequence_dataset.sequences[second_index].length;

    prefetch(*first_sequence_out);
    prefetch(*second_sequence_out);
}

sequence_t*
sequences_get(void)
{
    return g_sequence_dataset.sequences;
}

size_t
sequences_count(void)
{
    return g_sequence_dataset.sequence_count;
}

size_t
sequences_alignment_count(void)
{
    return g_sequence_dataset.alignment_count;
}
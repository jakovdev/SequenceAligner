#include "sequences.h"

#include "arch.h"
#include "biotypes.h"
#include "files.h"
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
    sequences_t sequences;
    sequence_count_t sequence_count;
    alignment_size_t alignment_count;
#ifdef USE_CUDA
    char* flat_sequences;
    sequence_offset_t* flat_offsets;
    quar_t* flat_lengths;
    size_t total_sequence_length;
    sequence_length_t max_sequence_length;
#endif
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

    size = (size + 7) & ~((size_t)7);

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
        curr->block = NULL;
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

#ifdef USE_CUDA
    if (g_sequence_dataset.flat_sequences)
    {
        free(g_sequence_dataset.flat_sequences);
        g_sequence_dataset.flat_sequences = NULL;
    }

    if (g_sequence_dataset.flat_offsets)
    {
        free(g_sequence_dataset.flat_offsets);
        g_sequence_dataset.flat_offsets = NULL;
    }

    if (g_sequence_dataset.flat_lengths)
    {
        free(g_sequence_dataset.flat_lengths);
        g_sequence_dataset.flat_lengths = NULL;
    }

    g_sequence_dataset.total_sequence_length = 0;
    g_sequence_dataset.max_sequence_length = 0;
#endif
}

static void
sequence_init(sequence_ptr_t sequence, const char* letters, sequence_length_t sequence_length)
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
similarity_pairwise(const sequence_ptr_t seq1, const sequence_ptr_t seq2)
{
    if (UNLIKELY(!seq1->length || !seq2->length))
    {
        return 0.0f;
    }

    sequence_length_t min_len = seq1->length < seq2->length ? seq1->length : seq2->length;
    size_t matches = 0;

#ifdef USE_SIMD
    sequence_length_t vec_limit = (min_len / BYTES) * BYTES;

    for (sequence_length_t i = 0; i < vec_limit; i += BYTES * 2)
    {
        prefetch(seq1->letters + i + BYTES);
        prefetch(seq2->letters + i + BYTES);
    }

    for (sequence_length_t i = 0; i < vec_limit; i += BYTES)
    {
        veci_t v1 = loadu((const veci_t*)(seq1->letters + i));
        veci_t v2 = loadu((const veci_t*)(seq2->letters + i));

#if defined(__AVX512F__) && defined(__AVX512BW__)
        num_t mask = cmpeq_epi8(v1, v2);
        matches += (size_t)__builtin_popcountll(mask);
#else
        num_t mask = movemask_epi8(cmpeq_epi8(v1, v2));
        matches += (size_t)__builtin_popcount(mask);
#endif
    }

    for (sequence_length_t i = vec_limit; i < min_len; i++)
    {
        matches += (seq1->letters[i] == seq2->letters[i]);
    }

#else
    for (sequence_length_t i = 0; i < min_len; i++)
    {
        matches += (seq1->letters[i] == seq2->letters[i]);
    }

#endif

    return (float)matches / (float)min_len;
}

bool
sequences_alloc_from_file(FileTextPtr input_file, float filter_threshold)
{
    if (!input_file || !input_file->text || !input_file->data.start)
    {
        print(ERROR, MSG_NONE, "SEQUENCES | Invalid file input");
        return false;
    }

    seq_pool_init();
    sequence_count_t total = input_file->data.total;
    sequences_t sequences = MALLOC(sequences, total);
    if (!sequences)
    {
        print(ERROR, MSG_NONE, "SEQUENCES | Failed to allocate memory for sequences");
        return false;
    }

    bool apply_filtering = filter_threshold > 0.0f;

    sequence_count_t sequence_count_current = 0;
    sequence_count_t filtered_count = 0;
    sequence_count_t skipped_count = 0;
    size_t total_sequence_length = 0;
    bool skip_long_sequences = false;
    bool asked_user_about_skipping = false;

#ifdef USE_CUDA
#include "args.h"
    const bool use_cuda = args_mode_cuda();
    sequence_length_t max_sequence_length = 0;
    sequence_offset_t* temp_offsets = NULL;
    quar_t* temp_lengths = NULL;
    if (use_cuda)
    {
        temp_offsets = MALLOC(temp_offsets, total);
        temp_lengths = MALLOC(temp_lengths, total);
        if (!temp_offsets || !temp_lengths)
        {
            print(ERROR, MSG_NONE, "SEQUENCES | Failed to allocate memory for CUDA arrays");
            free(temp_lengths);
            free(temp_offsets);
            free(sequences);
            return false;
        }
    }

#endif

    sequence_t current_sequence = { 0 };
    char* file_cursor = input_file->data.start;
    char* file_end = input_file->data.end;

    while (file_cursor < file_end && *file_cursor)
    {
        size_t next_sequence_length = file_sequence_next_length(input_file, file_cursor);

        if (!next_sequence_length)
        {
            file_sequence_next(input_file, &file_cursor);
            continue;
        }

        if (next_sequence_length > SEQUENCE_LENGTH_MAX)
        {
            if (!asked_user_about_skipping)
            {
                print(WARNING,
                      MSG_NONE,
                      "Found sequence longer than maximum allowed (%d letters)",
                      SEQUENCE_LENGTH_MAX);
                skip_long_sequences = print_yN("Skip sequences that are too long? [y/N]");
                asked_user_about_skipping = true;
            }

            if (skip_long_sequences)
            {
                file_sequence_next(input_file, &file_cursor);
                skipped_count++;
                continue;
            }

            else
            {
                print(ERROR,
                      MSG_NONE,
                      "SEQUENCES | Sequence too long: %zu letters (max: %d)",
                      next_sequence_length,
                      SEQUENCE_LENGTH_MAX);

                if (current_sequence.letters)
                {
                    free(current_sequence.letters);
                    current_sequence.letters = NULL;
                }

#ifdef USE_CUDA
                if (use_cuda)
                {
                    free(temp_lengths);
                    free(temp_offsets);
                }

#endif
                free(sequences);
                return false;
            }
        }

        if (next_sequence_length > current_sequence.length || !current_sequence.letters)
        {
            size_t new_capacity = next_sequence_length + 1;

            if (current_sequence.letters)
            {
                free(current_sequence.letters);
            }

            current_sequence.letters = MALLOC(current_sequence.letters, new_capacity);

            if (!current_sequence.letters)
            {
                print(ERROR, MSG_NONE, "SEQUENCES | Failed to allocate sequence buffer");
#ifdef USE_CUDA
                if (use_cuda)
                {
                    free(temp_lengths);
                    free(temp_offsets);
                }

#endif
                free(sequences);
                return false;
            }

            current_sequence.length = (sequence_length_t)next_sequence_length;
        }

        sequence_length_t sequence_length = (sequence_length_t)
            file_extract_sequence(input_file, &file_cursor, current_sequence.letters);

        bool should_include = true;

        if (apply_filtering && sequence_count_current > 0)
        {
            for (sequence_index_t j = 0; j < sequence_count_current; j++)
            {
                float similarity = similarity_pairwise(&current_sequence, &sequences[j]);

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
            sequence_init(&sequences[sequence_count_current],
                          current_sequence.letters,
                          sequence_length);
#ifdef USE_CUDA
            if (use_cuda)
            {
                temp_offsets[sequence_count_current] = (sequence_offset_t)total_sequence_length;
                temp_lengths[sequence_count_current] = (quar_t)sequence_length;
                if (max_sequence_length < sequence_length)
                {
                    max_sequence_length = sequence_length;
                }
            }
#endif
            total_sequence_length += sequence_length;
            sequence_count_current++;
        }

        sequence_count_t actual_count = sequence_count_current + filtered_count + skipped_count;
        int percentage = (int)(100 * actual_count / total);
        const char* progress_msg = apply_filtering ? "Filtering sequences" : "Loading sequences";
        print(PROGRESS, MSG_PERCENT(percentage), progress_msg);
    }

    free(current_sequence.letters);

    sequence_count_t sequence_count = sequence_count_current;
    if (UNLIKELY(sequence_count > SEQUENCE_COUNT_MAX))
    {
        print(ERROR, MSG_LOC(LAST), "Too many sequences: %u", sequence_count);
#ifdef USE_CUDA
        if (use_cuda)
        {
            free(temp_lengths);
            free(temp_offsets);
        }

#endif
        free(sequences);
        return false;
    }

    if (skipped_count > 0)
    {
        print(INFO, MSG_NONE, "Skipped %u sequences that were too long", skipped_count);
    }

    if (apply_filtering && filtered_count > 0 && filtered_count >= total / 4)
    {
        print(VERBOSE, MSG_NONE, "Reallocating memory to save %u sequence slots", filtered_count);
        sequences_t _sequences_new = REALLOC(sequences, sequence_count);
        if (_sequences_new)
        {
            sequences = _sequences_new;
        }

#ifdef USE_CUDA
        if (use_cuda)
        {
            sequence_offset_t* _offsets_new = REALLOC(temp_offsets, sequence_count);
            if (_offsets_new)
            {
                temp_offsets = _offsets_new;
            }

            quar_t* _lengths_new = REALLOC(temp_lengths, sequence_count);
            if (_lengths_new)
            {
                temp_lengths = _lengths_new;
            }
        }
#endif
    }

    if (apply_filtering)
    {
        print(DNA, MSG_NONE, "Loaded %u sequences (filtered %u)", sequence_count, filtered_count);
    }

    float sequence_average_length = (float)total_sequence_length / (float)sequence_count;

    print(INFO, MSG_NONE, "Average sequence length: %.2f", sequence_average_length);

    alignment_size_t alignment_count = (sequence_count * (sequence_count - 1)) / 2;

    g_sequence_dataset.sequences = sequences;
    g_sequence_dataset.sequence_count = sequence_count;
    g_sequence_dataset.alignment_count = alignment_count;

#ifdef USE_CUDA
    if (use_cuda)
    {
        g_sequence_dataset.flat_sequences = MALLOC(g_sequence_dataset.flat_sequences,
                                                   total_sequence_length);
        g_sequence_dataset.flat_offsets = MALLOC(g_sequence_dataset.flat_offsets, sequence_count);
        g_sequence_dataset.flat_lengths = MALLOC(g_sequence_dataset.flat_lengths, sequence_count);

        if (!g_sequence_dataset.flat_sequences || !g_sequence_dataset.flat_offsets ||
            !g_sequence_dataset.flat_lengths)
        {
            print(ERROR, MSG_NONE, "CUDA | Failed to allocate flattened arrays");
            free(temp_lengths);
            free(temp_offsets);
            free(sequences);
            return false;
        }

        memcpy(g_sequence_dataset.flat_offsets,
               temp_offsets,
               sequence_count * sizeof(sequence_offset_t));
        memcpy(g_sequence_dataset.flat_lengths, temp_lengths, sequence_count * sizeof(quar_t));

        size_t offset = 0;
        for (sequence_index_t i = 0; i < sequence_count; i++)
        {
            memcpy(g_sequence_dataset.flat_sequences + offset,
                   sequences[i].letters,
                   sequences[i].length);
            offset += sequences[i].length;
        }

        g_sequence_dataset.total_sequence_length = total_sequence_length;
        g_sequence_dataset.max_sequence_length = max_sequence_length;

        free(temp_lengths);
        free(temp_offsets);
    }

#endif

    return true;
}

void
sequences_get_pair(sequence_index_t first_sequence_index,
                   sequence_ptr_t* restrict first_sequence_out,
                   sequence_index_t second_sequence_index,
                   sequence_ptr_t* restrict second_sequence_out)
{
    *first_sequence_out = &g_sequence_dataset.sequences[first_sequence_index];
    *second_sequence_out = &g_sequence_dataset.sequences[second_sequence_index];

    prefetch((*first_sequence_out)->letters);
    prefetch((*second_sequence_out)->letters);
}

sequences_t
sequences_get(void)
{
    return g_sequence_dataset.sequences;
}

sequence_count_t
sequences_count(void)
{
    return g_sequence_dataset.sequence_count;
}

alignment_size_t
sequences_alignment_count(void)
{
    return g_sequence_dataset.alignment_count;
}

#ifdef USE_CUDA

char*
sequences_flattened(void)
{
    return g_sequence_dataset.flat_sequences;
}

sequence_offset_t*
sequences_offsets(void)
{
    return g_sequence_dataset.flat_offsets;
}

quar_t*
sequences_lengths(void)
{
    return g_sequence_dataset.flat_lengths;
}

size_t
sequences_total_length(void)
{
    return g_sequence_dataset.total_sequence_length;
}

sequence_length_t
sequences_max_length(void)
{
    return g_sequence_dataset.max_sequence_length;
}

#endif
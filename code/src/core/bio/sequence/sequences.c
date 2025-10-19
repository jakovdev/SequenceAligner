#include "core/bio/sequence/sequences.h"

#include <ctype.h>

#include "core/app/args.h"
#include "core/bio/score/matrices.h"
#include "core/bio/sequence/filtering.h"
#include "core/bio/types.h"
#include "core/io/files.h"
#include "system/arch.h"
#include "system/simd.h"
#include "util/benchmark.h"
#include "util/print.h"

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

#ifdef _MSC_VER
    atexit(seq_pool_free);
#endif
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
sequence_init(sequence_ptr_t pooled_sequence, const sequence_ptr_t temporary_sequence)
{
    pooled_sequence->length = temporary_sequence->length;

    pooled_sequence->letters = seq_pool_alloc(temporary_sequence->length + 1);
    if (pooled_sequence->letters)
    {
        memcpy(pooled_sequence->letters, temporary_sequence->letters, temporary_sequence->length);
        pooled_sequence->letters[temporary_sequence->length] = '\0';
    }
}

static bool
validate_sequence(const sequence_ptr_t sequence, SequenceType sequence_type)
{
    if (!sequence->letters || !sequence->length)
    {
        return false;
    }

    const char* valid_alphabet = NULL;
    size_t alphabet_size = 0;

    switch (sequence_type)
    {
        case SEQ_TYPE_AMINO:
            valid_alphabet = AMINO_ALPHABET;
            alphabet_size = AMINO_SIZE;
            break;
        case SEQ_TYPE_NUCLEOTIDE:
            valid_alphabet = NUCLEOTIDE_ALPHABET;
            alphabet_size = NUCLEOTIDE_SIZE;
            break;
        case SEQ_TYPE_INVALID:
        case SEQ_TYPE_COUNT:
        default:
            return false;
    }

    for (sequence_length_t i = 0; i < sequence->length; i++)
    {
        char c = (char)toupper(sequence->letters[i]);
        bool found = false;

        for (size_t j = 0; j < alphabet_size; j++)
        {
            if (c == valid_alphabet[j])
            {
                found = true;
                break;
            }
        }

        if (!found)
        {
            return false;
        }

        sequence->letters[i] = c;
    }

    return true;
}

bool
sequences_load_from_file(void)
{
    print_error_prefix("SEQUENCES");

    FileText input_file = { 0 };
    if (!file_text_open(&input_file, args_input()))
    {
        return false;
    }

    seq_pool_init();
    sequence_count_t total = file_sequence_total(&input_file);
    sequences_t sequences = MALLOC(sequences, total);
    if (!sequences)
    {
        print(ERROR, MSG_NONE, "Failed to allocate memory for sequences");
        file_text_close(&input_file);
        return false;
    }

    float filter_threshold = args_filter();
    bool apply_filtering = filter_threshold > 0.0f;

    sequence_count_t sequence_count_current = 0;
    sequence_count_t skipped_count = 0;
    size_t total_sequence_length = 0;
    bool skip_long_sequences = false;
    bool asked_user_about_skipping = false;
    bool skip_invalid_sequences = false;
    bool asked_user_about_invalid = false;
    sequence_count_t invalid_count = 0;
    SequenceType sequence_type = args_sequence_type();

    sequence_t sequence_current = { 0 };

    print(PROGRESS, MSG_PERCENT(0), "Loading sequences");
    print_error_prefix("FILE");

    for (sequence_count_t seq_index = 0; seq_index < total; seq_index++)
    {
        size_t next_sequence_length = file_sequence_next_length(&input_file);

        if (!next_sequence_length)
        {
            file_sequence_next(&input_file);
            continue;
        }

        if (next_sequence_length > SEQUENCE_LENGTH_MAX)
        {
            if (!asked_user_about_skipping)
            {
                bench_io_end();
                print(WARNING, MSG_NONE, "Found very large sequence (>%d)", SEQUENCE_LENGTH_MAX);
                skip_long_sequences = print_yN("Skip sequences that are too long? [y/N]");
                asked_user_about_skipping = true;
                bench_io_start();
            }

            if (skip_long_sequences)
            {
                file_sequence_next(&input_file);
                skipped_count++;
                continue;
            }

            else
            {
                print_error_prefix("SEQUENCES");
                print(ERROR, MSG_NONE, "Sequence too long: %zu letters", next_sequence_length);
                goto cleanup_sequence_current;
            }
        }

        if (next_sequence_length > sequence_current.length || !sequence_current.letters)
        {
            size_t new_capacity = next_sequence_length + 1;

            if (sequence_current.letters)
            {
                free(sequence_current.letters);
            }

            sequence_current.letters = MALLOC(sequence_current.letters, new_capacity);

            if (!sequence_current.letters)
            {
                print_error_prefix("SEQUENCES");
                print(ERROR, MSG_NONE, "Failed to allocate sequence buffer");
                goto cleanup_sequences;
            }
        }

        sequence_current.length = (sequence_length_t)next_sequence_length;

        file_extract_sequence(&input_file, sequence_current.letters);

        if (!validate_sequence(&sequence_current, sequence_type))
        {
            if (!asked_user_about_invalid)
            {
                bench_io_end();
                print(WARNING, MSG_LOC(FIRST), "Found sequence with invalid letters");
                print(WARNING, MSG_LOC(LAST), "Sequence: %s is invalid", sequence_current.letters);
                skip_invalid_sequences = print_yN("Skip sequences with invalid letters? [y/N]");
                asked_user_about_invalid = true;
                bench_io_start();
            }

            if (skip_invalid_sequences)
            {
                invalid_count++;
                continue;
            }

            else
            {
                print_error_prefix("SEQUENCES");
                print(ERROR, MSG_NONE, "Found sequence with invalid letters");
                goto cleanup_sequence_current;
            }
        }

        sequence_init(&sequences[sequence_count_current], &sequence_current);
        total_sequence_length += sequence_current.length;
        sequence_count_current++;

        sequence_count_t actual_count = sequence_count_current + skipped_count + invalid_count;
        int percentage = (int)(100 * actual_count / total);
        print(PROGRESS, MSG_PERCENT(percentage), "Loading sequences");
    }

    free(sequence_current.letters);

    print_error_prefix("SEQUENCES");

    sequence_count_t sequence_count = sequence_count_current;
    if (UNLIKELY(sequence_count > SEQUENCE_COUNT_MAX))
    {
        print(ERROR, MSG_NONE, "Too many sequences: %u", sequence_count);
        goto cleanup_sequences;
    }

    if (sequence_count < 2)
    {
        print(ERROR, MSG_NONE, "At least 2 sequences are required (found: %u)", sequence_count);
        goto cleanup_sequences;
    }

    if (skipped_count > 0)
    {
        print(INFO, MSG_NONE, "Skipped %u sequences that were too long", skipped_count);
    }

    if (invalid_count > 0)
    {
        print(INFO, MSG_NONE, "Skipped %u sequences with invalid letters", invalid_count);
    }

    sequence_count_t filtered_count = 0;
    if (!apply_filtering)
    {
        goto skip_filtering;
    }

    bench_io_end();
    bench_filter_start();
    print_error_prefix("FILTERING");

    bool* keep_flags = MALLOC(keep_flags, sequence_count);
    if (!keep_flags)
    {
        print(ERROR, MSG_NONE, "Failed to allocate memory for filtering flags");
        goto cleanup_sequences;
    }

    if (!filter_sequences(sequences, sequence_count, filter_threshold, keep_flags, &filtered_count))
    {
        free(keep_flags);
        goto cleanup_sequences;
    }

    sequence_count_t write_index = 0;
    total_sequence_length = 0;

    for (sequence_count_t read_index = 0; read_index < sequence_count; read_index++)
    {
        if (!keep_flags[read_index])
        {
            continue;
        }

        if (write_index != read_index)
        {
            sequences[write_index] = sequences[read_index];
        }

        total_sequence_length += sequences[write_index].length;
        write_index++;
    }

    sequence_count = write_index;
    free(keep_flags);

    bench_filter_end();
    bench_print_filter(filtered_count);
    bench_io_start();

    if (sequence_count < 2)
    {
        print(ERROR, MSG_NONE, "Filtering removed too many sequences");
        goto cleanup_sequences;
    }

    if (filtered_count > 0 && filtered_count >= total / 4)
    {
        print(VERBOSE, MSG_NONE, "Reallocating memory to save %u sequence slots", filtered_count);
        sequences_t _sequences_new = REALLOC(sequences, sequence_count);
        if (_sequences_new)
        {
            sequences = _sequences_new;
        }
    }

    print(DNA, MSG_NONE, "Loaded %u sequences (filtered %u)", sequence_count, filtered_count);
    goto already_printed;

skip_filtering:
    print(DNA, MSG_NONE, "Loaded %u sequences", sequence_count);

already_printed:;
    float sequence_average_length = (float)total_sequence_length / (float)sequence_count;

    print(INFO, MSG_NONE, "Average sequence length: %.2f", sequence_average_length);

    g_sequence_dataset.sequences = sequences;
    g_sequence_dataset.sequence_count = sequence_count;
    g_sequence_dataset.alignment_count = ((size_t)sequence_count * (sequence_count - 1)) / 2;

#ifdef _MSC_VER
    atexit(sequences_free);
#endif

#ifdef USE_CUDA
    if (!args_mode_cuda())
    {
        goto skip_cuda;
    }

    size_t total_length = 0;
    sequence_length_t cuda_max_length = 0;

    for (sequence_index_t i = 0; i < sequence_count; i++)
    {
        total_length += sequences[i].length;
        if (cuda_max_length < sequences[i].length)
        {
            cuda_max_length = sequences[i].length;
        }
    }

    g_sequence_dataset.flat_sequences = MALLOC(g_sequence_dataset.flat_sequences, total_length);
    g_sequence_dataset.flat_offsets = MALLOC(g_sequence_dataset.flat_offsets, sequence_count);
    g_sequence_dataset.flat_lengths = MALLOC(g_sequence_dataset.flat_lengths, sequence_count);

    if (!g_sequence_dataset.flat_sequences || !g_sequence_dataset.flat_offsets ||
        !g_sequence_dataset.flat_lengths)
    {
        print_error_prefix("CUDA");
        print(ERROR, MSG_NONE, "Failed to allocate flattened arrays");
        goto cleanup_sequences;
    }

    sequence_offset_t offset = 0;
    for (sequence_index_t i = 0; i < sequence_count; i++)
    {
        g_sequence_dataset.flat_offsets[i] = offset;
        g_sequence_dataset.flat_lengths[i] = (quar_t)sequences[i].length;
        memcpy(g_sequence_dataset.flat_sequences + offset,
               sequences[i].letters,
               sequences[i].length);
        offset += (sequence_offset_t)sequences[i].length;
    }

    g_sequence_dataset.total_sequence_length = total_length;
    g_sequence_dataset.max_sequence_length = cuda_max_length;

skip_cuda:
#endif

    file_text_close(&input_file);
    return true;

cleanup_sequence_current:
    if (sequence_current.letters)
    {
        free(sequence_current.letters);
    }

cleanup_sequences:
    free(sequences);
    file_text_close(&input_file);
    return false;
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

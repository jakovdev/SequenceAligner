#include "bio/sequence/sequences.h"

#include <ctype.h>
#include <string.h>
#include <stdlib.h>

#include "app/args.h"
#include "bio/score/matrices.h"
#include "bio/sequence/filtering.h"
#include "bio/types.h"
#include "io/files.h"
#include "system/compiler.h"
#include "system/memory.h"
#include "system/os.h"
#include "util/benchmark.h"
#include "util/print.h"

struct SeqMemBlock {
#define SB_SIZE (4 * MiB)
	char *block;
	size_t used;
	size_t capacity;
	struct SeqMemBlock *next;
};

static struct {
	struct SeqMemBlock *head;
	struct SeqMemBlock *current;
	size_t total_bytes;
	size_t total_allocated;
	size_t block_count;
} g_pool = { 0 };

static sequence_t *g_seqs;
static u64 g_align_n;
#ifdef USE_CUDA
static u64 g_seq_len_sum;
static u32 g_seq_len_max;
#endif
static u32 g_seq_n;

static void seq_pool_free(void)
{
	if (!g_pool.head)
		return;

	struct SeqMemBlock *curr = g_pool.head;
	while (curr) {
		struct SeqMemBlock *next = curr->next;
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

static void seq_pool_init(void)
{
	if (g_pool.head)
		return;

	g_pool.head = MALLOC(g_pool.head, 1);
	if (!g_pool.head)
		return;

	g_pool.head->block = alloc_huge_page(SB_SIZE);
	if (!g_pool.head->block) {
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
	atexit(seq_pool_free);
}

static char *seq_pool_alloc(size_t size)
{
	if (!g_pool.head) {
		seq_pool_init();
		if (!g_pool.head)
			return NULL;
	}

	size = (size + 7) & ~((size_t)7);

	if (g_pool.current->used + size > g_pool.current->capacity) {
		size_t new_block_size = SB_SIZE;
		if (size > new_block_size)
			new_block_size = size;

		struct SeqMemBlock *new_block = MALLOC(new_block, 1);
		if (!new_block)
			return NULL;

		new_block->block = MALLOC(new_block->block, new_block_size);
		if (!new_block->block) {
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

	char *result = g_pool.current->block + g_pool.current->used;
	g_pool.current->used += size;
	g_pool.total_allocated += size;

	return result;
}

static void sequences_free(void)
{
	if (g_seqs) {
		free(g_seqs);
		g_seqs = NULL;
	}
	g_align_n = 0;
#ifdef USE_CUDA
	g_seq_len_sum = 0;
	g_seq_len_max = 0;
#endif
	g_seq_n = 0;
}

static void sequence_init(sequence_t *const restrict pooled,
			  const sequence_t *const restrict temp)
{
	pooled->length = temp->length;

	pooled->letters = seq_pool_alloc(temp->length + 1);
	if (pooled->letters) {
		memcpy(pooled->letters, temp->letters, temp->length);
		pooled->letters[temp->length] = '\0';
	}
}

static bool validate_sequence(sequence_ptr_t sequence)
{
	if (!sequence->letters || !sequence->length)
		return false;

	const char *valid_alphabet = NULL;
	int alphabet_size = 0;

	switch (args_sequence_type()) {
	case SEQ_TYPE_AMINO:
		valid_alphabet = AMINO_ALPHABET;
		alphabet_size = AMINO_SIZE;
		break;
	case SEQ_TYPE_NUCLEO:
		valid_alphabet = NUCLEO_ALPHABET;
		alphabet_size = NUCLEO_SIZE;
		break;
	/* EXPANDABLE: enum SequenceType */
	case SEQ_TYPE_INVALID:
	case SEQ_TYPE_COUNT:
	default:
		UNREACHABLE();
	}

	for (u64 i = 0; i < sequence->length; i++) {
		char c = (char)toupper(sequence->letters[i]);
		bool found = false;

		for (int j = 0; j < alphabet_size; j++) {
			if (c == valid_alphabet[j]) {
				found = true;
				break;
			}
		}

		if (!found)
			return false;

		sequence->letters[i] = c;
	}

	return true;
}

static bool seq_len_valid(u64 length)
{
	const s32 gap_pen = args_gap_pen();
	if (!gap_pen)
		return length <= SEQUENCE_LENGTH_MAX;

	const u64 limit = SEQUENCE_LENGTH_MAX / (u64)gap_pen;
	if (length < limit)
		return true;

	if (gap_pen > 1000)
		print(M_NONE, WARNING "Very suspicious gap penalty (>1000)");
	else if (gap_pen > 100)
		print(M_NONE, WARNING "Unusually high gap penalty (>100)");

	print(M_NONE,
	      ERR "Sequence length " Pu64 " exceeds limits for gap penalty %d",
	      length, gap_pen);
	return false;
}

bool sequences_load_from_file(void)
{
	struct FileText input_file = { 0 };
	if (!file_text_open(&input_file, args_input()))
		return false;

	print_error_context("SEQUENCES");

	u32 total = file_sequence_total(&input_file);
	sequence_t *seqs = MALLOC(seqs, total);
	if (!seqs) {
		print(M_NONE, ERR "Failed to allocate memory for sequences");
		file_text_close(&input_file);
		return false;
	}

	double filter_threshold = args_filter();
	bool apply_filtering = filter_threshold > 0.0;

	u64 seqs_len_sum = 0;
#ifdef USE_CUDA
	u32 seq_len_max = 0;
#endif
	u32 seq_n_curr = 0;
	u32 seq_n_skip = 0;
	u32 seq_n_invalid = 0;
	bool skip_long = false;
	bool ask_long = false;
	bool skip_invalid = false;
	bool ask_invalid = false;

	sequence_t seq_curr = { 0 };

	print(M_PERCENT(0) "Loading sequences");
	bench_io_start();

	for (u32 seq_index = 0; seq_index < total; seq_index++) {
		u64 seq_len_next = file_sequence_next_length(&input_file);

		if (!seq_len_next) {
			file_sequence_next(&input_file);
			continue;
		}

		if (!seq_len_valid(seq_len_next)) {
			if (!ask_long) {
				bench_io_end();
				print(M_NONE,
				      WARNING
				      "Overflow from large sequence length: " Pu64,
				      seq_len_next);
				skip_long = print_yN("Skip long sequences?");
				ask_long = true;
				bench_io_start();
			}

			if (args_force() || skip_long) {
				file_sequence_next(&input_file);
				seq_n_skip++;
				continue;
			} else {
				print(M_NONE,
				      ERR "Sequence #" Pu32 " is too long",
				      seq_index + 1);
				goto cleanup_seq_curr_seqs;
			}
		}

		if (seq_len_next > seq_curr.length || !seq_curr.letters) {
			u64 count = seq_len_next + 1;

			if (seq_curr.letters)
				free(seq_curr.letters);

			seq_curr.letters = MALLOC(seq_curr.letters, count);

			if (!seq_curr.letters) {
				print(M_NONE,
				      ERR "Failed to allocate sequence");
				goto cleanup_seqs;
			}
		}

		seq_curr.length = seq_len_next;

		file_extract_entry(&input_file, seq_curr.letters);

		if (!validate_sequence(&seq_curr)) {
			if (!ask_invalid) {
				bench_io_end();
				print(M_LOC(FIRST), WARNING
				      "Found sequence with invalid letters");
				print(M_LOC(LAST),
				      WARNING "Sequence #" Pu32 " is invalid",
				      seq_index + 1);
				skip_invalid =
					print_yN("Skip invalid sequences?");
				ask_invalid = true;
				bench_io_start();
			}

			if (args_force() || skip_invalid) {
				seq_n_invalid++;
				continue;
			} else {
				print(M_NONE, ERR
				      "Found sequence with invalid letters");
				goto cleanup_seq_curr_seqs;
			}
		}

		sequence_init(&seqs[seq_n_curr], &seq_curr);
		if (!seqs[seq_n_curr].letters) {
			print(M_NONE, ERR "Failed to allocate sequence");
			goto cleanup_seq_curr_seqs;
		}

		seqs_len_sum += seq_curr.length;
#ifdef USE_CUDA
		if (args_mode_cuda() && seq_curr.length > seq_len_max)
			seq_len_max = (u32)seq_curr.length;
#endif
		seq_n_curr++;

		u32 seq_n_actual = seq_n_curr + seq_n_skip + seq_n_invalid;
		print(M_PROPORT(seq_n_actual / total) "Loading sequences");
	}

	bench_io_end();
	print(M_PERCENT(100) "Loading sequences");
	free(seq_curr.letters);

	u32 seq_n = seq_n_curr;
	if (seq_n < 2) {
		print(M_NONE,
		      ERR "At least 2 sequences are required, found " Pu32,
		      seq_n);
		goto cleanup_seqs;
	}

	if (seq_n_skip > 0)
		print(M_NONE,
		      INFO "Skipped " Pu32 " sequences that were too long",
		      seq_n_skip);

	if (seq_n_invalid > 0)
		print(M_NONE,
		      INFO "Skipped " Pu32 " sequences with invalid letters",
		      seq_n_invalid);

	u32 n_seqs_filtered = 0;
	if (!apply_filtering)
		goto skip_filtering;

	bench_filter_start();
	print_error_context("FILTERING");
	bool *keep_flags = MALLOC(keep_flags, seq_n);
	if (!keep_flags) {
		print(M_NONE,
		      ERR "Failed to allocate memory for filtering flags");
		goto cleanup_seqs;
	}

	if (!filter_sequences(seqs, seq_n, filter_threshold, keep_flags,
			      &n_seqs_filtered)) {
		free(keep_flags);
		goto cleanup_seqs;
	}

	u32 write_index = 0;
	seqs_len_sum = 0;

	for (u32 read_index = 0; read_index < seq_n; read_index++) {
		if (!keep_flags[read_index])
			continue;

		if (write_index != read_index)
			seqs[write_index] = seqs[read_index];

		seqs_len_sum += seqs[write_index].length;
#ifdef USE_CUDA
		if (args_mode_cuda() && seqs[write_index].length > seq_len_max)
			seq_len_max = (u32)seqs[write_index].length;
#endif
		write_index++;
	}

	seq_n = write_index;
	free(keep_flags);
	bench_filter_end();
	bench_filter_print(n_seqs_filtered);

	bench_io_start();
	if (seq_n < 2) {
		print(M_NONE, ERR "Filtering removed too many sequences");
		goto cleanup_seqs;
	}

	if (n_seqs_filtered > 0 && n_seqs_filtered >= total / 4) {
		print(M_NONE,
		      VERBOSE "Reallocating memory to save " Pu32
			      " sequence slots",
		      n_seqs_filtered);
		sequence_t *_sequences_new = REALLOC(seqs, seq_n);
		if (_sequences_new)
			seqs = _sequences_new;
	}

	print(M_NONE, INFO "Loaded " Pu32 " sequences (filtered " Pu32 ")",
	      seq_n, n_seqs_filtered);
	goto already_printed;

skip_filtering:
	print(M_NONE, INFO "Loaded " Pu32 " sequences", seq_n);

already_printed:
	double seq_len_avg = (double)seqs_len_sum / (double)seq_n;
	print(M_NONE, INFO "Average sequence length: %.2f", seq_len_avg);

	g_seqs = seqs;
	g_seq_n = seq_n;
	g_align_n = ((u64)seq_n * (seq_n - 1)) / 2;
#ifdef USE_CUDA
	g_seq_len_sum = seqs_len_sum;
	g_seq_len_max = seq_len_max;
#endif
	atexit(sequences_free);
	file_text_close(&input_file);
	return true;

cleanup_seq_curr_seqs:
	if (seq_curr.letters)
		free(seq_curr.letters);

cleanup_seqs:
	free(seqs);
	file_text_close(&input_file);
	return false;
}

sequence_t *sequence(u32 index)
{
	return &g_seqs[index];
}

sequence_t *sequences(void)
{
	return g_seqs;
}

u32 sequences_count(void)
{
	return g_seq_n;
}

u64 sequences_alignment_count(void)
{
	return g_align_n;
}

#ifdef USE_CUDA

u64 sequences_length_sum(void)
{
	return g_seq_len_sum;
}

u32 sequences_length_max(void)
{
	return g_seq_len_max;
}

#endif

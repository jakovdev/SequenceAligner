#include "core/bio/sequence/sequences.h"

#include <ctype.h>
#include <string.h>
#include <stdlib.h>

#include "core/app/args.h"
#include "core/bio/score/matrices.h"
#include "core/bio/sequence/filtering.h"
#include "core/bio/types.h"
#include "core/io/files.h"
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

static struct {
	sequence_t *seqs;
	u64 align_n;
#ifdef USE_CUDA
	char *seqs_flat;
	size_t seq_len_total;
	u32 *offs_flat;
	u32 *lens_flat;
	u32 seq_len_max;
#endif
	u32 seq_n;
} g_dataset = { 0 };

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
	if (g_dataset.seqs) {
		free(g_dataset.seqs);
		g_dataset.seqs = NULL;
	}

	g_dataset.seq_n = 0;
	g_dataset.align_n = 0;

#ifdef USE_CUDA
	if (g_dataset.seqs_flat) {
		free(g_dataset.seqs_flat);
		g_dataset.seqs_flat = NULL;
	}

	if (g_dataset.offs_flat) {
		free(g_dataset.offs_flat);
		g_dataset.offs_flat = NULL;
	}

	if (g_dataset.lens_flat) {
		free(g_dataset.lens_flat);
		g_dataset.lens_flat = NULL;
	}

	g_dataset.seq_len_total = 0;
	g_dataset.seq_len_max = 0;
#endif
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

static bool validate_sequence(sequence_ptr_t sequence,
			      enum SequenceType sequence_type)
{
	if (!sequence->letters || !sequence->length)
		return false;

	const char *valid_alphabet = NULL;
	int alphabet_size = 0;

	switch (sequence_type) {
	case SEQ_TYPE_AMINO:
		valid_alphabet = AMINO_ALPHABET;
		alphabet_size = AMINO_SIZE;
		break;
	case SEQ_TYPE_NUCLEO:
		valid_alphabet = NUCLEOTIDE_ALPHABET;
		alphabet_size = NUCLEOTIDE_SIZE;
		break;
	case SEQ_TYPE_INVALID:
	case SEQ_TYPE_COUNT:
	default:
		return false;
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
	const s32 gap_pen = args_gap_penalty();
	const bool linear = gap_pen > 0;
	if (!linear)
		return length <= SEQUENCE_LENGTH_MAX;

	u64 limit = SEQUENCE_LENGTH_MAX / (u64)gap_pen;
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

	u64 seq_len_total = 0;
	u32 seq_n_curr = 0;
	u32 seq_n_skip = 0;
	u32 seq_n_invalid = 0;
	bool skip_long = false;
	bool ask_long = false;
	bool skip_invalid = false;
	bool ask_invalid = false;
	enum SequenceType sequence_type = args_sequence_type();

	sequence_t seq_curr = { 0 };

	print(M_PERCENT(0) "Loading sequences");

	for (u32 seq_index = 0; seq_index < total; seq_index++) {
		u64 seq_len_next = file_sequence_next_length(&input_file);

		if (!seq_len_next) {
			file_sequence_next(&input_file);
			continue;
		}

		if (!seq_len_valid(seq_len_next)) {
			if (!ask_long) {
				print(M_NONE,
				      WARNING
				      "Overflow from large sequence length: " Pu64,
				      seq_len_next);
				bench_io_end();
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

		if (!validate_sequence(&seq_curr, sequence_type)) {
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

		seq_len_total += seq_curr.length;
		seq_n_curr++;

		u32 seq_n_actual = seq_n_curr + seq_n_skip + seq_n_invalid;
		print(M_PROPORT(seq_n_actual / total) "Loading sequences");
	}

	print(M_PERCENT(100) "Loading sequences");
	free(seq_curr.letters);

	u32 seq_n = seq_n_curr;
	if (UNLIKELY(seq_n > SEQUENCE_COUNT_MAX)) {
		print(M_NONE, ERR "Too many sequences: " Pu32, seq_n);
		goto cleanup_seqs;
	}

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

	bench_io_end();
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
	seq_len_total = 0;

	for (u32 read_index = 0; read_index < seq_n; read_index++) {
		if (!keep_flags[read_index])
			continue;

		if (write_index != read_index)
			seqs[write_index] = seqs[read_index];

		seq_len_total += seqs[write_index].length;
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
	double seq_len_avg = (double)seq_len_total / (double)seq_n;
	print(M_NONE, INFO "Average sequence length: %.2f", seq_len_avg);

	g_dataset.seqs = seqs;
	g_dataset.seq_n = seq_n;
	g_dataset.align_n = ((u64)seq_n * (seq_n - 1)) / 2;
	atexit(sequences_free);

#ifdef USE_CUDA
	if (!args_mode_cuda())
		goto skip_cuda;

	u32 cuda_max_length = 0;

	for (u32 i = 0; i < seq_n; i++) {
		if (cuda_max_length < seqs[i].length)
			cuda_max_length = (u32)seqs[i].length;
	}

	g_dataset.seqs_flat = MALLOC(g_dataset.seqs_flat, seq_len_total);
	g_dataset.offs_flat = MALLOC(g_dataset.offs_flat, seq_n);
	g_dataset.lens_flat = MALLOC(g_dataset.lens_flat, seq_n);

	if (!g_dataset.seqs_flat || !g_dataset.offs_flat ||
	    !g_dataset.lens_flat) {
		print_error_context("CUDA");
		print(M_NONE, ERR "Failed to allocate flattened arrays");
		goto cleanup_seqs;
	}

	u32 offset = 0;
	for (u32 i = 0; i < seq_n; i++) {
		g_dataset.offs_flat[i] = offset;
		g_dataset.lens_flat[i] = (u32)seqs[i].length;
		memcpy(g_dataset.seqs_flat + offset, seqs[i].letters,
		       seqs[i].length);
		offset += (u32)seqs[i].length;
	}

	g_dataset.seq_len_total = seq_len_total;
	g_dataset.seq_len_max = cuda_max_length;

skip_cuda:
#endif

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

sequence_t *sequence_get(u32 index)
{
	return &g_dataset.seqs[index];
}

sequence_t *sequences_get(void)
{
	return g_dataset.seqs;
}

u32 sequences_count(void)
{
	return g_dataset.seq_n;
}

u64 sequences_alignment_count(void)
{
	return g_dataset.align_n;
}

#ifdef USE_CUDA

char *sequences_flattened(void)
{
	return g_dataset.seqs_flat;
}

u32 *sequences_offsets(void)
{
	return g_dataset.offs_flat;
}

u32 *sequences_lengths(void)
{
	return g_dataset.lens_flat;
}

u64 sequences_length_total(void)
{
	return g_dataset.seq_len_total;
}

u32 sequences_length_max(void)
{
	return g_dataset.seq_len_max;
}

#endif

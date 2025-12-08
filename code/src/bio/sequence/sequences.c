#include "bio/sequence/sequences.h"

#include <ctype.h>
#include <string.h>
#include <stdlib.h>

#include "bio/score/matrices.h"
#include "bio/sequence/filtering.h"
#include "interface/seqalign_cuda.h"
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
static s64 g_alignments;
#ifdef USE_CUDA
static s64 g_seq_len_sum;
#endif
static s32 g_seq_len_max;
static s32 g_seq_n;

static void seq_pool_free(void)
{
	struct SeqMemBlock *curr = g_pool.head;
	while (curr) {
		struct SeqMemBlock *next = curr->next;
		free_aligned(curr->block);
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

static char *seq_pool_alloc(size_t size)
{
	if (!size)
		unreachable();

	if (!g_pool.head) {
		MALLOCA(g_pool.head, 1);
		if unlikely (!g_pool.head)
			return NULL;

		MALLOCA_CL(g_pool.head->block, SB_SIZE);
		if unlikely (!g_pool.head->block) {
			free(g_pool.head);
			g_pool.head = NULL;
			return NULL;
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

	size = (size + 7) & ~((size_t)7);

	if (g_pool.current->used + size > g_pool.current->capacity) {
		size_t new_block_size = SB_SIZE;
		if (size > new_block_size)
			new_block_size = size;

		struct SeqMemBlock *MALLOCA(new_block, 1);
		if unlikely (!new_block)
			return NULL;

		MALLOCA_CL(new_block->block, new_block_size);
		if unlikely (!new_block->block) {
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
	free_aligned(g_seqs);
	g_seqs = NULL;
	g_alignments = 0;
#ifdef USE_CUDA
	g_seq_len_sum = 0;
#endif
	g_seq_len_max = 0;
	g_seq_n = 0;
}

static bool sequence_init(sequence_t *const restrict pooled,
			  const sequence_t *const restrict temp)
{
	if (!pooled || SEQ_INVALID(temp))
		unreachable();

	pooled->length = temp->length;
	pooled->letters = seq_pool_alloc((size_t)temp->length + 1);
	if unlikely (!pooled->letters)
		return false;

	memcpy(pooled->letters, temp->letters, (size_t)temp->length);
	pooled->letters[temp->length] = '\0';
	return true;
}

static bool validate_sequence(sequence_ptr_t seq)
{
	if (SEQ_INVALID(seq))
		unreachable();

	const char *valid_alphabet = NULL;
	int alphabet_size = 0;

	switch (arg_sequence_type()) {
	case SEQ_TYPE_AMINO:
		valid_alphabet = AMINO_ALPHABET;
		alphabet_size = AMINO_SIZE;
		break;
	case SEQ_TYPE_NUCLEO:
		valid_alphabet = NUCLEO_ALPHABET;
		alphabet_size = NUCLEO_SIZE;
		break;
	case SEQ_TYPE_INVALID:
	case SEQ_TYPE_COUNT:
	default: /* NOTE: EXPANDABLE enum SequenceType */
		unreachable();
	}

	for (s32 i = 0; i < seq->length; i++) {
		char c = (char)toupper(seq->letters[i]);
		bool found = false;

		for (int j = 0; j < alphabet_size; j++) {
			if (c == valid_alphabet[j]) {
				found = true;
				break;
			}
		}

		if (!found)
			return false;

		seq->letters[i] = c;
	}

	return true;
}

static bool seq_len_valid(size_t len)
{
	if (!len)
		unreachable();

	const s32 gap_pen = -(arg_gap_pen());
	if (!gap_pen)
		return len <= SEQ_LEN_MAX;

	const size_t limit = SEQ_LEN_MAX / (size_t)gap_pen;
	if (len < limit)
		return true;

	if (gap_pen > 1000)
		pwarn("Very suspicious gap penalty (>1000)");
	else if (gap_pen > 100)
		pwarn("Unusually high gap penalty (>100)");

	perr("Sequence length %zu exceeds limits for gap penalty " Ps32, len,
	     gap_pen);
	return false;
}

bool sequences_load_from_file(void)
{
	struct FileText input_file = { 0 };
	if (!file_text_open(&input_file, arg_input()))
		return false;

	s32 total = file_sequence_total(&input_file);
	sequence_t *MALLOCA_CL(seqs, (size_t)total);
	if unlikely (!seqs) {
		perr("Out of memory allocating sequences");
		file_text_close(&input_file);
		return false;
	}

	s64 seq_len_sum = 0;
	s32 seq_len_max = 0;
	s32 seq_n_curr = 0;
	s32 seq_n_skip = 0;
	s32 seq_n_invalid = 0;
	bool skip_long = false;
	bool ask_long = false;
	bool skip_invalid = false;
	bool ask_invalid = false;

	sequence_t seq_curr = { 0 };

	bench_io_start();

	for (s32 seq_index = 0; seq_index < total; seq_index++) {
		const size_t seq_len = file_sequence_next_length(&input_file);
		if unlikely (!seq_len) {
			pwarn("Unexpected empty sequence #" Ps32
			      ", possible file corruption",
			      seq_index + 1);
			if (!print_yN("Continue loading sequences?"))
				goto cleanup_seq_curr_seqs;
			if unlikely (!file_sequence_next(&input_file)) {
				perr("Unexpected end of file, possible file corruption");
				goto cleanup_seq_curr_seqs;
			}
			continue;
		}

		if (!seq_len_valid(seq_len)) {
			if (!ask_long) {
				bench_io_end();
				pwarn("Overflow from large sequence length: %zu",
				      seq_len);
				skip_long = print_yN("Skip long sequences?");
				ask_long = true;
				bench_io_start();
			}

			if (skip_long) {
				if unlikely (!file_sequence_next(&input_file)) {
					perr("Unexpected end of file, possible file corruption");
					goto cleanup_seq_curr_seqs;
				}
				seq_n_skip++;
				continue;
			} else {
				perr("Sequence #" Ps32 " is too long",
				     seq_index + 1);
				goto cleanup_seq_curr_seqs;
			}
		}

		const s32 seq_len_safe = (s32)seq_len;
		if (seq_len_safe > seq_curr.length || !seq_curr.letters) {
			if (seq_curr.letters)
				free(seq_curr.letters);

			MALLOCA(seq_curr.letters, (size_t)(seq_len_safe + 1));
			if unlikely (!seq_curr.letters) {
				perr("Out of memory allocating sequence letters");
				goto cleanup_seqs;
			}
		}

		seq_curr.length = seq_len_safe;
		size_t len = file_extract_entry(&input_file, seq_curr.letters);
		if unlikely (!len || len != (size_t)seq_curr.length) {
			perr("Failed to extract sequence #" Ps32
			     ", possible file corruption",
			     seq_index + 1);
			goto cleanup_seq_curr_seqs;
		}

		if (!validate_sequence(&seq_curr)) {
			if (!ask_invalid) {
				bench_io_end();
				pwarn("Found sequence with invalid letters");
				pwarnl("Sequence #" Ps32 " is invalid",
				       seq_index + 1);
				skip_invalid =
					print_yN("Skip invalid sequences?");
				ask_invalid = true;
				bench_io_start();
			}

			if (skip_invalid) {
				seq_n_invalid++;
				continue;
			} else {
				perr("Found sequence with invalid letters");
				goto cleanup_seq_curr_seqs;
			}
		}

		if unlikely (!sequence_init(&seqs[seq_n_curr], &seq_curr)) {
			perr("Out of memory allocating into sequence pool");
			goto cleanup_seq_curr_seqs;
		}

		seq_len_sum += seq_curr.length;
		if (seq_curr.length > seq_len_max)
			seq_len_max = seq_curr.length;

		seq_n_curr++;
	}

	bench_io_end();
	free(seq_curr.letters);

	if (seq_n_skip > 0)
		pinfo("Skipped " Ps32 " sequences that were too long",
		      seq_n_skip);

	if (seq_n_invalid > 0)
		pinfo("Skipped " Ps32 " sequences with invalid letters",
		      seq_n_invalid);

	s32 seq_n = seq_n_curr;
	if (seq_n < SEQ_N_MIN) {
		perr("Not enough valid sequences loaded: " Ps32, seq_n);
		goto cleanup_seqs;
	}

	s32 seq_n_filter = 0;
	if (!arg_mode_filter())
		goto skip_filtering;

	bench_filter_start();
	bool *MALLOCA_CL(kept, (size_t)seq_n);
	if unlikely (!kept) {
		perr("Out of memory allocating filtering array");
		goto cleanup_seqs;
	}

	if unlikely (!filter_seqs(seqs, kept, seq_n, &seq_n_filter)) {
		free_aligned(kept);
		goto cleanup_seqs;
	}

	s32 write_index = 0;
	seq_len_sum = 0;

	for (s32 read_index = 0; read_index < seq_n; read_index++) {
		if (!kept[read_index])
			continue;

		if (write_index != read_index)
			seqs[write_index] = seqs[read_index];

		seq_len_sum += seqs[write_index].length;
		if (seqs[write_index].length > seq_len_max)
			seq_len_max = seqs[write_index].length;

		write_index++;
	}

	seq_n = write_index;
	free_aligned(kept);
	bench_filter_end();
	bench_filter_print();

	if (seq_n < 2) {
		perr("Filtering removed too many sequences (" Ps32 " remain)",
		     seq_n);
		goto cleanup_seqs;
	}

	if (seq_len_sum < SEQ_LEN_SUM_MIN) {
		perr("Not enough total sequence length after filtering: " Ps64,
		     seq_len_sum);
		goto cleanup_seqs;
	}

	if (seq_len_max > SEQ_LEN_MAX) {
		perr("Maximum sequence length after filtering exceeds limit: " Ps32,
		     seq_len_max);
		goto cleanup_seqs;
	}

	if (seq_n_filter > 0 && seq_n_filter >= total / 4) {
		pverb("Reallocating sequences to save memory");
		REALLOCA(seqs, (size_t)seq_n);
		else pverb("Failed to reallocate, continuing without");
	}

	pinfo("Loaded " Ps32 " sequences (filtered " Ps32 ")", seq_n,
	      seq_n_filter);

skip_filtering:
	pinfo("Average sequence length: %.2f",
	      (double)seq_len_sum / (double)seq_n);

	g_seqs = seqs;
	g_seq_n = seq_n;
	g_alignments = ((s64)seq_n * (seq_n - 1)) / 2;
#ifdef USE_CUDA
	g_seq_len_sum = seq_len_sum;
#endif
	g_seq_len_max = seq_len_max;
	file_text_close(&input_file);
	atexit(sequences_free);
	return true;

cleanup_seq_curr_seqs:
	if (seq_curr.letters)
		free(seq_curr.letters);

cleanup_seqs:
	free_aligned(seqs);
	file_text_close(&input_file);
	return false;
}

sequence_t *sequence(s32 index)
{
	if (!g_seqs || index < 0 || index >= g_seq_n)
		unreachable();

	return &g_seqs[index];
}

sequence_t *sequences_seqs(void)
{
	if unlikely (!g_seqs) {
		pdev("Sequences not loaded when accessing sequences");
		perr("Internal error accessing sequences");
		exit(EXIT_FAILURE);
	}

	return g_seqs;
}

s32 sequences_seq_n(void)
{
	if unlikely (!g_seqs) {
		pdev("Sequences not loaded when accessing sequence count");
		perr("Internal error accessing sequence count");
		exit(EXIT_FAILURE);
	}

	return g_seq_n;
}

s64 sequences_alignments(void)
{
	if unlikely (!g_seqs) {
		pdev("Sequences not loaded when accessing alignment count");
		perr("Internal error accessing alignment count");
		exit(EXIT_FAILURE);
	}

	return g_alignments;
}

s32 sequences_seq_len_max(void)
{
	if unlikely (!g_seqs) {
		pdev("Sequences not loaded when accessing max sequence length");
		perr("Internal error accessing max sequence length");
		exit(EXIT_FAILURE);
	}

	return g_seq_len_max;
}

#ifdef USE_CUDA

s64 sequences_seq_len_sum(void)
{
	if unlikely (!g_seqs) {
		pdev("Sequences not loaded when accessing sequence length sum");
		perr("Internal error accessing sequence length sum");
		exit(EXIT_FAILURE);
	}

	return g_seq_len_sum;
}

#endif

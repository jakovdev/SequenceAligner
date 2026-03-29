#include "bio/sequence/sequences.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#include "bio/score/matrices.h"
#include "bio/sequence/filtering.h"
#include "io/input.h"
#include "system/compiler.h"
#include "system/memory.h"
#include "util/benchmark.h"
#include "util/print.h"

#ifdef USE_CUDA
s64 *g_restrict g_indices;
#endif
s32 *g_restrict g_lengths;
s64 *g_restrict g_offsets;
char *g_restrict g_letters;
sequence_t *g_restrict g_seqs;
s64 g_alignments;
s32 g_seq_len_max;
s32 g_seq_n;

static bool globals_dirty;
static void sequences_free(void)
{
	if (!globals_dirty)
		return;

	free(g_lengths);
	free(g_offsets);
	free_aligned(g_letters);
	free_aligned(g_seqs);
#ifdef USE_CUDA
	free(g_indices);
	g_indices = NULL;
#endif
	g_seqs = NULL;
	g_letters = NULL;
	g_offsets = NULL;
	g_lengths = NULL;
	g_alignments = 0;
	g_seq_len_max = 0;
	g_seq_n = 0;
	globals_dirty = false;
}

static bool validate_sequence(char *restrict letters, s32 length)
{
	static bool valid[UCHAR_MAX + 1];
	static enum SequenceType cached_type = SEQ_TYPE_INVALID;

	enum SequenceType type = arg_sequence_type();
	const char *valid_alphabet = NULL;
	int alphabet_size = 0;

	switch (type) {
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

	if (cached_type != type) {
		memset(valid, 0, sizeof(valid));
		for (int i = 0; i < alphabet_size; i++)
			valid[(uchar)valid_alphabet[i]] = true;
		cached_type = type;
	}

	for (s32 i = 0; i < length; i++) {
		char c = (char)toupper((uchar)letters[i]);

		if (!valid[(uchar)c])
			return false;

		letters[i] = c;
	}

	return true;
}

static bool seq_len_valid(size_t len)
{
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
	sequences_free();
	struct ifile ifile = { 0 };
	if (!ifile_open(&ifile, arg_input()))
		return false;

	s32 total = ifile_sequence_count(&ifile);
	size_t capacity = (size_t)(total > 0 ? total : SEQ_N_MIN);
	size_t seq_curr_cap = SEQ_LEN_SUM_MIN;
	size_t letters_cap = PAGE_SIZE;
	s32 *MALLOCA(lengths, capacity);
	s64 *MALLOCA(offsets, capacity);
	char *MALLOCA(seq_curr, seq_curr_cap);
	char *MALLOCA_AL(letters, PAGE_SIZE, letters_cap);
	if unlikely (!lengths || !offsets || !seq_curr || !letters) {
		perr("Out of memory allocating sequences");
		goto cleanup_loading;
	}

	size_t letters_used = 0;
	s32 seq_len_max = 0;
	s32 seq_n_curr = 0;
	s32 seq_n_skip = 0;
	s32 seq_n_invalid = 0;
	bool skip_long = false;
	bool ask_long = false;
	bool skip_invalid = false;
	bool ask_invalid = false;

	bench_io_start();
	do {
		size_t seq_len = 0;
		ifile_sequence_length(&ifile, &seq_len);

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
				seq_n_skip++;
				continue;
			}

			perr("Sequence is too long");
			goto cleanup_loading;
		}

		if (seq_len + 1 > seq_curr_cap) {
			while (seq_len + 1 > seq_curr_cap)
				seq_curr_cap *= 2;
			REALLOCA(seq_curr, seq_curr_cap) {
				perr("Out of memory allocating sequence letters");
				goto cleanup_loading;
			}
		}

		ifile_sequence_extract(&ifile, seq_curr, seq_len);
		if (!validate_sequence(seq_curr, (s32)seq_len)) {
			if (!ask_invalid) {
				bench_io_end();
				pwarn("Found sequence with invalid letters");
				skip_invalid =
					print_yN("Skip invalid sequences?");
				ask_invalid = true;
				bench_io_start();
			}

			if (skip_invalid) {
				seq_n_invalid++;
				continue;
			}

			perr("Found sequence with invalid letters");
			goto cleanup_loading;
		}

		if ((size_t)seq_n_curr == capacity) {
			size_t next_cap = capacity * 2;
			REALLOCA(lengths, next_cap) {
				perr("Out of memory growing sequence lengths");
				goto cleanup_loading;
			}
			REALLOCA(offsets, next_cap) {
				perr("Out of memory growing sequence metadata");
				goto cleanup_loading;
			}
			capacity = next_cap;
		}

		if (letters_used + seq_len + 1 > letters_cap) {
			size_t next_cap = letters_cap;
			while (letters_used + seq_len + 1 > next_cap)
				next_cap *= 2;
			REALLOCA_AL(letters, PAGE_SIZE, letters_cap, next_cap) {
				perr("Out of memory growing sequence letters");
				goto cleanup_loading;
			}
			letters_cap = next_cap;
		}

		lengths[seq_n_curr] = (s32)seq_len;
		offsets[seq_n_curr++] = (s64)letters_used;
		memcpy(letters + letters_used, seq_curr, seq_len + 1);
		letters_used += seq_len + 1;
		if ((s32)seq_len > seq_len_max)
			seq_len_max = (s32)seq_len;
	} while (ifile_sequence_next(&ifile));
	bench_io_end();

	if (seq_n_skip > 0)
		pinfo("Skipped " Ps32 " sequences that were too long",
		      seq_n_skip);

	if (seq_n_invalid > 0)
		pinfo("Skipped " Ps32 " sequences with invalid letters",
		      seq_n_invalid);

	if (seq_n_curr < SEQ_N_MIN) {
		perr("Not enough valid sequences loaded: " Ps32, seq_n_curr);
		goto cleanup_loading;
	}

	if (letters_used < SEQ_LEN_SUM_MIN) {
		perr("Not enough total sequence letters: %zu", letters_used);
		goto cleanup_loading;
	}

	free(seq_curr);
	globals_dirty = true;
	g_seq_n = seq_n_curr;
	g_lengths = lengths;
	g_offsets = offsets;
	g_letters = letters;
	g_seq_len_max = seq_len_max;
	g_alignments = ((s64)g_seq_n * (g_seq_n - 1)) / 2;
	MALLOCA_AL(g_seqs, CACHE_LINE, (size_t)g_seq_n);
	if unlikely (!g_seqs) {
		perr("Out of memory allocating sequences");
		goto cleanup_globals;
	}

	for (s32 i = 0; i < g_seq_n; i++) {
		g_seqs[i].length = g_lengths[i];
		g_seqs[i].letters = g_letters + g_offsets[i];
	}

	if (!filter_seqs())
		goto cleanup_globals;

#ifdef USE_CUDA
	MALLOCA(g_indices, (size_t)g_seq_n);
	if unlikely (!g_indices) {
		perr("Out of memory allocating sequence indices");
		goto cleanup_globals;
	}
	for (s64 i = 0; i < g_seq_n; i++)
		g_indices[i] = (i * (i - 1)) / 2;
#endif

	pinfo("Average sequence length: %.2f",
	      (double)letters_used / (double)g_seq_n - 1.0);

	ifile_close(&ifile);
	return true;

cleanup_loading:
	free(lengths);
	free(offsets);
	free(seq_curr);
	free_aligned(letters);
cleanup_globals:
	sequences_free();
	ifile_close(&ifile);
	return false;
}

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

s64 ALIGNMENTS;
size_t LENGTHS_MAX;
s32 *g_restrict LENGTHS;
s64 *g_restrict OFFSETS;
char *g_restrict LETTERS;
sequence_t *g_restrict SEQS;
s32 SEQS_N;

static void sequences_free(void)
{
	free(LENGTHS);
	free(OFFSETS);
	free_aligned(LETTERS);
	free_aligned(SEQS);
	SEQS = NULL;
	LETTERS = NULL;
	OFFSETS = NULL;
	LENGTHS = NULL;
	ALIGNMENTS = 0;
	LENGTHS_MAX = 0;
	SEQS_N = 0;
}

static bool validate_sequence(s32 length, char letters[restrict static length])
{
	for (s32 i = 0; i < length; i++) {
		char c = (char)toupper((uchar)letters[i]);

		if (SEQ_LUT[(uchar)c] < 0)
			return false;

		letters[i] = c;
	}

	return true;
}

static bool validate_length(size_t len)
{
	const s32 gap_pen = -(GAP_PEN);
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

	size_t total = ifile.entries;
	if (total > SEQ_N_MAX) {
		perr("Too many sequences in input file: %zu (max: %u)", total,
		     SEQ_N_MAX);
		ifile_close(&ifile);
		return false;
	}

	if (total < SEQ_N_MIN) {
		perr("Not enough sequences in input file: %zu (min: %u)", total,
		     SEQ_N_MIN);
		ifile_close(&ifile);
		return false;
	}

	pinfo("Found %zu potential sequences", total);
	size_t capacity = PAGE_SIZE;
	MALLOCA(LENGTHS, total);
	MALLOCA(OFFSETS, total);
	MALLOCA_AL(SEQS, CACHE_LINE, total);
	MALLOCA_AL(LETTERS, PAGE_SIZE, capacity);
	if unlikely (!LENGTHS || !OFFSETS || !SEQS || !LETTERS) {
		perr("Out of memory allocating sequences");
		goto cleanup_seqs;
	}

	size_t letters_used = 0;
	s32 seq_n_curr = 0;
	s32 seq_n_long = 0;
	s32 seq_n_invalid = 0;
	int large = -1;
	int invalid = -1;

	bench_io_start();
	do {
		size_t length = ifile_entry_length(&ifile);
		if (!validate_length(length)) {
			if (large < 0) {
				bench_io_end();
				pwarn("Overflow from large sequence length: %zu",
				      length);
				large = print_yN("Skip long sequences?");
				bench_io_start();
			}

			if (large > 0) {
				seq_n_long++;
				continue;
			}

			perr("Sequence is too long");
			goto cleanup_seqs;
		}

		if (letters_used + length + 1 > capacity) {
			size_t old_cap = capacity;
			while (letters_used + length + 1 > capacity)
				capacity *= 2;
			REALLOCA_AL(LETTERS, PAGE_SIZE, old_cap, capacity) {
				perr("Out of memory growing sequence letters");
				goto cleanup_seqs;
			}
		}

		s32 seq_len = (s32)length;
		char *letters = LETTERS + letters_used;
		size_t written = ifile_entry_extract(&ifile, letters);
		if (written != length) {
			perr("Possible file corruption, expected %zu letters, got %zu",
			     length, written);
			goto cleanup_seqs;
		}

		if (!validate_sequence(seq_len, letters)) {
			if (invalid < 0) {
				bench_io_end();
				pwarn("Found sequence with invalid letters");
				invalid = print_yN("Skip invalid sequences?");
				bench_io_start();
			}

			if (invalid > 0) {
				seq_n_invalid++;
				continue;
			}

			perr("Found sequence with invalid letters");
			goto cleanup_seqs;
		}

		SEQS[seq_n_curr].length = seq_len;
		LENGTHS[seq_n_curr] = seq_len;
		OFFSETS[seq_n_curr++] = (s64)letters_used;
		letters_used += length + 1;
		if (length > LENGTHS_MAX)
			LENGTHS_MAX = length;
	} while (ifile_entry_next(&ifile));
	bench_io_end();

	if (seq_n_long > 0)
		pinfo("Skipped " Ps32 " sequences that were too long",
		      seq_n_long);

	if (seq_n_invalid > 0)
		pinfo("Skipped " Ps32 " sequences with invalid letters",
		      seq_n_invalid);

	if (seq_n_curr < SEQ_N_MIN) {
		perr("Not enough valid sequences loaded: " Ps32, seq_n_curr);
		goto cleanup_seqs;
	}

	if (letters_used < SEQ_LEN_SUM_MIN) {
		perr("Not enough total sequence letters: %zu", letters_used);
		goto cleanup_seqs;
	}

	SEQS_N = seq_n_curr;
	ALIGNMENTS = ((s64)SEQS_N * (SEQS_N - 1)) / 2;
	for (s32 i = 0; i < SEQS_N; i++)
		SEQS[i].letters = LETTERS + OFFSETS[i];

	if (!filter_seqs())
		goto cleanup_seqs;

	s64 used = OFFSETS[SEQS_N - 1] + LENGTHS[SEQS_N - 1] + 1;
	pinfo("Average sequence length: %.2f",
	      (double)used / (double)SEQS_N - 1.0);

	ifile_close(&ifile);
	return true;

cleanup_seqs:
	sequences_free();
	ifile_close(&ifile);
	return false;
}

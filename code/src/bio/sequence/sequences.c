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

s32 *g_restrict g_lengths;
s64 *g_restrict g_offsets;
char *g_restrict g_letters;
sequence_t *g_restrict g_seqs;
s64 g_alignments;
size_t g_seq_len_max;
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

	size_t total = (size_t)ifile_sequence_count(&ifile);
	size_t capacity = PAGE_SIZE;
	globals_dirty = true;
	MALLOCA(g_lengths, total);
	MALLOCA(g_offsets, total);
	MALLOCA_AL(g_seqs, CACHE_LINE, total);
	MALLOCA_AL(g_letters, PAGE_SIZE, capacity);
	if unlikely (!g_lengths || !g_offsets || !g_seqs || !g_letters) {
		perr("Out of memory allocating sequences");
		goto cleanup_seqs;
	}

	s64 letters_used = 0;
	s32 seq_n_curr = 0;
	s32 seq_n_long = 0;
	s32 seq_n_invalid = 0;
	int large = -1;
	int invalid = -1;

	bench_io_start();
	do {
		size_t length = 0;
		ifile_sequence_length(&ifile, &length);

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

		s32 seq_len = (s32)length;
		if (letters_used + seq_len + 1 > (s64)capacity) {
			size_t old_cap = capacity;
			while (letters_used + seq_len + 1 > (s64)capacity)
				capacity *= 2;
			REALLOCA_AL(g_letters, PAGE_SIZE, old_cap, capacity) {
				perr("Out of memory growing sequence letters");
				goto cleanup_seqs;
			}
		}

		char *letters = g_letters + letters_used;
		ifile_sequence_extract(&ifile, letters, length);
		if (!validate_sequence(letters, seq_len)) {
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

		g_seqs[seq_n_curr].length = seq_len;
		g_lengths[seq_n_curr] = seq_len;
		g_offsets[seq_n_curr++] = letters_used;
		letters_used += seq_len + 1;
		if (length > g_seq_len_max)
			g_seq_len_max = length;
	} while (ifile_sequence_next(&ifile));
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
		perr("Not enough total sequence letters: " Ps64, letters_used);
		goto cleanup_seqs;
	}

	g_seq_n = seq_n_curr;
	g_alignments = ((s64)g_seq_n * (g_seq_n - 1)) / 2;
	for (s32 i = 0; i < g_seq_n; i++)
		g_seqs[i].letters = g_letters + g_offsets[i];

	if (!filter_seqs())
		goto cleanup_seqs;

	s64 used = g_offsets[g_seq_n - 1] + g_lengths[g_seq_n - 1] + 1;
	pinfo("Average sequence length: %.2f",
	      (double)used / (double)g_seq_n - 1.0);

	ifile_close(&ifile);
	return true;

cleanup_seqs:
	sequences_free();
	ifile_close(&ifile);
	return false;
}

#include "bio/algorithm/alignment.h"

#include <print.h>
#include <progress.h>

#include "bio/algorithm/method/ga.h"
#include "bio/algorithm/method/nw.h"
#include "bio/algorithm/method/sw.h"
#include "bio/score/matrices.h"
#include "bio/sequence/sequences.h"
#include "bio/types.h"
#include "interface/seqalign_hdf5.h"
#include "system/memory.h"
#include "system/os.h"
#include "util/benchmark.h"
#include "util/macros.h"

size_t TABLE_SIZE;

bool align(struct sequences *dataset)
{
	static const typeof(&align_ga) ALIGN_METHODS[] = {
		[ALIGN_GOTOH_AFFINE] = align_ga,
		[ALIGN_NEEDLEMAN_WUNSCH] = align_nw,
		[ALIGN_SMITH_WATERMAN] = align_sw,
	};
	const typeof(&align_ga) method = ALIGN_METHODS[METHOD];
	const size_t total = (size_t)dataset->alignments;
	pinfo("Performing %zu pairwise alignments", total);
	if (!progress_start(total, THREAD_NUM, "Aligning sequences"))
		return false;

	bench_align_start();
	s64 total_checksum = 0;
	s32 seqs_n = dataset->seqs_n;
	TABLE_SIZE = (dataset->lengths_max + 1) * (dataset->lengths_max + 1);
	struct sequence *seqs = dataset->seqs;
#pragma omp parallel reduction(+ : total_checksum)
	{
		s32 *MALLOCA_AL(TABLE, CACHE_LINE, 3 * TABLE_SIZE);
		s32 *MALLOCA_AL(SEQ1I, CACHE_LINE, dataset->lengths_max);
		s32 *MALLOCA_AL(column_buffer, CACHE_LINE, (size_t)seqs_n);
		if unlikely (!TABLE || !SEQ1I || !column_buffer) {
			perr("Out of memory allocating alignment buffers");
			exit(EXIT_FAILURE);
		}
		s64 checksum = 0;
		s32 col;
#pragma omp for schedule(dynamic)
		for (col = 1; col < seqs_n; col++) {
			seq_ptr seq = &seqs[col];
			for (s32 i = 0; i < seq->length; ++i)
				SEQ1I[i] = SEQ_LUT[(uchar)seq->letters[i]];
			for (s32 row = 0; row < col; row++) {
				const s32 score =
					method(seq, &seqs[row], TABLE, SEQ1I);
				column_buffer[row] = score;
				checksum += score;
			}

			h5_matrix_column_set(col, column_buffer);
			progress_add((size_t)col);
		}

		progress_flush();
		total_checksum += checksum;
		free_aligned(column_buffer);
		free_aligned(SEQ1I);
		free_aligned(TABLE);
	}

	bench_align_end();
	progress_end();
	h5_checksum_set(total_checksum * 2);
	bench_align_print();
	return true;
}

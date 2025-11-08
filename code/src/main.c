#include "core/app/args.h"
#include "core/bio/algorithm/alignment.h"
#include "core/bio/score/scoring.h"
#include "core/bio/sequence/sequences.h"
#include "core/interface/seqalign_cuda.h"
#include "core/interface/seqalign_hdf5.h"
#include "system/os.h"
#include "system/types.h"
#include "util/benchmark.h"
#include "util/print.h"

int main(int argc, char *argv[])
{
	time_init();
	args_init(argc, argv);

	print(M_NONE, HEADER "SEQUENCE ALIGNER");
	args_print_config();
	print(M_NONE, SECTION "Setting Up Alignment");

	bench_io_start();
	if (!sequences_load_from_file())
		return 1;
	bench_io_end();

	u32 sequence_count = sequences_count();
	u64 total_alignments = sequences_alignment_count();
	if (args_mode_cuda() && !cuda_init())
		return 1;

	print_error_context("HDF5");
	bench_io_start();
	if (!h5_open(args_output(), sequence_count, args_compression(),
		     args_mode_write())) {
		bench_io_end();
		print(M_NONE,
		      ERR "Failed to create file, will use no-write mode");

		if (!args_force() && !print_yN("Do you want to continue?")) {
			print(M_LOC(LAST),
			      INFO "Exiting due to file creation failure");
			h5_close(1);
			return 1;
		}

		bench_io_start();
		h5_open(NULL, 0, 0, false);
	}

	if (!h5_sequences_store(sequences_get(), sequence_count)) {
		bench_io_end();
		print(M_NONE,
		      ERR "Failed to store sequences, will use no-write mode");

		if (!args_force() && !print_yN("Do you want to continue?")) {
			print(M_LOC(LAST),
			      INFO "Exiting due to sequence store failure");
			h5_close(1);
			return 1;
		}

		bench_io_start();
		h5_open(NULL, 0, 0, false);
	}

	bench_io_end();

	print(M_LOC(FIRST), VERBOSE "Initializing substitution matrix");
	scoring_init();

	print(M_NONE, SECTION "Performing Alignments");
	print(M_NONE, INFO "Will perform " Pu64 " pairwise alignments",
	      total_alignments);
	bool alignment_success = false;

#ifdef USE_CUDA
	if (args_mode_cuda())
		alignment_success = cuda_align();
	else
#endif
		alignment_success = align();

	if (!alignment_success) {
		print(M_NONE, ERR "Failed to perform alignments");
		h5_close(1);
		return 1;
	}

	if (!args_mode_write())
		print(M_NONE, INFO "Matrix checksum: " Ps64, h5_checksum());

	bench_align_print();

	bench_io_start();
	h5_close(0);
	bench_io_end();

	bench_io_print();

	bench_total_print(total_alignments);
	return 0;
}

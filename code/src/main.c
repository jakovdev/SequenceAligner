#include "bio/algorithm/alignment.h"
#include "bio/score/scoring.h"
#include "bio/sequence/sequences.h"
#include "io/files.h"
#include "interface/seqalign_cuda.h"
#include "interface/seqalign_hdf5.h"
#include "system/os.h"
#include "util/args.h"
#include "util/benchmark.h"
#include "util/print.h"

int main(int argc, char *argv[])
{
	time_init();

	if (!args_parse(argc, argv) || !args_validate()) {
		print(M_NONE, INFO "Use -h, --help for usage information");
		return 1;
	}

	print(M_NONE, HEADER "SEQUENCE ALIGNER");

	print(M_NONE, SECTION "Configuration");
	args_actions();

	print(M_NONE, SECTION "Setting Up Alignment");

	if (!sequences_load_from_file())
		return 1;

	print_error_context("HDF5");
	bench_io_start();
	if (!h5_open(arg_output(), sequences_count())) {
		bench_io_end();
		print(M_NONE,
		      ERR "Failed to create file, will use no-write mode");

		if (!print_yN("Do you want to continue?")) {
			print(M_LOC(LAST),
			      INFO "Exiting due to file creation failure");
			h5_close(1);
			return 1;
		}

		bench_io_start();
		h5_open(NULL, 0);
	}

	if (!h5_sequences_store(sequences(), sequences_count())) {
		bench_io_end();
		print(M_NONE,
		      ERR "Failed to store sequences, will use no-write mode");

		if (!print_yN("Do you want to continue?")) {
			print(M_LOC(LAST),
			      INFO "Exiting due to sequence store failure");
			h5_close(1);
			return 1;
		}

		bench_io_start();
		h5_open(NULL, 0);
	}

	bench_io_end();

	print(M_LOC(FIRST), VERBOSE "Initializing substitution matrix");
	scoring_init();

	print(M_NONE, SECTION "Performing Alignments");
	print(M_NONE, INFO "Will perform " Pu64 " pairwise alignments",
	      sequences_alignment_count());

	if (!(arg_mode_cuda() ? cuda_align() : align())) {
		print(M_NONE, ERR "Failed to perform alignments");
		h5_close(1);
		return 1;
	}

	if (!arg_mode_write())
		print(M_NONE, INFO "Matrix checksum: " Ps64, h5_checksum());

	bench_align_print();

	bench_io_start();
	h5_close(0);
	bench_io_end();

	bench_io_print();

	bench_total_print(sequences_alignment_count());
	return 0;
}

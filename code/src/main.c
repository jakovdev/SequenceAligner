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
		pinfo("Use %s -h, --help for usage information", argv[0]);
		return 1;
	}

	pheader("SEQUENCE ALIGNER");

	psection("Configuration");
	args_actions();

	psection("Setting Up Alignment");

	if (!sequences_load_from_file())
		return 1;

	perr_context("HDF5");
	bench_io_start();
	if (!h5_open(arg_output(), sequences_count())) {
		bench_io_end();
		perr("Failed to create file, will use no-write mode");

		if (!print_yN("Do you want to continue?")) {
			pinfol("Exiting due to file creation failure");
			h5_close(1);
			return 1;
		}

		bench_io_start();
		h5_open(NULL, 0);
	}

	if (!h5_sequences_store(sequences(), sequences_count())) {
		bench_io_end();
		perr("Failed to store sequences, will use no-write mode");

		if (!print_yN("Do you want to continue?")) {
			pinfol("Exiting due to sequence store failure");
			h5_close(1);
			return 1;
		}

		bench_io_start();
		h5_open(NULL, 0);
	}

	bench_io_end();

	pverb("Initializing substitution matrix");
	scoring_init();

	psection("Performing Alignments");
	pinfo("Will perform " Pu64 " pairwise alignments",
	      sequences_alignment_count());

	if (!(arg_mode_cuda() ? cuda_align() : align())) {
		perr("Failed to perform alignments");
		h5_close(1);
		return 1;
	}

	if (!arg_mode_write())
		pinfo("Matrix checksum: " Ps64, h5_checksum());

	bench_align_print();

	bench_io_start();
	h5_close(0);
	bench_io_end();

	bench_io_print();

	bench_total_print(sequences_alignment_count());
	return 0;
}

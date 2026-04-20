#include <args.h>
#include <print.h>

#include "bio/alignment.h"
#include "bio/sequences.h"
#include "interface/seqalign_cuda.h"
#include "interface/seqalign_hdf5.h"
#include "util/benchmark.h"

int main(int argc, char *argv[])
{
	if (!args_parse(argc, argv) || !args_validate()) {
		pinfo("Use %s -h, --help for usage information", argv[0]);
		return 1;
	}

	pheader("SEQUENCE ALIGNER");

	psection("Configuration");
	args_actions();

	psection("Setting Up Alignment");
	[[gnu::cleanup(input_free)]] struct input dataset = {};
	if (!input_load(&dataset) || !filter(&dataset))
		return 1;

	pinfo("Loaded %w32d sequences", dataset.seqs_n);
	pinfo("Average sequence length: %.2f", dataset.average_length);

	if (arg_mode_cuda() && !cuda_device_init())
		return 1;

	if (!h5_open(&dataset)) {
		cuda_device_close();
		return 1;
	}

	psection("Performing Alignments");
	if (!(arg_mode_cuda() ? cuda_align(&dataset) : align(&dataset))) {
		cuda_device_close();
		h5_close(1);
		return 1;
	}

	h5_close(0);
	bench_total_print((double)dataset.alignments);
	return 0;
}

#include <args.h>
#include <print.h>

#include "bio/alignment.h"
#include "bio/sequences.h"
#include "interface/seqalign_cuda.h"
#include "io/output.h"
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

	psection("Reading Dataset");
	[[gnu::cleanup(input_free)]] struct input dataset = {};
	if (!input_load(&dataset) || !filter(&dataset))
		return 1;

	pinfo("Loaded %d sequences", dataset.seqs_n);
	pinfo("Average sequence length: %.2f", dataset.average_length);
	bench_input_print();

	psection("Creating Similarity Matrix");
	[[gnu::cleanup(output_free)]] struct output sm = {};
	if (!output_load(&sm, &dataset))
		return 1;

	psection("Performing Alignments");
	if (!cuda_align(&dataset, &sm))
		return 1;

	psection("Writing Similarity Matrix");
	if (!output_flush(&sm))
		return 1;

	bench_total_print((double)dataset.alignments);
	return 0;
}

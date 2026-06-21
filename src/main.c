#include <args.h>
#include <print.h>

#include "interface/seqalign_cuda.h"
#include "io/input.h"
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
	[[gnu::cleanup(input_free)]] struct input in = {};
	if (!input_load(&in))
		return 1;

	[[gnu::cleanup(output_free)]] struct output out = {};
	if (!output_load(&out, in))
		return 1;

	psection("Performing Alignments");
	if (!cuda_align(in, out))
		return 1;

	if (!output_flush(&out))
		return 1;

	bench_total_print(alignments((s64)in.num));
	return 0;
}
